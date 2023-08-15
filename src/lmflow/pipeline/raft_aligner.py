#!/usr/bin/env python
# coding=utf-8
"""
The Aligner class simplifies the process of running alignment.
"""
import json
import logging
import numpy as np
import os
import sys
import time
from itertools import chain

import torch
import torch.distributed as dist
import transformers
from datasets import (
    set_caching_enabled,
    Dataset,
    DatasetDict,
    load_dataset
)
from transformers import (
    default_data_collator,
    pipeline,
    set_seed,
    AutoModelForCausalLM
)
from transformers.testing_utils import CaptureLogger

from lmflow.args import DatasetArguments
from lmflow.datasets.dataset import Dataset as LMFlowDataset
from lmflow.pipeline.base_aligner import BaseAligner
from lmflow.pipeline.utils.raft_trainer import RaftTrainer

logger = logging.getLogger(__name__)


class RaftAligner(BaseAligner):
    """
    Initializes the `RaftAligner` class with given arguments.

    Parameters
    ------------
    model_args : ModelArguments object.
        Contains the arguments required to load the model.
    
    data_args : DatasetArguments object.
        Contains the arguments required to load the dataset.

    raft_aligner_args : RaftAlignerArguments object.
        Contains the arguments required to perform alignment.

    args : Optional.
        Positional arguments.
    
    kwargs : Optional.
        Keyword arguments.

    """
    def __init__(self, model_args, data_args, aligner_args, *args, **kwargs):
        self.model_args = model_args
        self.data_args = data_args
        self.aligner_args = aligner_args

        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            handlers=[logging.StreamHandler(sys.stdout)],
        )

        self.INF = 88888888
        logger.setLevel(logging.INFO)

        output_reward_path = aligner_args.output_reward_path
        if output_reward_path is not None:
            os.makedirs(os.path.dirname(output_reward_path), exist_ok=True)
            # Deletes a maybe-exist file
            try:
                os.remove(output_reward_path)
            except OSError:
                pass

        ##########################
        self.raft_infer_samples_store_dir = aligner_args.raft_infer_set + "/my_infer_set.json"
        self.raft_filter_samples_store_dir = aligner_args.raft_filtered_set + "/my_filtered_set.json"
        self.raft_eval_samples_store_dir = model_args.model_name_or_path + "/eval_set/my_eval_set.json" 
        self.raft_rewards_store_dir = aligner_args.raft_exp_dir + "/reward_record.txt"
        self.raft_eval_rewards_store_dir = aligner_args.raft_exp_dir + "/reward_record.txt"
        self.raft_rank_store_path = aligner_args.raft_infer_set + "/training_reward_data.npy"
        self.if_save_ranks = aligner_args.save_ranks
        #########################

        # Set seed before initializing model.
        set_seed(aligner_args.seed)


    def _initialize_trainer(self, model, tokenizer, training_args):
        """
        This function takes the model and tokenizer as the input and initialize the trainer.
        """
        trainer = RaftTrainer(
            model=model,
            args=training_args,
            train_dataset=Dataset.from_dict({"text": [ " " ] }),
            eval_dataset=Dataset.from_dict({}),
            tokenizer=tokenizer,
            data_collator=default_data_collator,
            compute_metrics=None,
            preprocess_logits_for_metrics=None,
        )
        return trainer


    def _load_dataset(
        self,
        selected_dataset,
        model,
        tokenizer,
        model_args,
        data_args,
        training_args,
    ):
        '''
        This function prepares the dataset for every iteration.
        '''
        raw_datasets = selected_dataset

        if training_args.do_train:
            column_names = list(raw_datasets["train"].features)
        else:
            column_names = list(raw_datasets["validation"].features)
        text_column_name = "text" if "text" in column_names else column_names[0]

        # since this will be pickled to avoid _LazyModule error in Hasher force logger loading before tokenize_function
        tok_logger = transformers.utils.logging.get_logger("transformers.tokenization_utils_base")

        def tokenize_function(examples):
            with CaptureLogger(tok_logger) as cl:
                output = tokenizer(examples[text_column_name])
            # clm input could be much much longer than block_size
            if "Token indices sequence length is longer than the" in cl.out:
                tok_logger.warning(
                    "^^^^^^^^^^^^^^^^ Please ignore the warning above - this long input will be chunked into smaller bits"
                    " before being passed to the model."
                )
            return output

        with training_args.main_process_first(desc="dataset map tokenization"):
            if not data_args.streaming:
                tokenized_datasets = raw_datasets.map(
                    tokenize_function,
                    batched=True,
                    num_proc=data_args.preprocessing_num_workers,
                    remove_columns=column_names,
                    load_from_cache_file=not data_args.overwrite_cache,
                    desc="Running tokenizer on dataset",
                )
            else:
                tokenized_datasets = raw_datasets.map(
                    tokenize_function,
                    batched=True,
                    remove_columns=column_names,
                )

        if data_args.block_size is None:
            block_size = tokenizer.model_max_length
            if block_size > 1024:
                logger.warning(
                    "The chosen tokenizer supports a `model_max_length` that is longer than the default `block_size` value"
                    " of 1024. If you would like to use a longer `block_size` up to `tokenizer.model_max_length` you can"
                    " override this default with `--block_size xxx`."
                )
                block_size = 512
        else:
            if data_args.block_size > tokenizer.model_max_length:
                logger.warning(
                    f"The block_size passed ({data_args.block_size}) is larger than the maximum length for the model"
                    f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
                )
            block_size = min(data_args.block_size, tokenizer.model_max_length)

        # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
        def group_texts(examples):
            # Concatenate all texts.
            concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
            total_length = len(concatenated_examples[list(examples.keys())[0]])
            # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
            # customize this part to your needs.
            if total_length >= block_size:
                total_length = (total_length // block_size) * block_size
            # Split by chunks of max_len.
            result = {
                k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
                for k, t in concatenated_examples.items()
            }
            result["labels"] = result["input_ids"].copy()
            return result

        # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a remainder
        # for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value might be slower
        # to preprocess.
        #
        # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
        # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map

        with training_args.main_process_first(desc="grouping texts together"):
            group_batch_size = 1000
            if data_args.disable_group_texts:
                group_batch_size = 1
            if not data_args.streaming:
                lm_datasets = tokenized_datasets.map(
                    group_texts,
                    batched=True,
                    batch_size=group_batch_size,
                    num_proc=data_args.preprocessing_num_workers,
                    load_from_cache_file=not data_args.overwrite_cache,
                    desc=f"Grouping texts in chunks of {block_size}",
                )
            else:
                lm_datasets = tokenized_datasets.map(
                    group_texts,
                    batched=True,
                    batch_size=group_batch_size,
                )
    
        if training_args.do_train:
            if "train" not in tokenized_datasets:
                raise ValueError("--do_train requires a train dataset")
            train_dataset = lm_datasets["train"]
            if data_args.max_train_samples is not None:
                max_train_samples = min(len(train_dataset), data_args.max_train_samples)
                train_dataset = train_dataset.select(range(max_train_samples))

        return train_dataset


    def _load_input_dataset(self, dataset, tokenizer):
        """
        Load input dataset (i.e. prompt/question dataset) for training.

        Args:
            dataset: A Dataset object.
                The dataset to be loaded.

        Returns:
            dataloader (`torch.utils.data.DataLoader`):
                The dataloader for the dataset.
        """
        ds = dataset.get_backend_dataset()
        
        def tokenize(sample):
            sample["input_ids"] = tokenizer.encode(sample["text"])
            sample['input'] = tokenizer.decode(sample["input_ids"])
            return sample
        if self.mode == "raft_get_rewards":
            pass
        elif self.mode == "relabel":
            pass
        elif self.mode == "raft_get_ranks":
            pass
        elif self.mode == "raft_sample_rewards":
            pass
        elif self.mode == "eval_get_rewards":
            pass
        else:
            ds = ds.map(tokenize, batched=False)
            ds = ds.filter(lambda x: len(x["input_ids"]) <= 256)
        
        ds.set_format(type='torch')

        return ds

    def _clean_text(self, text):
        if len(text) == 0:
            return text
        stext = [x for x in text.split("###Human") if x]
        return stext[0].strip().strip("#") 

    def _discard_sample(self, text):
        if "#" in text:
            return True
        elif len(text) < 2: # delete empty sample
            return True
        return False
    
    def _sample_index(self, rewards, alpha, temperature=1):
        rewards = np.array(rewards)
        if np.all(rewards == -self.INF):
            return 0
        else:
            data_size = len(rewards)
            probs = np.exp(rewards/temperature) / np.sum(np.exp(rewards/temperature))
            idxs = np.random.choice(data_size, int(data_size * alpha), replace=False, p=probs)
            return idxs[0]

    def _raft_get_samples(
            self,
            model,
            batch_input,
            K=8,
            iter_id=0,
            local_rank=0,
            output_min_length=128,
            output_max_length=196,
            infer_batch_size=8,
            generation_kwargs={},
            tokenizer=None,
            training_args=None,
            reward_model=None,
            output_reward_path=None,
        ):
        """
        This function generates a batch of data by best-of-K policy from randomly sampled prompts
        """
        # we will get the batch dataset via Dataset.from_dict
        start_time = time.time()

        querys = batch_input['input']
        data_size = len(querys)

        reward_eva = []
        reward_train = []
        data = []
        input_texts = []
        responses = []
        all_outputs = []

        for i, query in enumerate(querys):
            input_texts = [query for _ in range(K)]
            
            gen_len = np.random.randint(output_min_length, output_max_length)
            generation_kwargs["max_new_tokens"] = gen_len
            inputs = tokenizer(input_texts, return_tensors="pt", padding=True).to(training_args.device)
            with torch.no_grad():
                outputs = model.generate(**inputs, **generation_kwargs)
            generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            generated_texts = [
                generated_text.replace(input_texts[i], "") for i, generated_text in enumerate(generated_texts)
            ]
            generated_texts = [
                self._clean_text(generated_text) for generated_text in generated_texts
            ]
            reward_eva.extend([1 for _ in range(K)])

            data.append({"input": querys[i], "output": [generated_texts[j] for j in range(K)]})

            input_texts = []

        world_size = int(os.getenv("WORLD_SIZE", "1"))
        all_process_data =[{}] * world_size
        dist.all_gather_object(all_process_data, data)

        all_process_eval_reward =[{}] * world_size
        dist.all_gather_object(all_process_eval_reward, reward_eva)
        all_process_train_set_reward =[{}] * world_size
        dist.all_gather_object(all_process_train_set_reward, reward_train)

        
        gathered_data = []
        gathered_reward = []
        gathered_train_reward = []

        for i in range(world_size):
            gathered_data.extend(all_process_data[i])
            gathered_reward.extend(all_process_eval_reward[i])
            gathered_train_reward.extend(all_process_train_set_reward[i])

        output_eval_dataset = {}
        output_eval_dataset['type'] = 'text_only'
        output_eval_dataset['instances'] = gathered_data
        import json
        if local_rank == 0:
            with open(self.raft_infer_samples_store_dir, 'w', encoding='utf8') as f:
                json.dump(output_eval_dataset, f, ensure_ascii=False)

        logger.info(f"collected data of {len(gathered_data)}")


    def _raft_get_rewards(
        self,
        batch_input,
        alpha=0.2,
        iter_id=0,
        local_rank=0,
        output_min_length=128,
        output_max_length=196,
        infer_batch_size=8,
        generation_kwargs={},
        tokenizer=None,
        training_args=None,
        reward_model=None,
        output_reward_path=None,
    ):
        """
        This function computes the rewards for the K responses given each prompt.
        We also collect the best of K samples into a filtered dataset.
        """
        # we will get the batch dataset via Dataset.from_dict

        kl_penalty = 0.001
        start_time = time.time()
        reward_eva = []
        reward_train = []
        querys = batch_input['input']
        responses = batch_input['output']
        K = len(responses[0])
        data = []
        reward_data = []
        kls = batch_input['kl']

        kl_eval = []
        kl_train = []
        for i in range(len(querys)):
            q = querys[i]
            tmp_responses = responses[i]
            texts_for_rewards = [q + r for r in tmp_responses]

            texts_for_reward_dataset = LMFlowDataset.create_from_dict({
                "type": "text_only",
                "instances": [
                    { "text": text } for text in texts_for_rewards
                ],
            })

            reward_dataset = reward_model.inference(texts_for_reward_dataset)
            rewards = [ sample["value"] for sample in reward_dataset.to_dict()["instances"] ]

            kl_rewards = kls[i]
            record_reward  = rewards[0] #- kl_penalty * kl_rewards[0]
            reward_eva.append(record_reward)
            
            kl_eval.append(kl_rewards[0])

                
            # we impose some post-detection and discard the samples with certain criteria.
            for kk in range(K):
                if self._discard_sample(tmp_responses[kk]):
                    rewards[kk] = -self.INF
            ################################
            
            regularized_rewards = [rewards[j] - 4.82 - kl_penalty * kl_rewards[j] for j in range(K)]

            idx_to_record = np.argmax(regularized_rewards)

            #idx_to_record = np.argmax(rewards)
            
            
            # if we discard all the samples, we do not record the sample 
            if rewards[idx_to_record] >= -20:
                data.append({"text": q + tmp_responses[idx_to_record]})
                reward_train.append(rewards[idx_to_record])
                reward_data.append(rewards)
                kl_train.append(kl_rewards[idx_to_record])
        


        world_size = int(os.getenv("WORLD_SIZE", "1"))
        all_process_list =[{}] * world_size

        mean_eval_reward = np.mean(reward_eva)
        mean_kl = np.mean(kl_eval)
        data_to_send = {
            'data': [[data[i], mean_eval_reward, reward_train[i],reward_data[i], mean_kl, kl_train[i]] for i in range(len(data))],
            'local_rank': local_rank
        }
        dist.all_gather_object(all_process_list, data_to_send)
        gathered_data = []
        gathered_reward = []
        gathered_train_reward = []
        gathered_rewards = []
        local_rank_sequence=[]
        gathered_kl = []
        gathered_train_kl = []
        for i in range(world_size):
            tmp_data = [tmp[0] for tmp in all_process_list[i]['data']]
            gathered_data.extend(tmp_data)

            tmp_reward = [tmp[1] for tmp in all_process_list[i]['data']]
            gathered_reward.extend(tmp_reward)

            tmp_train_reward = [tmp[2] for tmp in all_process_list[i]['data']]
            gathered_train_reward.extend(tmp_train_reward)

            tmp_rewards = [tmp[3] for tmp in all_process_list[i]['data']]
            gathered_rewards.append(tmp_rewards)


            tmp_kls = [tmp[4] for tmp in all_process_list[i]['data']]
            gathered_kl.extend(tmp_kls)

            tmp_train_kl = [tmp[5] for tmp in all_process_list[i]['data']]
            gathered_train_kl.extend(tmp_train_kl)

            tmp_local_rank = all_process_list[i]['local_rank']
            local_rank_sequence.append(tmp_local_rank)
        
        sorted_idx = np.argsort(local_rank_sequence)
        reward_list = [gathered_rewards[idx] for idx in sorted_idx]
        merged_reward_data = np.concatenate(reward_list, axis=0)
        logger.info(f"collected data of {len(gathered_data)}")
        logger.info(f"mean reward: {np.mean(gathered_reward)}, reward in train set: {np.mean(gathered_train_reward)}")
        

 
        # We store the training set for monitoring the RAFT training
        output_eval_dataset = {}
        output_eval_dataset['type'] = 'text_only'
        output_eval_dataset['instances'] = gathered_data
        import json
        if local_rank == 0:
            with open(self.raft_filter_samples_store_dir, 'w', encoding='utf8') as f:
                json.dump(output_eval_dataset, f, ensure_ascii=False)

            with open(self.raft_rewards_store_dir, 'a') as f:
                f.write(str(np.mean(gathered_reward)) + "   " + str(np.mean(gathered_train_reward)) 
                        + "  " + str(np.mean(gathered_kl)) + "   " + str(np.mean(gathered_train_kl)) + "\n")

            
            np.save(self.raft_rank_store_path, merged_reward_data)

    def _raft_sample_rewards(
        self,
        batch_input,
        alpha=0.2,
        temperature = 1,
        iter_id=0,
        local_rank=0,
        output_min_length=128,
        output_max_length=196,
        infer_batch_size=8,
        generation_kwargs={},
        tokenizer=None,
        training_args=None,
        reward_model=None,
        output_reward_path=None,
    ):
        """
        This function computes the rewards for the K responses given each prompt.
        We also collect the best of K samples into a filtered dataset.
        """
        # we will get the batch dataset via Dataset.from_dict
        start_time = time.time()
        reward_eva = []
        reward_train = []
        querys = batch_input['input']
        responses = batch_input['output']
        K = len(responses[0])
        data = []
        reward_data = []
        for i in range(len(querys)):
            q = querys[i]
            tmp_responses = responses[i]
            texts_for_rewards = [q + r for r in tmp_responses]

            texts_for_reward_dataset = LMFlowDataset.create_from_dict({
                "type": "text_only",
                "instances": [
                    { "text": text } for text in texts_for_rewards
                ],
            })

            reward_dataset = reward_model.inference(texts_for_reward_dataset)
            rewards = [ sample["value"] for sample in reward_dataset.to_dict()["instances"] ]
            
            reward_eva.append(rewards[0])

                
            # we impose some post-detection and discard the samples with certain criteria.
            for kk in range(K):
                if self._discard_sample(tmp_responses[kk]):
                    rewards[kk] = -self.INF
            ################################
            idx_to_record = self._sample_index(rewards=rewards, alpha = 1/K, temperature = temperature)
            
            
            # if we discard all the samples, we do not record the sample 
            if rewards[idx_to_record] != -self.INF:
                data.append({"text": q + tmp_responses[idx_to_record]})
                reward_train.append(rewards[idx_to_record])
                reward_data.append(rewards)

        world_size = int(os.getenv("WORLD_SIZE", "1"))
        all_process_list =[{}] * world_size

        mean_eval_reward = np.mean(reward_eva)

        data_to_send = {
            'data': [[data[i], mean_eval_reward, reward_train[i],reward_data[i]] for i in range(len(data))],
            'local_rank': local_rank
        }
        dist.all_gather_object(all_process_list, data_to_send)
        gathered_data = []
        gathered_reward = []
        gathered_train_reward = []
        gathered_rewards = []
        local_rank_sequence=[]
        for i in range(world_size):
            tmp_data = [tmp[0] for tmp in all_process_list[i]['data']]
            gathered_data.extend(tmp_data)

            tmp_reward = [tmp[1] for tmp in all_process_list[i]['data']]
            gathered_reward.extend(tmp_reward)

            tmp_train_reward = [tmp[2] for tmp in all_process_list[i]['data']]
            gathered_train_reward.extend(tmp_train_reward)

            tmp_rewards = [tmp[3] for tmp in all_process_list[i]['data']]
            gathered_rewards.append(tmp_rewards)

            tmp_local_rank = all_process_list[i]['local_rank']
            local_rank_sequence.append(tmp_local_rank)
        
        sorted_idx = np.argsort(local_rank_sequence)
        reward_list = [gathered_rewards[idx] for idx in sorted_idx]
        merged_reward_data = np.concatenate(reward_list, axis=0)
        logger.info(f"collected data of {len(gathered_data)}")
        logger.info(f"mean reward: {np.mean(gathered_reward)}, reward in train set: {np.mean(gathered_train_reward)}")
        

 
        # We store the training set for monitoring the RAFT training
        output_eval_dataset = {}
        output_eval_dataset['type'] = 'text_only'
        output_eval_dataset['instances'] = gathered_data
        import json
        if local_rank == 0:
            with open(self.raft_filter_samples_store_dir, 'w', encoding='utf8') as f:
                json.dump(output_eval_dataset, f, ensure_ascii=False)

            with open(self.raft_rewards_store_dir, 'a') as f:
                f.write(str(np.mean(gathered_reward)) + "   " + str(np.mean(gathered_train_reward)) + "\n")

            
            np.save(self.raft_rank_store_path, merged_reward_data)


    def _raft_get_ranks(
        self,
        batch_input,
        alpha=0.2,
        iter_id=0,
        local_rank=0,
        output_min_length=128,
        output_max_length=196,
        infer_batch_size=8,
        generation_kwargs={},
        tokenizer=None,
        training_args=None,
        reward_model=None,
        output_reward_path=None,
    ):
        """
        This function computes the rewards for the K responses given each prompt.
        We also collect the best of K samples into a filtered dataset.
        """
        # we will get the batch dataset via Dataset.from_dict
        start_time = time.time()
        reward_eva = []
        rank_reward_eva = []
        reward_train = []
        rank_reward_train =[]
        reward_data = []
        querys = batch_input['input']
        responses = batch_input['output']
        K = len(responses[0])
        data = []
        for i in range(len(querys)):
            q = querys[i]
            tmp_responses = responses[i]
            texts_for_rewards = [q + r for r in tmp_responses]

            texts_for_reward_dataset = LMFlowDataset.create_from_dict({
                "type": "text_only",
                "instances": [
                    { "text": text } for text in texts_for_rewards
                ],
            })

            reward_dataset = reward_model.inference(texts_for_reward_dataset)
            rewards = [ sample["value"] for sample in reward_dataset.to_dict()["instances"] ]
            rank_rewards = [ sample["rank_reward"] for sample in reward_dataset.to_dict()["instances"] ]
            
            # ranks = [sample["rank"] for sample in reward_dataset.to_dict()["instances"]]

            # if "#" in tmp_responses[0]:
            #     print("TTTT")
            #     rewards[0] = -1
            record_reward  = rewards[0]
            record_rank_reward = rank_rewards[0]
            reward_eva.append(record_reward)
            rank_reward_eva.append(record_rank_reward)
            # rank_data.append(ranks)

            #reward_eva.append(rewards[0])

                
            # we impose some post-detection and discard the samples with certain criteria.
            for kk in range(K):
                if self._discard_sample(tmp_responses[kk]):
                    rank_rewards[kk] = -self.INF
            ################################
            
            idx_to_record = np.argmax(rank_rewards)
            
            
            # if we discard all the samples, we do not record the sample 
            if rank_rewards[idx_to_record] != -self.INF:
                data.append({"text": q + tmp_responses[idx_to_record]})
                reward_train.append(rewards[idx_to_record])
                rank_reward_train.append(rank_rewards[idx_to_record])
                reward_data.append(rank_rewards)


        


        world_size = int(os.getenv("WORLD_SIZE", "1"))
        all_process_list =[{}] * world_size

        mean_eval_reward = np.mean(reward_eva,axis=0)
        mean_eval_rank_reward = np.mean(rank_reward_eva,axis=0)
        data_to_send = {
            'data': [[data[i], mean_eval_reward, reward_train[i], mean_eval_rank_reward, rank_reward_train[i],reward_data[i]] for i in range(len(data))],
            'local_rank': local_rank
        }
        dist.all_gather_object(all_process_list, data_to_send)
        
        
        if local_rank == 0:
            gathered_data = []
            gathered_reward = []
            gathered_rank_reward = []
            gathered_train_reward = []
            gathered_train_rank_reward = []
            gathered_rewards = []
            local_rank_sequence=[]

            for i in range(world_size):
                tmp_data = [tmp[0] for tmp in all_process_list[i]['data']]
                gathered_data.extend(tmp_data)

                tmp_reward = [tmp[1] for tmp in all_process_list[i]['data']]
                gathered_reward.extend(tmp_reward)

                tmp_train_reward = [tmp[2] for tmp in all_process_list[i]['data']]
                gathered_train_reward.extend(tmp_train_reward)

                tmp_reward = [tmp[3] for tmp in all_process_list[i]['data']]
                gathered_rank_reward.extend(tmp_reward)

                tmp_train_reward = [tmp[4] for tmp in all_process_list[i]['data']]
                gathered_train_rank_reward.extend(tmp_train_reward)

                tmp_rewards = [tmp[5] for tmp in all_process_list[i]['data']]
                gathered_rewards.append(tmp_rewards)

                tmp_local_rank = all_process_list[i]['local_rank']
                local_rank_sequence.append(tmp_local_rank)

            sorted_idx = np.argsort(local_rank_sequence)
            reward_list = [gathered_rewards[idx] for idx in sorted_idx]
            merged_reward_data = np.concatenate(reward_list, axis=0)


            logger.info(f"collected data of {len(gathered_data)}")
            logger.info(f"mean reward: {np.mean(gathered_reward,axis=0)}, reward in train set: {np.mean(gathered_train_reward,axis=0)}")
            logger.info(f"mean rank reward: {np.mean(gathered_rank_reward,axis=0)}, rank reward in train set: {np.mean(gathered_train_rank_reward,axis=0)}")
                
            # We store the training set for monitoring the RAFT training
            output_eval_dataset = {}
            output_eval_dataset['type'] = 'text_only'
            output_eval_dataset['instances'] = gathered_data
            with open(self.raft_filter_samples_store_dir, 'w', encoding='utf8') as f:
                json.dump(output_eval_dataset, f, ensure_ascii=False)

            with open(self.raft_rewards_store_dir, 'a') as f:
                f.write("mean reward: "+ str(list(np.mean(gathered_reward,axis=0))) + "   " + "reward in train set: " + str(list(np.mean(gathered_train_reward,axis=0))) + "mean rank reward: "+ str(np.mean(gathered_rank_reward,axis=0)) + "   " + "rank reward in train set: " + str(np.mean(gathered_train_rank_reward,axis=0)) + "\n")
            
            np.save(self.raft_rank_store_path, merged_reward_data)

    def _eval_get_samples(
            self,
            model,
            batch_input,
            K=8,
            iter_id=0,
            local_rank=0,
            output_min_length=128,
            output_max_length=196,
            infer_batch_size=8,
            generation_kwargs={},
            tokenizer=None,
            training_args=None,
            reward_model=None,
            output_reward_path=None,
        ):
        """
        This function generates a batch of data for evaluation.
        """
        # we will get the batch dataset via Dataset.from_dict
        start_time = time.time()
        
        generation_kwargs = {
            "min_length": 1,
            "top_k": 40,
            "top_p": 1.0,
            "do_sample": True,
            "pad_token_id": tokenizer.eos_token_id,
            "temperature":0.7,
        }
        querys = batch_input['input']
        data_size = len(querys)

        reward_eva = []
        reward_train = []
        data = []
        input_texts = []
        responses = []
        all_outputs = []
        infer_batch_size = 32

        for i in range(data_size):
            query = querys[i]
            input_texts.append(query)

            if (i + 1) % infer_batch_size == 0 or (i+1 == data_size):
                gen_len = np.random.randint(output_min_length, output_max_length)
                generation_kwargs["max_new_tokens"] = gen_len
                inputs = tokenizer(input_texts, return_tensors="pt", padding=True).to(training_args.device)
                with torch.no_grad():
                    outputs = model.generate(**inputs, **generation_kwargs)
                generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                generated_texts = [
                    generated_text.replace(input_texts[i], "") for i, generated_text in enumerate(generated_texts)
                ]
                generated_texts = [
                    self._clean_text(generated_text) for i, generated_text in enumerate(generated_texts)
                ]
                data.extend([{"input": input_texts[jj], "output": [generated_texts[jj]]} for jj in range(len(generated_texts)) ])

                reward_eva.extend([1 for jj in range(len(generated_texts))])
                responses.extend(generated_texts)
                input_texts = []
                reward_train.extend([1 for jj in range(len(generated_texts))])

        world_size = int(os.getenv("WORLD_SIZE", "1"))
        all_process_data =[{}] * world_size
        dist.all_gather_object(all_process_data, data)

        all_process_eval_reward =[{}] * world_size
        dist.all_gather_object(all_process_eval_reward, reward_eva)
        all_process_train_set_reward =[{}] * world_size
        dist.all_gather_object(all_process_train_set_reward, reward_train)

        
        gathered_data = []
        gathered_reward = []
        gathered_train_reward = []

        for i in range(world_size):
            gathered_data.extend(all_process_data[i])
            gathered_reward.extend(all_process_eval_reward[i])
            gathered_train_reward.extend(all_process_train_set_reward[i])

        output_eval_dataset = {}
        output_eval_dataset['type'] = 'text_only'
        output_eval_dataset['instances'] = gathered_data
        import json
        if local_rank == 0:
            with open(self.raft_eval_samples_store_dir, 'w', encoding='utf8') as f:
                logger.info(f"Opened the text file: {self.raft_eval_samples_store_dir}")
                json.dump(output_eval_dataset, f, ensure_ascii=False)

        logger.info(f"collected data of {len(gathered_data)}")
        
        ##### get the ppl
        ppl_dataset = load_dataset("json", data_files="/home/winnie/data/clean_hh_rlhf_uncerrtainty_study/eval_ppl_dataset/ppl_dataset.json", split="train", field="instances")

        texts = ppl_dataset['text']
        encodings = tokenizer("\n\n".join(texts), return_tensors="pt")
        max_length = 1024

        stride = 512
        seq_len = encodings.input_ids.size(1)

        nlls = []
        prev_end_loc = 0
    
        for begin_loc in range(0, seq_len, stride):
            end_loc = min(begin_loc + max_length, seq_len)
            trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
            input_ids = encodings.input_ids[:, begin_loc:end_loc].to(training_args.device)
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100

            with torch.no_grad():
                outputs = model(input_ids, labels=target_ids)

                # loss is calculated using CrossEntropyLoss which averages over input tokens.
                # Multiply it with trg_len to get the summation instead of average.
                # We will take average over all the tokens to get the true average
                # in the last step of this example.
                neg_log_likelihood = outputs.loss * trg_len

            nlls.append(neg_log_likelihood)

            prev_end_loc = end_loc
            if end_loc == seq_len:
                break

        ppl = torch.exp(torch.stack(nlls).sum() / end_loc)
        #diversity_metrics['ppl'] = ppl
        print(ppl)
        if local_rank == 0:
            with open(self.raft_eval_rewards_store_dir, 'a') as f:
                f.write(str(ppl) + "\n")
        #####

    def _eval_get_rewards(
        self,
        batch_input,
        alpha=0.2,
        iter_id=0,
        local_rank=0,
        output_min_length=128,
        output_max_length=196,
        infer_batch_size=8,
        generation_kwargs={},
        tokenizer=None,
        training_args=None,
        reward_model=None,
        output_reward_path=None,
    ):
        """
        :param batch_input: input prompts
        """
        # we will get the batch dataset via Dataset.from_dict
        start_time = time.time()
        reward_eva = []
        reward_train = []
        querys = batch_input['input']
        responses = batch_input['output']
        K = len(responses[0])
        data = []
        reward_data = []
        print("start generating responses")
        for i in range(len(querys)):
            q = querys[i]
            tmp_responses = responses[i]
            texts_for_rewards = [q + r for r in tmp_responses]

            texts_for_reward_dataset = LMFlowDataset.create_from_dict({
                "type": "text_only",
                "instances": [
                    { "text": text } for text in texts_for_rewards
                ],
            })

            reward_dataset = reward_model.inference(texts_for_reward_dataset)
            rewards = [ sample["value"] for sample in reward_dataset.to_dict()["instances"] ]

            # if "#" in tmp_responses[0]:
            #     print("TTTT")
            #     rewards[0] = -1
            record_reward  = rewards[0]
            reward_eva.append(record_reward)
            
            #reward_eva.append(rewards[0])

                
            # we impose some post-detection and discard the samples with certain criteria.
            for kk in range(K):
                if self._discard_sample(tmp_responses[kk]):
                    rewards[kk] = -self.INF
            ################################
            
            idx_to_record = np.argmax(rewards)
            
            
            # if we discard all the samples, we do not record the sample 
            if rewards[idx_to_record] != -self.INF:
                data.append({"text": q + tmp_responses[idx_to_record]})
                reward_train.append(rewards[idx_to_record])
                reward_data.append(rewards)

        world_size = int(os.getenv("WORLD_SIZE", "1"))
        all_process_list =[{}] * world_size

        mean_eval_reward = np.mean(reward_eva)

        data_to_send = {
            'data': [[data[i], mean_eval_reward, reward_train[i],reward_data[i]] for i in range(len(data))],
            'local_rank': local_rank
        }
        dist.all_gather_object(all_process_list, data_to_send)
        gathered_data = []
        gathered_reward = []
        gathered_train_reward = []
        gathered_rewards = []
        local_rank_sequence=[]
        for i in range(world_size):
            tmp_data = [tmp[0] for tmp in all_process_list[i]['data']]
            gathered_data.extend(tmp_data)

            tmp_reward = [tmp[1] for tmp in all_process_list[i]['data']]
            gathered_reward.extend(tmp_reward)

            tmp_train_reward = [tmp[2] for tmp in all_process_list[i]['data']]
            gathered_train_reward.extend(tmp_train_reward)

            tmp_rewards = [tmp[3] for tmp in all_process_list[i]['data']]
            gathered_rewards.append(tmp_rewards)

            tmp_local_rank = all_process_list[i]['local_rank']
            local_rank_sequence.append(tmp_local_rank)
        
        sorted_idx = np.argsort(local_rank_sequence)
        reward_list = [gathered_rewards[idx] for idx in sorted_idx]
        merged_reward_data = np.concatenate(reward_list, axis=0)
        logger.info(f"collected data of {len(gathered_data)}")
        logger.info(f"mean reward: {np.mean(gathered_reward)}, reward in train set: {np.mean(gathered_train_reward)}")
        

 
        # We store the training set for monitoring the RAFT training
        output_eval_dataset = {}
        output_eval_dataset['type'] = 'text_only'
        output_eval_dataset['instances'] = gathered_data
        import json
        if local_rank == 0:
            with open(self.raft_filter_samples_store_dir, 'w', encoding='utf8') as f:
                json.dump(output_eval_dataset, f, ensure_ascii=False)

            with open(self.raft_rewards_store_dir, 'a') as f:
                f.write(str(np.mean(gathered_reward)) + "   " + str(np.mean(gathered_train_reward)) + "\n")

            
            np.save(self.raft_rank_store_path, merged_reward_data)    
            
        

        
    def _relabel(
        self,
        #model,
        batch_input,
        alpha=0.2,
        iter_id=0,
        local_rank=0,
        output_min_length=128,
        output_max_length=196,
        infer_batch_size=8,
        generation_kwargs={},
        tokenizer=None,
        training_args=None,
        reward_model=None,
        output_reward_path=None
    ):
        """
        This function compute the reward for the original dataset and relabel them.
        """
        start_time = time.time()
        reward_record = []
        
        all_pos = batch_input['positive']
        all_neg = batch_input['negative']
        cnt = 0
        data = []
        rm_data = []
        for i in range(len(all_pos)):
            if (i+1) % 1000 == 0:
                print(i, cnt / i)
            texts_for_rewards = [all_pos[i], all_neg[i]]

            texts_for_reward_dataset = LMFlowDataset.create_from_dict({
                "type": "text_only",
                "instances": [
                    { "text": text } for text in texts_for_rewards
                ],
            })

            reward_dataset = reward_model.inference(texts_for_reward_dataset)
            rewards = [ sample["value"] for sample in reward_dataset.to_dict()["instances"] ]


            reward_record.append(rewards)

            if rewards[0] >= rewards[1]:
                data.append({"text": all_pos[i]})
                rm_data.append({"positive": all_pos[i], "negative": all_neg[i]})
                cnt += 1
            else:
                data.append({"text": all_neg[i]})
                rm_data.append({"positive": all_neg[i], "negative": all_pos[i]})


        world_size = int(os.getenv("WORLD_SIZE", "1"))
        all_process_list =[{}] * world_size
        

        data_to_send = [[data[i], reward_record[i], rm_data[i]] for i in range(len(data))]
        dist.all_gather_object(all_process_list, data_to_send)
        gathered_data = []
        gathered_reward = []
        gathered_rm_data = []
        for i in range(world_size):
            tmp_data = [tmp[0] for tmp in all_process_list[i]]
            gathered_data.extend(tmp_data)

            tmp_reward = [tmp[1] for tmp in all_process_list[i]]
            gathered_reward.extend(tmp_reward)

            tmp_train_reward = [tmp[2] for tmp in all_process_list[i]]
            gathered_rm_data.extend(tmp_train_reward)

    
        
        logger.info(f"collected data of {len(gathered_data)}")
        #logger.info(f"mean reward: {np.mean(gathered_reward)}, reward in train set: {np.mean(gathered_train_reward)}")
        

 
        # We store the training set for monitoring the RAFT training
        output_eval_dataset = {}
        output_eval_dataset['type'] = 'text_only'
        output_eval_dataset['instances'] = gathered_data

        output_rm_dataset = {}
        output_rm_dataset['type'] = 'text_only'
        output_rm_dataset['instances'] = gathered_rm_data
        import json
        if local_rank == 0:
            with open("/home/xiongwei/LMFlow/data/clean_hh_rlhf_uncerrtainty_study/rm/715_relabel_rm.json", 'w', encoding='utf8') as f:
                json.dump(output_rm_dataset, f, ensure_ascii=False)
            #with open("/home/xiongwei/LMFlow/data/clean_hh_rlhf_uncerrtainty_study/sft/clean_sft.json", 'w', encoding='utf8') as f:
            #    json.dump(output_eval_dataset, f, ensure_ascii=False)
        print(cnt / len(all_pos))

    def _summary(
        self,
        #model,
        batch_input,
        alpha=0.2,
        iter_id=0,
        local_rank=0,
        output_min_length=128,
        output_max_length=196,
        infer_batch_size=8,
        generation_kwargs={},
        tokenizer=None,
        training_args=None,
        reward_model=None,
        output_reward_path=None,
    ):
        """
        :param batch_input: input prompts
        """
        # we will get the batch dataset via Dataset.from_dict
        start_time = time.time()
        reward_eva = []
        reward_train = []
        querys = batch_input['input']
        responses = batch_input['output']
        K = len(responses[0])
        data = []
        for i in range(len(querys)):
            q = querys[i]
            tmp_responses = responses[i]
            #texts_for_rewards = [q + r for r in tmp_responses]
            texts_for_rewards = [q + tmp_responses[0]]

            texts_for_reward_dataset = LMFlowDataset.create_from_dict({
                "type": "text_only",
                "instances": [
                    { "text": text } for text in texts_for_rewards
                ],
            })

            reward_dataset = reward_model.inference(texts_for_reward_dataset)
            rewards = [ sample["value"] for sample in reward_dataset.to_dict()["instances"] ]
            #if "#" in tmp_responses[0]:
            #    rewards[0] = -1
            reward_eva.append(rewards[0])

                
            # we impose some post-detection and discard the samples with certain criteria.
            #for kk in range(K):
            #    if self._discard_sample(tmp_responses[kk]):
            #        rewards[kk] = -self.INF
            ################################
            
            idx_to_record = np.argmax(rewards)
            
            
            # if we discard all the samples, we do not record the sample 
            #if rewards[idx_to_record] != -self.INF:
            data.append({"text": q + tmp_responses[idx_to_record]})
            reward_train.append(rewards[idx_to_record])

        


        world_size = int(os.getenv("WORLD_SIZE", "1"))
        all_process_list =[{}] * world_size


        data_to_send = [[data[i], reward_eva[i], reward_train[i]] for i in range(len(data))]
        dist.all_gather_object(all_process_list, data_to_send)
        gathered_data = []
        gathered_reward = []
        gathered_train_reward = []
        for i in range(world_size):
            tmp_data = [tmp[0] for tmp in all_process_list[i]]
            gathered_data.extend(tmp_data)

            tmp_reward = [tmp[1] for tmp in all_process_list[i]]
            gathered_reward.extend(tmp_reward)

            tmp_train_reward = [tmp[2] for tmp in all_process_list[i]]
            gathered_train_reward.extend(tmp_train_reward)

        
        
        mean_eval_reward = np.mean(gathered_reward)
        std_eval_reward = np.std(gathered_reward)
        mean_train_reward = np.mean(gathered_train_reward)
        logger.info(f"collected data of {len(gathered_data)}")
        logger.info(f"mean reward: {mean_eval_reward}, std of reward: {std_eval_reward}, reward in train set: {mean_train_reward}")
        

 
        # We store the training set for monitoring the RAFT training
        output_eval_dataset = {}
        output_eval_dataset['type'] = 'text_only'
        output_eval_dataset['instances'] = gathered_data
        import json
        if local_rank == 0:
            tmp = np.random.randint(5000)
            with open("record_mmmm.txt", 'a') as f:
                f.write(str(tmp) + f"mean reward: {mean_eval_reward}, std of reward: {std_eval_reward}, reward in train set: {mean_train_reward}" + "\n") 
            np.save(str(tmp) + ".npy", gathered_reward)


    def _get_kl(self, query, responses, model, ref_model, tokenizer, device):
        """
        This function receives:
        query = ###Human:how are you? ###Assistant:
        responses = [Fine, Good, ..., Great] (K responses)
        we compute the conditional KL for each sample and return [KL-1, ..., KL-K]
        """
        kl_seq = []
        #query = test_texts[i]
        with torch.no_grad():
            query_inputs = tokenizer(query, return_tensors="pt", padding=True).to(device)
            query_len = query_inputs['input_ids'].shape[1]
            #print(query_inputs, query_len)
            for i in range(len(responses)):
                one_response = responses[i]
                #res_input = tokenizer(one_response, return_tensors="pt", padding=True).to(0)
                inputs = tokenizer(query + one_response, return_tensors="pt", padding=True).to(device)

                logits = model(**inputs)['logits']
                ref_logits = ref_model(**inputs)['logits']

                input_ids = inputs["input_ids"]
            
                logp = torch.nn.functional.log_softmax(logits[:, :-1, :], dim=2)
                logprobs = torch.gather(logp, 2, input_ids[:, 1:].unsqueeze(2)).squeeze(-1)

                ref_logp = torch.nn.functional.log_softmax(ref_logits[:, :-1, :], dim=2)
                ref_logprobs = torch.gather(ref_logp, 2, input_ids[:, 1:].unsqueeze(2)).squeeze(-1)

                
                # We compute the conditional KL only (p(y|x)) so we start from query_len - 1
                kl_pt = torch.sum(logprobs[:, query_len-1:] - ref_logprobs[:, query_len-1:], axis=1)
                #print(torch.sum(logprobs[:, query_len-1:] , axis=1))
                kl_seq.append(kl_pt.item())
        
        print(kl_seq, device)
        return kl_seq

    def _raft_get_samples_and_kl(
            self,
            model,
            batch_input,
            K=8,
            iter_id=0,
            local_rank=0,
            output_min_length=128,
            output_max_length=196,
            infer_batch_size=8,
            generation_kwargs={},
            tokenizer=None,
            training_args=None,
            reward_model=None,
            output_reward_path=None,
            ref_model=None
        ):
        """
        This function generates a batch of data by best-of-K policy from randomly sampled prompts
        """
        # we will get the batch dataset via Dataset.from_dict
        start_time = time.time()

        querys = batch_input['input']
        data_size = len(querys)

        reward_eva = []
        reward_train = []
        data = []
        input_texts = []
        responses = []
        all_outputs = []

        for i, query in enumerate(querys):
            input_texts = [query for _ in range(K)]
            
            gen_len = np.random.randint(output_min_length, output_max_length)
            generation_kwargs["max_new_tokens"] = gen_len
            inputs = tokenizer(input_texts, return_tensors="pt", padding=True).to(training_args.device)
            with torch.no_grad():
                outputs = model.generate(**inputs, **generation_kwargs)
            generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            generated_texts = [
                generated_text.replace(input_texts[i], "") for i, generated_text in enumerate(generated_texts)
            ]
            generated_texts = [
                self._clean_text(generated_text) for generated_text in generated_texts
            ]
            reward_eva.extend([1 for _ in range(K)])
            
            #q = querys[i]
            #tmp_responses = responses[i]
        
            iter_kl = self._get_kl(query, generated_texts, model, ref_model, tokenizer, training_args.device)
       
       

            data.append({"input": querys[i], "output": [generated_texts[j] for j in range(K)], "kl": iter_kl})

            input_texts = []

        world_size = int(os.getenv("WORLD_SIZE", "1"))
        all_process_data =[{}] * world_size
        dist.all_gather_object(all_process_data, data)

        all_process_eval_reward =[{}] * world_size
        dist.all_gather_object(all_process_eval_reward, reward_eva)
        all_process_train_set_reward =[{}] * world_size
        dist.all_gather_object(all_process_train_set_reward, reward_train)

        
        gathered_data = []
        gathered_reward = []
        gathered_train_reward = []

        for i in range(world_size):
            gathered_data.extend(all_process_data[i])
            gathered_reward.extend(all_process_eval_reward[i])
            gathered_train_reward.extend(all_process_train_set_reward[i])

        output_eval_dataset = {}
        output_eval_dataset['type'] = 'text_only'
        output_eval_dataset['instances'] = gathered_data
        import json
        if local_rank == 0:
            with open(self.raft_infer_samples_store_dir, 'w', encoding='utf8') as f:
                json.dump(output_eval_dataset, f, ensure_ascii=False)

        logger.info(f"collected data of {len(gathered_data)}")

    def _raft_get_kl(
        self,
        batch_input,
        model,
        ref_model,
        local_rank=0,
        tokenizer=None,
        training_args=None,
        output_reward_path=None,
    ):
        """
        This function computes the rewards for the K responses given each prompt.
        We also collect the best of K samples into a filtered dataset.
        """
        # we will get the batch dataset via Dataset.from_dict
        start_time = time.time()
        reward_eva = []
        reward_train = []
        querys = batch_input['input']
        responses = batch_input['output']
        K = len(responses[0])
        data = []
        reward_data = []
        kl = []
        for i in range(len(querys)):
            q = querys[i]
            tmp_responses = responses[i]
        
            iter_kl = self._get_kl(querys[i], responses[i], model, ref_model, tokenizer, training_args.device)
       
       
            data.append({"input": q, "output": tmp_responses, "kl": iter_kl})


        world_size = int(os.getenv("WORLD_SIZE", "1"))
        all_process_list =[{}] * world_size

        mean_eval_reward = np.mean(reward_eva)

        data_to_send = {
            'data': [[data[i], 1, 2, 3] for i in range(len(data))],
            'local_rank': local_rank
        }
        dist.all_gather_object(all_process_list, data_to_send)
        gathered_data = []
        gathered_reward = []
        gathered_train_reward = []
        gathered_rewards = []
        local_rank_sequence=[]
        for i in range(world_size):
            tmp_data = [tmp[0] for tmp in all_process_list[i]['data']]
            gathered_data.extend(tmp_data)

            tmp_reward = [tmp[1] for tmp in all_process_list[i]['data']]
            gathered_reward.extend(tmp_reward)

            tmp_train_reward = [tmp[2] for tmp in all_process_list[i]['data']]
            gathered_train_reward.extend(tmp_train_reward)

            tmp_rewards = [tmp[3] for tmp in all_process_list[i]['data']]
            gathered_rewards.append(tmp_rewards)

            tmp_local_rank = all_process_list[i]['local_rank']
            local_rank_sequence.append(tmp_local_rank)
        
        #sorted_idx = np.argsort(local_rank_sequence)
        ##reward_list = [gathered_rewards[idx] for idx in sorted_idx]
        #merged_reward_data = np.concatenate(reward_list, axis=0)
        #logger.info(f"collected data of {len(gathered_data)}")
        #logger.info(f"mean reward: {np.mean(gathered_reward)}, reward in train set: {np.mean(gathered_train_reward)}")
        

 
        # We store the training set for monitoring the RAFT training
        output_eval_dataset = {}
        output_eval_dataset['type'] = 'text_only'
        output_eval_dataset['instances'] = gathered_data
        import json
        if local_rank == 0:
            with open(self.raft_filter_samples_store_dir, 'w', encoding='utf8') as f:
                json.dump(output_eval_dataset, f, ensure_ascii=False)

           
           
            
           
           
    def align(self, model, dataset, reward_model, tokenizer=None):
        """
        Perform alignment for a model

        Parameters
        ------------
        model : BaseModel object.
        dataset: Dataset object.
            Input dataset for model to generate outputs. The input and output
                will then be feed into reward model to get the reward for
                alignment.
        reward_model: RegressionModel object.
        """
        aligner_args = self.aligner_args
        training_args = aligner_args
        model_args = self.model_args
        data_args = self.data_args
        world_size = int(os.getenv("WORLD_SIZE", "1"))

        self.mode = aligner_args.mode
        self.seed = aligner_args.raft_random_seed

        if not tokenizer:
            tokenizer = model.get_tokenizer()
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
            tokenizer.padding_side = "left"

        dataset = self._load_input_dataset(dataset, tokenizer)
        set_caching_enabled(False)

        wrapped_model = model
        model = model.get_backend_model()

        generation_kwargs = {
            #"min_length": 1,
            "top_k": 0.0,
            "top_p": 1.0,
            "do_sample": True,
            "pad_token_id": tokenizer.eos_token_id,
            "temperature":1.0,
        }


        print(self.mode, self.seed)

        #print(len(dataset['input']))
        share = int(len(dataset) / world_size) 
        dataset = dataset.select(np.arange(training_args.local_rank * share, (training_args.local_rank + 1)*share))

        set_seed(self.seed + training_args.local_rank)
        ###################
        np.random.seed(self.seed + training_args.local_rank)
        ###################
        ITERATION = aligner_args.num_raft_iteration
        collection_strategy = aligner_args.collection_strategy
        sft_batch_size = aligner_args.raft_batch_size

        if collection_strategy == "top":
            alpha = aligner_args.top_reward_percentage
            M = int(sft_batch_size / world_size / alpha)
            K = 1 
        elif collection_strategy == "local":
            K = int(1/aligner_args.top_reward_percentage)
            M = int(sft_batch_size / world_size)
        else:
            raise NotImplementedError("We only support two data collection strategies")


        self.store_dir = aligner_args.output_dir
        self.raft_eval_samples_store_dir = model_args.lora_model_path + "/eval_set/my_eval_set.json"

        self.reward_seq = []
        self.train_reawrd = []
        
        data_size = len(dataset)
        print(data_size)
        lr = training_args.learning_rate

        mode = aligner_args.mode


        assert mode != "xxx"
        if mode == "raft_get_samples":
            shuffled_dataset = dataset.shuffle(seed=self.seed)
            raft_trainer = self._initialize_trainer(model, tokenizer, training_args)
            raft_trainer.train(resume_from_checkpoint=False, is_first_time=True)
            random_idxs = np.arange(data_size)
            np.random.shuffle(random_idxs)
            end_idx = np.min([data_size, M])
            batch_input = shuffled_dataset.select(random_idxs[0 : end_idx])
            
            #batch_input = dataset

            ref_model = AutoModelForCausalLM.from_pretrained('/home/xiongwei/LMFlow/output_models/sft_llama_7b_2e-5_1epoch')
            raft_trainer2 = self._initialize_trainer(ref_model, tokenizer, training_args)
            raft_trainer2.train(resume_from_checkpoint=False, is_first_time=True)
            
            model.gradient_checkpointing_disable()
            model.config.use_cache = True

            start_time = time.time()

            selected_dataset = self._raft_get_samples_and_kl(
                    raft_trainer.tmp_model,
                    batch_input,
                    K,
                    0,
                    training_args.local_rank,
                    output_min_length=aligner_args.output_min_length,
                    output_max_length=aligner_args.output_max_length,
                    infer_batch_size=K,
                    generation_kwargs=generation_kwargs,
                    tokenizer=tokenizer,
                    training_args=training_args,
                    reward_model=reward_model,
                    output_reward_path=aligner_args.output_reward_path,
                    ref_model=ref_model
                    )
            end_time = time.time()
            logger.info("It takes %.2f s to inference one stage", end_time - start_time)
        elif mode == "raft_get_rewards":
            batch_input = dataset
            start_time = time.time()

            selected_dataset = self._raft_get_rewards(
                #raft_trainer.tmp_model,
                batch_input,
                0.2,
                0,
                training_args.local_rank,
                output_min_length=aligner_args.output_min_length,
                output_max_length=aligner_args.output_max_length,
                infer_batch_size=8,
                generation_kwargs=generation_kwargs,
                tokenizer=tokenizer,
                training_args=training_args,
                reward_model=reward_model,
                output_reward_path=aligner_args.output_reward_path,
            )
        elif mode == "eval_get_rewards":
            print("mode: eval_get_rewards")
            batch_input = dataset
            start_time = time.time()

            selected_dataset = self._eval_get_rewards(
                #raft_trainer.tmp_model,
                batch_input,
                0.2,
                0,
                training_args.local_rank,
                output_min_length=aligner_args.output_min_length,
                output_max_length=aligner_args.output_max_length,
                infer_batch_size=8,
                generation_kwargs=generation_kwargs,
                tokenizer=tokenizer,
                training_args=training_args,
                reward_model=reward_model,
                output_reward_path=aligner_args.output_reward_path,
            )
        elif mode == "raft_sample_rewards":
            batch_input = dataset
            start_time = time.time()

            selected_dataset = self._raft_sample_rewards(
                #raft_trainer.tmp_model,
                batch_input,
                0.2,
                0.5,
                0,
                training_args.local_rank,
                output_min_length=aligner_args.output_min_length,
                output_max_length=aligner_args.output_max_length,
                infer_batch_size=8,
                generation_kwargs=generation_kwargs,
                tokenizer=tokenizer,
                training_args=training_args,
                reward_model=reward_model,
                output_reward_path=aligner_args.output_reward_path,
            )
        elif mode == "raft_get_ranks":
            batch_input = dataset
            start_time = time.time()

            selected_dataset = self._raft_get_ranks(
                #raft_trainer.tmp_model,
                batch_input,
                0.2,
                0,
                training_args.local_rank,
                output_min_length=aligner_args.output_min_length,
                output_max_length=aligner_args.output_max_length,
                infer_batch_size=8,
                generation_kwargs=generation_kwargs,
                tokenizer=tokenizer,
                training_args=training_args,
                reward_model=reward_model,
                output_reward_path=aligner_args.output_reward_path,
            )
        elif mode == "eval_get_samples":
            batch_input = dataset
            start_time = time.time()
            raft_trainer = self._initialize_trainer(model, tokenizer, training_args)
            raft_trainer.train(resume_from_checkpoint=False, is_first_time=True)
            selected_dataset = self._eval_get_samples(
                raft_trainer.tmp_model,
                batch_input,
                K,
                0,
                training_args.local_rank,
                output_min_length=aligner_args.output_min_length,
                output_max_length=aligner_args.output_max_length,
                infer_batch_size=K,
                generation_kwargs=generation_kwargs,
                tokenizer=tokenizer,
                training_args=training_args,
                reward_model=reward_model,
                output_reward_path=aligner_args.output_reward_path)

            end_time = time.time()
            logger.info("It takes %.2f s to inference one stage", end_time - start_time)
        elif mode == "relabel":
            batch_input = dataset
            start_time = time.time()

            self._relabel(
                batch_input,
                reward_model = reward_model
            )

            end_time = time.time()
            logger.info("It takes %.2f s to relabel the dataset", end_time - start_time)

        elif mode == 'summary':
            batch_input = dataset
            start_time = time.time()

            self._summary(
                batch_input,
                reward_model=reward_model
            )
            


        return wrapped_model 
