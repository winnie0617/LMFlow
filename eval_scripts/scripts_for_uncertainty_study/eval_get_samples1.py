#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 Statistics and Machine Learning Research Group at HKUST. All rights reserved.
"""Alignment tuning example, such as RLHF."""

import logging
import os
import sys
sys.path.remove(os.path.abspath(os.path.dirname(sys.argv[0])))
from dataclasses import dataclass, field
from typing import Optional

from transformers import HfArgumentParser, pipeline, AutoTokenizer

from lmflow.args import (
    ModelArguments,
    DatasetArguments,
    AutoArguments,
)

from lmflow.datasets.dataset import Dataset
from lmflow.models.auto_model import AutoModel
from lmflow.pipeline.auto_pipeline import AutoPipeline

from peft import LoraConfig, PeftModel, PeftConfig

from trl import AutoModelForCausalLMWithValueHead

@dataclass
class RewardArguments:
    reward_type: Optional[str] = field(
        default="hf_pipeline",
        metadata={
            "help": (
                "type of reward model, support huggingface pipeline. Will"
                " support \"customized\" torch.nn.modules in the future."
            ),
        },
    )
    reward_model_or_path: Optional[str] = field(
        default=None,
        # default="lvwerra/distilbert-imdb", #"weqweasdas/hh_rlhf_rm",
        # default="/home/winnie/output_models/openllama_3b_rm_2sft_full_train_5e-6_1epoch_4x8bs_raw_dataset",

        metadata={
            "help": (
                "reward model name (huggingface) or its path"
            ),
        },
    )
    reward_task: Optional[str] = field(
        default="sentiment-analysis",
        metadata={
            "help": "type of reward task, such as sentiment-analysis, detoxic."
        },
    )
    reward_model_args: Optional[str] = field(
        default="return_all_scores=True, function_to_apply=\"none\", batch_size=1",
        metadata={
            "help": (
                "extra arguments required by different type of reward models."
            ),
        },
    )


def get_reward_function(reward_args, pipeline_args):
    args = reward_args
    reward_type = args.reward_type

    if reward_type == "hf_pipeline":

        # GPT-2 tokenizer has a pad token, but it is not eos_token by default. We need to set it to eos_token.
        # only for this model.
        rm_tokenizer = AutoTokenizer.from_pretrained(reward_args.reward_model_or_path)
        rm_tokenizer.pad_token = rm_tokenizer.eos_token
        rm_tokenizer.pad_token_id = rm_tokenizer.eos_token_id
        rm_tokenizer.padding_side = "left"
        
        hf_pipe = pipeline(
            reward_args.reward_task,
            model=reward_args.reward_model_or_path,
            device=f"cuda:{pipeline_args.local_rank}",
            tokenizer=rm_tokenizer
        )
        def reward_func(dataset: Dataset):
            if dataset.type != "text_only":
                raise NotImplementedError(
                    "reward function only accept \"text_only\" datasets"
                )
            pipe_kwargs = {
                "return_all_scores": True,
                "function_to_apply": "none",
                "batch_size": 1
            }

            data_dict = dataset.to_dict()
            texts_for_rewards = [
                sample["text"] for sample in data_dict["instances"]
            ]
            pipe_outputs = hf_pipe(texts_for_rewards, **pipe_kwargs)
            rewards = [output[0]["score"] for output in pipe_outputs]

            reward_dataset = Dataset.create_from_dict({
                "type": "float_only",
                "instances": [
                    { "value": reward } for reward in rewards
                ]
            })
            return reward_dataset

        return reward_func
    else:
        raise NotImplementedError("unsupported reward type \"{reward_type}\"")


def main():
	# Parses arguments
    pipeline_name = "raft_aligner"
    PipelineArguments = AutoArguments.get_pipeline_args_class(pipeline_name)

    parser = HfArgumentParser((
        ModelArguments,
        DatasetArguments,
        PipelineArguments,
        RewardArguments,
    ))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, pipeline_args, reward_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, pipeline_args, reward_args = parser.parse_args_into_dataclasses()


    # Initializes pipeline, dataset and model for reward training
    aligner = AutoPipeline.get_pipeline(
        pipeline_name=pipeline_name,
        model_args=model_args,
        data_args=data_args,
        pipeline_args=pipeline_args,
    )
    dataset = Dataset(data_args)
    # Load peft model

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    # config.model_name = peft_model_path
    # peft_config = PeftConfig.from_pretrained(peft_model_path)
    
    # peft_model_path = model_args.model_name_or_path
    # model = AutoModelForCausalLMWithValueHead.from_pretrained(
    #     peft_model_path,
    #     # load_in_8bit=True,
    #     peft_config=lora_config,

    # )
    model_args.lora_model_path = model_args.model_name_or_path
    model_args.model_name_or_path = "/home/winnie/output_models/sft_llama_7b_2e-5_1epoch"
    model = AutoModel.get_model(model_args, tune_strategy="none", ds_config=pipeline_args.deepspeed)
    # tokenizer = AutoTokenizer.from_pretrained(peft_model_path, use_fast=False)
    # tokenizer.add_special_tokens(
    #     {
    #         "eos_token": DEFAULT_EOS_TOKEN,
    #         "bos_token": DEFAULT_BOS_TOKEN,
    #         "unk_token": DEFAULT_UNK_TOKEN,
    #         "pad_token": DEFAULT_PAD_TOKEN,
    #     }
    # )
    # tokenizer.padding_side = "left"

    # Initializes reward function
    if reward_args.reward_model_or_path is not None:
        reward_function = get_reward_function(reward_args, pipeline_args)
        reward_model_args = ModelArguments(arch_type="text_regression")
        reward_model = AutoModel.get_model(reward_model_args)
        reward_model.register_inference_function(reward_function)
    else:
        reward_model = None

    # Aligns model with rewards
    aligned_model = aligner.align(
        model=model,
        dataset=dataset,
        # reward_model=reward_model,
        reward_model=reward_model,
        # tokenizer=tokenizer
    )


if __name__ == '__main__':
    main()