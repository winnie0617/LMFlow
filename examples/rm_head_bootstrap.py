from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
import os
import sys
# sys.path.remove(os.path.abspath(os.path.dirname(sys.argv[0])))
import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
# from peft import LoraConfig, TaskType, get_peft_model

from transformers import (
    AutoModel,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
)
from lmflow.args import (
    ModelArguments,
    DatasetArguments,
    AutoArguments,
)

from value_head import ModelWithValueHead

k = 3 #TODO: remove hard-coding

## Prepare training_args
pipeline_name = "finetuner"
PipelineArguments = AutoArguments.get_pipeline_args_class(pipeline_name)
parser = HfArgumentParser((ModelArguments, DatasetArguments, PipelineArguments))

if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
    model_args, data_args, pipeline_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
else:
    model_args, data_args, pipeline_args = parser.parse_args_into_dataclasses()

pipeline_args.remove_unused_columns = False 
pipeline_args.label_names = []

## Get model, by default we use lora to accelerate training
# peft_config = LoraConfig(
#     task_type=TaskType.SEQ_CLS,
#     r=16,
#     lora_alpha=32,
#     lora_dropout=0.1,
# )
# trust_remote_code=True if you want to use chatglm
pretrained = AutoModel.from_pretrained(model_args.model_name_or_path, torch_dtype=torch.bfloat16)
# Freeze all layers
for param in pretrained.parameters():
    param.requires_grad = False


## Get tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)

if "llama" in model_args.model_name_or_path:
    tokenizer.add_special_tokens(
        {
            "eos_token": "[PAD]",
            "bos_token": "</s>",
            "unk_token": "</s>",
            "pad_token": "</s>",
        }
    )
else:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

# We also need to add a pad_token for the model. Otherwise, the reward model cannot handle a batch of inputs
pretrained.config.pad_token_id = tokenizer.eos_token_id
assert pretrained.config.pad_token_id == tokenizer.pad_token_id

## Get the dataset
#TODO: change this to bootstrapping
def build_dataset(tokenizer, config):
    ''' 
    Return whole dataset without validation split
    We assume that we have preprocessed the dataset appropriately such that the sample is organized as follows:
    {"positive": prompt + answer_positive, "negative": prompt + answer_negative}, where the positive response is preferred.
    '''
    def tokenize(sample):
        tokenized_pos = tokenizer(sample['positive'], truncation=True)
        tokenized_neg = tokenizer(sample['negative'], truncation=True)
        sample["chosen_input_ids"] = tokenized_pos["input_ids"]
        sample["chosen_attention_mask"] = tokenized_pos["attention_mask"]
        sample["rejected_input_ids"] = tokenized_neg["input_ids"]
        sample["rejected_attention_mask"] = tokenized_neg["attention_mask"]
        return sample

    ds = load_dataset("json", data_files=config.dataset_path, split="train", field="instances")
    ds = ds.map(tokenize, batched=False)
    ds = ds.filter(lambda x: len(x["chosen_input_ids"]) <= 512 and len(x["rejected_input_ids"]) <= 512)

    return ds

## Define the trainer
def compute_metrics(eval_pred):
    result = {}
    pos_predictions_scores = eval_pred.predictions[0]
    neg_predictions_scores = eval_pred.predictions[1]
    # We assume that the first sample is preferred by default in groundtruth
    result['accuracy'] = np.sum(pos_predictions_scores >= neg_predictions_scores) / len(pos_predictions_scores)
    return result
    
class DataCollatorReward:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, data):
        batch = {}
        data_pos = []
        data_neg = []
        for sample in data:
            data_pos.append({"input_ids": sample['chosen_input_ids'], "attention_mask": sample["chosen_attention_mask"]})
            data_neg.append({"input_ids": sample['rejected_input_ids'], "attention_mask": sample["rejected_attention_mask"]})
        batch_pos = self.tokenizer.pad(data_pos, padding=True, return_tensors="pt")
        batch_neg = self.tokenizer.pad(data_neg, padding=True, return_tensors="pt")
        batch['chosen_input_ids'] = batch_pos['input_ids']
        batch['rejected_input_ids'] = batch_neg['input_ids']
        batch['chosen_attention_mask'] = batch_pos['attention_mask']
        batch['rejected_attention_mask'] = batch_neg['attention_mask']
        batch['return_loss'] = True
        return batch


class RMTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        chosen_rewards = model(input_ids=inputs["chosen_input_ids"], attention_mask=inputs["chosen_attention_mask"])[0]
        rejected_rewards = model(input_ids=inputs["rejected_input_ids"], attention_mask=inputs["rejected_attention_mask"])[0]
        loss = -nn.functional.logsigmoid(chosen_rewards - rejected_rewards).mean()
        if return_outputs:
            return loss, {"chosen_rewards": chosen_rewards, "rejected_rewards": rejected_rewards}
        return loss

data_collator = DataCollatorReward(tokenizer=tokenizer)

ds = build_dataset(tokenizer, data_args)
eval_dataset = None

if data_args.validation_split_percentage > 0:
    idx_gap = int((1-data_args.validation_split_percentage/100) * len(ds))
    train_dataset = ds.select(range(idx_gap))
    eval_dataset = ds.select(range(idx_gap, len(ds)))
else:
    train_dataset = ds

print("Training set: ", len(train_dataset), " Eval set: ", len(eval_dataset))
if not eval_dataset and pipeline_args.eval_steps > 0:
    raise valueerror("Cannot evaluate on an empty eval set")

for i in range(k):
    
    # Subsample from train and eval datasets separately
    idx_train = np.random.randint(len(train_dataset), size=len(train_dataset))
    train_sub = train_dataset.select(idx_train)
    idx_eval = np.random.randint(len(eval_dataset), size=len(eval_dataset))
    eval_sub = eval_dataset.select(idx_eval)
    
    model = ModelWithValueHead(pretrained, "output_models/finetune-gpt-neo/config.json") # TODO: remove hard coding

    trainer = RMTrainer(
        model=model,
        args=pipeline_args,
        train_dataset=train_sub,
        compute_metrics=compute_metrics,
        eval_dataset=eval_sub,
        data_collator=data_collator,
    )

    trainer.train()

    ## Save model
    # model_lora.save_pretrained(f"{pipeline_args.output_dir}/rm_{i}")
    model.save_head(f"{pipeline_args.output_dir}/head_{i}")

