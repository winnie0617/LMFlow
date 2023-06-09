from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
import os
import sys
sys.path.remove(os.path.abspath(os.path.dirname(sys.argv[0])))
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset
# from peft import LoraConfig, TaskType, get_peft_model

from transformers.modeling_utils import SequenceSummary


from transformers import (
    AutoModel,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    PretrainedConfig
)
from lmflow.args import (
    ModelArguments,
    DatasetArguments,
    AutoArguments,
)

from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
import wandb

k = 1 #TODO: remove hard-coding

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
# Load in model without head
model = AutoModel.from_pretrained(model_args.model_name_or_path, num_labels=1, torch_dtype=torch.bfloat16)

# model_lora = get_peft_model(model, peft_config)
# model_lora.print_trainable_parameters()

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
model.config.pad_token_id = tokenizer.eos_token_id
assert model.config.pad_token_id == tokenizer.pad_token_id

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
def compute_accuracy(outputs):
    # {"chosen_rewards": chosen_rewards, "rejected_rewards": rejected_rewards}
    result = {}
    pos_predictions_scores = outputs["chosen_rewards"].cpu().numpy()[0]
    neg_predictions_scores = outputs["rejected_rewards"].cpu().numpy()[0]
    # We assume that the first sample is preferred by default in groundtruth
    return np.sum(pos_predictions_scores >= neg_predictions_scores) / len(pos_predictions_scores)
    
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
        batch['chosen_input_ids'] = batch_pos['input_ids'].to(device)
        batch['rejected_input_ids'] = batch_neg['input_ids'].to(device)
        batch['chosen_attention_mask'] = batch_pos['attention_mask'].to(device)
        batch['rejected_attention_mask'] = batch_neg['attention_mask'].to(device)
        batch['return_loss'] = True
        return batch

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


def compute_loss(pretrained, value_head, inputs, return_outputs=False):
    chosen_hidden = pretrained(input_ids=inputs["chosen_input_ids"], attention_mask=inputs["chosen_attention_mask"])[0]
    rejected_hidden = pretrained(input_ids=inputs["rejected_input_ids"], attention_mask=inputs["rejected_attention_mask"])[0]
    chosen_rewards = value_head(chosen_hidden, None).squeeze(-1)
    rejected_rewards = value_head(rejected_hidden, None).squeeze(-1)
    loss = -nn.functional.logsigmoid(chosen_rewards - rejected_rewards).mean()
    if return_outputs:
        return loss, {"chosen_rewards": chosen_rewards, "rejected_rewards": rejected_rewards}
    return loss

def train(pretrained, train_data, eval_data):
    
    config = PretrainedConfig.from_pretrained("output_models/finetune-gpt-neo/config.json") # TODO: remove hard code
    config.num_labels = 1
    value_head = SequenceSummary(config).to(device)
    # print(value_head)

    train_dataloader = DataLoader(train_data, batch_size=pipeline_args.per_device_train_batch_size, collate_fn=data_collator)
    eval_dataloader = DataLoader(eval_data, batch_size=pipeline_args.per_device_eval_batch_size, collate_fn=data_collator)

    num_epochs = int(pipeline_args.num_train_epochs)
    acc_steps = pipeline_args.gradient_accumulation_steps
    num_training_steps = num_epochs * len(train_dataloader) // acc_steps
    optimizer = AdamW(list(value_head.parameters()), lr=pipeline_args.learning_rate, weight_decay=pipeline_args.weight_decay)
    lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

    best_val_loss = float("inf")

    # Logging
    run = wandb.init(
        # Set the project where this run will be logged
        project="rm_head",
        # Track hyperparameters and run metadata
        config={
            "learning_rate": pipeline_args.learning_rate,
            "per_device_train_batch_size": pipeline_args.per_device_train_batch_size,
            "gradient_accumulation_steps": pipeline_args.gradient_accumulation_steps,
            "epochs": num_epochs,
        })
    # wandb.init(settings=wandb.Settings(start_method="fork"))
    

    with tqdm(range(num_training_steps)) as progress_bar:
        loss_log = 0
        for epoch in range(num_epochs):
            # training
            value_head.train()
            for batch_i, batch in enumerate(train_dataloader):
                # Forward Pass
                loss = compute_loss(pretrained, value_head, batch, return_outputs=False)
                loss_log += loss
                # Logging
                if ((batch_i+1) % (pipeline_args.logging_steps * acc_steps) == 0):
                    wandb.log({"train/avg_loss": loss_log / pipeline_args.logging_steps / acc_steps,})
                    loss_log = 0
                # Normalize Gradients
                loss = loss / acc_steps
                loss.backward()
                # Gradient Accumulation
                if ((batch_i+1) % acc_steps == 0) or (batch_i+1 == len(train_dataloader)):
                    optimizer.step()
                    lr_scheduler.step()
                    progress_bar.update(1)
                    # Reset gradients, for the next accumulated batches
                    optimizer.zero_grad()

                # validation
                if (batch_i+1) % (pipeline_args.eval_steps * acc_steps) == 0 or (batch_i+1 == len(train_dataloader)):
                    value_head.eval()
                    curr_acc = 0
                    for batch_i, batch in enumerate(eval_dataloader):
                        with torch.no_grad():
                            # {"chosen_rewards": chosen_rewards, "rejected_rewards": rejected_rewards}
                            curr_loss, outputs = compute_loss(pretrained, value_head, batch, return_outputs=True)
                            loss += curr_loss
                            curr_acc += compute_accuracy(outputs)
                    avg_val_loss = loss / len(eval_dataloader)
                    wandb.log({"val/loss": avg_val_loss,})
                    wandb.log({"val/accuracy": curr_acc / len(eval_dataloader)})
                    tqdm.write(f"Validation loss: {avg_val_loss}")
                    # if avg_val_loss < best_val_loss:
                    #     print("Saving checkpoint!")
                    #     best_val_loss = avg_val_loss
                    #     torch.save({
                    #         'epoch': epoch,
                    #         'model_state_dict': value_head.state_dict(),
                    #         'optimizer_state_dict': optimizer.state_dict(),
                    #         'val_loss': best_val_loss,
                    #         },
                    #         f"{pipeline_args.output_dir}/head_{i}/checkpoints/epoch_{epoch}.pt"
                    #     )  
                    value_head.train()
    return value_head
# GPU
rank = torch.cuda.current_device()
device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')
# Instantiate model without head to prevent multiple copies
pretrained = AutoModel.from_pretrained("output_models/finetune-gpt-neo").to(device)
# # Freeze all layers
# for param in pretrained.parameters():
#     param.requires_grad = False
    
# for i in range(k):
    
# Subsample from train and eval datasets separately
idx_train = np.random.randint(len(train_dataset), size=len(train_dataset))
train_sub = train_dataset.select(idx_train)
idx_eval = np.random.randint(len(eval_dataset), size=len(eval_dataset))
eval_sub = eval_dataset.select(idx_eval)

value_head = train(pretrained, train_sub, eval_sub)
## Save model
torch.save(value_head.state_dict(), f"{pipeline_args.output_dir}/head_{rank}")




