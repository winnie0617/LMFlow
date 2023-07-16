from transformers import AutoTokenizer
from datasets import load_dataset
import torch

from value_head import ModelWithValueHead
from transformers import AutoModelForSequenceClassification, pipeline
from trl import AutoModelForCausalLMWithValueHead

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy

def build_dataset(model_name, split="train", data_dir="harmless-base", input_min_text_length=2, input_max_text_length=8):
    """
    Build dataset for training. This builds the dataset from `load_dataset`, one should
    customize this function to train the model on its own dataset.

    Args:
        dataset_name (`str`):
            The name of the dataset to be loaded.

    Returns:
        dataloader (`torch.utils.data.DataLoader`):
            The dataloader for the dataset.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    ds = load_dataset("Anthropic/hh-rlhf", data_dir=data_dir, split=split, use_auth_token=True)
    def tokenize(sample):
        tokenized_pos = tokenizer(sample['chosen'], truncation=True)
        # tokenized_neg = tokenizer(sample['rejected'], truncation=True)
        # tokenized_pos = tokenizer(sample['positive'], truncation=True)
        # tokenized_neg = tokenizer(sample['negative'], truncation=True)
        sample["chosen_input_ids"] = tokenized_pos["input_ids"] # need to be 2d
        # sample["chosen_attention_mask"] = [tokenized_pos["attention_mask"]]
        # sample["rejected_input_ids"] = [tokenized_neg["input_ids"]]
        # sample["rejected_attention_mask"] = [tokenized_neg["attention_mask"]]
        return sample

    # ds = load_dataset("json", data_files="data/anthropic/harmless-base/rm/train/dataset.json", split=split, field="instances")
    ds = ds.map(tokenize, batched=False)
    # format = {'type': 'torch'}
    # ds.set_format(**format)
    # ds = ds.filter(lambda x: len(x["chosen_input_ids"]) <= 512 and len(x["rejected_input_ids"]) <= 512)
    ds = ds.filter(lambda x: len(x["chosen_input_ids"]) <= 512)

    return ds

if __name__ == "__main__":
    model_name = "/home/winnie/LMFlow/output_models/finetune-gpt-neo"
    rm_name = "/home/winnie/LMFlow/output_models/rm_finetune-gpt-neo_bs_7_20"
    k = 7
    device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"  # to avoid a `pipeline` bug

    # We retrieve the dataloader by calling the `build_dataset` function.
    train_data = build_dataset(model_name, split="train")
    test_data = build_dataset(model_name, split="test", data_dir="helpful-base")


    # models = []
    # for i in range(k):
    #     model = AutoModelForSequenceClassification.from_pretrained(f"output_models/rm_finetune-gpt-neo_bs_1/rm_{i}").to(device)
    #     print(model)
    #     models.append(model)

    sent_kwargs = {"return_all_scores": True, "function_to_apply": "none", "batch_size": 1}

    sentiment_pipes = []
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    for i in range(k):
        sentiment_pipes.append(pipeline("sentiment-analysis", model=f"{rm_name}/rm_{i}", tokenizer=tokenizer, device=device))
  

    # Calculate reward std
    plt.figure(0)
    # plt.figure(1)
    # plt.figure(2)
    bins = np.linspace(0, 1, 10)
    d = {"train": train_data, "test": test_data}
    for label, data in d.items():
        print(label)
        stds = []
        means = []
        lengths = []
        acc = []
        n = 1000
        # for _, chosen_input in enumerate(zip(data["chosen"][:n], data["rejected"][:n])):
        for entry in data.select(range(n)):
            rewards = torch.zeros(k, 1)
            for i, sentiment_pipe in enumerate(sentiment_pipes):
                pipe_outputs = sentiment_pipe([entry["chosen"], entry["rejected"]], **sent_kwargs)
                chosen_score = torch.tensor(pipe_outputs[0][0]["score"])
                reject_score = torch.tensor(pipe_outputs[1][0]["score"])
                rewards[i,:] = chosen_score >= reject_score
            
            s = rewards.float().sum().item()
            freq = [s / k, (k-s) / k]
            rewards_std = entropy(freq)
            stds.append(rewards_std)
            acc.append([r.item() for r in rewards])

        plt.figure(0)
        plt.hist(stds, bins, label=label, alpha=0.7)
        acc = np.array(acc)
        print(acc.mean(axis=0))
    
    plt.figure(0)
    plt.title('Entropy')
    plt.legend(loc='upper right')
    plt.savefig("entropy_plot.png")

    # plt.figure(1)
    # plt.title('Std against Mean')
    # plt.legend(loc='upper right')
    # plt.savefig("mean_plot.png")

    # plt.figure(2)
    # plt.title('Mean against Response Length')
    # plt.legend(loc='upper right')
    # plt.savefig("mean_length_plot.png")

    # plt.figure(3)
    # plt.title('Std against Response Length')
    # plt.legend(loc='upper right')
    # plt.savefig("length_plot.png")


    

