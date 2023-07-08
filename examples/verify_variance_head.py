from transformers import AutoTokenizer
from datasets import load_dataset
import torch

from value_head import ModelWithValueHead
from transformers import AutoModel

import matplotlib.pyplot as plt
import numpy as np

import csv

def build_dataset(model_name, split="train", input_min_text_length=2, input_max_text_length=8):
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
    # load imdb with datasets
    ds = load_dataset("Anthropic/hh-rlhf", data_dir="harmless-base", split=split, use_auth_token=True)
    def tokenize(sample):
        sample["chosen_ids"] = tokenizer.encode(sample["rejected"])
        return sample

    ds = ds.map(tokenize, batched=False)
    ds.set_format(type="torch")
    ds = ds.filter(lambda x: len(x["chosen_ids"]) <= 512)
    return ds

if __name__ == "__main__":
    model_name = "/home/winnie/LMFlow/output_models/finetune-gpt-neo"
    rm_name = "/home/winnie/LMFlow/output_models/rm_head_1layer_finetune-gpt-neo_bs_10"
    k = 7
    device = 0 if torch.cuda.is_available() else "cpu"  # to avoid a `pipeline` bug

    # We retrieve the dataloader by calling the `build_dataset` function.
    train_data = build_dataset(model_name, split="train")
    test_data = build_dataset(model_name, split="test")

    # Load reward models and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    pretrained = AutoModel.from_pretrained("/home/winnie/LMFlow/output_models/finetune-gpt-neo", torch_dtype=torch.bfloat16).to(device)
    pretrained.config.pad_token_id = tokenizer.eos_token_id


    models = []
    for i in range(k):
        model = ModelWithValueHead(pretrained, "output_models/finetune-gpt-neo/config.json")
        # loaded_state = torch.load(f"{rm_name}/head_{i}")
        loaded_state = torch.load(f"{rm_name}/head_{i}", map_location="cuda:0")
        model.head.load_state_dict(loaded_state)
        model.to(device)
        models.append(model)
  

    # Calculate reward std
    plt.figure(0)
    plt.figure(1)
    plt.figure(2)
    bins = np.linspace(0, 1, 100)
    d = {"train": train_data, "test": test_data}
    d = {"test": test_data}
    for label, data in d.items():
        print(label)
        stds = []
        means = []
        lengths = []
        res = []
        n = 2000
        for input_num, chosen_input in enumerate(data["rejected"][:n]):
            rewards = torch.zeros(k, 1)
            res_curr = {} # store res
            for i, model in enumerate(models):
                model_inputs = tokenizer(chosen_input, padding=True, truncation=True, return_tensors="pt")
                model_inputs.to(device)
                model_outputs = model(input_ids=model_inputs["input_ids"], attention_mask=model_inputs["attention_mask"])
                r = torch.tensor(model_outputs)
                rewards[i,:] = r
                res_curr[i] = r.float().item()

            # rewards_std = rewards.std(axis=0)
            # if rewards_std.float().item() > 0.5 or rewards_std.float().item() < 0.15:
            #     print(chosen_input)
            #     print(rewards_std.float().item())
            #     print(rewards)

            query, response = chosen_input.rsplit("\nAssistant: ", 1)

            # stds.append(rewards_std.float().item())
            # means.append(rewards.mean(axis=0).float().item())
            # lengths.append(len(response))
            res_curr["length"] = len(response)
            res.append(res_curr)
        
        with open(f'bootstrap_res_{label}.csv', 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=res[0].keys())
            writer.writeheader()
            writer.writerows(res)

            
    #     plt.figure(0)
    #     plt.hist(stds, bins, label=label, alpha=0.7)
    #     plt.figure(1)
    #     plt.scatter(means, stds, label=label, alpha=0.5)
    #     plt.figure(2)
    #     plt.scatter(lengths, means, label=label, alpha=0.5)
    #     plt.figure(3)
    #     plt.scatter(lengths, stds, label=label, alpha=0.5)
    
    # plt.figure(0)
    # plt.title('Std Distribution')
    # plt.legend(loc='upper right')
    # plt.savefig("std_plot.png")

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




    

