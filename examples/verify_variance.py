from transformers import AutoTokenizer
from datasets import load_dataset
import torch

from value_head import ModelWithValueHead
from transformers import AutoModel, pipeline
from trl import AutoModelForCausalLMWithValueHead

# import matplotlib.pyplot as plt
import numpy as np

import csv

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
    # load imdb with datasets
    ds = load_dataset("Anthropic/hh-rlhf", data_dir=data_dir, split=split, use_auth_token=True)
    def tokenize(sample):
        sample["chosen_ids"] = tokenizer.encode(sample["rejected"])
        return sample

    ds = ds.map(tokenize, batched=False)
    ds.set_format(type="torch")
    ds = ds.filter(lambda x: len(x["chosen_ids"]) <= 512)
    return ds

if __name__ == "__main__":
    model_name = "/home/winnie/LMFlow/output_models/finetune-gpt-neo"
    rm_name = "/home/winnie/LMFlow/output_models/rm_finetune-gpt-neo_bs_7_20"
    k = 7
    
    device = 0 if torch.cuda.is_available() else "cpu"  # to avoid a `pipeline` bug

    # We retrieve the dataloader by calling the `build_dataset` function.
    train_data = build_dataset(model_name, split="train")
    test_data = build_dataset(model_name, split="test", data_dir="helpful-base")

    # Load reward models and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    pretrained = AutoModel.from_pretrained("/home/winnie/LMFlow/output_models/finetune-gpt-neo", torch_dtype=torch.bfloat16).to(device)
    pretrained.config.pad_token_id = tokenizer.eos_token_id

    # models = []

    # for i in range(k):
    #     model = AutoModelForCausalLMWithValueHead.from_pretrained(f"output_models/rm_finetune-gpt-neo_bs_3_merged/rm_{i}").to(device)
    #     models.append(model)
    sent_kwargs = {"return_all_scores": True, "function_to_apply": "none", "batch_size": 1}

    sentiment_pipes = []
    for i in range(k):
        sentiment_pipes.append(pipeline("sentiment-analysis", model=f"{rm_name}/rm_{i}", device=device, tokenizer=tokenizer))
    # dir = "/home/share/rewardmodels"
    # files = ["rm_2sft_full_train_2epoch_gpt_neo_2_7B_exp5", "rm_2sft_full_train_2epoch_gpt_neo_2_7B_exp7", "rm_2sft_full_train_2epoch_gpt_neo_2_7B_exp6"]
    # import os
    # # for filename in os.listdir(dir):
    # for filename in files:
    #     f = os.path.join(dir, filename)
    #     sentiment_pipes.append(pipeline("sentiment-analysis", model=f, device=device, tokenizer=tokenizer))

  

    # Calculate reward std
    bins = np.linspace(0, 1, 100)
    d = {"train": train_data, "test": test_data}
    for label, data in d.items():
        print(label)
        stds = []
        means = []
        lengths = []
        res = []
        n = 2000
        for input_num, chosen_input in enumerate(data["chosen"][:n]):
            rewards = torch.zeros(k, 1)
            res_curr = {} # store res
            for i, sentiment_pipe in enumerate(sentiment_pipes):
                pipe_outputs = sentiment_pipe(chosen_input, **sent_kwargs)
                r = torch.tensor([output[0]["score"] for output in pipe_outputs])
                rewards[i,:] = r
                res_curr[i] = r.float().item()

            # if rewards_std.float().item() > 0.5 or rewards_std.float().item() < 0.15:
            #     print(chosen_input)
            #     print(rewards_std.float().item())
            #     q(rewards)


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


    

