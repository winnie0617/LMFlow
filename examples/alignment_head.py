# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from dataclasses import dataclass, field
from typing import Optional

import torch
from datasets import load_dataset
from peft import LoraConfig
from tqdm import tqdm
from transformers import AutoTokenizer, HfArgumentParser, pipeline

from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer, set_seed
from trl.core import LengthSampler

from value_head import ModelWithValueHead
from transformers import AutoModel, PretrainedConfig


########################################################################
# This is a fully working simple example to use trl with accelerate.
#
# This example fine-tunes a GPT2 model on the IMDB dataset using PPO
# (proximal policy optimization).
# in any of the following settings (with the same script):
#   - single CPU or single GPU
#   - fp16 (mixed-precision) or fp32 (normal precision)
#
# To run it in each of these various modes, first initialize the accelerate
# configuration with `accelerate config`
#
########################################################################

########################################################################
# NOTE for to train with a 8-bit model a more recent version of
# transformers is required, full dependecies for this example:
# pip install  bitsandbytes datasets accelerate loralib
# pip install  git+https://github.com/huggingface/transformers.git@main
# pip install peft
########################################################################

# We first define the configuration of the experiment, defining the model, the dataset,
# the training parameters, and the PPO parameters.
# Check the default arguments in the `PPOConfig` class for more details.
# If you want to log with tensorboard, add the kwarg
# `accelerator_kwargs={"logging_dir": PATH_TO_LOGS}` to the PPOConfig.


# Define and parse arguments.
@dataclass
class ScriptArguments:
    """
    The name of the Casual LM model we wish to fine with PPO
    """

    # NOTE: gpt2 models use Conv1D instead of Linear layers which are not yet supported in 8 bit mode
    # models like gpt-neo* models are more suitable
    # model_name: Optional[str] = field(default="edbeeching/gpt-neo-1.3B-imdb", metadata={"help": "the model name"})
    log_with: Optional[str] = field(default="wandb", metadata={"help": "use 'wandb' to log with wandb"})
    # learning_rate: Optional[float] = field(default=1.41e-6, metadata={"help": "the learning rate"})
    merge_model_adapter: Optional[bool] = field(default=True, metadata={"help": "the learning rate"})


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

gpt = True
if gpt:
    model_name = "/home/winnie/LMFlow/output_models/finetune-gpt-neo"
else:
    model_name = "/home/winnie/trl/examples/sentiment/scripts/my_models/finetune-with-lora-llama-merged"
rm_name = "/home/winnie/LMFlow/output_models/rm_head_1layer_finetune-gpt-neo_bs_10"
k = 3


config = PPOConfig(
    model_name=model_name,
    learning_rate=1e-5,
    # init_kl_coef=0.05,
    log_with=script_args.log_with,
    batch_size=1,
    mini_batch_size=1,
    gradient_accumulation_steps=32,
    optimize_cuda_cache=True,
)

# Configure tokenizer
tokenizer = AutoTokenizer.from_pretrained(config.model_name)
if gpt:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

# We then define the arguments to pass to the sentiment analysis pipeline.
# We set `return_all_scores` to True to get the sentiment score for each token.
sent_kwargs = {"return_all_scores": True, "function_to_apply": "none", "batch_size": config.mini_batch_size}


# Below is an example function to build the dataset. In our case, we use the IMDB dataset
# from the `datasets` library. One should customize this function to train the model on
# its own dataset.
def build_dataset(config, dataset_name="imdb", input_min_text_length=2, input_max_text_length=8):
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
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if gpt:
        tokenizer.pad_token = tokenizer.eos_token
    # load imdb with datasets
    ds = load_dataset("Anthropic/hh-rlhf", data_dir="harmless-base", split="train", use_auth_token=True)
    def tokenize(sample):
        # query = sample["chosen"].split("\nAssistant: ")[0].split("\nHuman: ")[1]
        query, response = sample["chosen"].rsplit("\nAssistant: ", 1)
        sample["query"] = query + "\nAssistant: "
        sample["input_ids"] = tokenizer.encode(sample["query"])
        return sample

    ds = ds.map(tokenize, batched=False)
    ds.set_format(type="torch")
    return ds


# We retrieve the dataloader by calling the `build_dataset` function.
dataset = build_dataset(config)


def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])


# set seed before initializing value head for deterministic eval
set_seed(config.seed)


"""### Apply LoRA
Here comes the magic with `peft`! Let's load a `PeftModel` and specify that we are going to use low-rank adapters (LoRA) using `get_peft_model` utility function from `peft`.
"""


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=['q_proj', 'v_proj'],
)

# Now let's build the model, the reference model, and the tokenizer.
model = AutoModelForCausalLMWithValueHead.from_pretrained(
    config.model_name,
    load_in_8bit=True,
    device_map="balanced",
    # max_memory={0: "800MB", 1: "800MB"},
    peft_config=lora_config,
)

print_trainable_parameters(model)

# GPT-2 tokenizer has a pad token, but it is not eos_token by default. We need to set it to eos_token.
# only for this model.
if gpt:
    tokenizer.pad_token = tokenizer.eos_token

optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config.learning_rate)
# We then build the PPOTrainer, passing the model, the reference model, the tokenizer
ppo_trainer = PPOTrainer(
    config, model, ref_model=None, tokenizer=tokenizer, dataset=dataset, data_collator=collator, optimizer=optimizer
)



# We then build the sentiment analysis pipeline, passing the model name and the
# sentiment analysis pipeline arguments. Let's also make sure to set the device
# to the same device as the PPOTrainer.
device = ppo_trainer.accelerator.device
if ppo_trainer.accelerator.num_processes == 1:
    device = 0 if torch.cuda.is_available() else "cpu"  # to avoid a `pipeline` bug

# Use the same pretrained model object to
pretrained = AutoModel.from_pretrained("/home/winnie/LMFlow/output_models/finetune-gpt-neo", torch_dtype=torch.bfloat16).to(device)
# head_config = PretrainedConfig.from_pretrained("output_models/finetune-gpt-neo/config.json")

# We also need to add a pad_token for the model. Otherwise, the reward model cannot handle a batch of inputs
pretrained.config.pad_token_id = tokenizer.eos_token_id
assert pretrained.config.pad_token_id == tokenizer.pad_token_id

models = []
for i in range(k):
    model = ModelWithValueHead(pretrained, "output_models/finetune-gpt-neo/config.json")
    # loaded_state = torch.load(f"{rm_name}/head_{i}")
    loaded_state = torch.load(f"{rm_name}/head_{i}", map_location="cuda:0")
    model.head.load_state_dict(loaded_state)
    model.to(device)
    models.append(model)
    # print(model)


# We then define the arguments to pass to the `generate` function. These arguments
# are passed to the `generate` function of the PPOTrainer, which is a wrapper around
# the `generate` function of the trained model.
generation_kwargs = {
    "min_length": -1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
    "eos_token_id": -1,
}
output_min_length = 10
output_max_length = 128
output_length_sampler = LengthSampler(output_min_length, output_max_length)


for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
    query_tensors = batch["input_ids"]

    # model.gradient_checkpointing_disable()
    # model.pretrained_model.config.use_cache = True
    # Get response from Causal LM
    response_tensors = []
    for query in query_tensors:
        gen_len = output_length_sampler()
        generation_kwargs["max_new_tokens"] = gen_len
        response = ppo_trainer.generate(query, **generation_kwargs)
        response_tensors.append(response.squeeze()[-gen_len:])
    batch["response"] = [tokenizer.decode(r.squeeze()) for r in response_tensors]
    # Compute sentiment score
    texts = [q + r for q, r in zip(batch["query"], batch["response"])]
    # Get hidden layer from pretrained
    # 

    rewards = torch.zeros(k, config.batch_size)
    for i, model in enumerate(models):
        model_inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        model_inputs.to(device)
        model_outputs = model(input_ids=model_inputs["input_ids"], attention_mask=model_inputs["attention_mask"])
        rewards[i,:] = torch.tensor([output for output in model_outputs])

    rewards_mean = rewards.mean(axis=0)
    rewards_std = rewards.std(axis=0)

    # print("mean")
    # print(rewards_mean)
    # print("std")
    # print(rewards_std)
    penalized_reward = rewards_mean - 0.1 * rewards_std # TODO: remove hard-coding
    # print("penalized")
    # print(penalized_reward)

    for t, r, s in zip(texts, torch.transpose(rewards, 0, 1), rewards_std):
        print(t)
        print(r)
        print(s)
    
    penalized_reward = [torch.tensor(m, dtype=torch.float16) for m in penalized_reward] # reward has to be a list of tensor

    # Run PPO step
    # model.gradient_checkpointing_enable()
    # model.pretrained_model.config.use_cache = False

    stats = ppo_trainer.step(query_tensors, response_tensors, penalized_reward)
    ppo_trainer.log_stats(stats, batch, penalized_reward, rewards_std) # TODO: update logger

# model.push_to_hub(f"{script_args.model_name}-ppo-sentiment")
