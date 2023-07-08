from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, HfArgumentParser
from peft import PeftConfig, PeftModel
import torch

model = AutoModelForSequenceClassification.from_pretrained("output_models/finetune-gpt-neo", num_labels=1, torch_dtype=torch.bfloat16)

lora_path = "output_models/rm_finetune-gpt-neo_bs_6_old"
output_path = "output_models/rm_finetune-gpt-neo_bs_6"

for i in range(6):
    model_lora = PeftModel.from_pretrained(model, f"{lora_path}/rm_{i}")
    model_lora.eval()
    model_merged = model_lora.merge_and_unload()
    print(model_merged)
    model_merged.save_pretrained(f"{output_path}/rm_{i}")