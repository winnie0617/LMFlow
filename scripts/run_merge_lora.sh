lora_path=output_models/rm_gpt2_bs_10/rm_0
output_path=output_models/rm_gpt2_bs_10/rm_merged_0
python examples/merge_lora.py \
    --lora_model_path ${lora_path} \
    --tokenizer_name 
    --output_model_path ${output_path}
