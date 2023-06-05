# lora_path=output_models/rm
# output_path=output_models/rm_merged
# base_path=gpt2
lora_path=output_models/rm_finetune-gpt-neo_bs_2
output_path=output_models/rm_finetune-gpt-neo_bs_1_merged
base_path=output_models/finetune-gpt-neo
for i in {1..1}
do
   python examples/merge_lora_trl.py \
    --adapter_model_name ${lora_path}/rm_${i} \
    --output_name ${output_path}/rm_${i} \
    --base_model_name ${base_path} 
done

