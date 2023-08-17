#!/bin/bash

count=5

base_dir="/home/winnie/trl/logs_trl/ppo_llama_klcoef0.01"
reward_model="/home/winnie/output_models/openllama_3b_rm_2sft_full_train_5e-6_1epoch_4x8bs_raw_dataset"

for (( i=1; i<=$count; i++ )); do
  model_dir="${base_dir}/batch_${i}"
  CUDA_VISIBLE_DEVICES="0,1" ./scripts_for_uncertainty_study/eval_get_rewards.sh $model_dir/eval_set /home/winnie/LMFlow/eval_scripts/output_models/test_infer ${base_dir} ${reward_model}
done