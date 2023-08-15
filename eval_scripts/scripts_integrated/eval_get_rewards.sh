#!/bin/bash

count=6

base_dir="/home/winnie/trl/logs_trl/ppo_llama_klcoef0.01"
reward_model="output_models/rms/0715_relabel_rm_gpt_neo_2_7b_5e-6_1epoch_1"

for (( i=1; i<=$count; i++ )); do
  model_dir="${base_dir}/model${i}"
  CUDA_VISIBLE_DEVICES="0,1" ./scripts_for_uncertainty_study/eval_get_rewards.sh $model_dir/eval_set /home/xiongwei/LMFlow/output_models/test_infer ${base_dir} ${reward_model}
done