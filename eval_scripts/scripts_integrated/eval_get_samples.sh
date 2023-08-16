#!/bin/bash

count=27
base_dir="/home/winnie/ppo_llama_klcoef0.01"

for (( i=1; i<=$count; i++ )); do
  model_dir="${base_dir}/batch_${i}"
  mkdir $model_dir/eval_set
  CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" ./scripts_for_uncertainty_study/eval_get_samples.sh $model_dir $base_dir
done