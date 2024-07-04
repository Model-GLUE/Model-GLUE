#!/usr/bin/bash

current_dir=$(pwd)
python -u ./heuristic_merge.py \
  --GPU_idx "0" \
  --tasks "mmlu_tiny,winogrande,arc_challenge,gsm8k,humanevalsynthesize-js,mbpp" \
  --models "migtissera/Synthia-7B-v1.2,teknium/OpenHermes-7B,meta-llama/Llama-2-7b-chat-hf,lmsys/vicuna-7b-v1.5" \
  --base_model "meta-llama/Llama-2-7b-hf" \
  --root_path $current_dir \
  --metric "acc" \
  --order "descending" \
  --do_coefficient_search True \
  --sim_of_delta_param False \
  --merge_method "linear" \
  --verbose True \
  --seed 42



