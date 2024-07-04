import optuna
import subprocess
import os
import json
import yaml
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch")
import torch
from merge_tools import build_model_config, merge_in_memory

env = os.environ.copy()
env['MKL_THREADING_LAYER'] = 'GNU'
env['MKL_SERVICE_FORCE_INTEL'] = '1'

# hyper parameter of merging
root_path = '/PATH/TO/WORKSPACE'
GPU_idx = '5'
path_of_bigcode = '/PATH/TO/bigcode-evaluation-harness'
n_of_group = 2
model_list = ["lmsys/vicuna-7b-v1.5", "meta-llama/Llama-2-7b-chat-hf", "teknium/OpenHermes-7B", "garage-bAInd/Platypus2-7B", "neuralmagic/Llama-2-7b-evolcodealpaca", "meta-math/MetaMath-7B-V1.0", "migtissera/Synthia-7B-v1.2", "PygmalionAI/pygmalion-2-7b", "stanford-oval/Llama-2-7b-WikiChat-fused", "cognitivecomputations/dolphin-llama2-7b", "Severus27/BeingWell_llama2_7b", "GOAT-AI/GOAT-7B-Community"]
model_idx_path = './pytorch_model.bin.index.json'
output_path = './output.txt'

# hyper parameter of cma-es
popsize = 15
sigma0 = 1 / 6
seed = 42
n_trials = 500


def execute_merge(merged_model_state_dict, model_state_dict_list, coe_list, range_list, n_layers=32):
    if merged_model_state_dict == None:
        merged_model_state_dict = {}

    with open(model_idx_path, 'r') as f:
        index = json.load(f)
    model_structure = index["weight_map"]
    for tensor_name in model_structure.keys():
        flag = 0
        for selected_layer in range_list:
            # print(selected_layer, range_list)
            if selected_layer == n_layers:
                match_str = 'lm_head.weight'
            elif selected_layer == n_layers + 1:
                match_str = 'model.embed_tokens.weight'
            elif selected_layer == n_layers + 2:
                match_str = 'model.norm.weight'
            else:
                match_str = f"layers.{selected_layer}."
            if match_str in tensor_name:
                flag = 1
                break
        if flag == 1 and 'rotary_emb.inv_freq' not in tensor_name:
            # do merge
            for model_state_dict, weight in zip(model_state_dict_list, coe_list):
                if match_str == 'lm_head.weight' or match_str == 'model.embed_tokens.weight':

                    if tensor_name in merged_model_state_dict:
                        merged_model_state_dict[tensor_name] += torch.tensor(model_state_dict[tensor_name][:32000, :], dtype=torch.bfloat16) * torch.tensor(weight, dtype=torch.bfloat16)
                    else:
                        merged_model_state_dict[tensor_name] = torch.tensor(model_state_dict[tensor_name][:32000, :], dtype=torch.bfloat16) * torch.tensor(weight,
                                                                                              dtype=torch.float)
                else:
                    if tensor_name in merged_model_state_dict:
                        merged_model_state_dict[tensor_name] += torch.tensor(model_state_dict[tensor_name], dtype=torch.bfloat16) * torch.tensor(weight, dtype=torch.bfloat16)
                    else:
                        merged_model_state_dict[tensor_name] = torch.tensor(model_state_dict[tensor_name], dtype=torch.bfloat16) * torch.tensor(weight,
                                                                                                                                                dtype=torch.float)

    return merged_model_state_dict

def normalize_list(coe_list):
    return [x / sum(coe_list) for x in coe_list]

def merge_models_group(coefficient_list, model_list, n_of_group, n_layers=32, group_size=4, save_path='./tmp_model'):
    if len(model_list) < group_size:
        group_size = len(model_list)
    coe_by_range_list = []
    for j in range(n_of_group):
        tmp_coe_by_range_list = []
        for k in range(len(model_list)):
            i = j * len(model_list) + k
            tmp_coe_by_range_list.append(coefficient_list[i])
        coe_by_range_list.append(normalize_list(tmp_coe_by_range_list))

    layer_idxs = list(range(n_layers))
    layer_idxs_by_group = [layer_idxs[i:i + n_layers // n_of_group] for i in range(0, n_layers, n_layers // n_of_group)]

    # lm_head and embed
    coe_by_range_list.append(normalize_list(coefficient_list[-len(model_list):]))
    coe_by_range_list.append(normalize_list(coefficient_list[-len(model_list):]))
    coe_by_range_list.append(normalize_list(coefficient_list[-len(model_list):]))
    layer_idxs_by_group.extend([[n_layers], [n_layers + 1], [n_layers + 2]])

    coe_by_group_list = []
    for coe_by_range in coe_by_range_list:
        coe_group_tmp = [coe_by_range[i:i + group_size] for i in range(0, len(coe_by_range), group_size)]
        if coe_by_group_list == []:
            for i in range(0, len(coe_by_range), group_size):
                coe_by_group_list.append([])

        for i in range(len(coe_by_range) // group_size):
            coe_by_group_list[i].append(coe_group_tmp[i])

    merged_model_state_dict = None
    model_group_list = [model_list[i:i + group_size] for i in range(0, len(model_list), group_size)]


    for model_group, coe_group in zip(model_group_list, coe_by_group_list):
        model2del = []
        model_state_dict_list = []
        for model_name in model_group:
            model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
            model_state_dict_list.append(model.state_dict())
            model2del.append(model)

        for ranges, coe_group_by_range in zip(layer_idxs_by_group, coe_group):
            # print(ranges, coe_group_by_range)
            merged_model_state_dict = execute_merge(merged_model_state_dict, model_state_dict_list, coe_group_by_range, ranges, n_layers)

            # print(merged_model_state_dict)
        # for i in range(len(model2del)):
        # del model2del
        # del model_state_dict_list

    model2save = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-2-7b-chat-hf', torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-chat-hf')
    model2save.load_state_dict(merged_model_state_dict)
    model2save.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    # print(model_group_list)
    # print(layer_idxs_by_group)
    # print(range_list)



def objective(trial, model_list, n_of_group, range_max, range_min, merge_method='linear'):
    if n_of_group == 1:
        if 'dare' in merge_method or 'ties' in merge_method:
            n = len(model_list) * 2
        else:
            n = len(model_list)

        param_list = [
            trial.suggest_float(f"weight_{i}", range_min, range_max)
            for i in range(n)
        ]
        
        weight_list = param_list[:len(model_list)]
        density_list = param_list[len(model_list):]
        cfg = build_model_config(model_list=model_list, weight_list=weight_list, 
                                 density_list=density_list, merge_method=merge_method, 
                                 base_model='meta-llama/Llama-2-7b-hf' if 'dare' in merge_method or 'ties' in merge_method else None)
        model = merge_in_memory(None, cfg)
        pass
        # call original merge method of mergekit, merge_in_memory
    else:
        n = len(model_list) * (n_of_group + 1)
        param_list = [
            trial.suggest_float(f"weight_{i}", range_min, range_max)
            for i in range(n)
        ]

        merge_models_group(param_list, model_list, n_of_group)


    try:
        shell_script_content = f"""#!/bin/bash
result_float=0
tasks=("mmlu_tiny" "gsm8k" "arc_challenge" "winogrande")
k=0
for task in "${{tasks[@]}}"; do
    CUDA_VISIBLE_DEVICES={GPU_idx} lm_eval --model hf \
    --model_args pretrained=./tmp_model,trust_remote_code=True,dtype="bfloat16" \
    --tasks $task \
    --batch_size 1 \
    --output_path ./${{task}}_tmp.json
    ((k++))
done

export CUDA_VISIBLE_DEVICES={GPU_idx}
python {path_of_bigcode}/main.py \
    --model {root_path}/tmp_model \
    --tasks mbpp \
    --max_length_generation 1024 \
    --do_sample True \
    --n_samples 1 \
    --top_p 0.95 \
    --batch_size 1 \
    --temperature 0.2 \
    --precision bf16 \
    --trust_remote_code \
    --allow_code_execution \
    --use_auth_token \
    --metric_output_path {root_path}/mbpp_tmp.json

export CUDA_VISIBLE_DEVICES={GPU_idx}
python {path_of_bigcode}/main.py \
    --model {root_path}/tmp_model \
    --tasks humanevalsynthesize-js \
    --max_length_generation 512 \
    --prompt instruct \
    --do_sample True \
    --n_samples 1 \
    --top_p 0.95 \
    --batch_size 1 \
    --temperature 0.2 \
    --precision bf16 \
    --trust_remote_code \
    --allow_code_execution \
    --use_auth_token \
    --metric_output_path {root_path}/humanevalsynthesize-js_tmp.json
"""
        script_filename = 'tmp_script.sh'
        with open(script_filename, 'w') as script_file:
            script_file.write(shell_script_content)

        os.chmod(script_filename, 0o755)

        result = subprocess.run(['./tmp_script.sh'], capture_output=True, text=True, shell=True, env=env)

        if result.returncode == 0:
            results_list = []
            sum_results = 0.0
            task_list = ["mmlu_tiny", "gsm8k", "arc_challenge", "winogrande", "mbpp", "humanevalsynthesize-js"]
            metric_list = ["acc,none", "exact_match,get-answer", "acc_norm,none", "acc,none", "pass@1", "pass@1"]
            for task, metric in zip(task_list, metric_list):
                with open(f'{task}_tmp.json', 'r') as file:
                    r = json.load(file)
                    if task == 'mbpp':
                        tmp_r = float(r[task][metric])
                    elif task == 'humanevalsynthesize-js':
                        tmp_r = float(r[task][metric])
                        os.remove(f'logs.json')
                    else:
                        tmp_r = float(r['results'][task][metric])
                    sum_results += tmp_r
                    results_list.append(tmp_r)
                os.remove(f'{task}_tmp.json')
            print('------------------------------')
            print('param_list: ', param_list)
            print('acc per task: ', results_list)
            print('sum acc: ', sum_results)
            return -float(sum_results)
        else:
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            print("Return Code:", result.returncode)
            return 0.0
    except:
        return 0.0


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--tasks", type=str, required=True)
    # parser.add_argument("--model_list", type=str, required=True)
    # parser.add_argument("--root_path", type=str, required=True)
    # parser.add_argument("--GPU_idx", type=str, required=True)
    # parser.add_argument("--verbose", type=bool, default=False)
    # parser.add_argument("--cache_dir", type=str, default='/workspace/.cache')
    # parser.add_argument("--merge_method", type=str, default='linear')
    # parser.add_argument("--seed", type=int, default=42, help="Random seed")
    # parser.add_argument("--popsize", type=int, default=10)
    # parser.add_argument("--sigma0", type=float, default=1/6)
    # parser.add_argument("--n_trials", type=int, default=200)
    # parser.add_argument("--num_group", type=int, default=1, required=True)
    # parser.add_argument("--range_max", type=float, default=1.0)
    # parser.add_argument("--range_min", type=float, default=0.0)
    # parser.add_argument("--output_path", type=str, default='./output.txt')
    
    # global args
    # args = parser.parse_args()
    # os.environ['CUDA_VISIBLE_DEVICES']=args.GPU_idx
    # models = args.model_list.split(',')

    # popsize = 4+3*ln(num_param)
    # sys.stdout = open(args.output_path, 'a')
    # sampler = optuna.samplers.CmaEsSampler(popsize=args.popsize, sigma0=args.sigma0, seed=args.seed)
    # study = optuna.create_study(sampler=sampler)

    # study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)
    # sys.stdout.close()
    # sys.stdout = sys.__stdout__

    sys.stdout = open(output_path, 'a')
    sampler = optuna.samplers.CmaEsSampler(popsize=popsize, sigma0=sigma0, seed=seed)
    study = optuna.create_study(sampler=sampler)

    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    sys.stdout.close()
    sys.stdout = sys.__stdout__