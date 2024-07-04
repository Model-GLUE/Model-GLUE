import subprocess
import os
import json
from transformers import AutoModelForCausalLM
import torch
import argparse
import torch.nn.functional as F
import json
import subprocess
import yaml
import sys
from typing import Dict
import csv
from eval_tools import eval_bigcode, eval_lm, evaluate_model
from typing import List
from merge_tools import build_model_config, run_merge, MergeOptions

env = os.environ.copy()
env['MKL_THREADING_LAYER'] = 'GNU'
env['MKL_SERVICE_FORCE_INTEL'] = '1'

def get_sim(model_a: str, model_b: str, sim_of_delta_param: bool, per_column: bool, base_model_path='meta-llama/Llama-2-7b-hf'):

    def cosine_similarity(vector1, vector2):
        dot_product = torch.dot(vector1, vector2)
        norm_vector1 = torch.norm(vector1)
        norm_vector2 = torch.norm(vector2)

        similarity = dot_product / (norm_vector1 * norm_vector2)

        return similarity
    if sim_of_delta_param:
        base_model = AutoModelForCausalLM.from_pretrained(base_model_path)
        base_model_state_dict = base_model.state_dict()

    ft_model1 = AutoModelForCausalLM.from_pretrained(model_a)
    ft_model2 = AutoModelForCausalLM.from_pretrained(model_b)
    ft_model1_state_dict = ft_model1.state_dict()
    ft_model2_state_dict = ft_model2.state_dict()

    return_dic = {}
    return_dic['model_a'] = model_a
    return_dic['model_b'] = model_b

    sim_list_attn = []
    sim_list_mlp = []
    for name, param in ft_model1_state_dict.items():

        if 'self_attn' in name or 'mlp' in name:
            # print(f"Parameter name: {name}, Shape: {param.shape}")
            # sim = cosine_similarity_m(param, ft_model_state_dict[name])
            if sim_of_delta_param:
                param_a = param - base_model_state_dict[name]
                param_b = ft_model2_state_dict[name] - base_model_state_dict[name]
            else:
                param_a = param
                param_b = ft_model2_state_dict[name]
            if per_column:
                similarity_scores = F.cosine_similarity(param_a, param_b, dim=0)
                sim = torch.mean(similarity_scores)
                # print(sim)
            else:
                sim = cosine_similarity(param_a.view(-1), param_b.view(-1))
            if 'self_attn' in name:
                sim_list_attn.append(sim.item())
            elif 'mlp' in name:
                sim_list_mlp.append(sim.item())

    sim_list_all = sim_list_attn + sim_list_mlp
    return_dic['sim_attn'] = sum(sim_list_attn) / len(sim_list_attn)
    return_dic['sim_mlp'] = sum(sim_list_mlp) / len(sim_list_mlp)
    return_dic['sim_all'] = sum(sim_list_all) / len(sim_list_all)

    del ft_model1
    del ft_model2
    if sim_of_delta_param:
        del base_model
    return return_dic

def select_a_model(current_model, models, metric):
    selected_model = ''
    if 'sim' in metric:
        init_sim = -1
        for model in models:
            sims = get_sim(current_model, model, args.sim_of_delta_param, True)
            # sim = sims['sim_all']
            sim = sims['sim_mlp']
            if init_sim == -1:
                selected_model = model
                init_sim = sim
            else:
                if metric == 'dissim':
                    if sim < init_sim:
                        selected_model = model
                        init_sim = sim
                elif metric == 'sim':
                    if sim > init_sim:
                        selected_model = model
                        init_sim = sim
            if args.verbose:
                print(f"[log] {metric} of {args.init_model} {model} : {sim}")
    elif metric == 'acc':
        selected_model = models[0]

    models.remove(selected_model)
    if args.verbose:
        print(f"[log] selected model: {selected_model}")
        print(f"[log] remained models: {models}")
    return selected_model, models

def get_order_of_models(model_list, task_list, descending=True, cache_file='./cached_accs.csv'):
    def read_csv_file(filename):
        data = {}

        with open(filename, 'r') as file:
            reader = csv.reader(file)
            headers = next(reader)
            for header in headers:
                data[header] = {}

            for row in reader:
                for i, value in enumerate(row):
                    data[headers[i]][row[0]] = value
        return data

    if cache_file:
        search_table = read_csv_file(cache_file)
        score_list = []
        for model in model_list:
            tmp_score = 0.0
            for task in task_list:
                tmp_score += float(search_table[task][model])
            score_list.append(tmp_score)

        sorted_models = [x for _, x in sorted(zip(score_list, model_list), key=lambda x: x[0], reverse=descending)]
        return sorted_models

def clustering(wild_models, sim_of_delta_param, threshold=0.95):
    model_families = []
    for model in wild_models:
        if model_families == []:
            model_families.append([model])
        else:
            for i, fml in enumerate(model_families):
                sim = get_sim(fml[0], model, sim_of_delta_param, True)
                if sim['sim_all'] > threshold:
                    model_families[i].append(model)
                    break

    return model_families


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=str, required=True)
    parser.add_argument("--models", type=str, required=True)
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--root_path", type=str, required=True)
    parser.add_argument("--GPU_idx", type=str, required=True)
    parser.add_argument("--metric", type=str, required=True)
    parser.add_argument("--sim_of_delta_param", type=bool, default=False)
    parser.add_argument("--verbose", type=bool, default=False)
    parser.add_argument("--cache_dir", type=str, default='/workspace/.cache')
    parser.add_argument("--merge_method", type=str, default='linear')
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--do_coefficient_search", type=bool, default=False)
    parser.add_argument("--order", type=str, default='descending')
    parser.add_argument("--tmp_model_path", type=str, default='./tmp_model')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES']=args.GPU_idx
    model_list = args.models.split(',')
    task_list = args.tasks.split(',')

    if args.metric == "acc":
        model_list = get_order_of_models(model_list, task_list, descending=True, cache_file='./files/cached_accs.csv')

    current_model = args.init_model
    best_result_in_all_round = evaluate_model(None, None, bigcode_cfg_path='./files/generation_config.json', task_list=task_list, model_path=current_model)

    if args.verbose:
        print(f"[log] acc of init model: {best_result_in_all_round}")
    selected_model_list = []
    coeffi_list = []
    for round in range(len(model_list)):
        if args.verbose:
            print(f"[log] round: {round}")
        # get a model to merge
        selected_model, model_list = select_a_model(current_model, model_list, args.metric)

        best_result_in_this_round = 0
        best_coefficient = 1
        if args.do_coefficient_search == True:
            # do coefficient search
            for i in range(1, 10):
                coefficient_1 = i / 10.0
                coefficient_2 = 1 - coefficient_1
                
                cfg = build_model_config(model_list=[current_model, selected_model], weight_list=[coefficient_1, coefficient_2],
                                         merge_method=args.merge_method, density_list=None)
                
                model2eval, tokenizer = run_merge(cfg, options=MergeOptions(), base_model=args.base_model)
                if model2eval:
                    model2eval.to('cuda')

                res_acc = evaluate_model(model2eval, tokenizer, bigcode_cfg_path='./files/generation_config.json', task_list=task_list, model_path=args.tmp_model_path)

                # subprocess.run(f'rm -r {args.root_path}/tmp_model', shell=True)
                if args.verbose:
                    print(f"[log] {coefficient_1} * {current_model} + {coefficient_2} * {selected_model} = {res_acc}")

                if res_acc >= best_result_in_this_round:
                    best_coefficient = coefficient_1
                    best_result_in_this_round = res_acc
        else:
            # simply averaging
            coefficient_1 = (len(selected_model_list) + 1) / (len(selected_model_list) + 2)
            coefficient_2 = 1 - coefficient_1

            cfg = build_model_config(model_list=[current_model, selected_model], weight_list=[coefficient_1, coefficient_2],
                            merge_method=args.merge_method, density_list=None)
            
            model2eval, tokenizer = run_merge(cfg, options=MergeOptions(), base_model=args.base_model)

            res_acc = evaluate_model(model2eval, tokenizer, bigcode_cfg_path='./files/generation_config.json', task_list=task_list, model_path=args.tmp_model_path)

            # subprocess.run(f'rm -r {args.root_path}/tmp_model', shell=True)
            if args.verbose:
                print(f"[log] {coefficient_1} * {current_model} + {coefficient_2} * {selected_model} = {res_acc}")

            if res_acc >= best_result_in_this_round:
                best_coefficient = coefficient_1
                best_result_in_this_round = res_acc

        if args.verbose:
            print(f"[log] Got best result of coefficient search!  {best_result_in_this_round} = {best_coefficient} * {current_model} + {1 - best_coefficient} * {selected_model}")

        # get the best model of this round
        if best_result_in_this_round >= best_result_in_all_round:
            if args.verbose:
                print(f"[log] best_result_in_this_round: {best_result_in_this_round} > best_result_in_last_round: {best_result_in_all_round}")

            save_path=f'{args.root_path}/best_model_in_round_{round}'
            cfg = build_model_config(model_list=[current_model, selected_model], weight_list=[coefficient_1, coefficient_2],
                            merge_method=args.merge_method, density_list=None)
            
            _, _ = run_merge(cfg, options=MergeOptions(), out_path=save_path, base_model=args.base_model)
            
            current_model = save_path

            selected_model_list.append(selected_model)
            coeffi_list.append(best_coefficient)
            best_result_in_all_round = best_result_in_this_round

            if args.verbose:
                print(f"[log] Got best result of round {round}!  {best_result_in_this_round} = {best_coefficient} * {current_model} + {1 - best_coefficient} * {selected_model}")
        else:
            print(f"[log] There is no improvement in round {round}, so drop {selected_model}!")

    print(selected_model_list)
    print(coeffi_list)
    print(best_result_in_all_round)


    best_model_list = [args.init_model] + selected_model_list
    best_coeffi_list = []
    for i, model in enumerate(best_model_list):
        coefficient_tmp = 1.0
        for j, coefficient in enumerate(coeffi_list):
            if i - 1 == j:
                coefficient_tmp *= (1 - coefficient)
            elif i - 1 > j:
                continue
            elif i - 1 < j:
                coefficient_tmp *= coefficient
        best_coeffi_list.append(coefficient_tmp)
    print(best_coeffi_list)
    print(best_model_list)

    # if args.merge_method != 'slerp':
    #     cfg = build_model_config(model_list=best_model_list, weight_list=best_coeffi_list,
    #             merge_method=args.merge_method, density_list=None)
        
    #     _, _ = run_merge(cfg, options=MergeOptions(), out_path='./searched_model', base_model=args.base_model)
