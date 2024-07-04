import json
import argparse
import torch
from accelerate import Accelerator
from typing import List
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

from bigcode_eval.evaluator import Evaluator
from bigcode_eval.tasks import ALL_TASKS
import lm_eval

def eval_bigcode(model, tokenizer, task_names, args):

    accelerator = Accelerator()
    if accelerator.is_main_process:
        print(f"Selected Tasks: {task_names}")

    if not tokenizer.eos_token:
        if tokenizer.bos_token:
            tokenizer.eos_token = tokenizer.bos_token
            print("bos_token used as eos_token")
        else:
            raise ValueError("No eos_token or bos_token found")
    try:
        tokenizer.pad_token = tokenizer.eos_token
    except AttributeError:
        print("Not setting pad_token to eos_token")
        pass

    evaluator = Evaluator(accelerator, model, tokenizer, args)
    results = {}
    for idx, task in enumerate(task_names):
        results[task] = evaluator.evaluate(
            task, intermediate_generations=None
        )
    return results


def eval_lm(model, tokenizer, task_names, args, bootstrap_iters=0, num_fewshot=0, batch_size=1):
    lm_eval.tasks.initialize_tasks()
    res = lm_eval.simple_evaluate(
            model='hf',
            pretrained=model,
            tokenizer=tokenizer,
            tasks=task_names,
            num_fewshot=num_fewshot,
            batch_size=batch_size,
            model_args=args,
            bootstrap_iters=bootstrap_iters,
        )
    return res['results']

def evaluate_model(model2eval, tokenizer, bigcode_cfg_path: str, task_list: List, model_path: str = None):
    if model_path:
        model_kwargs = {'revision': None, 'trust_remote_code': True, 'use_auth_token': True, 'cache_dir': '/root/autodl-tmp/workspace/.cache', 'torch_dtype': torch.bfloat16}
        model2eval = AutoModelForCausalLM.from_pretrained(
            model_path,
            **model_kwargs,
        ).to("cuda")
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            truncation_side="left",
            padding_side="right", 
        )
    tasks_bigcode = ['humanevalsynthesize-js', 'mbpp']
    tasks_lm_eval = ['mmlu_tiny', 'winogrande', "gsm8k", 'arc_challenge']

    metrics = {'arc_challenge': 'acc_norm,none', 'mmlu_tiny': 'acc,none', 'winogrande': 'acc,none', 'gsm8k': 'exact_match,get-answer', 'humanevalsynthesize-js': 'pass@1', 'mbpp': 'pass@1'}

    with open(bigcode_cfg_path, "r") as f:
        bigcode_cfg = json.load(f)
    args_mbpp = argparse.Namespace()
    args_mbpp.__dict__.update(bigcode_cfg)
    args_he = argparse.Namespace()
    bigcode_cfg["prompt"]="instruct"
    bigcode_cfg["max_length_generation"]=512
    args_he.__dict__.update(bigcode_cfg)
    args_bigcode = {'humanevalsynthesize-js': args_he, 'mbpp': args_mbpp}

    args_lm_eval={
        "trust_remote_code": True,
        "dtype": "bfloat16"
    }
    results = 0.0
    for task in task_list:
        if task in tasks_bigcode:
            res = eval_bigcode(model2eval, tokenizer, [task], args_bigcode[task])
        elif task in tasks_lm_eval:
            res = eval_lm(model2eval, tokenizer, [task], args_lm_eval)
        else:
            raise ValueError("Wrong task name: ", task)
        results += res[task][metrics[task]]
    
    return results / len(task_list)