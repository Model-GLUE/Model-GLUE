# Copyright (C) 2024 Charles O. Goddard
#
# This software is free software: you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# This software is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program. If not, see http://www.gnu.org/licenses/.

import logging
import os
import shutil
from typing import Dict, Optional, List

import tqdm
import transformers
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from mergekit.architecture import ArchitectureInfo, get_architecture_info
from mergekit.graph import Executor
from mergekit.io.tasks import LoaderCache
from mergekit.options import MergeOptions
from mergekit.plan import MergePlanner
from mergekit.io.tasks import LoaderCache, ReturnTensor
from mergekit.tokenizer import TokenizerInfo
from mergekit.config import (
    InputModelDefinition,
    MergeConfiguration,
    ParameterSetting,
)


def run_merge(
    merge_config: MergeConfiguration,
    options: MergeOptions,
    out_path: str = None,
    base_model: str = "meta-llama/Llama-2-7b-hf",
):
    if options.random_seed is not None:
        transformers.trainer_utils.set_seed(options.random_seed)

    if not merge_config.models and not merge_config.slices:
        raise RuntimeError("No output requested")

    model_arch_info = [
        get_architecture_info(m.config(trust_remote_code=options.trust_remote_code))
        for m in merge_config.referenced_models()
    ]
    if not options.allow_crimes:
        if not all(a == model_arch_info[0] for a in model_arch_info[1:]):
            raise RuntimeError(
                "Must specify --allow-crimes to attempt to mix different architectures"
            )
    arch_info = model_arch_info[0]

    # initialize loader cache and set options
    loader_cache = LoaderCache()
    loader_cache.setup(options=options)

    # create config for output model
    cfg_out = _model_out_config(
        merge_config, arch_info, trust_remote_code=options.trust_remote_code
    )

    # warm up loader cache
    for m in (
        pbar := tqdm.tqdm(
            merge_config.referenced_models(),
            desc="Warmup loader cache",
            disable=options.quiet,
        )
    ):
        loader_cache.get(m)
    del pbar

    logging.info("Planning operations")

    if out_path:
        targets = MergePlanner(
                merge_config,
                arch_info,
                options=options,
                out_model_config=cfg_out,
        ).plan_to_disk(out_path=out_path)

        exec = Executor(
            tasks=targets,
            math_device="cuda" if options.cuda else "cpu",
            storage_device="cuda" if options.low_cpu_memory else "cpu",
        )

        tokenizer = None
        for tensor_task, value in exec.run(quiet=True):
            if isinstance(value, TokenizerInfo):
                tokenizer = value.tokenizer

        logging.info("Saving config")
        cfg_out.save_pretrained(out_path)

        if tokenizer == None:
            tokenizer = AutoTokenizer.from_pretrained(base_model)
            logging.info("Saving tokenizer")
            tokenizer.save_pretrained(out_path, safe_serialization=True)

        return None, None
    
    else:
        targets = MergePlanner(
            merge_config,
            arch_info,
            options=options,
            out_model_config=cfg_out,
        ).plan_in_memory()

        executor = Executor(
            tasks=targets,
            math_device="cuda" if options.cuda else "cpu",
            storage_device="cuda" if options.low_cpu_memory else "cpu",
        )

        tokenizer = AutoTokenizer.from_pretrained(base_model)
        config = AutoConfig.from_pretrained(base_model)
        model = AutoModelForCausalLM.from_config(config)
        param_dict = dict(model.named_parameters())

        # for name, param in model.named_parameters():
        #     if 'model.layers.0.self_attn.q_proj.weight' in name:
        #         print(param)

        for tensor_task, value in executor.run(quiet=True):
            assert isinstance(tensor_task, ReturnTensor)
            name = tensor_task.weight_info.name
            if name in param_dict:
                param_dict[name].data.copy_(value, non_blocking=True)
        
        # for name, param in model.named_parameters():
        #     if 'model.layers.0.self_attn.q_proj.weight' in name:
        #         print(param)


        return model, tokenizer

def _model_out_config(
    config: MergeConfiguration,
    arch_info: ArchitectureInfo,
    trust_remote_code: bool = False,
) -> transformers.PretrainedConfig:
    """Return a configuration for the resulting model."""
    if config.base_model:
        res = config.base_model.config(trust_remote_code=trust_remote_code)
    else:
        res = config.referenced_models()[0].config(trust_remote_code=trust_remote_code)
    if config.dtype:
        res.torch_dtype = config.dtype

    if config.slices:
        try:
            num_layers = sum(
                s.sources[0].layer_range[1] - s.sources[0].layer_range[0]
                for s in config.slices
            )
            setattr(res, arch_info.num_layers_config_key(), num_layers)
        except Exception as e:
            logging.warning(
                "Unable to set number of layers in output config - you may need to manually correct it.",
                exc_info=e,
            )

    return res

def build_model_config(
    model_list: List,
    weight_list: List,
    merge_method: str,
    density_list: List = None,
    dtype: str = "bfloat16",
    base_model: Optional[str] = None,
):

    if density_list:
        assert len(model_list) == len(weight_list) == len(density_list), "Merging parameters do not match"
    else:
        assert len(model_list) == len(weight_list), "Merging parameters do not match"
    models=[]
    for i in range(len(model_list)):
        params = {"weight": weight_list[i]}
        if density_list:
            params["density"] = density_list[i]
        models.append(
            InputModelDefinition(
                model=model_list[i],
                parameters=params,
            )
        )

    config = MergeConfiguration(
        merge_method=merge_method,
        base_model=base_model,
        models=models,
        dtype=dtype,
        parameters=params,
    )

    return config

__all__ = ["MergeOptions", "merge_in_memory"]