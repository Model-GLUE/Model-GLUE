import os
import shutil
import sys
from typing import Optional

from fire import Fire
from tqdm import tqdm
from transformers import AutoTokenizer

from mergekit.architecture import MISTRAL_INFO
from mergekit.common import ModelReference, dtype_from_name
from mergekit.io import LazyTensorLoader, TensorWriter
from .mixtral_model_level import MixtralModelLevelHybridConfig


def get_script_path():
    return os.path.dirname(os.path.realpath(sys.argv[0]))


def build(
        expert1_dir: str,
        expert2_dir: str,
        dense_dir: str,
        moe_dir: str,
        num_dense_layers: int,
        out_path: str,
        out_dtype: Optional[str] = "bfloat16",
        more_expert_dir: str = None,
):
    """
    Concatenate dense and mixture-of-experts models into a hybrid model. The frist `num_dense_layers` layers of the
    hybrid model will be from the dense model, and the rest will be from the mixture-of-experts model.

    Parameters
    ----------
    out_dtype
    expert1_dir: str,
        Expert 1's directory or repo name
    expert2_dir: str,
        Expert 2's directory or repo name
    dense_dir : str
        The directory containing the dense model.
    moe_dir : str
        The directory containing the mixture-of-experts model.
    num_dense_layers : int
        The number of layers to take from the dense model.
    out_path : str
        The directory to save the hybrid model.
    out_dtype : str, optional
        The dtype to save the hybrid model in. Can be one of "float32", "bfloat16", "float16". Defaults to "bfloat16".
    more_expert_dir : str, optional
        More expert repo name, split by comma
    """
    expert1_model = ModelReference.parse(expert1_dir)
    expert2_model = ModelReference.parse(expert2_dir)
    expert_models = []
    if more_expert_dir is not None:
        more_expert_dir = more_expert_dir.split(',')
        for expert_model in more_expert_dir:
            expert_models.append(ModelReference.parse(expert_model))
    moe_model = ModelReference.parse(moe_dir)
    dense_model = ModelReference.parse(dense_dir)

    moe_config = moe_model.config(trust_remote_code=False)
    out_config = MixtralModelLevelHybridConfig(**moe_config.to_dict())
    out_config.num_dense_layers = num_dense_layers
    out_config.num_experts_per_tok = 1
    out_config.use_cache = False
    out_config.save_pretrained(out_path)

    out_config = MixtralModelLevelHybridConfig.from_pretrained(out_path)
    out_config.auto_map[
        "AutoModelForCausalLM"] = "modeling_mixtral_model_level_hybrid.MixtralModelLevelHybridForCausalLM"
    out_config.save_pretrained(out_path)
    modeling_file_path = os.path.join(get_script_path(), "mixtral_model_level_hybrid",
                                      "modeling_mixtral_model_level_hybrid.py")
    shutil.copy(modeling_file_path, os.path.join(out_path, "modeling_mixtral_model_level_hybrid.py"))

    expert1_loader = LazyTensorLoader(expert1_model.tensor_index(), lazy_unpickle=False)
    expert2_loader = LazyTensorLoader(expert2_model.tensor_index(), lazy_unpickle=False)
    
    if expert_models is not None:
        more_expert_loaders = []
        for expert_model in expert_models:
            more_expert_loaders.append(LazyTensorLoader(expert_model.tensor_index(), lazy_unpickle=False))
        
    expert_loaders = [expert1_loader, expert2_loader]
    if expert_models is not None:
        expert_loaders.extend(more_expert_loaders)

    moe_loader = LazyTensorLoader(moe_model.tensor_index(), lazy_unpickle=False)
    dense_loader = LazyTensorLoader(dense_model.tensor_index(), lazy_unpickle=False)

    writer = TensorWriter(out_path=out_path)
    out_dtype = dtype_from_name(out_dtype)

    print("Copying parameters...")
    for tensor_name in MISTRAL_INFO.pre_weight_names + MISTRAL_INFO.post_weight_names:
        tensor = dense_loader.get_tensor(tensor_name)
        writer.save_tensor(
            tensor_name, tensor.to(dtype=out_dtype), clone=False
        )

    for name_format in tqdm(MISTRAL_INFO.layer_weight_formats()):
        for layer_idx in range(out_config.num_hidden_layers):
            tensor_name = name_format.format(idx=layer_idx)
            if '.gate.' in tensor_name:
                continue
            if layer_idx >= num_dense_layers:
                for expert_index in range(moe_config.num_local_experts):
                    expert_name = tensor_name.replace(
                        f"layers.{layer_idx}", f"layers.{layer_idx}.experts.{expert_index}"
                    )
                    tensor = expert_loaders[expert_index].get_tensor(tensor_name)
                    writer.save_tensor(
                        expert_name, tensor.to(dtype=out_dtype), clone=True
                    )
                continue
            writer.save_tensor(
                tensor_name, dense_loader.get_tensor(tensor_name).to(dtype=out_dtype)
            )

    gate_layer_idx = num_dense_layers

    tensor_name = f"model.layers.{gate_layer_idx}.block_sparse_moe.gate.weight"
    tensor = moe_loader.get_tensor(tensor_name)
    writer.save_tensor("model.gate.weight", tensor.to(dtype=out_dtype))

    writer.finalize()

    print("Saving tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(dense_model.model.path, revision=dense_model.model.revision)
    tokenizer.padding_side = "left"
    tokenizer.pad_token_id = tokenizer.bos_token_id
    tokenizer.save_pretrained(out_path, safe_serialization=True)

    print("Done!")


if __name__ == "__main__":
    Fire(build)
