# -*- coding: utf-8 -*-
# @Author: pingzhili
# @Time: 2024/2/14

from .configuration_mixtral_model_level_hybrid import (
    MixtralModelLevelHybridConfig
)
from .modeling_mixtral_model_level_hybrid import (
    MixtralModelLevelHybridModel,
    MixtralModelLevelHybridForCausalLM,
)

MixtralModelLevelHybridConfig.register_for_auto_class("AutoConfig")
MixtralModelLevelHybridModel.register_for_auto_class("AutoModel")
MixtralModelLevelHybridForCausalLM.register_for_auto_class("AutoModelForCausalLM")
