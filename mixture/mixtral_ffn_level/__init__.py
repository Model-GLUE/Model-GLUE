from .configuration_mixtral_hybrid import (
    MixtralHybridConfig
)
from .modeling_mixtral_hybrid import (
    MixtralHybridModel,
    MixtralHybridConfig,
    MixtralHybridForCausalLM,
    MixtralHybridDecoderLayer
)

MixtralHybridConfig.register_for_auto_class("AutoConfig")
MixtralHybridModel.register_for_auto_class("AutoModel")
MixtralHybridForCausalLM.register_for_auto_class("AutoModelForCausalLM")
