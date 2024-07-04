from .configuration_mixtral_block_level import (
    MixtralBlockLevelConfig
)
from .modeling_mixtral_block_level import (
    MixtralBlockLevelModel,
    MixtralBlockLevelForCausalLM,
    MixtralBlockLevelSparseDecoderLayer
)

MixtralBlockLevelConfig.register_for_auto_class("AutoConfig")
MixtralBlockLevelModel.register_for_auto_class("AutoModel")
MixtralBlockLevelForCausalLM.register_for_auto_class("AutoModelForCausalLM")
