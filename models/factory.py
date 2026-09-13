"""Model construction for Relational BabyLM backends."""

import torch.nn as nn

from models.attention.dat.config import DatLMConfig
from models.attention.dat.lm import DatDecoderLM
from models.attention.transformer.config import SelfAttentionLMConfig
from models.attention.transformer.lm import SelfAttentionDecoderLM


ModelConfig = SelfAttentionLMConfig | DatLMConfig


def build_model_config_from_dict(model_type: str, raw_config: dict[str, object]) -> ModelConfig:
    if model_type == "self_attention":
        return SelfAttentionLMConfig(**raw_config)
    if model_type == "dat":
        return DatLMConfig(**raw_config)
    raise ValueError(f"Unsupported model_type: {model_type}")


def build_model(model_type: str, model_config: object) -> nn.Module:
    if model_type == "self_attention":
        if not isinstance(model_config, SelfAttentionLMConfig):
            raise TypeError(
                "self_attention model_type requires SelfAttentionLMConfig, "
                f"got {type(model_config).__name__}"
            )
        return SelfAttentionDecoderLM(model_config)
    if model_type == "dat":
        if not isinstance(model_config, DatLMConfig):
            raise TypeError(
                "dat model_type requires DatLMConfig, "
                f"got {type(model_config).__name__}"
            )
        return DatDecoderLM(model_config)
    raise ValueError(f"Unsupported model_type: {model_type}")
