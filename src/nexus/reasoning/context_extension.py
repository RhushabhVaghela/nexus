#!/usr/bin/env python3
"""
Context Length Extension Module

Extends model context length using advanced positional encoding techniques:
- RoPE Scaling (Linear, Dynamic)
- YaRN (Yet another RoPE extensioN)
- NTK-aware interpolation
- Sliding Window Attention

Based on research from:
- LongRoPE (Microsoft)
- YaRN (Anthropic/Together)
- DeepSeek context extension
"""

import math
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Tuple, Union, List
from enum import Enum

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class ScalingType(Enum):
    """Types of context length scaling."""
    LINEAR = "linear"
    DYNAMIC = "dynamic"
    YARN = "yarn"
    NTK = "ntk"
    LONGROPE = "longrope"


@dataclass
class ContextExtensionConfig:
    """Configuration for context length extension."""
    target_length: int = 131072  # 128K default
    original_length: int = 32768
    scaling_type: ScalingType = ScalingType.YARN
    rope_theta: float = 10000.0
    rope_scaling_factor: float = 0.0
    yarn_alpha: float = 1.0
    yarn_beta_fast: float = 32.0
    yarn_beta_slow: float = 1.0
    yarn_mscale: float = 1.0
    yarn_mscale_all_dim: float = 0.0
    ntk_alpha: float = 0.0
    use_sliding_window: bool = False
    sliding_window_size: int = 4096
    use_flash_attention: bool = True
    gradient_checkpointing: bool = True


class RoPEScaler:
    """Rotary Position Embedding Scaler."""
    
    def __init__(self, config: ContextExtensionConfig):
        self.config = config
        self.scaling_factor = self._compute_scaling_factor()
    
    def _compute_scaling_factor(self) -> float:
        if self.config.rope_scaling_factor > 0:
            return self.config.rope_scaling_factor
        return self.config.target_length / self.config.original_length
    
    def get_rope_config(self) -> Dict[str, Any]:
        if self.config.scaling_type == ScalingType.LINEAR:
            return {"type": "linear", "factor": self.scaling_factor}
        elif self.config.scaling_type == ScalingType.DYNAMIC:
            return {"type": "dynamic", "factor": self.scaling_factor}
        elif self.config.scaling_type == ScalingType.YARN:
            return {
                "type": "yarn",
                "factor": self.scaling_factor,
                "original_max_position_embeddings": self.config.original_length,
                "attention_factor": self._compute_yarn_attention_factor(),
                "beta_fast": self.config.yarn_beta_fast,
                "beta_slow": self.config.yarn_beta_slow,
            }
        elif self.config.scaling_type == ScalingType.NTK:
            alpha = self.config.ntk_alpha if self.config.ntk_alpha > 0 else (self.scaling_factor ** 2) - 1
            new_theta = self.config.rope_theta * (alpha + 1) ** (64 / (64 - 2))
            return {"type": "ntk", "factor": self.scaling_factor, "rope_theta": new_theta}
        else:
            raise ValueError(f"Unknown scaling type: {self.config.scaling_type}")
    
    def _compute_yarn_attention_factor(self) -> float:
        s = self.scaling_factor
        if s <= 1:
            return 1.0
        return math.sqrt(0.1 * math.log(s) + 1)


class ContextExtender:
    """Main class for extending model context length."""
    
    def __init__(self, config: Optional[ContextExtensionConfig] = None):
        self.config = config or ContextExtensionConfig()
        self.scaler = RoPEScaler(self.config)
    
    def get_model_kwargs(self) -> Dict[str, Any]:
        kwargs = {
            "rope_scaling": self.scaler.get_rope_config(),
            "max_position_embeddings": self.config.target_length,
        }
        if self.config.use_flash_attention:
            kwargs["attn_implementation"] = "flash_attention_2"
        return kwargs
    
    def extend_model_config(self, model_config: Any) -> Any:
        model_config.max_position_embeddings = self.config.target_length
        model_config.rope_scaling = self.scaler.get_rope_config()
        if not hasattr(model_config, 'original_max_position_embeddings'):
            model_config.original_max_position_embeddings = self.config.original_length
        logger.info(f"Extended context: {self.config.original_length} → {self.config.target_length}")
        return model_config
    
    def prepare_training_config(self) -> Dict[str, Any]:
        return {
            "gradient_checkpointing": self.config.gradient_checkpointing,
            "use_flash_attention": self.config.use_flash_attention,
            "max_length": self.config.target_length,
            "sliding_window": self.config.sliding_window_size if self.config.use_sliding_window else None,
        }


def create_context_extender(
    target_length: int = 131072,
    original_length: int = 32768,
    scaling_type: str = "yarn",
    **kwargs
) -> ContextExtender:
    config = ContextExtensionConfig(
        target_length=target_length,
        original_length=original_length,
        scaling_type=ScalingType(scaling_type),
        **kwargs
    )
    return ContextExtender(config)


if __name__ == "__main__":
    extender = create_context_extender(target_length=131072, original_length=32768, scaling_type="yarn")
    print(f"Context Extension: {extender.config.original_length:,} → {extender.config.target_length:,}")
    print(f"RoPE Config: {extender.scaler.get_rope_config()}")
