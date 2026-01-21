#!/usr/bin/env python3
"""
training_methods.py
SOTA Training Methods Configuration for LLM fine-tuning.

Supported Methods:
- SFT (Supervised Fine-Tuning)
- LoRA (Low-Rank Adaptation)
- QLoRA (Quantized LoRA)
- DPO (Direct Preference Optimization)
- GRPO (Group Relative Policy Optimization)
- ORPO (Odds Ratio Preference Optimization)
- PPO/RLHF (Proximal Policy Optimization)
- Distillation (Knowledge Distillation)

Usage:
    from src.training_methods import TrainingMethod, get_training_config
    
    config = get_training_config(TrainingMethod.QLORA)
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List


class TrainingMethod(Enum):
    """Available SOTA training methods."""
    
    # Basic Fine-Tuning
    SFT = "sft"                  # Supervised Fine-Tuning (full weights)
    
    # Parameter-Efficient Methods (PEFT)
    LORA = "lora"               # Low-Rank Adaptation
    QLORA = "qlora"             # Quantized LoRA (4-bit)
    DORA = "dora"               # Weight-Decomposed LoRA (2024)
    
    # Preference-Based Methods
    DPO = "dpo"                 # Direct Preference Optimization
    GRPO = "grpo"               # Group Relative Policy Optimization (DeepSeek)
    ORPO = "orpo"               # Odds Ratio Preference Optimization
    PPO = "ppo"                 # Proximal Policy Optimization (RLHF)
    
    # Knowledge Transfer
    DISTILLATION = "distillation"  # Knowledge Distillation
    
    # Continued Training
    CPT = "cpt"                 # Continued Pre-Training


@dataclass
class TrainingMethodConfig:
    """Configuration for a training method."""
    
    method: TrainingMethod
    description: str
    
    # PEFT-specific
    use_peft: bool = False
    lora_r: int = 16                    # LoRA rank
    lora_alpha: int = 32                # LoRA alpha
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"])
    
    # Quantization
    use_quantization: bool = False
    quantization_bits: int = 4          # 4 or 8
    
    # Preference-based
    use_preference_data: bool = False
    beta: float = 0.1                   # DPO/GRPO beta parameter
    
    # Distillation
    use_distillation: bool = False
    temperature: float = 2.0
    distillation_alpha: float = 0.5
    
    # General
    learning_rate: float = 2e-5
    epochs: int = 3
    gradient_accumulation_steps: int = 4
    warmup_ratio: float = 0.1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "method": self.method.value,
            "description": self.description,
            "use_peft": self.use_peft,
            "lora_r": self.lora_r,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "use_quantization": self.use_quantization,
            "quantization_bits": self.quantization_bits,
            "use_preference_data": self.use_preference_data,
            "beta": self.beta,
            "use_distillation": self.use_distillation,
            "temperature": self.temperature,
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
        }


# Pre-defined configurations for each method
TRAINING_CONFIGS: Dict[TrainingMethod, TrainingMethodConfig] = {
    TrainingMethod.SFT: TrainingMethodConfig(
        method=TrainingMethod.SFT,
        description="Supervised Fine-Tuning - Full weight updates",
        learning_rate=2e-5,
    ),
    
    TrainingMethod.LORA: TrainingMethodConfig(
        method=TrainingMethod.LORA,
        description="Low-Rank Adaptation - Parameter-efficient fine-tuning",
        use_peft=True,
        lora_r=16,
        lora_alpha=32,
        learning_rate=1e-4,  # LoRA uses higher LR
    ),
    
    TrainingMethod.QLORA: TrainingMethodConfig(
        method=TrainingMethod.QLORA,
        description="Quantized LoRA - 4-bit quantization with LoRA",
        use_peft=True,
        use_quantization=True,
        quantization_bits=4,
        lora_r=64,           # QLoRA typically uses higher rank
        lora_alpha=16,
        learning_rate=2e-4,
    ),
    
    TrainingMethod.DORA: TrainingMethodConfig(
        method=TrainingMethod.DORA,
        description="Weight-Decomposed LoRA - Improved LoRA variant (2024)",
        use_peft=True,
        lora_r=16,
        lora_alpha=32,
        learning_rate=1e-4,
    ),
    
    TrainingMethod.DPO: TrainingMethodConfig(
        method=TrainingMethod.DPO,
        description="Direct Preference Optimization - Simplified RLHF",
        use_preference_data=True,
        beta=0.1,
        learning_rate=5e-7,
    ),
    
    TrainingMethod.GRPO: TrainingMethodConfig(
        method=TrainingMethod.GRPO,
        description="Group Relative Policy Optimization - DeepSeek method",
        use_preference_data=True,
        beta=0.1,
        learning_rate=1e-6,
    ),
    
    TrainingMethod.ORPO: TrainingMethodConfig(
        method=TrainingMethod.ORPO,
        description="Odds Ratio Preference Optimization - Combined SFT+Preference",
        use_preference_data=True,
        beta=0.1,
        learning_rate=8e-6,
    ),
    
    TrainingMethod.PPO: TrainingMethodConfig(
        method=TrainingMethod.PPO,
        description="Proximal Policy Optimization - Classic RLHF",
        use_preference_data=True,
        learning_rate=1e-6,
    ),
    
    TrainingMethod.DISTILLATION: TrainingMethodConfig(
        method=TrainingMethod.DISTILLATION,
        description="Knowledge Distillation - Learn from teacher model",
        use_distillation=True,
        temperature=2.0,
        distillation_alpha=0.5,
        learning_rate=2e-5,
    ),
    
    TrainingMethod.CPT: TrainingMethodConfig(
        method=TrainingMethod.CPT,
        description="Continued Pre-Training - Domain adaptation",
        learning_rate=5e-6,
        epochs=1,
    ),
}


def get_training_config(method: TrainingMethod) -> TrainingMethodConfig:
    """Get the default configuration for a training method."""
    return TRAINING_CONFIGS.get(method, TRAINING_CONFIGS[TrainingMethod.SFT])


def get_all_methods() -> List[str]:
    """Get list of all available training method names."""
    return [m.value for m in TrainingMethod]


def parse_training_method(method_str: str) -> TrainingMethod:
    """Parse a string to TrainingMethod enum."""
    method_str = method_str.lower().strip()
    
    for method in TrainingMethod:
        if method.value == method_str:
            return method
    
    raise ValueError(f"Unknown training method: {method_str}. Available: {get_all_methods()}")


if __name__ == "__main__":
    print("Available Training Methods:")
    print("=" * 60)
    for method, config in TRAINING_CONFIGS.items():
        print(f"\n{method.value.upper()}: {config.description}")
        print(f"  Learning Rate: {config.learning_rate}")
        if config.use_peft:
            print(f"  LoRA: r={config.lora_r}, alpha={config.lora_alpha}")
        if config.use_quantization:
            print(f"  Quantization: {config.quantization_bits}-bit")
        if config.use_preference_data:
            print(f"  Preference-based: beta={config.beta}")
        if config.use_distillation:
            print(f"  Distillation: temp={config.temperature}, alpha={config.distillation_alpha}")
