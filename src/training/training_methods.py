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
- Distillation (Knowledge Distillation with Temperature Scheduling)

Usage:
    from src.training_methods import TrainingMethod, get_training_config
    
    config = get_training_config(TrainingMethod.DISTILLATION)
    
    # With temperature scheduling
    config = get_training_config(
        TrainingMethod.DISTILLATION,
        temperature_schedule="cosine",  # or "linear"
        initial_temperature=5.0,
        final_temperature=1.0
    )
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Callable
import math


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


class TemperatureSchedule:
    """
    Temperature scheduling for distillation based on paper 2601.15394.
    Replaces fixed T=2.0 with decay schedule from T=5 to T=1.
    """
    
    @staticmethod
    def constant(temperature: float, **kwargs) -> float:
        """Constant temperature (backward compatible)."""
        return temperature
    
    @staticmethod
    def linear(current_step: int, total_steps: int, 
               initial_temperature: float = 5.0, 
               final_temperature: float = 1.0,
               **kwargs) -> float:
        """
        Linear temperature decay.
        T(t) = T_initial - (T_initial - T_final) * (t / T_total)
        """
        if total_steps == 0:
            return final_temperature
        progress = min(current_step / total_steps, 1.0)
        return initial_temperature - (initial_temperature - final_temperature) * progress
    
    @staticmethod
    def cosine(current_step: int, total_steps: int,
               initial_temperature: float = 5.0,
               final_temperature: float = 1.0,
               **kwargs) -> float:
        """
        Cosine temperature decay.
        T(t) = T_final + (T_initial - T_final) * 0.5 * (1 + cos(π * t / T_total))
        """
        if total_steps == 0:
            return final_temperature
        progress = min(current_step / total_steps, 1.0)
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        return final_temperature + (initial_temperature - final_temperature) * cosine_decay
    
    @staticmethod
    def exponential(current_step: int, total_steps: int,
                    initial_temperature: float = 5.0,
                    final_temperature: float = 1.0,
                    decay_rate: float = 0.95,
                    **kwargs) -> float:
        """
        Exponential temperature decay.
        T(t) = T_final + (T_initial - T_final) * decay_rate^(t/T_total)
        """
        if total_steps == 0:
            return final_temperature
        progress = min(current_step / total_steps, 1.0)
        decay = decay_rate ** progress
        return final_temperature + (initial_temperature - final_temperature) * decay
    
    @staticmethod
    def get_schedule(schedule_type: str) -> Callable:
        """Get temperature schedule function by name."""
        schedules = {
            "constant": TemperatureSchedule.constant,
            "linear": TemperatureSchedule.linear,
            "cosine": TemperatureSchedule.cosine,
            "exponential": TemperatureSchedule.exponential,
        }
        return schedules.get(schedule_type, TemperatureSchedule.constant)


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
    # Legacy: fixed temperature for backward compatibility
    temperature: float = 2.0
    distillation_alpha: float = 0.5
    
    # Temperature Scheduling (Paper 2601.15394)
    use_temperature_schedule: bool = False
    temperature_schedule: str = "cosine"  # "constant", "linear", "cosine", "exponential"
    initial_temperature: float = 5.0      # T=5 at start
    final_temperature: float = 1.0        # T=1 at end
    
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
            "use_temperature_schedule": self.use_temperature_schedule,
            "temperature_schedule": self.temperature_schedule,
            "initial_temperature": self.initial_temperature,
            "final_temperature": self.final_temperature,
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
        }
    
    def get_temperature(self, current_step: int = 0, total_steps: int = 1) -> float:
        """
        Get the temperature for the current training step.
        
        Args:
            current_step: Current training step
            total_steps: Total number of training steps
            
        Returns:
            Temperature value for current step
        """
        if not self.use_temperature_schedule:
            # Backward compatible: return fixed temperature
            return self.temperature
        
        schedule_fn = TemperatureSchedule.get_schedule(self.temperature_schedule)
        return schedule_fn(
            current_step=current_step,
            total_steps=total_steps,
            initial_temperature=self.initial_temperature,
            final_temperature=self.final_temperature
        )


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
        description="Knowledge Distillation - Learn from teacher model with temperature scheduling",
        use_distillation=True,
        # Legacy: maintain backward compatibility with fixed temperature
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


def get_training_config(method: TrainingMethod, 
                        use_temperature_schedule: bool = False,
                        temperature_schedule: str = "cosine",
                        initial_temperature: float = 5.0,
                        final_temperature: float = 1.0,
                        **kwargs) -> TrainingMethodConfig:
    """
    Get the default configuration for a training method.
    
    Args:
        method: Training method enum
        use_temperature_schedule: Enable temperature scheduling for distillation
        temperature_schedule: Type of schedule ("linear", "cosine", "exponential", "constant")
        initial_temperature: Starting temperature (default: 5.0)
        final_temperature: Ending temperature (default: 1.0)
        **kwargs: Additional configuration overrides
        
    Returns:
        TrainingMethodConfig instance
    """
    config = TRAINING_CONFIGS.get(method, TRAINING_CONFIGS[TrainingMethod.SFT]).__class__(
        **TRAINING_CONFIGS.get(method, TRAINING_CONFIGS[TrainingMethod.SFT]).__dict__
    )
    
    # Apply temperature scheduling if requested
    if use_temperature_schedule and method == TrainingMethod.DISTILLATION:
        config.use_temperature_schedule = True
        config.temperature_schedule = temperature_schedule
        config.initial_temperature = initial_temperature
        config.final_temperature = final_temperature
    
    # Apply any additional overrides
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    return config


def get_distillation_config_with_schedule(
    schedule_type: str = "cosine",
    initial_temp: float = 5.0,
    final_temp: float = 1.0,
    **kwargs
) -> TrainingMethodConfig:
    """
    Convenience function to get distillation config with temperature scheduling.
    
    Args:
        schedule_type: "linear", "cosine", or "exponential"
        initial_temp: Initial temperature (default 5.0 as per paper)
        final_temp: Final temperature (default 1.0 as per paper)
        **kwargs: Additional configuration options
        
    Returns:
        TrainingMethodConfig for distillation with scheduling enabled
    """
    return get_training_config(
        TrainingMethod.DISTILLATION,
        use_temperature_schedule=True,
        temperature_schedule=schedule_type,
        initial_temperature=initial_temp,
        final_temperature=final_temp,
        **kwargs
    )


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


def create_distillation_callback(config: TrainingMethodConfig):
    """
    Create a callback function for temperature scheduling during training.
    
    This callback can be integrated with training loops to update temperature
    at each step.
    
    Args:
        config: Training configuration with temperature scheduling
        
    Returns:
        Callback function that takes (current_step, total_steps) and returns temperature
    """
    if not config.use_distillation:
        raise ValueError("Temperature scheduling is only applicable to distillation")
    
    def temperature_callback(current_step: int, total_steps: int) -> float:
        return config.get_temperature(current_step, total_steps)
    
    return temperature_callback


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
            if config.use_temperature_schedule:
                print(f"  Distillation: With {config.temperature_schedule} schedule")
                print(f"    Temperature: {config.initial_temperature} → {config.final_temperature}")
            else:
                print(f"  Distillation: Fixed temp={config.temperature}")
            print(f"  Alpha: {config.distillation_alpha}")
    
    # Demonstrate temperature scheduling
    print("\n" + "=" * 60)
    print("Temperature Scheduling Demonstration")
    print("=" * 60)
    
    total_steps = 1000
    for schedule_type in ["linear", "cosine", "exponential"]:
        print(f"\n{schedule_type.capitalize()} Schedule (T=5.0 → T=1.0):")
        config = get_distillation_config_with_schedule(schedule_type)
        for step in [0, 250, 500, 750, 1000]:
            temp = config.get_temperature(step, total_steps)
            print(f"  Step {step:4d}/{total_steps}: T = {temp:.3f}")
