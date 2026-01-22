"""Reasoning module for advanced model training."""
from .cot_generator import CoTGenerator, CoTConfig, ThinkingTrace, ReasoningType
from .reward_functions import (
    RewardFunction, CombinedReward, CorrectnessReward, FormatReward,
    LengthReward, ConsistencyReward, ProcessReward, RewardConfig, RewardResult,
    create_reward_function
)
from .context_extension import (
    ContextExtender, ContextExtensionConfig, RoPEScaler, ScalingType,
    create_context_extender
)

__all__ = [
    # CoT Generation
    "CoTGenerator", "CoTConfig", "ThinkingTrace", "ReasoningType",
    # Rewards
    "RewardFunction", "CombinedReward", "CorrectnessReward", "FormatReward",
    "LengthReward", "ConsistencyReward", "ProcessReward", "RewardConfig",
    "RewardResult", "create_reward_function",
    # Context Extension
    "ContextExtender", "ContextExtensionConfig", "RoPEScaler", "ScalingType",
    "create_context_extender",
]
