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
from .ring_attention import (
    RingAttention, RingAttentionConfig, RingAttentionWrapper, RingCommunicator,
    create_ring_attention
)
from .bookmark_indexation import (
    BookmarkIndexation, BookmarkConfig, BookmarkEntry, BookmarkIndex,
    TieredKVCache, DiskCache, StorageTier, create_bookmark_indexation
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
    # Ring Attention (multi-GPU)
    "RingAttention", "RingAttentionConfig", "RingAttentionWrapper",
    "RingCommunicator", "create_ring_attention",
    # Bookmark Indexation (tiered storage)
    "BookmarkIndexation", "BookmarkConfig", "BookmarkEntry", "BookmarkIndex",
    "TieredKVCache", "DiskCache", "StorageTier", "create_bookmark_indexation",
]
