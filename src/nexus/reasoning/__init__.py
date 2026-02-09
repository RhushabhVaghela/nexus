"""Reasoning module for advanced model training."""

import importlib as _importlib

# ---------------------------------------------------------------------------
# Lazy imports — no submodule is loaded until an attribute is first accessed.
# ---------------------------------------------------------------------------
_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    # cot_generator.py
    "CoTGenerator": (".cot_generator", "CoTGenerator"),
    "CoTConfig": (".cot_generator", "CoTConfig"),
    "ThinkingTrace": (".cot_generator", "ThinkingTrace"),
    "ReasoningType": (".cot_generator", "ReasoningType"),
    # reward_functions.py
    "RewardFunction": (".reward_functions", "RewardFunction"),
    "CombinedReward": (".reward_functions", "CombinedReward"),
    "CorrectnessReward": (".reward_functions", "CorrectnessReward"),
    "FormatReward": (".reward_functions", "FormatReward"),
    "LengthReward": (".reward_functions", "LengthReward"),
    "ConsistencyReward": (".reward_functions", "ConsistencyReward"),
    "ProcessReward": (".reward_functions", "ProcessReward"),
    "RewardConfig": (".reward_functions", "RewardConfig"),
    "RewardResult": (".reward_functions", "RewardResult"),
    "create_reward_function": (".reward_functions", "create_reward_function"),
    # context_extension.py
    "ContextExtender": (".context_extension", "ContextExtender"),
    "ContextExtensionConfig": (".context_extension", "ContextExtensionConfig"),
    "RoPEScaler": (".context_extension", "RoPEScaler"),
    "ScalingType": (".context_extension", "ScalingType"),
    "create_context_extender": (".context_extension", "create_context_extender"),
    # ring_attention.py
    "RingAttention": (".ring_attention", "RingAttention"),
    "RingAttentionConfig": (".ring_attention", "RingAttentionConfig"),
    "RingAttentionWrapper": (".ring_attention", "RingAttentionWrapper"),
    "RingCommunicator": (".ring_attention", "RingCommunicator"),
    "create_ring_attention": (".ring_attention", "create_ring_attention"),
    # bookmark_indexation.py
    "BookmarkIndexation": (".bookmark_indexation", "BookmarkIndexation"),
    "BookmarkConfig": (".bookmark_indexation", "BookmarkConfig"),
    "BookmarkEntry": (".bookmark_indexation", "BookmarkEntry"),
    "BookmarkIndex": (".bookmark_indexation", "BookmarkIndex"),
    "TieredKVCache": (".bookmark_indexation", "TieredKVCache"),
    "DiskCache": (".bookmark_indexation", "DiskCache"),
    "StorageTier": (".bookmark_indexation", "StorageTier"),
    "create_bookmark_indexation": (
        ".bookmark_indexation",
        "create_bookmark_indexation",
    ),
}


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name]
        module = _importlib.import_module(module_path, __name__)
        value = getattr(module, attr_name)
        globals()[name] = value  # Cache for subsequent access
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return list(__all__)


__all__ = [
    # CoT Generation
    "CoTGenerator",
    "CoTConfig",
    "ThinkingTrace",
    "ReasoningType",
    # Rewards
    "RewardFunction",
    "CombinedReward",
    "CorrectnessReward",
    "FormatReward",
    "LengthReward",
    "ConsistencyReward",
    "ProcessReward",
    "RewardConfig",
    "RewardResult",
    "create_reward_function",
    # Context Extension
    "ContextExtender",
    "ContextExtensionConfig",
    "RoPEScaler",
    "ScalingType",
    "create_context_extender",
    # Ring Attention (multi-GPU)
    "RingAttention",
    "RingAttentionConfig",
    "RingAttentionWrapper",
    "RingCommunicator",
    "create_ring_attention",
    # Bookmark Indexation (tiered storage)
    "BookmarkIndexation",
    "BookmarkConfig",
    "BookmarkEntry",
    "BookmarkIndex",
    "TieredKVCache",
    "DiskCache",
    "StorageTier",
    "create_bookmark_indexation",
]
