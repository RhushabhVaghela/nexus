"""
Nexus Optimizations Module

Research-backed optimization implementations for achieving 100+ tokens/second:

# Sequential Dependency Solutions (Blocker #1):
1. Layer Pipelining (EasySpec, SpecPipe, FlowSpec) - layer_pipelining.py
2. Adaptive Layer Skipping (SWIFT, LayerSkip, AdaSkip) - adaptive_layer_skipping.py
3. Semi-Autoregressive Decoding (SPACE) - semi_autoregressive.py

# Decompression Overhead Solutions (Blocker #2):
4. Async Decompression (nvCOMP-style) - async_decompression.py
5. Optimized Compression (ZSTD + quantization) - compression_optimized.py

# Forward Pass Time Solutions (Blocker #3):
6. Layer Fusion (NVIDIA Blackwell-style) - layer_fusion.py
7. Early Exit + Dynamic Routing (LayerSkip, DASH) - early_exit_routing.py
8. Low-Rank Attention + Sparsity - low_rank_attention.py

# Additional Research Implementations:
- ARMOR Pruning (arxiv:2510.05528) - armor_pruning.py
- SuffixDecoding (arxiv:2411.04975) - suffix_decoding.py
- Chimera Decoder (arxiv:2402.15758) - chimera_decoder.py
"""

import importlib as _importlib

# ---------------------------------------------------------------------------
# Lazy imports — no submodule is loaded until an attribute is first accessed.
# ---------------------------------------------------------------------------
_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    # layer_pipelining.py
    "LayerPipeliningOptimizer": (".layer_pipelining", "LayerPipeliningOptimizer"),
    "SpeculativeLayerExecutor": (".layer_pipelining", "SpeculativeLayerExecutor"),
    # adaptive_layer_skipping.py
    "AdaptiveLayerSkipper": (".adaptive_layer_skipping", "AdaptiveLayerSkipper"),
    "SWIFTSkipper": (".adaptive_layer_skipping", "SWIFTSkipper"),
    "LayerSkipIntegration": (".adaptive_layer_skipping", "LayerSkipIntegration"),
    # semi_autoregressive.py
    "SemiAutoregressiveDecoder": (".semi_autoregressive", "SemiAutoregressiveDecoder"),
    "SPACEDecoder": (".semi_autoregressive", "SPACEDecoder"),
    # async_decompression.py
    "AsyncDecompressor": (".async_decompression", "AsyncDecompressor"),
    "CUDAStreamManager": (".async_decompression", "CUDAStreamManager"),
    # compression_optimized.py
    "OptimizedCompressor": (".compression_optimized", "OptimizedCompressor"),
    "ZSTDQuantizedCompressor": (".compression_optimized", "ZSTDQuantizedCompressor"),
    # layer_fusion.py
    "LayerFusionOptimizer": (".layer_fusion", "LayerFusionOptimizer"),
    "FusedAttentionFFN": (".layer_fusion", "FusedAttentionFFN"),
    # early_exit_routing.py
    "EarlyExitRouter": (".early_exit_routing", "EarlyExitRouter"),
    "DynamicLayerRouter": (".early_exit_routing", "DynamicLayerRouter"),
    # low_rank_attention.py
    "LowRankAttention": (".low_rank_attention", "LowRankAttention"),
    "SparseAttentionOptimizer": (".low_rank_attention", "SparseAttentionOptimizer"),
    # armor_pruning.py
    "ARMORPruner": (".armor_pruning", "ARMORPruner"),
    "AdaptiveMaskGenerator": (".armor_pruning", "AdaptiveMaskGenerator"),
    "SparsityScheduler": (".armor_pruning", "SparsityScheduler"),
    # suffix_decoding.py
    "SuffixTrie": (".suffix_decoding", "SuffixTrie"),
    "SuffixCache": (".suffix_decoding", "SuffixCache"),
    "SuffixDecoder": (".suffix_decoding", "SuffixDecoder"),
    "SuffixDecodingIntegration": (".suffix_decoding", "SuffixDecodingIntegration"),
    # chimera_decoder.py
    "ChimeraHead": (".chimera_decoder", "ChimeraHead"),
    "ChimeraConfig": (".chimera_decoder", "ChimeraConfig"),
    "ChimeraWrapper": (".chimera_decoder", "ChimeraWrapper"),
    "ChimeraTrainer": (".chimera_decoder", "ChimeraTrainer"),
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
    return list(__all__) + ["__version__", "__all__"]


__all__ = [
    # Layer Pipelining
    "LayerPipeliningOptimizer",
    "SpeculativeLayerExecutor",
    # Adaptive Layer Skipping
    "AdaptiveLayerSkipper",
    "SWIFTSkipper",
    "LayerSkipIntegration",
    # Semi-Autoregressive Decoding
    "SemiAutoregressiveDecoder",
    "SPACEDecoder",
    # Async Decompression
    "AsyncDecompressor",
    "CUDAStreamManager",
    # Optimized Compression
    "OptimizedCompressor",
    "ZSTDQuantizedCompressor",
    # Layer Fusion
    "LayerFusionOptimizer",
    "FusedAttentionFFN",
    # Early Exit Routing
    "EarlyExitRouter",
    "DynamicLayerRouter",
    # Low-Rank Attention
    "LowRankAttention",
    "SparseAttentionOptimizer",
    # ARMOR Pruning
    "ARMORPruner",
    "AdaptiveMaskGenerator",
    "SparsityScheduler",
    # SuffixDecoding
    "SuffixTrie",
    "SuffixCache",
    "SuffixDecoder",
    "SuffixDecodingIntegration",
    # Chimera
    "ChimeraHead",
    "ChimeraConfig",
    "ChimeraWrapper",
    "ChimeraTrainer",
]
