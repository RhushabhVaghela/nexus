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

from .layer_pipelining import LayerPipeliningOptimizer, SpeculativeLayerExecutor
from .adaptive_layer_skipping import (
    AdaptiveLayerSkipper,
    SWIFTSkipper,
    LayerSkipIntegration,
)
from .semi_autoregressive import SemiAutoregressiveDecoder, SPACEDecoder
from .async_decompression import AsyncDecompressor, CUDAStreamManager
from .compression_optimized import OptimizedCompressor, ZSTDQuantizedCompressor
from .layer_fusion import LayerFusionOptimizer, FusedAttentionFFN
from .early_exit_routing import EarlyExitRouter, DynamicLayerRouter
from .low_rank_attention import LowRankAttention, SparseAttentionOptimizer
from .armor_pruning import ARMORPruner, AdaptiveMaskGenerator, SparsityScheduler
from .suffix_decoding import (
    SuffixTrie,
    SuffixCache,
    SuffixDecoder,
    SuffixDecodingIntegration,
)
from .chimera_decoder import ChimeraHead, ChimeraConfig, ChimeraWrapper, ChimeraTrainer

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

__version__ = "1.1.0"
