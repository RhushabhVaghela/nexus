"""
Nexus SLI (Selective Layer Inference) Module

Advanced Selective Layer Inference with caching, quantization,
and I/O optimization for running large models on limited GPU memory.
"""

from .layer_cache import (
    LayerCache,
    LayerCacheManager,
    get_layer_cache,
    CacheEntry,
    CacheStats,
)

from .quantization import (
    LayerQuantizer,
    AdaptiveQuantizer,
    QuantizationConfig,
    QuantizationMode,
    QuantizationRegistry,
    quantize_layer,
    dequantize_layer,
    get_int8_config,
    get_nf4_config,
    get_fp4_config,
    get_mixed_precision_config,
)

from .io_optimizer import (
    IOOptimizer,
    AsyncLayerPrefetcher,
    ComputeIOOverlap,
    SSDWearLeveling,
    ParallelDownloader,
    IOPriority,
    IORequest,
    IOStats,
    get_io_optimizer,
)

__all__ = [
    # Layer Cache
    'LayerCache',
    'LayerCacheManager',
    'get_layer_cache',
    'CacheEntry',
    'CacheStats',
    
    # Quantization
    'LayerQuantizer',
    'AdaptiveQuantizer',
    'QuantizationConfig',
    'QuantizationMode',
    'QuantizationRegistry',
    'quantize_layer',
    'dequantize_layer',
    'get_int8_config',
    'get_nf4_config',
    'get_fp4_config',
    'get_mixed_precision_config',
    
    # I/O Optimization
    'IOOptimizer',
    'AsyncLayerPrefetcher',
    'ComputeIOOverlap',
    'SSDWearLeveling',
    'ParallelDownloader',
    'IOPriority',
    'IORequest',
    'IOStats',
    'get_io_optimizer',
]
