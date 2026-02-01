"""
Nexus SLI (Selective Layer Inference) Module

Advanced Selective Layer Inference with caching, quantization,
and I/O optimization for running large models on limited GPU memory.

Phase 1 & 2 Features:
- NVFP4 Streaming Loader: Hardware-accelerated 4-bit quantization
- QAD Distillation Loss: Knowledge transfer from FP32 to NVFP4
- Nested Update Scheduler: Efficient three-tier training
- Hierarchical Layer Cache: Three-tier caching system
- Advanced SLI Integrator: Unified integration of all components
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

# Phase 1 & 2: NVFP4-QAD and Nested Learning
from .nvfp4_loader import (
    NVFP4StreamingLoader,
    NVFP4Quantizer,
    NVFP4Config,
    NVFP4Mode,
    QuantizedTensor,
    NVFP4QuantizationError,
    get_nvfp4_config,
    quantize_to_nvfp4,
    dequantize_from_nvfp4,
    NVFP4_AVAILABLE,
)

from .qad_loss import (
    QADDistillationLoss,
    QADLossConfig,
    QADLossType,
    QADLossStats,
    PerLayerQADLoss,
    QADLossError,
    get_qad_loss_config,
    compute_distillation_loss,
)

from .nested_scheduler import (
    NestedUpdateScheduler,
    NestedUpdateConfig,
    UpdateGroup,
    UpdateStats,
    NestedSchedulerError,
    get_nested_scheduler,
    create_attention_focused_scheduler,
)

from .hierarchical_cache import (
    HierarchicalLayerCache,
    HierarchicalCacheConfig,
    HierarchicalCacheEntry,
    CacheTier,
    EvictionPolicy,
    HierarchicalCacheError,
)

from .advanced_sli_integrator import (
    AdvancedSLIIntegrator,
    AdvancedSLIConfig,
    LayerInfo,
    AdvancedSLIError,
    create_advanced_integrator,
)

# Legacy imports for backward compatibility
from .universal_sli_integrator import (
    UniversalSLIIntegrator,
    SequentialLayerIntegrator,
)

from .exceptions import (
    SLIError,
    UnsupportedArchitectureError,
    WeightLoadingError,
    LayerCreationError,
    MoEConfigurationError,
    FormatDetectionError,
    WeightMapError,
)

__all__ = [
    # Layer Cache (Original)
    'LayerCache',
    'LayerCacheManager',
    'get_layer_cache',
    'CacheEntry',
    'CacheStats',
    
    # Quantization (Original)
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
    
    # I/O Optimization (Original)
    'IOOptimizer',
    'AsyncLayerPrefetcher',
    'ComputeIOOverlap',
    'SSDWearLeveling',
    'ParallelDownloader',
    'IOPriority',
    'IORequest',
    'IOStats',
    'get_io_optimizer',
    
    # NVFP4 Streaming Loader (Phase 1)
    'NVFP4StreamingLoader',
    'NVFP4Quantizer',
    'NVFP4Config',
    'NVFP4Mode',
    'QuantizedTensor',
    'NVFP4QuantizationError',
    'get_nvfp4_config',
    'quantize_to_nvfp4',
    'dequantize_from_nvfp4',
    'NVFP4_AVAILABLE',
    
    # QAD Distillation Loss (Phase 1)
    'QADDistillationLoss',
    'QADLossConfig',
    'QADLossType',
    'QADLossStats',
    'PerLayerQADLoss',
    'QADLossError',
    'get_qad_loss_config',
    'compute_distillation_loss',
    
    # Nested Update Scheduler (Phase 1)
    'NestedUpdateScheduler',
    'NestedUpdateConfig',
    'UpdateGroup',
    'UpdateStats',
    'NestedSchedulerError',
    'get_nested_scheduler',
    'create_attention_focused_scheduler',
    
    # Hierarchical Layer Cache (Phase 2)
    'HierarchicalLayerCache',
    'HierarchicalCacheConfig',
    'HierarchicalCacheEntry',
    'CacheTier',
    'EvictionPolicy',
    'HierarchicalCacheError',
    
    # Advanced SLI Integrator (Phase 2)
    'AdvancedSLIIntegrator',
    'AdvancedSLIConfig',
    'LayerInfo',
    'AdvancedSLIError',
    'create_advanced_integrator',
    
    # Universal SLI (Legacy)
    'UniversalSLIIntegrator',
    'SequentialLayerIntegrator',
    
    # Exceptions
    'SLIError',
    'UnsupportedArchitectureError',
    'WeightLoadingError',
    'LayerCreationError',
    'MoEConfigurationError',
    'FormatDetectionError',
    'WeightMapError',
]

__version__ = "2.0.0"
