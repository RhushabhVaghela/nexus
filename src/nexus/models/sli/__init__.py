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
    EnhancedPrefetchBuffer,
    AccessPattern,
    LockFreeQueue,
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

from .sliding_window_buffer import (
    SlidingWindowBuffer,
    AdaptiveSlidingWindow,
    SlidingWindowConfig,
    WindowState,
    WindowEntry,
    WindowStats,
)

# New P1 Features
from .prefetch_engine import (
    PrefetchEngine,
    PrefetchConfig,
    PrefetchStats,
    PatternPredictor,
    PrefetchPattern,
    PrefetchPriority,
    create_prefetch_engine,
)

from .activation_cache import (
    ActivationCache,
    ActivationCacheConfig,
    ActivationCacheStats,
    ActivationCacheEntry,
    CacheInvalidationStrategy,
    CompressionType,
    ActivationCacheManager,
    get_activation_cache,
)

from .compressed_storage import (
    CompressedLayerStorage,
    LayerCompressor,
    CompressionConfig,
    CompressionAlgorithm,
    QuantizationType,
    CompressedEntry,
    compress_layer_to_storage,
    load_compressed_layer,
)

from .storage_tier_manager import (
    StorageTierManager,
    StorageTierConfig,
    StorageTier,
    AccessPattern as TierAccessPattern,
    TieredEntry,
    TierStats,
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

from .architecture_registry import ArchitectureRegistry

__all__ = [
    # Architecture Registry
    "ArchitectureRegistry",
    # Layer Cache (Original)
    "LayerCache",
    "LayerCacheManager",
    "get_layer_cache",
    "CacheEntry",
    "CacheStats",
    # Quantization (Original)
    "LayerQuantizer",
    "AdaptiveQuantizer",
    "QuantizationConfig",
    "QuantizationMode",
    "QuantizationRegistry",
    "quantize_layer",
    "dequantize_layer",
    "get_int8_config",
    "get_nf4_config",
    "get_fp4_config",
    "get_mixed_precision_config",
    # I/O Optimization (Enhanced)
    "IOOptimizer",
    "AsyncLayerPrefetcher",
    "ComputeIOOverlap",
    "SSDWearLeveling",
    "ParallelDownloader",
    "EnhancedPrefetchBuffer",
    "AccessPattern",
    "LockFreeQueue",
    "IOPriority",
    "IORequest",
    "IOStats",
    "get_io_optimizer",
    # Sliding Window Buffer
    "SlidingWindowBuffer",
    "AdaptiveSlidingWindow",
    "SlidingWindowConfig",
    "WindowState",
    "WindowEntry",
    "WindowStats",
    # Smart Layer Prefetching
    "PrefetchEngine",
    "PrefetchConfig",
    "PrefetchStats",
    "PatternPredictor",
    "PrefetchPattern",
    "PrefetchPriority",
    "create_prefetch_engine",
    # Better Activation Caching
    "ActivationCache",
    "ActivationCacheConfig",
    "ActivationCacheStats",
    "ActivationCacheEntry",
    "CacheInvalidationStrategy",
    "CompressionType",
    "ActivationCacheManager",
    "get_activation_cache",
    # Compressed Storage
    "CompressedLayerStorage",
    "LayerCompressor",
    "CompressionConfig",
    "CompressionAlgorithm",
    "QuantizationType",
    "CompressedEntry",
    "compress_layer_to_storage",
    "load_compressed_layer",
    # Storage Tier Manager
    "StorageTierManager",
    "StorageTierConfig",
    "StorageTier",
    "TieredEntry",
    "TierStats",
    # NVFP4 Streaming Loader (Phase 1)
    "NVFP4StreamingLoader",
    "NVFP4Quantizer",
    "NVFP4Config",
    "NVFP4Mode",
    "QuantizedTensor",
    "NVFP4QuantizationError",
    "get_nvfp4_config",
    "quantize_to_nvfp4",
    "dequantize_from_nvfp4",
    "NVFP4_AVAILABLE",
    # QAD Distillation Loss (Phase 1)
    "QADDistillationLoss",
    "QADLossConfig",
    "QADLossType",
    "QADLossStats",
    "PerLayerQADLoss",
    "QADLossError",
    "get_qad_loss_config",
    "compute_distillation_loss",
    # Nested Update Scheduler (Phase 1)
    "NestedUpdateScheduler",
    "NestedUpdateConfig",
    "UpdateGroup",
    "UpdateStats",
    "NestedSchedulerError",
    "get_nested_scheduler",
    "create_attention_focused_scheduler",
    # Hierarchical Layer Cache (Phase 2)
    "HierarchicalLayerCache",
    "HierarchicalCacheConfig",
    "HierarchicalCacheEntry",
    "CacheTier",
    "EvictionPolicy",
    "HierarchicalCacheError",
    # Advanced SLI Integrator (Phase 2)
    "AdvancedSLIIntegrator",
    "AdvancedSLIConfig",
    "LayerInfo",
    "AdvancedSLIError",
    "create_advanced_integrator",
    # Universal SLI (Legacy)
    "UniversalSLIIntegrator",
    "SequentialLayerIntegrator",
    # Exceptions
    "SLIError",
    "UnsupportedArchitectureError",
    "WeightLoadingError",
    "LayerCreationError",
    "MoEConfigurationError",
    "FormatDetectionError",
    "WeightMapError",
    # Storage Tier Access Pattern (alias)
    "TierAccessPattern",
]


# ---------------------------------------------------------------------------
# Lazy imports for additional SLI modules (avoid heavy torch load at import)
# ---------------------------------------------------------------------------
_LAZY_IMPORTS = {
    # cxl_pim_integration.py — CXL/PIM hardware integration
    "CXLDeviceConfig": (".cxl_pim_integration", "CXLDeviceConfig"),
    "CXLProcessingUnit": (".cxl_pim_integration", "CXLProcessingUnit"),
    "CXLMemoryPool": (".cxl_pim_integration", "CXLMemoryPool"),
    "CXLPIMInference": (".cxl_pim_integration", "CXLPIMInference"),
    "CXLIntegration": (".cxl_pim_integration", "CXLIntegration"),
    "HybridGPUPIM": (".cxl_pim_integration", "HybridGPUPIM"),
    # layer_factory.py — universal layer construction
    "UniversalLayerFactory": (".layer_factory", "UniversalLayerFactory"),
    # moe_handler.py — Mixture-of-Experts support
    "MoEConfig": (".moe_handler", "MoEConfig"),
    "MoEHandler": (".moe_handler", "MoEHandler"),
    # multi_head_latent_attention.py — MLA attention
    "MultiHeadLatentAttention": (
        ".multi_head_latent_attention",
        "MultiHeadLatentAttention",
    ),
    "TransMLAConverter": (".multi_head_latent_attention", "TransMLAConverter"),
    "MLAConfig": (".multi_head_latent_attention", "MLAConfig"),
    "MLASLIIntegrator": (".multi_head_latent_attention", "MLASLIIntegrator"),
    # nested_learning.py — nested/progressive learning strategies
    "NestedStrategy": (".nested_learning", "NestedStrategy"),
    "NestedLearningConfig": (".nested_learning", "NestedLearningConfig"),
    "LayerHierarchy": (".nested_learning", "LayerHierarchy"),
    "ProgressiveUnfreezer": (".nested_learning", "ProgressiveUnfreezer"),
    "HierarchicalDistiller": (".nested_learning", "HierarchicalDistiller"),
    "AdaptiveLRScheduler": (".nested_learning", "AdaptiveLRScheduler"),
    "NestedDropout": (".nested_learning", "NestedDropout"),
    "NestedLearning": (".nested_learning", "NestedLearning"),
    "CurriculumSampler": (".nested_learning", "CurriculumSampler"),
    "apply_nested_learning": (".nested_learning", "apply_nested_learning"),
    # nvfp4_qad.py — QAD-aware NVFP4 quantization
    # NOTE: NVFP4Config/NVFP4Quantizer already exported from nvfp4_loader;
    #       only non-colliding names are lazy-loaded here.
    "NVFP4Format": (".nvfp4_qad", "NVFP4Format"),
    "NVFP4Linear": (".nvfp4_qad", "NVFP4Linear"),
    "NVFP4QAD": (".nvfp4_qad", "NVFP4QAD"),
    "quantize_to_nvfp4": (".nvfp4_qad", "quantize_to_nvfp4"),
    # test_time_compute.py — test-time compute scaling
    "ComputeBudget": (".test_time_compute", "ComputeBudget"),
    "TestTimeConfig": (".test_time_compute", "TestTimeConfig"),
    "PromptComplexityAnalyzer": (".test_time_compute", "PromptComplexityAnalyzer"),
    "Verifier": (".test_time_compute", "Verifier"),
    "TestTimeComputeScaler": (".test_time_compute", "TestTimeComputeScaler"),
    "TestTimeSLIIntegration": (".test_time_compute", "TestTimeSLIIntegration"),
    # weight_loader.py — universal weight loading
    "UniversalWeightLoader": (".weight_loader", "UniversalWeightLoader"),
}


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        import importlib

        module_path, attr_name = _LAZY_IMPORTS[name]
        module = importlib.import_module(module_path, package=__name__)
        return getattr(module, attr_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# Extend __all__ with lazy-loaded names
__all__ += list(_LAZY_IMPORTS.keys())
