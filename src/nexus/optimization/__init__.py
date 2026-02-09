"""
Nexus Optimization — application-level optimization engines.

Includes KV-cache optimization and Remotion video generation engine.

Note: For model-level optimizations (pruning, fusion, etc.), see ``nexus.optimizations``.
"""

import importlib as _importlib

# ---------------------------------------------------------------------------
# Lazy imports — no submodule is loaded until an attribute is first accessed.
# ---------------------------------------------------------------------------
_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    # kv_cache.py
    "OptimizedKVCache": (".kv_cache", "OptimizedKVCache"),
    "RepetitionAwareCacheManager": (".kv_cache", "RepetitionAwareCacheManager"),
    "InferenceOptimizer": (".kv_cache", "InferenceOptimizer"),
    "KVCacheIntegration": (".kv_cache", "KVCacheIntegration"),
    "create_optimized_cache": (".kv_cache", "create_optimized_cache"),
    "create_cache_manager": (".kv_cache", "create_cache_manager"),
    # remotion_engine.py
    "RemotionExplainerEngine": (".remotion_engine", "RemotionExplainerEngine"),
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
    "OptimizedKVCache",
    "RepetitionAwareCacheManager",
    "InferenceOptimizer",
    "KVCacheIntegration",
    "create_optimized_cache",
    "create_cache_manager",
    "RemotionExplainerEngine",
]
