"""
Nexus Inference — optimized inference engine with quantization and batching.
"""

import importlib as _importlib

# ---------------------------------------------------------------------------
# Lazy imports — no submodule is loaded until an attribute is first accessed.
# ---------------------------------------------------------------------------
_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "OptimizedInference": (".optimized_inference", "OptimizedInference"),
    "OptimizedInferenceConfig": (".optimized_inference", "OptimizedInferenceConfig"),
    "OptimizationRegistry": (".optimized_inference", "OptimizationRegistry"),
    "create_optimized_inference": (
        ".optimized_inference",
        "create_optimized_inference",
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
    "OptimizedInference",
    "OptimizedInferenceConfig",
    "OptimizationRegistry",
    "create_optimized_inference",
]
