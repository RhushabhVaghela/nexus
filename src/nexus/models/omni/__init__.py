# src/omni/__init__.py
"""Omni model support for Qwen2.5-Omni models."""

from .loader import OmniModelLoader, OmniModelConfig, load_omni_model

# ---------------------------------------------------------------------------
# Lazy imports for heavier omni modules
# ---------------------------------------------------------------------------
_LAZY_IMPORTS = {
    # inference.py — OmniModel inference pipeline
    "GenerationConfig": (".inference", "GenerationConfig"),
    "OmniInference": (".inference", "OmniInference"),
}


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        import importlib

        module_path, attr_name = _LAZY_IMPORTS[name]
        module = importlib.import_module(module_path, package=__name__)
        return getattr(module, attr_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "OmniModelLoader",
    "OmniModelConfig",
    "load_omni_model",
] + list(_LAZY_IMPORTS.keys())
