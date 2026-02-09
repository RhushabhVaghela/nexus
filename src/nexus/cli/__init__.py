"""
src/cli/__init__.py

Nexus CLI package with integrated polish features.
"""

from .completion import install_completion, show_completion_script

# ---------------------------------------------------------------------------
# Lazy imports
# ---------------------------------------------------------------------------
_LAZY_IMPORTS = {
    "main": (".nexus_cli", "main"),
}


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        import importlib

        module_path, attr_name = _LAZY_IMPORTS[name]
        module = importlib.import_module(module_path, package=__name__)
        return getattr(module, attr_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["install_completion", "show_completion_script", "main"]
