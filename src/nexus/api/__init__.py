"""
Nexus API — REST endpoints for the Explainer service.

All public symbols are lazy-imported to avoid pulling in torch at package
import time (explainer_api imports RemotionExplainerEngine → torch).
"""

import importlib as _importlib

__all__ = [
    "ExplainerRequest",
    "ExplainerResponse",
    "ErrorResponse",
    "HealthResponse",
]

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "ExplainerRequest": (".explainer_api", "ExplainerRequest"),
    "ExplainerResponse": (".explainer_api", "ExplainerResponse"),
    "ErrorResponse": (".explainer_api", "ErrorResponse"),
    "HealthResponse": (".explainer_api", "HealthResponse"),
}


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name]
        module = _importlib.import_module(module_path, __name__)
        value = getattr(module, attr_name)
        globals()[name] = value  # cache for subsequent access
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return list(__all__)
