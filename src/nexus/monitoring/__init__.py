"""
Prometheus Monitoring for Nexus

Provides metrics collection, HTTP endpoint, and Grafana dashboards.

Author: Nexus Team
"""

import importlib as _importlib

# ---------------------------------------------------------------------------
# Lazy imports — no submodule is loaded until an attribute is first accessed.
# ---------------------------------------------------------------------------
_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    # metrics_server.py
    "MetricsServer": (".metrics_server", "MetricsServer"),
    "start_metrics_server": (".metrics_server", "start_metrics_server"),
    "stop_metrics_server": (".metrics_server", "stop_metrics_server"),
    "get_metrics_server": (".metrics_server", "get_metrics_server"),
    # collectors.py
    "MetricsCollector": (".collectors", "MetricsCollector"),
    "InferenceMetricsCollector": (".collectors", "InferenceMetricsCollector"),
    "CacheMetricsCollector": (".collectors", "CacheMetricsCollector"),
    "SystemMetricsCollector": (".collectors", "SystemMetricsCollector"),
    "get_collector": (".collectors", "get_collector"),
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
    # Server
    "MetricsServer",
    "start_metrics_server",
    "stop_metrics_server",
    "get_metrics_server",
    # Collectors
    "MetricsCollector",
    "InferenceMetricsCollector",
    "CacheMetricsCollector",
    "SystemMetricsCollector",
    "get_collector",
]
