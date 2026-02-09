"""
Nexus Core Module

Provides core functionality for the Nexus system:
- Unified exception hierarchy
- Resilience patterns (circuit breaker, retry, bulkhead)
- Orchestration capabilities

Example:
    from nexus.core.exceptions import CircuitBreakerOpen, RateLimitExceeded
    from nexus.core.resilience import CircuitBreaker
"""

# Core exceptions
from .exceptions import (
    NexusBaseError,
    NexusValueError,
    NexusTypeError,
    NexusIOError,
    NexusConfigError,
    NexusRuntimeError,
    NexusTimeoutError,
    CircuitBreakerOpen,
    RateLimitExceeded,
    BulkheadFull,
    BulkheadTimeout,
    SLIError,
    TrainingError,
    InferenceError,
    # SLI-specific exceptions
    UnsupportedArchitectureError,
    WeightLoadingError,
    LayerCreationError,
    MoEConfigurationError,
    FormatDetectionError,
    WeightMapError,
)

__all__ = [
    # Base exceptions
    "NexusBaseError",
    "NexusValueError",
    "NexusTypeError",
    "NexusIOError",
    "NexusConfigError",
    "NexusRuntimeError",
    "NexusTimeoutError",
    # Resilience exceptions
    "CircuitBreakerOpen",
    "RateLimitExceeded",
    "BulkheadFull",
    "BulkheadTimeout",
    # Domain-specific exceptions
    "SLIError",
    "TrainingError",
    "InferenceError",
    # SLI-specific exceptions
    "UnsupportedArchitectureError",
    "WeightLoadingError",
    "LayerCreationError",
    "MoEConfigurationError",
    "FormatDetectionError",
    "WeightMapError",
]

# ---------------------------------------------------------------------------
# Lazy imports for heavier core modules
# ---------------------------------------------------------------------------
_LAZY_IMPORTS = {
    # resilience.py — full resilience toolkit
    "CircuitState": (".resilience", "CircuitState"),
    "CircuitBreakerConfig": (".resilience", "CircuitBreakerConfig"),
    "RetryConfig": (".resilience", "RetryConfig"),
    "CircuitBreaker": (".resilience", "CircuitBreaker"),
    "RetryPolicy": (".resilience", "RetryPolicy"),
    "Bulkhead": (".resilience", "Bulkhead"),
    "Timeout": (".resilience", "Timeout"),
    "ResilientClient": (".resilience", "ResilientClient"),
    "circuit_breaker": (".resilience", "circuit_breaker"),
    "retry": (".resilience", "retry"),
    "timeout": (".resilience", "timeout"),
    # config.py — model/system configuration
    "ModelInfo": (".config", "ModelInfo"),
    "NexusCoreConfig": (
        ".config",
        "NexusConfig",
    ),  # aliased to avoid clash with models.config
    # capability_audit.py — runtime capability auditing
    "audit_capabilities": (".capability_audit", "audit_capabilities"),
    # capability_registry.py — capability/reasoning registry
    "ReasoningLevel": (".capability_registry", "ReasoningLevel"),
    "Capability": (".capability_registry", "Capability"),
    "CapabilityRegistry": (".capability_registry", "CapabilityRegistry"),
    # metrics_tracker.py — training/benchmark metrics
    "TrainingMetrics": (".metrics_tracker", "TrainingMetrics"),
    "ValidationMetrics": (".metrics_tracker", "ValidationMetrics"),
    "BenchmarkMetrics": (".metrics_tracker", "BenchmarkMetrics"),
    "MetricsTracker": (".metrics_tracker", "MetricsTracker"),
    "ProgressTracker": (".metrics_tracker", "ProgressTracker"),
    "run_with_progress": (".metrics_tracker", "run_with_progress"),
    "discover_datasets": (".metrics_tracker", "discover_datasets"),
    "get_all_datasets": (".metrics_tracker", "get_all_datasets"),
    "get_capability_datasets": (".metrics_tracker", "get_capability_datasets"),
    # inventory.py — model inventory management
    "load_inventory": (".inventory", "load_inventory"),
}


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        import importlib

        module_path, attr_name = _LAZY_IMPORTS[name]
        module = importlib.import_module(module_path, package=__name__)
        return getattr(module, attr_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ += list(_LAZY_IMPORTS.keys())
