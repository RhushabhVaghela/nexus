"""
Nexus Core Module

Provides core functionality for the Nexus system:
- Unified exception hierarchy
- Resilience patterns (circuit breaker, retry, bulkhead)
- Orchestration capabilities

Example:
    from src.core.exceptions import CircuitBreakerOpen, RateLimitExceeded
    from src.core.resilience import CircuitBreaker
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
