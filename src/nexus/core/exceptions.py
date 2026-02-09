"""
Unified Nexus Exception Hierarchy

Base exceptions for all Nexus modules:
- NexusBaseError - Base exception
  - NexusValueError - Invalid values
  - NexusTypeError - Type errors
  - NexusIOError - I/O errors
  - NexusConfigError - Configuration errors
  - NexusRuntimeError - Runtime errors
    - CircuitBreakerOpen
    - RateLimitExceeded
    - BulkheadFull
    - BulkheadTimeout
    - NexusTimeoutError
    - SLIError
    - TrainingError
    - InferenceError
"""

import warnings
from typing import Optional, Any


class NexusBaseError(Exception):
    """Base exception for all Nexus errors."""

    def __init__(self, message: str = "Nexus error occurred", **kwargs: Any):
        self.message = message
        # Store additional context
        for key, value in kwargs.items():
            setattr(self, key, value)
        super().__init__(message)


class NexusValueError(NexusBaseError, ValueError):
    """Invalid value error."""

    pass


class NexusTypeError(NexusBaseError, TypeError):
    """Type error."""

    pass


class NexusIOError(NexusBaseError, IOError):
    """I/O error."""

    pass


class NexusConfigError(NexusBaseError, ValueError):
    """Configuration error."""

    pass


class NexusRuntimeError(NexusBaseError, RuntimeError):
    """Runtime error."""

    pass


class NexusTimeoutError(NexusRuntimeError, TimeoutError):
    """Timeout error."""

    pass


class CircuitBreakerOpen(NexusRuntimeError):
    """Circuit breaker is open.

    Raised when a circuit breaker rejects requests because it has detected
    too many failures and is in the OPEN state.

    Attributes:
        name: Name of the circuit breaker
        last_error: The last error that triggered the circuit to open
    """

    def __init__(
        self, name: str, last_error: Optional[str] = None, message: Optional[str] = None
    ):
        self.name = name
        self.last_error = last_error
        display_message = (
            message or f"Circuit breaker '{name}' is OPEN. Last error: {last_error}"
        )
        super().__init__(display_message)

    def __reduce__(self):
        """Support pickling by reconstructing with all arguments."""
        return (self.__class__, (self.name, self.last_error, None))


class RateLimitExceeded(NexusRuntimeError):
    """Rate limit exceeded.

    Raised when a request is rejected due to rate limiting.

    Attributes:
        key: The rate limit key (e.g., user ID, API key, IP address)
        limit: The rate limit that was exceeded
        retry_after: Seconds until the client can retry
    """

    def __init__(
        self, key: str, limit: int, retry_after: float, message: Optional[str] = None
    ):
        self.key = key
        self.limit = limit
        self.retry_after = retry_after
        display_message = message or (
            f"Rate limit exceeded for '{key}'. "
            f"Limit: {limit}, retry after: {retry_after:.2f}s"
        )
        super().__init__(display_message)

    def __reduce__(self):
        """Support pickling by reconstructing with all arguments."""
        return (self.__class__, (self.key, self.limit, self.retry_after, None))


class BulkheadFull(NexusRuntimeError):
    """Bulkhead is full.

    Raised when a bulkhead (resource pool) has reached its capacity.
    """

    pass


class BulkheadTimeout(NexusRuntimeError):
    """Bulkhead timeout.

    Raised when an operation times out while waiting for bulkhead resources.
    """

    pass


class SLIError(NexusRuntimeError):
    """SLI (Service Level Indicator) error.

    Base exception for all SLI-related errors.
    """

    pass


class UnsupportedArchitectureError(SLIError):
    """Raised when an unsupported architecture is encountered."""

    def __init__(self, model_type: str, architectures: list = None):
        self.model_type = model_type
        self.architectures = architectures or []
        msg = f"Unsupported architecture: model_type='{model_type}'"
        if self.architectures:
            msg += f", architectures={self.architectures}"
        msg += ". This architecture family is not yet supported by Universal SLI."
        super().__init__(msg)


class WeightLoadingError(SLIError):
    """Raised when weight loading fails."""

    def __init__(
        self, weight_name: str, shard_name: str = None, cause: Exception = None
    ):
        self.weight_name = weight_name
        self.shard_name = shard_name
        self.cause = cause
        msg = f"Failed to load weight: {weight_name}"
        if shard_name:
            msg += f" from shard: {shard_name}"
        if cause:
            msg += f". Cause: {str(cause)}"
        super().__init__(msg)


class LayerCreationError(SLIError):
    """Raised when layer creation fails."""

    def __init__(self, layer_idx: int, family_id: str, cause: Exception = None):
        self.layer_idx = layer_idx
        self.family_id = family_id
        self.cause = cause
        msg = f"Failed to create layer {layer_idx} for family '{family_id}'"
        if cause:
            msg += f". Cause: {str(cause)}"
        super().__init__(msg)


class MoEConfigurationError(SLIError):
    """Raised when MoE configuration is invalid or unsupported."""

    def __init__(self, moe_type: str, message: str = None):
        self.moe_type = moe_type
        msg = message or f"Invalid or unsupported MoE configuration: {moe_type}"
        super().__init__(msg)


class FormatDetectionError(SLIError):
    """Raised when weight format cannot be detected."""

    def __init__(self, model_id: str, attempted_formats: list = None):
        self.model_id = model_id
        self.attempted_formats = attempted_formats or []
        msg = f"Could not detect weight format for model: {model_id}"
        if self.attempted_formats:
            msg += f". Attempted formats: {self.attempted_formats}"
        super().__init__(msg)


class WeightMapError(SLIError):
    """Raised when weight map is missing or invalid."""

    def __init__(self, model_id: str, index_file: str = None):
        self.model_id = model_id
        self.index_file = index_file
        msg = f"Weight map error for model: {model_id}"
        if index_file:
            msg += f". Index file: {index_file}"
        super().__init__(msg)


class TrainingError(NexusRuntimeError):
    """Training error.

    Base exception for all training-related errors.
    """

    pass


class InferenceError(NexusRuntimeError):
    """Inference error.

    Base exception for all inference-related errors.
    """

    pass


# Backward compatibility aliases for deprecated exception classes


def _create_deprecation_warning(old_path: str, new_path: str) -> str:
    """Create a deprecation warning message."""
    return (
        f"{old_path} is deprecated. "
        f"Please use {new_path} from nexus.core.exceptions instead. "
        f"This will be removed in a future version."
    )


# ============================================================================
# DEPRECATED: CircuitBreakerOpen in core/resilience.py
# ============================================================================


class _DeprecatedCircuitBreakerOpen(Exception):
    """Deprecated: Use CircuitBreakerOpen from nexus.core.exceptions instead."""

    pass


# ============================================================================
# DEPRECATED: CircuitBreakerOpen in utils/circuit_breaker.py
# ============================================================================


def _get_utils_circuit_breaker_circuit_breaker_open():
    """Return the deprecated CircuitBreakerOpen from utils.circuit_breaker."""
    warnings.warn(
        _create_deprecation_warning(
            "CircuitBreakerOpen from utils.circuit_breaker",
            "CircuitBreakerOpen from nexus.core.exceptions",
        ),
        DeprecationWarning,
        stacklevel=2,
    )
    from nexus.utils.circuit_breaker import CircuitBreakerOpen as Original

    return Original


# ============================================================================
# DEPRECATED: RateLimitExceeded in security/rate_limiter.py
# ============================================================================


def _get_security_rate_limiter_rate_limit_exceeded():
    """Return the deprecated RateLimitExceeded from security.rate_limiter."""
    warnings.warn(
        _create_deprecation_warning(
            "RateLimitExceeded from security.rate_limiter",
            "RateLimitExceeded from nexus.core.exceptions",
        ),
        DeprecationWarning,
        stacklevel=2,
    )
    from nexus.security.rate_limiter import RateLimitExceeded as Original

    return Original


# ============================================================================
# DEPRECATED: RateLimitExceeded in utils/rate_limiter.py
# ============================================================================


def _get_utils_rate_limiter_rate_limit_exceeded():
    """Return the deprecated RateLimitExceeded from utils.rate_limiter."""
    warnings.warn(
        _create_deprecation_warning(
            "RateLimitExceeded from utils.rate_limiter",
            "RateLimitExceeded from nexus.core.exceptions",
        ),
        DeprecationWarning,
        stacklevel=2,
    )
    from nexus.utils.rate_limiter import RateLimitExceeded as Original

    return Original
