"""
Resilience Patterns for Nexus

Implements circuit breaker pattern and other resilience mechanisms:
- Circuit Breaker: Prevent cascading failures
- Retry with Exponential Backoff: Handle transient failures
- Bulkhead Isolation: Resource pooling
- Timeout Management: Prevent hanging operations

Author: Nexus Team
"""

import time
import threading
import logging
import warnings
from typing import Callable, Optional, Any, Dict, List, Type, Union, Tuple
from enum import Enum
from dataclasses import dataclass, field
from functools import wraps
import random

# Import unified exceptions for backward compatibility
from nexus.core.exceptions import (
    CircuitBreakerOpen as UnifiedCircuitBreakerOpen,
)

logger = logging.getLogger(__name__)


def _warn_deprecated_circuit_breaker_open():
    """Warn about deprecated CircuitBreakerOpen in this module."""
    warnings.warn(
        "CircuitBreakerOpen from nexus.core.resilience is deprecated. "
        "Please use CircuitBreakerOpen from nexus.core.exceptions instead. "
        "This will be removed in a future version.",
        DeprecationWarning,
        stacklevel=3,
    )


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, rejecting requests
    HALF_OPEN = "half_open"  # Testing if recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""

    failure_threshold: int = 5  # Failures before opening
    recovery_timeout: float = 30.0  # Seconds before half-open
    half_open_max_calls: int = 3  # Test calls in half-open
    success_threshold: int = 2  # Successes to close
    expected_exception: Type[Exception] = Exception


@dataclass
class RetryConfig:
    """Configuration for retry mechanism."""

    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    retryable_exceptions: Tuple[Type[Exception], ...] = (Exception,)


class CircuitBreaker:
    """
    Circuit Breaker pattern implementation.

    Prevents cascading failures by rejecting requests when
    a service is failing repeatedly.

    Example:
        >>> breaker = CircuitBreaker(CircuitBreakerConfig())
        >>>
        >>> @breaker
        ... def unstable_service():
        ...     # Might fail
        ...     pass
        >>>
        >>> result = unstable_service()  # Protected by circuit breaker

    States:
        - CLOSED: Normal operation, requests pass through
        - OPEN: Service failing, requests rejected immediately
        - HALF_OPEN: Testing recovery with limited requests
    """

    def __init__(
        self, config: Optional[CircuitBreakerConfig] = None, name: str = "default"
    ):
        self.config = config or CircuitBreakerConfig()
        self.name = name

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: Optional[float] = None
        self._half_open_calls = 0

        self._lock = threading.RLock()

        # Statistics
        self._stats = {
            "calls": 0,
            "successes": 0,
            "failures": 0,
            "rejected": 0,
            "state_transitions": 0,
        }

    def __call__(self, func: Callable) -> Callable:
        """Decorator to protect function with circuit breaker."""

        @wraps(func)
        def wrapper(*args, **kwargs):
            return self.call(func, *args, **kwargs)

        return wrapper

    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function with circuit breaker protection.

        Args:
            func: Function to protect
            *args, **kwargs: Arguments to function

        Returns:
            Function result

        Raises:
            CircuitBreakerOpen: If circuit is open
            Exception: If function fails
        """
        with self._lock:
            self._update_state()

            if self._state == CircuitState.OPEN:
                self._stats["rejected"] += 1
                raise CircuitBreakerOpen(
                    f"Circuit '{self.name}' is OPEN. "
                    f"Last failure: {self._last_failure_time}"
                )

            if self._state == CircuitState.HALF_OPEN:
                if self._half_open_calls >= self.config.half_open_max_calls:
                    self._stats["rejected"] += 1
                    raise CircuitBreakerOpen(
                        f"Circuit '{self.name}' is HALF_OPEN, max calls reached"
                    )
                self._half_open_calls += 1

            self._stats["calls"] += 1

        # Execute outside lock
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except self.config.expected_exception as e:
            self._on_failure()
            raise

    def _update_state(self):
        """Update circuit state based on time and failures."""
        if self._state == CircuitState.OPEN:
            # Check if recovery timeout has passed
            if (
                self._last_failure_time
                and time.time() - self._last_failure_time
                >= self.config.recovery_timeout
            ):
                logger.info(f"Circuit '{self.name}' transitioning to HALF_OPEN")
                self._state = CircuitState.HALF_OPEN
                self._half_open_calls = 0
                self._stats["state_transitions"] += 1

    def _on_success(self):
        """Handle successful call."""
        with self._lock:
            self._stats["successes"] += 1

            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.config.success_threshold:
                    logger.info(f"Circuit '{self.name}' transitioning to CLOSED")
                    self._state = CircuitState.CLOSED
                    self._failure_count = 0
                    self._success_count = 0
                    self._stats["state_transitions"] += 1
            else:
                self._failure_count = 0

    def _on_failure(self):
        """Handle failed call."""
        with self._lock:
            self._stats["failures"] += 1
            self._failure_count += 1
            self._last_failure_time = time.time()

            if self._state == CircuitState.HALF_OPEN:
                logger.warning(
                    f"Circuit '{self.name}' transitioning to OPEN (failure in half-open)"
                )
                self._state = CircuitState.OPEN
                self._stats["state_transitions"] += 1
            elif self._failure_count >= self.config.failure_threshold:
                logger.warning(
                    f"Circuit '{self.name}' transitioning to OPEN ({self._failure_count} failures)"
                )
                self._state = CircuitState.OPEN
                self._stats["state_transitions"] += 1

    @property
    def state(self) -> CircuitState:
        """Get current circuit state."""
        with self._lock:
            return self._state

    def get_stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics."""
        with self._lock:
            return {
                **self._stats,
                "state": self._state.value,
                "failure_count": self._failure_count,
                "success_count": self._success_count,
                "is_open": self._state == CircuitState.OPEN,
            }

    def manual_reset(self):
        """Manually reset circuit to CLOSED state."""
        with self._lock:
            logger.info(f"Circuit '{self.name}' manually reset to CLOSED")
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._success_count = 0
            self._half_open_calls = 0
            self._stats["state_transitions"] += 1


class CircuitBreakerOpen(UnifiedCircuitBreakerOpen):
    """Circuit Breaker Open Exception.

    .. deprecated::
        Use :class:`src.core.exceptions.CircuitBreakerOpen` instead.
        This class will be removed in a future version.

    Exception raised when circuit breaker is open.
    Maintained for backward compatibility - now inherits from unified exception.
    """

    def __init__(self, name: str, last_error: Optional[str] = None):
        _warn_deprecated_circuit_breaker_open()
        super().__init__(name, last_error)


class RetryPolicy:
    """
    Retry mechanism with exponential backoff.

    Example:
        >>> retry = RetryPolicy(RetryConfig(max_attempts=3))
        >>>
        >>> @retry
        ... def unreliable_operation():
        ...     # Might fail transiently
        ...     pass
        >>>
        >>> result = unreliable_operation()  # Auto-retries on failure
    """

    def __init__(self, config: Optional[RetryConfig] = None):
        self.config = config or RetryConfig()

    def __call__(self, func: Callable) -> Callable:
        """Decorator to add retry logic."""

        @wraps(func)
        def wrapper(*args, **kwargs):
            return self.execute(func, *args, **kwargs)

        return wrapper

    def execute(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function with retry logic.

        Args:
            func: Function to retry
            *args, **kwargs: Arguments to function

        Returns:
            Function result

        Raises:
            Exception: If all retries exhausted
        """
        last_exception = None

        for attempt in range(1, self.config.max_attempts + 1):
            try:
                return func(*args, **kwargs)
            except self.config.retryable_exceptions as e:
                last_exception = e

                if attempt < self.config.max_attempts:
                    delay = self._calculate_delay(attempt)
                    logger.warning(
                        f"Attempt {attempt} failed: {e}. Retrying in {delay:.2f}s..."
                    )
                    time.sleep(delay)
                else:
                    logger.error(f"All {self.config.max_attempts} attempts failed")

        raise last_exception

    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay for retry attempt."""
        delay = self.config.base_delay * (self.config.exponential_base ** (attempt - 1))
        delay = min(delay, self.config.max_delay)

        if self.config.jitter:
            # Add random jitter (±25%)
            delay *= 0.75 + 0.5 * random.random()

        return delay


class Bulkhead:
    """
    Bulkhead pattern for resource isolation.

    Limits concurrent operations to prevent resource exhaustion.

    Example:
        >>> bulkhead = Bulkhead(max_concurrent=10, max_queue=20)
        >>>
        >>> @bulkhead
        ... def limited_operation():
        ...     # Resource-intensive operation
        ...     pass
    """

    def __init__(
        self, max_concurrent: int = 10, max_queue: int = 20, queue_timeout: float = 30.0
    ):
        self.max_concurrent = max_concurrent
        self.max_queue = max_queue
        self.queue_timeout = queue_timeout

        self._semaphore = threading.Semaphore(max_concurrent)
        self._queue_lock = threading.Lock()
        self._queued = 0

        self._stats = {
            "executed": 0,
            "queued": 0,
            "rejected": 0,
            "timeouts": 0,
        }

    def __call__(self, func: Callable) -> Callable:
        """Decorator to limit concurrent execution."""

        @wraps(func)
        def wrapper(*args, **kwargs):
            return self.execute(func, *args, **kwargs)

        return wrapper

    def execute(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function with bulkhead protection.

        Args:
            func: Function to execute
            *args, **kwargs: Function arguments

        Returns:
            Function result

        Raises:
            BulkheadFull: If bulkhead is full
            BulkheadTimeout: If queued too long
        """
        # Check queue capacity
        with self._queue_lock:
            if self._queued >= self.max_queue:
                self._stats["rejected"] += 1
                raise BulkheadFull(
                    f"Bulkhead full. Max concurrent: {self.max_concurrent}, "
                    f"Max queue: {self.max_queue}"
                )
            self._queued += 1
            self._stats["queued"] += 1

        try:
            # Wait for semaphore with timeout
            acquired = self._semaphore.acquire(timeout=self.queue_timeout)

            if not acquired:
                self._stats["timeouts"] += 1
                raise BulkheadTimeout(
                    f"Timeout waiting for bulkhead after {self.queue_timeout}s"
                )

            try:
                self._stats["executed"] += 1
                return func(*args, **kwargs)
            finally:
                self._semaphore.release()
        finally:
            with self._queue_lock:
                self._queued -= 1

    def get_stats(self) -> Dict[str, Any]:
        """Get bulkhead statistics."""
        with self._queue_lock:
            return {
                **self._stats,
                "current_queue": self._queued,
                "available_slots": self.max_concurrent
                - (self.max_concurrent - self._semaphore._value),
            }


class BulkheadFull(Exception):
    """Exception raised when bulkhead is full."""

    pass


class BulkheadTimeout(Exception):
    """Exception raised when bulkhead queue times out."""

    pass


class Timeout:
    """
    Timeout wrapper for function execution.

    Note: This uses threading and works for I/O bound operations.
    For CPU-bound, consider using signal-based timeouts on Unix.

    Example:
        >>> timeout = Timeout(seconds=5.0)
        >>>
        >>> @timeout
        ... def slow_operation():
        ...     time.sleep(10)
        >>>
        >>> try:
        ...     slow_operation()
        ... except TimeoutError:
        ...     print("Operation timed out")
    """

    def __init__(self, seconds: float):
        self.seconds = seconds

    def __call__(self, func: Callable) -> Callable:
        """Decorator to add timeout."""

        @wraps(func)
        def wrapper(*args, **kwargs):
            return self.execute(func, *args, **kwargs)

        return wrapper

    def execute(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with timeout."""
        result = [None]
        exception = [None]

        def target():
            try:
                result[0] = func(*args, **kwargs)
            except Exception as e:
                exception[0] = e

        thread = threading.Thread(target=target)
        thread.daemon = True
        thread.start()
        thread.join(timeout=self.seconds)

        if thread.is_alive():
            raise TimeoutError(f"Function timed out after {self.seconds} seconds")

        if exception[0] is not None:
            raise exception[0]

        return result[0]


class ResilientClient:
    """
    Combined resilience client with circuit breaker, retry, and bulkhead.

    Example:
        >>> client = ResilientClient(
        ...     circuit_config=CircuitBreakerConfig(),
        ...     retry_config=RetryConfig(),
        ...     max_concurrent=5
        ... )
        >>>
        >>> @client.protect
        ... def critical_operation():
        ...     # Critical operation
        ...     pass
    """

    def __init__(
        self,
        circuit_config: Optional[CircuitBreakerConfig] = None,
        retry_config: Optional[RetryConfig] = None,
        max_concurrent: int = 10,
    ):
        self.circuit = CircuitBreaker(circuit_config)
        self.retry = RetryPolicy(retry_config)
        self.bulkhead = Bulkhead(max_concurrent=max_concurrent)

    def protect(self, func: Callable) -> Callable:
        """
        Apply all resilience patterns to function.

        Order: Bulkhead -> Circuit Breaker -> Retry -> Function
        """

        @wraps(func)
        def wrapper(*args, **kwargs):
            # Apply in order
            return self.bulkhead.execute(
                lambda: self.circuit.call(
                    lambda: self.retry.execute(func, *args, **kwargs)
                )
            )

        return wrapper

    def get_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get combined statistics."""
        return {
            "circuit_breaker": self.circuit.get_stats(),
            "bulkhead": self.bulkhead.get_stats(),
        }


# Convenience functions
def circuit_breaker(
    failure_threshold: int = 5, recovery_timeout: float = 30.0, name: str = "default"
):
    """Decorator to create circuit breaker."""
    config = CircuitBreakerConfig(
        failure_threshold=failure_threshold, recovery_timeout=recovery_timeout
    )
    breaker = CircuitBreaker(config, name)
    return breaker


def retry(max_attempts: int = 3, base_delay: float = 1.0, max_delay: float = 60.0):
    """Decorator to create retry policy."""
    config = RetryConfig(
        max_attempts=max_attempts, base_delay=base_delay, max_delay=max_delay
    )
    policy = RetryPolicy(config)
    return policy


def timeout(seconds: float):
    """Decorator to add timeout."""
    return Timeout(seconds)


# Example usage
if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)

    print("Resilience Patterns Demo")
    print("=" * 50)

    # Circuit Breaker demo
    print("\n1. Circuit Breaker:")

    breaker = CircuitBreaker(
        CircuitBreakerConfig(failure_threshold=3, recovery_timeout=5.0),
        name="test_service",
    )

    failure_count = 0

    @breaker
    def unstable_service():
        global failure_count
        failure_count += 1
        if failure_count <= 5:
            raise ValueError(f"Simulated failure #{failure_count}")
        return "Success!"

    for i in range(10):
        try:
            result = unstable_service()
            print(f"  Call {i + 1}: {result}")
        except CircuitBreakerOpen as e:
            print(f"  Call {i + 1}: Circuit Open - {e}")
        except Exception as e:
            print(f"  Call {i + 1}: Failed - {e}")

    print(f"\nStats: {breaker.get_stats()}")

    # Retry demo
    print("\n2. Retry with Exponential Backoff:")

    retry_policy = RetryPolicy(RetryConfig(max_attempts=3, base_delay=0.1))

    attempt = 0

    @retry_policy
    def eventually_succeeds():
        global attempt
        attempt += 1
        if attempt < 3:
            raise ValueError(f"Attempt {attempt} failed")
        return f"Succeeded on attempt {attempt}"

    attempt = 0
    result = eventually_succeeds()
    print(f"  Result: {result}")

    print("\n" + "=" * 50)
