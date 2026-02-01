"""
Retry Logic with Exponential Backoff for Nexus

Provides resilient retry mechanisms with exponential backoff, jitter,
and circuit breaker integration for external service calls.
"""

import asyncio
import functools
import logging
import random
import time
from dataclasses import dataclass, field
from typing import (
    Any, Callable, Dict, List, Optional, Set, Type, TypeVar, Union,
    Awaitable
)
from enum import Enum, auto

logger = logging.getLogger(__name__)

T = TypeVar('T')


class BackoffStrategy(Enum):
    """Backoff strategies for retry delays."""
    EXPONENTIAL = auto()      # 2^n * base_delay
    LINEAR = auto()           # n * base_delay
    FIXED = auto()            # constant delay
    FIBONACCI = auto()        # fibonacci(n) * base_delay


@dataclass
class RetryConfig:
    """Configuration for retry behavior.
    
    Attributes:
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay between retries in seconds
        max_delay: Maximum delay between retries in seconds
        backoff_strategy: Strategy for calculating retry delays
        jitter: Whether to add random jitter to delays
        jitter_max: Maximum jitter in seconds
        exponential_base: Base for exponential backoff (default 2)
        retryable_exceptions: Set of exception types that trigger retry
        non_retryable_exceptions: Set of exception types that don't retry
        on_retry: Optional callback called on each retry attempt
        on_success: Optional callback called on successful retry
        on_failure: Optional callback called when all retries exhausted
        timeout: Optional timeout for each attempt in seconds
        circuit_breaker: Optional circuit breaker name to use
    """
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    backoff_strategy: BackoffStrategy = BackoffStrategy.EXPONENTIAL
    jitter: bool = True
    jitter_max: float = 1.0
    exponential_base: float = 2.0
    retryable_exceptions: Set[Type[Exception]] = field(
        default_factory=lambda: {Exception}
    )
    non_retryable_exceptions: Set[Type[Exception]] = field(default_factory=set)
    on_retry: Optional[Callable[[int, Exception, float], None]] = None
    on_success: Optional[Callable[[int, float], None]] = None
    on_failure: Optional[Callable[[Exception], None]] = None
    timeout: Optional[float] = None
    circuit_breaker: Optional[str] = None

    def __post_init__(self):
        """Validate configuration."""
        if self.max_retries < 0:
            raise ValueError("max_retries must be >= 0")
        if self.base_delay < 0:
            raise ValueError("base_delay must be >= 0")
        if self.max_delay < self.base_delay:
            raise ValueError("max_delay must be >= base_delay")
        if self.exponential_base < 1:
            raise ValueError("exponential_base must be >= 1")


class RetryExhausted(Exception):
    """Raised when all retry attempts have been exhausted.
    
    Attributes:
        attempts: Number of attempts made
        last_exception: The last exception that caused failure
        total_time: Total time spent retrying in seconds
    """
    
    def __init__(
        self,
        message: str,
        attempts: int,
        last_exception: Optional[Exception] = None,
        total_time: float = 0.0
    ):
        super().__init__(message)
        self.attempts = attempts
        self.last_exception = last_exception
        self.total_time = total_time
    
    def __str__(self) -> str:
        base_msg = f"RetryExhausted after {self.attempts} attempt(s)"
        if self.last_exception:
            base_msg += f": {self.last_exception}"
        if self.total_time > 0:
            base_msg += f" (total time: {self.total_time:.2f}s)"
        return base_msg


class RetryStats:
    """Statistics for retry operations."""
    
    def __init__(self):
        self.attempts: int = 0
        self.successes: int = 0
        self.failures: int = 0
        self.total_delay: float = 0.0
        self.exceptions: List[Exception] = []
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
    
    @property
    def duration(self) -> float:
        """Get total duration in seconds."""
        if self.start_time is None:
            return 0.0
        end = self.end_time or time.time()
        return end - self.start_time
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert stats to dictionary."""
        return {
            "attempts": self.attempts,
            "successes": self.successes,
            "failures": self.failures,
            "total_delay": self.total_delay,
            "duration": self.duration,
            "exception_types": [type(e).__name__ for e in self.exceptions]
        }


def calculate_delay(
    attempt: int,
    config: RetryConfig
) -> float:
    """Calculate delay for a retry attempt.
    
    Args:
        attempt: Current attempt number (0-indexed)
        config: Retry configuration
        
    Returns:
        Delay in seconds
    """
    if config.backoff_strategy == BackoffStrategy.FIXED:
        delay = config.base_delay
    elif config.backoff_strategy == BackoffStrategy.LINEAR:
        delay = config.base_delay * (attempt + 1)
    elif config.backoff_strategy == BackoffStrategy.EXPONENTIAL:
        delay = config.base_delay * (config.exponential_base ** attempt)
    elif config.backoff_strategy == BackoffStrategy.FIBONACCI:
        # Fibonacci sequence: 1, 1, 2, 3, 5, 8, ...
        a, b = 1, 1
        for _ in range(attempt):
            a, b = b, a + b
        delay = config.base_delay * a
    else:
        delay = config.base_delay
    
    # Apply max delay cap
    delay = min(delay, config.max_delay)
    
    # Add jitter to prevent thundering herd
    if config.jitter:
        jitter = random.uniform(0, config.jitter_max)
        delay += jitter
    
    return delay


def should_retry(
    exception: Exception,
    config: RetryConfig
) -> bool:
    """Determine if an exception should trigger a retry.
    
    Args:
        exception: The exception that was raised
        config: Retry configuration
        
    Returns:
        True if the operation should be retried
    """
    # Check non-retryable exceptions first (higher priority)
    for exc_type in config.non_retryable_exceptions:
        if isinstance(exception, exc_type):
            return False
    
    # Check if exception is in retryable list
    for exc_type in config.retryable_exceptions:
        if isinstance(exception, exc_type):
            return True
    
    return False


def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    backoff_strategy: BackoffStrategy = BackoffStrategy.EXPONENTIAL,
    jitter: bool = True,
    jitter_max: float = 1.0,
    exponential_base: float = 2.0,
    retryable_exceptions: Optional[Set[Type[Exception]]] = None,
    non_retryable_exceptions: Optional[Set[Type[Exception]]] = None,
    on_retry: Optional[Callable[[int, Exception, float], None]] = None,
    on_success: Optional[Callable[[int, float], None]] = None,
    on_failure: Optional[Callable[[Exception], None]] = None,
    timeout: Optional[float] = None,
    circuit_breaker: Optional[str] = None
) -> Callable:
    """Decorator for retry with exponential backoff.
    
    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay between retries in seconds
        max_delay: Maximum delay between retries in seconds
        backoff_strategy: Strategy for calculating retry delays
        jitter: Whether to add random jitter to delays
        jitter_max: Maximum jitter in seconds
        exponential_base: Base for exponential backoff
        retryable_exceptions: Exception types that trigger retry
        non_retryable_exceptions: Exception types that don't retry
        on_retry: Callback called on each retry (attempt, exception, delay)
        on_success: Callback called on successful retry (attempts, total_time)
        on_failure: Callback called when retries exhausted (exception)
        timeout: Timeout for each attempt in seconds
        circuit_breaker: Circuit breaker name to use
        
    Returns:
        Decorated function
        
    Example:
        @retry_with_backoff(max_retries=5, base_delay=2.0)
        def call_api():
            return requests.get("https://api.example.com")
    """
    config = RetryConfig(
        max_retries=max_retries,
        base_delay=base_delay,
        max_delay=max_delay,
        backoff_strategy=backoff_strategy,
        jitter=jitter,
        jitter_max=jitter_max,
        exponential_base=exponential_base,
        retryable_exceptions=retryable_exceptions or {Exception},
        non_retryable_exceptions=non_retryable_exceptions or set(),
        on_retry=on_retry,
        on_success=on_success,
        on_failure=on_failure,
        timeout=timeout,
        circuit_breaker=circuit_breaker
    )
    
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            return _retry_sync(func, config, *args, **kwargs)
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs) -> T:
            return await _retry_async(func, config, *args, **kwargs)
        
        # Attach retry config for introspection
        wrapper.retry_config = config  # type: ignore
        async_wrapper.retry_config = config  # type: ignore
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else wrapper
    
    return decorator


def _retry_sync(
    func: Callable[..., T],
    config: RetryConfig,
    *args,
    **kwargs
) -> T:
    """Execute function with synchronous retry logic.
    
    Args:
        func: Function to execute
        config: Retry configuration
        *args: Function positional arguments
        **kwargs: Function keyword arguments
        
    Returns:
        Function result
        
    Raises:
        RetryExhausted: If all retries are exhausted
        Exception: Original exception if not retryable
    """
    stats = RetryStats()
    stats.start_time = time.time()
    
    # Check circuit breaker if configured
    if config.circuit_breaker:
        try:
            from .circuit_breaker import get_circuit_breaker_registry
            registry = get_circuit_breaker_registry()
            breaker = registry.get(config.circuit_breaker)
            if breaker and breaker.state.name == "OPEN":
                raise RetryExhausted(
                    f"Circuit breaker '{config.circuit_breaker}' is OPEN",
                    attempts=0
                )
        except ImportError:
            pass
    
    last_exception: Optional[Exception] = None
    
    for attempt in range(config.max_retries + 1):
        stats.attempts = attempt + 1
        
        try:
            if config.timeout:
                import signal
                
                def timeout_handler(signum, frame):
                    raise TimeoutError(f"Operation timed out after {config.timeout}s")
                
                old_handler = signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(int(config.timeout))
                
                try:
                    result = func(*args, **kwargs)
                finally:
                    signal.alarm(0)
                    signal.signal(signal.SIGALRM, old_handler)
            else:
                result = func(*args, **kwargs)
            
            stats.successes += 1
            stats.end_time = time.time()
            
            if config.on_success:
                try:
                    config.on_success(attempt + 1, stats.duration)
                except Exception as e:
                    logger.warning(f"on_success callback failed: {e}")
            
            return result
            
        except Exception as e:
            last_exception = e
            stats.exceptions.append(e)
            
            # Check if we should retry
            if not should_retry(e, config):
                logger.debug(f"Exception {type(e).__name__} is not retryable")
                raise
            
            # Check if we've exhausted retries
            if attempt >= config.max_retries:
                stats.failures += 1
                stats.end_time = time.time()
                
                if config.on_failure:
                    try:
                        config.on_failure(e)
                    except Exception as cb_err:
                        logger.warning(f"on_failure callback failed: {cb_err}")
                
                raise RetryExhausted(
                    f"All {config.max_retries + 1} attempts failed",
                    attempts=attempt + 1,
                    last_exception=e,
                    total_time=stats.duration
                ) from e
            
            # Calculate and apply delay
            delay = calculate_delay(attempt, config)
            stats.total_delay += delay
            
            logger.debug(
                f"Retry {attempt + 1}/{config.max_retries} for {func.__name__} "
                f"after {type(e).__name__}: waiting {delay:.2f}s"
            )
            
            if config.on_retry:
                try:
                    config.on_retry(attempt + 1, e, delay)
                except Exception as cb_err:
                    logger.warning(f"on_retry callback failed: {cb_err}")
            
            time.sleep(delay)
    
    # This should never be reached
    raise RetryExhausted(
        "Unexpected end of retry loop",
        attempts=stats.attempts,
        last_exception=last_exception,
        total_time=stats.duration
    )


async def _retry_async(
    func: Callable[..., Awaitable[T]],
    config: RetryConfig,
    *args,
    **kwargs
) -> T:
    """Execute async function with retry logic.
    
    Args:
        func: Async function to execute
        config: Retry configuration
        *args: Function positional arguments
        **kwargs: Function keyword arguments
        
    Returns:
        Function result
        
    Raises:
        RetryExhausted: If all retries are exhausted
        Exception: Original exception if not retryable
    """
    stats = RetryStats()
    stats.start_time = time.time()
    
    # Check circuit breaker if configured
    if config.circuit_breaker:
        try:
            from .circuit_breaker import get_circuit_breaker_registry
            registry = get_circuit_breaker_registry()
            breaker = registry.get(config.circuit_breaker)
            if breaker and breaker.state.name == "OPEN":
                raise RetryExhausted(
                    f"Circuit breaker '{config.circuit_breaker}' is OPEN",
                    attempts=0
                )
        except ImportError:
            pass
    
    last_exception: Optional[Exception] = None
    
    for attempt in range(config.max_retries + 1):
        stats.attempts = attempt + 1
        
        try:
            if config.timeout:
                result = await asyncio.wait_for(
                    func(*args, **kwargs),
                    timeout=config.timeout
                )
            else:
                result = await func(*args, **kwargs)
            
            stats.successes += 1
            stats.end_time = time.time()
            
            if config.on_success:
                try:
                    config.on_success(attempt + 1, stats.duration)
                except Exception as e:
                    logger.warning(f"on_success callback failed: {e}")
            
            return result
            
        except Exception as e:
            last_exception = e
            stats.exceptions.append(e)
            
            # Check if we should retry
            if not should_retry(e, config):
                logger.debug(f"Exception {type(e).__name__} is not retryable")
                raise
            
            # Check if we've exhausted retries
            if attempt >= config.max_retries:
                stats.failures += 1
                stats.end_time = time.time()
                
                if config.on_failure:
                    try:
                        config.on_failure(e)
                    except Exception as cb_err:
                        logger.warning(f"on_failure callback failed: {cb_err}")
                
                raise RetryExhausted(
                    f"All {config.max_retries + 1} attempts failed",
                    attempts=attempt + 1,
                    last_exception=e,
                    total_time=stats.duration
                ) from e
            
            # Calculate and apply delay
            delay = calculate_delay(attempt, config)
            stats.total_delay += delay
            
            logger.debug(
                f"Retry {attempt + 1}/{config.max_retries} for {func.__name__} "
                f"after {type(e).__name__}: waiting {delay:.2f}s"
            )
            
            if config.on_retry:
                try:
                    config.on_retry(attempt + 1, e, delay)
                except Exception as cb_err:
                    logger.warning(f"on_retry callback failed: {cb_err}")
            
            await asyncio.sleep(delay)
    
    # This should never be reached
    raise RetryExhausted(
        "Unexpected end of retry loop",
        attempts=stats.attempts,
        last_exception=last_exception,
        total_time=stats.duration
    )


class Retryable:
    """Class-based retry wrapper for more control."""
    
    def __init__(self, config: Optional[RetryConfig] = None, **kwargs):
        """Initialize with configuration.
        
        Args:
            config: RetryConfig instance, or
            **kwargs: Keyword arguments to create RetryConfig
        """
        if config is None:
            config = RetryConfig(**kwargs)
        self.config = config
        self.stats = RetryStats()
    
    def __call__(self, func: Callable[..., T]) -> Callable[..., T]:
        """Decorate a function.
        
        Args:
            func: Function to wrap
            
        Returns:
            Wrapped function with retry logic
        """
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            self.stats = RetryStats()
            return _retry_sync(func, self.config, *args, **kwargs)
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs) -> T:
            self.stats = RetryStats()
            return await _retry_async(func, self.config, *args, **kwargs)
        
        wrapper.retryable = self  # type: ignore
        async_wrapper.retryable = self  # type: ignore
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else wrapper
    
    def get_stats(self) -> RetryStats:
        """Get current retry statistics."""
        return self.stats


def retry_call(
    func: Callable[..., T],
    *args,
    max_retries: int = 3,
    base_delay: float = 1.0,
    retryable_exceptions: Optional[Set[Type[Exception]]] = None,
    **kwargs
) -> T:
    """Call a function with retry logic (imperative style).
    
    Args:
        func: Function to call
        *args: Positional arguments for function
        max_retries: Maximum retry attempts
        base_delay: Base delay between retries
        retryable_exceptions: Exceptions that trigger retry
        **kwargs: Keyword arguments for function
        
    Returns:
        Function result
        
    Raises:
        RetryExhausted: If all retries exhausted
    """
    config = RetryConfig(
        max_retries=max_retries,
        base_delay=base_delay,
        retryable_exceptions=retryable_exceptions or {Exception}
    )
    return _retry_sync(func, config, *args, **kwargs)


async def retry_call_async(
    func: Callable[..., Awaitable[T]],
    *args,
    max_retries: int = 3,
    base_delay: float = 1.0,
    retryable_exceptions: Optional[Set[Type[Exception]]] = None,
    **kwargs
) -> T:
    """Call an async function with retry logic (imperative style).
    
    Args:
        func: Async function to call
        *args: Positional arguments for function
        max_retries: Maximum retry attempts
        base_delay: Base delay between retries
        retryable_exceptions: Exceptions that trigger retry
        **kwargs: Keyword arguments for function
        
    Returns:
        Function result
        
    Raises:
        RetryExhausted: If all retries exhausted
    """
    config = RetryConfig(
        max_retries=max_retries,
        base_delay=base_delay,
        retryable_exceptions=retryable_exceptions or {Exception}
    )
    return await _retry_async(func, config, *args, **kwargs)


# Pre-configured retry decorators for common scenarios

network_retry = retry_with_backoff(
    max_retries=5,
    base_delay=1.0,
    max_delay=30.0,
    retryable_exceptions={
        ConnectionError,
        TimeoutError,
        OSError,
        IOError
    }
)
"""Retry decorator optimized for network operations."""

database_retry = retry_with_backoff(
    max_retries=3,
    base_delay=0.5,
    max_delay=10.0,
    retryable_exceptions={
        ConnectionError,
        TimeoutError
    }
)
"""Retry decorator optimized for database operations."""

api_retry = retry_with_backoff(
    max_retries=3,
    base_delay=2.0,
    max_delay=60.0,
    retryable_exceptions={
        ConnectionError,
        TimeoutError,
        Exception  # For HTTP status errors
    },
    non_retryable_exceptions={
        ValueError,
        TypeError,
        KeyError
    }
)
"""Retry decorator optimized for API calls."""


def with_retry(
    exceptions: Optional[Set[Type[Exception]]] = None,
    retries: int = 3,
    delay: float = 1.0
):
    """Simple retry decorator with sensible defaults.
    
    Args:
        exceptions: Exception types to retry on
        retries: Number of retry attempts
        delay: Base delay between retries
        
    Returns:
        Decorated function
        
    Example:
        @with_retry(retries=5, delay=2.0)
        def fetch_data():
            return api.get_data()
    """
    return retry_with_backoff(
        max_retries=retries,
        base_delay=delay,
        retryable_exceptions=exceptions or {Exception}
    )


# Export all public members
__all__ = [
    'BackoffStrategy',
    'RetryConfig',
    'RetryExhausted',
    'RetryStats',
    'Retryable',
    'calculate_delay',
    'retry_with_backoff',
    'retry_call',
    'retry_call_async',
    'should_retry',
    'network_retry',
    'database_retry',
    'api_retry',
    'with_retry',
]