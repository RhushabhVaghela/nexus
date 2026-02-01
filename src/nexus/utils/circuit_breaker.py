"""
Circuit Breaker Pattern Implementation for Nexus

Provides fault tolerance for external service calls by preventing
cascading failures and allowing graceful degradation.
"""

import asyncio
import logging
import time
from enum import Enum, auto
from functools import wraps
from typing import Any, Callable, Dict, Optional, Set, TypeVar, Union
from dataclasses import dataclass, field
from contextlib import asynccontextmanager
import threading

logger = logging.getLogger(__name__)

T = TypeVar('T')


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = auto()      # Normal operation - requests pass through
    OPEN = auto()        # Failure threshold exceeded - requests blocked
    HALF_OPEN = auto()   # Testing if service has recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    failure_threshold: int = 5
    recovery_timeout: float = 30.0  # seconds
    half_open_max_calls: int = 3
    success_threshold: int = 2
    excluded_exceptions: Set[type] = field(default_factory=set)
    expected_exception: Optional[type] = None


class CircuitBreakerOpen(Exception):
    """Raised when circuit breaker is open."""
    
    def __init__(self, name: str, last_error: Optional[str] = None):
        self.name = name
        self.last_error = last_error
        super().__init__(f"Circuit breaker '{name}' is OPEN. Last error: {last_error}")


class CircuitBreaker:
    """
    Circuit breaker implementation for external service calls.
    
    States:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Failure threshold exceeded, requests fail fast
    - HALF_OPEN: Testing recovery with limited requests
    """
    
    def __init__(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None,
        on_state_change: Optional[Callable[[CircuitState, CircuitState], None]] = None
    ):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.on_state_change = on_state_change
        
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._half_open_calls = 0
        self._last_failure_time: Optional[float] = None
        self._last_error: Optional[str] = None
        
        self._lock = threading.RLock()
        self._async_lock = asyncio.Lock()
        
        logger.info(f"Circuit breaker '{name}' initialized in CLOSED state")
    
    @property
    def state(self) -> CircuitState:
        """Get current circuit state."""
        with self._lock:
            return self._state
    
    @property
    def metrics(self) -> Dict[str, Any]:
        """Get circuit breaker metrics."""
        with self._lock:
            return {
                "name": self.name,
                "state": self._state.name,
                "failure_count": self._failure_count,
                "success_count": self._success_count,
                "half_open_calls": self._half_open_calls,
                "last_failure_time": self._last_failure_time,
                "last_error": self._last_error,
                "config": {
                    "failure_threshold": self.config.failure_threshold,
                    "recovery_timeout": self.config.recovery_timeout,
                    "half_open_max_calls": self.config.half_open_max_calls,
                    "success_threshold": self.config.success_threshold,
                }
            }
    
    def _transition_to(self, new_state: CircuitState) -> None:
        """Transition to a new state."""
        old_state = self._state
        if old_state != new_state:
            self._state = new_state
            logger.warning(
                f"Circuit breaker '{self.name}' transitioned: "
                f"{old_state.name} -> {new_state.name}"
            )
            
            if self.on_state_change:
                try:
                    self.on_state_change(old_state, new_state)
                except Exception as e:
                    logger.error(f"State change callback error: {e}")
            
            # Reset counters on state change
            if new_state == CircuitState.CLOSED:
                self._failure_count = 0
                self._success_count = 0
                self._half_open_calls = 0
            elif new_state == CircuitState.OPEN:
                self._half_open_calls = 0
                self._success_count = 0
            elif new_state == CircuitState.HALF_OPEN:
                self._failure_count = 0
                self._half_open_calls = 0
    
    def _can_execute(self) -> bool:
        """Check if request can be executed."""
        with self._lock:
            if self._state == CircuitState.CLOSED:
                return True
            
            elif self._state == CircuitState.OPEN:
                # Check if recovery timeout has passed
                if self._last_failure_time is not None:
                    elapsed = time.time() - self._last_failure_time
                    if elapsed >= self.config.recovery_timeout:
                        logger.info(
                            f"Circuit '{self.name}' recovery timeout elapsed, "
                            f"transitioning to HALF_OPEN"
                        )
                        self._transition_to(CircuitState.HALF_OPEN)
                        return True
                return False
            
            elif self._state == CircuitState.HALF_OPEN:
                if self._half_open_calls < self.config.half_open_max_calls:
                    self._half_open_calls += 1
                    return True
                return False
            
            return False
    
    def _on_success(self) -> None:
        """Handle successful execution."""
        with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.config.success_threshold:
                    logger.info(
                        f"Circuit '{self.name}' success threshold reached, "
                        f"transitioning to CLOSED"
                    )
                    self._transition_to(CircuitState.CLOSED)
            else:
                self._failure_count = max(0, self._failure_count - 1)
    
    def _on_failure(self, error: Exception) -> None:
        """Handle failed execution."""
        with self._lock:
            # Check if exception should be excluded
            if type(error) in self.config.excluded_exceptions:
                return
            
            self._failure_count += 1
            self._last_failure_time = time.time()
            self._last_error = str(error)
            
            if self._state == CircuitState.HALF_OPEN:
                logger.warning(
                    f"Circuit '{self.name}' failed in HALF_OPEN, "
                    f"transitioning to OPEN"
                )
                self._transition_to(CircuitState.OPEN)
            elif self._state == CircuitState.CLOSED:
                if self._failure_count >= self.config.failure_threshold:
                    logger.warning(
                        f"Circuit '{self.name}' failure threshold reached, "
                        f"transitioning to OPEN"
                    )
                    self._transition_to(CircuitState.OPEN)
    
    def call(self, func: Callable[..., T], *args, **kwargs) -> T:
        """
        Execute function with circuit breaker protection.
        
        Args:
            func: Function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Function result
            
        Raises:
            CircuitBreakerOpen: If circuit is open
            Exception: Original exception from function
        """
        if not self._can_execute():
            raise CircuitBreakerOpen(self.name, self._last_error)
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure(e)
            raise
    
    async def call_async(self, func: Callable[..., Any], *args, **kwargs) -> Any:
        """
        Execute async function with circuit breaker protection.
        
        Args:
            func: Async function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Function result
            
        Raises:
            CircuitBreakerOpen: If circuit is open
            Exception: Original exception from function
        """
        async with self._async_lock:
            can_execute = self._can_execute()
        
        if not can_execute:
            raise CircuitBreakerOpen(self.name, self._last_error)
        
        try:
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure(e)
            raise
    
    @asynccontextmanager
    async def acquire(self):
        """Async context manager for circuit breaker."""
        async with self._async_lock:
            can_execute = self._can_execute()
        
        if not can_execute:
            raise CircuitBreakerOpen(self.name, self._last_error)
        
        try:
            yield self
            self._on_success()
        except Exception as e:
            self._on_failure(e)
            raise
    
    def __call__(self, func: Callable) -> Callable:
        """Decorator for circuit breaker protection."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            return self.call(func, *args, **kwargs)
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            return await self.call_async(func, *args, **kwargs)
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else wrapper
    
    def force_open(self) -> None:
        """Manually open the circuit."""
        with self._lock:
            self._transition_to(CircuitState.OPEN)
            self._last_failure_time = time.time()
    
    def force_close(self) -> None:
        """Manually close the circuit."""
        with self._lock:
            self._transition_to(CircuitState.CLOSED)
    
    def reset(self) -> None:
        """Reset circuit breaker to initial state."""
        with self._lock:
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._success_count = 0
            self._half_open_calls = 0
            self._last_failure_time = None
            self._last_error = None


class CircuitBreakerRegistry:
    """Registry for managing multiple circuit breakers."""
    
    def __init__(self):
        self._breakers: Dict[str, CircuitBreaker] = {}
        self._lock = threading.RLock()
    
    def register(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None,
        on_state_change: Optional[Callable[[CircuitState, CircuitState], None]] = None
    ) -> CircuitBreaker:
        """Register a new circuit breaker."""
        with self._lock:
            if name in self._breakers:
                raise ValueError(f"Circuit breaker '{name}' already registered")
            
            breaker = CircuitBreaker(name, config, on_state_change)
            self._breakers[name] = breaker
            return breaker
    
    def get(self, name: str) -> Optional[CircuitBreaker]:
        """Get circuit breaker by name."""
        with self._lock:
            return self._breakers.get(name)
    
    def remove(self, name: str) -> bool:
        """Remove circuit breaker from registry."""
        with self._lock:
            if name in self._breakers:
                del self._breakers[name]
                return True
            return False
    
    def get_all_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get metrics for all circuit breakers."""
        with self._lock:
            return {name: breaker.metrics for name, breaker in self._breakers.items()}
    
    def reset_all(self) -> None:
        """Reset all circuit breakers."""
        with self._lock:
            for breaker in self._breakers.values():
                breaker.reset()


# Global registry instance
_global_registry = CircuitBreakerRegistry()


def get_circuit_breaker_registry() -> CircuitBreakerRegistry:
    """Get the global circuit breaker registry."""
    return _global_registry


def circuit_breaker(
    name: str,
    failure_threshold: int = 5,
    recovery_timeout: float = 30.0,
    half_open_max_calls: int = 3,
    success_threshold: int = 2,
    excluded_exceptions: Optional[Set[type]] = None
):
    """
    Decorator for applying circuit breaker protection.
    
    Args:
        name: Circuit breaker name
        failure_threshold: Number of failures before opening
        recovery_timeout: Seconds before attempting recovery
        half_open_max_calls: Max calls in half-open state
        success_threshold: Successes needed to close
        excluded_exceptions: Exception types to ignore
    """
    config = CircuitBreakerConfig(
        failure_threshold=failure_threshold,
        recovery_timeout=recovery_timeout,
        half_open_max_calls=half_open_max_calls,
        success_threshold=success_threshold,
        excluded_exceptions=excluded_exceptions or set()
    )
    
    registry = get_circuit_breaker_registry()
    
    def decorator(func: Callable) -> Callable:
        breaker = registry.get(name)
        if breaker is None:
            breaker = registry.register(name, config)
        
        return breaker(func)
    
    return decorator


# Pre-configured circuit breakers for common services

MODEL_LOADER_CIRCUIT = CircuitBreakerConfig(
    failure_threshold=3,
    recovery_timeout=60.0,
    half_open_max_calls=1,
    success_threshold=1
)

API_CALL_CIRCUIT = CircuitBreakerConfig(
    failure_threshold=5,
    recovery_timeout=30.0,
    half_open_max_calls=2,
    success_threshold=2
)

DATABASE_CIRCUIT = CircuitBreakerConfig(
    failure_threshold=3,
    recovery_timeout=10.0,
    half_open_max_calls=1,
    success_threshold=1
)

EXTERNAL_SERVICE_CIRCUIT = CircuitBreakerConfig(
    failure_threshold=5,
    recovery_timeout=60.0,
    half_open_max_calls=3,
    success_threshold=2
)