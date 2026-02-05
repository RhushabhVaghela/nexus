"""
tests/unit/test_circuit_breaker.py
Comprehensive tests for circuit breaker functionality.

Tests cover:
- Circuit state transitions (CLOSED -> OPEN -> HALF_OPEN -> CLOSED)
- Failure threshold handling
- Recovery timeout
- Success threshold
- Half-open max calls
- Excluded exceptions
- Decorator functionality
"""

import pytest
import time
import asyncio
from unittest.mock import Mock, MagicMock, patch

from src.utils.circuit_breaker import (
    CircuitState,
    CircuitBreakerConfig,
    CircuitBreakerOpen,
    CircuitBreaker,
    CircuitBreakerRegistry,
    get_circuit_breaker_registry,
    circuit_breaker,
    MODEL_LOADER_CIRCUIT,
    API_CALL_CIRCUIT,
    DATABASE_CIRCUIT,
    EXTERNAL_SERVICE_CIRCUIT,
)


class TestCircuitState:
    """Test CircuitState enum."""
    
    def test_closed_state(self):
        """Test CLOSED state."""
        assert CircuitState.CLOSED.name == "CLOSED"
    
    def test_open_state(self):
        """Test OPEN state."""
        assert CircuitState.OPEN.name == "OPEN"
    
    def test_half_open_state(self):
        """Test HALF_OPEN state."""
        assert CircuitState.HALF_OPEN.name == "HALF_OPEN"


class TestCircuitBreakerConfig:
    """Test CircuitBreakerConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = CircuitBreakerConfig()
        
        assert config.failure_threshold == 5
        assert config.recovery_timeout == 30.0
        assert config.half_open_max_calls == 3
        assert config.success_threshold == 2
        assert config.excluded_exceptions == set()
        assert config.expected_exception is None
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = CircuitBreakerConfig(
            failure_threshold=3,
            recovery_timeout=10.0,
            half_open_max_calls=1,
            success_threshold=1,
            excluded_exceptions={ValueError}
        )
        
        assert config.failure_threshold == 3
        assert config.recovery_timeout == 10.0
        assert config.half_open_max_calls == 1
        assert config.success_threshold == 1
        assert ValueError in config.excluded_exceptions


class TestCircuitBreakerOpen:
    """Test CircuitBreakerOpen exception."""
    
    def test_exception_creation(self):
        """Test creating exception."""
        exc = CircuitBreakerOpen("test_circuit", "last error message")
        
        assert exc.name == "test_circuit"
        assert exc.last_error == "last error message"
        assert "test_circuit" in str(exc)
        assert "last error message" in str(exc)


class TestCircuitBreaker:
    """Test CircuitBreaker class."""
    
    def test_initial_state(self):
        """Test initial state is CLOSED."""
        cb = CircuitBreaker("test")
        
        assert cb.state == CircuitState.CLOSED
    
    def test_successful_call(self):
        """Test successful function call."""
        cb = CircuitBreaker("test")
        
        def success_func():
            return "success"
        
        result = cb.call(success_func)
        assert result == "success"
    
    def test_failed_call(self):
        """Test failed function call."""
        cb = CircuitBreaker("test")
        
        def fail_func():
            raise ValueError("Test error")
        
        with pytest.raises(ValueError):
            cb.call(fail_func)
    
    def test_failure_count_increment(self):
        """Test failure count increments."""
        cb = CircuitBreaker("test", config=CircuitBreakerConfig(failure_threshold=5))
        
        def fail_func():
            raise ValueError("Test error")
        
        for _ in range(3):
            try:
                cb.call(fail_func)
            except ValueError:
                pass
        
        assert cb.metrics["failure_count"] == 3
    
    def test_opens_after_threshold(self):
        """Test circuit opens after failure threshold."""
        cb = CircuitBreaker("test", config=CircuitBreakerConfig(failure_threshold=3))
        
        def fail_func():
            raise ValueError("Test error")
        
        # Trigger failures to open circuit
        for _ in range(3):
            try:
                cb.call(fail_func)
            except ValueError:
                pass
        
        assert cb.state == CircuitState.OPEN
    
    def test_blocks_when_open(self):
        """Test calls are blocked when circuit is open."""
        cb = CircuitBreaker("test", config=CircuitBreakerConfig(failure_threshold=1))
        
        def fail_func():
            raise ValueError("Test error")
        
        # Open the circuit
        try:
            cb.call(fail_func)
        except ValueError:
            pass
        
        # Next call should be blocked
        def any_func():
            return "should not execute"
        
        with pytest.raises(CircuitBreakerOpen):
            cb.call(any_func)
    
    def test_half_open_after_timeout(self):
        """Test circuit transitions to half-open after timeout."""
        config = CircuitBreakerConfig(failure_threshold=1, recovery_timeout=0.1)
        cb = CircuitBreaker("test", config=config)
        
        # Open the circuit
        try:
            cb.call(lambda: (_ for _ in ()).throw(ValueError("error")))
        except (ValueError, CircuitBreakerOpen):
            pass
        
        assert cb.state == CircuitState.OPEN
        
        # Wait for recovery timeout
        time.sleep(0.15)
        
        # Next call should transition to half-open
        def success_func():
            return "success"
        
        # This call should work and transition to half-open
        result = cb.call(success_func)
        assert result == "success"
        assert cb.state == CircuitState.HALF_OPEN
    
    def test_closes_after_success_threshold(self):
        """Test circuit closes after success threshold in half-open."""
        config = CircuitBreakerConfig(
            failure_threshold=1,
            recovery_timeout=0.1,
            success_threshold=2,
            half_open_max_calls=5
        )
        cb = CircuitBreaker("test", config=config)
        
        # Open the circuit
        try:
            cb.call(lambda: (_ for _ in ()).throw(ValueError("error")))
        except (ValueError, CircuitBreakerOpen):
            pass
        
        # Wait for recovery timeout
        time.sleep(0.15)
        
        # Make successful calls to reach success threshold
        for _ in range(2):
            cb.call(lambda: "success")
        
        assert cb.state == CircuitState.CLOSED
    
    def test_reopens_on_failure_in_half_open(self):
        """Test circuit reopens on failure in half-open state."""
        config = CircuitBreakerConfig(
            failure_threshold=1,
            recovery_timeout=0.1,
            success_threshold=2
        )
        cb = CircuitBreaker("test", config=config)
        
        # Open the circuit
        try:
            cb.call(lambda: (_ for _ in ()).throw(ValueError("error")))
        except (ValueError, CircuitBreakerOpen):
            pass
        
        # Wait for recovery timeout
        time.sleep(0.15)
        
        # First call to transition to half-open (success)
        cb.call(lambda: "success")
        assert cb.state == CircuitState.HALF_OPEN
        
        # Failure should reopen
        try:
            cb.call(lambda: (_ for _ in ()).throw(ValueError("error")))
        except ValueError:
            pass
        
        assert cb.state == CircuitState.OPEN
    
    def test_excluded_exceptions(self):
        """Test excluded exceptions don't count as failures."""
        config = CircuitBreakerConfig(
            failure_threshold=3,
            excluded_exceptions={ValueError}
        )
        cb = CircuitBreaker("test", config=config)
        
        # These should not count as failures
        for _ in range(5):
            try:
                cb.call(lambda: (_ for _ in ()).throw(ValueError("error")))
            except ValueError:
                pass
        
        # Circuit should still be closed
        assert cb.state == CircuitState.CLOSED
    
    def test_force_open(self):
        """Test manually forcing circuit open."""
        cb = CircuitBreaker("test")
        
        cb.force_open()
        
        assert cb.state == CircuitState.OPEN
        assert cb.metrics["state"] == "OPEN"
    
    def test_force_close(self):
        """Test manually forcing circuit closed."""
        cb = CircuitBreaker("test", config=CircuitBreakerConfig(failure_threshold=1))
        
        # Open the circuit
        try:
            cb.call(lambda: (_ for _ in ()).throw(ValueError("error")))
        except (ValueError, CircuitBreakerOpen):
            pass
        
        cb.force_close()
        
        assert cb.state == CircuitState.CLOSED
    
    def test_reset(self):
        """Test resetting circuit breaker."""
        cb = CircuitBreaker("test", config=CircuitBreakerConfig(failure_threshold=1))
        
        # Open the circuit
        try:
            cb.call(lambda: (_ for _ in ()).throw(ValueError("error")))
        except (ValueError, CircuitBreakerOpen):
            pass
        
        cb.reset()
        
        assert cb.state == CircuitState.CLOSED
        assert cb.metrics["failure_count"] == 0
        assert cb.metrics["success_count"] == 0
    
    def test_metrics(self):
        """Test getting metrics."""
        cb = CircuitBreaker("test")
        
        metrics = cb.metrics
        
        assert "name" in metrics
        assert "state" in metrics
        assert "failure_count" in metrics
        assert "success_count" in metrics
        assert "config" in metrics
    
    def test_call_with_args(self):
        """Test calling with arguments."""
        cb = CircuitBreaker("test")
        
        def func_with_args(a, b, c=None):
            return f"{a}-{b}-{c}"
        
        result = cb.call(func_with_args, 1, 2, c=3)
        assert result == "1-2-3"


class TestCircuitBreakerAsync:
    """Test async circuit breaker functionality."""
    
    @pytest.mark.asyncio
    async def test_async_successful_call(self):
        """Test successful async function call."""
        cb = CircuitBreaker("test")
        
        async def async_success():
            return "success"
        
        result = await cb.call_async(async_success)
        assert result == "success"
    
    @pytest.mark.asyncio
    async def test_async_failed_call(self):
        """Test failed async function call."""
        cb = CircuitBreaker("test")
        
        async def async_fail():
            raise ValueError("Test error")
        
        with pytest.raises(ValueError):
            await cb.call_async(async_fail)
    
    @pytest.mark.asyncio
    async def test_async_blocks_when_open(self):
        """Test async calls are blocked when circuit is open."""
        cb = CircuitBreaker("test", config=CircuitBreakerConfig(failure_threshold=1))
        
        # Open the circuit
        try:
            await cb.call_async(lambda: (_ for _ in ()).throw(ValueError("error")))
        except (ValueError, CircuitBreakerOpen, TypeError):
            pass
        
        async def any_func():
            return "should not execute"
        
        with pytest.raises(CircuitBreakerOpen):
            await cb.call_async(any_func)
    
    @pytest.mark.asyncio
    async def test_async_context_manager(self):
        """Test async context manager."""
        cb = CircuitBreaker("test")
        
        async with cb.acquire():
            pass  # Successful operation
        
        # Circuit should still be closed
        assert cb.state == CircuitState.CLOSED


class TestCircuitBreakerAsDecorator:
    """Test circuit breaker as decorator."""
    
    def test_decorator_sync_function(self):
        """Test decorating sync function."""
        cb = CircuitBreaker("test")
        
        @cb
        def decorated_func():
            return "decorated"
        
        result = decorated_func()
        assert result == "decorated"
    
    @pytest.mark.asyncio
    async def test_decorator_async_function(self):
        """Test decorating async function."""
        cb = CircuitBreaker("test")
        
        @cb
        async def async_decorated():
            return "async decorated"
        
        result = await async_decorated()
        assert result == "async decorated"


class TestCircuitBreakerRegistry:
    """Test CircuitBreakerRegistry class."""
    
    def test_register(self):
        """Test registering a circuit breaker."""
        registry = CircuitBreakerRegistry()
        cb = registry.register("test_cb")
        
        assert cb.name == "test_cb"
        assert registry.get("test_cb") is cb
    
    def test_register_duplicate(self):
        """Test registering duplicate name raises error."""
        registry = CircuitBreakerRegistry()
        registry.register("test_cb")
        
        with pytest.raises(ValueError):
            registry.register("test_cb")
    
    def test_get_missing(self):
        """Test getting non-existent circuit breaker."""
        registry = CircuitBreakerRegistry()
        
        assert registry.get("missing") is None
    
    def test_remove(self):
        """Test removing circuit breaker."""
        registry = CircuitBreakerRegistry()
        registry.register("test_cb")
        
        assert registry.remove("test_cb") is True
        assert registry.get("test_cb") is None
        assert registry.remove("test_cb") is False
    
    def test_get_all_metrics(self):
        """Test getting metrics for all circuit breakers."""
        registry = CircuitBreakerRegistry()
        registry.register("cb1")
        registry.register("cb2")
        
        metrics = registry.get_all_metrics()
        
        assert "cb1" in metrics
        assert "cb2" in metrics
    
    def test_reset_all(self):
        """Test resetting all circuit breakers."""
        registry = CircuitBreakerRegistry()
        cb = registry.register("cb1", config=CircuitBreakerConfig(failure_threshold=1))
        
        # Open the circuit
        try:
            cb.call(lambda: (_ for _ in ()).throw(ValueError("error")))
        except (ValueError, CircuitBreakerOpen):
            pass
        
        registry.reset_all()
        
        assert cb.state == CircuitState.CLOSED


class TestGlobalRegistry:
    """Test global registry functions."""
    
    def test_get_circuit_breaker_registry_singleton(self):
        """Test get_circuit_breaker_registry returns singleton."""
        registry1 = get_circuit_breaker_registry()
        registry2 = get_circuit_breaker_registry()
        
        assert registry1 is registry2


class TestCircuitBreakerDecorator:
    """Test circuit_breaker decorator function."""
    
    def test_decorator_with_registry(self):
        """Test decorator using global registry."""
        registry = get_circuit_breaker_registry()
        
        @circuit_breaker("test_decorator", failure_threshold=3)
        def my_function():
            return "success"
        
        result = my_function()
        assert result == "success"
        
        # Should be registered
        assert registry.get("test_decorator") is not None


class TestPredefinedConfigs:
    """Test predefined circuit breaker configurations."""
    
    def test_model_loader_circuit(self):
        """Test MODEL_LOADER_CIRCUIT config."""
        assert MODEL_LOADER_CIRCUIT.failure_threshold == 3
        assert MODEL_LOADER_CIRCUIT.recovery_timeout == 60.0
    
    def test_api_call_circuit(self):
        """Test API_CALL_CIRCUIT config."""
        assert API_CALL_CIRCUIT.failure_threshold == 5
        assert API_CALL_CIRCUIT.recovery_timeout == 30.0
    
    def test_database_circuit(self):
        """Test DATABASE_CIRCUIT config."""
        assert DATABASE_CIRCUIT.failure_threshold == 3
        assert DATABASE_CIRCUIT.recovery_timeout == 10.0
    
    def test_external_service_circuit(self):
        """Test EXTERNAL_SERVICE_CIRCUIT config."""
        assert EXTERNAL_SERVICE_CIRCUIT.failure_threshold == 5
        assert EXTERNAL_SERVICE_CIRCUIT.recovery_timeout == 60.0


class TestEdgeCases:
    """Test edge cases."""
    
    def test_half_open_max_calls(self):
        """Test half-open max calls limit."""
        config = CircuitBreakerConfig(
            failure_threshold=1,
            recovery_timeout=0.1,
            half_open_max_calls=2
        )
        cb = CircuitBreaker("test", config=config)
        
        # Open the circuit
        try:
            cb.call(lambda: (_ for _ in ()).throw(ValueError("error")))
        except (ValueError, CircuitBreakerOpen):
            pass
        
        # Wait for timeout
        time.sleep(0.15)
        
        # Use up all half-open calls
        cb._transition_to(CircuitState.HALF_OPEN)
        cb._half_open_calls = 2
        
        # Next call should be blocked
        with pytest.raises(CircuitBreakerOpen):
            cb.call(lambda: "success")
    
    def test_success_decrements_failure_count(self):
        """Test success decrements failure count in closed state."""
        config = CircuitBreakerConfig(failure_threshold=5)
        cb = CircuitBreaker("test", config=config)
        
        # Add some failures (but not enough to open)
        for _ in range(3):
            try:
                cb.call(lambda: (_ for _ in ()).throw(ValueError("error")))
            except ValueError:
                pass
        
        assert cb.metrics["failure_count"] == 3
        
        # Success should decrement
        cb.call(lambda: "success")
        
        assert cb.metrics["failure_count"] == 2
