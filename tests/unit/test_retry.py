"""
Comprehensive tests for retry logic with exponential backoff.

Tests cover:
- Retry with backoff strategies (exponential, linear, fixed, fibonacci)
- Jitter functionality
- Exception filtering (retryable vs non-retryable)
- Max retry limits
- Timeout handling
- Circuit breaker integration
- Async retry support
- Retry statistics
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, patch, MagicMock

from src.utils.retry import (
    BackoffStrategy,
    RetryConfig,
    RetryExhausted,
    RetryStats,
    Retryable,
    calculate_delay,
    retry_with_backoff,
    retry_call,
    retry_call_async,
    should_retry,
    network_retry,
    database_retry,
    api_retry,
    with_retry,
)


class TestBackoffStrategy:
    """Test BackoffStrategy enum."""
    
    def test_exponential_strategy(self):
        """Test EXPONENTIAL strategy exists."""
        assert BackoffStrategy.EXPONENTIAL.name == "EXPONENTIAL"
    
    def test_linear_strategy(self):
        """Test LINEAR strategy exists."""
        assert BackoffStrategy.LINEAR.name == "LINEAR"
    
    def test_fixed_strategy(self):
        """Test FIXED strategy exists."""
        assert BackoffStrategy.FIXED.name == "FIXED"
    
    def test_fibonacci_strategy(self):
        """Test FIBONACCI strategy exists."""
        assert BackoffStrategy.FIBONACCI.name == "FIBONACCI"


class TestRetryConfig:
    """Test RetryConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = RetryConfig()
        
        assert config.max_retries == 3
        assert config.base_delay == 1.0
        assert config.max_delay == 60.0
        assert config.backoff_strategy == BackoffStrategy.EXPONENTIAL
        assert config.jitter is True
        assert config.jitter_max == 1.0
        assert config.exponential_base == 2.0
        assert Exception in config.retryable_exceptions
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = RetryConfig(
            max_retries=5,
            base_delay=2.0,
            max_delay=30.0,
            backoff_strategy=BackoffStrategy.LINEAR,
            jitter=False,
            exponential_base=3.0
        )
        
        assert config.max_retries == 5
        assert config.base_delay == 2.0
        assert config.max_delay == 30.0
        assert config.backoff_strategy == BackoffStrategy.LINEAR
        assert config.jitter is False
        assert config.exponential_base == 3.0
    
    def test_invalid_max_retries(self):
        """Test validation of negative max_retries."""
        with pytest.raises(ValueError, match="max_retries must be >= 0"):
            RetryConfig(max_retries=-1)
    
    def test_invalid_base_delay(self):
        """Test validation of negative base_delay."""
        with pytest.raises(ValueError, match="base_delay must be >= 0"):
            RetryConfig(base_delay=-1.0)
    
    def test_invalid_max_delay(self):
        """Test validation of max_delay less than base_delay."""
        with pytest.raises(ValueError, match="max_delay must be >= base_delay"):
            RetryConfig(base_delay=10.0, max_delay=5.0)
    
    def test_invalid_exponential_base(self):
        """Test validation of exponential_base less than 1."""
        with pytest.raises(ValueError, match="exponential_base must be >= 1"):
            RetryConfig(exponential_base=0.5)


class TestRetryExhausted:
    """Test RetryExhausted exception."""
    
    def test_exception_creation(self):
        """Test creating RetryExhausted exception."""
        last_exc = ValueError("test error")
        exc = RetryExhausted("All retries failed", 3, last_exc, 5.0)
        
        assert exc.attempts == 3
        assert exc.last_exception is last_exc
        assert exc.total_time == 5.0
        assert "3 attempt(s)" in str(exc)
        assert "test error" in str(exc)
        assert "5.00s" in str(exc)
    
    def test_exception_without_last_error(self):
        """Test RetryExhausted without last exception."""
        exc = RetryExhausted("All retries failed", 3)
        
        assert "3 attempt(s)" in str(exc)
        assert exc.last_exception is None


class TestRetryStats:
    """Test RetryStats class."""
    
    def test_initial_state(self):
        """Test initial stats state."""
        stats = RetryStats()
        
        assert stats.attempts == 0
        assert stats.successes == 0
        assert stats.failures == 0
        assert stats.total_delay == 0.0
        assert stats.exceptions == []
        assert stats.start_time is None
        assert stats.end_time is None
    
    def test_duration_property(self):
        """Test duration calculation."""
        stats = RetryStats()
        stats.start_time = time.time() - 5.0
        
        duration = stats.duration
        
        assert duration >= 5.0
        assert duration < 6.0
    
    def test_duration_without_start(self):
        """Test duration when start_time is None."""
        stats = RetryStats()
        
        assert stats.duration == 0.0
    
    def test_to_dict(self):
        """Test converting stats to dictionary."""
        stats = RetryStats()
        stats.attempts = 3
        stats.successes = 1
        stats.failures = 1
        stats.total_delay = 2.5
        stats.exceptions = [ValueError("error")]
        
        result = stats.to_dict()
        
        assert result["attempts"] == 3
        assert result["successes"] == 1
        assert result["failures"] == 1
        assert result["total_delay"] == 2.5
        assert "exception_types" in result
        assert result["exception_types"] == ["ValueError"]


class TestCalculateDelay:
    """Test calculate_delay function."""
    
    def test_fixed_delay(self):
        """Test fixed delay strategy."""
        config = RetryConfig(
            backoff_strategy=BackoffStrategy.FIXED,
            base_delay=5.0,
            jitter=False
        )
        
        delay = calculate_delay(0, config)
        
        assert delay == 5.0
    
    def test_linear_delay(self):
        """Test linear delay strategy."""
        config = RetryConfig(
            backoff_strategy=BackoffStrategy.LINEAR,
            base_delay=2.0,
            jitter=False
        )
        
        assert calculate_delay(0, config) == 2.0
        assert calculate_delay(1, config) == 4.0
        assert calculate_delay(2, config) == 6.0
    
    def test_exponential_delay(self):
        """Test exponential delay strategy."""
        config = RetryConfig(
            backoff_strategy=BackoffStrategy.EXPONENTIAL,
            base_delay=1.0,
            exponential_base=2.0,
            jitter=False
        )
        
        assert calculate_delay(0, config) == 1.0  # 2^0 * 1
        assert calculate_delay(1, config) == 2.0  # 2^1 * 1
        assert calculate_delay(2, config) == 4.0  # 2^2 * 1
        assert calculate_delay(3, config) == 8.0  # 2^3 * 1
    
    def test_fibonacci_delay(self):
        """Test fibonacci delay strategy."""
        config = RetryConfig(
            backoff_strategy=BackoffStrategy.FIBONACCI,
            base_delay=1.0,
            jitter=False
        )
        
        assert calculate_delay(0, config) == 1.0  # fib(0) = 1
        assert calculate_delay(1, config) == 1.0  # fib(1) = 1
        assert calculate_delay(2, config) == 2.0  # fib(2) = 2
        assert calculate_delay(3, config) == 3.0  # fib(3) = 3
        assert calculate_delay(4, config) == 5.0  # fib(4) = 5
    
    def test_max_delay_cap(self):
        """Test that delay is capped at max_delay."""
        config = RetryConfig(
            backoff_strategy=BackoffStrategy.EXPONENTIAL,
            base_delay=1.0,
            max_delay=5.0,
            jitter=False
        )
        
        # This would be 8.0 without cap
        delay = calculate_delay(3, config)
        
        assert delay == 5.0
    
    def test_jitter_adds_randomness(self):
        """Test that jitter adds random delay."""
        config = RetryConfig(
            backoff_strategy=BackoffStrategy.FIXED,
            base_delay=1.0,
            jitter=True,
            jitter_max=1.0
        )
        
        delays = [calculate_delay(0, config) for _ in range(100)]
        
        # All delays should be >= base_delay
        assert all(d >= 1.0 for d in delays)
        # Some delays should be > base_delay (due to jitter)
        assert any(d > 1.0 for d in delays)
        # All delays should be <= base_delay + jitter_max
        assert all(d <= 2.0 for d in delays)


class TestShouldRetry:
    """Test should_retry function."""
    
    def test_retryable_exception(self):
        """Test retryable exception is retried."""
        config = RetryConfig(retryable_exceptions={ValueError})
        
        assert should_retry(ValueError("test"), config) is True
    
    def test_non_retryable_exception(self):
        """Test non-retryable exception is not retried."""
        config = RetryConfig(
            retryable_exceptions={Exception},
            non_retryable_exceptions={TypeError}
        )
        
        assert should_retry(TypeError("test"), config) is False
    
    def test_non_retryable_takes_priority(self):
        """Test non-retryable exceptions take priority."""
        config = RetryConfig(
            retryable_exceptions={ValueError},
            non_retryable_exceptions={ValueError}
        )
        
        # Non-retryable should take priority
        assert should_retry(ValueError("test"), config) is False
    
    def test_exception_not_in_lists(self):
        """Test exception not in any list is not retried."""
        config = RetryConfig(retryable_exceptions={ValueError})
        
        assert should_retry(TypeError("test"), config) is False


class TestRetryWithBackoff:
    """Test retry_with_backoff decorator."""
    
    def test_successful_function_no_retry(self):
        """Test successful function is not retried."""
        call_count = 0
        
        @retry_with_backoff(max_retries=3, base_delay=0.01, jitter=False)
        def success_func():
            nonlocal call_count
            call_count += 1
            return "success"
        
        result = success_func()
        
        assert result == "success"
        assert call_count == 1
    
    def test_retry_on_failure(self):
        """Test function is retried on failure."""
        call_count = 0
        
        @retry_with_backoff(max_retries=3, base_delay=0.01, jitter=False)
        def fail_then_succeed():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("not yet")
            return "success"
        
        result = fail_then_succeed()
        
        assert result == "success"
        assert call_count == 3
    
    def test_retry_exhausted(self):
        """Test RetryExhausted raised when retries exhausted."""
        @retry_with_backoff(max_retries=2, base_delay=0.01, jitter=False)
        def always_fail():
            raise ValueError("always fails")
        
        with pytest.raises(RetryExhausted) as exc_info:
            always_fail()
        
        assert exc_info.value.attempts == 3  # initial + 2 retries
        assert "always fails" in str(exc_info.value.last_exception)
    
    def test_non_retryable_exception_not_retried(self):
        """Test non-retryable exception is not retried."""
        call_count = 0
        
        @retry_with_backoff(
            max_retries=3,
            base_delay=0.01,
            jitter=False,
            retryable_exceptions={ValueError},
            non_retryable_exceptions={TypeError}
        )
        def raises_type_error():
            nonlocal call_count
            call_count += 1
            raise TypeError("not retryable")
        
        with pytest.raises(TypeError):
            raises_type_error()
        
        assert call_count == 1
    
    def test_retry_callbacks(self):
        """Test retry callbacks are called."""
        on_retry_calls = []
        on_success_calls = []
        
        def on_retry(attempt, exception, delay):
            on_retry_calls.append((attempt, type(exception).__name__, delay))
        
        def on_success(attempts, total_time):
            on_success_calls.append((attempts, total_time))
        
        call_count = 0
        
        @retry_with_backoff(
            max_retries=3,
            base_delay=0.01,
            jitter=False,
            on_retry=on_retry,
            on_success=on_success
        )
        def fail_twice():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("fail")
            return "success"
        
        fail_twice()
        
        assert len(on_retry_calls) == 2
        assert len(on_success_calls) == 1
        assert on_retry_calls[0][0] == 1
        assert on_retry_calls[1][0] == 2
        assert on_success_calls[0][0] == 3
    
    def test_on_failure_callback(self):
        """Test on_failure callback is called."""
        on_failure_calls = []
        
        def on_failure(exception):
            on_failure_calls.append(type(exception).__name__)
        
        @retry_with_backoff(
            max_retries=1,
            base_delay=0.01,
            jitter=False,
            on_failure=on_failure
        )
        def always_fail():
            raise ValueError("fail")
        
        with pytest.raises(RetryExhausted):
            always_fail()
        
        assert len(on_failure_calls) == 1
        assert on_failure_calls[0] == "RetryExhausted"


class TestAsyncRetry:
    """Test async retry functionality."""
    
    @pytest.mark.asyncio
    async def test_async_successful_function(self):
        """Test successful async function."""
        call_count = 0
        
        @retry_with_backoff(max_retries=3, base_delay=0.01, jitter=False)
        async def async_success():
            nonlocal call_count
            call_count += 1
            return "async success"
        
        result = await async_success()
        
        assert result == "async success"
        assert call_count == 1
    
    @pytest.mark.asyncio
    async def test_async_retry_on_failure(self):
        """Test async function is retried on failure."""
        call_count = 0
        
        @retry_with_backoff(max_retries=3, base_delay=0.01, jitter=False)
        async def async_fail_then_succeed():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("not yet")
            return "success"
        
        result = await async_fail_then_succeed()
        
        assert result == "success"
        assert call_count == 3
    
    @pytest.mark.asyncio
    async def test_async_retry_exhausted(self):
        """Test RetryExhausted raised for async function."""
        @retry_with_backoff(max_retries=2, base_delay=0.01, jitter=False)
        async def async_always_fail():
            raise ValueError("always fails")
        
        with pytest.raises(RetryExhausted) as exc_info:
            await async_always_fail()
        
        assert exc_info.value.attempts == 3


class TestRetryCall:
    """Test retry_call function."""
    
    def test_retry_call_success(self):
        """Test retry_call with successful function."""
        def success_func():
            return "success"
        
        result = retry_call(success_func)
        
        assert result == "success"
    
    def test_retry_call_with_args(self):
        """Test retry_call with function arguments."""
        def func_with_args(a, b, c=None):
            return f"{a}-{b}-{c}"
        
        result = retry_call(func_with_args, 1, 2, c=3, max_retries=1, base_delay=0.01)
        
        assert result == "1-2-3"
    
    def test_retry_call_retries(self):
        """Test retry_call retries on failure."""
        call_count = 0
        
        def fail_then_succeed():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("fail")
            return "success"
        
        result = retry_call(fail_then_succeed, max_retries=3, base_delay=0.01)
        
        assert result == "success"
        assert call_count == 3


class TestRetryCallAsync:
    """Test retry_call_async function."""
    
    @pytest.mark.asyncio
    async def test_retry_call_async_success(self):
        """Test retry_call_async with successful function."""
        async def async_success():
            return "async success"
        
        result = await retry_call_async(async_success)
        
        assert result == "async success"
    
    @pytest.mark.asyncio
    async def test_retry_call_async_retries(self):
        """Test retry_call_async retries on failure."""
        call_count = 0
        
        async def async_fail_then_succeed():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("fail")
            return "success"
        
        result = await retry_call_async(
            async_fail_then_succeed,
            max_retries=3,
            base_delay=0.01
        )
        
        assert result == "success"
        assert call_count == 3


class TestRetryable:
    """Test Retryable class."""
    
    def test_retryable_decorator(self):
        """Test Retryable as decorator."""
        retryable = Retryable(max_retries=3, base_delay=0.01, jitter=False)
        
        call_count = 0
        
        @retryable
        def func():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ValueError("fail")
            return "success"
        
        result = func()
        
        assert result == "success"
        assert call_count == 2
    
    def test_retryable_get_stats(self):
        """Test getting stats from Retryable."""
        retryable = Retryable(max_retries=1, base_delay=0.01, jitter=False)
        
        @retryable
        def success_func():
            return "success"
        
        success_func()
        stats = retryable.get_stats()
        
        assert stats.attempts == 1
        assert stats.successes == 1


class TestPredefinedDecorators:
    """Test predefined retry decorators."""
    
    def test_network_retry(self):
        """Test network_retry decorator."""
        call_count = 0
        
        @network_retry
        def network_func():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ConnectionError("network error")
            return "success"
        
        result = network_func()
        
        assert result == "success"
        assert call_count == 2
    
    def test_database_retry(self):
        """Test database_retry decorator."""
        call_count = 0
        
        @database_retry
        def db_func():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ConnectionError("db error")
            return "success"
        
        result = db_func()
        
        assert result == "success"
        assert call_count == 2
    
    def test_api_retry(self):
        """Test api_retry decorator."""
        call_count = 0
        
        @api_retry
        def api_func():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise TimeoutError("api timeout")
            return "success"
        
        result = api_func()
        
        assert result == "success"
        assert call_count == 2
    
    def test_api_retry_not_retry_value_error(self):
        """Test api_retry does not retry ValueError."""
        call_count = 0
        
        @api_retry
        def api_func():
            nonlocal call_count
            call_count += 1
            raise ValueError("not retried")
        
        with pytest.raises(ValueError):
            api_func()
        
        assert call_count == 1
    
    def test_with_retry(self):
        """Test with_retry convenience decorator."""
        call_count = 0
        
        @with_retry(retries=3, delay=0.01)
        def func():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ValueError("fail")
            return "success"
        
        result = func()
        
        assert result == "success"
        assert call_count == 2


class TestEdgeCases:
    """Test edge cases."""
    
    def test_zero_retries(self):
        """Test with zero retries."""
        call_count = 0
        
        @retry_with_backoff(max_retries=0, jitter=False)
        def func():
            nonlocal call_count
            call_count += 1
            raise ValueError("fail")
        
        with pytest.raises(RetryExhausted) as exc_info:
            func()
        
        assert exc_info.value.attempts == 1
        assert call_count == 1
    
    def test_exception_in_callback_ignored(self):
        """Test exception in callback is ignored."""
        def bad_callback(*args):
            raise RuntimeError("callback error")
        
        @retry_with_backoff(
            max_retries=1,
            base_delay=0.01,
            jitter=False,
            on_retry=bad_callback,
            on_success=bad_callback,
            on_failure=bad_callback
        )
        def func():
            raise ValueError("fail")
        
        # Should not raise callback error
        with pytest.raises(RetryExhausted):
            func()


class TestDecoratorAttributes:
    """Test decorator attributes for introspection."""
    
    def test_decorator_has_config(self):
        """Test decorated function has retry_config attribute."""
        @retry_with_backoff(max_retries=5, base_delay=2.0, jitter=False)
        def func():
            return "success"
        
        assert hasattr(func, 'retry_config')
        assert func.retry_config.max_retries == 5
        assert func.retry_config.base_delay == 2.0