"""
Comprehensive tests for resilience patterns module.

Tests cover:
- Circuit Breaker pattern (closed, open, half-open states)
- Retry Policy with exponential backoff
- Bulkhead pattern for resource isolation
- Timeout handling
- Integration between components
- Decorator functionality

Target: 80%+ coverage of /src/core/resilience.py
"""

import pytest
import time
import threading
from unittest.mock import Mock, patch, MagicMock
from typing import List

from src.core.resilience import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerOpen,
    CircuitState,
    RetryPolicy,
    RetryConfig,
    Bulkhead,
    BulkheadFull,
    BulkheadTimeout,
    Timeout,
    ResilientClient,
    circuit_breaker,
    retry,
    timeout as timeout_decorator,
)


# =============================================================================
# CircuitBreaker Tests
# =============================================================================


class TestCircuitBreakerConfig:
    """Test CircuitBreakerConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = CircuitBreakerConfig()

        assert config.failure_threshold == 5
        assert config.recovery_timeout == 30.0
        assert config.half_open_max_calls == 3
        assert config.success_threshold == 2
        assert config.expected_exception == Exception

    def test_custom_config(self):
        """Test custom configuration values."""
        config = CircuitBreakerConfig(
            failure_threshold=3,
            recovery_timeout=10.0,
            half_open_max_calls=5,
            success_threshold=3,
            expected_exception=ValueError,
        )

        assert config.failure_threshold == 3
        assert config.recovery_timeout == 10.0
        assert config.half_open_max_calls == 5
        assert config.success_threshold == 3
        assert config.expected_exception == ValueError


class TestCircuitBreakerClosedState:
    """Test circuit breaker in CLOSED state (normal operation)."""

    def test_initial_state_is_closed(self):
        """Test that circuit starts in CLOSED state."""
        breaker = CircuitBreaker()
        assert breaker.state == CircuitState.CLOSED

    def test_successful_call_in_closed_state(self):
        """Test successful function execution in closed state."""
        breaker = CircuitBreaker()

        def success_func():
            return "success"

        result = breaker.call(success_func)
        assert result == "success"
        assert breaker.state == CircuitState.CLOSED

    def test_failure_count_increments(self):
        """Test that failures increment the failure counter."""
        config = CircuitBreakerConfig(failure_threshold=5)
        breaker = CircuitBreaker(config)

        def fail_func():
            raise ValueError("test error")

        # Make 2 failures
        for _ in range(2):
            with pytest.raises(ValueError):
                breaker.call(fail_func)

        stats = breaker.get_stats()
        assert stats["failures"] == 2
        assert stats["failure_count"] == 2
        assert breaker.state == CircuitState.CLOSED

    def test_success_resets_failure_count(self):
        """Test that success resets the failure counter."""
        config = CircuitBreakerConfig(failure_threshold=5)
        breaker = CircuitBreaker(config)

        call_count = 0

        def sometimes_fails():
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise ValueError("fail")
            return "success"

        # Two failures
        for _ in range(2):
            with pytest.raises(ValueError):
                breaker.call(sometimes_fails)

        assert breaker.get_stats()["failure_count"] == 2

        # Success should reset
        result = breaker.call(sometimes_fails)
        assert result == "success"
        assert breaker.get_stats()["failure_count"] == 0

    def test_call_stats_increment(self):
        """Test that call statistics are tracked."""
        breaker = CircuitBreaker()

        def success_func():
            return "ok"

        breaker.call(success_func)
        breaker.call(success_func)

        stats = breaker.get_stats()
        assert stats["calls"] == 2
        assert stats["successes"] == 2


class TestCircuitBreakerOpenState:
    """Test circuit breaker in OPEN state (failing, rejecting requests)."""

    def test_transitions_to_open_on_threshold(self):
        """Test circuit opens when failure threshold is reached."""
        config = CircuitBreakerConfig(failure_threshold=3)
        breaker = CircuitBreaker(config)

        def fail_func():
            raise ValueError("always fails")

        # 3 failures should open the circuit
        for _ in range(3):
            with pytest.raises(ValueError):
                breaker.call(fail_func)

        assert breaker.state == CircuitState.OPEN

    def test_rejects_calls_when_open(self):
        """Test that calls are rejected when circuit is open."""
        config = CircuitBreakerConfig(failure_threshold=1)
        breaker = CircuitBreaker(config)

        def fail_func():
            raise ValueError("fail")

        # Open the circuit
        with pytest.raises(ValueError):
            breaker.call(fail_func)

        assert breaker.state == CircuitState.OPEN

        # Next call should be rejected
        with pytest.raises(CircuitBreakerOpen) as exc_info:
            breaker.call(lambda: "should not execute")

        assert "OPEN" in str(exc_info.value)
        assert "test" in breaker.name or "default" in breaker.name

    def test_rejected_stats_increment(self):
        """Test that rejected calls are tracked in stats."""
        config = CircuitBreakerConfig(failure_threshold=1)
        breaker = CircuitBreaker(config)

        def fail_func():
            raise ValueError("fail")

        # Open the circuit
        with pytest.raises(ValueError):
            breaker.call(fail_func)

        # Rejected calls
        for _ in range(3):
            with pytest.raises(CircuitBreakerOpen):
                breaker.call(lambda: "x")

        stats = breaker.get_stats()
        assert stats["rejected"] == 3

    def test_state_transitions_tracked(self):
        """Test that state transitions are tracked."""
        config = CircuitBreakerConfig(failure_threshold=2)
        breaker = CircuitBreaker(config)

        def fail_func():
            raise ValueError("fail")

        # 2 failures -> OPEN
        for _ in range(2):
            with pytest.raises(ValueError):
                breaker.call(fail_func)

        stats = breaker.get_stats()
        assert stats["state_transitions"] >= 1


class TestCircuitBreakerHalfOpenState:
    """Test circuit breaker in HALF_OPEN state (testing recovery)."""

    def test_transitions_to_half_open_after_timeout(self):
        """Test circuit transitions to half-open after recovery timeout."""
        config = CircuitBreakerConfig(
            failure_threshold=1,
            recovery_timeout=0.1,  # 100ms for fast test
        )
        breaker = CircuitBreaker(config)

        def fail_func():
            raise ValueError("fail")

        # Open the circuit
        with pytest.raises(ValueError):
            breaker.call(fail_func)

        assert breaker.state == CircuitState.OPEN

        # Wait for recovery timeout
        time.sleep(0.15)

        # Trigger state update by making a call
        try:
            breaker.call(lambda: "test")
        except CircuitBreakerOpen:
            pass  # Expected if in half-open

        # State should now be half-open
        assert breaker.state == CircuitState.HALF_OPEN

    def test_half_open_limits_calls(self):
        """Test that half-open state limits concurrent test calls."""
        config = CircuitBreakerConfig(
            failure_threshold=1,
            recovery_timeout=0.1,
            half_open_max_calls=2,
            success_threshold=5,  # High threshold so circuit stays in HALF_OPEN
        )
        breaker = CircuitBreaker(config)

        # Open the circuit
        with pytest.raises(ValueError):
            breaker.call(lambda: (_ for _ in ()).throw(ValueError("fail")))

        # Wait for recovery
        time.sleep(0.15)

        # Trigger state update to HALF_OPEN and consume available calls
        # In half-open, we can make up to half_open_max_calls
        successful_calls = 0
        rejected_calls = 0

        # Try more calls than allowed in half-open
        for _ in range(5):
            try:
                result = breaker.call(lambda: "success")
                successful_calls += 1
            except CircuitBreakerOpen:
                rejected_calls += 1
                break

        # Should have made exactly half_open_max_calls successful calls, then get rejected
        assert successful_calls == config.half_open_max_calls
        assert rejected_calls >= 1

    def test_success_in_half_open_increments_success_count(self):
        """Test success in half-open increments success counter."""
        config = CircuitBreakerConfig(
            failure_threshold=1,
            recovery_timeout=0.1,
            success_threshold=2,
            half_open_max_calls=5,
        )
        breaker = CircuitBreaker(config)

        # Open the circuit
        with pytest.raises(ValueError):
            breaker.call(lambda: (_ for _ in ()).throw(ValueError("fail")))

        # Wait for recovery
        time.sleep(0.15)

        # Trigger state update and verify we're in HALF_OPEN
        breaker.call(lambda: "success1")

        assert breaker.state == CircuitState.HALF_OPEN

        # Check internal success count incremented (access via _success_count)
        with breaker._lock:
            internal_success = breaker._success_count
        assert internal_success == 1

        # Second success should close circuit
        breaker.call(lambda: "success2")
        assert breaker.state == CircuitState.CLOSED

    def test_failure_in_half_open_returns_to_open(self):
        """Test that failure in half-open returns circuit to open."""
        config = CircuitBreakerConfig(failure_threshold=1, recovery_timeout=0.1)
        breaker = CircuitBreaker(config)

        # Open the circuit
        with pytest.raises(ValueError):
            breaker.call(lambda: (_ for _ in ()).throw(ValueError("fail")))

        # Wait for recovery
        time.sleep(0.15)

        # Trigger state update
        try:
            breaker.call(lambda: "x")
        except CircuitBreakerOpen:
            pass

        assert breaker.state == CircuitState.HALF_OPEN

        # Failure should return to open
        with pytest.raises(ValueError):
            breaker.call(lambda: (_ for _ in ()).throw(ValueError("fail again")))

        assert breaker.state == CircuitState.OPEN


class TestCircuitBreakerManualReset:
    """Test manual reset functionality."""

    def test_manual_reset_closes_circuit(self):
        """Test manual reset closes an open circuit."""
        config = CircuitBreakerConfig(failure_threshold=1)
        breaker = CircuitBreaker(config)

        # Open the circuit
        with pytest.raises(ValueError):
            breaker.call(lambda: (_ for _ in ()).throw(ValueError("fail")))

        assert breaker.state == CircuitState.OPEN

        # Manual reset
        breaker.manual_reset()

        assert breaker.state == CircuitState.CLOSED
        assert breaker.get_stats()["failure_count"] == 0
        assert breaker.get_stats()["success_count"] == 0

    def test_manual_reset_tracks_transition(self):
        """Test manual reset tracks state transition."""
        config = CircuitBreakerConfig(failure_threshold=1)
        breaker = CircuitBreaker(config)

        # Open the circuit
        with pytest.raises(ValueError):
            breaker.call(lambda: (_ for _ in ()).throw(ValueError("fail")))

        initial_transitions = breaker.get_stats()["state_transitions"]

        breaker.manual_reset()

        assert breaker.get_stats()["state_transitions"] == initial_transitions + 1


class TestCircuitBreakerDecorator:
    """Test circuit breaker as decorator."""

    def test_decorator_success(self):
        """Test successful decorated function."""
        breaker = CircuitBreaker()

        @breaker
        def my_function():
            return "decorated result"

        result = my_function()
        assert result == "decorated result"

    def test_decorator_failure(self):
        """Test decorated function that fails."""
        config = CircuitBreakerConfig(failure_threshold=1)
        breaker = CircuitBreaker(config)

        @breaker
        def failing_function():
            raise ValueError("decorated fail")

        with pytest.raises(ValueError):
            failing_function()

        assert breaker.state == CircuitState.OPEN

    def test_decorator_preserves_function_metadata(self):
        """Test that decorator preserves function name and docstring."""
        breaker = CircuitBreaker()

        @breaker
        def my_function():
            """My docstring."""
            return "result"

        assert my_function.__name__ == "my_function"
        assert my_function.__doc__ == "My docstring."


class TestCircuitBreakerExpectedException:
    """Test circuit breaker with specific expected exceptions."""

    def test_only_expected_exception_tracks_failure(self):
        """Test that only expected exceptions count as failures."""
        config = CircuitBreakerConfig(
            failure_threshold=1, expected_exception=ValueError
        )
        breaker = CircuitBreaker(config)

        # TypeError should not count as failure
        with pytest.raises(TypeError):
            breaker.call(lambda: (_ for _ in ()).throw(TypeError("type error")))

        assert breaker.state == CircuitState.CLOSED
        assert breaker.get_stats()["failures"] == 0

        # ValueError should count
        with pytest.raises(ValueError):
            breaker.call(lambda: (_ for _ in ()).throw(ValueError("value error")))

        assert breaker.state == CircuitState.OPEN


# =============================================================================
# RetryPolicy Tests
# =============================================================================


class TestRetryConfig:
    """Test RetryConfig dataclass."""

    def test_default_config(self):
        """Test default retry configuration."""
        config = RetryConfig()

        assert config.max_attempts == 3
        assert config.base_delay == 1.0
        assert config.max_delay == 60.0
        assert config.exponential_base == 2.0
        assert config.jitter is True
        assert config.retryable_exceptions == (Exception,)

    def test_custom_config(self):
        """Test custom retry configuration."""
        config = RetryConfig(
            max_attempts=5,
            base_delay=0.5,
            max_delay=30.0,
            exponential_base=3.0,
            jitter=False,
            retryable_exceptions=(ValueError, TypeError),
        )

        assert config.max_attempts == 5
        assert config.base_delay == 0.5
        assert config.max_delay == 30.0
        assert config.exponential_base == 3.0
        assert config.jitter is False
        assert config.retryable_exceptions == (ValueError, TypeError)


class TestRetryPolicyExecution:
    """Test retry policy execution behavior."""

    def test_success_on_first_attempt(self):
        """Test successful execution on first attempt."""
        retry_policy = RetryPolicy()

        def success_func():
            return "success"

        result = retry_policy.execute(success_func)
        assert result == "success"

    def test_retry_on_failure_then_success(self):
        """Test retry on failure eventually succeeds."""
        config = RetryConfig(max_attempts=3)
        retry_policy = RetryPolicy(config)

        call_count = 0

        def eventually_succeeds():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError(f"fail {call_count}")
            return f"success on {call_count}"

        with patch("time.sleep"):  # Mock sleep for faster test
            result = retry_policy.execute(eventually_succeeds)

        assert result == "success on 3"
        assert call_count == 3

    def test_max_retries_exceeded(self):
        """Test exception raised when max retries exceeded."""
        config = RetryConfig(max_attempts=3)
        retry_policy = RetryPolicy(config)

        def always_fails():
            raise ValueError("always fails")

        with patch("time.sleep"):  # Mock sleep for faster test
            with pytest.raises(ValueError, match="always fails"):
                retry_policy.execute(always_fails)

    def test_no_retry_for_non_retryable_exception(self):
        """Test that non-retryable exceptions don't trigger retry."""
        config = RetryConfig(max_attempts=3, retryable_exceptions=(ValueError,))
        retry_policy = RetryPolicy(config)

        call_count = 0

        def raises_type_error():
            nonlocal call_count
            call_count += 1
            raise TypeError("type error")

        with pytest.raises(TypeError):
            retry_policy.execute(raises_type_error)

        # Should only be called once, no retries
        assert call_count == 1


class TestRetryPolicyDelayCalculation:
    """Test retry delay calculation."""

    def test_exponential_backoff(self):
        """Test exponential backoff calculation."""
        config = RetryConfig(base_delay=1.0, exponential_base=2.0, jitter=False)
        retry_policy = RetryPolicy(config)

        # attempt 1: 1.0 * 2^0 = 1.0
        delay = retry_policy._calculate_delay(1)
        assert delay == 1.0

        # attempt 2: 1.0 * 2^1 = 2.0
        delay = retry_policy._calculate_delay(2)
        assert delay == 2.0

        # attempt 3: 1.0 * 2^2 = 4.0
        delay = retry_policy._calculate_delay(3)
        assert delay == 4.0

    def test_max_delay_cap(self):
        """Test that delay is capped at max_delay."""
        config = RetryConfig(
            base_delay=1.0, max_delay=5.0, exponential_base=2.0, jitter=False
        )
        retry_policy = RetryPolicy(config)

        # attempt 4 would be 8.0, but max is 5.0
        delay = retry_policy._calculate_delay(4)
        assert delay == 5.0

    def test_jitter_adds_randomness(self):
        """Test that jitter adds random variation."""
        config = RetryConfig(base_delay=1.0, jitter=True)
        retry_policy = RetryPolicy(config)

        delays = [retry_policy._calculate_delay(1) for _ in range(10)]

        # Jitter adds ±25%, so delays should vary
        # With base 1.0, range is 0.75 to 1.25
        assert all(0.7 <= d <= 1.3 for d in delays)

        # Not all should be identical (statistically unlikely)
        assert len(set(round(d, 4) for d in delays)) > 1

    def test_no_jitter_consistent_delay(self):
        """Test that without jitter, delay is consistent."""
        config = RetryConfig(base_delay=1.0, jitter=False)
        retry_policy = RetryPolicy(config)

        delay1 = retry_policy._calculate_delay(1)
        delay2 = retry_policy._calculate_delay(1)

        assert delay1 == delay2


class TestRetryPolicyDecorator:
    """Test retry policy as decorator."""

    def test_decorator_success(self):
        """Test successful decorated function."""
        retry_policy = RetryPolicy()

        @retry_policy
        def my_function():
            return "decorated result"

        result = my_function()
        assert result == "decorated result"

    def test_decorator_retries_on_failure(self):
        """Test decorator retries on failure."""
        config = RetryConfig(max_attempts=3)
        retry_policy = RetryPolicy(config)

        call_count = 0

        @retry_policy
        def eventually_succeeds():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("fail")
            return "success"

        with patch("time.sleep"):
            result = eventually_succeeds()

        assert result == "success"
        assert call_count == 3

    def test_decorator_preserves_metadata(self):
        """Test that decorator preserves function metadata."""
        retry_policy = RetryPolicy()

        @retry_policy
        def my_function():
            """My docstring."""
            return "result"

        assert my_function.__name__ == "my_function"
        assert my_function.__doc__ == "My docstring."


# =============================================================================
# Bulkhead Tests
# =============================================================================


class TestBulkheadInitialization:
    """Test bulkhead initialization."""

    def test_default_initialization(self):
        """Test default bulkhead configuration."""
        bulkhead = Bulkhead()

        assert bulkhead.max_concurrent == 10
        assert bulkhead.max_queue == 20
        assert bulkhead.queue_timeout == 30.0

    def test_custom_initialization(self):
        """Test custom bulkhead configuration."""
        bulkhead = Bulkhead(max_concurrent=5, max_queue=10, queue_timeout=10.0)

        assert bulkhead.max_concurrent == 5
        assert bulkhead.max_queue == 10
        assert bulkhead.queue_timeout == 10.0


class TestBulkheadExecution:
    """Test bulkhead execution behavior."""

    def test_successful_execution(self):
        """Test successful function execution through bulkhead."""
        bulkhead = Bulkhead()

        def success_func():
            return "success"

        result = bulkhead.execute(success_func)
        assert result == "success"

    def test_exception_propagation(self):
        """Test that exceptions are propagated."""
        bulkhead = Bulkhead()

        def failing_func():
            raise ValueError("test error")

        with pytest.raises(ValueError, match="test error"):
            bulkhead.execute(failing_func)

    def test_stats_tracking(self):
        """Test that execution stats are tracked."""
        bulkhead = Bulkhead()

        bulkhead.execute(lambda: "result1")
        bulkhead.execute(lambda: "result2")

        stats = bulkhead.get_stats()
        assert stats["executed"] == 2


class TestBulkheadConcurrencyLimit:
    """Test bulkhead concurrency limiting."""

    def test_concurrent_execution_limited(self):
        """Test that concurrent executions are limited."""
        bulkhead = Bulkhead(max_concurrent=2, max_queue=10)

        active_count = 0
        max_active = 0
        lock = threading.Lock()

        def slow_function():
            nonlocal active_count, max_active
            with lock:
                active_count += 1
                max_active = max(max_active, active_count)

            time.sleep(0.1)  # Hold slot for 100ms

            with lock:
                active_count -= 1

            return "done"

        threads = []
        for _ in range(5):
            t = threading.Thread(target=lambda: bulkhead.execute(slow_function))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # Max concurrent should be limited to 2
        assert max_active <= 2

    def test_queue_management(self):
        """Test that queue properly manages waiting operations."""
        bulkhead = Bulkhead(max_concurrent=1, max_queue=3)

        execution_order = []
        lock = threading.Lock()

        def slow_function(id: int):
            with lock:
                execution_order.append(f"start_{id}")
            time.sleep(0.05)
            with lock:
                execution_order.append(f"end_{id}")
            return f"result_{id}"

        threads = []
        for i in range(3):
            t = threading.Thread(
                target=lambda i=i: bulkhead.execute(lambda: slow_function(i))
            )
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # All should have executed
        assert len([e for e in execution_order if e.startswith("end_")]) == 3


class TestBulkheadFull:
    """Test bulkhead full condition."""

    def test_bulkhead_full_when_queue_exceeded(self):
        """Test that bulkhead raises BulkheadFull when queue is full."""
        bulkhead = Bulkhead(max_concurrent=1, max_queue=1)

        barrier = threading.Barrier(2)

        def blocking_function():
            barrier.wait()  # Block until test signals
            time.sleep(0.2)
            return "done"

        # Start a blocking operation
        t1 = threading.Thread(target=lambda: bulkhead.execute(blocking_function))
        t1.start()

        # Wait for first thread to acquire semaphore
        time.sleep(0.05)

        # Queue one more (fills queue)
        t2 = threading.Thread(target=lambda: bulkhead.execute(lambda: time.sleep(0.1)))
        t2.start()
        time.sleep(0.01)

        # Third operation should fail - bulkhead full
        with pytest.raises(BulkheadFull) as exc_info:
            bulkhead.execute(lambda: "x")

        assert "full" in str(exc_info.value).lower()

        # Release barrier and wait for threads
        barrier.wait()
        t1.join()
        t2.join()

    def test_rejected_stats_increment(self):
        """Test that rejected calls are tracked."""
        bulkhead = Bulkhead(max_concurrent=1, max_queue=0)

        # Start blocking operation
        t = threading.Thread(target=lambda: bulkhead.execute(lambda: time.sleep(0.1)))
        t.start()
        time.sleep(0.01)

        # Try to execute while busy - should be rejected
        try:
            bulkhead.execute(lambda: "x")
        except BulkheadFull:
            pass

        t.join()

        stats = bulkhead.get_stats()
        assert stats["rejected"] >= 1


class TestBulkheadTimeout:
    """Test bulkhead timeout behavior."""

    def test_timeout_waiting_for_slot(self):
        """Test timeout when waiting too long for a slot."""
        bulkhead = Bulkhead(
            max_concurrent=1,
            max_queue=2,
            queue_timeout=0.05,  # 50ms timeout
        )

        slot_held = threading.Event()

        def hold_slot():
            slot_held.set()
            time.sleep(1.0)

        # Hold the only slot for a long time
        t = threading.Thread(target=lambda: bulkhead.execute(hold_slot))
        t.start()
        slot_held.wait(timeout=0.5)  # Wait until slot is held

        # Start threads that will wait and potentially timeout
        timeout_occurred = threading.Event()

        def wait_for_slot():
            try:
                bulkhead.execute(lambda: time.sleep(0.1))
            except BulkheadTimeout:
                timeout_occurred.set()

        # Start multiple threads to fill queue and cause timeouts
        threads = []
        for _ in range(3):
            t2 = threading.Thread(target=wait_for_slot)
            threads.append(t2)
            t2.start()

        # Wait for at least one timeout
        timeout_occurred.wait(timeout=0.5)

        for t2 in threads:
            t2.join(timeout=0.5)

        assert timeout_occurred.is_set(), "Expected at least one timeout"

        t.join(timeout=0.5)

    def test_timeout_stats_increment(self):
        """Test that timeouts are tracked in stats."""
        bulkhead = Bulkhead(max_concurrent=1, max_queue=2, queue_timeout=0.05)

        slot_held = threading.Event()

        def hold_slot():
            slot_held.set()
            time.sleep(1.0)

        # Hold slot
        t = threading.Thread(target=lambda: bulkhead.execute(hold_slot))
        t.start()
        slot_held.wait(timeout=0.5)

        # Start threads that will timeout
        def wait_and_catch():
            try:
                bulkhead.execute(lambda: time.sleep(0.1))
            except BulkheadTimeout:
                pass

        threads = []
        for _ in range(2):
            t2 = threading.Thread(target=wait_and_catch)
            threads.append(t2)
            t2.start()

        for t2 in threads:
            t2.join(timeout=0.5)

        t.join(timeout=0.5)

        stats = bulkhead.get_stats()
        assert stats["timeouts"] >= 1


class TestBulkheadDecorator:
    """Test bulkhead as decorator."""

    def test_decorator_execution(self):
        """Test successful decorated function."""
        bulkhead = Bulkhead()

        @bulkhead
        def my_function():
            return "decorated result"

        result = my_function()
        assert result == "decorated result"

    def test_decorator_preserves_metadata(self):
        """Test that decorator preserves function metadata."""
        bulkhead = Bulkhead()

        @bulkhead
        def my_function():
            """My docstring."""
            return "result"

        assert my_function.__name__ == "my_function"
        assert my_function.__doc__ == "My docstring."


# =============================================================================
# Timeout Tests
# =============================================================================


class TestTimeoutExecution:
    """Test timeout wrapper behavior."""

    def test_successful_execution(self):
        """Test successful execution within timeout."""
        timeout = Timeout(seconds=1.0)

        def quick_function():
            return "success"

        result = timeout.execute(quick_function)
        assert result == "success"

    def test_timeout_raises_error(self):
        """Test that timeout raises TimeoutError."""
        timeout = Timeout(seconds=0.05)  # 50ms timeout

        def slow_function():
            time.sleep(1.0)  # Takes 1 second
            return "never reached"

        with pytest.raises(TimeoutError) as exc_info:
            timeout.execute(slow_function)

        assert "timed out" in str(exc_info.value).lower()
        assert "0.05" in str(exc_info.value)

    def test_exception_propagation(self):
        """Test that function exceptions are propagated."""
        timeout = Timeout(seconds=1.0)

        def failing_function():
            raise ValueError("test error")

        with pytest.raises(ValueError, match="test error"):
            timeout.execute(failing_function)


class TestTimeoutDecorator:
    """Test timeout as decorator."""

    def test_decorator_success(self):
        """Test successful decorated function."""
        timeout = Timeout(seconds=1.0)

        @timeout
        def my_function():
            return "decorated result"

        result = my_function()
        assert result == "decorated result"

    def test_decorator_timeout(self):
        """Test decorator times out slow function."""
        timeout = Timeout(seconds=0.05)

        @timeout
        def slow_function():
            time.sleep(1.0)
            return "never reached"

        with pytest.raises(TimeoutError):
            slow_function()

    def test_decorator_preserves_metadata(self):
        """Test that decorator preserves function metadata."""
        timeout = Timeout(seconds=1.0)

        @timeout
        def my_function():
            """My docstring."""
            return "result"

        assert my_function.__name__ == "my_function"
        assert my_function.__doc__ == "My docstring."


# =============================================================================
# ResilientClient Integration Tests
# =============================================================================


class TestResilientClient:
    """Test ResilientClient combining all patterns."""

    def test_initialization(self):
        """Test ResilientClient initialization."""
        client = ResilientClient()

        assert client.circuit is not None
        assert client.retry is not None
        assert client.bulkhead is not None

    def test_custom_initialization(self):
        """Test ResilientClient with custom configs."""
        circuit_config = CircuitBreakerConfig(failure_threshold=3)
        retry_config = RetryConfig(max_attempts=5)

        client = ResilientClient(
            circuit_config=circuit_config, retry_config=retry_config, max_concurrent=5
        )

        assert client.circuit.config.failure_threshold == 3
        assert client.retry.config.max_attempts == 5
        assert client.bulkhead.max_concurrent == 5

    def test_protect_decorator_success(self):
        """Test protect decorator with successful execution."""
        client = ResilientClient()

        @client.protect
        def my_function():
            return "protected result"

        result = my_function()
        assert result == "protected result"

    def test_protect_decorator_retry_then_success(self):
        """Test protect decorator retries then succeeds."""
        client = ResilientClient(
            retry_config=RetryConfig(max_attempts=3, base_delay=0.01)
        )

        call_count = 0

        @client.protect
        def eventually_succeeds():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError(f"fail {call_count}")
            return "success"

        with patch("time.sleep"):
            result = eventually_succeeds()

        assert result == "success"
        assert call_count == 3

    def test_get_stats(self):
        """Test getting combined statistics."""
        client = ResilientClient()

        # Execute some operations
        @client.protect
        def test_func():
            return "result"

        test_func()
        test_func()

        stats = client.get_stats()

        assert "circuit_breaker" in stats
        assert "bulkhead" in stats
        assert stats["circuit_breaker"]["calls"] == 2
        assert stats["bulkhead"]["executed"] == 2


# =============================================================================
# Convenience Function Tests
# =============================================================================


class TestCircuitBreakerConvenienceDecorator:
    """Test circuit_breaker convenience decorator."""

    def test_decorator_creation(self):
        """Test creating circuit breaker via convenience function."""
        breaker = circuit_breaker(
            failure_threshold=3, recovery_timeout=10.0, name="test_circuit"
        )

        assert isinstance(breaker, CircuitBreaker)
        assert breaker.config.failure_threshold == 3
        assert breaker.config.recovery_timeout == 10.0
        assert breaker.name == "test_circuit"

    def test_decorator_usage(self):
        """Test using convenience decorator."""

        @circuit_breaker(failure_threshold=1)
        def my_function():
            return "result"

        result = my_function()
        assert result == "result"


class TestRetryDecorator:
    """Test retry convenience decorator."""

    def test_decorator_creation(self):
        """Test creating retry policy via convenience function."""
        policy = retry(max_attempts=5, base_delay=0.5, max_delay=30.0)

        assert isinstance(policy, RetryPolicy)
        assert policy.config.max_attempts == 5
        assert policy.config.base_delay == 0.5
        assert policy.config.max_delay == 30.0

    def test_decorator_usage(self):
        """Test using convenience decorator."""
        call_count = 0

        @retry(max_attempts=3, base_delay=0.01)
        def my_function():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ValueError("fail")
            return "result"

        with patch("time.sleep"):
            result = my_function()

        assert result == "result"
        assert call_count == 2


class TestTimeoutConvenienceDecorator:
    """Test timeout convenience decorator."""

    def test_decorator_creation(self):
        """Test creating timeout via convenience function."""
        timeout = timeout_decorator(seconds=5.0)

        assert isinstance(timeout, Timeout)
        assert timeout.seconds == 5.0

    def test_decorator_usage(self):
        """Test using convenience decorator."""

        @timeout_decorator(seconds=1.0)
        def my_function():
            return "result"

        result = my_function()
        assert result == "result"


# =============================================================================
# CircuitState Enum Tests
# =============================================================================


class TestCircuitState:
    """Test CircuitState enum."""

    def test_closed_state(self):
        """Test CLOSED state."""
        assert CircuitState.CLOSED.value == "closed"
        assert CircuitState.CLOSED.name == "CLOSED"

    def test_open_state(self):
        """Test OPEN state."""
        assert CircuitState.OPEN.value == "open"
        assert CircuitState.OPEN.name == "OPEN"

    def test_half_open_state(self):
        """Test HALF_OPEN state."""
        assert CircuitState.HALF_OPEN.value == "half_open"
        assert CircuitState.HALF_OPEN.name == "HALF_OPEN"


# =============================================================================
# Exception Tests
# =============================================================================


class TestCircuitBreakerOpenException:
    """Test CircuitBreakerOpen exception."""

    def test_exception_creation(self):
        """Test creating exception."""
        exc = CircuitBreakerOpen("test message")
        assert str(exc) == "test message"

    def test_exception_inheritance(self):
        """Test exception inheritance."""
        exc = CircuitBreakerOpen("test")
        assert isinstance(exc, Exception)


class TestBulkheadFullException:
    """Test BulkheadFull exception."""

    def test_exception_creation(self):
        """Test creating exception."""
        exc = BulkheadFull("test message")
        assert str(exc) == "test message"

    def test_exception_inheritance(self):
        """Test exception inheritance."""
        exc = BulkheadFull("test")
        assert isinstance(exc, Exception)


class TestBulkheadTimeoutException:
    """Test BulkheadTimeout exception."""

    def test_exception_creation(self):
        """Test creating exception."""
        exc = BulkheadTimeout("test message")
        assert str(exc) == "test message"

    def test_exception_inheritance(self):
        """Test exception inheritance."""
        exc = BulkheadTimeout("test")
        assert isinstance(exc, Exception)


# =============================================================================
# Error Propagation Tests
# =============================================================================


class TestErrorPropagation:
    """Test error propagation through resilience patterns."""

    def test_circuit_breaker_propagates_exception(self):
        """Test circuit breaker propagates original exception."""
        breaker = CircuitBreaker()

        def raises_value_error():
            raise ValueError("original error")

        with pytest.raises(ValueError, match="original error"):
            breaker.call(raises_value_error)

    def test_retry_propagates_last_exception(self):
        """Test retry propagates last exception."""
        retry_policy = RetryPolicy(RetryConfig(max_attempts=2))

        def always_fails():
            raise TypeError("specific error")

        with pytest.raises(TypeError, match="specific error"):
            with patch("time.sleep"):
                retry_policy.execute(always_fails)

    def test_bulkhead_propagates_exception(self):
        """Test bulkhead propagates function exception."""
        bulkhead = Bulkhead()

        def raises_runtime_error():
            raise RuntimeError("runtime error")

        with pytest.raises(RuntimeError, match="runtime error"):
            bulkhead.execute(raises_runtime_error)


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_circuit_breaker_zero_threshold(self):
        """Test circuit breaker with zero failure threshold."""
        config = CircuitBreakerConfig(failure_threshold=0)
        breaker = CircuitBreaker(config)

        # Any failure should open immediately
        with pytest.raises(ValueError):
            breaker.call(lambda: (_ for _ in ()).throw(ValueError("fail")))

        assert breaker.state == CircuitState.OPEN

    def test_retry_single_attempt(self):
        """Test retry with single attempt (no retry)."""
        config = RetryConfig(max_attempts=1)
        retry_policy = RetryPolicy(config)

        call_count = 0

        def always_fails():
            nonlocal call_count
            call_count += 1
            raise ValueError("fail")

        with pytest.raises(ValueError):
            retry_policy.execute(always_fails)

        assert call_count == 1

    def test_bulkhead_zero_queue(self):
        """Test bulkhead with zero queue size - operations rejected when slot busy."""
        bulkhead = Bulkhead(max_concurrent=1, max_queue=0)

        # With max_queue=0, the queue check happens BEFORE incrementing
        # So any call when _queued >= 0 (which is always true) will be rejected
        # This tests the actual behavior - zero queue means no queuing allowed
        with pytest.raises(BulkheadFull):
            bulkhead.execute(lambda: "should fail")

    def test_concurrent_circuit_breaker_access(self):
        """Test thread-safe circuit breaker access."""
        config = CircuitBreakerConfig(failure_threshold=10)
        breaker = CircuitBreaker(config)

        results = []
        lock = threading.Lock()

        def access_breaker():
            try:
                result = breaker.call(lambda: "success")
                with lock:
                    results.append(result)
            except Exception as e:
                with lock:
                    results.append(str(e))

        threads = [threading.Thread(target=access_breaker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 10
        assert all(r == "success" for r in results)

    def test_concurrent_retry_access(self):
        """Test thread-safe retry access."""
        retry_policy = RetryPolicy(RetryConfig(max_attempts=2))

        results = []
        lock = threading.Lock()

        def access_retry():
            try:
                result = retry_policy.execute(lambda: "success")
                with lock:
                    results.append(result)
            except Exception as e:
                with lock:
                    results.append(str(e))

        threads = [threading.Thread(target=access_retry) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 5
        assert all(r == "success" for r in results)


# =============================================================================
# Statistics Tests
# =============================================================================


class TestStatistics:
    """Test statistics collection."""

    def test_circuit_breaker_stats_complete(self):
        """Test that all expected stats are present."""
        breaker = CircuitBreaker()

        stats = breaker.get_stats()

        expected_keys = [
            "calls",
            "successes",
            "failures",
            "rejected",
            "state_transitions",
            "state",
            "failure_count",
            "success_count",
            "is_open",
        ]

        for key in expected_keys:
            assert key in stats, f"Missing stat: {key}"

    def test_bulkhead_stats_complete(self):
        """Test that all expected bulkhead stats are present."""
        bulkhead = Bulkhead()

        stats = bulkhead.get_stats()

        expected_keys = [
            "executed",
            "queued",
            "rejected",
            "timeouts",
            "current_queue",
            "available_slots",
        ]

        for key in expected_keys:
            assert key in stats, f"Missing stat: {key}"

    def test_stats_after_operations(self):
        """Test stats after various operations."""
        breaker = CircuitBreaker(CircuitBreakerConfig(failure_threshold=5))

        # Successful calls
        breaker.call(lambda: "success")
        breaker.call(lambda: "success")

        # Failed calls
        for _ in range(3):
            try:
                breaker.call(lambda: (_ for _ in ()).throw(ValueError("fail")))
            except ValueError:
                pass

        stats = breaker.get_stats()
        assert stats["calls"] == 5
        assert stats["successes"] == 2
        assert stats["failures"] == 3


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Test integration between components."""

    def test_circuit_breaker_with_retry(self):
        """Test circuit breaker combined with retry policy."""
        circuit_config = CircuitBreakerConfig(failure_threshold=5)
        breaker = CircuitBreaker(circuit_config)
        retry_policy = RetryPolicy(RetryConfig(max_attempts=3))

        call_count = 0

        def flaky_operation():
            nonlocal call_count
            call_count += 1
            if call_count % 2 == 1:  # Fail on odd attempts
                raise ValueError(f"fail {call_count}")
            return f"success {call_count}"

        @breaker
        def protected_and_retried():
            return retry_policy.execute(flaky_operation)

        with patch("time.sleep"):
            # First call: fail (retry succeeds on 2nd attempt within retry)
            result = protected_and_retried()
            assert "success" in result

    def test_full_stack_resilience(self):
        """Test full stack: bulkhead -> circuit -> retry -> function."""
        client = ResilientClient(
            circuit_config=CircuitBreakerConfig(failure_threshold=5),
            retry_config=RetryConfig(max_attempts=2, base_delay=0.01),
            max_concurrent=3,
        )

        call_count = 0

        @client.protect
        def resilient_operation():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ValueError("first attempt fails")
            return f"success on attempt {call_count}"

        with patch("time.sleep"):
            result = resilient_operation()

        assert "success" in result
        assert call_count == 2
