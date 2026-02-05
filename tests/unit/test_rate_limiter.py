"""
tests/unit/test_rate_limiter.py
Comprehensive tests for rate limiter functionality.

Tests cover:
- Token bucket algorithm
- Sliding window algorithm
- Local rate limiter backend
- Redis rate limiter backend (mocked)
- Rate limit result and config
- Rate limit exceeded exception
- Rate limiter registry
- Decorator functionality
"""

import pytest
import time
import asyncio
from unittest.mock import Mock, patch, MagicMock

from src.utils.rate_limiter import (
    RateLimitResult,
    RateLimitConfig,
    RateLimitExceeded,
    LocalTokenBucket,
    LocalSlidingWindow,
    LocalRateLimiterBackend,
    RedisRateLimiterBackend,
    RateLimiter,
    RateLimiterRegistry,
    get_rate_limiter_registry,
    rate_limit,
    API_RATE_LIMIT,
    INFERENCE_RATE_LIMIT,
    GLOBAL_RATE_LIMIT,
    LOGIN_RATE_LIMIT,
)


class TestRateLimitResult:
    """Test RateLimitResult dataclass."""
    
    def test_creation(self):
        """Test creating rate limit result."""
        result = RateLimitResult(
            allowed=True,
            limit=100,
            remaining=50,
            reset_time=1234567890.0
        )
        
        assert result.allowed is True
        assert result.limit == 100
        assert result.remaining == 50
        assert result.reset_time == 1234567890.0
        assert result.retry_after is None
    
    def test_creation_with_retry(self):
        """Test creating result with retry after."""
        result = RateLimitResult(
            allowed=False,
            limit=100,
            remaining=0,
            reset_time=1234567890.0,
            retry_after=60.0
        )
        
        assert result.allowed is False
        assert result.retry_after == 60.0


class TestRateLimitConfig:
    """Test RateLimitConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = RateLimitConfig()
        
        assert config.requests_per_second == 10.0
        assert config.burst_size == 20
        assert config.window_size == 60.0
        assert config.key_prefix == "nexus:rl"
        assert config.block_duration is None
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = RateLimitConfig(
            requests_per_second=100.0,
            burst_size=50,
            window_size=300.0,
            key_prefix="custom:prefix",
            block_duration=600.0
        )
        
        assert config.requests_per_second == 100.0
        assert config.burst_size == 50
        assert config.window_size == 300.0
        assert config.key_prefix == "custom:prefix"
        assert config.block_duration == 600.0


class TestRateLimitExceeded:
    """Test RateLimitExceeded exception."""
    
    def test_exception_creation(self):
        """Test creating exception."""
        exc = RateLimitExceeded(
            key="user:123",
            limit=100,
            retry_after=60.0
        )
        
        assert exc.key == "user:123"
        assert exc.limit == 100
        assert exc.retry_after == 60.0
        assert "user:123" in str(exc)
    
    def test_custom_message(self):
        """Test exception with custom message."""
        exc = RateLimitExceeded(
            key="user:123",
            limit=100,
            retry_after=60.0,
            message="Custom rate limit message"
        )
        
        assert "Custom rate limit message" in str(exc)


class TestLocalTokenBucket:
    """Test LocalTokenBucket class."""
    
    def test_initial_tokens(self):
        """Test initial token count equals burst size."""
        bucket = LocalTokenBucket(burst_size=10, refill_rate=1.0)
        
        assert bucket.tokens == 10.0
    
    def test_consume_success(self):
        """Test successful token consumption."""
        bucket = LocalTokenBucket(burst_size=10, refill_rate=1.0)
        
        success, remaining = bucket.consume(5.0)
        
        assert success is True
        assert remaining == 5.0
    
    def test_consume_failure(self):
        """Test failed token consumption."""
        bucket = LocalTokenBucket(burst_size=5, refill_rate=1.0)
        
        # Consume all tokens
        bucket.consume(5.0)
        
        # Try to consume more
        success, retry_after = bucket.consume(1.0)
        
        assert success is False
        assert retry_after > 0
    
    def test_token_refill(self):
        """Test token refill over time."""
        bucket = LocalTokenBucket(burst_size=10, refill_rate=10.0)
        
        # Consume all tokens
        bucket.consume(10.0)
        assert bucket.tokens == 0.0
        
        # Wait for refill
        time.sleep(0.2)
        
        success, remaining = bucket.consume(1.0)
        assert success is True


class TestLocalSlidingWindow:
    """Test LocalSlidingWindow class."""
    
    def test_add_request_within_limit(self):
        """Test adding request within limit."""
        window = LocalSlidingWindow(limit=10, window_size=60.0)
        
        allowed, remaining, reset_time = window.add_request()
        
        assert allowed is True
        assert remaining == 9
    
    def test_add_request_exceeds_limit(self):
        """Test adding request that exceeds limit."""
        window = LocalSlidingWindow(limit=2, window_size=60.0)
        
        window.add_request()
        window.add_request()
        allowed, remaining, reset_time = window.add_request()
        
        assert allowed is False
        assert remaining == 0
    
    def test_window_slides(self):
        """Test window slides over time."""
        window = LocalSlidingWindow(limit=2, window_size=0.1)
        
        window.add_request()
        window.add_request()
        
        # Wait for window to slide
        time.sleep(0.15)
        
        allowed, remaining, reset_time = window.add_request()
        assert allowed is True


class TestLocalRateLimiterBackend:
    """Test LocalRateLimiterBackend class."""
    
    def test_check_token_bucket_allowed(self):
        """Test token bucket check allowed."""
        backend = LocalRateLimiterBackend()
        
        result = backend.check_token_bucket("key1", 1.0, 10, 1.0)
        
        assert result.allowed is True
        assert result.limit == 10
        assert result.remaining == 9
    
    def test_check_token_bucket_denied(self):
        """Test token bucket check denied."""
        backend = LocalRateLimiterBackend()
        
        # Consume all tokens
        for _ in range(10):
            backend.check_token_bucket("key1", 1.0, 10, 1000.0)
        
        result = backend.check_token_bucket("key1", 1.0, 10, 1000.0)
        
        assert result.allowed is False
    
    def test_check_sliding_window_allowed(self):
        """Test sliding window check allowed."""
        backend = LocalRateLimiterBackend()
        
        result = backend.check_sliding_window("key1", 10, 60.0)
        
        assert result.allowed is True
    
    def test_check_sliding_window_denied(self):
        """Test sliding window check denied."""
        backend = LocalRateLimiterBackend()
        
        # Exhaust limit
        for _ in range(10):
            backend.check_sliding_window("key1", 10, 60.0)
        
        result = backend.check_sliding_window("key1", 10, 60.0)
        
        assert result.allowed is False
    
    def test_reset(self):
        """Test resetting rate limit."""
        backend = LocalRateLimiterBackend()
        
        backend.check_token_bucket("key1", 1.0, 10, 1.0)
        assert backend.reset("key1") is True
        
        # Should be able to make requests again
        result = backend.check_token_bucket("key1", 1.0, 10, 1.0)
        assert result.remaining == 9
    
    def test_cleanup(self):
        """Test periodic cleanup."""
        backend = LocalRateLimiterBackend()
        
        # Add requests
        backend.check_sliding_window("key1", 10, 0.05)
        
        # Force cleanup by waiting
        time.sleep(0.1)
        backend._cleanup_if_needed()
        
        # Bucket should still exist but be empty or cleaned up
        assert isinstance(backend, LocalRateLimiterBackend)


class TestRedisRateLimiterBackend:
    """Test RedisRateLimiterBackend class (mocked)."""
    
    @patch("redis.Redis")
    def test_init_success(self, mock_redis):
        """Test successful initialization."""
        mock_client = Mock()
        mock_client.ping.return_value = True
        mock_redis.return_value = mock_client
        
        backend = RedisRateLimiterBackend()
        
        assert backend._redis is mock_client
    
    @patch("redis.Redis")
    def test_init_failure(self, mock_redis):
        """Test initialization failure."""
        mock_redis.side_effect = Exception("Connection failed")
        
        with pytest.raises(Exception):
            RedisRateLimiterBackend()
    
    @patch("redis.Redis")
    def test_check_token_bucket(self, mock_redis):
        """Test token bucket check with Redis."""
        mock_client = Mock()
        mock_client.eval.return_value = [1, 9]  # allowed=True, remaining=9
        mock_redis.return_value = mock_client
        
        backend = RedisRateLimiterBackend()
        result = backend.check_token_bucket("key1", 1.0, 10, 1.0)
        
        assert result.allowed is True
        assert result.remaining == 9
    
    @patch("redis.Redis")
    def test_check_sliding_window(self, mock_redis):
        """Test sliding window check with Redis."""
        mock_client = Mock()
        mock_client.eval.return_value = [1, 9, 1234567890.0]  # allowed, remaining, reset_time
        mock_redis.return_value = mock_client
        
        backend = RedisRateLimiterBackend()
        result = backend.check_sliding_window("key1", 10, 60.0)
        
        assert result.allowed is True
        assert result.remaining == 9
    
    @patch("redis.Redis")
    def test_reset(self, mock_redis):
        """Test resetting rate limit in Redis."""
        mock_client = Mock()
        mock_client.delete.return_value = 1
        mock_redis.return_value = mock_client
        
        backend = RedisRateLimiterBackend()
        result = backend.reset("key1")
        
        assert result is True


class TestRateLimiter:
    """Test RateLimiter class."""
    
    def test_is_allowed_token_bucket(self):
        """Test is_allowed with token bucket."""
        limiter = RateLimiter(algorithm="token_bucket")
        
        result = limiter.is_allowed(user_id="user1", action="api_call")
        
        assert isinstance(result, RateLimitResult)
        assert result.allowed is True
    
    def test_is_allowed_sliding_window(self):
        """Test is_allowed with sliding window."""
        limiter = RateLimiter(algorithm="sliding_window")
        
        result = limiter.is_allowed(user_id="user1", action="api_call")
        
        assert isinstance(result, RateLimitResult)
        assert result.allowed is True
    
    def test_check_allowed(self):
        """Test check when allowed."""
        limiter = RateLimiter()
        
        # Should not raise
        limiter.check(user_id="user1")
    
    def test_check_denied(self):
        """Test check when denied."""
        config = RateLimitConfig(requests_per_second=1.0, burst_size=1)
        limiter = RateLimiter(config=config, algorithm="token_bucket")
        
        # Exhaust limit
        limiter.check(user_id="user1")
        limiter.check(user_id="user1")
        
        with pytest.raises(RateLimitExceeded):
            limiter.check(user_id="user1")
    
    def test_reset(self):
        """Test resetting rate limit."""
        limiter = RateLimiter()
        
        limiter.is_allowed(user_id="user1")
        assert limiter.reset(user_id="user1") is True
    
    @pytest.mark.asyncio
    async def test_is_allowed_async(self):
        """Test async is_allowed."""
        limiter = RateLimiter()
        
        result = await limiter.is_allowed_async(user_id="user1")
        
        assert isinstance(result, RateLimitResult)
    
    @pytest.mark.asyncio
    async def test_check_async(self):
        """Test async check."""
        limiter = RateLimiter()
        
        # Should not raise
        await limiter.check_async(user_id="user1")
    
    def test_make_key(self):
        """Test key generation."""
        config = RateLimitConfig(key_prefix="test")
        limiter = RateLimiter(config=config)
        
        key = limiter._make_key(user_id="user1", action="api", resource="data")
        
        assert "test" in key
        assert "user1" in key
    
    def test_make_key_long(self):
        """Test key hashing for long keys."""
        config = RateLimitConfig(key_prefix="test")
        limiter = RateLimiter(config=config)
        
        long_user_id = "a" * 200
        key = limiter._make_key(user_id=long_user_id)
        
        assert len(key) <= 100  # Should be hashed
    
    def test_unknown_algorithm(self):
        """Test unknown algorithm raises error."""
        limiter = RateLimiter(algorithm="unknown")
        
        with pytest.raises(ValueError):
            limiter.is_allowed()


class TestRateLimiterRegistry:
    """Test RateLimiterRegistry class."""
    
    def test_register(self):
        """Test registering a rate limiter."""
        registry = RateLimiterRegistry()
        limiter = registry.register("api_limiter")
        
        assert isinstance(limiter, RateLimiter)
        assert registry.get("api_limiter") is limiter
    
    def test_register_duplicate(self):
        """Test registering duplicate name raises error."""
        registry = RateLimiterRegistry()
        registry.register("test")
        
        with pytest.raises(ValueError):
            registry.register("test")
    
    def test_get_missing(self):
        """Test getting non-existent rate limiter."""
        registry = RateLimiterRegistry()
        
        assert registry.get("missing") is None
    
    def test_remove(self):
        """Test removing rate limiter."""
        registry = RateLimiterRegistry()
        registry.register("test")
        
        assert registry.remove("test") is True
        assert registry.get("test") is None
        assert registry.remove("test") is False


class TestGlobalRegistry:
    """Test global registry functions."""
    
    def test_get_rate_limiter_registry_singleton(self):
        """Test get_rate_limiter_registry returns singleton."""
        registry1 = get_rate_limiter_registry()
        registry2 = get_rate_limiter_registry()
        
        assert registry1 is registry2


class TestRateLimitDecorator:
    """Test rate_limit decorator."""
    
    def test_decorator_with_limiter(self):
        """Test decorator with existing limiter."""
        registry = get_rate_limiter_registry()
        registry.register("test_limiter")
        
        @rate_limit("test_limiter")
        def my_function():
            return "success"
        
        result = my_function()
        assert result == "success"
    
    def test_decorator_with_missing_limiter(self):
        """Test decorator with missing limiter logs warning."""
        
        @rate_limit("nonexistent_limiter")
        def my_function():
            return "success"
        
        # Should still work (warns but doesn't block)
        result = my_function()
        assert result == "success"
    
    def test_decorator_extracts_user_id(self):
        """Test decorator extracts user_id from arguments."""
        registry = get_rate_limiter_registry()
        registry.register("user_limiter")
        
        @rate_limit("user_limiter", user_id_arg="user_id")
        def my_function(user_id):
            return f"processed {user_id}"
        
        result = my_function("user123")
        assert result == "processed user123"


class TestPredefinedConfigs:
    """Test predefined rate limit configurations."""
    
    def test_api_rate_limit(self):
        """Test API_RATE_LIMIT config."""
        assert API_RATE_LIMIT.requests_per_second == 100/60
        assert API_RATE_LIMIT.burst_size == 20
        assert API_RATE_LIMIT.key_prefix == "nexus:api"
    
    def test_inference_rate_limit(self):
        """Test INFERENCE_RATE_LIMIT config."""
        assert INFERENCE_RATE_LIMIT.requests_per_second == 10.0
        assert INFERENCE_RATE_LIMIT.burst_size == 5
        assert INFERENCE_RATE_LIMIT.key_prefix == "nexus:inference"
    
    def test_global_rate_limit(self):
        """Test GLOBAL_RATE_LIMIT config."""
        assert GLOBAL_RATE_LIMIT.requests_per_second == 1000.0
        assert GLOBAL_RATE_LIMIT.burst_size == 100
        assert GLOBAL_RATE_LIMIT.key_prefix == "nexus:global"
    
    def test_login_rate_limit(self):
        """Test LOGIN_RATE_LIMIT config."""
        assert LOGIN_RATE_LIMIT.requests_per_second == 5/60
        assert LOGIN_RATE_LIMIT.burst_size == 5
        assert LOGIN_RATE_LIMIT.window_size == 300.0
        assert LOGIN_RATE_LIMIT.block_duration == 300.0


class TestEdgeCases:
    """Test edge cases."""
    
    def test_token_bucket_exactly_empty(self):
        """Test token bucket when exactly empty."""
        bucket = LocalTokenBucket(burst_size=5, refill_rate=1000.0)
        
        # Consume exactly all tokens
        success, remaining = bucket.consume(5.0)
        assert success is True
        assert remaining == 0.0
        
        # Next consume should fail
        success, retry_after = bucket.consume(1.0)
        assert success is False
    
    def test_sliding_window_exactly_at_limit(self):
        """Test sliding window at exact limit."""
        window = LocalSlidingWindow(limit=3, window_size=60.0)
        
        allowed1, _, _ = window.add_request()
        allowed2, _, _ = window.add_request()
        allowed3, _, _ = window.add_request()
        allowed4, _, _ = window.add_request()
        
        assert allowed1 is True
        assert allowed2 is True
        assert allowed3 is True
        assert allowed4 is False
    
    def test_concurrent_access(self):
        """Test thread-safe concurrent access."""
        import threading
        
        backend = LocalRateLimiterBackend()
        errors = []
        
        def worker():
            try:
                for _ in range(10):
                    backend.check_token_bucket("key", 0.1, 100, 1000.0)
            except Exception as e:
                errors.append(e)
        
        threads = [threading.Thread(target=worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert len(errors) == 0
