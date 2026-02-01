"""
Rate Limiter Implementation for Nexus

Provides rate limiting with token bucket and sliding window algorithms.
Supports both local and Redis backends for distributed rate limiting.
"""

import asyncio
import hashlib
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Set, Tuple, Union
from collections import deque
from functools import wraps
import threading

logger = logging.getLogger(__name__)


@dataclass
class RateLimitResult:
    """Result of a rate limit check."""
    allowed: bool
    limit: int
    remaining: int
    reset_time: float
    retry_after: Optional[float] = None
    window_start: Optional[float] = None


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""
    requests_per_second: float = 10.0
    burst_size: int = 20
    window_size: float = 60.0  # seconds for sliding window
    key_prefix: str = "nexus:rl"
    block_duration: Optional[float] = None  # Block duration after exceeding limit


class RateLimitExceeded(Exception):
    """Raised when rate limit is exceeded."""
    
    def __init__(
        self,
        key: str,
        limit: int,
        retry_after: float,
        message: Optional[str] = None
    ):
        self.key = key
        self.limit = limit
        self.retry_after = retry_after
        super().__init__(
            message or f"Rate limit exceeded for '{key}'. "
            f"Limit: {limit}, retry after: {retry_after:.2f}s"
        )


class RateLimiterBackend(ABC):
    """Abstract base class for rate limiter backends.
    
    Implementations must provide concrete methods for:
    - Token bucket rate limiting
    - Sliding window rate limiting
    - Rate limit reset functionality
    """
    
    @abstractmethod
    def check_token_bucket(
        self,
        key: str,
        tokens: float,
        burst_size: int,
        refill_rate: float
    ) -> RateLimitResult:
        """Check rate limit using token bucket algorithm.
        
        Args:
            key: Unique identifier for the rate limit bucket
            tokens: Number of tokens to consume
            burst_size: Maximum number of tokens in the bucket
            refill_rate: Rate at which tokens are refilled (tokens per second)
            
        Returns:
            RateLimitResult indicating if request is allowed
        """
        pass
    
    @abstractmethod
    def check_sliding_window(
        self,
        key: str,
        limit: int,
        window_size: float
    ) -> RateLimitResult:
        """Check rate limit using sliding window algorithm.
        
        Args:
            key: Unique identifier for the rate limit window
            limit: Maximum number of requests in the window
            window_size: Size of the sliding window in seconds
            
        Returns:
            RateLimitResult indicating if request is allowed
        """
        pass
    
    @abstractmethod
    def reset(self, key: str) -> bool:
        """Reset rate limit for key.
        
        Args:
            key: Unique identifier to reset
            
        Returns:
            True if rate limit was reset, False if key was not found
        """
        pass


class LocalTokenBucket:
    """Local in-memory token bucket implementation."""
    
    def __init__(self, burst_size: int, refill_rate: float):
        self.burst_size = burst_size
        self.refill_rate = refill_rate
        self.tokens = float(burst_size)
        self.last_update = time.time()
        self._lock = threading.Lock()
    
    def consume(self, tokens: float = 1.0) -> Tuple[bool, float]:
        """
        Try to consume tokens from bucket.
        
        Returns:
            Tuple of (success, current_tokens)
        """
        with self._lock:
            now = time.time()
            elapsed = now - self.last_update
            
            # Refill tokens
            self.tokens = min(
                self.burst_size,
                self.tokens + elapsed * self.refill_rate
            )
            self.last_update = now
            
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True, self.tokens
            
            # Calculate retry after
            needed = tokens - self.tokens
            retry_after = needed / self.refill_rate
            return False, retry_after


class LocalSlidingWindow:
    """Local in-memory sliding window implementation."""
    
    def __init__(self, limit: int, window_size: float):
        self.limit = limit
        self.window_size = window_size
        self.requests: deque = deque()
        self._lock = threading.Lock()
    
    def add_request(self) -> Tuple[bool, int, float]:
        """
        Add request to window.
        
        Returns:
            Tuple of (allowed, remaining, reset_time)
        """
        with self._lock:
            now = time.time()
            window_start = now - self.window_size
            
            # Remove old requests outside window
            while self.requests and self.requests[0] <= window_start:
                self.requests.popleft()
            
            if len(self.requests) < self.limit:
                self.requests.append(now)
                remaining = self.limit - len(self.requests)
                reset_time = self.requests[0] + self.window_size if self.requests else now + self.window_size
                return True, remaining, reset_time
            
            reset_time = self.requests[0] + self.window_size
            retry_after = reset_time - now
            return False, 0, retry_after


class LocalRateLimiterBackend(RateLimiterBackend):
    """Local in-memory rate limiter backend."""
    
    def __init__(self):
        self._token_buckets: Dict[str, LocalTokenBucket] = {}
        self._sliding_windows: Dict[str, LocalSlidingWindow] = {}
        self._lock = threading.Lock()
        self._last_cleanup = time.time()
    
    def _cleanup_if_needed(self):
        """Periodically cleanup expired entries."""
        now = time.time()
        if now - self._last_cleanup > 300:  # Cleanup every 5 minutes
            with self._lock:
                # Cleanup empty sliding windows
                for key in list(self._sliding_windows.keys()):
                    window = self._sliding_windows[key]
                    window_start = now - window.window_size
                    while window.requests and window.requests[0] <= window_start:
                        window.requests.popleft()
                    if not window.requests:
                        del self._sliding_windows[key]
            self._last_cleanup = now
    
    def check_token_bucket(
        self,
        key: str,
        tokens: float,
        burst_size: int,
        refill_rate: float
    ) -> RateLimitResult:
        """Check rate limit using token bucket algorithm."""
        self._cleanup_if_needed()
        
        with self._lock:
            if key not in self._token_buckets:
                self._token_buckets[key] = LocalTokenBucket(burst_size, refill_rate)
        
        bucket = self._token_buckets[key]
        allowed, remaining = bucket.consume(tokens)
        
        return RateLimitResult(
            allowed=allowed,
            limit=burst_size,
            remaining=int(remaining) if allowed else 0,
            reset_time=bucket.last_update + (burst_size - remaining) / refill_rate,
            retry_after=remaining if not allowed else None
        )
    
    def check_sliding_window(
        self,
        key: str,
        limit: int,
        window_size: float
    ) -> RateLimitResult:
        """Check rate limit using sliding window algorithm."""
        self._cleanup_if_needed()
        
        with self._lock:
            if key not in self._sliding_windows:
                self._sliding_windows[key] = LocalSlidingWindow(limit, window_size)
        
        window = self._sliding_windows[key]
        allowed, remaining, reset_time = window.add_request()
        
        return RateLimitResult(
            allowed=allowed,
            limit=limit,
            remaining=remaining,
            reset_time=reset_time,
            retry_after=reset_time - time.time() if not allowed else None,
            window_start=window.requests[0] if window.requests else None
        )
    
    def reset(self, key: str) -> bool:
        """Reset rate limit for key."""
        with self._lock:
            found = False
            if key in self._token_buckets:
                del self._token_buckets[key]
                found = True
            if key in self._sliding_windows:
                del self._sliding_windows[key]
                found = True
            return found


class RedisRateLimiterBackend(RateLimiterBackend):
    """Redis-based distributed rate limiter backend."""
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        key_prefix: str = "nexus:rl"
    ):
        self.key_prefix = key_prefix
        
        try:
            import redis
            self._redis = redis.Redis(
                host=host,
                port=port,
                db=db,
                password=password,
                decode_responses=True
            )
            self._redis.ping()
            logger.info(f"Connected to Redis at {host}:{port}")
        except ImportError:
            raise ImportError("redis package required for RedisRateLimiterBackend")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise
    
    def _make_key(self, key: str) -> str:
        """Create Redis key with prefix."""
        return f"{self.key_prefix}:{key}"
    
    def check_token_bucket(
        self,
        key: str,
        tokens: float,
        burst_size: int,
        refill_rate: float
    ) -> RateLimitResult:
        """Check rate limit using token bucket algorithm with Redis."""
        redis_key = self._make_key(f"tb:{key}")
        now = time.time()
        
        pipe = self._redis.pipeline()
        
        # Lua script for atomic token bucket operation
        script = """
        local key = KEYS[1]
        local tokens = tonumber(ARGV[1])
        local burst_size = tonumber(ARGV[2])
        local refill_rate = tonumber(ARGV[3])
        local now = tonumber(ARGV[4])
        local ttl = math.ceil(burst_size / refill_rate) + 1
        
        local bucket = redis.call('HMGET', key, 'tokens', 'last_update')
        local current_tokens = tonumber(bucket[1]) or burst_size
        local last_update = tonumber(bucket[2]) or now
        
        -- Refill tokens
        local elapsed = now - last_update
        current_tokens = math.min(burst_size, current_tokens + elapsed * refill_rate)
        
        local allowed = 0
        if current_tokens >= tokens then
            current_tokens = current_tokens - tokens
            allowed = 1
        end
        
        redis.call('HMSET', key, 'tokens', current_tokens, 'last_update', now)
        redis.call('EXPIRE', key, ttl)
        
        return {allowed, current_tokens}
        """
        
        try:
            result = self._redis.eval(
                script, 1, redis_key, tokens, burst_size, refill_rate, now
            )
            allowed = result[0] == 1
            remaining = result[1]
            
            return RateLimitResult(
                allowed=allowed,
                limit=burst_size,
                remaining=int(remaining),
                reset_time=now + (burst_size - remaining) / refill_rate,
                retry_after=(tokens - remaining) / refill_rate if not allowed else None
            )
        except Exception as e:
            logger.error(f"Redis token bucket error: {e}")
            # Fallback to allow request
            return RateLimitResult(allowed=True, limit=burst_size, remaining=1, reset_time=now)
    
    def check_sliding_window(
        self,
        key: str,
        limit: int,
        window_size: float
    ) -> RateLimitResult:
        """Check rate limit using sliding window algorithm with Redis."""
        redis_key = self._make_key(f"sw:{key}")
        now = time.time()
        window_start = now - window_size
        
        pipe = self._redis.pipeline()
        
        # Lua script for atomic sliding window operation
        script = """
        local key = KEYS[1]
        local limit = tonumber(ARGV[1])
        local window_size = tonumber(ARGV[2])
        local now = tonumber(ARGV[3])
        local window_start = tonumber(ARGV[4])
        
        -- Remove old entries
        redis.call('ZREMRANGEBYSCORE', key, 0, window_start)
        
        -- Count current entries
        local count = redis.call('ZCARD', key)
        
        local allowed = 0
        if count < limit then
            redis.call('ZADD', key, now, now)
            allowed = 1
        end
        
        redis.call('EXPIRE', key, math.ceil(window_size))
        
        -- Get oldest entry for reset time
        local oldest = redis.call('ZRANGE', key, 0, 0, 'WITHSCORES')
        local reset_time = oldest[2] or (now + window_size)
        
        return {allowed, limit - count - allowed, reset_time}
        """
        
        try:
            result = self._redis.eval(
                script, 1, redis_key, limit, window_size, now, window_start
            )
            allowed = result[0] == 1
            remaining = result[1]
            reset_time = result[2]
            
            return RateLimitResult(
                allowed=allowed,
                limit=limit,
                remaining=remaining,
                reset_time=reset_time,
                retry_after=reset_time - now if not allowed else None,
                window_start=window_start
            )
        except Exception as e:
            logger.error(f"Redis sliding window error: {e}")
            return RateLimitResult(allowed=True, limit=limit, remaining=1, reset_time=now)
    
    def reset(self, key: str) -> bool:
        """Reset rate limit for key."""
        tb_key = self._make_key(f"tb:{key}")
        sw_key = self._make_key(f"sw:{key}")
        
        result = self._redis.delete(tb_key, sw_key)
        return result > 0


class RateLimiter:
    """Rate limiter with support for multiple algorithms and backends."""
    
    def __init__(
        self,
        backend: Optional[RateLimiterBackend] = None,
        config: Optional[RateLimitConfig] = None,
        algorithm: str = "token_bucket"
    ):
        self.backend = backend or LocalRateLimiterBackend()
        self.config = config or RateLimitConfig()
        self.algorithm = algorithm
        self._async_lock = asyncio.Lock()
    
    def _make_key(
        self,
        user_id: Optional[str] = None,
        action: Optional[str] = None,
        resource: Optional[str] = None
    ) -> str:
        """Create rate limit key from components."""
        parts = [self.config.key_prefix]
        if action:
            parts.append(action)
        if resource:
            parts.append(resource)
        if user_id:
            parts.append(user_id)
        
        key = ":".join(parts)
        # Hash long keys
        if len(key) > 100:
            key = hashlib.md5(key.encode()).hexdigest()
        return key
    
    def is_allowed(
        self,
        user_id: Optional[str] = None,
        action: Optional[str] = None,
        resource: Optional[str] = None,
        tokens: float = 1.0
    ) -> RateLimitResult:
        """
        Check if request is allowed under rate limit.
        
        Args:
            user_id: User identifier
            action: Action being performed
            resource: Resource being accessed
            tokens: Number of tokens to consume
            
        Returns:
            RateLimitResult with allowance status
        """
        key = self._make_key(user_id, action, resource)
        
        if self.algorithm == "token_bucket":
            return self.backend.check_token_bucket(
                key,
                tokens,
                self.config.burst_size,
                self.config.requests_per_second
            )
        elif self.algorithm == "sliding_window":
            return self.backend.check_sliding_window(
                key,
                int(self.config.requests_per_second * self.config.window_size),
                self.config.window_size
            )
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")
    
    async def is_allowed_async(
        self,
        user_id: Optional[str] = None,
        action: Optional[str] = None,
        resource: Optional[str] = None,
        tokens: float = 1.0
    ) -> RateLimitResult:
        """Async version of is_allowed."""
        async with self._async_lock:
            return self.is_allowed(user_id, action, resource, tokens)
    
    def check(
        self,
        user_id: Optional[str] = None,
        action: Optional[str] = None,
        resource: Optional[str] = None,
        tokens: float = 1.0
    ) -> None:
        """
        Check rate limit and raise exception if exceeded.
        
        Raises:
            RateLimitExceeded: If rate limit is exceeded
        """
        result = self.is_allowed(user_id, action, resource, tokens)
        
        if not result.allowed:
            key = self._make_key(user_id, action, resource)
            raise RateLimitExceeded(
                key=key,
                limit=result.limit,
                retry_after=result.retry_after or 1.0
            )
    
    async def check_async(
        self,
        user_id: Optional[str] = None,
        action: Optional[str] = None,
        resource: Optional[str] = None,
        tokens: float = 1.0
    ) -> None:
        """Async version of check."""
        result = await self.is_allowed_async(user_id, action, resource, tokens)
        
        if not result.allowed:
            key = self._make_key(user_id, action, resource)
            raise RateLimitExceeded(
                key=key,
                limit=result.limit,
                retry_after=result.retry_after or 1.0
            )
    
    def reset(
        self,
        user_id: Optional[str] = None,
        action: Optional[str] = None,
        resource: Optional[str] = None
    ) -> bool:
        """Reset rate limit for given key components."""
        key = self._make_key(user_id, action, resource)
        return self.backend.reset(key)


class RateLimiterRegistry:
    """Registry for managing multiple rate limiters."""
    
    def __init__(self):
        self._limiters: Dict[str, RateLimiter] = {}
        self._lock = threading.Lock()
    
    def register(
        self,
        name: str,
        backend: Optional[RateLimiterBackend] = None,
        config: Optional[RateLimitConfig] = None,
        algorithm: str = "token_bucket"
    ) -> RateLimiter:
        """Register a new rate limiter."""
        with self._lock:
            if name in self._limiters:
                raise ValueError(f"Rate limiter '{name}' already registered")
            
            limiter = RateLimiter(backend, config, algorithm)
            self._limiters[name] = limiter
            return limiter
    
    def get(self, name: str) -> Optional[RateLimiter]:
        """Get rate limiter by name."""
        with self._lock:
            return self._limiters.get(name)
    
    def remove(self, name: str) -> bool:
        """Remove rate limiter from registry."""
        with self._lock:
            if name in self._limiters:
                del self._limiters[name]
                return True
            return False


# Global registry
_global_registry = RateLimiterRegistry()


def get_rate_limiter_registry() -> RateLimiterRegistry:
    """Get the global rate limiter registry."""
    return _global_registry


def rate_limit(
    limiter_name: str,
    user_id_arg: Optional[str] = None,
    action: Optional[str] = None,
    resource: Optional[str] = None,
    tokens: float = 1.0
):
    """
    Decorator for applying rate limiting.
    
    Args:
        limiter_name: Name of registered rate limiter
        user_id_arg: Function argument name containing user ID
        action: Action identifier
        resource: Resource identifier
        tokens: Tokens to consume per call
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            registry = get_rate_limiter_registry()
            limiter = registry.get(limiter_name)
            
            if limiter is None:
                logger.warning(f"Rate limiter '{limiter_name}' not found, skipping")
                return func(*args, **kwargs)
            
            # Extract user_id from arguments if specified
            user_id = None
            if user_id_arg:
                if user_id_arg in kwargs:
                    user_id = kwargs[user_id_arg]
                elif args:
                    # Try to get from positional args
                    import inspect
                    sig = inspect.signature(func)
                    params = list(sig.parameters.keys())
                    if user_id_arg in params:
                        idx = params.index(user_id_arg)
                        if idx < len(args):
                            user_id = args[idx]
            
            limiter.check(user_id, action, resource, tokens)
            return func(*args, **kwargs)
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            registry = get_rate_limiter_registry()
            limiter = registry.get(limiter_name)
            
            if limiter is None:
                logger.warning(f"Rate limiter '{limiter_name}' not found, skipping")
                if asyncio.iscoroutinefunction(func):
                    return await func(*args, **kwargs)
                return func(*args, **kwargs)
            
            # Extract user_id from arguments if specified
            user_id = None
            if user_id_arg:
                if user_id_arg in kwargs:
                    user_id = kwargs[user_id_arg]
                elif args:
                    import inspect
                    sig = inspect.signature(func)
                    params = list(sig.parameters.keys())
                    if user_id_arg in params:
                        idx = params.index(user_id_arg)
                        if idx < len(args):
                            user_id = args[idx]
            
            await limiter.check_async(user_id, action, resource, tokens)
            
            if asyncio.iscoroutinefunction(func):
                return await func(*args, **kwargs)
            return func(*args, **kwargs)
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else wrapper
    return decorator


# Pre-configured rate limiters

# API endpoints - 100 requests per minute per user
API_RATE_LIMIT = RateLimitConfig(
    requests_per_second=100/60,
    burst_size=20,
    window_size=60.0,
    key_prefix="nexus:api"
)

# Model inference - 10 requests per second per user
INFERENCE_RATE_LIMIT = RateLimitConfig(
    requests_per_second=10.0,
    burst_size=5,
    window_size=60.0,
    key_prefix="nexus:inference"
)

# Global limits - 1000 requests per second total
GLOBAL_RATE_LIMIT = RateLimitConfig(
    requests_per_second=1000.0,
    burst_size=100,
    window_size=60.0,
    key_prefix="nexus:global"
)

# Login attempts - 5 per minute per IP
LOGIN_RATE_LIMIT = RateLimitConfig(
    requests_per_second=5/60,
    burst_size=5,
    window_size=300.0,  # 5 minute window
    block_duration=300.0,  # 5 minute block
    key_prefix="nexus:login"
)