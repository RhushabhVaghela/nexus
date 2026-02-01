"""
Rate Limiter Module for Nexus API

Implements token bucket algorithm for rate limiting with support
for per-user and per-endpoint limits.
"""

import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
from typing import Callable, Dict, Optional, Set, Tuple, Any, List
import logging

logger = logging.getLogger(__name__)


class RateLimitStrategy(Enum):
    """Rate limiting strategies."""
    FIXED_WINDOW = "fixed_window"
    SLIDING_WINDOW = "sliding_window"
    TOKEN_BUCKET = "token_bucket"


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""
    requests_per_second: float = 10.0
    burst_size: int = 20
    window_size: int = 60  # seconds for fixed/sliding window
    strategy: RateLimitStrategy = RateLimitStrategy.TOKEN_BUCKET
    block_duration: int = 300  # seconds to block if limit exceeded
    key_prefix: str = ""
    
    def __post_init__(self):
        if self.requests_per_second <= 0:
            raise ValueError("requests_per_second must be positive")
        if self.burst_size <= 0:
            raise ValueError("burst_size must be positive")


@dataclass
class TokenBucket:
    """Token bucket for rate limiting."""
    capacity: int
    refill_rate: float  # tokens per second
    tokens: float = field(default=0.0)
    last_refill: float = field(default_factory=time.time)
    lock: threading.Lock = field(default_factory=threading.Lock)
    
    def consume(self, tokens: int = 1) -> Tuple[bool, float]:
        """
        Attempt to consume tokens from the bucket.
        
        Args:
            tokens: Number of tokens to consume
            
        Returns:
            Tuple of (success, retry_after_seconds)
        """
        with self.lock:
            now = time.time()
            # Refill tokens based on time elapsed
            elapsed = now - self.last_refill
            self.tokens = min(
                self.capacity,
                self.tokens + elapsed * self.refill_rate
            )
            self.last_refill = now
            
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True, 0.0
            
            # Calculate wait time
            tokens_needed = tokens - self.tokens
            wait_time = tokens_needed / self.refill_rate
            return False, wait_time
    
    def reset(self):
        """Reset the bucket to full capacity."""
        with self.lock:
            self.tokens = self.capacity
            self.last_refill = time.time()


@dataclass
class FixedWindow:
    """Fixed window rate limiter."""
    window_size: int  # seconds
    max_requests: int
    window_start: float = field(default_factory=time.time)
    count: int = field(default=0)
    lock: threading.Lock = field(default_factory=threading.Lock)
    
    def consume(self, tokens: int = 1) -> Tuple[bool, float]:
        """
        Attempt to consume a request.
        
        Args:
            tokens: Number of requests (usually 1)
            
        Returns:
            Tuple of (success, retry_after_seconds)
        """
        with self.lock:
            now = time.time()
            
            # Check if we need to reset the window
            if now - self.window_start >= self.window_size:
                self.window_start = now
                self.count = 0
            
            if self.count + tokens <= self.max_requests:
                self.count += tokens
                return True, 0.0
            
            # Calculate retry after
            retry_after = self.window_size - (now - self.window_start)
            return False, retry_after
    
    def reset(self):
        """Reset the window."""
        with self.lock:
            self.window_start = time.time()
            self.count = 0


@dataclass
class SlidingWindow:
    """Sliding window rate limiter."""
    window_size: int  # seconds
    max_requests: int
    requests: List[float] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)
    
    def consume(self, tokens: int = 1) -> Tuple[bool, float]:
        """
        Attempt to consume a request.
        
        Args:
            tokens: Number of requests (usually 1)
            
        Returns:
            Tuple of (success, retry_after_seconds)
        """
        with self.lock:
            now = time.time()
            window_start = now - self.window_size
            
            # Remove requests outside the window
            self.requests = [t for t in self.requests if t > window_start]
            
            if len(self.requests) + tokens <= self.max_requests:
                self.requests.extend([now] * tokens)
                return True, 0.0
            
            # Calculate retry after based on oldest request
            if self.requests:
                retry_after = self.window_size - (now - self.requests[0])
            else:
                retry_after = self.window_size
            return False, retry_after
    
    def reset(self):
        """Reset the sliding window."""
        with self.lock:
            self.requests.clear()


class RateLimiter:
    """
    Rate limiter supporting multiple strategies and per-key limits.
    """
    
    def __init__(self, default_config: Optional[RateLimitConfig] = None):
        """
        Initialize the rate limiter.
        
        Args:
            default_config: Default rate limit configuration
        """
        self._default_config = default_config or RateLimitConfig()
        self._buckets: Dict[str, TokenBucket] = {}
        self._fixed_windows: Dict[str, FixedWindow] = {}
        self._sliding_windows: Dict[str, SlidingWindow] = {}
        self._blocked_keys: Dict[str, float] = {}  # key -> unblock_time
        self._configs: Dict[str, RateLimitConfig] = {}
        self._lock = threading.RLock()
        self._cleanup_interval = 300  # seconds
        self._last_cleanup = time.time()
        
        # Start background cleanup thread
        self._cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self._cleanup_thread.start()
    
    def configure_key(self, key: str, config: RateLimitConfig):
        """
        Configure rate limiting for a specific key.
        
        Args:
            key: Rate limit key (e.g., user ID, endpoint)
            config: Rate limit configuration
        """
        with self._lock:
            self._configs[key] = config
            # Clear existing limiters for this key to apply new config
            self._buckets.pop(key, None)
            self._fixed_windows.pop(key, None)
            self._sliding_windows.pop(key, None)
    
    def is_allowed(
        self,
        key: str,
        tokens: int = 1,
        config: Optional[RateLimitConfig] = None
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Check if a request is allowed.
        
        Args:
            key: Rate limit key
            tokens: Number of tokens/requests to consume
            config: Optional config override
            
        Returns:
            Tuple of (is_allowed, metadata)
            metadata contains: retry_after, remaining, reset_time
        """
        self._cleanup_if_needed()
        
        with self._lock:
            # Check if key is blocked
            unblock_time = self._blocked_keys.get(key)
            if unblock_time and time.time() < unblock_time:
                return False, {
                    "retry_after": int(unblock_time - time.time()),
                    "remaining": 0,
                    "reset_time": int(unblock_time)
                }
            elif unblock_time:
                del self._blocked_keys[key]
            
            # Get configuration
            cfg = config or self._configs.get(key) or self._default_config
            
            # Get or create limiter
            allowed, retry_after = self._consume(key, tokens, cfg)
            
            if not allowed:
                # Block the key temporarily
                block_until = time.time() + cfg.block_duration
                self._blocked_keys[key] = block_until
                
                return False, {
                    "retry_after": int(retry_after),
                    "remaining": 0,
                    "reset_time": int(time.time() + retry_after)
                }
            
            # Calculate remaining and reset
            remaining = self._get_remaining(key, cfg)
            reset_time = self._get_reset_time(key, cfg)
            
            return True, {
                "retry_after": 0,
                "remaining": remaining,
                "reset_time": reset_time
            }
    
    def _consume(self, key: str, tokens: int, config: RateLimitConfig) -> Tuple[bool, float]:
        """Consume tokens using the configured strategy."""
        if config.strategy == RateLimitStrategy.TOKEN_BUCKET:
            bucket = self._buckets.get(key)
            if bucket is None:
                bucket = TokenBucket(
                    capacity=config.burst_size,
                    refill_rate=config.requests_per_second
                )
                self._buckets[key] = bucket
            return bucket.consume(tokens)
        
        elif config.strategy == RateLimitStrategy.FIXED_WINDOW:
            window = self._fixed_windows.get(key)
            if window is None:
                window = FixedWindow(
                    window_size=config.window_size,
                    max_requests=int(config.requests_per_second * config.window_size)
                )
                self._fixed_windows[key] = window
            return window.consume(tokens)
        
        elif config.strategy == RateLimitStrategy.SLIDING_WINDOW:
            sw = self._sliding_windows.get(key)
            if sw is None:
                sw = SlidingWindow(
                    window_size=config.window_size,
                    max_requests=int(config.requests_per_second * config.window_size)
                )
                self._sliding_windows[key] = sw
            return sw.consume(tokens)
        
        else:
            raise ValueError(f"Unknown strategy: {config.strategy}")
    
    def _get_remaining(self, key: str, config: RateLimitConfig) -> int:
        """Get remaining requests for a key."""
        if config.strategy == RateLimitStrategy.TOKEN_BUCKET:
            bucket = self._buckets.get(key)
            if bucket:
                # Trigger a refill
                bucket.consume(0)
                return int(bucket.tokens)
            return config.burst_size
        
        elif config.strategy == RateLimitStrategy.FIXED_WINDOW:
            window = self._fixed_windows.get(key)
            if window:
                return max(0, window.max_requests - window.count)
            return int(config.requests_per_second * config.window_size)
        
        elif config.strategy == RateLimitStrategy.SLIDING_WINDOW:
            sw = self._sliding_windows.get(key)
            if sw:
                # Clean and count
                now = time.time()
                window_start = now - sw.window_size
                sw.requests = [t for t in sw.requests if t > window_start]
                return max(0, sw.max_requests - len(sw.requests))
            return int(config.requests_per_second * config.window_size)
        
        return 0
    
    def _get_reset_time(self, key: str, config: RateLimitConfig) -> int:
        """Get the reset time for a key."""
        if config.strategy == RateLimitStrategy.TOKEN_BUCKET:
            bucket = self._buckets.get(key)
            if bucket:
                tokens_needed = bucket.capacity - bucket.tokens
                if tokens_needed > 0:
                    return int(time.time() + tokens_needed / bucket.refill_rate)
            return int(time.time())
        
        elif config.strategy == RateLimitStrategy.FIXED_WINDOW:
            window = self._fixed_windows.get(key)
            if window:
                return int(window.window_start + window.window_size)
            return int(time.time() + config.window_size)
        
        elif config.strategy == RateLimitStrategy.SLIDING_WINDOW:
            sw = self._sliding_windows.get(key)
            if sw and sw.requests:
                return int(sw.requests[0] + sw.window_size)
            return int(time.time() + config.window_size)
        
        return int(time.time())
    
    def reset_key(self, key: str):
        """Reset rate limiting for a key."""
        with self._lock:
            bucket = self._buckets.get(key)
            if bucket:
                bucket.reset()
            
            window = self._fixed_windows.get(key)
            if window:
                window.reset()
            
            sw = self._sliding_windows.get(key)
            if sw:
                sw.reset()
            
            self._blocked_keys.pop(key, None)
    
    def block_key(self, key: str, duration: int):
        """
        Manually block a key.
        
        Args:
            key: Key to block
            duration: Block duration in seconds
        """
        with self._lock:
            self._blocked_keys[key] = time.time() + duration
    
    def unblock_key(self, key: str):
        """Unblock a key."""
        with self._lock:
            self._blocked_keys.pop(key, None)
    
    def get_stats(self, key: Optional[str] = None) -> Dict[str, Any]:
        """
        Get rate limiter statistics.
        
        Args:
            key: Optional specific key to get stats for
            
        Returns:
            Dictionary of statistics
        """
        with self._lock:
            if key:
                cfg = self._configs.get(key) or self._default_config
                return {
                    "key": key,
                    "remaining": self._get_remaining(key, cfg),
                    "reset_time": self._get_reset_time(key, cfg),
                    "is_blocked": key in self._blocked_keys
                }
            
            return {
                "total_buckets": len(self._buckets),
                "total_fixed_windows": len(self._fixed_windows),
                "total_sliding_windows": len(self._sliding_windows),
                "blocked_keys": len(self._blocked_keys),
                "configured_keys": len(self._configs)
            }
    
    def _cleanup_if_needed(self):
        """Run cleanup if interval has passed."""
        if time.time() - self._last_cleanup > self._cleanup_interval:
            self._cleanup()
    
    def _cleanup(self):
        """Clean up expired entries."""
        with self._lock:
            now = time.time()
            
            # Clean blocked keys
            expired = [k for k, t in self._blocked_keys.items() if now > t]
            for k in expired:
                del self._blocked_keys[k]
            
            self._last_cleanup = now
    
    def _cleanup_loop(self):
        """Background cleanup thread."""
        while True:
            time.sleep(self._cleanup_interval)
            try:
                self._cleanup()
            except Exception as e:
                logger.error(f"Error in rate limiter cleanup: {e}")


def rate_limit(
    limiter: RateLimiter,
    key_func: Callable[..., str],
    config: Optional[RateLimitConfig] = None
) -> Callable:
    """
    Decorator to apply rate limiting to a function.
    
    Args:
        limiter: RateLimiter instance
        key_func: Function to extract rate limit key from arguments
        config: Optional RateLimitConfig override
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            key = key_func(*args, **kwargs)
            allowed, metadata = limiter.is_allowed(key, config=config)
            
            if not allowed:
                raise RateLimitExceeded(
                    f"Rate limit exceeded. Retry after {metadata['retry_after']} seconds."
                )
            
            # Add rate limit headers/info to result if it's a dict
            result = func(*args, **kwargs)
            
            return result
        
        return wrapper
    return decorator


class RateLimitExceeded(Exception):
    """Exception raised when rate limit is exceeded."""
    pass


class PerUserRateLimiter:
    """Rate limiter with per-user default configuration."""
    
    def __init__(
        self,
        default_requests_per_second: float = 10.0,
        default_burst_size: int = 20
    ):
        """
        Initialize per-user rate limiter.
        
        Args:
            default_requests_per_second: Default rate limit
            default_burst_size: Default burst capacity
        """
        default_config = RateLimitConfig(
            requests_per_second=default_requests_per_second,
            burst_size=default_burst_size
        )
        self._limiter = RateLimiter(default_config)
        self._default_config = default_config
    
    def is_allowed(
        self,
        user_id: str,
        endpoint: Optional[str] = None,
        tokens: int = 1
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Check if request is allowed for user.
        
        Args:
            user_id: User identifier
            endpoint: Optional endpoint for per-endpoint limiting
            tokens: Number of tokens to consume
            
        Returns:
            Tuple of (is_allowed, metadata)
        """
        key = f"user:{user_id}"
        if endpoint:
            key = f"{key}:endpoint:{endpoint}"
        
        return self._limiter.is_allowed(key, tokens)
    
    def configure_user(
        self,
        user_id: str,
        requests_per_second: float,
        burst_size: int,
        endpoint: Optional[str] = None
    ):
        """
        Configure rate limiting for a user.
        
        Args:
            user_id: User identifier
            requests_per_second: Rate limit
            burst_size: Burst capacity
            endpoint: Optional specific endpoint
        """
        key = f"user:{user_id}"
        if endpoint:
            key = f"{key}:endpoint:{endpoint}"
        
        config = RateLimitConfig(
            requests_per_second=requests_per_second,
            burst_size=burst_size
        )
        self._limiter.configure_key(key, config)
    
    def get_limiter(self) -> RateLimiter:
        """Get underlying rate limiter."""
        return self._limiter


class PerEndpointRateLimiter:
    """Rate limiter with per-endpoint configuration."""
    
    def __init__(
        self,
        default_requests_per_second: float = 100.0,
        default_burst_size: int = 200
    ):
        """
        Initialize per-endpoint rate limiter.
        
        Args:
            default_requests_per_second: Default rate limit
            default_burst_size: Default burst capacity
        """
        default_config = RateLimitConfig(
            requests_per_second=default_requests_per_second,
            burst_size=default_burst_size
        )
        self._limiter = RateLimiter(default_config)
        self._default_config = default_config
    
    def is_allowed(
        self,
        endpoint: str,
        user_id: Optional[str] = None,
        tokens: int = 1
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Check if request is allowed for endpoint.
        
        Args:
            endpoint: Endpoint identifier
            user_id: Optional user for combined limiting
            tokens: Number of tokens to consume
            
        Returns:
            Tuple of (is_allowed, metadata)
        """
        key = f"endpoint:{endpoint}"
        if user_id:
            key = f"user:{user_id}:{key}"
        
        return self._limiter.is_allowed(key, tokens)
    
    def configure_endpoint(
        self,
        endpoint: str,
        requests_per_second: float,
        burst_size: int
    ):
        """
        Configure rate limiting for an endpoint.
        
        Args:
            endpoint: Endpoint identifier
            requests_per_second: Rate limit
            burst_size: Burst capacity
        """
        key = f"endpoint:{endpoint}"
        config = RateLimitConfig(
            requests_per_second=requests_per_second,
            burst_size=burst_size
        )
        self._limiter.configure_key(key, config)
    
    def get_limiter(self) -> RateLimiter:
        """Get underlying rate limiter."""
        return self._limiter


# Global rate limiter instance
_global_rate_limiter: Optional[RateLimiter] = None


def get_rate_limiter() -> RateLimiter:
    """Get or create global rate limiter instance."""
    global _global_rate_limiter
    if _global_rate_limiter is None:
        _global_rate_limiter = RateLimiter()
    return _global_rate_limiter


def set_rate_limiter(limiter: RateLimiter) -> None:
    """Set global rate limiter instance."""
    global _global_rate_limiter
    _global_rate_limiter = limiter
