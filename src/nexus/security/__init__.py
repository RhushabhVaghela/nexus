"""
Security module for Nexus.

Provides security audit, input validation, content safety filtering,
authentication (API keys + JWT), and rate limiting.
"""

from .audit import (
    SecurityAuditor,
    InputValidator,
    ContentFilter,
    InjectionDetector,
    get_security_auditor,
    security_check,
)

from .auth import (
    Permission,
    APIKey,
    JWTClaims,
    APIKeyManager,
    JWTTokenManager,
    Authenticator,
    require_auth,
    rate_limit_key,
    get_authenticator,
    set_authenticator,
)

from .rate_limiter import (
    RateLimitStrategy,
    RateLimitConfig,
    RateLimiter,
    RateLimitExceeded,
    PerUserRateLimiter,
    PerEndpointRateLimiter,
    rate_limit,
    get_rate_limiter,
    set_rate_limiter,
)

__all__ = [
    # audit
    "SecurityAuditor",
    "InputValidator",
    "ContentFilter",
    "InjectionDetector",
    "get_security_auditor",
    "security_check",
    # auth
    "Permission",
    "APIKey",
    "JWTClaims",
    "APIKeyManager",
    "JWTTokenManager",
    "Authenticator",
    "require_auth",
    "rate_limit_key",
    "get_authenticator",
    "set_authenticator",
    # rate_limiter
    "RateLimitStrategy",
    "RateLimitConfig",
    "RateLimiter",
    "RateLimitExceeded",
    "PerUserRateLimiter",
    "PerEndpointRateLimiter",
    "rate_limit",
    "get_rate_limiter",
    "set_rate_limiter",
]
