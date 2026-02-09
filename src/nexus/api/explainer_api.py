"""
Nexus Universal Explainer API

Secure API endpoints for the Nexus Universal Explainer Engine.
Implements comprehensive security including authentication, rate limiting,
input validation, and security headers.

Security Features:
- JWT/API Key Authentication
- Rate Limiting (Token Bucket Algorithm)
- Input Validation & Injection Detection
- Security Headers (HSTS, X-Frame-Options, etc.)
- Comprehensive Security Logging
- CORS with Restricted Origins
"""

import os
import logging
import time
import uuid
from typing import Any, Dict, Optional, Tuple
from datetime import datetime

try:
    from fastapi import (
        FastAPI,
        HTTPException,
        Request,
        Security,
        Header,
        Depends,
        Query,
    )

    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
try:
    from fastapi.middleware.cors import CORSMiddleware

    CORS_AVAILABLE = True
except ImportError:
    CORS_AVAILABLE = False
try:
    from pydantic import BaseModel, Field

    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
try:
    from starlette.responses import Response

    STARLETTE_AVAILABLE = True
except ImportError:
    STARLETTE_AVAILABLE = False
try:
    from starlette.middleware.base import BaseHTTPMiddleware

    BASE_MIDDLEWARE_AVAILABLE = True
except ImportError:
    BASE_MIDDLEWARE_AVAILABLE = False

# Module-level guard: this entire module requires fastapi, pydantic, and starlette
_missing = []
if not FASTAPI_AVAILABLE:
    _missing.append("fastapi")
if not PYDANTIC_AVAILABLE:
    _missing.append("pydantic")
if not STARLETTE_AVAILABLE:
    _missing.append("starlette")
if not CORS_AVAILABLE:
    _missing.append("fastapi (CORS middleware)")
if not BASE_MIDDLEWARE_AVAILABLE:
    _missing.append("starlette (BaseHTTPMiddleware)")
if _missing:
    raise ImportError(
        f"The explainer_api module requires {', '.join(_missing)}. "
        f"Install with: pip install {' '.join(_missing)}"
    )

from nexus.optimization.remotion_engine import RemotionExplainerEngine
from nexus.security.auth import get_authenticator, Authenticator, Permission, JWTClaims
from nexus.security.audit import (
    get_security_auditor,
    SecurityAuditor,
    SecurityReport,
    SecurityException,
)
from nexus.security.rate_limiter import get_rate_limiter, RateLimiter, RateLimitConfig

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='{"timestamp": "%(asctime)s", "level": "%(levelname)s", "logger": "%(name)s", "message": "%(message)s"}',
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ============================================================================
# SECURITY CONFIGURATION
# ============================================================================


# CORS Configuration - FIXED: Validate origins, never allow wildcard
def get_allowed_origins() -> list:
    """Get and validate allowed CORS origins."""
    env_origins = os.environ.get("ALLOWED_ORIGINS", "").strip()
    if not env_origins:
        # Default to localhost only for security
        return ["http://localhost:3000", "http://localhost:8080"]

    origins = [o.strip() for o in env_origins.split(",") if o.strip()]

    # Security: Never allow wildcard in production
    if "*" in origins:
        logger.warning("SECURITY: Wildcard CORS origin (*) detected and removed!")
        origins = [o for o in origins if o != "*"]

    return origins if origins else ["http://localhost:3000", "http://localhost:8080"]


ALLOWED_ORIGINS = get_allowed_origins()

# Rate Limiting Configuration
RATE_LIMIT_REQUESTS_PER_SECOND = float(os.environ.get("RATE_LIMIT_RPS", "10"))
RATE_LIMIT_BURST_SIZE = int(os.environ.get("RATE_LIMIT_BURST", "20"))


# Authentication Configuration - FIXED: Require JWT_SECRET in production
def get_jwt_secret() -> str:
    """Get JWT secret, requiring it in production."""
    secret = os.environ.get("JWT_SECRET")
    if not secret:
        # For development only - raise error in production
        if os.environ.get("NEXUS_ENV") == "production":
            raise RuntimeError(
                "SECURITY: JWT_SECRET must be set in production! "
                "Set NEXUS_ENV=development for auto-generated secrets (not recommended for production)."
            )
        # Generate temporary secret for development
        import secrets

        secret = secrets.token_hex(32)
        logger.warning(
            "SECURITY: Using auto-generated JWT secret. Set JWT_SECRET for production!"
        )
    return secret


JWT_SECRET = get_jwt_secret()
JWT_EXPIRY_HOURS = int(os.environ.get("JWT_EXPIRY_HOURS", "24"))

# Security Headers Configuration
SECURITY_HEADERS = {
    "X-Frame-Options": "DENY",
    "X-Content-Type-Options": "nosniff",
    "X-XSS-Protection": "1; mode=block",
    "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
    "Content-Security-Policy": "default-src 'self'",
    "Referrer-Policy": "strict-origin-when-cross-origin",
    "Permissions-Policy": "geolocation=(), microphone=(), camera=()",
}

# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================


class ExplainerRequest(BaseModel):
    """Validated request model for explainer generation."""

    prompt: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="The prompt for explainer generation",
        example="Explain how quantum computing works with visual animations",
    )
    model_path: str = Field(
        default=REMOTION_EXPLAINER_DIR,
        description="Path to the trained model",
    )
    narrate: bool = Field(default=False, description="Whether to include narration")

    class Config:
        json_schema_extra = {
            "example": {
                "prompt": "Explain how quantum computing works",
                "model_path": REMOTION_EXPLAINER_DIR,
                "narrate": False,
            }
        }


class ExplainerResponse(BaseModel):
    """Response model for explainer generation."""

    success: bool
    video_url: Optional[str] = None
    tsx_preview: Optional[str] = None
    request_id: str
    processing_time_ms: Optional[float] = None


class ErrorResponse(BaseModel):
    """Standard error response."""

    error: str
    detail: Optional[str] = None
    request_id: str
    timestamp: str


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    engine_ready: bool
    version: str = "1.0.0"


class TokenRequest(BaseModel):
    """Request model for JWT token generation."""

    user_id: str = Field(
        ..., min_length=1, max_length=256, description="Unique user identifier"
    )
    permissions: list = Field(
        default=["inference"],
        description="List of permission scopes (read, write, admin, inference, training)",
    )


class APIKeyRequest(BaseModel):
    """Request model for API key generation."""

    name: str = Field(
        default="Default Key",
        min_length=1,
        max_length=256,
        description="Human-readable name for the key",
    )
    permissions: list = Field(
        default=["inference"],
        description="List of permission scopes",
    )


# ============================================================================
# SECURITY MIDDLEWARE
# ============================================================================


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Middleware to add security headers to all responses."""

    def __init__(self, app, headers: Dict[str, str] = None):
        super().__init__(app)
        self.headers = headers or SECURITY_HEADERS

    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)

        # Add security headers
        for header_name, header_value in self.headers.items():
            response.headers[header_name] = header_value

        return response


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware to log all API requests."""

    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        request.state.request_id = request_id

        start_time = time.time()

        # Log request
        logger.info(
            {
                "event": "request_start",
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "client_ip": request.client.host if request.client else "unknown",
                "user_agent": request.headers.get("user-agent", "unknown"),
            }
        )

        try:
            response = await call_next(request)
            processing_time = (time.time() - start_time) * 1000

            # Log response
            logger.info(
                {
                    "event": "request_complete",
                    "request_id": request_id,
                    "status_code": response.status_code,
                    "processing_time_ms": round(processing_time, 2),
                }
            )

            # Add request ID to response
            response.headers["X-Request-ID"] = request_id

            return response

        except Exception as e:
            processing_time = (time.time() - start_time) * 1000

            logger.error(
                {
                    "event": "request_error",
                    "request_id": request_id,
                    "error": str(e),
                    "processing_time_ms": round(processing_time, 2),
                    "error_type": type(e).__name__,
                }
            )

            raise


# ============================================================================
# API INITIALIZATION
# ============================================================================

app = FastAPI(
    title="Nexus Universal Explainer API",
    description="Secure API for generating AI-powered video explanations with comprehensive security controls",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Global instances (lazy loaded)
_authenticator: Optional[Authenticator] = None
_rate_limiter: Optional[RateLimiter] = None
_security_auditor: Optional[SecurityAuditor] = None
_engine: Optional[RemotionExplainerEngine] = None


def get_authenticator_instance() -> Authenticator:
    """Get or create authenticator instance."""
    global _authenticator
    if _authenticator is None:
        _authenticator = Authenticator(secret_key=JWT_SECRET)
    return _authenticator


def get_rate_limiter_instance() -> RateLimiter:
    """Get or create rate limiter instance."""
    global _rate_limiter
    if _rate_limiter is None:
        config = RateLimitConfig(
            requests_per_second=RATE_LIMIT_REQUESTS_PER_SECOND,
            burst_size=RATE_LIMIT_BURST_SIZE,
        )
        _rate_limiter = RateLimiter(default_config=config)
    return _rate_limiter


def get_security_auditor_instance() -> SecurityAuditor:
    """Get or create security auditor instance."""
    global _security_auditor
    if _security_auditor is None:
        _security_auditor = get_security_auditor()
    return _security_auditor


# ============================================================================
# MIDDLEWARE SETUP
# ============================================================================

# 1. CORS Middleware (Restricted Origins)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-Request-ID"],
)

# 2. Security Headers Middleware
app.add_middleware(SecurityHeadersMiddleware)

# 3. Request Logging Middleware
app.add_middleware(RequestLoggingMiddleware)


# ============================================================================
# DEPENDENCIES
# ============================================================================


async def verify_authentication(
    authorization: Optional[str] = Header(None, alias="Authorization"),
) -> Dict[str, Any]:
    """
    Verify authentication and return auth context.

    Args:
        authorization: Authorization header value

    Returns:
        Authentication context dict

    Raises:
        HTTPException: If authentication fails
    """
    auth = get_authenticator_instance()

    is_valid, error, context = auth.authenticate_request(authorization)

    if not is_valid:
        logger.warning(
            {
                "event": "auth_failure",
                "error": error,
                "ip": "unknown",  # Would need to pass request for real IP
            }
        )
        raise HTTPException(status_code=401, detail=error)

    logger.info(
        {
            "event": "auth_success",
            "user_id": context.get("user_id") or context.get("key_id"),
            "auth_type": context.get("type"),
        }
    )

    return context


async def check_rate_limit(
    auth_context: Dict[str, Any] = Depends(verify_authentication),
    request: Request = None,  # noqa: FastAPI injects; None fallback for testing only
) -> Dict[str, Any]:
    """
    Check rate limit for the request.

    Args:
        auth_context: Authentication context from verify_authentication
        request: FastAPI request object

    Returns:
        Auth context if rate limit passed

    Raises:
        HTTPException: If rate limit exceeded
    """
    limiter = get_rate_limiter_instance()

    # Generate rate limit key based on auth type
    if auth_context.get("type") == "jwt":
        rate_key = f"jwt:{auth_context.get('user_id', 'unknown')}"
    else:
        rate_key = f"apikey:{auth_context.get('key_id', 'unknown')}"

    is_allowed, metadata = limiter.is_allowed(rate_key)

    if not is_allowed:
        logger.warning(
            {
                "event": "rate_limit_exceeded",
                "key": rate_key,
                "retry_after": metadata.get("retry_after"),
            }
        )

        raise HTTPException(
            status_code=429,
            detail={
                "error": "Rate limit exceeded",
                "retry_after": metadata.get("retry_after", 60),
            },
        )

    # Add rate limit headers to request state for response
    if request:
        request.state.rate_limit_remaining = metadata.get("remaining", 0)
        request.state.rate_limit_reset = metadata.get("reset_time", 0)

    return auth_context


def audit_input(prompt: str, request_id: str) -> SecurityReport:
    """
    Audit input for security violations.

    Args:
        prompt: Input prompt to audit
        request_id: Request identifier for logging

    Returns:
        SecurityReport with audit results

    Raises:
        HTTPException: If security violation detected
    """
    auditor = get_security_auditor_instance()

    try:
        report = auditor.audit_input(
            text=prompt,
            context="explainer_api.generate",
            metadata={
                "request_id": request_id,
                "timestamp": datetime.utcnow().isoformat(),
            },
        )
    except SecurityException as exc:
        # SecurityAuditor raises SecurityException when block_on_violation=True
        violation = exc.violation if hasattr(exc, "violation") else None
        violation_type = violation.type.value if violation else "unknown"
        logger.warning(
            {
                "event": "security_violation",
                "request_id": request_id,
                "violations": [violation_type],
            }
        )
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Security violation detected",
                "violations": [violation_type],
                "request_id": request_id,
            },
        )

    if not report.passed:
        violation_types = [v.type.value for v in report.violations]
        logger.warning(
            {
                "event": "security_violation",
                "request_id": request_id,
                "violations": violation_types,
            }
        )

        raise HTTPException(
            status_code=400,
            detail={
                "error": "Security violation detected",
                "violations": violation_types,
                "request_id": request_id,
            },
        )

    return report


# ============================================================================
# API ENDPOINTS
# ============================================================================


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    global _engine

    logger.info({"event": "startup", "message": "Initializing Nexus Explainer API"})

    # Initialize engine with a default model path if it exists
    default_model = REMOTION_EXPLAINER_DIR
    if os.path.exists(default_model):
        try:
            _engine = RemotionExplainerEngine(model_path=default_model)
            logger.info({"event": "engine_initialized", "model_path": default_model})
        except Exception as e:
            logger.warning({"event": "engine_init_failed", "error": str(e)})

    # Initialize security components
    _ = get_authenticator_instance()
    _ = get_rate_limiter_instance()
    _ = get_security_auditor_instance()

    logger.info(
        {
            "event": "startup_complete",
            "security_features": [
                "authentication",
                "rate_limiting",
                "input_validation",
                "security_headers",
                "request_logging",
            ],
        }
    )


@app.post(
    "/generate",
    response_model=ExplainerResponse,
    responses={
        400: {
            "model": ErrorResponse,
            "description": "Invalid request or security violation",
        },
        401: {"model": ErrorResponse, "description": "Authentication required"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
    summary="Generate Explainer Video",
    description="Generate an AI-powered video explanation based on the provided prompt",
)
async def generate_explanation(
    request: Request,
    payload: ExplainerRequest,
    auth_context: Dict[str, Any] = Depends(check_rate_limit),
) -> ExplainerResponse:
    """
    Generate a video explanation based on the provided prompt.

    Requires authentication via JWT token or API key.

    Security measures:
    - Authentication required
    - Rate limited (10 requests/second, 20 burst)
    - Input validated for injection attacks
    """
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))

    # Audit input for security violations
    audit_input(payload.prompt, request_id)

    start_time = time.time()
    global _engine

    logger.info(
        {
            "event": "generation_start",
            "request_id": request_id,
            "user_id": auth_context.get("user_id") or auth_context.get("key_id"),
            "narrate": payload.narrate,
        }
    )

    try:
        # Initialize engine if needed
        if _engine is None:
            try:
                _engine = RemotionExplainerEngine(model_path=payload.model_path)
            except Exception as e:
                logger.error(
                    {
                        "event": "engine_init_error",
                        "request_id": request_id,
                        "error": str(e),
                    }
                )
                raise HTTPException(
                    status_code=500,
                    detail="Failed to initialize model. Check server logs for details.",
                )

        # Generate video
        video_path = _engine.generate_video(
            prompt=payload.prompt, narrate=payload.narrate
        )

        # Read the TSX for preview
        tsx_path = _engine.remotion_dir / "src" / "GeneratedScene.tsx"
        tsx_code = None
        if tsx_path.exists():
            with open(tsx_path, "r") as f:
                tsx_code = f.read()

        processing_time = (time.time() - start_time) * 1000

        logger.info(
            {
                "event": "generation_success",
                "request_id": request_id,
                "processing_time_ms": round(processing_time, 2),
            }
        )

        return ExplainerResponse(
            success=True,
            video_url=video_path,
            tsx_preview=tsx_code,
            request_id=request_id,
            processing_time_ms=round(processing_time, 2),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            {
                "event": "generation_error",
                "request_id": request_id,
                "error": str(e),
                "error_type": type(e).__name__,
            }
        )
        raise HTTPException(
            status_code=500,
            detail="An internal error occurred. Check server logs for details.",
        )


@app.get(
    "/health",
    response_model=HealthResponse,
    responses={
        200: {"description": "Service healthy"},
        503: {"description": "Service unhealthy"},
    },
    summary="Health Check",
    description="Check the health status of the API service",
)
async def health_check() -> HealthResponse:
    """
    Health check endpoint.

    Returns the status of the service and whether the engine is ready.
    Does not require authentication.
    """
    engine_ready = _engine is not None

    status = "healthy" if engine_ready else "degraded"

    return HealthResponse(status=status, engine_ready=engine_ready)


@app.get(
    "/security/audit-log",
    response_model=Dict[str, Any],
    responses={
        200: {"description": "Audit log retrieved"},
        401: {"description": "Authentication required"},
    },
    summary="Get Security Audit Log",
    description="Retrieve security audit log entries (requires authentication)",
)
async def get_audit_log(
    auth_context: Dict[str, Any] = Depends(verify_authentication),
    limit: int = Query(default=100, ge=1, le=1000, description="Max entries to return"),
) -> Dict[str, Any]:
    """
    Get security audit log.

    Requires authentication with admin privileges.

    Args:
        limit: Maximum number of entries to return (default: 100)
    """
    # Check for admin permission
    permissions = auth_context.get("permissions", [])
    if not any(p.value == "admin" for p in permissions):
        # For API keys, check the permission set
        if isinstance(auth_context.get("permissions"), set):
            admin_perms = auth_context["permissions"]
        else:
            admin_perms = {p.value for p in permissions}

        if "admin" not in admin_perms:
            raise HTTPException(status_code=403, detail="Admin privileges required")

    auditor = get_security_auditor_instance()

    return auditor.get_violation_summary()


@app.get(
    "/rate-limit/stats",
    response_model=Dict[str, Any],
    responses={
        200: {"description": "Rate limit stats retrieved"},
        401: {"description": "Authentication required"},
    },
    summary="Get Rate Limit Statistics",
    description="Retrieve rate limiter statistics (requires authentication)",
)
async def get_rate_limit_stats(
    auth_context: Dict[str, Any] = Depends(verify_authentication),
) -> Dict[str, Any]:
    """
    Get rate limiter statistics.

    Returns overall rate limiter statistics.
    """
    limiter = get_rate_limiter_instance()
    return limiter.get_stats()


# ============================================================================
# AUTHENTICATION UTILITY ENDPOINTS
# ============================================================================


@app.post(
    "/auth/token",
    response_model=Dict[str, Any],
    responses={
        200: {"description": "Token generated successfully"},
        401: {"description": "Invalid credentials"},
    },
    summary="Generate JWT Token",
    description="Generate a JWT token for API authentication",
)
async def generate_token(request: Request, payload: TokenRequest) -> Dict[str, Any]:
    """
    Generate a JWT token.

    This is a utility endpoint for development/testing.
    In production, tokens should be generated through a proper auth service.
    """
    auth = get_authenticator_instance()

    # Convert permission strings to Permission enums
    permission_enums = []
    for perm in payload.permissions:
        try:
            permission_enums.append(Permission(perm))
        except ValueError:
            permission_enums.append(Permission.INFERENCE)

    # Generate token
    token = auth.jwt.generate_token(
        user_id=payload.user_id,
        permissions=permission_enums,
        expires_in=None,  # Uses default (24 hours)
    )

    logger.info(
        {
            "event": "token_generated",
            "user_id": payload.user_id,
            "permissions": payload.permissions,
        }
    )

    return {
        "access_token": token,
        "token_type": "bearer",
        "expires_in": 86400,  # 24 hours in seconds
    }


@app.post(
    "/auth/api-key",
    response_model=Dict[str, Any],
    responses={
        200: {"description": "API key generated successfully"},
        401: {"description": "Authentication required"},
    },
    summary="Generate API Key",
    description="Generate an API key for authentication",
)
async def generate_api_key(request: Request, payload: APIKeyRequest) -> Dict[str, Any]:
    """
    Generate an API key.

    This is a utility endpoint for development/testing.
    """
    auth = get_authenticator_instance()

    # Convert permission strings to Permission enums
    permission_enums = []
    for perm in payload.permissions:
        try:
            permission_enums.append(Permission(perm))
        except ValueError:
            permission_enums.append(Permission.INFERENCE)

    # Generate API key
    raw_key, api_key = auth.api_keys.generate_key(
        name=payload.name,
        permissions=permission_enums,
        expires_in_days=365,
        rate_limit=1000,
    )

    logger.info(
        {"event": "api_key_generated", "key_id": api_key.key_id, "name": payload.name}
    )

    return {
        "api_key": raw_key,
        "key_id": api_key.key_id,
        "name": api_key.name,
        "permissions": [p.value for p in api_key.permissions],
        "rate_limit": api_key.rate_limit,
        "expires_at": api_key.expires_at.isoformat() if api_key.expires_at else None,
    }


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import uvicorn
from nexus.config.paths import REMOTION_EXPLAINER_DIR

    # Run with uvicorn
    uvicorn.run(
        "nexus.api.explainer_api:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
        access_log=True,
    )
