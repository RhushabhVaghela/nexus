"""
Authentication Module for Nexus API

Provides JWT token validation, API key management, and secure
token generation for Nexus API endpoints.
"""

import hashlib
import hmac
import secrets
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from functools import wraps
from typing import Callable, Dict, List, Optional, Set, Tuple, Any

import jwt
from jwt.exceptions import ExpiredSignatureError, InvalidTokenError


class Permission(Enum):
    """API permission levels."""
    READ = "read"
    WRITE = "write"
    ADMIN = "admin"
    INFERENCE = "inference"
    TRAINING = "training"


@dataclass
class APIKey:
    """Represents an API key with metadata."""
    key_id: str
    hashed_key: str
    name: str
    permissions: Set[Permission]
    created_at: datetime
    expires_at: Optional[datetime]
    last_used: Optional[datetime]
    rate_limit: int = 1000  # requests per hour
    is_active: bool = True
    metadata: Dict[str, Any] = None


@dataclass
class JWTClaims:
    """JWT token claims."""
    sub: str  # Subject (user ID)
    iat: float  # Issued at
    exp: float  # Expiration
    scopes: List[str]  # Permissions
    jti: str  # JWT ID (for revocation)
    metadata: Optional[Dict[str, Any]] = None


class APIKeyManager:
    """Manages API key lifecycle and validation."""
    
    def __init__(self, secret_key: Optional[str] = None):
        """
        Initialize the API key manager.
        
        Args:
            secret_key: Master secret for key hashing (auto-generated if None)
        """
        self._keys: Dict[str, APIKey] = {}  # key_id -> APIKey
        self._hashed_key_lookup: Dict[str, str] = {}  # hashed_key -> key_id
        self._secret = secret_key or secrets.token_hex(32)
        self._revoked_keys: Set[str] = set()
    
    def generate_key(
        self,
        name: str,
        permissions: List[Permission],
        expires_in_days: Optional[int] = 365,
        rate_limit: int = 1000,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Tuple[str, APIKey]:
        """
        Generate a new API key.
        
        Args:
            name: Human-readable name for the key
            permissions: List of permissions
            expires_in_days: Key expiration in days (None for no expiration)
            rate_limit: Rate limit (requests per hour)
            metadata: Additional metadata
            
        Returns:
            Tuple of (raw_key, APIKey object)
        """
        # Generate secure random key
        raw_key = f"nx_{secrets.token_urlsafe(32)}"
        key_id = secrets.token_hex(16)
        
        # Hash the key for storage
        hashed_key = self._hash_key(raw_key)
        
        # Calculate expiration
        created_at = datetime.utcnow()
        expires_at = None
        if expires_in_days:
            expires_at = created_at + timedelta(days=expires_in_days)
        
        # Create API key record
        api_key = APIKey(
            key_id=key_id,
            hashed_key=hashed_key,
            name=name,
            permissions=set(permissions),
            created_at=created_at,
            expires_at=expires_at,
            last_used=None,
            rate_limit=rate_limit,
            is_active=True,
            metadata=metadata or {}
        )
        
        # Store
        self._keys[key_id] = api_key
        self._hashed_key_lookup[hashed_key] = key_id
        
        return raw_key, api_key
    
    def validate_key(self, raw_key: str) -> Optional[APIKey]:
        """
        Validate an API key.
        
        Args:
            raw_key: The API key to validate
            
        Returns:
            APIKey if valid, None otherwise
        """
        if not raw_key or not raw_key.startswith("nx_"):
            return None
        
        hashed_key = self._hash_key(raw_key)
        key_id = self._hashed_key_lookup.get(hashed_key)
        
        if not key_id:
            return None
        
        api_key = self._keys.get(key_id)
        if not api_key:
            return None
        
        # Check if revoked
        if key_id in self._revoked_keys:
            return None
        
        # Check if active
        if not api_key.is_active:
            return None
        
        # Check expiration
        if api_key.expires_at and datetime.utcnow() > api_key.expires_at:
            return None
        
        # Update last used
        api_key.last_used = datetime.utcnow()
        
        return api_key
    
    def revoke_key(self, key_id: str) -> bool:
        """
        Revoke an API key.
        
        Args:
            key_id: The key ID to revoke
            
        Returns:
            True if revoked, False if not found
        """
        if key_id in self._keys:
            self._revoked_keys.add(key_id)
            self._keys[key_id].is_active = False
            return True
        return False
    
    def delete_key(self, key_id: str) -> bool:
        """
        Permanently delete an API key.
        
        Args:
            key_id: The key ID to delete
            
        Returns:
            True if deleted, False if not found
        """
        if key_id in self._keys:
            api_key = self._keys[key_id]
            del self._hashed_key_lookup[api_key.hashed_key]
            del self._keys[key_id]
            self._revoked_keys.discard(key_id)
            return True
        return False
    
    def get_key(self, key_id: str) -> Optional[APIKey]:
        """Get API key by ID (without validating)."""
        return self._keys.get(key_id)
    
    def list_keys(self) -> List[APIKey]:
        """List all API keys."""
        return list(self._keys.values())
    
    def rotate_key(self, key_id: str) -> Optional[Tuple[str, APIKey]]:
        """
        Rotate an API key (invalidate old, create new with same permissions).
        
        Args:
            key_id: The key ID to rotate
            
        Returns:
            Tuple of (new_raw_key, APIKey) or None if key not found
        """
        old_key = self._keys.get(key_id)
        if not old_key:
            return None
        
        # Generate new key with same permissions
        new_raw_key, new_api_key = self.generate_key(
            name=f"{old_key.name} (rotated)",
            permissions=list(old_key.permissions),
            rate_limit=old_key.rate_limit,
            metadata={**old_key.metadata, "rotated_from": key_id}
        )
        
        # Revoke old key
        self.revoke_key(key_id)
        
        return new_raw_key, new_api_key
    
    def _hash_key(self, raw_key: str) -> str:
        """Hash an API key for storage."""
        return hmac.new(
            self._secret.encode(),
            raw_key.encode(),
            hashlib.sha256
        ).hexdigest()


class JWTTokenManager:
    """Manages JWT token generation and validation."""
    
    def __init__(self, secret_key: Optional[str] = None, algorithm: str = "HS256"):
        """
        Initialize JWT token manager.
        
        Args:
            secret_key: Secret key for signing (auto-generated if None)
            algorithm: JWT algorithm to use
        """
        self._secret = secret_key or secrets.token_hex(32)
        self._algorithm = algorithm
        self._revoked_tokens: Set[str] = set()  # Set of jti (JWT IDs)
        self._token_lifetime = timedelta(hours=24)  # Default token lifetime
    
    def generate_token(
        self,
        user_id: str,
        permissions: List[Permission],
        expires_in: Optional[timedelta] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate a JWT token.
        
        Args:
            user_id: Subject/user identifier
            permissions: List of permissions
            expires_in: Token lifetime (default: 24 hours)
            metadata: Additional claims
            
        Returns:
            JWT token string
        """
        now = datetime.utcnow()
        exp = now + (expires_in or self._token_lifetime)
        
        claims = {
            "sub": user_id,
            "iat": now,
            "exp": exp,
            "scopes": [p.value for p in permissions],
            "jti": secrets.token_urlsafe(16),
            "metadata": metadata or {}
        }
        
        return jwt.encode(claims, self._secret, algorithm=self._algorithm)
    
    def validate_token(self, token: str) -> Optional[JWTClaims]:
        """
        Validate a JWT token.
        
        Args:
            token: JWT token string
            
        Returns:
            JWTClaims if valid, None otherwise
        """
        try:
            payload = jwt.decode(token, self._secret, algorithms=[self._algorithm])
            
            # Check if token is revoked
            jti = payload.get("jti")
            if jti and jti in self._revoked_tokens:
                return None
            
            return JWTClaims(
                sub=payload["sub"],
                iat=payload["iat"],
                exp=payload["exp"],
                scopes=payload.get("scopes", []),
                jti=jti,
                metadata=payload.get("metadata")
            )
        except (ExpiredSignatureError, InvalidTokenError):
            return None
    
    def revoke_token(self, jti: str) -> bool:
        """
        Revoke a token by its JTI.
        
        Args:
            jti: JWT ID to revoke
            
        Returns:
            True if added to revocation list
        """
        self._revoked_tokens.add(jti)
        return True
    
    def revoke_token_by_value(self, token: str) -> bool:
        """
        Revoke a token by its full value.
        
        Args:
            token: JWT token string
            
        Returns:
            True if revoked successfully
        """
        try:
            payload = jwt.decode(
                token, 
                self._secret, 
                algorithms=[self._algorithm],
                options={"verify_exp": False}  # Allow revoking expired tokens
            )
            jti = payload.get("jti")
            if jti:
                self._revoked_tokens.add(jti)
                return True
        except InvalidTokenError:
            pass
        return False
    
    def refresh_token(self, token: str, expires_in: Optional[timedelta] = None) -> Optional[str]:
        """
        Refresh a valid token (generates new token with extended expiry).
        
        Args:
            token: Current valid JWT token
            expires_in: New expiration time
            
        Returns:
            New token string or None if current token is invalid
        """
        claims = self.validate_token(token)
        if not claims:
            return None
        
        # Revoke old token
        self.revoke_token(claims.jti)
        
        # Generate new token
        permissions = [Permission(s) for s in claims.scopes]
        return self.generate_token(
            user_id=claims.sub,
            permissions=permissions,
            expires_in=expires_in,
            metadata=claims.metadata
        )
    
    def has_permission(self, token: str, permission: Permission) -> bool:
        """
        Check if token has specific permission.
        
        Args:
            token: JWT token string
            permission: Permission to check
            
        Returns:
            True if token has permission
        """
        claims = self.validate_token(token)
        if not claims:
            return False
        return permission.value in claims.scopes or Permission.ADMIN.value in claims.scopes


class Authenticator:
    """Combined authentication handler for API keys and JWT tokens."""
    
    def __init__(self, secret_key: Optional[str] = None):
        """
        Initialize authenticator.
        
        Args:
            secret_key: Master secret for both API keys and JWT
        """
        self.api_keys = APIKeyManager(secret_key)
        self.jwt = JWTTokenManager(secret_key)
    
    def authenticate_request(
        self,
        auth_header: Optional[str] = None
    ) -> Tuple[bool, Optional[str], Optional[Dict[str, Any]]]:
        """
        Authenticate a request from Authorization header.
        
        Args:
            auth_header: Authorization header value (e.g., "Bearer <token>" or "ApiKey <key>")
            
        Returns:
            Tuple of (is_valid, error_message, auth_context)
        """
        if not auth_header:
            return False, "Missing authorization header", None
        
        parts = auth_header.split()
        if len(parts) != 2:
            return False, "Invalid authorization format", None
        
        scheme, credential = parts
        
        if scheme.lower() == "bearer":
            # JWT token
            claims = self.jwt.validate_token(credential)
            if not claims:
                return False, "Invalid or expired token", None
            
            context = {
                "type": "jwt",
                "user_id": claims.sub,
                "permissions": [Permission(s) for s in claims.scopes],
                "jti": claims.jti,
                "metadata": claims.metadata
            }
            return True, None, context
        
        elif scheme.lower() == "apikey":
            # API key
            api_key = self.api_keys.validate_key(credential)
            if not api_key:
                return False, "Invalid API key", None
            
            context = {
                "type": "api_key",
                "key_id": api_key.key_id,
                "permissions": api_key.permissions,
                "rate_limit": api_key.rate_limit,
                "metadata": api_key.metadata
            }
            return True, None, context
        
        else:
            return False, f"Unsupported authentication scheme: {scheme}", None


def require_auth(
    authenticator: Authenticator,
    permissions: Optional[List[Permission]] = None
) -> Callable:
    """
    Decorator to require authentication for a function.
    
    Args:
        authenticator: Authenticator instance
        permissions: Required permissions (optional)
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # This is a simplified version - in practice you'd extract
            # auth header from request context (e.g., Flask request, FastAPI Request)
            auth_header = kwargs.get("authorization") or (
                args[0].headers.get("Authorization") if args else None
            )
            
            is_valid, error, context = authenticator.authenticate_request(auth_header)
            
            if not is_valid:
                raise PermissionError(error)
            
            # Check permissions
            if permissions:
                user_permissions = context.get("permissions", set())
                if Permission.ADMIN not in user_permissions:
                    for perm in permissions:
                        if perm not in user_permissions:
                            raise PermissionError(f"Missing required permission: {perm.value}")
            
            # Add auth context to kwargs
            kwargs["auth_context"] = context
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


def rate_limit_key(auth_context: Dict[str, Any]) -> str:
    """
    Generate a rate limit key from auth context.
    
    Args:
        auth_context: Authentication context
        
    Returns:
        Rate limit key string
    """
    if auth_context["type"] == "jwt":
        return f"jwt:{auth_context['user_id']}"
    else:
        return f"apikey:{auth_context['key_id']}"


# Global authenticator instance (for use with dependency injection)
_global_authenticator: Optional[Authenticator] = None


def get_authenticator() -> Authenticator:
    """Get or create global authenticator instance."""
    global _global_authenticator
    if _global_authenticator is None:
        _global_authenticator = Authenticator()
    return _global_authenticator


def set_authenticator(authenticator: Authenticator) -> None:
    """Set global authenticator instance."""
    global _global_authenticator
    _global_authenticator = authenticator
