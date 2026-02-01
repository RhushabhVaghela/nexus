"""
Security Tests

Tests for authentication, authorization, and security features.
"""

import pytest
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import jwt

# Import the auth module
import sys
sys.path.insert(0, 'src')
from nexus.security.auth import (
    APIKeyManager, JWTTokenManager, Authenticator,
    Permission, APIKey, JWTClaims, require_auth
)


class TestAPIKeyManager:
    """Test API key management."""
    
    def test_generate_key(self):
        """Test API key generation."""
        manager = APIKeyManager()
        raw_key, api_key = manager.generate_key(
            name="Test Key",
            permissions=[Permission.READ],
            expires_in_days=30
        )
        
        assert raw_key.startswith("nx_")
        assert api_key.name == "Test Key"
        assert Permission.READ in api_key.permissions
        assert api_key.is_active
    
    def test_validate_key_success(self):
        """Test successful key validation."""
        manager = APIKeyManager()
        raw_key, api_key = manager.generate_key(
            name="Test Key",
            permissions=[Permission.READ, Permission.WRITE]
        )
        
        validated = manager.validate_key(raw_key)
        assert validated is not None
        assert validated.key_id == api_key.key_id
        assert validated.last_used is not None
    
    def test_validate_key_invalid(self):
        """Test validation of invalid key."""
        manager = APIKeyManager()
        
        validated = manager.validate_key("invalid_key")
        assert validated is None
    
    def test_validate_key_revoked(self):
        """Test validation of revoked key."""
        manager = APIKeyManager()
        raw_key, api_key = manager.generate_key(
            name="Test Key",
            permissions=[Permission.READ]
        )
        
        # Revoke the key
        manager.revoke_key(api_key.key_id)
        
        validated = manager.validate_key(raw_key)
        assert validated is None
    
    def test_validate_key_expired(self):
        """Test validation of expired key."""
        manager = APIKeyManager()
        raw_key, api_key = manager.generate_key(
            name="Test Key",
            permissions=[Permission.READ],
            expires_in_days=-1  # Already expired
        )
        
        validated = manager.validate_key(raw_key)
        assert validated is None
    
    def test_delete_key(self):
        """Test key deletion."""
        manager = APIKeyManager()
        raw_key, api_key = manager.generate_key(
            name="Test Key",
            permissions=[Permission.READ]
        )
        
        # Delete the key
        assert manager.delete_key(api_key.key_id)
        
        # Should not be valid anymore
        validated = manager.validate_key(raw_key)
        assert validated is None
    
    def test_rotate_key(self):
        """Test key rotation."""
        manager = APIKeyManager()
        raw_key, api_key = manager.generate_key(
            name="Test Key",
            permissions=[Permission.READ]
        )
        
        # Rotate the key
        new_raw_key, new_api_key = manager.rotate_key(api_key.key_id)
        
        assert new_raw_key is not None
        assert new_raw_key != raw_key
        
        # Old key should be invalid
        assert manager.validate_key(raw_key) is None
        # New key should be valid
        assert manager.validate_key(new_raw_key) is not None
    
    def test_list_keys(self):
        """Test listing all keys."""
        manager = APIKeyManager()
        
        manager.generate_key(name="Key 1", permissions=[Permission.READ])
        manager.generate_key(name="Key 2", permissions=[Permission.WRITE])
        
        keys = manager.list_keys()
        assert len(keys) == 2


class TestJWTTokenManager:
    """Test JWT token management."""
    
    def test_generate_token(self):
        """Test JWT token generation."""
        manager = JWTTokenManager()
        token = manager.generate_token(
            user_id="user123",
            permissions=[Permission.READ, Permission.WRITE]
        )
        
        assert token is not None
        assert isinstance(token, str)
    
    def test_validate_token_success(self):
        """Test successful token validation."""
        manager = JWTTokenManager()
        token = manager.generate_token(
            user_id="user123",
            permissions=[Permission.READ]
        )
        
        claims = manager.validate_token(token)
        assert claims is not None
        assert claims.sub == "user123"
        assert "read" in claims.scopes
    
    def test_validate_token_expired(self):
        """Test validation of expired token."""
        manager = JWTTokenManager()
        
        # Generate token that expires immediately
        token = manager.generate_token(
            user_id="user123",
            permissions=[Permission.READ],
            expires_in=timedelta(seconds=-1)
        )
        
        claims = manager.validate_token(token)
        assert claims is None
    
    def test_validate_token_invalid(self):
        """Test validation of invalid token."""
        manager = JWTTokenManager()
        
        claims = manager.validate_token("invalid.token.here")
        assert claims is None
    
    def test_revoke_token(self):
        """Test token revocation."""
        manager = JWTTokenManager()
        token = manager.generate_token(
            user_id="user123",
            permissions=[Permission.READ]
        )
        
        claims = manager.validate_token(token)
        assert claims is not None
        
        # Revoke token
        manager.revoke_token(claims.jti)
        
        # Should no longer be valid
        claims = manager.validate_token(token)
        assert claims is None
    
    def test_refresh_token(self):
        """Test token refresh."""
        manager = JWTTokenManager()
        token = manager.generate_token(
            user_id="user123",
            permissions=[Permission.READ]
        )
        
        # Refresh token
        new_token = manager.refresh_token(token)
        
        assert new_token is not None
        assert new_token != token
        
        # Old token should be invalid
        assert manager.validate_token(token) is None
        # New token should be valid
        assert manager.validate_token(new_token) is not None
    
    def test_has_permission(self):
        """Test permission checking."""
        manager = JWTTokenManager()
        token = manager.generate_token(
            user_id="user123",
            permissions=[Permission.READ, Permission.WRITE]
        )
        
        assert manager.has_permission(token, Permission.READ)
        assert manager.has_permission(token, Permission.WRITE)
        assert not manager.has_permission(token, Permission.ADMIN)
    
    def test_admin_has_all_permissions(self):
        """Test that ADMIN permission grants all access."""
        manager = JWTTokenManager()
        token = manager.generate_token(
            user_id="admin123",
            permissions=[Permission.ADMIN]
        )
        
        assert manager.has_permission(token, Permission.READ)
        assert manager.has_permission(token, Permission.WRITE)
        assert manager.has_permission(token, Permission.INFERENCE)


class TestAuthenticator:
    """Test combined authenticator."""
    
    def test_authenticate_jwt(self):
        """Test JWT authentication."""
        auth = Authenticator()
        token = auth.jwt.generate_token(
            user_id="user123",
            permissions=[Permission.READ]
        )
        
        is_valid, error, context = auth.authenticate_request(f"Bearer {token}")
        
        assert is_valid
        assert error is None
        assert context["type"] == "jwt"
        assert context["user_id"] == "user123"
    
    def test_authenticate_api_key(self):
        """Test API key authentication."""
        auth = Authenticator()
        raw_key, api_key = auth.api_keys.generate_key(
            name="Test Key",
            permissions=[Permission.READ]
        )
        
        is_valid, error, context = auth.authenticate_request(f"ApiKey {raw_key}")
        
        assert is_valid
        assert error is None
        assert context["type"] == "api_key"
        assert context["key_id"] == api_key.key_id
    
    def test_authenticate_missing_header(self):
        """Test authentication with missing header."""
        auth = Authenticator()
        
        is_valid, error, context = auth.authenticate_request(None)
        
        assert not is_valid
        assert "Missing" in error
    
    def test_authenticate_invalid_format(self):
        """Test authentication with invalid header format."""
        auth = Authenticator()
        
        is_valid, error, context = auth.authenticate_request("InvalidHeader")
        
        assert not is_valid
        assert "Invalid" in error
    
    def test_authenticate_invalid_token(self):
        """Test authentication with invalid token."""
        auth = Authenticator()
        
        is_valid, error, context = auth.authenticate_request("Bearer invalid_token")
        
        assert not is_valid
        assert "Invalid" in error or "expired" in error.lower()
    
    def test_authenticate_unsupported_scheme(self):
        """Test authentication with unsupported scheme."""
        auth = Authenticator()
        
        is_valid, error, context = auth.authenticate_request("Basic dXNlcjpwYXNz")
        
        assert not is_valid
        assert "Unsupported" in error


class TestRequireAuthDecorator:
    """Test require_auth decorator."""
    
    def test_require_auth_success(self):
        """Test successful authentication with decorator."""
        auth = Authenticator()
        token = auth.jwt.generate_token(
            user_id="user123",
            permissions=[Permission.READ]
        )
        
        @require_auth(auth)
        def protected_function(auth_context=None):
            return "Success"
        
        result = protected_function(authorization=f"Bearer {token}")
        assert result == "Success"
    
    def test_require_auth_failure(self):
        """Test authentication failure with decorator."""
        auth = Authenticator()
        
        @require_auth(auth)
        def protected_function(auth_context=None):
            return "Success"
        
        with pytest.raises(PermissionError):
            protected_function(authorization="Bearer invalid")
    
    def test_require_auth_with_permission(self):
        """Test permission checking with decorator."""
        auth = Authenticator()
        token = auth.jwt.generate_token(
            user_id="user123",
            permissions=[Permission.READ]
        )
        
        @require_auth(auth, permissions=[Permission.WRITE])
        def protected_function(auth_context=None):
            return "Success"
        
        # Should fail due to missing permission
        with pytest.raises(PermissionError):
            protected_function(authorization=f"Bearer {token}")


class TestSecurityEdgeCases:
    """Test security edge cases."""
    
    def test_key_hashing(self):
        """Test that raw keys are properly hashed."""
        manager = APIKeyManager()
        raw_key, api_key = manager.generate_key(
            name="Test",
            permissions=[Permission.READ]
        )
        
        # Stored key should be hashed, not raw
        assert api_key.hashed_key != raw_key
        assert len(api_key.hashed_key) == 64  # SHA-256 hex
    
    def test_token_tampering(self):
        """Test detection of tampered tokens."""
        manager = JWTTokenManager(secret_key="secret1")
        token = manager.generate_token(
            user_id="user123",
            permissions=[Permission.READ]
        )
        
        # Tamper with token
        tampered_token = token[:-10] + "tamperedxx"
        
        # Should fail validation
        claims = manager.validate_token(tampered_token)
        assert claims is None
    
    def test_wrong_secret_key(self):
        """Test validation with wrong secret key."""
        manager1 = JWTTokenManager(secret_key="secret1")
        manager2 = JWTTokenManager(secret_key="secret2")
        
        token = manager1.generate_token(
            user_id="user123",
            permissions=[Permission.READ]
        )
        
        # Should fail with different secret
        claims = manager2.validate_token(token)
        assert claims is None
    
    def test_key_metadata(self):
        """Test API key metadata."""
        manager = APIKeyManager()
        raw_key, api_key = manager.generate_key(
            name="Test",
            permissions=[Permission.READ],
            metadata={"project": "test", "team": "ai"}
        )
        
        assert api_key.metadata["project"] == "test"
        assert api_key.metadata["team"] == "ai"


class TestRateLimitKey:
    """Test rate limit key generation."""
    
    def test_jwt_rate_limit_key(self):
        """Test rate limit key from JWT context."""
        from nexus.security.auth import rate_limit_key
        
        context = {
            "type": "jwt",
            "user_id": "user123"
        }
        
        key = rate_limit_key(context)
        assert key == "jwt:user123"
    
    def test_api_key_rate_limit_key(self):
        """Test rate limit key from API key context."""
        from nexus.security.auth import rate_limit_key
        
        context = {
            "type": "api_key",
            "key_id": "key_abc123"
        }
        
        key = rate_limit_key(context)
        assert key == "apikey:key_abc123"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
