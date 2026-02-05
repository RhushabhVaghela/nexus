# Nexus Security Documentation

## Overview

Security is a fundamental design principle of the Nexus platform. This documentation provides comprehensive coverage of all security mechanisms, authentication methods, authorization patterns, best practices, and audit logging capabilities. Understanding these security controls is essential for administrators configuring access policies, developers integrating with the platform, and security auditors assessing the system.

The Nexus platform implements defense-in-depth security with multiple layers of protection spanning network security, authentication and authorization, data protection, input validation, and comprehensive audit logging. The security architecture is designed to meet enterprise compliance requirements including SOC 2, ISO 27001, and GDPR while maintaining the performance needed for production AI workloads.

This document covers authentication mechanisms including API keys and OAuth 2.0, authorization patterns with role-based and attribute-based access control, network security configurations, data protection measures, input validation and sanitization, security monitoring, and audit logging. For operational security procedures, see the Deployment Guide.

## Installation

### Security Dependencies

```bash
# Core security dependencies
pip install nexus-security

# For OAuth 2.0 support
pip install python-jose[cryptography]  # JWT handling
pip install authlib                    # OAuth client

# For rate limiting
pip install redis                      # Redis backend for rate limits
pip install python-ratelimit

# For encryption
pip install cryptography               # Fernet encryption
pip install pym enca                   # NaCl encryption

# For audit logging
pip install elasticsearch              # Audit log storage
pip install azure-identity             # Azure AD integration
pip install boto3                      # AWS integration
```

### Security Module Installation

```bash
# Install all security modules
pip install nexus-security[all]

# Install specific modules
pip install nexus-security[auth]       # Authentication only
pip install nexus-security[encryption] # Encryption only
pip install nexus-security[audit]      # Audit logging only
```

## Usage

### Quick Start with Security

```python
from nexus.security import SecurityConfig, SecurityManager
from nexus.security.auth import Authenticator, TokenManager
from nexus.security.audit import AuditLogger

# Configure security settings
config = SecurityConfig(
    # Authentication configuration
    authentication=AuthenticationConfig(
        enabled=True,
        provider="oauth",  # "api_key", "oauth", "sso"
        api_key_rotation_days=90,
        oauth_config=OAuthConfig(
            client_id="your-client-id",
            client_secret="your-client-secret",
            issuer="https://auth.example.com",
            audience="nexus-api"
        )
    ),
    
    # Authorization configuration
    authorization=AuthorizationConfig(
        enabled=True,
        policy="rbac",  # "rbac", "abac", "hybrid"
        rbac=RBACConfig(
            roles=["admin", "developer", "viewer"],
            permissions={
                "admin": ["read", "write", "delete", "admin"],
                "developer": ["read", "write"],
                "viewer": ["read"]
            }
        )
    ),
    
    # Encryption configuration
    encryption=EncryptionConfig(
        enabled=True,
        algorithm="aes-256-gcm",
        key_rotation_days=365,
        encrypt_at_rest=True,
        encrypt_in_transit=True
    ),
    
    # Rate limiting configuration
    rate_limiting=RateLimitingConfig(
        enabled=True,
        backend="redis",
        default_limits={
            "requests_per_minute": 1000,
            "tokens_per_minute": 100000,
            "concurrent_requests": 10
        }
    ),
    
    # Audit configuration
    audit=AuditConfig(
        enabled=True,
        backend="elasticsearch",
        log_level="info",
        sensitive_fields=["password", "token", "api_key"],
        retention_days=365
    )
)

# Initialize security manager
security = SecurityManager(config)

# Apply security middleware to application
app = security.wrap_application(app)
```

### Authentication Setup

```python
from nexus.security.auth import Authenticator, APIKeyAuth, OAuthAuth

# API Key Authentication
api_key_auth = APIKeyAuth(
    header_name="Authorization",
    prefix="Bearer",
    key_length=32,
    hash_algorithm="sha256",
    rotation_required=True
)

# OAuth 2.0 Authentication
oauth_auth = OAuthAuth(
    client_id="your-client-id",
    client_secret="your-client-secret",
    authorization_url="https://auth.example.com/authorize",
    token_url="https://auth.example.com/token",
    refresh_url="https://auth.example.com/refresh",
    scopes=["openid", "profile", "email", "nexus:api"],
    audiences=["nexus-api"],
    jwks_uri="https://auth.example.com/.well-known/jwks.json"
)

# Combined authenticator
authenticator = Authenticator(
    providers=[api_key_auth, oauth_auth],
    default_provider="oauth",
    token_expiry_seconds=3600,
    refresh_token_expiry_seconds=86400
)

# Authenticate request
async def authenticate_request(request):
    token = request.headers.get("Authorization")
    if not token:
        raise AuthenticationError("Missing authentication")
    
    user = await authenticator.authenticate(token)
    return user
```

## Authentication Methods

### API Key Authentication

```python
from nexus.security.auth import APIKeyManager

# Create API key manager
key_manager = APIKeyManager(
    algorithm="sha256",
    key_length=64,
    prefix="nexus_",
    hash_salt="unique-salt-value",
    expiry_days=90,
    scopes=["read", "write"]
)

# Generate new API key
api_key = key_manager.create_key(
    name="production-api-key",
    user_id="user_123",
    scopes=["read", "write", "execute"],
    expires_in_days=90,
    rate_limit=1000
)

print(f"API Key: {api_key.key}")
print(f"Key ID: {api_key.key_id}")
print(f"Expires: {api_key.expires_at}")

# Verify API key
user = key_manager.verify_key(api_key.key)
print(f"User: {user.id}")
print(f"Scopes: {user.scopes}")

# Rotate API key
new_key = key_manager.rotate_key(
    old_key_id=api_key.key_id,
    copy_scopes=True,
    keep_old_for_hours=24  # Grace period
)

# Revoke API key
key_manager.revoke_key(api_key.key_id, reason="security_incident")

# List user keys
keys = key_manager.list_keys(user_id="user_123")
for key in keys:
    print(f"{key.name}: {key.key_id} ({key.created_at})")
```

### OAuth 2.0 Authentication

```python
from nexus.security.auth import OAuthManager, TokenManager

# Initialize OAuth manager
oauth = OAuthManager(
    client_id="your-client-id",
    client_secret="your-client-secret",
    redirect_uri="https://your-app.com/callback",
    authorization_endpoint="https://auth.example.com/authorize",
    token_endpoint="https://auth.example.com/token",
    userinfo_endpoint="https://auth.example.com/userinfo",
    jwks_uri="https://auth.example.com/.well-known/jwks.json",
    scopes=["openid", "profile", "email", "nexus:api"],
    audiences=["nexus-api"]
)

# Generate authorization URL
auth_url = oauth.get_authorization_url(
    state="random-state-string",
    nonce="random-nonce",
    redirect_uri="https://your-app.com/callback"
)
print(f"Redirect user to: {auth_url}")

# Exchange authorization code for tokens
tokens = oauth.exchange_code(
    code="authorization-code",
    redirect_uri="https://your-app.com/callback"
)
print(f"Access Token: {tokens.access_token}")
print(f"Refresh Token: {tokens.refresh_token}")
print(f"Expires In: {tokens.expires_in}")

# Refresh access token
new_tokens = oauth.refresh_token(refresh_token=current_refresh)
access_token = new_tokens.access_token

# Verify token
user_info = oauth.verify_token(access_token)
print(f"User: {user_info.sub}")
print(f"Email: {user_info.email}")
print(f"Roles: {user_info.roles}")

# Revoke tokens
oauth.revoke_token(access_token)
```

### JWT Token Management

```python
from nexus.security.auth import TokenManager, JWKSManager

# Create token manager
token_manager = TokenManager(
    secret_key="your-secret-key",
    algorithm="RS256",  # "HS256", "RS256", "ES256"
    access_token_expiry=3600,  # 1 hour
    refresh_token_expiry=86400,  # 24 hours
    issuer="nexus-platform",
    audience="nexus-api"
)

# Generate access token
access_token = token_manager.create_access_token(
    user_id="user_123",
    roles=["developer"],
    scopes=["read", "write"],
    metadata={"org_id": "org_456"}
)

# Generate refresh token
refresh_token = token_manager.create_refresh_token(
    user_id="user_123",
    device_id="device_789"
)

# Verify and decode token
payload = token_manager.verify_token(access_token)
print(f"User: {payload.sub}")
print(f"Roles: {payload.roles}")
print(f"Scopes: {payload.scope}")

# JWKS for token verification
jwks_manager = JWKSManager(
    private_key=private_key,
    public_key=public_key,
    key_id="key-1",
    algorithm="RS256",
    rotate_keys=True,
    key_lifetime_days=90
)

# Get public keys for token verification
public_keys = jwks_manager.get_jwks()
print(public_keys)
```

## Authorization Patterns

### Role-Based Access Control (RBAC)

```python
from nexus.security.auth import RBACAuthorizer, Permission

# Define roles and permissions
rbac = RBACAuthorizer(
    roles={
        "admin": Permission(
            actions=["read", "write", "delete", "admin", "execute"],
            resources=["*"],
            conditions=None
        ),
        "developer": Permission(
            actions=["read", "write", "execute"],
            resources=["models:*", "datasets:*", "training:*"],
            conditions={
                "org_match": True
            }
        ),
        "viewer": Permission(
            actions=["read"],
            resources=["models:read", "datasets:read"],
            conditions=None
        ),
        "service": Permission(
            actions=["read", "write"],
            resources=["models:*"],
            conditions={
                "service_account": True
            }
        )
    },
    default_role="viewer",
    super_role="admin"
)

# Check permissions
can_write = rbac.check_permission(
    user_roles=["developer"],
    action="write",
    resource="models:test-model"
)
print(f"Can write model: {can_write}")  # True

can_delete = rbac.check_permission(
    user_roles=["developer"],
    action="delete",
    resource="models:test-model"
)
print(f"Can delete model: {can_delete}")  # False

can_admin = rbac.check_permission(
    user_roles=["admin"],
    action="admin",
    resource="*"
)
print(f"Full admin access: {can_admin}")  # True

# Resource-level permissions
rbac.grant_permission(
    user_id="user_123",
    role="developer",
    resource="models:my-model",
    expires_at="2024-12-31"
)

rbac.revoke_permission(
    user_id="user_123",
    role="developer",
    resource="models:my-model"
)
```

### Attribute-Based Access Control (ABAC)

```python
from nexus.security.auth import ABACAuthorizer, AttributeRule

# Define attribute-based rules
abac = ABACAuthorizer(
    rules=[
        AttributeRule(
            name="org_member_access",
            description="Organization members can access org resources",
            conditions={
                "subject.org_id == resource.org_id",
                "subject.active == true"
            },
            actions=["read", "write"],
            priority=100
        ),
        AttributeRule(
            name="time_based_access",
            description="Access allowed only during business hours",
            conditions={
                "current_time.hour >= 9",
                "current_time.hour <= 18",
                "current_time.weekday <= 5"
            },
            actions=["read", "write", "delete"],
            priority=50
        ),
        AttributeRule(
            name="location_based_access",
            description="Access restricted to allowed locations",
            conditions={
                "subject.ip_country in ['US', 'CA', 'UK']"
            },
            actions=["*"],
            priority=75
        ),
        AttributeRule(
            name="sensitive_data_protection",
            description="Sensitive data requires additional verification",
            conditions={
                "resource.sensitivity == 'high'",
                "subject.mfa_enabled == true"
            },
            actions=["read"],
            priority=200
        )
    ],
    default_deny=True
)

# Evaluate access request
decision = abac.evaluate(
    subject=Subject(
        user_id="user_123",
        org_id="org_456",
        active=True,
        mfa_enabled=True,
        ip_country="US"
    ),
    resource=Resource(
        id="dataset:customer-data",
        org_id="org_456",
        sensitivity="high"
    ),
    action="read",
    context=Context(
        current_time=datetime.now(),
        ip_address="192.168.1.1"
    )
)

print(f"Access granted: {decision.granted}")
print(f"Reason: {decision.reason}")
```

### Fine-Grained Authorization

```python
from nexus.security.auth import FineGrainedAuthorizer, Policy

# Define authorization policies
authorizer = FineGrainedAuthorizer(
    policies=[
        Policy(
            name="model_access_policy",
            rules=[
                # Owners have full access
                Rule(
                    when={"subject.id == resource.owner_id"},
                    then=Decision.ALLOW_ALL
                ),
                # Team members can read and write
                Rule(
                    when={
                        "subject.id in resource.team_members",
                        "subject.org_id == resource.org_id"
                    },
                    then=Decision.ALLOW.with_actions(["read", "write"])
                ),
                # Public models can be read by anyone
                Rule(
                    when={"resource.visibility == 'public'"},
                    then=Decision.ALLOW.with_action("read")
                )
            ]
        ),
        Policy(
            name="training_access_policy",
            rules=[
                # Admin can manage all training jobs
                Rule(
                    when={"subject.has_role('admin')"},
                    then=Decision.ALLOW_ALL
                ),
                # Users can access their own training jobs
                Rule(
                    when={"subject.id == resource.user_id"},
                    then=Decision.ALLOW_ALL
                ),
                # Team leads can access team jobs
                Rule(
                    when={
                        "subject.has_role('team_lead')",
                        "subject.team_id == resource.team_id"
                    },
                    then=Decision.ALLOW.with_actions(["read", "cancel"])
                )
            ]
        )
    ],
    combine_decisions="all_permit"  # "all_permit", "first_permit", "explicit"
)

# Check authorization
decision = authorizer.check(
    user=user,
    resource=model,
    action="write"
)
```

## Security Best Practices

### Input Validation and Sanitization

```python
from nexus.security.validation import (
    InputValidator,
    OutputValidator,
    Sanitizer,
    SchemaValidator
)

# Configure input validation
validator = InputValidator(
    # String validation
    max_length=10000,
    allow_empty=False,
    strip_whitespace=True,
    
    # Content validation
    allowed_content_types=["text/plain", "application/json"],
    blocked_patterns=[
        r"<script.*?>.*?</script>",
        r"javascript:",
        r"on\w+=",
        r"SELECT.*FROM",
        r"DROP.*TABLE"
    ],
    
    # Type validation
    enforce_types=True,
    coerce_types=True,
    
    # File upload validation
    max_file_size_mb=100,
    allowed_extensions=[".txt", ".json", ".csv", ".parquet"],
    scan_for_malware=True,
    
    # Injection prevention
    sql_injection_check=True,
    xss_check=True,
    command_injection_check=True
)

# Validate inputs
try:
    validated = validator.validate(
        data={
            "prompt": user_input,
            "max_tokens": max_tokens,
            "temperature": temperature
        },
        schema={
            "prompt": {"type": "string", "max_length": 4096},
            "max_tokens": {"type": "integer", "min": 1, "max": 4096},
            "temperature": {"type": "number", "min": 0, "max": 2}
        }
    )
except ValidationError as e:
    print(f"Validation failed: {e.message}")

# Sanitize inputs
sanitizer = Sanitizer(
    remove_html_tags=True,
    escape_html_entities=True,
    normalize_unicode=True,
    remove_control_chars=True,
    max_length=10000
)

clean_input = sanitizer.sanitize(user_input)
```

### Data Protection

```python
from nexus.security.encryption import (
    EncryptionManager,
    KeyManager,
    DataMasker
)

# Initialize encryption manager
encryption = EncryptionManager(
    algorithm="aes-256-gcm",
    key_length=32,
    iv_length=12,
    tag_length=16,
    key_provider="aws-kms",  # "aws-kms", "azure-keyvault", "hashicorp-vault", "local"
    key_id="your-kms-key-id"
)

# Encrypt sensitive data
encrypted = encryption.encrypt(
    data=b"sensitive data",
    key_id="default",
    associated_data={"user_id": "user_123"}
)
print(f"Encrypted: {encrypted.ciphertext}")
print(f"IV: {encrypted.iv}")
print(f"Tag: {encrypted.tag}")

# Decrypt data
decrypted = encryption.decrypt(
    ciphertext=encrypted.ciphertext,
    iv=encrypted.iv,
    tag=encrypted.tag,
    associated_data={"user_id": "user_123"}
)
print(f"Decrypted: {decrypted}")

# Key management
key_manager = KeyManager(
    provider="aws-kms",
    region="us-east-1",
    key_rotation_days=365,
    automatic_rotation=True
)

# Generate new key
new_key = key_manager.generate_key(
    alias="nexus-data-key",
    usage="encrypt",
    key_material_length=32
)

# Rotate keys
key_manager.rotate_key(
    key_alias="nexus-data-key",
    re_encrypt_existing=True,
    grace_period_days=7
)

# Data masking for logs and responses
masker = DataMasker(
    patterns=[
        (r"\b\d{3}-\d{2}-\d{4}\b", "***-**-****"),  # SSN
        (r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "***@***.***"),  # Email
        (r"\b\d{16}\b", "****-****-****-****"),  # Credit card
        (r"(?i)(api[_-]?key)[=:]\s*([A-Za-z0-9-_]+)", r"\1: ***"),  # API key
        (r"(?i)(bearer)\s+([A-Za-z0-9-_]+\.[A-Za-z0-9-_]+)", r"\1 ***")  # JWT
    ],
    mask_character="*"
)

masked = masker.mask(log_message)
print(masked)
```

### Network Security

```python
from nexus.security.network import (
    NetworkPolicy,
    FirewallManager,
    TLSPolicy
)

# Configure TLS
tls = TLSPolicy(
    minimum_version="TLSv1.3",
    cipher_suites=[
        "TLS_AES_256_GCM_SHA384",
        "TLS_AES_128_GCM_SHA256",
        "TLS_CHACHA20_POLY1305_SHA256"
    ],
    certificate=CERTConfig(
        provider="letsencrypt",
        domains=["api.nexus.example.com"],
        renewal_days=30,
        ocsp_stapling=True
    ),
    hsts=HeaderConfig(
        enabled=True,
        max_age=31536000,
        include_subdomains=True,
        preload=True
    )
)

# Network firewall configuration
firewall = FirewallManager(
    default_action="deny",
    rules=[
        FirewallRule(
            name="allow-api-traffic",
            action="allow",
            source="10.0.0.0/8",
            destination="0.0.0.0/0",
            ports=[443],
            protocols=["tcp"],
            description="Allow internal API traffic"
        ),
        FirewallRule(
            name="allow-monitoring",
            action="allow",
            source="monitoring.nexus.example.com",
            destination="0.0.0.0/0",
            ports=[9090, 3000],
            protocols=["tcp"]
        ),
        FirewallRule(
            name="block-ssh-from-internet",
            action="deny",
            source="0.0.0.0/0",
            destination="0.0.0.0/0",
            ports=[22],
            protocols=["tcp"],
            description="Block SSH from internet"
        )
    ],
    rate_limit=1000,  # Connections per second
    connection_timeout=30
)

# Network policy
network = NetworkPolicy(
    isolation="namespace",
    allowed_egress=[
        EgressRule(
            to=["kubernetes.io/api/v1/pods"],
            ports=[443],
            domains=["*.huggingface.co", "*.openai.com"]
        ),
        EgressRule(
            to=["cloudprovider.com"],
            ports=[443],
            for_resources=["model-registry", "checkpoint-storage"]
        )
    ],
    allowed_ingress=[
        IngressRule(
            from=["ingress-controller.nexus.svc"],
            ports=[8080]
        )
    ],
    dns_policy="CoreDNS",
    host_network=False,
    host_ipc=False,
    host_pid=False
)
```

### Rate Limiting

```python
from nexus.security.rate_limiting import (
    RateLimiter,
    SlidingWindowLog,
    TokenBucket
)

# Configure rate limiter
limiter = RateLimiter(
    backend="redis",
    prefix="nexus:ratelimit",
    
    # Global limits
    global_limits=RateLimit(
        requests_per_second=10000,
        tokens_per_minute=1000000
    ),
    
    # User limits
    user_limits=RateLimit(
        requests_per_minute=1000,
        tokens_per_minute=100000,
        concurrent_requests=10
    ),
    
    # Model-specific limits
    model_limits={
        "large-model": RateLimit(
            requests_per_minute=100,
            tokens_per_minute=50000
        ),
        "small-model": RateLimit(
            requests_per_minute=1000,
            tokens_per_minute=500000
        )
    },
    
    # Endpoint-specific limits
    endpoint_limits={
        "/v1/completions": RateLimit(
            requests_per_minute=500,
            tokens_per_minute=100000
        ),
        "/v1/chat/completions": RateLimit(
            requests_per_minute=500,
            tokens_per_minute=100000
        ),
        "/v1/embeddings": RateLimit(
            requests_per_minute=2000,
            tokens_per_minute=200000
        )
    },
    
    # Algorithm configuration
    algorithm="sliding_window_log",  # "sliding_window_log", "token_bucket", "fixed_window"
    
    # Response configuration
    headers=True,
    retry_after_header=True,
    custom_message="Rate limit exceeded. Please retry later."
)

# Check rate limit for user
remaining, reset_time = limiter.check(
    user_id="user_123",
    endpoint="/v1/completions",
    model="large-model"
)
print(f"Remaining: {remaining}")
print(f"Reset in: {reset_time} seconds")

# Custom rate limit for specific use case
custom_limiter = TokenBucket(
    capacity=100,
    refill_rate=10,  # tokens per second
    tokens_per_request=1
)

# Apply rate limit decorator
@limiter.limit(user_key="user_id")
async def api_endpoint(request):
    pass
```

## Audit Logging

### Audit Logger Configuration

```python
from nexus.security.audit import (
    AuditLogger,
    AuditEvent,
    AuditSink
)

# Configure audit logger
audit_logger = AuditLogger(
    enabled=True,
    
    # Backend configuration
    backend="elasticsearch",
    hosts=["elasticsearch.nexus.svc:9200"],
    index="nexus-audit-logs",
    doc_type="audit_event",
    
    # Buffer configuration
    buffer_size=1000,
    flush_interval=10,  # seconds
    batch_size=100,
    async_write=True,
    
    # Retention
    retention_days=365,
    archive_after_days=90,
    archive_storage="s3",
    archive_bucket="nexus-audit-archive",
    
    # Filtering
    exclude_patterns=[
        "/health",
        "/metrics",
        "/v1/rate-limit-status"
    ],
    sensitive_fields=[
        "api_key",
        "access_token",
        "refresh_token",
        "password",
        "secret",
        "credential"
    ],
    
    # Enrichment
    enrich_request=True,
    enrich_with_user_agent=True,
    enrich_with_geoip=True,
    
    # Encryption
    encrypt_records=True,
    encryption_key="your-encryption-key"
)

# Create audit sink
sink = AuditSink(
    type="elasticsearch",
    config={
        "hosts": ["elasticsearch.nexus.svc:9200"],
        "index": "nexus-audit-logs",
        "bulk_size": 100
    }
)

# Add another sink for alerting
alert_sink = AuditSink(
    type="alertmanager",
    config={
        "url": "http://alertmanager.nexus.svc:9093",
        "labels": {
            "severity": "security",
            "component": "audit"
        }
    }
)

audit_logger.add_sink(sink)
audit_logger.add_sink(alert_sink)
```

### Logging Audit Events

```python
from nexus.security.audit import AuditEvent, AuditAction

# Create audit event
event = AuditEvent(
    id="evt_12345",
    timestamp=datetime.utcnow(),
    action=AuditAction.INVOKE_MODEL,
    actor=Actor(
        id="user_123",
        type="user",
        email="user@example.com",
        roles=["developer"],
        org_id="org_456"
    ),
    resource=Resource(
        id="model:llama-3-8b",
        type="model",
        name="meta-llama/Llama-3-8b-instruct"
    ),
    context=Context(
        ip_address="192.168.1.100",
        user_agent="Nexus-Python/1.0",
        request_id="req_abc123",
        session_id="sess_xyz789"
    ),
    details={
        "prompt_tokens": 50,
        "completion_tokens": 200,
        "model_version": "1.0",
        "inference_time_ms": 150
    },
    outcome="success",
    risk_level="low"
)

# Log the event
audit_logger.log(event)

# Batch logging
events = [
    AuditEvent(
        action=AuditAction.READ_MODEL,
        actor=Actor(id="user_123"),
        resource=Resource(id="model:test"),
        outcome="success"
    ),
    AuditEvent(
        action=AuditAction.CREATE_TRAINING,
        actor=Actor(id="user_123"),
        resource=Resource(id="training:job-456"),
        outcome="success"
    )
]
audit_logger.log_batch(events)

# Conditional logging
audit_logger.log_if(
    condition=event.action in [AuditAction.WRITE, AuditAction.DELETE],
    event=event
)
```

### Audit Event Types

```python
from nexus.security.audit import AuditAction

# Authentication events
AUDIT_AUTHENTICATION = [
    AuditAction.LOGIN_SUCCESS,
    AuditAction.LOGIN_FAILURE,
    AuditAction.LOGOUT,
    AuditAction.TOKEN_CREATED,
    AuditAction.TOKEN_REVOKED,
    AuditAction.TOKEN_EXPIRED,
    AuditAction.SESSION_CREATED,
    AuditAction.SESSION_TERMINATED,
    AuditAction.PASSWORD_CHANGED,
    AuditAction.MFA_ENABLED,
    AuditAction.MFA_DISABLED,
    AuditAction.API_KEY_CREATED,
    AuditAction.API_KEY_REVOKED,
    AuditAction.API_KEY_ROTATED
]

# Authorization events
AUDIT_AUTHORIZATION = [
    AuditAction.ACCESS_GRANTED,
    AuditAction.ACCESS_DENIED,
    AuditAction.PERMISSION_GRANTED,
    AuditAction.PERMISSION_REVOKED,
    AuditAction.ROLE_ASSIGNED,
    AuditAction.ROLE_REMOVED,
    AuditAction.POLICY_CHANGED
]

# Data events
AUDIT_DATA = [
    AuditAction.READ_MODEL,
    AuditAction.WRITE_MODEL,
    AuditAction.DELETE_MODEL,
    AuditAction.READ_DATASET,
    AuditAction.WRITE_DATASET,
    AuditAction.DELETE_DATASET,
    AuditAction.EXPORT_DATA,
    AuditAction.IMPORT_DATA
]

# Inference events
AUDIT_INFERENCE = [
    AuditAction.INVOKE_MODEL,
    AuditAction.INFERENCE_COMPLETED,
    AuditAction.INFERENCE_FAILED,
    AuditAction.BATCH_SUBMITTED,
    AuditAction.BATCH_COMPLETED
]

# Training events
AUDIT_TRAINING = [
    AuditAction.CREATE_TRAINING,
    AuditAction.START_TRAINING,
    AuditAction.STOP_TRAINING,
    AuditAction.TRAINING_COMPLETED,
    AuditAction.TRAINING_FAILED,
    AuditAction.CHECKPOINT_CREATED,
    AuditAction.CHECKPOINT_RESTORED
]

# Administrative events
AUDIT_ADMINISTRATION = [
    AuditAction.CONFIG_CHANGED,
    AuditAction.USER_CREATED,
    AuditAction.USER_DELETED,
    AuditAction.USER_MODIFIED,
    AuditAction.ORG_CREATED,
    AuditAction.ORG_MODIFIED,
    AuditAction.SETTINGS_CHANGED,
    AuditAction.INTEGRATION_ADDED,
    AuditAction.INTEGRATION_REMOVED
]

# Security events
AUDIT_SECURITY = [
    AuditAction.SECURITY_ALERT,
    AuditAction.SUSPICIOUS_ACTIVITY,
    AuditAction.BRUTE_FORCE_DETECTED,
    AuditAction.INTRUSION_DETECTED,
    AuditAction.VULNERABILITY_FOUND,
    AuditAction.COMPLIANCE_VIOLATION
]
```

### Querying Audit Logs

```python
from nexus.security.audit import AuditQuery

# Query audit logs
query = AuditQuery(
    backend="elasticsearch",
    index="nexus-audit-logs",
    
    # Filters
    filters={
        "actor.id": "user_123",
        "action": ["INVOKE_MODEL", "READ_MODEL"],
        "outcome": "success",
        "timestamp": {
            "gte": "2024-01-01",
            "lte": "2024-12-31"
        }
    },
    
    # Pagination
    size=100,
    from_=0,
    
    # Sorting
    sort=[{"timestamp": {"order": "desc"}}],
    
    # Aggregation
    aggregations={
        "action_breakdown": {
            "terms": {"field": "action"}
        },
        "daily_volume": {
            "date_histogram": {
                "field": "timestamp",
                "calendar_interval": "day"
            }
        }
    }
)

# Execute query
results = audit_logger.query(query)

# Get events
for event in results.events:
    print(f"{event.timestamp}: {event.action} by {event.actor.id}")

# Get aggregations
for agg_name, agg_result in results.aggregations.items():
    print(f"{agg_name}: {agg_result}")

# Generate compliance report
report = audit_logger.generate_compliance_report(
    standard="SOC2",
    start_date="2024-01-01",
    end_date="2024-12-31"
)
print(report.summary)
```

## Security Monitoring

### Security Alerts

```python
from nexus.security.monitoring import (
    SecurityMonitor,
    AlertRule,
    AlertManager
)

# Configure security monitor
monitor = SecurityMonitor(
    # Metrics source
    metrics_backend="prometheus",
    query_interval=60,  # seconds
    
    # Alert rules
    alert_rules=[
        AlertRule(
            name="failed_login_spike",
            condition='rate(auth_login_failure[5m]) > 10',
            severity="warning",
            description="More than 10 failed logins in 5 minutes",
            labels={"component": "authentication"},
            annotations={
                "summary": "Failed login spike detected",
                "description": "Rate: {{ $value }} failed logins/min"
            }
        ),
        AlertRule(
            name="api_key_exposure",
            condition='rate(api_key_exposed[1h]) > 0',
            severity="critical",
            description="API key potentially exposed in logs",
            labels={"component": "security"},
            annotations={
                "summary": "API key exposure detected",
                "action": "Rotate affected API keys immediately"
            }
        ),
        AlertRule(
            name="rate_limit_violation",
            condition='rate(rate_limit_exceeded[5m]) > 100',
            severity="warning",
            description="High rate of rate limit violations",
            labels={"component": "rate_limiting"}
        ),
        AlertRule(
            name="suspicious_ip",
            condition='rate(requests_from_new_ip[1h]) > 50',
            severity="warning",
            description="Requests from many new IP addresses",
            labels={"component": "access"}
        )
    ],
    
    # Alert routing
    alertmanager_url="http://alertmanager.nexus.svc:9093",
    receiver="security-team",
    
    # Suppression
    suppression_rules=[
        SuppressionRule(
            name="maintenance_window",
            condition="maintenance_mode == true",
            duration=3600
        )
    ]
)

# Start monitoring
monitor.start()

# Get security status
status = monitor.get_status()
print(f"Overall status: {status.overall_status}")
print(f"Active alerts: {status.active_alerts}")
print(f"Last check: {status.last_check}")

# Manually trigger alert
monitor.trigger_alert(
    name="security_incident",
    severity="critical",
    description="Suspicious activity detected",
    labels={"incident_id": "INC-123"}
)
```

### Vulnerability Scanning

```python
from nexus.security.vulnerability import (
    VulnerabilityScanner,
    DependencyScanner,
    SecretScanner
)

# Configure vulnerability scanner
scanner = VulnerabilityScanner(
    # Dependencies to scan
    dependencies=True,
    container_images=True,
    source_code=True,
    
    # Vulnerability databases
    sources=[
        "cve.mitre.org",
        "nvd.nist.gov",
        "github.com/advisories"
    ],
    
    # Severity thresholds
    critical_threshold=0,
    high_threshold=7,
    medium_threshold=4,
    
    # Scanning schedule
    schedule="0 2 * * 0",  # Weekly on Sunday at 2 AM
    
    # Reporting
    report_format="html",
    report_path="/reports/vulnerability"
)

# Scan dependencies
results = scanner.scan_dependencies(
    requirements_file="requirements.txt",
    packages=["nexus-core", "torch", "transformers"]
)

# Check for vulnerabilities
for vuln in results.vulnerabilities:
    print(f"{vuln.id}: {vuln.description}")
    print(f"  Severity: {vuln.severity}")
    print(f"  Package: {vuln.package}@{vuln.version}")
    print(f"  Fix: {vuln.fixed_in}")

# Scan for secrets
secret_scanner = SecretScanner(
    patterns=[
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "API_KEY",
        "PRIVATE_KEY",
        "PASSWORD",
        "SECRET"
    ],
    exclude_patterns=["tests/", "docs/"],
    entropy_threshold=4.5
)

secrets = secret_scanner.scan(
    paths=["/src", "/config"],
    report_path="/reports/secrets"
)

for secret in secrets.found:
    print(f"{secret.file}:{secret.line}: Potential secret")
```

## Compliance and Governance

### Compliance Reports

```python
from nexus.security.compliance import (
    ComplianceManager,
    ComplianceReport,
    PolicyValidator
)

# Configure compliance manager
compliance = ComplianceManager(
    # Standards to validate against
    standards=["SOC2", "ISO27001", "GDPR", "HIPAA"],
    
    # Check intervals
    continuous_validation=True,
    validation_interval=3600,
    
    # Reporting
    report_format="pdf",
    report_retention_days=2555,  # 7 years for compliance
    
    # Evidence collection
    collect_evidence=True,
    evidence_retention_days=365
)

# Run compliance check
report = compliance.validate(
    standard="SOC2",
    scope="all",
    from_date="2024-01-01",
    to_date="2024-12-31"
)

# View results
print(f"Standard: {report.standard}")
print(f"Status: {report.status}")  # "compliant", "non_compliant", "partial"
print(f"Score: {report.score}%")
print(f"Controls checked: {report.controls_checked}")
print(f"Controls passed: {report.controls_passed}")
print(f"Controls failed: {report.controls_failed}")

# View failed controls
for control in report.failed_controls:
    print(f"\nControl: {control.id}")
    print(f"  Description: {control.description}")
    print(f"  Failure reason: {control.failure_reason}")
    print(f"  Remediation: {control.remediation}")

# Generate evidence report
evidence = compliance.collect_evidence(
    control_ids=["CC6.1", "CC6.2", "CC6.3"],
    format="zip"
)
```

### Security Policies

```python
from nexus.security.policies import (
    SecurityPolicy,
    PolicyEnforcer,
    PolicyViolation
)

# Define security policy
policy = SecurityPolicy(
    name="data-protection-policy",
    version="1.0",
    controls=[
        Control(
            id="DP-001",
            name="Encryption at Rest",
            description="All data must be encrypted at rest",
            enforcement="mandatory",
            validation="encryption_check"
        ),
        Control(
            id="DP-002",
            name="Encryption in Transit",
            description="All data must be encrypted in transit",
            enforcement="mandatory",
            validation="tls_check"
        ),
        Control(
            id="DP-003",
            name="Access Logging",
            description="All data access must be logged",
            enforcement="mandatory",
            validation="audit_check"
        ),
        Control(
            id="DP-004",
            name="Data Retention",
            description="Data must be retained according to policy",
            enforcement="mandatory",
            validation="retention_check"
        ),
        Control(
            id="DP-005",
            name="Data Masking",
            description="Sensitive data must be masked in logs",
            enforcement="advisory",
            validation="masking_check"
        )
    ]
)

# Enforce policy
enforcer = PolicyEnforcer(
    policy=policy,
    mode="enforce",  # "audit", "enforce", "warn"
    exceptions={
        "emergency_breakglass": {
            "requires_approval": True,
            "max_duration_hours": 4,
            "audit_required": True
        }
    }
)

# Check resource compliance
violations = enforcer.check(resource=dataset)
if violations:
    for violation in violations:
        print(f"Violation: {violation.control_id}")
        print(f"  Message: {violation.message}")
        print(f"  Remediation: {violation.remediation}")

# Request exception
exception = enforcer.request_exception(
    resource_id="dataset:sensitive",
    control_id="DP-005",
    reason="Testing",
    duration_hours=2
)
```

## API Reference

### Authentication Classes

```python
class APIKeyManager:
    """Manager for API key authentication."""
    
    def __init__(
        self,
        algorithm: str = "sha256",
        key_length: int = 64,
        prefix: str = "nexus_",
        hash_salt: str,
        expiry_days: int = 90,
        scopes: List[str] = None
    ):
        """Initialize API key manager.
        
        Args:
            algorithm: Hash algorithm for key storage
            key_length: Length of generated keys
            prefix: Prefix for key IDs
            hash_salt: Salt for key hashing
            expiry_days: Default key expiry
            scopes: Default scopes for keys
        """
        
    def create_key(
        self,
        name: str,
        user_id: str,
        scopes: List[str] = None,
        expires_in_days: int = None,
        rate_limit: int = None
    ) -> APIKey:
        """Create new API key.
        
        Args:
            name: Human-readable key name
            user_id: User identifier
            scopes: Permission scopes
            expires_in_days: Custom expiry
            rate_limit: Custom rate limit
            
        Returns:
            APIKey object with key details
        """
        
    def verify_key(self, key: str) -> User:
        """Verify API key and return user.
        
        Args:
            key: API key to verify
            
        Returns:
            User object if valid
            
        Raises:
            InvalidKeyError: If key is invalid
            ExpiredKeyError: If key is expired
        """
        
    def rotate_key(
        self,
        old_key_id: str,
        copy_scopes: bool = True,
        keep_old_for_hours: int = 0
    ) -> APIKey:
        """Rotate API key.
        
        Args:
            old_key_id: Key to rotate
            copy_scopes: Copy scopes from old key
            keep_old_hours: Grace period for old key
            
        Returns:
            New APIKey object
        """
        
    def revoke_key(self, key_id: str, reason: str = None) -> None:
        """Revoke API key.
        
        Args:
            key_id: Key to revoke
            reason: Revocation reason
        """
```

### Authorization Classes

```python
class RBACAuthorizer:
    """Role-based access control authorizer."""
    
    def __init__(
        self,
        roles: Dict[str, Permission],
        default_role: str = "viewer",
        super_role: str = "admin"
    ):
        """Initialize RBAC authorizer.
        
        Args:
            roles: Role definitions with permissions
            default_role: Default role for new users
            super_role: Role with full access
        """
        
    def check_permission(
        self,
        user_roles: List[str],
        action: str,
        resource: str,
        conditions: Dict = None
    ) -> bool:
        """Check if user can perform action on resource.
        
        Args:
            user_roles: User's roles
            action: Action to perform
            resource: Resource identifier
            conditions: Additional conditions
            
        Returns:
            True if allowed, False otherwise
        """
        
    def grant_permission(
        self,
        user_id: str,
        role: str,
        resource: str = None,
        expires_at: str = None
    ) -> None:
        """Grant permission to user.
        
        Args:
            user_id: User identifier
            role: Role to grant
            resource: Specific resource (optional)
            expires_at: Expiration timestamp
        """
        
    def revoke_permission(
        self,
        user_id: str,
        role: str,
        resource: str = None
    ) -> None:
        """Revoke permission from user.
        
        Args:
            user_id: User identifier
            role: Role to revoke
            resource: Specific resource
        """

class ABACAuthorizer:
    """Attribute-based access control authorizer."""
    
    def __init__(
        self,
        rules: List[AttributeRule],
        default_deny: bool = True
    ):
        """Initialize ABAC authorizer.
        
        Args:
            rules: Attribute-based rules
            default_deny: Default decision if no rule matches
        """
        
    def evaluate(
        self,
        subject: Subject,
        resource: Resource,
        action: str,
        context: Context
    ) -> Decision:
        """Evaluate access request.
        
        Args:
            subject: Requesting entity
            resource: Target resource
            action: Action to perform
            context: Request context
            
        Returns:
            Access decision with reason
        """
```

### Encryption Classes

```python
class EncryptionManager:
    """Manager for data encryption operations."""
    
    def __init__(
        self,
        algorithm: str = "aes-256-gcm",
        key_length: int = 32,
        key_provider: str = "local",
        **kwargs
    ):
        """Initialize encryption manager.
        
        Args:
            algorithm: Encryption algorithm
            key_length: Key length in bytes
            key_provider: Key management provider
        """
        
    def encrypt(
        self,
        data: bytes,
        key_id: str = "default",
        associated_data: Dict = None
    ) -> EncryptedData:
        """Encrypt data.
        
        Args:
            data: Plaintext data
            key_id: Key to use
            associated_data: Additional authenticated data
            
        Returns:
            Encrypted data with IV and tag
        """
        
    def decrypt(
        self,
        ciphertext: bytes,
        iv: bytes,
        tag: bytes,
        key_id: str = "default",
        associated_data: Dict = None
    ) -> bytes:
        """Decrypt data.
        
        Args:
            ciphertext: Encrypted data
            iv: Initialization vector
            tag: Authentication tag
            key_id: Key to use
            associated_data: Additional authenticated data
            
        Returns:
            Decrypted plaintext
        """
```

### Audit Classes

```python
class AuditLogger:
    """Logger for security audit events."""
    
    def __init__(
        self,
        enabled: bool = True,
        backend: str = "elasticsearch",
        **kwargs
    ):
        """Initialize audit logger.
        
        Args:
            enabled: Enable audit logging
            backend: Storage backend
        """
        
    def log(self, event: AuditEvent) -> None:
        """Log audit event.
        
        Args:
            event: Audit event to log
        """
        
    def log_batch(self, events: List[AuditEvent]) -> None:
        """Log batch of audit events.
        
        Args:
            events: List of events to log
        """
        
    def query(self, query: AuditQuery) -> AuditResults:
        """Query audit logs.
        
        Args:
            query: Query parameters
            
        Returns:
            Query results with events and aggregations
        """
        
    def generate_compliance_report(
        self,
        standard: str,
        start_date: str,
        end_date: str
    ) -> ComplianceReport:
        """Generate compliance report.
        
        Args:
            standard: Compliance standard
            start_date: Report start date
            end_date: Report end date
            
        Returns:
            Compliance report
        """
```

## Examples

### Complete Security Setup

```python
from nexus.security import SecurityManager

# Configure comprehensive security
security = SecurityManager(
    config=SecurityConfig(
        authentication=AuthenticationConfig(
            enabled=True,
            provider="oauth",
            api_key_rotation_days=90,
            mfa_required=True
        ),
        authorization=AuthorizationConfig(
            enabled=True,
            policy="rbac",
            rbac=RBACConfig(
                roles={
                    "admin": Permission.ALL,
                    "developer": Permission(
                        actions=["read", "write", "execute"],
                        resources=["models:*", "datasets:*"]
                    ),
                    "viewer": Permission(
                        actions=["read"],
                        resources=["models:read", "datasets:read"]
                    )
                }
            )
        ),
        encryption=EncryptionConfig(
            enabled=True,
            algorithm="aes-256-gcm",
            key_rotation_days=365,
            encrypt_at_rest=True,
            encrypt_in_transit=True
        ),
        rate_limiting=RateLimitingConfig(
            enabled=True,
            backend="redis",
            default_limits={
                "requests_per_minute": 1000,
                "tokens_per_minute": 100000
            }
        ),
        audit=AuditConfig(
            enabled=True,
            backend="elasticsearch",
            retention_days=365,
            log_level="info"
        )
    )
)

# Wrap application with security
app = security.wrap_application(app)

# Configure middleware
app.add_middleware(SecurityMiddleware)
app.add_middleware(AuthenticationMiddleware)
app.add_middleware(AuthorizationMiddleware)
app.add_middleware(RateLimitMiddleware)
app.add_middleware(AuditMiddleware)

# Add security headers
@app.middleware("http")
async def add_security_headers(request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = "max-age=31536000"
    return response
```

### Custom Authentication Provider

```python
from nexus.security.auth import BaseAuthProvider, AuthResult

class CustomAuthProvider(BaseAuthProvider):
    """Custom authentication provider."""
    
    def __init__(self, api_endpoint: str, api_key: str):
        self.api_endpoint = api_endpoint
        self.api_key = api_key
        
    async def authenticate(
        self,
        credentials: str,
        request: Request
    ) -> AuthResult:
        """Authenticate credentials.
        
        Args:
            credentials: Authentication credentials
            request: HTTP request
            
        Returns:
            Authentication result
        """
        # Validate credentials format
        if not credentials.startswith("Custom "):
            return AuthResult(
                success=False,
                error="Invalid credential format"
            )
        
        token = credentials[7:]  # Remove "Custom " prefix
        
        # Call custom auth service
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.api_endpoint}/validate",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={"token": token}
            )
            
        if response.status_code == 200:
            user_data = response.json()
            return AuthResult(
                success=True,
                user=User(
                    id=user_data["user_id"],
                    email=user_data["email"],
                    roles=user_data["roles"]
                )
            )
        else:
            return AuthResult(
                success=False,
                error="Invalid credentials"
            )
    
    def get_scheme(self) -> str:
        """Return authentication scheme name."""
        return "Custom"
```

### Security Event Handler

```python
from nexus.security.events import SecurityEventHandler

class CustomEventHandler(SecurityEventHandler):
    """Custom security event handler."""
    
    async def handle_login_failure(
        self,
        event: AuditEvent,
        context: HandlerContext
    ) -> HandlerResponse:
        """Handle login failure event."""
        # Check for brute force
        failures = await self.get_recent_failures(
            ip=event.context.ip_address,
            window_minutes=15
        )
        
        if failures >= 5:
            # Block IP temporarily
            await self.block_ip(
                ip=event.context.ip_address,
                duration_minutes=30,
                reason="Brute force detection"
            )
            
            # Send alert
            await self.send_alert(
                severity="high",
                title="Potential brute force attack",
                description=f"5+ failed logins from {event.context.ip_address}"
            )
        
        return HandlerResponse(handled=True)
    
    async def handle_data_export(
        self,
        event: AuditEvent,
        context: HandlerContext
    ) -> HandlerResponse:
        """Handle data export event."""
        # Verify export is authorized
        if not await self.is_export_authorized(event):
            return HandlerResponse(
                handled=True,
                action="block",
                reason="Unauthorized data export"
            )
        
        # Add export to compliance tracking
        await self.track_export(event)
        
        return HandlerResponse(handled=True)
```

## See Also

- **[Architecture Overview](ARCHITECTURE.md)** - System architecture details
- **[API Reference](API_REFERENCE.md)** - Detailed API documentation
- **[Pipeline Guide](PIPELINE_GUIDE.md)** - Pipeline configuration
- **[Deployment Guide](DEPLOYMENT.md)** - Production deployment instructions
- **[Security Best Practices](../SECURITY_BEST_PRACTICES.md)** - Additional security guidelines
- **[Compliance Guide](../COMPLIANCE.md)** - Compliance requirements and procedures
