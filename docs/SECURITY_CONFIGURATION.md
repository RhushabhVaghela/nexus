# Nexus API Security Configuration

## Overview

This document describes the comprehensive security measures implemented in the Nexus Universal Explainer API.

## Security Architecture

### Layer 1: Authentication

The API supports two authentication methods:

1. **JWT Tokens** (Recommended for web applications)
   - Format: `Authorization: Bearer <token>`
   - Auto-generated expiration (24 hours default)
   - Scoped permissions
   - Token revocation support

2. **API Keys** (Recommended for server-to-server)
   - Format: `Authorization: ApiKey <key>`
   - Prefix: `nx_`
   - Customizable rate limits
   - Expiration dates

### Layer 2: Rate Limiting

Implemented using the **Token Bucket Algorithm**:

| Endpoint | Requests/Second | Burst Size | Block Duration |
|----------|-----------------|------------|----------------|
| `/generate` | 10 | 20 | 300s |
| `/auth/*` | 5 | 10 | 300s |
| `/health` | 100 | 200 | 60s |

### Layer 3: Input Validation

Comprehensive security audit including:

- **Prompt Injection Detection**: Blocks attempts to override system instructions
- **Jailbreak Detection**: Identifies jailbreak patterns (DAN, developer mode, etc.)
- **Code Injection Detection**: Prevents XSS, JavaScript injection
- **SQL Injection Detection**: Blocks SQL keyword abuse
- **Command Injection Detection**: Prevents shell command injection
- **Content Filtering**: Blocks toxic content and PII

### Layer 4: Security Headers

All responses include these headers:

| Header | Value | Purpose |
|--------|-------|---------|
| `X-Frame-Options` | `DENY` | Prevents clickjacking |
| `X-Content-Type-Options` | `nosniff` | Prevents MIME sniffing |
| `X-XSS-Protection` | `1; mode=block` | XSS filter |
| `Strict-Transport-Security` | `max-age=31536000; includeSubDomains` | HSTS |
| `Content-Security-Policy` | `default-src 'self'` | CSP |
| `Referrer-Policy` | `strict-origin-when-cross-origin` | Referrer control |
| `Permissions-Policy` | `geolocation=(), microphone=(), camera=()` | Feature restrictions |

## Configuration

### Environment Variables

```bash
# CORS Configuration
ALLOWED_ORIGINS=http://localhost:3000,https://myapp.com

# Rate Limiting
RATE_LIMIT_RPS=10           # Requests per second
RATE_LIMIT_BURST=20         # Burst capacity

# Authentication
JWT_SECRET=your-secret-key   # Auto-generated if not set
JWT_EXPIRY_HOURS=24          # Token expiration
```

### CORS Configuration

```python
# In production, specify exact origins:
ALLOWED_ORIGINS=https://app.example.com,https://www.example.com

# Never use wildcard in production:
# WRONG: allow_origins=["*"]
# RIGHT: allow_origins=["https://your-domain.com"]
```

## API Endpoints

### Protected Endpoints

All endpoints except `/health` require authentication:

```bash
# Generate explanation (requires inference permission)
POST /generate
Authorization: Bearer <token>

# Health check (no auth required)
GET /health

# Security audit log (requires admin)
GET /security/audit-log
Authorization: Bearer <token>
```

### Authentication Endpoints

```bash
# Generate JWT token
POST /auth/token
Content-Type: application/json
{
    "user_id": "user123",
    "permissions": ["inference", "read"]
}

# Generate API key
POST /auth/api-key
Authorization: Bearer <admin-token>
Content-Type: application/json
{
    "name": "Production Key",
    "permissions": ["inference"],
    "rate_limit": 1000
}
```

## Security Logging

All security events are logged:

```json
{
  "timestamp": "2024-01-15T10:30:00",
  "level": "WARNING",
  "event": "auth_failure",
  "error": "Invalid or expired token",
  "ip": "192.168.1.1"
}
```

### Logged Events

- Authentication attempts (success/failure)
- Security violations
- Rate limit exceeded
- Generation requests
- Audit log access

## Error Responses

### Authentication Error (401)

```json
{
  "error": "Missing authorization header",
  "request_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

### Rate Limit Error (429)

```json
{
  "error": "Rate limit exceeded",
  "retry_after": 45,
  "request_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

### Security Violation (400)

```json
{
  "error": "Security violation detected",
  "violations": ["prompt_injection", "code_injection"],
  "request_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

## Testing Security

### Authentication Test

```bash
# Generate token
curl -X POST http://localhost:8000/auth/token \
  -H "Content-Type: application/json" \
  -d '{"user_id": "test_user", "permissions": ["inference"]}'

# Use token
curl -X POST http://localhost:8000/generate \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Explain quantum computing"}'
```

### Rate Limit Test

```bash
# Rapid requests to test rate limiting
for i in {1..25}; do
  curl -X POST http://localhost:8000/generate \
    -H "Authorization: Bearer <token>" \
    -d "prompt=test" &
done
```

### Security Header Test

```bash
curl -I http://localhost:8000/health
# Should include all security headers
```

## Security Checklist

- [ ] CORS origins restricted to specific domains
- [ ] JWT_SECRET set in production
- [ ] Rate limits appropriate for use case
- [ ] Admin API keys secured
- [ ] Audit logs monitored
- [ ] TLS/HTTPS enabled in production
- [ ] Security headers present
- [ ] Input validation enabled

## Best Practices

1. **Rotate Keys Regularly**: Use API key rotation every 90 days
2. **Monitor Logs**: Set up alerts for auth failures
3. **Least Privilege**: Grant minimum required permissions
4. **HTTPS Only**: Never expose API over HTTP in production
5. **Rate Limit Appropriately**: Adjust limits based on use case
