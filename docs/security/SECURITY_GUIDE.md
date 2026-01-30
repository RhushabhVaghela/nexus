# Nexus Production Security Guide

## Overview

This document provides comprehensive security documentation for the Nexus production deployment. It covers all implemented security features, operational procedures, and best practices.

## Security Architecture

### Defense in Depth

Nexus implements multiple layers of security:

1. **Input Layer**: Validation, sanitization, injection detection
2. **Access Layer**: Rate limiting, authentication, authorization
3. **Service Layer**: Circuit breakers, resource limits
4. **Monitoring Layer**: Metrics, tracing, audit logging
5. **Infrastructure Layer**: Health checks, container security

## Security Components

### 1. Input Validation (src/security/audit.py)

#### InputValidator

Validates and sanitizes all user inputs:

- **Maximum Length**: 10,000 characters default
- **Encoding**: UTF-8/ASCII enforcement
- **Null Byte Detection**: Blocks null injection attempts
- **Control Character Filtering**: Removes suspicious characters

```python
from src.security.audit import InputValidator

validator = InputValidator(max_input_length=5000)
violations = validator.validate(user_input, context="api_request")
sanitized = validator.sanitize(user_input)
```

#### InjectionDetector

Detects various injection attacks:

- **Prompt Injection**: Detects attempts to override system instructions
- **Jailbreak Detection**: Identifies DAN and similar patterns
- **Code Injection**: Detects XSS, script injection
- **SQL Injection**: Identifies SQL keywords and patterns
- **Command Injection**: Detects shell command injection

```python
from src.security.audit import InjectionDetector

detector = InjectionDetector()
violations = detector.scan(user_input)
```

#### ContentFilter

Filters content for safety:

- **Blocklist**: Configurable blocked words/patterns
- **PII Detection**: Identifies emails, SSNs, credit cards
- **Auto-redaction**: Removes PII from outputs

```python
from src.security.audit import ContentFilter

filter = ContentFilter(blocklist={"spam", "abuse"})
violations = filter.check_content(text)
filtered, report = filter.filter_output(model_response)
```

### 2. Security Auditor

Central coordinator for all security checks:

```python
from src.security.audit import get_security_auditor, SecurityLevel

# Configure
auditor = configure_security_auditor(
    max_input_length=10000,
    blocklist={"blocked_word"},
    min_level=SecurityLevel.MEDIUM,
    block_on_violation=True
)

# Audit input
try:
    report = auditor.audit_input(user_input, context="chat_api")
    if not report.passed:
        logger.warning(f"Security violations: {report.violations}")
except SecurityException as e:
    # Handle blocked request
    pass

# Audit and filter output
filtered_output, report = auditor.audit_output(model_response)
```

### 3. Circuit Breakers (src/utils/circuit_breaker.py)

Protects against cascading failures:

#### States

- **CLOSED**: Normal operation, requests pass through
- **OPEN**: Failure threshold exceeded, requests fail fast
- **HALF_OPEN**: Testing recovery with limited requests

```python
from src.utils.circuit_breaker import (
    circuit_breaker,
    get_circuit_breaker_registry,
    MODEL_LOADER_CIRCUIT
)

# Decorator usage
@circuit_breaker("model_loader", failure_threshold=3, recovery_timeout=60.0)
def load_model(model_path):
    # Model loading logic
    pass

# Manual usage
registry = get_circuit_breaker_registry()
cb = registry.get("api_calls")

try:
    result = cb.call(external_api_call, arg1, arg2)
except CircuitBreakerOpen:
    # Handle circuit open - fail fast
    pass

# Get metrics
metrics = cb.metrics
all_metrics = registry.get_all_metrics()
```

#### Pre-configured Circuits

- **MODEL_LOADER_CIRCUIT**: 3 failures, 60s recovery
- **API_CALL_CIRCUIT**: 5 failures, 30s recovery
- **DATABASE_CIRCUIT**: 3 failures, 10s recovery
- **EXTERNAL_SERVICE_CIRCUIT**: 5 failures, 60s recovery

### 4. Rate Limiting (src/utils/rate_limiter.py)

Prevents abuse and ensures fair usage:

#### Algorithms

- **Token Bucket**: Burst-friendly rate limiting
- **Sliding Window**: Strict rate limiting with window tracking

```python
from src.utils.rate_limiter import (
    rate_limit,
    get_rate_limiter_registry,
    RateLimiterConfig,
    RedisRateLimiterBackend
)

# Configure rate limiter
config = RateLimitConfig(
    requests_per_second=10.0,
    burst_size=5,
    window_size=60.0
)

# Local backend
limiter = get_rate_limiter_registry().register(
    "api",
    config=config,
    algorithm="token_bucket"
)

# Redis backend for distributed rate limiting
redis_backend = RedisRateLimiterBackend(
    host="redis.example.com",
    port=6379
)
limiter = get_rate_limiter_registry().register(
    "api_global",
    backend=redis_backend,
    config=config
)

# Decorator usage
@rate_limit("api", user_id_arg="user_id", action="chat")
def process_chat(user_id: str, message: str):
    # Process chat
    pass

# Manual check
try:
    limiter.check(user_id="user123", action="chat")
    # Proceed with request
except RateLimitExceeded as e:
    # Return 429 Too Many Requests
    pass
```

#### Pre-configured Limits

- **API_RATE_LIMIT**: 100 req/min per user
- **INFERENCE_RATE_LIMIT**: 10 req/sec per user
- **GLOBAL_RATE_LIMIT**: 1000 req/sec total
- **LOGIN_RATE_LIMIT**: 5 req/min per IP (5 min block)

### 5. Metrics and Monitoring (src/utils/metrics.py)

Comprehensive observability:

#### GPU Metrics

- GPU utilization percentage
- Memory usage (allocated/total)
- Temperature (requires pynvml)
- Power draw

#### Inference Metrics

- Request count (total, by status)
- Request duration histogram
- Tokens generated
- Tokens per second
- Queue size
- Batch size distribution

#### API Metrics

- Request count (by endpoint, method, status)
- Request/response size
- Active connections
- Rate limit hits

#### Training Metrics

- Epochs and steps
- Loss and learning rate
- Samples per second
- Gradient norm

```python
from src.utils.metrics import get_metrics_manager, timed

manager = get_metrics_manager()

# Record inference
manager.inference.record_request(
    model="nexus-1.6",
    duration=2.5,
    tokens=150,
    status="success"
)

# Track with context manager
with manager.inference.track_request(model="nexus-1.6"):
    # Run inference
    pass

# Record error
manager.record_error("inference", exception)

# Update system metrics
manager.update_system_metrics()

# Export for Prometheus
prometheus_data = manager.get_prometheus_export()

# Decorator for timing
@timed("my_function", labels={"component": "parser"})
def parse_input(text):
    # Parsing logic
    pass
```

### 6. Distributed Tracing (src/utils/tracing.py)

OpenTelemetry-compatible tracing:

```python
from src.utils.tracing import (
    configure_tracing,
    trace_span,
    get_tracer_manager
)

# Configure
tracer = configure_tracing(
    service_name="nexus",
    jaeger_host="jaeger.example.com",
    jaeger_port=6831,
    sample_rate=1.0
)

# Decorator usage
@trace_span("generate_response", kind="server")
def generate_response(prompt):
    # Generation logic
    pass

# Manual usage
with tracer.start_span("database_query", kind="client") as span:
    span.set_attribute("db.table", "users")
    span.set_attribute("db.operation", "select")
    # Execute query
    span.add_event("query_complete", {"rows": 10})

# Get current span
span = tracer.get_current_span()
if span:
    span.set_attribute("user.id", user_id)
```

### 7. Health Checks (src/utils/health.py)

Kubernetes-compatible health monitoring:

```python
from src.utils.health import (
    configure_health_checks,
    check_health,
    liveness_probe,
    readiness_probe,
    startup_probe,
    HealthCheckRegistry,
    SystemHealthCheck,
    GPUHealthCheck
)

# Configure
registry = configure_health_checks(
    check_gpu=True,
    check_system=True,
    model_path="/models/nexus-1.6"
)

# Add custom check
class CustomCheck(HealthCheck):
    def check(self):
        # Custom health logic
        return HealthCheckResult(...)

registry.register(CustomCheck())

# Run checks
report = check_health()
print(report.to_dict())

# Kubernetes probes
status_code, body = liveness_probe()
status_code, body = readiness_probe()
status_code, body = startup_probe()
```

## Configuration

See `config/production.yaml` for all production settings.

### Environment Variables

```bash
# Security
NEXUS_SECURITY_ENABLED=true
NEXUS_SECURITY_BLOCK_ON_VIOLATION=true
NEXUS_SECURITY_MIN_LEVEL=medium

# Rate Limiting
NEXUS_RATE_LIMIT_BACKEND=redis
NEXUS_REDIS_HOST=localhost
NEXUS_REDIS_PORT=6379

# Tracing
NEXUS_TRACING_ENABLED=true
NEXUS_JAEGER_HOST=localhost
NEXUS_JAEGER_PORT=6831

# Metrics
NEXUS_METRICS_ENABLED=true
NEXUS_METRICS_PORT=9090
```

## Operational Procedures

### Deployment Checklist

- [ ] Security auditor configured with appropriate blocklist
- [ ] Circuit breakers registered for all external calls
- [ ] Rate limiting enabled (Redis for multi-instance)
- [ ] Metrics endpoint exposed
- [ ] Jaeger agent configured
- [ ] Health checks registered
- [ ] Kubernetes probes configured
- [ ] Logging configured (JSON format for production)

### Monitoring Dashboards

#### Key Metrics to Monitor

1. **Error Rates**
   - `nexus_errors_total` by component
   - Circuit breaker state changes
   - Rate limit hits

2. **Performance**
   - `nexus_inference_request_duration_seconds`
   - `nexus_inference_tokens_per_second`
   - Queue sizes

3. **Resources**
   - `nexus_system_cpu_percent`
   - `nexus_system_memory_used_bytes`
   - `nexus_gpu_utilization_percent`
   - `nexus_gpu_temperature_celsius`

4. **Security**
   - Security violations by type
   - Blocked requests
   - Injection attempts

### Alerting Rules

```yaml
# High error rate
- alert: HighErrorRate
  expr: rate(nexus_errors_total[5m]) > 0.05
  for: 5m
  severity: critical

# Circuit breaker open
- alert: CircuitBreakerOpen
  expr: nexus_circuit_breaker_state == 1  # OPEN state
  severity: warning

# High GPU temperature
- alert: HighGPUTemperature
  expr: nexus_gpu_temperature_celsius > 85
  severity: warning

# Rate limiting triggered
- alert: RateLimitTriggered
  expr: rate(nexus_api_rate_limit_hits_total[1m]) > 10
  severity: info
```

### Incident Response

#### Security Incident

1. **Immediate**: Block offending IP/user
2. **Short-term**: Review security logs
3. **Medium-term**: Update blocklists and patterns
4. **Long-term**: Security review and improvements

#### Performance Incident

1. **Immediate**: Check circuit breaker states
2. **Short-term**: Review metrics and traces
3. **Medium-term**: Adjust rate limits and resources
4. **Long-term**: Capacity planning

## Security Best Practices

### Input Handling

1. Always validate and sanitize inputs
2. Use the security auditor for all external inputs
3. Log security violations
4. Rate limit by user and globally

### Model Security

1. Use circuit breakers for model loading
2. Monitor GPU health
3. Validate model outputs with content filter
4. Log all inference requests

### API Security

1. Implement authentication/authorization
2. Use rate limiting on all endpoints
3. Validate all parameters
4. Return generic error messages to clients

### Infrastructure

1. Run with least privilege
2. Use secrets management
3. Enable audit logging
4. Regular security updates

## Troubleshooting

### Circuit Breaker Issues

```python
# Check all circuit breakers
from src.utils.circuit_breaker import get_circuit_breaker_registry
registry = get_circuit_breaker_registry()
for name, metrics in registry.get_all_metrics().items():
    print(f"{name}: {metrics['state']}, failures: {metrics['failure_count']}")

# Reset a circuit breaker
cb = registry.get("model_loader")
if cb:
    cb.reset()
```

### Rate Limiting Issues

```python
# Check rate limiter status
from src.utils.rate_limiter import get_rate_limiter_registry
registry = get_rate_limiter_registry()
for name in registry._limiters.keys():
    print(f"Rate limiter: {name}")
```

### Security Violations

```python
# Review security audit log
from src.security.audit import get_security_auditor
auditor = get_security_auditor()
summary = auditor.get_violation_summary()
print(f"Total audits: {summary['total_audits']}")
print(f"Violations by type: {summary['violations_by_type']}")

# Get recent failed audits
failed = auditor.get_audit_log(limit=10, failed_only=True)
```

## Compliance

### Data Protection

- PII detection and redaction
- Audit logging
- Data retention policies

### Access Control

- Authentication required
- Role-based access control
- API key management

### Audit Trail

- All requests logged
- Security events recorded
- Metrics retained

---

## Quick Reference

### Import Statements

```python
# Security
from src.security.audit import (
    get_security_auditor,
    InputValidator,
    InjectionDetector,
    ContentFilter,
    SecurityLevel,
    SecurityException
)

# Circuit Breaker
from src.utils.circuit_breaker import (
    circuit_breaker,
    get_circuit_breaker_registry,
    CircuitBreakerOpen,
    MODEL_LOADER_CIRCUIT,
    API_CALL_CIRCUIT
)

# Rate Limiting
from src.utils.rate_limiter import (
    rate_limit,
    get_rate_limiter_registry,
    RateLimitExceeded,
    RateLimitConfig,
    RedisRateLimiterBackend,
    API_RATE_LIMIT
)

# Metrics
from src.utils.metrics import (
    get_metrics_manager,
    timed,
    increment_counter
)

# Tracing
from src.utils.tracing import (
    configure_tracing,
    trace_span,
    get_tracer_manager
)

# Health
from src.utils.health import (
    configure_health_checks,
    check_health,
    liveness_probe,
    readiness_probe,
    startup_probe
)
```

### Common Operations

```bash
# Health check endpoints
curl http://localhost:8080/health/live    # Liveness
curl http://localhost:8080/health/ready   # Readiness
curl http://localhost:8080/health/startup # Startup

# Metrics endpoint
curl http://localhost:9090/metrics

# View logs
kubectl logs -f deployment/nexus

# Check circuit breaker
python -c "from src.utils.circuit_breaker import get_circuit_breaker_registry; print(get_circuit_breaker_registry().get_all_metrics())"
```

---

**Version**: 1.0.0

**Last Updated**: 2026-01-30
