"""
Nexus Utils - Production Readiness Modules

This package provides essential utilities for production deployments:

- Circuit Breaker Pattern: Fault tolerance for external service calls
- Retry Logic: Exponential backoff with jitter for resilient operations  
- Metrics: Prometheus-compatible metrics collection
- Health Checks: Kubernetes-compatible health probes
- Structured Logging: JSON logging with rotation and compression

Example:
    from nexus.utils import (
        CircuitBreaker,
        retry_with_backoff,
        get_metrics_manager,
        get_health_registry,
        setup_rotating_logger
    )
"""

# Circuit Breaker
from .circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerOpen,
    CircuitBreakerRegistry,
    CircuitState,
    API_CALL_CIRCUIT,
    DATABASE_CIRCUIT,
    EXTERNAL_SERVICE_CIRCUIT,
    MODEL_LOADER_CIRCUIT,
    circuit_breaker,
    get_circuit_breaker_registry,
)

# Retry Logic
from .retry import (
    BackoffStrategy,
    RetryConfig,
    RetryExhausted,
    RetryStats,
    Retryable,
    api_retry,
    calculate_delay,
    database_retry,
    network_retry,
    retry_call,
    retry_call_async,
    retry_with_backoff,
    should_retry,
    with_retry,
)

# Metrics
from .metrics import (
    APIMetrics,
    GPUMetricsCollector,
    InferenceMetrics,
    LocalMetric,
    MetricType,
    MetricValue,
    MetricsCollector,
    MetricsManager,
    SystemMetrics,
    TrainingMetrics,
    get_metrics_manager,
    increment_counter,
    timed,
)

# Health Checks
from .health import (
    APIHealthCheck,
    DatabaseHealthCheck,
    GPUHealthCheck,
    HealthCheck,
    HealthCheckRegistry,
    HealthCheckResult,
    HealthReport,
    HealthStatus,
    ModelHealthCheck,
    ProbeType,
    SystemHealthCheck,
    check_health,
    check_health_async,
    configure_health_checks,
    get_health_registry,
    liveness_probe,
    readiness_probe,
    startup_probe,
)

# Logging
from .logging_config import (
    GzipRotatingFileHandler,
    JSONFormatter,
    LoggerConfig,
    TimedGzipRotatingFileHandler,
    init_logging,
    log_benchmark_progress,
    log_completion,
    log_header,
    log_progress,
    log_structured,
    setup_daily_logger,
    setup_logger,
    setup_logger_advanced,
    setup_rotating_logger,
    shutdown_logging,
)

__all__ = [
    # Circuit Breaker
    "CircuitBreaker",
    "CircuitBreakerConfig",
    "CircuitBreakerOpen",
    "CircuitBreakerRegistry",
    "CircuitState",
    "API_CALL_CIRCUIT",
    "DATABASE_CIRCUIT",
    "EXTERNAL_SERVICE_CIRCUIT",
    "MODEL_LOADER_CIRCUIT",
    "circuit_breaker",
    "get_circuit_breaker_registry",
    # Retry
    "BackoffStrategy",
    "RetryConfig",
    "RetryExhausted",
    "RetryStats",
    "Retryable",
    "api_retry",
    "calculate_delay",
    "database_retry",
    "network_retry",
    "retry_call",
    "retry_call_async",
    "retry_with_backoff",
    "should_retry",
    "with_retry",
    # Metrics
    "APIMetrics",
    "GPUMetricsCollector",
    "InferenceMetrics",
    "LocalMetric",
    "MetricType",
    "MetricValue",
    "MetricsCollector",
    "MetricsManager",
    "SystemMetrics",
    "TrainingMetrics",
    "get_metrics_manager",
    "increment_counter",
    "timed",
    # Health
    "APIHealthCheck",
    "DatabaseHealthCheck",
    "GPUHealthCheck",
    "HealthCheck",
    "HealthCheckRegistry",
    "HealthCheckResult",
    "HealthReport",
    "HealthStatus",
    "ModelHealthCheck",
    "ProbeType",
    "SystemHealthCheck",
    "check_health",
    "check_health_async",
    "configure_health_checks",
    "get_health_registry",
    "liveness_probe",
    "readiness_probe",
    "startup_probe",
    # Logging
    "GzipRotatingFileHandler",
    "JSONFormatter",
    "LoggerConfig",
    "TimedGzipRotatingFileHandler",
    "init_logging",
    "log_benchmark_progress",
    "log_completion",
    "log_header",
    "log_progress",
    "log_structured",
    "setup_daily_logger",
    "setup_logger",
    "setup_logger_advanced",
    "setup_rotating_logger",
    "shutdown_logging",
]