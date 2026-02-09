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

Note: Imports are lazy-loaded to avoid the cascading torch/CUDA memory
      allocation that occurs when all submodules are imported eagerly.
      Submodules like metrics.py and memory_guard.py import torch at
      module level, which memory-maps ~5 GB of CUDA libraries each.
      Eager importing of all submodules inflates VmSize to 64+ GB,
      causing MemoryError under Linux heuristic overcommit (mode 0).
"""

import importlib
from typing import Any

# ─────────────────────────────────────────────────────────────────────────────
# Lazy import registry: attribute_name -> (submodule, object_name)
# ─────────────────────────────────────────────────────────────────────────────

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    # Circuit Breaker (.circuit_breaker)
    "CircuitBreaker": (".circuit_breaker", "CircuitBreaker"),
    "CircuitBreakerConfig": (".circuit_breaker", "CircuitBreakerConfig"),
    "CircuitBreakerOpen": (".circuit_breaker", "CircuitBreakerOpen"),
    "CircuitBreakerRegistry": (".circuit_breaker", "CircuitBreakerRegistry"),
    "CircuitState": (".circuit_breaker", "CircuitState"),
    "API_CALL_CIRCUIT": (".circuit_breaker", "API_CALL_CIRCUIT"),
    "DATABASE_CIRCUIT": (".circuit_breaker", "DATABASE_CIRCUIT"),
    "EXTERNAL_SERVICE_CIRCUIT": (".circuit_breaker", "EXTERNAL_SERVICE_CIRCUIT"),
    "MODEL_LOADER_CIRCUIT": (".circuit_breaker", "MODEL_LOADER_CIRCUIT"),
    "circuit_breaker": (".circuit_breaker", "circuit_breaker"),
    "get_circuit_breaker_registry": (
        ".circuit_breaker",
        "get_circuit_breaker_registry",
    ),
    # Retry Logic (.retry)
    "BackoffStrategy": (".retry", "BackoffStrategy"),
    "RetryConfig": (".retry", "RetryConfig"),
    "RetryExhausted": (".retry", "RetryExhausted"),
    "RetryStats": (".retry", "RetryStats"),
    "Retryable": (".retry", "Retryable"),
    "api_retry": (".retry", "api_retry"),
    "calculate_delay": (".retry", "calculate_delay"),
    "database_retry": (".retry", "database_retry"),
    "network_retry": (".retry", "network_retry"),
    "retry_call": (".retry", "retry_call"),
    "retry_call_async": (".retry", "retry_call_async"),
    "retry_with_backoff": (".retry", "retry_with_backoff"),
    "should_retry": (".retry", "should_retry"),
    "with_retry": (".retry", "with_retry"),
    # Metrics (.metrics) — HEAVY: imports torch
    "APIMetrics": (".metrics", "APIMetrics"),
    "GPUMetricsCollector": (".metrics", "GPUMetricsCollector"),
    "InferenceMetrics": (".metrics", "InferenceMetrics"),
    "LocalMetric": (".metrics", "LocalMetric"),
    "MetricType": (".metrics", "MetricType"),
    "MetricValue": (".metrics", "MetricValue"),
    "MetricsCollector": (".metrics", "MetricsCollector"),
    "MetricsManager": (".metrics", "MetricsManager"),
    "SystemMetrics": (".metrics", "SystemMetrics"),
    "TrainingMetrics": (".metrics", "TrainingMetrics"),
    "get_metrics_manager": (".metrics", "get_metrics_manager"),
    "increment_counter": (".metrics", "increment_counter"),
    "timed": (".metrics", "timed"),
    # Health Checks (.health) — HEAVY: imports memory_guard -> torch
    "APIHealthCheck": (".health", "APIHealthCheck"),
    "DatabaseHealthCheck": (".health", "DatabaseHealthCheck"),
    "GPUHealthCheck": (".health", "GPUHealthCheck"),
    "HealthCheck": (".health", "HealthCheck"),
    "HealthCheckRegistry": (".health", "HealthCheckRegistry"),
    "HealthCheckResult": (".health", "HealthCheckResult"),
    "HealthReport": (".health", "HealthReport"),
    "HealthStatus": (".health", "HealthStatus"),
    "ModelHealthCheck": (".health", "ModelHealthCheck"),
    "ProbeType": (".health", "ProbeType"),
    "SystemHealthCheck": (".health", "SystemHealthCheck"),
    "check_health": (".health", "check_health"),
    "check_health_async": (".health", "check_health_async"),
    "configure_health_checks": (".health", "configure_health_checks"),
    "get_health_registry": (".health", "get_health_registry"),
    "liveness_probe": (".health", "liveness_probe"),
    "readiness_probe": (".health", "readiness_probe"),
    "startup_probe": (".health", "startup_probe"),
    # Logging (.logging_config) — lightweight
    "GzipRotatingFileHandler": (".logging_config", "GzipRotatingFileHandler"),
    "JSONFormatter": (".logging_config", "JSONFormatter"),
    "LoggerConfig": (".logging_config", "LoggerConfig"),
    "TimedGzipRotatingFileHandler": (".logging_config", "TimedGzipRotatingFileHandler"),
    "init_logging": (".logging_config", "init_logging"),
    "log_benchmark_progress": (".logging_config", "log_benchmark_progress"),
    "log_completion": (".logging_config", "log_completion"),
    "log_header": (".logging_config", "log_header"),
    "log_progress": (".logging_config", "log_progress"),
    "log_structured": (".logging_config", "log_structured"),
    "setup_daily_logger": (".logging_config", "setup_daily_logger"),
    "setup_logger": (".logging_config", "setup_logger"),
    "setup_logger_advanced": (".logging_config", "setup_logger_advanced"),
    "setup_rotating_logger": (".logging_config", "setup_rotating_logger"),
    "shutdown_logging": (".logging_config", "shutdown_logging"),
    # Rate Limiter (.rate_limiter)
    "RateLimitConfig": (".rate_limiter", "RateLimitConfig"),
    "RateLimitExceeded": (".rate_limiter", "RateLimitExceeded"),
    "RateLimitResult": (".rate_limiter", "RateLimitResult"),
    "RateLimiterBackend": (".rate_limiter", "RateLimiterBackend"),
    "LocalRateLimiterBackend": (".rate_limiter", "LocalRateLimiterBackend"),
    "LocalTokenBucket": (".rate_limiter", "LocalTokenBucket"),
    "LocalSlidingWindow": (".rate_limiter", "LocalSlidingWindow"),
    "RedisRateLimiterBackend": (".rate_limiter", "RedisRateLimiterBackend"),
    "RateLimiter": (".rate_limiter", "RateLimiter"),
    "RateLimiterRegistry": (".rate_limiter", "RateLimiterRegistry"),
    "rate_limit": (".rate_limiter", "rate_limit"),
    "get_rate_limiter_registry": (".rate_limiter", "get_rate_limiter_registry"),
}


def __getattr__(name: str) -> Any:
    """Lazy-load attributes from submodules on first access."""
    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name]
        module = importlib.import_module(module_path, package=__name__)
        value = getattr(module, attr_name)
        # Cache in module namespace so __getattr__ isn't called again
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Support autocomplete/introspection by listing all lazy exports."""
    return list(__all__) + list(globals().keys())


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
    # Health Checks
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
    # Rate Limiter
    "RateLimitConfig",
    "RateLimitExceeded",
    "RateLimitResult",
    "RateLimiterBackend",
    "LocalRateLimiterBackend",
    "LocalTokenBucket",
    "LocalSlidingWindow",
    "RedisRateLimiterBackend",
    "RateLimiter",
    "RateLimiterRegistry",
    "rate_limit",
    "get_rate_limiter_registry",
]
