# Monitoring Setup Guide

Complete guide for setting up monitoring with Prometheus and Grafana for Nexus performance tracking.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [Prometheus Setup](#prometheus-setup)
- [Grafana Setup](#grafana-setup)
- [Metrics Reference](#metrics-reference)
- [Alerting](#alerting)
- [Dashboards](#dashboards)
- [Troubleshooting](#troubleshooting)

## Overview

Nexus provides comprehensive monitoring capabilities:

- **Metrics Collection**: Automatic collection of inference, cache, and system metrics
- **Prometheus Export**: Native Prometheus metrics endpoint
- **Grafana Integration**: Pre-built dashboards for visualization
- **Custom Metrics**: Easy extension for application-specific metrics

### Metrics Categories

1. **Inference Metrics**: Request count, latency, throughput, errors
2. **Cache Metrics**: Hit rate, size, evictions, compression ratio
3. **System Metrics**: CPU, GPU, memory, disk usage
4. **Custom Metrics**: Application-specific metrics

## Architecture

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Nexus      │────▶│  Prometheus  │────▶│   Grafana    │
│  Metrics     │     │   Server     │     │  Dashboards  │
└──────────────┘     └──────────────┘     └──────────────┘
       │                    │                    │
       ▼                    ▼                    ▼
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Inference   │     │  Time-Series │     │  Visualize   │
│  Cache       │     │   Database   │     │  Alert       │
│  System      │     │              │     │              │
└──────────────┘     └──────────────┘     └──────────────┘
```

## Quick Start

### 1. Start Metrics Server

```python
from nexus.monitoring.metrics_server import start_metrics_server

# Start metrics server
server = start_metrics_server(
    host="0.0.0.0",  # Listen on all interfaces
    port=9090        # Prometheus scrape port
)

print("Metrics server running at http://localhost:9090/metrics")
```

### 2. Register Collectors

```python
from nexus.monitoring.collectors import (
    InferenceMetricsCollector,
    CacheMetricsCollector,
    SystemMetricsCollector,
    register_all_collectors,
)
from prometheus_client import CollectorRegistry

# Create registry
registry = CollectorRegistry()

# Register all collectors
register_all_collectors(registry)

# Or register individually
inference_collector = InferenceMetricsCollector()
inference_collector.set_registry(registry)
inference_collector.register_metrics()
```

### 3. Record Metrics

```python
# Record inference request
inference_collector.record_request(
    model="llama-7b",
    duration_seconds=0.5,
    tokens_generated=25,
    success=True
)

# Record cache hit
cache_collector = CacheMetricsCollector()
cache_collector.record_hit("activation_cache", "memory")

# Update system metrics
system_collector = SystemMetricsCollector()
system_collector.update_metrics()
```

### 4. View Metrics

```bash
# View raw metrics
curl http://localhost:9090/metrics

# Check health
curl http://localhost:9090/health
```

## Prometheus Setup

### Docker Deployment

```yaml
# docker-compose.yml
version: '3.8'

services:
  prometheus:
    image: prom/prometheus:latest
    container_name: nexus-prometheus
    ports:
      - "9091:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.enable-lifecycle'

  grafana:
    image: grafana/grafana:latest
    container_name: nexus-grafana
    ports:
      - "3000:3000"
    volumes:
      - grafana_data:/var/lib/grafana
      - ./grafana/dashboards:/etc/grafana/provisioning/dashboards
      - ./grafana/datasources:/etc/grafana/provisioning/datasources
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin

volumes:
  prometheus_data:
  grafana_data:
```

### Prometheus Configuration

```yaml
# prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'nexus'
    static_configs:
      - targets: ['host.docker.internal:9090']
    metrics_path: /metrics
    scrape_interval: 5s
```

### Verification

```bash
# Start services
docker-compose up -d

# Check Prometheus targets
curl http://localhost:9091/api/v1/targets

# Query metrics
open http://localhost:9091/graph
```

## Grafana Setup

### Initial Setup

1. Access Grafana at `http://localhost:3000`
2. Login with admin/admin
3. Add Prometheus data source:
   - URL: `http://prometheus:9090`
   - Access: Server

### Dashboard Provisioning

```yaml
# grafana/datasources/datasource.yml
apiVersion: 1
datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
```

### Pre-built Dashboards

Create `grafana/dashboards/nexus-dashboard.json`:

```json
{
  "dashboard": {
    "title": "Nexus Performance",
    "panels": [
      {
        "title": "Request Rate",
        "targets": [{
          "expr": "rate(nexus_inference_requests_total[5m])",
          "legendFormat": "{{model}} - {{status}}"
        }],
        "type": "graph"
      },
      {
        "title": "Latency (p99)",
        "targets": [{
          "expr": "histogram_quantile(0.99, rate(nexus_inference_request_duration_seconds_bucket[5m]))",
          "legendFormat": "{{model}}"
        }],
        "type": "graph"
      },
      {
        "title": "Cache Hit Rate",
        "targets": [{
          "expr": "nexus_cache_hit_rate",
          "legendFormat": "{{cache_type}}"
        }],
        "type": "stat"
      },
      {
        "title": "GPU Utilization",
        "targets": [{
          "expr": "nexus_system_gpu_utilization_percent",
          "legendFormat": "GPU {{gpu}}"
        }],
        "type": "graph"
      }
    ]
  }
}
```

## Metrics Reference

### Inference Metrics

| Metric | Type | Description | Labels |
|--------|------|-------------|--------|
| `nexus_inference_requests_total` | Counter | Total inference requests | model, status |
| `nexus_inference_request_duration_seconds` | Histogram | Request latency | model |
| `nexus_inference_tokens_generated_total` | Counter | Total tokens generated | model |
| `nexus_inference_tokens_per_request` | Histogram | Tokens per request | model |
| `nexus_inference_tokens_per_second` | Gauge | Current throughput | model |
| `nexus_inference_errors_total` | Counter | Total errors | model, error_type |
| `nexus_inference_requests_in_flight` | Gauge | Active requests | model |
| `nexus_inference_time_to_first_token_seconds` | Histogram | TTFT | model |

### Cache Metrics

| Metric | Type | Description | Labels |
|--------|------|-------------|--------|
| `nexus_cache_hits_total` | Counter | Cache hits | cache_type, tier |
| `nexus_cache_misses_total` | Counter | Cache misses | cache_type, tier |
| `nexus_cache_evictions_total` | Counter | Evictions | cache_type |
| `nexus_cache_size_bytes` | Gauge | Current size | cache_type, tier |
| `nexus_cache_entries` | Gauge | Number of entries | cache_type, tier |
| `nexus_cache_hit_rate` | Gauge | Hit rate (0-1) | cache_type |
| `nexus_cache_utilization_ratio` | Gauge | Utilization (0-1) | cache_type, tier |

### System Metrics

| Metric | Type | Description | Labels |
|--------|------|-------------|--------|
| `nexus_system_cpu_usage_percent` | Gauge | CPU usage | cpu |
| `nexus_system_memory_usage_bytes` | Gauge | Memory usage | type |
| `nexus_system_memory_usage_percent` | Gauge | Memory percent | |
| `nexus_system_gpu_utilization_percent` | Gauge | GPU utilization | gpu |
| `nexus_system_gpu_memory_usage_bytes` | Gauge | GPU memory | gpu, type |
| `nexus_system_gpu_temperature_celsius` | Gauge | GPU temp | gpu |
| `nexus_system_gpu_power_draw_watts` | Gauge | GPU power | gpu |

### Prometheus Query Examples

```promql
# Request rate by model
sum by (model) (rate(nexus_inference_requests_total[5m]))

# Error rate
sum(rate(nexus_inference_errors_total[5m])) / sum(rate(nexus_inference_requests_total[5m]))

# p99 latency by model
histogram_quantile(0.99, 
  sum by (model, le) (rate(nexus_inference_request_duration_seconds_bucket[5m]))
)

# Cache hit rate by type
sum by (cache_type) (nexus_cache_hits_total) / 
  (sum by (cache_type) (nexus_cache_hits_total) + sum by (cache_type) (nexus_cache_misses_total))

# GPU memory usage
nexus_system_gpu_memory_usage_bytes{gpu="0", type="used"} / 
  nexus_system_gpu_memory_usage_bytes{gpu="0", type="total"}
```

## Alerting

### Prometheus Alerts

```yaml
# alerts.yml
groups:
  - name: nexus-alerts
    rules:
      - alert: HighErrorRate
        expr: |
          sum(rate(nexus_inference_errors_total[5m])) / 
          sum(rate(nexus_inference_requests_total[5m])) > 0.05
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High error rate detected"
          
      - alert: HighLatency
        expr: |
          histogram_quantile(0.99, 
            sum(rate(nexus_inference_request_duration_seconds_bucket[5m]))
          ) > 2
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High p99 latency"
          
      - alert: LowCacheHitRate
        expr: nexus_cache_hit_rate < 0.7
        for: 10m
        labels:
          severity: info
        annotations:
          summary: "Low cache hit rate"
          
      - alert: GPUHighTemperature
        expr: nexus_system_gpu_temperature_celsius > 85
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "GPU overheating"
          
      - alert: GPUOutOfMemory
        expr: |
          nexus_system_gpu_memory_usage_bytes{gpu="0", type="used"} / 
          nexus_system_gpu_memory_usage_bytes{gpu="0", type="total"} > 0.95
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "GPU memory nearly full"
```

### Alertmanager Configuration

```yaml
# alertmanager.yml
global:
  smtp_smarthost: 'localhost:587'
  smtp_from: 'alerts@example.com'

route:
  receiver: 'default'
  routes:
    - match:
        severity: critical
      receiver: 'pagerduty'

receivers:
  - name: 'default'
    email_configs:
      - to: 'oncall@example.com'
        
  - name: 'pagerduty'
    pagerduty_configs:
      - service_key: '<your-key>'
```

## Dashboards

### Inference Dashboard

Key panels:

- Request rate (requests/sec)
- Latency percentiles (p50, p90, p99)
- Error rate
- Token throughput
- Model-specific breakdowns

### Cache Dashboard

Key panels:

- Hit rate over time
- Memory usage
- Disk usage
- Eviction rate
- Compression ratio

### System Dashboard

Key panels:

- CPU usage per core
- Memory usage
- GPU utilization per device
- GPU memory usage
- GPU temperature
- Network I/O

### SLO Dashboard

Key panels:

- Availability (99.9% target)
- Latency SLO (p99 < 500ms)
- Error budget remaining

## Troubleshooting

### Metrics Not Appearing

**Check metrics server**:

```bash
curl http://localhost:9090/metrics
```

**Verify collector registration**:

```python
from nexus.monitoring.collectors import get_collector
collector = get_collector("inference")
print(collector._metrics)  # Should show registered metrics
```

**Check Prometheus targets**:

```bash
curl http://localhost:9091/api/v1/targets | jq .
```

### High Memory Usage

**Reduce metric retention**:

```yaml
# prometheus.yml
global:
  scrape_interval: 30s  # Increase interval
  
storage:
  tsdb:
    retention.time: 7d  # Reduce retention
```

**Limit cardinality**:

```python
# Avoid high-cardinality labels
# Bad: user_id, request_id
# Good: model, status
```

### Grafana Not Showing Data

**Check data source**:

```bash
curl http://localhost:9091/api/v1/query?query=up
```

**Verify dashboard queries**:

```promql
# Test query in Grafana Explore
nexus_inference_requests_total
```

## Best Practices

1. **Scrape Interval**: 5-15s for real-time, 30s+ for historical
2. **Label Cardinality**: Keep <1000 unique combinations
3. **Metric Names**: Use `nexus_<subsystem>_<metric>` format
4. **Dashboard Refresh**: 5-10s for live, 30s+ for summary
5. **Retention**: 15 days default, 30+ days for SLO tracking
6. **Alerts**: Use `for` clause to prevent flapping

## Advanced Usage

### Custom Metrics

```python
from nexus.monitoring.collectors import MetricsCollector
from prometheus_client import Counter, Histogram

class CustomMetricsCollector(MetricsCollector):
    def __init__(self):
        super().__init__(namespace="nexus", subsystem="custom")
    
    def register_metrics(self):
        self._create_counter(
            "custom_events_total",
            "Total custom events",
            ["event_type"]
        )
        self._create_histogram(
            "custom_operation_duration_seconds",
            "Custom operation duration",
            ["operation"]
        )
```

### Distributed Tracing Integration

```python
from nexus.monitoring.collectors import InferenceMetricsCollector

# Add trace ID to metrics
class TracedInferenceCollector(InferenceMetricsCollector):
    def record_request(self, model, duration_seconds, tokens_generated, success=True, trace_id=None):
        super().record_request(model, duration_seconds, tokens_generated, success)
        if trace_id:
            # Add to tracing system
            pass
```

## Additional Resources

- [Prometheus Documentation](https://prometheus.io/docs/)
- [Grafana Documentation](https://grafana.com/docs/)
- [Prometheus Best Practices](https://prometheus.io/docs/practices/)
- [Performance Optimizations Guide](./PERFORMANCE_OPTIMIZATIONS.md)
