"""
Metrics and Monitoring Implementation for Nexus

Provides Prometheus-compatible metrics collection with custom metrics
for all components including GPU utilization tracking.
"""

import functools
import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union
import threading

logger = logging.getLogger(__name__)


# Try to import prometheus_client
try:
    from prometheus_client import (
        Counter, Histogram, Gauge, Info,
        CollectorRegistry, push_to_gateway,
        generate_latest, CONTENT_TYPE_LATEST
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logger.warning("prometheus_client not available, metrics will be local only")

# Try to import torch for GPU metrics
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class MetricType:
    """Metric types."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"
    INFO = "info"


@dataclass
class MetricValue:
    """Single metric value."""
    name: str
    value: float
    labels: Dict[str, str] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class LocalMetric:
    """Local in-memory metric storage."""
    
    def __init__(self, name: str, metric_type: str, description: str = ""):
        self.name = name
        self.metric_type = metric_type
        self.description = description
        self._values: Dict[str, MetricValue] = {}
        self._lock = threading.Lock()
    
    def _make_key(self, labels: Dict[str, str]) -> str:
        """Create unique key from labels."""
        if not labels:
            return "__default__"
        return ":".join(f"{k}={v}" for k, v in sorted(labels.items()))
    
    def inc(self, value: float = 1.0, labels: Optional[Dict[str, str]] = None) -> None:
        """Increment counter metric."""
        labels = labels or {}
        key = self._make_key(labels)
        
        with self._lock:
            if key in self._values:
                self._values[key].value += value
                self._values[key].timestamp = time.time()
            else:
                self._values[key] = MetricValue(self.name, value, labels)
    
    def set(self, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Set gauge metric."""
        labels = labels or {}
        key = self._make_key(labels)
        
        with self._lock:
            self._values[key] = MetricValue(self.name, value, labels)
    
    def observe(self, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Observe histogram value."""
        labels = labels or {}
        key = self._make_key(labels)
        
        with self._lock:
            if key not in self._values:
                self._values[key] = MetricValue(
                    self.name, 0, labels, 
                    metadata={"count": 0, "sum": 0.0, "values": []}
                )
            
            mv = self._values[key]
            mv.timestamp = time.time()
            if not hasattr(mv, 'metadata'):
                mv.metadata = {"count": 0, "sum": 0.0, "values": []}
            
            mv.metadata["count"] += 1
            mv.metadata["sum"] += value
            mv.metadata["values"].append(value)
            mv.value = mv.metadata["sum"] / mv.metadata["count"]
    
    def get_value(self, labels: Optional[Dict[str, str]] = None) -> Optional[float]:
        """Get current value."""
        labels = labels or {}
        key = self._make_key(labels)
        
        with self._lock:
            if key in self._values:
                return self._values[key].value
            return None
    
    def get_all_values(self) -> List[MetricValue]:
        """Get all values for this metric."""
        with self._lock:
            return list(self._values.values())


class MetricsCollector:
    """Metrics collector with Prometheus integration."""
    
    def __init__(
        self,
        namespace: str = "nexus",
        subsystem: Optional[str] = None,
        registry: Optional[Any] = None
    ):
        self.namespace = namespace
        self.subsystem = subsystem
        self.registry = registry
        self._metrics: Dict[str, Union[LocalMetric, Any]] = {}
        self._lock = threading.Lock()
        
        if not PROMETHEUS_AVAILABLE:
            self.registry = None
    
    def _make_name(self, name: str) -> str:
        """Create full metric name."""
        parts = [self.namespace]
        if self.subsystem:
            parts.append(self.subsystem)
        parts.append(name)
        return "_".join(parts)
    
    def create_counter(
        self,
        name: str,
        description: str = "",
        labelnames: Optional[List[str]] = None
    ) -> Union[Counter, LocalMetric]:
        """Create a counter metric."""
        full_name = self._make_name(name)
        
        with self._lock:
            if name in self._metrics:
                return self._metrics[name]
            
            if PROMETHEUS_AVAILABLE:
                counter = Counter(
                    full_name, description,
                    labelnames=labelnames or [],
                    registry=self.registry
                )
                self._metrics[name] = counter
                return counter
            else:
                metric = LocalMetric(full_name, MetricType.COUNTER, description)
                self._metrics[name] = metric
                return metric
    
    def create_gauge(
        self,
        name: str,
        description: str = "",
        labelnames: Optional[List[str]] = None
    ) -> Union[Gauge, LocalMetric]:
        """Create a gauge metric."""
        full_name = self._make_name(name)
        
        with self._lock:
            if name in self._metrics:
                return self._metrics[name]
            
            if PROMETHEUS_AVAILABLE:
                gauge = Gauge(
                    full_name, description,
                    labelnames=labelnames or [],
                    registry=self.registry
                )
                self._metrics[name] = gauge
                return gauge
            else:
                metric = LocalMetric(full_name, MetricType.GAUGE, description)
                self._metrics[name] = metric
                return metric
    
    def create_histogram(
        self,
        name: str,
        description: str = "",
        labelnames: Optional[List[str]] = None,
        buckets: Optional[List[float]] = None
    ) -> Union[Histogram, LocalMetric]:
        """Create a histogram metric."""
        full_name = self._make_name(name)
        
        with self._lock:
            if name in self._metrics:
                return self._metrics[name]
            
            if PROMETHEUS_AVAILABLE:
                kwargs = {
                    "labelnames": labelnames or [],
                    "registry": self.registry
                }
                if buckets:
                    kwargs["buckets"] = buckets
                
                histogram = Histogram(full_name, description, **kwargs)
                self._metrics[name] = histogram
                return histogram
            else:
                metric = LocalMetric(full_name, MetricType.HISTOGRAM, description)
                self._metrics[name] = metric
                return metric
    
    def create_info(
        self,
        name: str,
        description: str = ""
    ) -> Union[Info, LocalMetric]:
        """Create an info metric."""
        full_name = self._make_name(name)
        
        with self._lock:
            if name in self._metrics:
                return self._metrics[name]
            
            if PROMETHEUS_AVAILABLE:
                info = Info(full_name, description, registry=self.registry)
                self._metrics[name] = info
                return info
            else:
                metric = LocalMetric(full_name, MetricType.INFO, description)
                self._metrics[name] = metric
                return metric
    
    def get_metric(self, name: str) -> Optional[Union[LocalMetric, Any]]:
        """Get metric by name."""
        with self._lock:
            return self._metrics.get(name)
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all metrics."""
        with self._lock:
            return dict(self._metrics)
    
    def export_prometheus(self) -> bytes:
        """Export metrics in Prometheus format."""
        if PROMETHEUS_AVAILABLE:
            return generate_latest(self.registry or CollectorRegistry())
        else:
            # Generate simple text format
            lines = []
            for name, metric in self._metrics.items():
                if isinstance(metric, LocalMetric):
                    lines.append(f"# HELP {metric.name} {metric.description}")
                    lines.append(f"# TYPE {metric.name} {metric.metric_type}")
                    for mv in metric.get_all_values():
                        label_str = ",".join(f'{k}="{v}"' for k, v in mv.labels.items())
                        if label_str:
                            lines.append(f"{metric.name}{{{label_str}}} {mv.value}")
                        else:
                            lines.append(f"{metric.name} {mv.value}")
            return "\n".join(lines).encode()


class GPUMetricsCollector:
    """Collector for GPU metrics."""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics = metrics_collector
        self._setup_metrics()
    
    def _setup_metrics(self):
        """Setup GPU metrics."""
        self.gpu_utilization = self.metrics.create_gauge(
            "gpu_utilization_percent",
            "GPU utilization percentage",
            labelnames=["device", "gpu_name"]
        )
        self.gpu_memory_used = self.metrics.create_gauge(
            "gpu_memory_used_bytes",
            "GPU memory used in bytes",
            labelnames=["device", "gpu_name"]
        )
        self.gpu_memory_total = self.metrics.create_gauge(
            "gpu_memory_total_bytes",
            "GPU memory total in bytes",
            labelnames=["device", "gpu_name"]
        )
        self.gpu_temperature = self.metrics.create_gauge(
            "gpu_temperature_celsius",
            "GPU temperature in celsius",
            labelnames=["device", "gpu_name"]
        )
        self.gpu_power_draw = self.metrics.create_gauge(
            "gpu_power_draw_watts",
            "GPU power draw in watts",
            labelnames=["device", "gpu_name"]
        )
    
    def update(self):
        """Update GPU metrics."""
        if not TORCH_AVAILABLE:
            return
        
        try:
            if not torch.cuda.is_available():
                return
            
            for i in range(torch.cuda.device_count()):
                device_name = torch.cuda.get_device_name(i)
                labels = {"device": f"cuda:{i}", "gpu_name": device_name}
                
                # Memory metrics
                memory_stats = torch.cuda.memory_stats(i)
                allocated = memory_stats.get("allocated_bytes.all.current", 0)
                reserved = memory_stats.get("reserved_bytes.all.current", 0)
                
                self.gpu_memory_used.labels(**labels).set(allocated)
                self.gpu_memory_total.labels(**labels).set(reserved)
                
                # Utilization - requires pynvml for accurate readings
                try:
                    import pynvml
                    pynvml.nvmlInit()
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    
                    # Utilization
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    self.gpu_utilization.labels(**labels).set(util.gpu)
                    
                    # Temperature
                    temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                    self.gpu_temperature.labels(**labels).set(temp)
                    
                    # Power
                    power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert to watts
                    self.gpu_power_draw.labels(**labels).set(power)
                    
                except ImportError:
                    # Fallback to torch memory-based estimate
                    pass
                except Exception as e:
                    logger.debug(f"Error getting GPU metrics for device {i}: {e}")
                    
        except Exception as e:
            logger.error(f"Error updating GPU metrics: {e}")


class InferenceMetrics:
    """Metrics for model inference."""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics = metrics_collector
        self._setup_metrics()
    
    def _setup_metrics(self):
        """Setup inference metrics."""
        self.requests_total = self.metrics.create_counter(
            "inference_requests_total",
            "Total inference requests",
            labelnames=["model", "status"]
        )
        self.request_duration = self.metrics.create_histogram(
            "inference_request_duration_seconds",
            "Inference request duration",
            labelnames=["model"],
            buckets=[.005, .01, .025, .05, .075, .1, .25, .5, .75, 1.0, 2.5, 5.0, 7.5, 10.0]
        )
        self.tokens_generated = self.metrics.create_counter(
            "inference_tokens_generated_total",
            "Total tokens generated",
            labelnames=["model"]
        )
        self.tokens_per_second = self.metrics.create_gauge(
            "inference_tokens_per_second",
            "Tokens per second generation rate",
            labelnames=["model"]
        )
        self.queue_size = self.metrics.create_gauge(
            "inference_queue_size",
            "Current inference queue size"
        )
        self.batch_size = self.metrics.create_histogram(
            "inference_batch_size",
            "Inference batch size",
            buckets=[1, 2, 4, 8, 16, 32, 64]
        )
    
    def record_request(
        self,
        model: str,
        duration: float,
        tokens: int,
        status: str = "success"
    ):
        """Record inference request metrics."""
        self.requests_total.labels(model=model, status=status).inc()
        self.request_duration.labels(model=model).observe(duration)
        self.tokens_generated.labels(model=model).inc(tokens)
        
        if duration > 0:
            tps = tokens / duration
            self.tokens_per_second.labels(model=model).set(tps)
    
    @contextmanager
    def track_request(self, model: str):
        """Context manager to track request duration."""
        start = time.time()
        try:
            yield self
            self.record_request(model, time.time() - start, 0, "success")
        except Exception as e:
            self.record_request(model, time.time() - start, 0, "error")
            raise


class APIMetrics:
    """Metrics for API calls."""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics = metrics_collector
        self._setup_metrics()
    
    def _setup_metrics(self):
        """Setup API metrics."""
        self.requests_total = self.metrics.create_counter(
            "api_requests_total",
            "Total API requests",
            labelnames=["endpoint", "method", "status"]
        )
        self.request_duration = self.metrics.create_histogram(
            "api_request_duration_seconds",
            "API request duration",
            labelnames=["endpoint", "method"],
            buckets=[.001, .005, .01, .025, .05, .1, .25, .5, 1.0, 2.5]
        )
        self.request_size = self.metrics.create_histogram(
            "api_request_size_bytes",
            "API request size in bytes",
            labelnames=["endpoint"],
            buckets=[100, 1000, 10000, 100000, 1000000]
        )
        self.response_size = self.metrics.create_histogram(
            "api_response_size_bytes",
            "API response size in bytes",
            labelnames=["endpoint"],
            buckets=[100, 1000, 10000, 100000, 1000000]
        )
        self.active_connections = self.metrics.create_gauge(
            "api_active_connections",
            "Number of active connections"
        )
        self.rate_limit_hits = self.metrics.create_counter(
            "api_rate_limit_hits_total",
            "Total rate limit hits",
            labelnames=["endpoint"]
        )
    
    @contextmanager
    def track_request(self, endpoint: str, method: str = "GET"):
        """Context manager to track API request."""
        start = time.time()
        try:
            yield self
            self.requests_total.labels(endpoint=endpoint, method=method, status="200").inc()
        except Exception as e:
            status = getattr(e, 'status_code', '500')
            self.requests_total.labels(endpoint=endpoint, method=method, status=str(status)).inc()
            raise
        finally:
            self.request_duration.labels(endpoint=endpoint, method=method).observe(time.time() - start)


class TrainingMetrics:
    """Metrics for training operations."""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics = metrics_collector
        self._setup_metrics()
    
    def _setup_metrics(self):
        """Setup training metrics."""
        self.epochs_total = self.metrics.create_counter(
            "training_epochs_total",
            "Total training epochs",
            labelnames=["model", "stage"]
        )
        self.steps_total = self.metrics.create_counter(
            "training_steps_total",
            "Total training steps",
            labelnames=["model", "stage"]
        )
        self.loss = self.metrics.create_gauge(
            "training_loss",
            "Current training loss",
            labelnames=["model", "stage"]
        )
        self.learning_rate = self.metrics.create_gauge(
            "training_learning_rate",
            "Current learning rate",
            labelnames=["model"]
        )
        self.samples_per_second = self.metrics.create_gauge(
            "training_samples_per_second",
            "Training throughput",
            labelnames=["model"]
        )
        self.grad_norm = self.metrics.create_gauge(
            "training_grad_norm",
            "Gradient norm",
            labelnames=["model"]
        )


class SystemMetrics:
    """System-level metrics."""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics = metrics_collector
        self._setup_metrics()
        self._gpu_collector = GPUMetricsCollector(metrics_collector)
    
    def _setup_metrics(self):
        """Setup system metrics."""
        self.info = self.metrics.create_info(
            "system_info",
            "System information"
        )
        self.cpu_percent = self.metrics.create_gauge(
            "system_cpu_percent",
            "CPU utilization percentage"
        )
        self.memory_used = self.metrics.create_gauge(
            "system_memory_used_bytes",
            "System memory used in bytes"
        )
        self.memory_total = self.metrics.create_gauge(
            "system_memory_total_bytes",
            "System memory total in bytes"
        )
        self.disk_used = self.metrics.create_gauge(
            "system_disk_used_bytes",
            "Disk space used in bytes",
            labelnames=["mount"]
        )
        self.disk_total = self.metrics.create_gauge(
            "system_disk_total_bytes",
            "Disk space total in bytes",
            labelnames=["mount"]
        )
        self.uptime_seconds = self.metrics.create_counter(
            "system_uptime_seconds",
            "System uptime in seconds"
        )
    
    def update(self):
        """Update system metrics."""
        try:
            import psutil
            
            # CPU
            self.cpu_percent.set(psutil.cpu_percent(interval=None))
            
            # Memory
            mem = psutil.virtual_memory()
            self.memory_used.set(mem.used)
            self.memory_total.set(mem.total)
            
            # Disk
            for partition in psutil.disk_partitions():
                try:
                    usage = psutil.disk_usage(partition.mountpoint)
                    labels = {"mount": partition.mountpoint}
                    self.disk_used.labels(**labels).set(usage.used)
                    self.disk_total.labels(**labels).set(usage.total)
                except:
                    pass
            
            # GPU
            self._gpu_collector.update()
            
        except ImportError:
            pass
        except Exception as e:
            logger.error(f"Error updating system metrics: {e}")


class MetricsManager:
    """Central manager for all metrics."""
    
    def __init__(self):
        self.registry = CollectorRegistry() if PROMETHEUS_AVAILABLE else None
        self.collector = MetricsCollector(registry=self.registry)
        
        # Component metrics
        self.inference = InferenceMetrics(self.collector)
        self.api = APIMetrics(self.collector)
        self.training = TrainingMetrics(self.collector)
        self.system = SystemMetrics(self.collector)
        
        # General metrics
        self.errors_total = self.collector.create_counter(
            "errors_total",
            "Total errors",
            labelnames=["component", "error_type"]
        )
        self.warnings_total = self.collector.create_counter(
            "warnings_total",
            "Total warnings",
            labelnames=["component"]
        )
    
    def record_error(self, component: str, error: Exception):
        """Record an error."""
        error_type = type(error).__name__
        self.errors_total.labels(component=component, error_type=error_type).inc()
    
    def record_warning(self, component: str):
        """Record a warning."""
        self.warnings_total.labels(component=component).inc()
    
    def update_system_metrics(self):
        """Update all system metrics."""
        self.system.update()
    
    def get_prometheus_export(self) -> bytes:
        """Get metrics in Prometheus format."""
        return self.collector.export_prometheus()
    
    def push_to_gateway(self, gateway: str, job: str = "nexus"):
        """Push metrics to Prometheus push gateway."""
        if PROMETHEUS_AVAILABLE and self.registry:
            try:
                push_to_gateway(gateway, job=job, registry=self.registry)
            except Exception as e:
                logger.error(f"Failed to push metrics: {e}")


# Global metrics manager
_metrics_manager: Optional[MetricsManager] = None
_lock = threading.Lock()


def get_metrics_manager() -> MetricsManager:
    """Get or create global metrics manager."""
    global _metrics_manager
    if _metrics_manager is None:
        with _lock:
            if _metrics_manager is None:
                _metrics_manager = MetricsManager()
    return _metrics_manager


def timed(metric_name: str, labels: Optional[Dict[str, str]] = None):
    """Decorator to time function execution."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            manager = get_metrics_manager()
            histogram = manager.collector.create_histogram(
                f"{metric_name}_duration_seconds",
                f"Duration of {metric_name}"
            )
            
            start = time.time()
            try:
                return func(*args, **kwargs)
            finally:
                duration = time.time() - start
                if isinstance(histogram, LocalMetric):
                    histogram.observe(duration, labels)
                else:
                    if labels:
                        histogram.labels(**labels).observe(duration)
                    else:
                        histogram.observe(duration)
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            manager = get_metrics_manager()
            histogram = manager.collector.create_histogram(
                f"{metric_name}_duration_seconds",
                f"Duration of {metric_name}"
            )
            
            start = time.time()
            try:
                return await func(*args, **kwargs)
            finally:
                duration = time.time() - start
                if isinstance(histogram, LocalMetric):
                    histogram.observe(duration, labels)
                else:
                    if labels:
                        histogram.labels(**labels).observe(duration)
                    else:
                        histogram.observe(duration)
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else wrapper
    return decorator


def increment_counter(metric_name: str, labels: Optional[Dict[str, str]] = None, value: float = 1.0):
    """Increment a counter metric."""
    manager = get_metrics_manager()
    counter = manager.collector.create_counter(metric_name, f"Counter for {metric_name}")
    
    if isinstance(counter, LocalMetric):
        counter.inc(value, labels)
    else:
        if labels:
            counter.labels(**labels).inc(value)
        else:
            counter.inc(value)