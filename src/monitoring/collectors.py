"""
Metrics Collectors for Nexus Monitoring

Collects and exposes various metrics:
- Inference metrics (latency, throughput, tokens/sec)
- Cache metrics (hits, misses, size)
- System metrics (GPU/CPU usage, memory)

Author: Nexus Team
"""

import time
import threading
import logging
from typing import Dict, Optional, Any, List, Callable
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import torch

# Prometheus client
try:
    from prometheus_client import (
        Counter,
        Histogram,
        Gauge,
        Info,
        CollectorRegistry,
    )

    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    # Fallback stubs so type annotations don't raise NameError
    Counter = None
    Histogram = None
    Gauge = None
    Info = None
    CollectorRegistry = None

# Try to import psutil for system metrics
try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# Try to import pynvml for GPU metrics
try:
    import pynvml

    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False

logger = logging.getLogger(__name__)


class MetricsCollector(ABC):
    """Base class for metrics collectors."""

    def __init__(self, namespace: str = "nexus", subsystem: str = ""):
        self.namespace = namespace
        self.subsystem = subsystem
        self.registry: Optional[CollectorRegistry] = None
        self._metrics: Dict[str, Any] = {}

    def set_registry(self, registry: CollectorRegistry):
        """Set the Prometheus registry."""
        self.registry = registry

    @abstractmethod
    def register_metrics(self):
        """Register Prometheus metrics."""
        pass

    @abstractmethod
    def collect(self) -> Dict[str, Any]:
        """Collect current metric values."""
        pass

    def _create_counter(
        self, name: str, description: str, labels: List[str] = None
    ) -> Counter:
        """Create a counter metric."""
        if not PROMETHEUS_AVAILABLE:
            return None

        full_name = f"{self.subsystem}_{name}" if self.subsystem else name
        counter = Counter(
            full_name,
            description,
            labels or [],
            namespace=self.namespace,
            registry=self.registry,
        )
        self._metrics[name] = counter
        return counter

    def _create_histogram(
        self,
        name: str,
        description: str,
        labels: List[str] = None,
        buckets: List[float] = None,
    ) -> Histogram:
        """Create a histogram metric."""
        if not PROMETHEUS_AVAILABLE:
            return None

        full_name = f"{self.subsystem}_{name}" if self.subsystem else name
        histogram = Histogram(
            full_name,
            description,
            labels or [],
            buckets=buckets or Histogram.DEFAULT_BUCKETS,
            namespace=self.namespace,
            registry=self.registry,
        )
        self._metrics[name] = histogram
        return histogram

    def _create_gauge(
        self, name: str, description: str, labels: List[str] = None
    ) -> Gauge:
        """Create a gauge metric."""
        if not PROMETHEUS_AVAILABLE:
            return None

        full_name = f"{self.subsystem}_{name}" if self.subsystem else name
        gauge = Gauge(
            full_name,
            description,
            labels or [],
            namespace=self.namespace,
            registry=self.registry,
        )
        self._metrics[name] = gauge
        return gauge

    def _create_info(self, name: str, description: str) -> Info:
        """Create an info metric."""
        if not PROMETHEUS_AVAILABLE:
            return None

        full_name = f"{self.subsystem}_{name}" if self.subsystem else name
        info = Info(
            full_name, description, namespace=self.namespace, registry=self.registry
        )
        self._metrics[name] = info
        return info


class InferenceMetricsCollector(MetricsCollector):
    """
    Collector for inference-related metrics.

    Metrics:
    - Request count and latency
    - Token throughput
    - Generation length
    - Error rate
    """

    def __init__(self, namespace: str = "nexus"):
        super().__init__(namespace, "inference")
        self._request_count = 0
        self._error_count = 0
        self._total_tokens = 0
        self._lock = threading.Lock()

    def register_metrics(self):
        """Register inference metrics."""
        # Counters
        self._create_counter(
            "requests_total", "Total number of inference requests", ["model", "status"]
        )
        self._create_counter(
            "tokens_generated_total", "Total number of tokens generated", ["model"]
        )
        self._create_counter(
            "errors_total", "Total number of inference errors", ["model", "error_type"]
        )

        # Histograms
        self._create_histogram(
            "request_duration_seconds",
            "Inference request duration in seconds",
            ["model"],
            buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
        )
        self._create_histogram(
            "tokens_per_request",
            "Number of tokens per request",
            ["model"],
            buckets=[1, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000],
        )
        self._create_histogram(
            "time_to_first_token_seconds",
            "Time to first token in seconds",
            ["model"],
            buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
        )

        # Gauges
        self._create_gauge(
            "requests_in_flight",
            "Number of requests currently being processed",
            ["model"],
        )
        self._create_gauge(
            "tokens_per_second", "Current token generation rate", ["model"]
        )

        # Info
        self._create_info("build_info", "Build information about the inference server")

    def collect(self) -> Dict[str, Any]:
        """Collect current metrics."""
        with self._lock:
            return {
                "request_count": self._request_count,
                "error_count": self._error_count,
                "total_tokens": self._total_tokens,
                "error_rate": self._error_count / max(1, self._request_count),
            }

    def record_request(
        self,
        model: str,
        duration_seconds: float,
        tokens_generated: int,
        success: bool = True,
        error_type: str = "",
    ):
        """Record an inference request."""
        with self._lock:
            self._request_count += 1
            self._total_tokens += tokens_generated

            if not success:
                self._error_count += 1

        # Update Prometheus metrics
        if "requests_total" in self._metrics:
            status = "success" if success else "error"
            self._metrics["requests_total"].labels(model=model, status=status).inc()

        if success and "request_duration_seconds" in self._metrics:
            self._metrics["request_duration_seconds"].labels(model=model).observe(
                duration_seconds
            )

        if "tokens_generated_total" in self._metrics:
            self._metrics["tokens_generated_total"].labels(model=model).inc(
                tokens_generated
            )

        if "tokens_per_request" in self._metrics:
            self._metrics["tokens_per_request"].labels(model=model).observe(
                tokens_generated
            )

        if not success and "errors_total" in self._metrics:
            self._metrics["errors_total"].labels(
                model=model, error_type=error_type
            ).inc()

    def record_time_to_first_token(self, model: str, time_seconds: float):
        """Record time to first token."""
        if "time_to_first_token_seconds" in self._metrics:
            self._metrics["time_to_first_token_seconds"].labels(model=model).observe(
                time_seconds
            )

    def set_requests_in_flight(self, model: str, count: int):
        """Set number of requests in flight."""
        if "requests_in_flight" in self._metrics:
            self._metrics["requests_in_flight"].labels(model=model).set(count)

    def set_tokens_per_second(self, model: str, tps: float):
        """Set tokens per second."""
        if "tokens_per_second" in self._metrics:
            self._metrics["tokens_per_second"].labels(model=model).set(tps)


class CacheMetricsCollector(MetricsCollector):
    """
    Collector for cache-related metrics.

    Metrics:
    - Cache hits and misses
    - Cache size and utilization
    - Eviction rate
    """

    def __init__(self, namespace: str = "nexus"):
        super().__init__(namespace, "cache")
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._lock = threading.Lock()

    def register_metrics(self):
        """Register cache metrics."""
        # Counters
        self._create_counter(
            "hits_total", "Total number of cache hits", ["cache_type", "tier"]
        )
        self._create_counter(
            "misses_total", "Total number of cache misses", ["cache_type", "tier"]
        )
        self._create_counter(
            "evictions_total", "Total number of cache evictions", ["cache_type"]
        )

        # Gauges
        self._create_gauge(
            "size_bytes", "Current cache size in bytes", ["cache_type", "tier"]
        )
        self._create_gauge(
            "entries", "Number of entries in cache", ["cache_type", "tier"]
        )
        self._create_gauge("hit_rate", "Cache hit rate (0-1)", ["cache_type"])
        self._create_gauge(
            "utilization_ratio", "Cache utilization ratio (0-1)", ["cache_type", "tier"]
        )

    def collect(self) -> Dict[str, Any]:
        """Collect current metrics."""
        with self._lock:
            total = self._hits + self._misses
            return {
                "hits": self._hits,
                "misses": self._misses,
                "evictions": self._evictions,
                "hit_rate": self._hits / total if total > 0 else 0.0,
            }

    def record_hit(self, cache_type: str, tier: str = "memory"):
        """Record a cache hit."""
        with self._lock:
            self._hits += 1

        if "hits_total" in self._metrics:
            self._metrics["hits_total"].labels(cache_type=cache_type, tier=tier).inc()

    def record_miss(self, cache_type: str, tier: str = "memory"):
        """Record a cache miss."""
        with self._lock:
            self._misses += 1

        if "misses_total" in self._metrics:
            self._metrics["misses_total"].labels(cache_type=cache_type, tier=tier).inc()

    def record_eviction(self, cache_type: str):
        """Record a cache eviction."""
        with self._lock:
            self._evictions += 1

        if "evictions_total" in self._metrics:
            self._metrics["evictions_total"].labels(cache_type=cache_type).inc()

    def set_cache_size(self, cache_type: str, tier: str, bytes_used: int):
        """Set cache size in bytes."""
        if "size_bytes" in self._metrics:
            self._metrics["size_bytes"].labels(cache_type=cache_type, tier=tier).set(
                bytes_used
            )

    def set_entries(self, cache_type: str, tier: str, count: int):
        """Set number of cache entries."""
        if "entries" in self._metrics:
            self._metrics["entries"].labels(cache_type=cache_type, tier=tier).set(count)

    def update_hit_rate(self, cache_type: str):
        """Update hit rate gauge."""
        if "hit_rate" in self._metrics:
            with self._lock:
                total = self._hits + self._misses
                rate = self._hits / total if total > 0 else 0.0
            self._metrics["hit_rate"].labels(cache_type=cache_type).set(rate)


class SystemMetricsCollector(MetricsCollector):
    """
    Collector for system metrics.

    Metrics:
    - CPU usage
    - Memory usage
    - GPU usage and memory
    - Disk I/O
    - Network I/O
    """

    def __init__(self, namespace: str = "nexus"):
        super().__init__(namespace, "system")
        self._gpu_initialized = False
        if PYNVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self._gpu_initialized = True
                self._gpu_count = pynvml.nvmlDeviceGetCount()
            except:
                logger.warning("Failed to initialize NVML for GPU metrics")

    def register_metrics(self):
        """Register system metrics."""
        # CPU metrics
        self._create_gauge("cpu_usage_percent", "CPU usage percentage", ["cpu"])
        self._create_gauge("cpu_frequency_mhz", "CPU frequency in MHz", ["cpu"])

        # Memory metrics
        self._create_gauge(
            "memory_usage_bytes",
            "Memory usage in bytes",
            ["type"],  # used, free, available
        )
        self._create_gauge("memory_usage_percent", "Memory usage percentage")

        # GPU metrics
        if self._gpu_initialized:
            self._create_gauge(
                "gpu_utilization_percent", "GPU utilization percentage", ["gpu"]
            )
            self._create_gauge(
                "gpu_memory_usage_bytes",
                "GPU memory usage in bytes",
                ["gpu", "type"],  # used, free, total
            )
            self._create_gauge(
                "gpu_temperature_celsius", "GPU temperature in celsius", ["gpu"]
            )
            self._create_gauge(
                "gpu_power_draw_watts", "GPU power draw in watts", ["gpu"]
            )

        # Memory guard metrics (WSL-aware pressure + swap)
        self._create_gauge(
            "memory_pressure_level",
            "Memory pressure level (0=safe, 1=elevated, 2=high, 3=critical, 4=deadly)",
        )
        self._create_gauge(
            "swap_usage_percent",
            "Swap usage percentage",
        )

    def collect(self) -> Dict[str, Any]:
        """Collect current system metrics."""
        metrics = {}

        if PSUTIL_AVAILABLE:
            # CPU
            metrics["cpu_percent"] = psutil.cpu_percent(interval=None)
            metrics["cpu_count"] = psutil.cpu_count()

            # Memory
            mem = psutil.virtual_memory()
            metrics["memory_percent"] = mem.percent
            metrics["memory_used_gb"] = mem.used / 1e9
            metrics["memory_available_gb"] = mem.available / 1e9

        # GPU
        if self._gpu_initialized:
            metrics["gpu"] = []
            for i in range(self._gpu_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)

                gpu_info = {
                    "index": i,
                    "utilization": util.gpu,
                    "memory_used_gb": mem.used / 1e9,
                    "memory_total_gb": mem.total / 1e9,
                }

                # Try to get temperature
                try:
                    temp = pynvml.nvmlDeviceGetTemperature(
                        handle, pynvml.NVML_TEMPERATURE_GPU
                    )
                    gpu_info["temperature"] = temp
                except Exception as e:
                    logger.warning(f"Failed to get GPU temperature: {e}")

                # Try to get power
                try:
                    power = (
                        pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
                    )  # Convert mW to W
                    gpu_info["power_draw"] = power
                except Exception as e:
                    logger.warning(f"Failed to get GPU power usage: {e}")

                metrics["gpu"].append(gpu_info)

        # Memory guard: swap + pressure
        try:
            from src.utils.memory_guard import guard, MemoryPressure

            if guard is not None:
                snap = guard.snapshot()
                pressure_order = list(MemoryPressure)
                metrics["memory_pressure"] = pressure_order.index(snap.pressure)
                metrics["swap_used_gb"] = snap.swap_used_gb
                metrics["swap_total_gb"] = snap.swap_total_gb
                metrics["swap_percent"] = snap.swap_percent
        except Exception:
            pass

        return metrics

    def update_metrics(self):
        """Update Prometheus gauges with current values."""
        if not PSUTIL_AVAILABLE:
            return

        # CPU
        if "cpu_usage_percent" in self._metrics:
            cpu_percent = psutil.cpu_percent(interval=None, percpu=True)
            for i, percent in enumerate(cpu_percent):
                self._metrics["cpu_usage_percent"].labels(cpu=str(i)).set(percent)

        # Memory
        if "memory_usage_percent" in self._metrics:
            mem = psutil.virtual_memory()
            self._metrics["memory_usage_percent"].set(mem.percent)

            if "memory_usage_bytes" in self._metrics:
                self._metrics["memory_usage_bytes"].labels(type="used").set(mem.used)
                self._metrics["memory_usage_bytes"].labels(type="free").set(mem.free)
                self._metrics["memory_usage_bytes"].labels(type="available").set(
                    mem.available
                )

        # GPU
        if self._gpu_initialized:
            for i in range(self._gpu_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)

                if "gpu_utilization_percent" in self._metrics:
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    self._metrics["gpu_utilization_percent"].labels(gpu=str(i)).set(
                        util.gpu
                    )

                if "gpu_memory_usage_bytes" in self._metrics:
                    mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    self._metrics["gpu_memory_usage_bytes"].labels(
                        gpu=str(i), type="used"
                    ).set(mem.used)
                    self._metrics["gpu_memory_usage_bytes"].labels(
                        gpu=str(i), type="free"
                    ).set(mem.free)
                    self._metrics["gpu_memory_usage_bytes"].labels(
                        gpu=str(i), type="total"
                    ).set(mem.total)

                if "gpu_temperature_celsius" in self._metrics:
                    try:
                        temp = pynvml.nvmlDeviceGetTemperature(
                            handle, pynvml.NVML_TEMPERATURE_GPU
                        )
                        self._metrics["gpu_temperature_celsius"].labels(gpu=str(i)).set(
                            temp
                        )
                    except Exception as e:
                        logger.warning(f"Failed to get GPU temperature: {e}")

                if "gpu_power_draw_watts" in self._metrics:
                    try:
                        power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
                        self._metrics["gpu_power_draw_watts"].labels(gpu=str(i)).set(
                            power
                        )
                    except Exception as e:
                        logger.warning(f"Failed to get GPU power usage: {e}")

        # Memory guard: update pressure + swap gauges
        try:
            from src.utils.memory_guard import guard, MemoryPressure

            if guard is not None:
                snap = guard.snapshot()
                pressure_order = list(MemoryPressure)
                if "memory_pressure_level" in self._metrics:
                    self._metrics["memory_pressure_level"].set(
                        pressure_order.index(snap.pressure)
                    )
                if "swap_usage_percent" in self._metrics:
                    self._metrics["swap_usage_percent"].set(snap.swap_percent)
        except Exception:
            pass

    def __del__(self):
        """Cleanup."""
        if self._gpu_initialized:
            try:
                pynvml.nvmlShutdown()
            except Exception as e:
                logger.warning(f"Failed to shutdown NVML: {e}")


# Global collectors
_collectors: Dict[str, MetricsCollector] = {}
_collectors_lock = threading.Lock()


def get_collector(
    name: str, collector_class: type = None
) -> Optional[MetricsCollector]:
    """
    Get or create a global collector.

    Args:
        name: Collector name
        collector_class: Class to instantiate if collector doesn't exist

    Returns:
        MetricsCollector instance or None
    """
    with _collectors_lock:
        if name not in _collectors and collector_class is not None:
            _collectors[name] = collector_class()
        return _collectors.get(name)


def register_all_collectors(registry: CollectorRegistry):
    """Register all collectors with the given registry."""
    inference = get_collector("inference", InferenceMetricsCollector)
    cache = get_collector("cache", CacheMetricsCollector)
    system = get_collector("system", SystemMetricsCollector)

    for collector in [inference, cache, system]:
        if collector:
            collector.set_registry(registry)
            collector.register_metrics()
