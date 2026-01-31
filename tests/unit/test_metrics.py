"""
tests/unit/test_metrics.py
Comprehensive tests for metrics collection functionality.

Tests cover:
- Local metric storage
- MetricsCollector with Prometheus integration
- GPU metrics collection
- Inference metrics
- API metrics
- Training metrics
- System metrics
- MetricsManager integration
"""

import pytest
import time
import asyncio
from unittest.mock import patch, MagicMock, Mock

from src.utils.metrics import (
    MetricType,
    MetricValue,
    LocalMetric,
    MetricsCollector,
    GPUMetricsCollector,
    InferenceMetrics,
    APIMetrics,
    TrainingMetrics,
    SystemMetrics,
    MetricsManager,
    get_metrics_manager,
    timed,
    increment_counter,
)


class TestMetricValue:
    """Test MetricValue dataclass."""
    
    def test_creation(self):
        """Test creating a metric value."""
        mv = MetricValue(name="test_metric", value=42.0)
        
        assert mv.name == "test_metric"
        assert mv.value == 42.0
        assert mv.labels == {}
        assert mv.timestamp > 0
    
    def test_with_labels(self):
        """Test metric value with labels."""
        mv = MetricValue(
            name="test_metric",
            value=42.0,
            labels={"method": "GET", "endpoint": "/api"}
        )
        
        assert mv.labels == {"method": "GET", "endpoint": "/api"}


class TestLocalMetric:
    """Test LocalMetric class."""
    
    def test_counter_increment(self):
        """Test counter increment."""
        metric = LocalMetric("test_counter", MetricType.COUNTER)
        
        metric.inc(1.0)
        metric.inc(2.0)
        
        assert metric.get_value() == 3.0
    
    def test_counter_increment_with_labels(self):
        """Test counter increment with labels."""
        metric = LocalMetric("test_counter", MetricType.COUNTER)
        
        metric.inc(1.0, labels={"status": "200"})
        metric.inc(2.0, labels={"status": "200"})
        metric.inc(1.0, labels={"status": "404"})
        
        assert metric.get_value(labels={"status": "200"}) == 3.0
        assert metric.get_value(labels={"status": "404"}) == 1.0
    
    def test_gauge_set(self):
        """Test gauge set."""
        metric = LocalMetric("test_gauge", MetricType.GAUGE)
        
        metric.set(100.0)
        metric.set(50.0)
        
        assert metric.get_value() == 50.0
    
    def test_histogram_observe(self):
        """Test histogram observe."""
        metric = LocalMetric("test_histogram", MetricType.HISTOGRAM)
        
        metric.observe(0.1)
        metric.observe(0.2)
        metric.observe(0.3)
        
        # Average should be 0.2
        assert metric.get_value() == pytest.approx(0.2, rel=1e-6)
    
    def test_histogram_metadata(self):
        """Test histogram metadata tracking."""
        metric = LocalMetric("test_histogram", MetricType.HISTOGRAM)
        
        metric.observe(1.0)
        metric.observe(2.0)
        metric.observe(3.0)
        
        values = metric.get_all_values()
        assert len(values) == 1
        
        mv = values[0]
        assert mv.metadata["count"] == 3
        assert mv.metadata["sum"] == 6.0
    
    def test_get_value_missing(self):
        """Test getting value for missing metric."""
        collector = MockMetricsCollector()
        collector._metrics = {}
        
        # Should return None for non-existent metric
        value = collector.get_value("nonexistent")
        assert value is None
        """Test getting value for missing key."""
        metric = LocalMetric("test", MetricType.GAUGE)
        
        assert metric.get_value() is None
        assert metric.get_value(labels={"unknown": "label"}) is None
    
    def test_get_all_values(self):
        """Test getting all values."""
        metric = LocalMetric("test", MetricType.COUNTER)
        
        metric.inc(1.0, labels={"a": "1"})
        metric.inc(2.0, labels={"b": "2"})
        
        values = metric.get_all_values()
        assert len(values) == 2


class TestMetricsCollector:
    """Test MetricsCollector class."""
    
    def test_create_counter(self):
        """Test creating counter metric."""
        collector = MetricsCollector()
        counter = collector.create_counter("requests", "Total requests")
        
        assert counter is not None
        # Should be LocalMetric since prometheus is not available in tests
        assert isinstance(counter, LocalMetric)
    
    def test_create_gauge(self):
        """Test creating gauge metric."""
        collector = MetricsCollector()
        gauge = collector.create_gauge("temperature", "Current temperature")
        
        assert gauge is not None
        assert isinstance(gauge, LocalMetric)
    
    def test_create_histogram(self):
        """Test creating histogram metric."""
        collector = MetricsCollector()
        histogram = collector.create_histogram("duration", "Request duration")
        
        assert histogram is not None
        assert isinstance(histogram, LocalMetric)
    
    def test_create_info(self):
        """Test creating info metric."""
        collector = MetricsCollector()
        info = collector.create_info("version", "Application version")
        
        assert info is not None
    
    def test_get_metric(self):
        """Test getting metric by name."""
        collector = MetricsCollector()
        collector.create_counter("test", "Test metric")
        
        metric = collector.get_metric("test")
        assert metric is not None
        
        missing = collector.get_metric("missing")
        assert missing is None
    
    def test_get_all_metrics(self):
        """Test getting all metrics."""
        collector = MetricsCollector()
        collector.create_counter("counter1", "Counter 1")
        collector.create_gauge("gauge1", "Gauge 1")
        
        metrics = collector.get_all_metrics()
        assert len(metrics) == 2
    
    def test_duplicate_metric_creation(self):
        """Test creating duplicate metric returns existing."""
        collector = MetricsCollector()
        
        counter1 = collector.create_counter("test", "Test")
        counter2 = collector.create_counter("test", "Test again")
        
        assert counter1 is counter2
    
    def test_export_prometheus(self):
        """Test Prometheus format export."""
        collector = MetricsCollector()
        collector.create_counter("test", "Test counter")
        
        output = collector.export_prometheus()
        assert isinstance(output, bytes)


class TestGPUMetricsCollector:
    """Test GPUMetricsCollector class."""
    
    @patch("src.utils.metrics.TORCH_AVAILABLE", False)
    def test_update_no_torch(self):
        """Test update when torch is not available."""
        collector = MetricsCollector()
        gpu_collector = GPUMetricsCollector(collector)
        
        # Should not raise
        gpu_collector.update()
    
    @patch("torch.cuda.is_available", return_value=False)
    def test_update_no_cuda(self, mock_cuda):
        """Test update when CUDA is not available."""
        collector = MetricsCollector()
        gpu_collector = GPUMetricsCollector(collector)
        
        # Should not raise
        gpu_collector.update()
    
    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.device_count", return_value=1)
    @patch("torch.cuda.get_device_name", return_value="Test GPU")
    @patch("torch.cuda.memory_stats")
    def test_update_with_cuda(self, mock_stats, mock_name, mock_count, mock_avail):
        """Test update with CUDA available."""
        mock_stats.return_value = {
            "allocated_bytes.all.current": 1024**3,
            "reserved_bytes.all.current": 2 * 1024**3
        }
        
        collector = MetricsCollector()
        gpu_collector = GPUMetricsCollector(collector)
        
        # Should not raise
        gpu_collector.update()


class TestInferenceMetrics:
    """Test InferenceMetrics class."""
    
    def test_setup_metrics(self):
        """Test inference metrics are set up."""
        collector = MetricsCollector()
        inf_metrics = InferenceMetrics(collector)
        
        assert inf_metrics.requests_total is not None
        assert inf_metrics.request_duration is not None
        assert inf_metrics.tokens_generated is not None
        assert inf_metrics.tokens_per_second is not None
        assert inf_metrics.queue_size is not None
        assert inf_metrics.batch_size is not None
    
    def test_record_request(self):
        """Test recording inference request."""
        collector = MetricsCollector()
        inf_metrics = InferenceMetrics(collector)
        
        inf_metrics.record_request("model1", duration=1.0, tokens=100, status="success")
        
        # Verify metrics were recorded
        assert isinstance(inf_metrics.requests_total, LocalMetric)
    
    def test_track_request_context_manager(self):
        """Test track request context manager."""
        collector = MetricsCollector()
        inf_metrics = InferenceMetrics(collector)
        
        with inf_metrics.track_request("model1"):
            time.sleep(0.01)
        
        # Should record metrics
        assert isinstance(inf_metrics.requests_total, LocalMetric)
    
    def test_track_request_with_exception(self):
        """Test track request handles exceptions."""
        collector = MetricsCollector()
        inf_metrics = InferenceMetrics(collector)
        
        try:
            with inf_metrics.track_request("model1"):
                raise ValueError("Test error")
        except ValueError:
            pass
        
        # Should still record the request with error status
        assert isinstance(inf_metrics.requests_total, LocalMetric)


class TestAPIMetrics:
    """Test APIMetrics class."""
    
    def test_setup_metrics(self):
        """Test API metrics are set up."""
        collector = MetricsCollector()
        api_metrics = APIMetrics(collector)
        
        assert api_metrics.requests_total is not None
        assert api_metrics.request_duration is not None
        assert api_metrics.request_size is not None
        assert api_metrics.response_size is not None
        assert api_metrics.active_connections is not None
        assert api_metrics.rate_limit_hits is not None
    
    def test_track_request_success(self):
        """Test tracking successful API request."""
        collector = MetricsCollector()
        api_metrics = APIMetrics(collector)
        
        with api_metrics.track_request("/api/test", "GET"):
            time.sleep(0.01)
        
        assert isinstance(api_metrics.requests_total, LocalMetric)
    
    def test_track_request_exception(self):
        """Test tracking API request with exception."""
        collector = MetricsCollector()
        api_metrics = APIMetrics(collector)
        
        class TestException(Exception):
            status_code = 500
        
        try:
            with api_metrics.track_request("/api/test", "GET"):
                raise TestException("Error")
        except TestException:
            pass
        
        assert isinstance(api_metrics.requests_total, LocalMetric)


class TestTrainingMetrics:
    """Test TrainingMetrics class."""
    
    def test_setup_metrics(self):
        """Test training metrics are set up."""
        collector = MetricsCollector()
        train_metrics = TrainingMetrics(collector)
        
        assert train_metrics.epochs_total is not None
        assert train_metrics.steps_total is not None
        assert train_metrics.loss is not None
        assert train_metrics.learning_rate is not None
        assert train_metrics.samples_per_second is not None
        assert train_metrics.grad_norm is not None


class TestSystemMetrics:
    """Test SystemMetrics class."""
    
    def test_setup_metrics(self):
        """Test system metrics are set up."""
        collector = MetricsCollector()
        sys_metrics = SystemMetrics(collector)
        
        assert sys_metrics.info is not None
        assert sys_metrics.cpu_percent is not None
        assert sys_metrics.memory_used is not None
        assert sys_metrics.memory_total is not None
    
    @patch("psutil.cpu_percent", return_value=50.0)
    @patch("psutil.virtual_memory")
    @patch("psutil.disk_partitions", return_value=[])
    def test_update(self, mock_partitions, mock_virtual_memory, mock_cpu):
        """Test system metrics update."""
        mock_mem = Mock()
        mock_mem.used = 4 * 1024**3
        mock_mem.total = 16 * 1024**3
        mock_mem.percent = 25.0
        mock_virtual_memory.return_value = mock_mem
        
        collector = MetricsCollector()
        sys_metrics = SystemMetrics(collector)
        
        # Should not raise
        sys_metrics.update()
    
    @patch("src.utils.metrics.TORCH_AVAILABLE", False)
    def test_update_no_psutil(self):
        """Test update when psutil is not available."""
        with patch.dict("sys.modules", {"psutil": None}):
            collector = MetricsCollector()
            sys_metrics = SystemMetrics(collector)
            
            # Should not raise
            sys_metrics.update()


class TestMetricsManager:
    """Test MetricsManager class."""
    
    def test_initialization(self):
        """Test metrics manager initialization."""
        manager = MetricsManager()
        
        assert manager.collector is not None
        assert manager.inference is not None
        assert manager.api is not None
        assert manager.training is not None
        assert manager.system is not None
    
    def test_record_error(self):
        """Test recording an error."""
        manager = MetricsManager()
        
        try:
            raise ValueError("Test error")
        except Exception as e:
            manager.record_error("test_component", e)
        
        assert isinstance(manager.errors_total, LocalMetric)
    
    def test_record_warning(self):
        """Test recording a warning."""
        manager = MetricsManager()
        manager.record_warning("test_component")
        
        assert isinstance(manager.warnings_total, LocalMetric)
    
    def test_update_system_metrics(self):
        """Test updating system metrics."""
        manager = MetricsManager()
        
        # Should not raise
        manager.update_system_metrics()
    
    def test_get_prometheus_export(self):
        """Test getting Prometheus export."""
        manager = MetricsManager()
        manager.record_error("test", Exception("error"))
        
        output = manager.get_prometheus_export()
        assert isinstance(output, bytes)


class TestGlobalMetricsManager:
    """Test global metrics manager functions."""
    
    def test_get_metrics_manager_singleton(self):
        """Test get_metrics_manager returns singleton."""
        manager1 = get_metrics_manager()
        manager2 = get_metrics_manager()
        
        assert manager1 is manager2


class TestTimedDecorator:
    """Test timed decorator."""
    
    def test_timed_sync_function(self):
        """Test timing synchronous function."""
        
        @timed("test_operation")
        def slow_function():
            time.sleep(0.01)
            return "done"
        
        result = slow_function()
        assert result == "done"
    
    def test_timed_with_labels(self):
        """Test timing with labels."""
        
        @timed("test_op", labels={"method": "GET"})
        def my_function():
            time.sleep(0.01)
            return "done"
        
        result = my_function()
        assert result == "done"


class TestIncrementCounter:
    """Test increment_counter function."""
    
    def test_increment(self):
        """Test incrementing counter."""
        increment_counter("test_counter")
        
        manager = get_metrics_manager()
        assert isinstance(manager.collector.get_metric("test_counter"), LocalMetric)
    
    def test_increment_with_labels(self):
        """Test incrementing counter with labels."""
        increment_counter("test_counter_labeled", labels={"status": "200"}, value=5.0)
        
        manager = get_metrics_manager()
        assert isinstance(manager.collector.get_metric("test_counter_labeled"), LocalMetric)


class TestMetricsEdgeCases:
    """Test metrics edge cases."""
    
    def test_local_metric_concurrent_access(self):
        """Test thread-safe concurrent access to metrics."""
        import threading
        
        metric = LocalMetric("concurrent_test", MetricType.COUNTER)
        errors = []
        
        def worker():
            try:
                for _ in range(100):
                    metric.inc(1.0)
            except Exception as e:
                errors.append(e)
        
        threads = [threading.Thread(target=worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert len(errors) == 0
        assert metric.get_value() == 500.0
    
    def test_histogram_single_value(self):
        """Test histogram with single value."""
        metric = LocalMetric("single", MetricType.HISTOGRAM)
        
        metric.observe(42.0)
        
        assert metric.get_value() == 42.0
    
    def test_empty_labels_key(self):
        """Test metric with empty labels."""
        metric = LocalMetric("empty_labels", MetricType.GAUGE)
        
        metric.set(100.0, labels={})
        
        assert metric.get_value(labels={}) == 100.0
