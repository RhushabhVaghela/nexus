"""
Unit tests for Nexus Monitoring System

This module contains comprehensive tests for the monitoring infrastructure including:
- Metrics server functionality
- Prometheus metrics collection
- Inference metrics collector
- Cache metrics collector
- System metrics collector

Author: Nexus Team
"""

import unittest
import time
import threading
from unittest.mock import Mock, patch, MagicMock, call
from http.server import HTTPServer
import json

from nexus.monitoring.metrics_server import (
    MetricsHandler,
    MetricsServer,
    start_metrics_server,
    stop_metrics_server,
    get_metrics_server,
    _global_server,
)

from nexus.monitoring.collectors import (
    PROMETHEUS_AVAILABLE,
    PSUTIL_AVAILABLE,
    PYNVML_AVAILABLE,
    MetricsCollector,
    InferenceMetricsCollector,
    CacheMetricsCollector,
    SystemMetricsCollector,
    get_collector,
    register_all_collectors,
)


class TestMetricsHandler(unittest.TestCase):
    """Test MetricsHandler HTTP request handler."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.handler = MetricsHandler
        self.handler.registry = None
    
    @patch('nexus.monitoring.metrics_server.generate_latest')
    def test_handle_metrics_success(self, mock_generate):
        """Test successful metrics endpoint handling."""
        mock_generate.return_value = b'fake_metrics_data'
        
        # Mock request handler setup
        mock_request = MagicMock()
        mock_client_address = ('127.0.0.1', 12345)
        mock_server = MagicMock()
        
        handler = self.handler(mock_request, mock_client_address, mock_server)
        handler.path = '/metrics'
        handler.wfile = MagicMock()
        
        handler.do_GET()
        
        handler.wfile.write.assert_called_with(b'fake_metrics_data')
    
    @patch('nexus.monitoring.metrics_server.generate_latest')
    def test_handle_metrics_error(self, mock_generate):
        """Test metrics endpoint error handling."""
        mock_generate.side_effect = Exception("Generation error")
        
        mock_request = MagicMock()
        mock_client_address = ('127.0.0.1', 12345)
        mock_server = MagicMock()
        
        handler = self.handler(mock_request, mock_client_address, mock_server)
        handler.path = '/metrics'
        handler.wfile = MagicMock()
        handler.error_message_format = "Error: %s"
        
        handler.do_GET()
        
        # Should handle error gracefully
        handler.wfile.write.assert_called()
    
    def test_handle_health(self):
        """Test health check endpoint."""
        mock_request = MagicMock()
        mock_client_address = ('127.0.0.1', 12345)
        mock_server = MagicMock()
        
        handler = self.handler(mock_request, mock_client_address, mock_server)
        handler.path = '/health'
        handler.wfile = MagicMock()
        
        handler.do_GET()
        
        handler.wfile.write.assert_called_with(b'{"status": "healthy"}')
    
    def test_handle_root(self):
        """Test root endpoint."""
        mock_request = MagicMock()
        mock_client_address = ('127.0.0.1', 12345)
        mock_server = MagicMock()
        
        handler = self.handler(mock_request, mock_client_address, mock_server)
        handler.path = '/'
        handler.wfile = MagicMock()
        
        handler.do_GET()
        
        handler.wfile.write.assert_called()
        response = handler.wfile.write.call_args[0][0]
        self.assertIn(b'Nexus Metrics Server', response)
    
    def test_handle_404(self):
        """Test 404 handling for unknown paths."""
        mock_request = MagicMock()
        mock_client_address = ('127.0.0.1', 12345)
        mock_server = MagicMock()
        
        handler = self.handler(mock_request, mock_client_address, mock_server)
        handler.path = '/unknown'
        handler.wfile = MagicMock()
        handler.error_message_format = "Not Found"
        
        handler.do_GET()
        
        handler.wfile.write.assert_called_with(b'Not Found')


class TestMetricsServer(unittest.TestCase):
    """Test MetricsServer class."""
    
    def test_initialization(self):
        """Test server initialization."""
        server = MetricsServer(host="localhost", port=9090)
        
        self.assertEqual(server.host, "localhost")
        self.assertEqual(server.port, 9090)
        self.assertFalse(server._running)
        self.assertIsNone(server._server)
        self.assertIsNone(server._thread)
    
    @patch('nexus.monitoring.metrics_server.HTTPServer')
    @patch('nexus.monitoring.metrics_server.threading.Thread')
    def test_start_server(self, mock_thread, mock_http_server):
        """Test starting the metrics server."""
        mock_server_instance = MagicMock()
        mock_http_server.return_value = mock_server_instance
        mock_thread_instance = MagicMock()
        mock_thread.return_value = mock_thread_instance
        
        server = MetricsServer(host="localhost", port=9090)
        server.start()
        
        mock_http_server.assert_called_once()
        mock_thread.assert_called_once()
        self.assertTrue(server._running)
    
    @patch('nexus.monitoring.metrics_server.HTTPServer')
    def test_start_already_running(self, mock_http_server):
        """Test starting already running server."""
        server = MetricsServer(host="localhost", port=9090)
        server._running = True
        
        server.start()
        
        mock_http_server.assert_not_called()
    
    @patch('nexus.monitoring.metrics_server.HTTPServer')
    @patch('nexus.monitoring.metrics_server.threading.Thread')
    def test_stop_server(self, mock_thread, mock_http_server):
        """Test stopping the metrics server."""
        mock_server_instance = MagicMock()
        mock_http_server.return_value = mock_server_instance
        mock_thread_instance = MagicMock()
        mock_thread.return_value = mock_thread_instance
        
        server = MetricsServer(host="localhost", port=9090)
        server.start()
        server.stop()
        
        mock_server_instance.shutdown.assert_called_once()
        mock_thread_instance.join.assert_called_once()
        self.assertFalse(server._running)
    
    def test_stop_not_running(self):
        """Test stopping not running server."""
        server = MetricsServer(host="localhost", port=9090)
        
        # Should not raise any errors
        server.stop()
        self.assertFalse(server._running)
    
    def test_get_registry(self):
        """Test getting Prometheus registry."""
        server = MetricsServer(host="localhost", port=9090)
        
        registry = server.get_registry()
        
        self.assertIsNotNone(registry)
    
    def test_is_running(self):
        """Test checking if server is running."""
        server = MetricsServer(host="localhost", port=9090)
        
        self.assertFalse(server.is_running())
        
        server._running = True
        self.assertTrue(server.is_running())


class TestStartStopMetricsServer(unittest.TestCase):
    """Test start_metrics_server and stop_metrics_server functions."""
    
    def tearDown(self):
        """Clean up global server."""
        global _global_server
        _global_server = None
    
    @patch('nexus.monitoring.metrics_server.MetricsServer')
    def test_start_metrics_server(self, mock_server_class):
        """Test starting global metrics server."""
        mock_server = MagicMock()
        mock_server_class.return_value = mock_server
        
        result = start_metrics_server(host="localhost", port=9090)
        
        mock_server_class.assert_called_once_with("localhost", 9090, None)
        mock_server.start.assert_called_once()
        self.assertEqual(result, mock_server)
    
    @patch('nexus.monitoring.metrics_server.MetricsServer')
    def test_start_metrics_server_existing(self, mock_server_class):
        """Test starting when server already exists."""
        mock_server = MagicMock()
        
        global _global_server
        _global_server = mock_server
        
        result = start_metrics_server()
        
        mock_server_class.assert_not_called()
        self.assertEqual(result, mock_server)
    
    @patch('nexus.monitoring.metrics_server.MetricsServer')
    def test_stop_metrics_server(self, mock_server_class):
        """Test stopping global metrics server."""
        mock_server = MagicMock()
        
        global _global_server
        _global_server = mock_server
        
        stop_metrics_server()
        
        mock_server.stop.assert_called_once()
        self.assertIsNone(_global_server)
    
    def test_stop_metrics_server_no_server(self):
        """Test stopping when no server exists."""
        global _global_server
        _global_server = None
        
        # Should not raise any errors
        stop_metrics_server()
    
    @patch('nexus.monitoring.metrics_server.MetricsServer')
    def test_get_metrics_server(self, mock_server_class):
        """Test getting global metrics server."""
        mock_server = MagicMock()
        
        global _global_server
        _global_server = mock_server
        
        result = get_metrics_server()
        
        self.assertEqual(result, mock_server)


class TestInferenceMetricsCollector(unittest.TestCase):
    """Test InferenceMetricsCollector class."""
    
    def test_initialization(self):
        """Test collector initialization."""
        collector = InferenceMetricsCollector()
        
        self.assertEqual(collector.namespace, "nexus")
        self.assertEqual(collector.subsystem, "inference")
        self.assertEqual(collector._request_count, 0)
        self.assertEqual(collector._error_count, 0)
        self.assertEqual(collector._total_tokens, 0)
    
    def test_collect(self):
        """Test collecting metrics."""
        collector = InferenceMetricsCollector()
        collector._request_count = 100
        collector._error_count = 5
        collector._total_tokens = 1000
        
        metrics = collector.collect()
        
        self.assertEqual(metrics['request_count'], 100)
        self.assertEqual(metrics['error_count'], 5)
        self.assertEqual(metrics['total_tokens'], 1000)
        self.assertEqual(metrics['error_rate'], 0.05)
    
    def test_record_request_success(self):
        """Test recording successful request."""
        collector = InferenceMetricsCollector()
        
        collector.record_request(
            model="test_model",
            duration_seconds=0.5,
            tokens_generated=10,
            success=True
        )
        
        self.assertEqual(collector._request_count, 1)
        self.assertEqual(collector._error_count, 0)
        self.assertEqual(collector._total_tokens, 10)
    
    def test_record_request_error(self):
        """Test recording failed request."""
        collector = InferenceMetricsCollector()
        
        collector.record_request(
            model="test_model",
            duration_seconds=0.5,
            tokens_generated=0,
            success=False,
            error_type="cuda_error"
        )
        
        self.assertEqual(collector._request_count, 1)
        self.assertEqual(collector._error_count, 1)
    
    def test_record_time_to_first_token(self):
        """Test recording time to first token."""
        collector = InferenceMetricsCollector()
        
        # Should not raise any errors even without metrics
        collector.record_time_to_first_token("test_model", 0.1)
    
    def test_set_requests_in_flight(self):
        """Test setting requests in flight."""
        collector = InferenceMetricsCollector()
        
        # Should not raise any errors even without metrics
        collector.set_requests_in_flight("test_model", 5)
    
    def test_set_tokens_per_second(self):
        """Test setting tokens per second."""
        collector = InferenceMetricsCollector()
        
        # Should not raise any errors even without metrics
        collector.set_tokens_per_second("test_model", 50.0)


class TestCacheMetricsCollector(unittest.TestCase):
    """Test CacheMetricsCollector class."""
    
    def test_initialization(self):
        """Test collector initialization."""
        collector = CacheMetricsCollector()
        
        self.assertEqual(collector.namespace, "nexus")
        self.assertEqual(collector.subsystem, "cache")
        self.assertEqual(collector._hits, 0)
        self.assertEqual(collector._misses, 0)
        self.assertEqual(collector._evictions, 0)
    
    def test_collect(self):
        """Test collecting metrics."""
        collector = CacheMetricsCollector()
        collector._hits = 80
        collector._misses = 20
        collector._evictions = 10
        
        metrics = collector.collect()
        
        self.assertEqual(metrics['hits'], 80)
        self.assertEqual(metrics['misses'], 20)
        self.assertEqual(metrics['evictions'], 10)
        self.assertEqual(metrics['hit_rate'], 0.8)
    
    def test_record_hit_memory(self):
        """Test recording memory hit."""
        collector = CacheMetricsCollector()
        
        collector.record_hit("activation_cache", tier="memory")
        
        self.assertEqual(collector._hits, 1)
        self.assertEqual(collector._misses, 0)
    
    def test_record_hit_disk(self):
        """Test recording disk hit."""
        collector = CacheMetricsCollector()
        
        collector.record_hit("activation_cache", tier="disk")
        
        self.assertEqual(collector._hits, 1)
    
    def test_record_miss(self):
        """Test recording cache miss."""
        collector = CacheMetricsCollector()
        
        collector.record_miss("activation_cache", tier="memory")
        
        self.assertEqual(collector._misses, 1)
    
    def test_record_eviction(self):
        """Test recording eviction."""
        collector = CacheMetricsCollector()
        
        collector.record_eviction("activation_cache")
        
        self.assertEqual(collector._evictions, 1)
    
    def test_set_cache_size(self):
        """Test setting cache size."""
        collector = CacheMetricsCollector()
        
        # Should not raise any errors even without metrics
        collector.set_cache_size("activation_cache", "memory", 1000000)
    
    def test_set_entries(self):
        """Test setting cache entries count."""
        collector = CacheMetricsCollector()
        
        # Should not raise any errors even without metrics
        collector.set_entries("activation_cache", "memory", 100)
    
    def test_update_hit_rate(self):
        """Test updating hit rate."""
        collector = CacheMetricsCollector()
        collector._hits = 80
        collector._misses = 20
        
        # Should not raise any errors even without metrics
        collector.update_hit_rate("activation_cache")


class TestSystemMetricsCollector(unittest.TestCase):
    """Test SystemMetricsCollector class."""
    
    def test_initialization(self):
        """Test collector initialization."""
        collector = SystemMetricsCollector()
        
        self.assertEqual(collector.namespace, "nexus")
        self.assertEqual(collector.subsystem, "system")
    
    @patch('nexus.monitoring.collectors.PSUTIL_AVAILABLE', True)
    @patch('nexus.monitoring.collectors.psutil')
    def test_collect_with_psutil(self, mock_psutil):
        """Test collecting system metrics with psutil."""
        mock_psutil.cpu_percent.return_value = 50.0
        mock_psutil.cpu_count.return_value = 8
        mock_psutil.virtual_memory.return_value = MagicMock(
            percent=75.0,
            used=8000000000,
            available=4000000000
        )
        
        collector = SystemMetricsCollector()
        metrics = collector.collect()
        
        self.assertEqual(metrics['cpu_percent'], 50.0)
        self.assertEqual(metrics['cpu_count'], 8)
        self.assertEqual(metrics['memory_percent'], 75.0)
    
    @patch('nexus.monitoring.collectors.PSUTIL_AVAILABLE', False)
    def test_collect_without_psutil(self):
        """Test collecting system metrics without psutil."""
        collector = SystemMetricsCollector()
        metrics = collector.collect()
        
        # Should still return empty dict without errors
        self.assertIsInstance(metrics, dict)
    
    @patch('nexus.monitoring.collectors.PSUTIL_AVAILABLE', True)
    @patch('nexus.monitoring.collectors.psutil')
    def test_update_metrics_with_psutil(self, mock_psutil):
        """Test updating Prometheus metrics with psutil."""
        mock_psutil.cpu_percent.return_value = [50.0, 60.0, 70.0, 80.0]
        mock_psutil.virtual_memory.return_value = MagicMock(
            percent=75.0,
            used=8000000000,
            free=2000000000,
            available=4000000000
        )
        
        collector = SystemMetricsCollector()
        
        # Create mock metrics
        collector._metrics = {
            'cpu_usage_percent': MagicMock(),
            'memory_usage_percent': MagicMock(),
            'memory_usage_bytes': MagicMock()
        }
        
        collector.update_metrics()
        
        collector._metrics['memory_usage_percent'].set.assert_called_with(75.0)


class TestGetCollector(unittest.TestCase):
    """Test get_collector function."""
    
    def tearDown(self):
        """Clean up collectors."""
        from nexus.monitoring.collectors import _collectors
        _collectors.clear()
    
    def test_get_collector_create_new(self):
        """Test creating new collector."""
        collector = get_collector("test_collector", InferenceMetricsCollector)
        
        self.assertIsInstance(collector, InferenceMetricsCollector)
    
    def test_get_collector_existing(self):
        """Test getting existing collector."""
        collector1 = get_collector("test_collector", InferenceMetricsCollector)
        collector2 = get_collector("test_collector", InferenceMetricsCollector)
        
        self.assertIs(collector1, collector2)
    
    def test_get_collector_no_class(self):
        """Test getting collector without class."""
        collector = get_collector("nonexistent_collector")
        
        self.assertIsNone(collector)


class TestRegisterAllCollectors(unittest.TestCase):
    """Test register_all_collectors function."""
    
    def tearDown(self):
        """Clean up collectors."""
        from nexus.monitoring.collectors import _collectors
        _collectors.clear()
    
    def test_register_all_collectors(self):
        """Test registering all collectors."""
        mock_registry = MagicMock()
        
        register_all_collectors(mock_registry)
        
        # Should have registered all collectors
        self.assertIn("inference", get_collector("inference"))
        self.assertIn("cache", str(type(get_collector("cache"))))
        self.assertIn("system", str(type(get_collector("system"))))


class TestMetricsCollectorBase(unittest.TestCase):
    """Test base MetricsCollector class."""
    
    def test_initialization(self):
        """Test base collector initialization."""
        collector = MetricsCollector(namespace="test", subsystem="inference")
        
        self.assertEqual(collector.namespace, "test")
        self.assertEqual(collector.subsystem, "inference")
        self.assertIsNone(collector.registry)
    
    def test_set_registry(self):
        """Test setting registry."""
        collector = MetricsCollector()
        mock_registry = MagicMock()
        
        collector.set_registry(mock_registry)
        
        self.assertEqual(collector.registry, mock_registry)


if __name__ == '__main__':
    unittest.main()
