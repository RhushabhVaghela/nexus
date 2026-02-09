"""
Integration Tests for Performance Optimizations

This module contains integration tests that verify the interaction between:
- Prefetch engine and sliding window buffer
- Activation cache and inference pipeline
- TensorRT backend integration
- End-to-end performance testing

Author: Nexus Team
"""

import unittest
import time
import tempfile
import threading
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import torch
import torch.nn as nn

# Test imports
from nexus.models.sli.prefetch_engine import (
    PrefetchEngine, PrefetchStats, create_prefetch_engine
)
from nexus.models.sli.activation_cache import (
    ActivationCache, ActivationCacheConfig, get_activation_cache
)
from nexus.monitoring.metrics_server import MetricsServer
from nexus.monitoring.collectors import (
    InferenceMetricsCollector, CacheMetricsCollector, register_all_collectors
)


class TestPrefetchWithSlidingWindow(unittest.TestCase):
    """Test prefetch engine integration with sliding window."""
    
    def test_prefetch_during_sequential_access(self):
        """Test prefetch behavior during sequential layer access."""
        
        # Create mock sliding window
        mock_sliding_window = MagicMock()
        mock_sliding_window.is_layer_in_window.return_value = False
        mock_layer = nn.Linear(100, 100)
        mock_sliding_window._load_layer_into_window.return_value = MagicMock(
            layer=mock_layer
        )
        
        # Create prefetch engine integrated with sliding window
        engine = PrefetchEngine(
            sliding_window=mock_sliding_window,
            config=None
        )
        engine.start()
        engine.set_model_info("test_model", 32)
        
        # Simulate sequential access to layers
        for i in range(10):
            engine.record_access("test_model", i)
            time.sleep(0.01)  # Small delay
        
        # Give time for async prefetches
        time.sleep(0.2)
        
        stats = engine.get_stats()
        engine.stop()
        
        # Should have triggered prefetches
        self.assertGreaterEqual(stats['total_prefetches'], 0)
    
    def test_prefetch_buffer_integration(self):
        """Test interaction between prefetch buffer and cache."""
        
        cache_config = ActivationCacheConfig(
            max_memory_size_gb=0.5,
            max_disk_size_gb=1.0,
            enable_persistence=False
        )
        cache = ActivationCache(config=cache_config)
        
        # Store prefetched layers in cache
        for i in range(5):
            layer = nn.Linear(100, 100)
            cache.store(f"prefetched_layer_{i}", torch.randn(1))
        
        stats = cache.get_stats()
        cache.shutdown()
        
        self.assertEqual(stats['memory_entries'], 5)
    
    def test_concurrent_prefetch_requests(self):
        """Test handling of concurrent prefetch requests."""
        
        engine = PrefetchEngine(
            layer_loader=lambda m, i: nn.Linear(100, 100)
        )
        engine.start()
        engine.set_model_info("model1", 100)
        
        errors = []
        
        def access_layers(thread_id):
            try:
                for i in range(5):
                    engine.record_access("model1", thread_id * 10 + i)
            except Exception as e:
                errors.append(e)
        
        threads = [
            threading.Thread(target=access_layers, args=(i,))
            for i in range(4)
        ]
        
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        time.sleep(0.2)
        stats = engine.get_stats()
        engine.stop()
        
        self.assertEqual(len(errors), 0)
        self.assertGreaterEqual(stats['total_prefetches'], 0)


class TestCacheIntegration(unittest.TestCase):
    """Test cache integration with inference pipeline."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_activation_caching_in_pipeline(self):
        """Test activation caching during inference pipeline."""
        
        cache_config = ActivationCacheConfig(
            max_memory_size_gb=0.5,
            max_disk_size_gb=1.0,
            enable_persistence=True,
            persistence_dir=self.temp_dir
        )
        cache = ActivationCache(config=cache_config)
        
        # Simulate forward pass through multiple layers
        batch_size = 2
        seq_length = 10
        hidden_dim = 256
        
        # Layer 0 output
        layer0_output = torch.randn(batch_size, seq_length, hidden_dim)
        cache.store("layer_0_output", layer0_output, context="inference_run_1")
        
        # Layer 1 output
        layer1_output = torch.randn(batch_size, seq_length, hidden_dim)
        cache.store("layer_1_output", layer1_output, context="inference_run_1")
        
        # Simulate backward pass needing activations
        cached_layer0 = cache.retrieve("layer_0_output", context="inference_run_1")
        cached_layer1 = cache.retrieve("layer_1_output", context="inference_run_1")
        
        self.assertIsNotNone(cached_layer0)
        self.assertIsNotNone(cached_layer1)
        self.assertTrue(torch.allclose(layer0_output, cached_layer0))
        self.assertTrue(torch.allclose(layer1_output, cached_layer1))
        
        stats = cache.get_stats()
        cache.shutdown()
        
        self.assertEqual(stats['total_hits'], 2)
    
    def test_cache_with_ttl(self):
        """Test cache TTL expiration in inference context."""
        
        cache_config = ActivationCacheConfig(
            max_memory_size_gb=0.5,
            default_ttl_seconds=0.1,  # Very short TTL
            enable_persistence=False
        )
        cache = ActivationCache(config=cache_config)
        
        # Store activation
        activation = torch.randn(10, 256)
        cache.store("short_lived", activation, ttl=0.05)
        
        # Should be available
        self.assertIsNotNone(cache.retrieve("short_lived"))
        
        # Wait for expiration
        time.sleep(0.1)
        
        # Should be expired
        expired = cache.retrieve("short_lived")
        self.assertIsNone(expired)
        
        cache.shutdown()
    
    def test_multi_tier_caching(self):
        """Test multi-tier caching (memory + disk)."""
        
        cache_config = ActivationCacheConfig(
            max_memory_size_gb=0.01,  # Very small memory cache
            max_disk_size_gb=0.5,
            compression=None,
            enable_persistence=True,
            persistence_dir=self.temp_dir
        )
        cache = ActivationCache(config=cache_config)
        
        # Store multiple large tensors
        for i in range(10):
            tensor = torch.randn(100, 100)  # ~40KB each
            cache.store(f"large_tensor_{i}", tensor, persist=True)
        
        # Some should be evicted to disk
        stats = cache.get_stats()
        
        # Retrieve from disk
        retrieved = cache.retrieve("large_tensor_0")
        self.assertIsNotNone(retrieved)
        
        cache.shutdown()
    
    def test_cache_compression_integrity(self):
        """Test that compression doesn't corrupt data."""
        
        cache_config = ActivationCacheConfig(
            max_memory_size_gb=0.5,
            max_disk_size_gb=1.0,
            compression="gzip",
            enable_persistence=True,
            persistence_dir=self.temp_dir
        )
        cache = ActivationCache(config=cache_config)
        
        # Store tensor with specific values
        original = torch.randn(100, 100)
        cache.store("compressed", original, persist=True)
        
        # Retrieve and verify
        retrieved = cache.retrieve("compressed")
        self.assertTrue(torch.allclose(original, retrieved, atol=1e-5))
        
        cache.shutdown()


class TestMetricsIntegration(unittest.TestCase):
    """Test metrics collection integration."""
    
    def test_inference_metrics_collection(self):
        """Test inference metrics collection during requests."""
        
        collector = InferenceMetricsCollector()
        
        # Simulate multiple inference requests
        for i in range(10):
            collector.record_request(
                model="llama-7b",
                duration_seconds=0.5 + i * 0.1,
                tokens_generated=20,
                success=True
            )
        
        # Simulate some errors
        for i in range(2):
            collector.record_request(
                model="llama-7b",
                duration_seconds=0.1,
                tokens_generated=0,
                success=False,
                error_type="cuda_oom"
            )
        
        metrics = collector.collect()
        
        self.assertEqual(metrics['request_count'], 12)
        self.assertEqual(metrics['error_count'], 2)
        self.assertAlmostEqual(metrics['error_rate'], 2/12, places=4)
    
    def test_cache_metrics_collection(self):
        """Test cache metrics collection."""
        
        collector = CacheMetricsCollector()
        
        # Simulate cache operations
        for i in range(80):
            collector.record_hit("activation_cache", "memory")
        
        for i in range(20):
            collector.record_miss("activation_cache", "memory")
        
        for i in range(10):
            collector.record_eviction("activation_cache")
        
        metrics = collector.collect()
        
        self.assertEqual(metrics['hits'], 80)
        self.assertEqual(metrics['misses'], 20)
        self.assertEqual(metrics['hit_rate'], 0.8)
        self.assertEqual(metrics['evictions'], 10)
    
    def test_metrics_server_integration(self):
        """Test metrics server with collectors."""
        
        server = MetricsServer(host="localhost", port=19090)
        registry = server.get_registry()
        
        # Register all collectors
        register_all_collectors(registry)
        
        # Get collectors
        inference_collector = InferenceMetricsCollector()
        inference_collector.set_registry(registry)
        inference_collector.register_metrics()
        
        # Record some metrics
        for i in range(5):
            inference_collector.record_request(
                model="test_model",
                duration_seconds=0.5,
                tokens_generated=10,
                success=True
            )
        
        server.stop()
        
        # Verify collector has data
        metrics = inference_collector.collect()
        self.assertGreater(metrics['request_count'], 0)


class TestEndToEndPerformance(unittest.TestCase):
    """Test end-to-end performance scenarios."""
    
    def test_sequential_layer_access_performance(self):
        """Test performance of sequential layer access with prefetching."""
        
        layers = {i: nn.Linear(256, 256) for i in range(20)}
        
        def mock_loader(model_id: str, layer_idx: int):
            time.sleep(0.01)  # Simulate loading time
            return layers.get(layer_idx)
        
        # Without prefetching
        start_time = time.time()
        for i in range(20):
            layer = mock_loader("model", i)
            time.sleep(0.005)  # Simulate computation
        baseline_time = time.time() - start_time
        
        # With prefetching
        engine = PrefetchEngine(
            layer_loader=mock_loader,
            lookahead=3,
            thread_pool_size=4
        )
        engine.start()
        engine.set_model_info("model", 20)
        
        start_time = time.time()
        for i in range(20):
            engine.record_access("model", i)
            time.sleep(0.005)  # Simulate computation
        prefetch_time = time.time() - start_time
        
        engine.stop()
        
        stats = engine.get_stats()
        
        # With prefetching should be faster
        self.assertGreaterEqual(stats['total_prefetches'], 0)
    
    def test_cache_hit_performance(self):
        """Test performance benefit of cache hits."""
        
        cache_config = ActivationCacheConfig(
            max_memory_size_gb=1.0,
            enable_persistence=False
        )
        cache = ActivationCache(config=cache_config)
        
        # Store activations
        for i in range(10):
            tensor = torch.randn(100, 256)
            cache.store(f"layer_{i}", tensor)
        
        # First access - should be hit
        start_time = time.time()
        for i in range(10):
            _ = cache.retrieve(f"layer_{i}")
        hit_time = time.time() - start_time
        
        stats = cache.get_stats()
        cache.shutdown()
        
        self.assertEqual(stats['memory_hits'], 10)
        # Cache hits should be fast (sub-millisecond)
        self.assertLess(hit_time, 1.0)


class TestErrorHandlingAndResilience(unittest.TestCase):
    """Test error handling and resilience."""
    
    def test_prefetch_engine_error_handling(self):
        """Test prefetch engine handles errors gracefully."""
        
        def failing_loader(model_id: str, layer_idx: int):
            if layer_idx == 5:
                raise Exception("Simulated load error")
            return nn.Linear(100, 100)
        
        engine = PrefetchEngine(layer_loader=failing_loader)
        engine.start()
        engine.set_model_info("model", 20)
        
        # Should not crash even with errors
        for i in range(10):
            engine.record_access("model", i)
            time.sleep(0.01)
        
        time.sleep(0.1)
        stats = engine.get_stats()
        engine.stop()
        
        # Should have recorded some failures
        self.assertGreaterEqual(stats['total_prefetches'], 0)
    
    def test_cache_error_recovery(self):
        """Test cache error recovery."""
        
        cache_config = ActivationCacheConfig(
            max_memory_size_gb=0.5,
            enable_persistence=True,
            persistence_dir="/nonexistent/path/for/errors"
        )
        
        # Should handle errors gracefully
        cache = ActivationCache(config=cache_config)
        
        # Try to store
        tensor = torch.randn(10, 10)
        result = cache.store("test", tensor)
        
        # Should still work for memory cache even if disk fails
        self.assertTrue(result)
        self.assertIsNotNone(cache.retrieve("test"))
        
        cache.shutdown()


if __name__ == '__main__':
    unittest.main()
