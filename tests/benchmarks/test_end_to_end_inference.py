"""
End-to-End Inference Pipeline Benchmarks

Comprehensive benchmarks for full inference pipeline:
- Full pipeline with all optimizations enabled
- Component interaction performance
- Memory usage throughout pipeline
- Concurrent request handling
- Latency distribution analysis

Usage:
    python -m pytest tests/benchmarks/test_end_to_end_inference.py -v

Author: Nexus Team
"""

import pytest
import time
import statistics
import threading
import tempfile
from typing import Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock, patch, MagicMock

import torch
import torch.nn as nn

from src.models.sli.prefetch_engine import PrefetchEngine, create_prefetch_engine
from src.models.sli.activation_cache import ActivationCache, ActivationCacheConfig
from src.monitoring.metrics_server import MetricsServer
from src.monitoring.collectors import InferenceMetricsCollector


def benchmark_full_pipeline(
    num_layers: int = 24,
    batch_size: int = 1,
    seq_length: int = 512,
    num_iterations: int = 5
) -> Dict[str, float]:
    """
    Benchmark full inference pipeline with all optimizations.
    
    Returns:
        Dictionary with pipeline timing results
    """
    results = {
        'baseline_ms': 0,
        'with_prefetch_ms': 0,
        'with_cache_ms': 0,
        'with_all_ms': 0,
    }
    
    # Create mock model layers
    layers = {i: nn.Linear(512, 512) for i in range(num_layers)}
    
    def layer_loader(model_id: str, layer_idx: int) -> nn.Module:
        time.sleep(0.005)  # Simulate 5ms load time
        return layers[layer_idx]
    
    # Baseline: Sequential loading without any optimization
    times = []
    for _ in range(num_iterations):
        start = time.time()
        for i in range(num_layers):
            layer = layer_loader("model", i)
            # Simulate forward pass
            dummy_input = torch.randn(batch_size, seq_length, 512)
            _ = layer(dummy_input[:, 0, :])  # Process first token
            time.sleep(0.002)  # Compute time
        times.append((time.time() - start) * 1000)
    
    results['baseline_ms'] = statistics.mean(times)
    
    # With prefetching
    with tempfile.TemporaryDirectory() as temp_dir:
        cache_config = ActivationCacheConfig(
            max_memory_size_gb=0.5,
            max_disk_size_gb=1.0,
            enable_persistence=False
        )
        cache = ActivationCache(config=cache_config)
        
        engine = create_prefetch_engine(
            layer_loader=layer_loader,
            lookahead=3,
            thread_pool_size=4
        )
        engine.start()
        engine.set_model_info("model", num_layers)
        
        times = []
        for _ in range(num_iterations):
            start = time.time()
            for i in range(num_layers):
                engine.record_access("model", i)
                # Check prefetch buffer
                prefetched = engine.get_prefetched_layer(f"model_layer_{i}")
                if prefetched is None:
                    layer = layer_loader("model", i)
                else:
                    layer = prefetched
                
                dummy_input = torch.randn(batch_size, seq_length, 512)
                _ = layer(dummy_input[:, 0, :])
                time.sleep(0.002)
            times.append((time.time() - start) * 1000)
        
        results['with_all_ms'] = statistics.mean(times)
        
        engine.stop()
        cache.shutdown()
    
    # Calculate speedups
    results['prefetch_speedup'] = results['baseline_ms'] / results['with_all_ms']
    results['time_saved_ms'] = results['baseline_ms'] - results['with_all_ms']
    
    return results


def benchmark_concurrent_inference(
    num_requests: int = 10,
    concurrency: int = 4,
    num_layers: int = 12
) -> Dict[str, float]:
    """
    Benchmark concurrent request handling.
    
    Returns:
        Dictionary with concurrent request statistics
    """
    def layer_loader(model_id: str, layer_idx: int) -> nn.Module:
        time.sleep(0.003)
        return nn.Linear(256, 256)
    
    engine = create_prefetch_engine(
        layer_loader=layer_loader,
        thread_pool_size=8
    )
    engine.start()
    engine.set_model_info("model", num_layers)
    
    def process_request(request_id: int) -> float:
        start = time.time()
        for i in range(num_layers):
            engine.record_access("model", i)
            time.sleep(0.001)
        return (time.time() - start) * 1000
    
    # Sequential processing
    start = time.time()
    sequential_times = [process_request(i) for i in range(num_requests)]
    sequential_total = (time.time() - start) * 1000
    
    # Concurrent processing
    start = time.time()
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        concurrent_times = list(executor.map(process_request, range(num_requests)))
    concurrent_total = (time.time() - start) * 1000
    
    engine.stop()
    
    return {
        'sequential_total_ms': sequential_total,
        'concurrent_total_ms': concurrent_total,
        'sequential_avg_ms': statistics.mean(sequential_times),
        'concurrent_avg_ms': statistics.mean(concurrent_times),
        'concurrent_speedup': sequential_total / concurrent_total,
        'throughput_improvement': num_requests / (concurrent_total / 1000),
    }


def benchmark_memory_footprint(
    num_layers: int = 24,
    layer_size: int = 256
) -> Dict[str, float]:
    """
    Benchmark memory footprint of optimizations.
    
    Returns:
        Dictionary with memory statistics
    """
    import psutil
    import os
    
    process = psutil.Process(os.getpid())
    
    # Baseline
    baseline_mem = process.memory_info().rss
    
    # With cache
    cache_config = ActivationCacheConfig(
        max_memory_size_gb=1.0,
        enable_persistence=False
    )
    cache = ActivationCache(config=cache_config)
    
    # Store some activations
    for i in range(10):
        tensor = torch.randn(layer_size, layer_size)
        cache.store(f"layer_{i}", tensor)
    
    cache_mem = process.memory_info().rss
    
    # With prefetch engine
    engine = create_prefetch_engine(
        layer_loader=lambda m, i: nn.Linear(layer_size, layer_size),
        thread_pool_size=4
    )
    engine.start()
    
    engine_mem = process.memory_info().rss
    
    # Cleanup
    engine.stop()
    cache.shutdown()
    
    return {
        'baseline_mb': baseline_mem / (1024 * 1024),
        'cache_overhead_mb': (cache_mem - baseline_mem) / (1024 * 1024),
        'engine_overhead_mb': (engine_mem - cache_mem) / (1024 * 1024),
        'total_overhead_mb': (engine_mem - baseline_mem) / (1024 * 1024),
    }


def benchmark_latency_distribution(
    num_requests: int = 100,
    num_layers: int = 12
) -> Dict[str, List[float]]:
    """
    Benchmark latency distribution (p50, p90, p99).
    
    Returns:
        Dictionary with latency percentiles
    """
    def layer_loader(model_id: str, layer_idx: int) -> nn.Module:
        # Add some variability
        import random
        time.sleep(0.002 + random.random() * 0.002)
        return nn.Linear(256, 256)
    
    engine = create_prefetch_engine(layer_loader=layer_loader)
    engine.start()
    engine.set_model_info("model", num_layers)
    
    latencies = []
    for _ in range(num_requests):
        start = time.time()
        for i in range(num_layers):
            engine.record_access("model", i)
        latencies.append((time.time() - start) * 1000)
    
    engine.stop()
    
    latencies.sort()
    
    return {
        'latencies_ms': latencies,
        'p50_ms': latencies[int(num_requests * 0.5)],
        'p90_ms': latencies[int(num_requests * 0.9)],
        'p95_ms': latencies[int(num_requests * 0.95)],
        'p99_ms': latencies[int(num_requests * 0.99)],
        'mean_ms': statistics.mean(latencies),
        'std_ms': statistics.stdev(latencies) if len(latencies) > 1 else 0,
    }


def benchmark_component_interaction(
    num_iterations: int = 10
) -> Dict[str, float]:
    """
    Benchmark interaction between prefetch and cache.
    
    Returns:
        Dictionary with interaction timing
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        # Initialize both components
        cache_config = ActivationCacheConfig(
            max_memory_size_gb=0.5,
            enable_persistence=True,
            persistence_dir=temp_dir
        )
        cache = ActivationCache(config=cache_config)
        
        def layer_loader(model_id: str, layer_idx: int) -> nn.Module:
            # Check cache first
            cached = cache.retrieve(f"layer_{layer_idx}", context="model")
            if cached is not None:
                return nn.Linear(256, 256)  # Use cached info
            
            time.sleep(0.005)
            layer = nn.Linear(256, 256)
            
            # Store activation pattern
            cache.store(f"layer_{layer_idx}", torch.randn(1), context="model")
            
            return layer
        
        engine = create_prefetch_engine(
            layer_loader=layer_loader,
            lookahead=2
        )
        engine.start()
        engine.set_model_info("model", 16)
        
        times = []
        for _ in range(num_iterations):
            start = time.time()
            for i in range(16):
                engine.record_access("model", i)
            times.append((time.time() - start) * 1000)
        
        engine.stop()
        cache.shutdown()
        
        return {
            'avg_time_ms': statistics.mean(times),
            'min_time_ms': min(times),
            'max_time_ms': max(times),
        }


# Pytest benchmark tests
@pytest.mark.benchmark
class TestEndToEndBenchmarks:
    """End-to-end inference benchmark tests."""
    
    def test_full_pipeline_performance(self):
        """Benchmark full pipeline with all optimizations."""
        results = benchmark_full_pipeline(
            num_layers=12,
            batch_size=1,
            seq_length=256,
            num_iterations=3
        )
        
        print(f"\nFull Pipeline Performance:")
        print(f"  Baseline: {results['baseline_ms']:.2f} ms")
        print(f"  With optimizations: {results['with_all_ms']:.2f} ms")
        print(f"  Speedup: {results['prefetch_speedup']:.2f}x")
        print(f"  Time saved: {results['time_saved_ms']:.2f} ms")
    
    def test_concurrent_request_handling(self):
        """Benchmark concurrent request handling."""
        results = benchmark_concurrent_inference(
            num_requests=8,
            concurrency=4,
            num_layers=8
        )
        
        print(f"\nConcurrent Request Handling:")
        print(f"  Sequential total: {results['sequential_total_ms']:.2f} ms")
        print(f"  Concurrent total: {results['concurrent_total_ms']:.2f} ms")
        print(f"  Speedup: {results['concurrent_speedup']:.2f}x")
        print(f"  Throughput: {results['throughput_improvement']:.1f} req/s")
        
        assert results['concurrent_speedup'] > 1.0, "Concurrent should be faster"
    
    def test_memory_footprint(self):
        """Benchmark memory footprint."""
        results = benchmark_memory_footprint(
            num_layers=16,
            layer_size=256
        )
        
        print(f"\nMemory Footprint:")
        print(f"  Baseline: {results['baseline_mb']:.1f} MB")
        print(f"  Cache overhead: {results['cache_overhead_mb']:.1f} MB")
        print(f"  Engine overhead: {results['engine_overhead_mb']:.1f} MB")
        print(f"  Total overhead: {results['total_overhead_mb']:.1f} MB")
        
        assert results['total_overhead_mb'] < 500, "Overhead should be reasonable"
    
    def test_latency_distribution(self):
        """Benchmark latency distribution."""
        results = benchmark_latency_distribution(
            num_requests=50,
            num_layers=8
        )
        
        print(f"\nLatency Distribution:")
        print(f"  Mean: {results['mean_ms']:.2f} ms")
        print(f"  P50: {results['p50_ms']:.2f} ms")
        print(f"  P90: {results['p90_ms']:.2f} ms")
        print(f"  P95: {results['p95_ms']:.2f} ms")
        print(f"  P99: {results['p99_ms']:.2f} ms")
        print(f"  Std: {results['std_ms']:.2f} ms")
        
        assert results['p99_ms'] < results['p50_ms'] * 3, "P99 should not be too high"
    
    def test_component_interaction(self):
        """Benchmark component interaction."""
        results = benchmark_component_interaction(
            num_iterations=5
        )
        
        print(f"\nComponent Interaction:")
        print(f"  Avg time: {results['avg_time_ms']:.2f} ms")
        print(f"  Min time: {results['min_time_ms']:.2f} ms")
        print(f"  Max time: {results['max_time_ms']:.2f} ms")


if __name__ == "__main__":
    print("=" * 60)
    print("End-to-End Inference Pipeline Benchmarks")
    print("=" * 60)
    
    # Full pipeline
    results = benchmark_full_pipeline(num_iterations=3)
    print(f"\nPipeline Speedup: {results['prefetch_speedup']:.2f}x")
    
    # Concurrent
    results = benchmark_concurrent_inference(num_requests=8)
    print(f"Concurrent Speedup: {results['concurrent_speedup']:.2f}x")
    
    print("\n" + "=" * 60)
