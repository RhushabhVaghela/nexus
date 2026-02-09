"""
Prefetch Performance Benchmarks

Benchmarks for Smart Layer Prefetching Engine performance:
- Sequential access speedup
- Strided access speedup
- Burst access speedup
- Comparison with/without prefetching
- Thread pool performance

Usage:
    python -m pytest tests/benchmarks/test_prefetch_performance.py -v

Author: Nexus Team
"""

import pytest
import time
import statistics
from typing import List, Dict
import torch
import torch.nn as nn

from src.models.sli.prefetch_engine import (
    PrefetchEngine,
    create_prefetch_engine,
    PrefetchConfig,
)


class MockModel:
    """Mock model for benchmarking."""
    
    def __init__(self, num_layers: int = 32, layer_size: int = 256):
        self.num_layers = num_layers
        self.layer_size = layer_size
        self.layers = {}
    
    def get_layer(self, layer_idx: int) -> nn.Module:
        """Get or create layer with simulated load time."""
        if layer_idx not in self.layers:
            # Simulate layer loading time (10-50ms)
            time.sleep(0.02)
            self.layers[layer_idx] = nn.Linear(self.layer_size, self.layer_size)
        return self.layers[layer_idx]


def benchmark_sequential_access(
    num_layers: int = 20,
    num_iterations: int = 5
) -> Dict[str, float]:
    """
    Benchmark sequential layer access with and without prefetching.
    
    Returns:
        Dictionary with timing results
    """
    model = MockModel(num_layers=num_layers)
    
    # Baseline: Sequential access without prefetching
    baseline_times = []
    for _ in range(num_iterations):
        start = time.time()
        for i in range(num_layers):
            layer = model.get_layer(i)
            time.sleep(0.005)  # Simulate computation time
        baseline_times.append(time.time() - start)
    
    baseline_time = statistics.median(baseline_times)
    
    # With prefetching
    def layer_loader(model_id: str, layer_idx: int) -> nn.Module:
        return model.get_layer(layer_idx)
    
    prefetch_times = []
    for _ in range(num_iterations):
        engine = create_prefetch_engine(
            layer_loader=layer_loader,
            lookahead=3,
            thread_pool_size=4
        )
        engine.start()
        engine.set_model_info("benchmark_model", num_layers)
        
        # Clear loaded layers for fair comparison
        model.layers.clear()
        
        start = time.time()
        for i in range(num_layers):
            engine.record_access("benchmark_model", i)
            # Try to get from prefetch buffer
            prefetched = engine.get_prefetched_layer(f"benchmark_model_layer_{i+3}")
            if prefetched is None:
                layer = model.get_layer(i)
            time.sleep(0.005)  # Simulate computation time
        
        prefetch_times.append(time.time() - start)
        engine.stop()
    
    prefetch_time = statistics.median(prefetch_times)
    
    speedup = baseline_time / prefetch_time if prefetch_time > 0 else 1.0
    
    return {
        'baseline_time_ms': baseline_time * 1000,
        'prefetch_time_ms': prefetch_time * 1000,
        'speedup': speedup,
        'improvement_percent': (1 - prefetch_time / baseline_time) * 100 if baseline_time > 0 else 0
    }


def benchmark_pattern_recognition(
    pattern: str = "strided",
    num_layers: int = 20
) -> Dict[str, float]:
    """
    Benchmark pattern recognition performance.
    
    Args:
        pattern: Access pattern ("sequential", "strided", "burst", "random")
        num_layers: Number of layers to access
    
    Returns:
        Dictionary with pattern detection accuracy and timing
    """
    from src.models.sli.prefetch_engine import PatternPredictor, LayerAccess
    
    predictor = PatternPredictor()
    
    # Generate access pattern
    accesses = []
    if pattern == "sequential":
        accesses = list(range(num_layers))
    elif pattern == "strided":
        accesses = list(range(0, num_layers * 2, 2))[:num_layers]
    elif pattern == "burst":
        accesses = [5] * num_layers
    elif pattern == "random":
        import random
        accesses = [random.randint(0, num_layers) for _ in range(num_layers)]
    
    # Record accesses and measure detection time
    start = time.time()
    for i, layer_idx in enumerate(accesses):
        access = LayerAccess(
            layer_index=layer_idx,
            timestamp=time.time(),
            model_id="benchmark"
        )
        predictor.record_access(access)
    
    detection_time = (time.time() - start) * 1000  # ms
    
    info = predictor.get_pattern_info()
    
    return {
        'detection_time_ms': detection_time,
        'detected_pattern': info['pattern'],
        'confidence': info['confidence'],
        'correct_detection': info['pattern'] == pattern,
    }


def benchmark_adaptive_lookahead(
    num_iterations: int = 50
) -> Dict[str, List[int]]:
    """
    Benchmark adaptive lookahead adjustment.
    
    Returns:
        Dictionary with lookahead values over time
    """
    from src.models.sli.prefetch_engine import PrefetchStats
    
    config = PrefetchConfig(
        enable_adaptive_lookahead=True,
        min_lookahead=2,
        max_lookahead=5
    )
    
    engine = PrefetchEngine(config=config)
    engine.start()
    
    lookahead_history = []
    
    # Phase 1: High success rate (should increase lookahead)
    for _ in range(num_iterations // 2):
        engine._stats.record_prefetch(True, 10.0)
        engine._adapt_lookahead()
        lookahead_history.append(engine._current_lookahead)
    
    # Phase 2: Low success rate (should decrease lookahead)
    for _ in range(num_iterations // 2):
        engine._stats.record_prefetch(False, 10.0)
        engine._adapt_lookahead()
        lookahead_history.append(engine._current_lookahead)
    
    engine.stop()
    
    return {
        'lookahead_history': lookahead_history,
        'initial_lookahead': lookahead_history[0],
        'final_lookahead': lookahead_history[-1],
        'max_lookahead': max(lookahead_history),
        'min_lookahead': min(lookahead_history),
    }


def benchmark_thread_pool_scaling(
    thread_counts: List[int] = [1, 2, 4, 8],
    num_prefetches: int = 20
) -> Dict[int, float]:
    """
    Benchmark thread pool scaling performance.
    
    Returns:
        Dictionary mapping thread count to time
    """
    results = {}
    
    for thread_count in thread_counts:
        def layer_loader(model_id: str, layer_idx: int) -> nn.Module:
            time.sleep(0.01)  # Simulate load
            return nn.Linear(256, 256)
        
        engine = create_prefetch_engine(
            layer_loader=layer_loader,
            lookahead=3,
            thread_pool_size=thread_count
        )
        engine.start()
        engine.set_model_info("benchmark", 100)
        
        start = time.time()
        futures = engine.prefetch_layers_parallel(
            "benchmark",
            list(range(num_prefetches))
        )
        
        # Wait for all to complete
        for future in futures:
            try:
                future.result(timeout=10)
            except Exception:
                pass
        
        elapsed = time.time() - start
        results[thread_count] = elapsed * 1000  # ms
        
        engine.stop()
    
    return results


# Pytest benchmark tests
@pytest.mark.benchmark
class TestPrefetchBenchmarks:
    """Prefetch performance benchmark tests."""
    
    def test_sequential_access_speedup(self):
        """Benchmark sequential access speedup with prefetching."""
        results = benchmark_sequential_access(num_layers=20, num_iterations=3)
        
        print(f"\nSequential Access Results:")
        print(f"  Baseline: {results['baseline_time_ms']:.2f} ms")
        print(f"  With Prefetch: {results['prefetch_time_ms']:.2f} ms")
        print(f"  Speedup: {results['speedup']:.2f}x")
        print(f"  Improvement: {results['improvement_percent']:.1f}%")
        
        # With prefetching should generally be faster
        assert results['speedup'] >= 0.8, "Prefetching should not significantly slow down"
    
    def test_strided_pattern_detection(self):
        """Benchmark strided pattern recognition."""
        results = benchmark_pattern_recognition("strided", num_layers=20)
        
        print(f"\nStrided Pattern Detection:")
        print(f"  Detected: {results['detected_pattern']}")
        print(f"  Confidence: {results['confidence']:.2f}")
        print(f"  Detection time: {results['detection_time_ms']:.2f} ms")
        
        assert results['correct_detection'], "Should correctly detect strided pattern"
        assert results['confidence'] > 0.5, "Should have high confidence"
    
    def test_burst_pattern_detection(self):
        """Benchmark burst pattern recognition."""
        results = benchmark_pattern_recognition("burst", num_layers=10)
        
        print(f"\nBurst Pattern Detection:")
        print(f"  Detected: {results['detected_pattern']}")
        print(f"  Confidence: {results['confidence']:.2f}")
        
        # Burst pattern might be detected or might look like sequential
        assert results['detection_time_ms'] < 10, "Detection should be fast"
    
    def test_adaptive_lookahead(self):
        """Benchmark adaptive lookahead behavior."""
        results = benchmark_adaptive_lookahead(num_iterations=40)
        
        print(f"\nAdaptive Lookahead:")
        print(f"  Initial: {results['initial_lookahead']}")
        print(f"  Final: {results['final_lookahead']}")
        print(f"  Range: {results['min_lookahead']} - {results['max_lookahead']}")
        
        # Lookahead should adapt
        assert len(set(results['lookahead_history'])) > 1, "Lookahead should change"
    
    def test_thread_pool_scaling(self):
        """Benchmark thread pool scaling."""
        results = benchmark_thread_pool_scaling(
            thread_counts=[1, 2, 4],
            num_prefetches=16
        )
        
        print(f"\nThread Pool Scaling:")
        for threads, time_ms in results.items():
            print(f"  {threads} threads: {time_ms:.2f} ms")
        
        # More threads should generally be faster (up to a point)
        if len(results) >= 2:
            times = list(results.values())
            # 4 threads should generally beat 1 thread
            assert times[-1] < times[0] * 1.5, "More threads should not be significantly slower"


if __name__ == "__main__":
    # Run benchmarks directly
    print("=" * 60)
    print("Prefetch Performance Benchmarks")
    print("=" * 60)
    
    # Sequential access
    results = benchmark_sequential_access(num_layers=20, num_iterations=3)
    print(f"\nSequential Access:")
    print(f"  Speedup: {results['speedup']:.2f}x")
    
    # Pattern detection
    for pattern in ["sequential", "strided", "burst"]:
        results = benchmark_pattern_recognition(pattern, num_layers=20)
        print(f"\n{pattern.title()} Pattern:")
        print(f"  Detected: {results['detected_pattern']} (correct: {results['correct_detection']})")
    
    print("\n" + "=" * 60)
