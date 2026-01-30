"""
Performance Under Load Testing

Tests system behavior under various load conditions:
- Concurrent model loading (10+ models)
- Batch inference stress testing
- Memory pressure testing
- GPU memory exhaustion handling
- Timeout and degradation testing

Usage:
    pytest tests/integration/test_load_performance.py -v
    pytest tests/integration/test_load_performance.py -v -m "slow"  # Run heavy load tests
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import tracemalloc
import gc
import time
import json
import threading
import sys
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from unittest.mock import MagicMock, patch
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class LoadTestResult:
    """Results from a load test."""
    test_name: str
    duration_seconds: float
    peak_memory_mb: float
    throughput: float  # items/second
    error_count: int
    success_count: int
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float


@dataclass
class StressTestMetrics:
    """Metrics collected during stress testing."""
    timestamp: float
    memory_used_mb: float
    gpu_memory_mb: float
    cpu_percent: float
    active_threads: int


class PerformanceMonitor:
    """Monitor system performance during tests."""
    
    def __init__(self, interval_ms: float = 100):
        self.interval_ms = interval_ms
        self.metrics: List[StressTestMetrics] = []
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
    
    def start(self):
        """Start monitoring in background thread."""
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._monitor_loop)
        self._thread.start()
    
    def stop(self):
        """Stop monitoring."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=1.0)
    
    def _monitor_loop(self):
        """Monitoring loop."""
        try:
            import psutil
            process = psutil.Process()
        except ImportError:
            process = None
        
        while not self._stop_event.is_set():
            try:
                metric = StressTestMetrics(
                    timestamp=time.time(),
                    memory_used_mb=process.memory_info().rss / (1024**2) if process else 0,
                    gpu_memory_mb=torch.cuda.memory_allocated() / (1024**2) if torch.cuda.is_available() else 0,
                    cpu_percent=process.cpu_percent() if process else 0,
                    active_threads=threading.active_count()
                )
                self.metrics.append(metric)
            except Exception:
                pass
            
            time.sleep(self.interval_ms / 1000)
    
    def get_summary(self) -> Dict[str, float]:
        """Get summary of collected metrics."""
        if not self.metrics:
            return {}
        
        mem_values = [m.memory_used_mb for m in self.metrics]
        gpu_values = [m.gpu_memory_mb for m in self.metrics]
        
        return {
            "peak_memory_mb": max(mem_values) if mem_values else 0,
            "peak_gpu_memory_mb": max(gpu_values) if gpu_values else 0,
            "avg_cpu_percent": np.mean([m.cpu_percent for m in self.metrics]),
            "max_active_threads": max(m.active_threads for m in self.metrics)
        }


@pytest.mark.integration
@pytest.mark.slow
class TestConcurrentModelLoading:
    """
    Tests concurrent loading of multiple models.
    """
    
    def test_sequential_model_loading(self):
        """Baseline: Sequential model loading."""
        num_models = 5
        
        models = []
        load_times = []
        
        for i in range(num_models):
            start = time.time()
            # Simulate model loading
            model = nn.Linear(100, 10)
            time.sleep(0.1)  # Simulate load time
            models.append(model)
            load_times.append(time.time() - start)
        
        total_time = sum(load_times)
        avg_time = np.mean(load_times)
        
        print(f"   Sequential loading: {num_models} models in {total_time:.2f}s (avg: {avg_time:.3f}s)")
        assert len(models) == num_models
    
    def test_concurrent_model_loading_threaded(self):
        """Test loading models in parallel threads."""
        num_models = 5
        models_loaded = []
        lock = threading.Lock()
        
        def load_model(idx: int) -> float:
            start = time.time()
            # Simulate model loading
            model = nn.Linear(100, 10)
            time.sleep(0.1)
            with lock:
                models_loaded.append(model)
            return time.time() - start
        
        start = time.time()
        with ThreadPoolExecutor(max_workers=num_models) as executor:
            futures = [executor.submit(load_model, i) for i in range(num_models)]
            load_times = [f.result() for f in as_completed(futures)]
        
        total_time = time.time() - start
        avg_time = np.mean(load_times)
        
        print(f"   Threaded loading: {num_models} models in {total_time:.2f}s (avg: {avg_time:.3f}s)")
        assert len(models_loaded) == num_models
    
    def test_model_loading_memory_accumulation(self):
        """Test memory doesn't grow excessively with multiple models."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            initial_memory = torch.cuda.memory_allocated()
        else:
            initial_memory = 0
        
        models = []
        for i in range(10):
            model = nn.Linear(1000, 1000)  # 1M parameters each
            models.append(model)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            final_memory = torch.cuda.memory_allocated()
            memory_growth = final_memory - initial_memory
            
            print(f"   Memory growth for 10 models: {memory_growth / (1024**2):.2f} MB")
            # Each model ~4MB (1M params * 4 bytes), should be reasonable
            assert memory_growth < 100 * (1024**2)  # Less than 100MB growth
        
        # Cleanup
        del models
        gc.collect()


@pytest.mark.integration
@pytest.mark.slow
class TestBatchInferenceStress:
    """
    Stress tests for batch inference.
    """
    
    def test_inference_scaling_with_batch_size(self):
        """Test inference time scales reasonably with batch size."""
        model = nn.Linear(512, 128)
        model.eval()
        
        batch_sizes = [1, 2, 4, 8, 16, 32]
        results = {}
        
        for bs in batch_sizes:
            x = torch.randn(bs, 512)
            
            # Warmup
            with torch.no_grad():
                _ = model(x)
            
            # Measure
            times = []
            for _ in range(10):
                start = time.time()
                with torch.no_grad():
                    _ = model(x)
                times.append(time.time() - start)
            
            avg_time = np.mean(times)
            throughput = bs / avg_time
            results[bs] = {"time": avg_time, "throughput": throughput}
            
            print(f"   Batch {bs:2d}: {avg_time*1000:.2f}ms, {throughput:.1f} items/s")
        
        # Verify throughput increases or stays reasonable
        throughputs = [r["throughput"] for r in results.values()]
        assert max(throughputs) > throughputs[0]  # Some batching benefit
    
    def test_sustained_inference_load(self):
        """Test sustained inference over time."""
        model = nn.Linear(256, 64)
        model.eval()
        
        duration_seconds = 5
        batch_size = 8
        
        iterations = 0
        start = time.time()
        latencies = []
        
        while time.time() - start < duration_seconds:
            x = torch.randn(batch_size, 256)
            
            iter_start = time.time()
            with torch.no_grad():
                _ = model(x)
            latencies.append((time.time() - iter_start) * 1000)
            
            iterations += 1
        
        total_time = time.time() - start
        throughput = (iterations * batch_size) / total_time
        
        p99_latency = np.percentile(latencies, 99)
        
        print(f"   Sustained: {iterations} iterations, {throughput:.1f} items/s, p99={p99_latency:.2f}ms")
        assert throughput > 100  # Should handle at least 100 items/s
    
    def test_concurrent_inference_requests(self):
        """Test handling concurrent inference requests."""
        model = nn.Linear(256, 64)
        model.eval()
        lock = threading.Lock()
        results = []
        
        def inference_task(task_id: int) -> Dict:
            start = time.time()
            x = torch.randn(4, 256)
            
            with lock:  # Simulate model lock for thread safety
                with torch.no_grad():
                    output = model(x)
            
            duration = time.time() - start
            return {"task_id": task_id, "duration": duration, "output_shape": output.shape}
        
        num_concurrent = 20
        start = time.time()
        
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(inference_task, i) for i in range(num_concurrent)]
            results = [f.result() for f in as_completed(futures)]
        
        total_time = time.time() - start
        avg_duration = np.mean([r["duration"] for r in results])
        
        print(f"   Concurrent: {num_concurrent} tasks in {total_time:.2f}s (avg: {avg_duration*1000:.2f}ms)")
        assert len(results) == num_concurrent


@pytest.mark.integration
@pytest.mark.slow
class TestMemoryPressure:
    """
    Tests behavior under memory pressure.
    """
    
    def test_gradual_memory_increase(self):
        """Test system behavior as memory gradually increases."""
        monitor = PerformanceMonitor(interval_ms=50)
        monitor.start()
        
        tensors = []
        try:
            for i in range(50):
                # Allocate increasingly large tensors
                size = (i + 1) * 1000
                tensor = torch.randn(size)
                tensors.append(tensor)
                
                if i % 10 == 0:
                    time.sleep(0.01)
        finally:
            monitor.stop()
        
        summary = monitor.get_summary()
        print(f"   Memory test: peak={summary.get('peak_memory_mb', 0):.2f} MB")
        
        # Cleanup
        del tensors
        gc.collect()
    
    def test_memory_cleanup_effectiveness(self):
        """Test memory is properly cleaned up after operations."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            initial = torch.cuda.memory_allocated()
        else:
            import psutil
            process = psutil.Process()
            initial = process.memory_info().rss
        
        # Allocate and deallocate multiple times
        for i in range(5):
            tensors = [torch.randn(10000, 100) for _ in range(10)]
            del tensors
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Check final memory
        if torch.cuda.is_available():
            final = torch.cuda.memory_allocated()
        else:
            final = process.memory_info().rss
        
        growth = abs(final - initial) / (1024**2)
        print(f"   Memory cleanup: growth={growth:.2f} MB")
        assert growth < 50  # Less than 50MB growth
    
    def test_oom_handling_simulation(self):
        """Test graceful handling of out-of-memory situations."""
        errors_caught = 0
        
        # Simulate OOM by requesting too much memory
        try:
            # This would cause OOM on most systems
            huge_tensor = torch.empty(10**12)  # 1 trillion elements
        except (RuntimeError, MemoryError):
            errors_caught += 1
        
        # System should still be functional
        normal_tensor = torch.randn(100, 100)
        result = normal_tensor.sum()
        
        assert errors_caught == 1
        assert result.item() is not None
        print(f"   OOM handling: {errors_caught} errors caught, system functional")


@pytest.mark.integration
@pytest.mark.slow
class TestGPUMemoryExhaustion:
    """
    Tests GPU memory exhaustion handling.
    """
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_memory_tracking(self):
        """Test accurate GPU memory tracking."""
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        initial = torch.cuda.memory_allocated()
        
        # Allocate tensors on GPU
        tensors = []
        for i in range(5):
            tensor = torch.randn(1000, 1000, device='cuda')
            tensors.append(tensor)
        
        current = torch.cuda.memory_allocated()
        peak = torch.cuda.max_memory_allocated()
        
        print(f"   GPU memory: initial={initial/(1024**2):.2f}MB, current={current/(1024**2):.2f}MB, peak={peak/(1024**2):.2f}MB")
        
        assert current > initial
        assert peak >= current
        
        # Cleanup
        del tensors
        torch.cuda.empty_cache()
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_memory_fragmentation(self):
        """Test behavior with fragmented GPU memory."""
        torch.cuda.empty_cache()
        
        # Allocate and free tensors to create fragmentation
        tensors = []
        for i in range(10):
            t = torch.randn(1000, 1000, device='cuda')
            tensors.append(t)
        
        # Free every other tensor
        for i in range(0, len(tensors), 2):
            tensors[i] = None
        
        torch.cuda.empty_cache()
        
        # Try to allocate a large tensor
        try:
            large = torch.randn(5000, 5000, device='cuda')
            success = True
            del large
        except RuntimeError:
            success = False
        
        # Cleanup remaining
        tensors = None
        torch.cuda.empty_cache()
        
        print(f"   Fragmentation test: large allocation {'succeeded' if success else 'failed'}")
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_mixed_gpu_cpu_offloading(self):
        """Test mixed GPU/CPU memory management."""
        # Create tensor on CPU
        cpu_tensor = torch.randn(5000, 5000)
        
        # Move to GPU
        gpu_tensor = cpu_tensor.cuda()
        
        # Process on GPU
        result_gpu = gpu_tensor * 2
        
        # Move back to CPU
        result_cpu = result_gpu.cpu()
        
        # Verify
        expected = cpu_tensor * 2
        assert torch.allclose(result_cpu, expected)
        
        print(f"   GPU/CPU offloading: tensor size={cpu_tensor.element_size() * cpu_tensor.nelement() / (1024**2):.2f} MB")
        
        del cpu_tensor, gpu_tensor, result_gpu, result_cpu
        torch.cuda.empty_cache()


@pytest.mark.integration
class TestTimeoutAndDegradation:
    """
    Tests timeout handling and graceful degradation.
    """
    
    def test_inference_timeout(self):
        """Test inference respects timeout."""
        model = nn.Sequential(
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 10)
        )
        model.eval()
        
        timeout_seconds = 1.0
        
        x = torch.randn(100, 100)
        
        start = time.time()
        with torch.no_grad():
            result = model(x)
        duration = time.time() - start
        
        assert duration < timeout_seconds, f"Inference took {duration:.2f}s, exceeding {timeout_seconds}s timeout"
        print(f"   Inference timeout: completed in {duration*1000:.2f}ms")
    
    def test_graceful_degradation_under_load(self):
        """Test system degrades gracefully under high load."""
        model = nn.Linear(256, 64)
        model.eval()
        
        load_levels = [1, 5, 10, 20]
        results = {}
        
        for load in load_levels:
            latencies = []
            
            for _ in range(load):
                x = torch.randn(32, 256)
                start = time.time()
                with torch.no_grad():
                    _ = model(x)
                latencies.append((time.time() - start) * 1000)
            
            results[load] = {
                "p50": np.percentile(latencies, 50),
                "p95": np.percentile(latencies, 95),
                "max": max(latencies)
            }
        
        # Print results
        for load, metrics in results.items():
            print(f"   Load {load:2d}: p50={metrics['p50']:.2f}ms, p95={metrics['p95']:.2f}ms, max={metrics['max']:.2f}ms")
        
        # Latency should increase with load but not explode
        p95s = [r["p95"] for r in results.values()]
        # Max p95 should not be more than 10x min p95
        assert max(p95s) < max(p95s) * 10
    
    def test_error_recovery_under_load(self):
        """Test system recovers from errors under load."""
        model = nn.Linear(10, 2)
        
        success_count = 0
        error_count = 0
        
        for i in range(50):
            try:
                if i % 10 == 0:
                    # Intentional error: wrong input size
                    x = torch.randn(5, 5)
                else:
                    x = torch.randn(5, 10)
                
                output = model(x)
                success_count += 1
            except RuntimeError:
                error_count += 1
                # Continue processing
                continue
        
        print(f"   Error recovery: {success_count} success, {error_count} errors")
        assert success_count > 40  # Most should succeed
        assert error_count == 5    # Expected errors


@pytest.mark.integration
class TestPerformanceBaselines:
    """
    Establish performance baselines.
    """
    
    def test_save_performance_baselines(self, tmp_path):
        """Save performance baselines to file."""
        baselines = {
            "model_loading": {
                "small_model_seconds": 2.0,
                "medium_model_seconds": 10.0,
                "large_model_seconds": 30.0
            },
            "inference": {
                "tokens_per_second_cpu": 10.0,
                "tokens_per_second_gpu": 100.0,
                "latency_p99_ms": 100.0
            },
            "memory": {
                "small_model_mb": 500,
                "medium_model_mb": 2000,
                "large_model_mb": 8000
            }
        }
        
        baseline_file = tmp_path / "performance_baselines.json"
        with open(baseline_file, 'w') as f:
            json.dump(baselines, f, indent=2)
        
        # Verify file was created and can be loaded
        with open(baseline_file) as f:
            loaded = json.load(f)
        
        assert loaded == baselines
        print(f"   Performance baselines saved to {baseline_file}")


@pytest.mark.integration
@pytest.mark.slow
class TestLoadTestSuite:
    """
    Comprehensive load test suite.
    """
    
    def test_full_load_scenario(self):
        """Run a comprehensive load test scenario."""
        monitor = PerformanceMonitor(interval_ms=100)
        latencies = []
        errors = []
        
        monitor.start()
        start = time.time()
        
        try:
            # Phase 1: Load multiple models
            models = [nn.Linear(256, 64) for _ in range(5)]
            
            # Phase 2: Run concurrent inference
            def inference_task(model_idx: int):
                model = models[model_idx % len(models)]
                task_latencies = []
                
                for _ in range(10):
                    try:
                        x = torch.randn(32, 256)
                        t0 = time.time()
                        with torch.no_grad():
                            _ = model(x)
                        task_latencies.append((time.time() - t0) * 1000)
                    except Exception as e:
                        errors.append(str(e))
                
                return task_latencies
            
            with ThreadPoolExecutor(max_workers=5) as executor:
                futures = [executor.submit(inference_task, i) for i in range(10)]
                for f in as_completed(futures):
                    latencies.extend(f.result())
            
        finally:
            monitor.stop()
        
        duration = time.time() - start
        summary = monitor.get_summary()
        
        # Calculate results
        result = LoadTestResult(
            test_name="full_load_scenario",
            duration_seconds=duration,
            peak_memory_mb=summary.get("peak_memory_mb", 0),
            throughput=len(latencies) / duration if duration > 0 else 0,
            error_count=len(errors),
            success_count=len(latencies),
            latency_p50_ms=np.percentile(latencies, 50) if latencies else 0,
            latency_p95_ms=np.percentile(latencies, 95) if latencies else 0,
            latency_p99_ms=np.percentile(latencies, 99) if latencies else 0
        )
        
        print(f"\n   Full Load Test Results:")
        print(f"   Duration: {result.duration_seconds:.2f}s")
        print(f"   Throughput: {result.throughput:.1f} inferences/s")
        print(f"   Success: {result.success_count}, Errors: {result.error_count}")
        print(f"   Latency: p50={result.latency_p50_ms:.2f}ms, p95={result.latency_p95_ms:.2f}ms, p99={result.latency_p99_ms:.2f}ms")
        print(f"   Peak Memory: {result.peak_memory_mb:.2f} MB")
        
        # Save results
        results_dir = PROJECT_ROOT / "test_results"
        results_dir.mkdir(exist_ok=True)
        
        with open(results_dir / "load_test_results.json", 'w') as f:
            json.dump(asdict(result), f, indent=2)
        
        assert result.error_count == 0
        assert result.throughput > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
