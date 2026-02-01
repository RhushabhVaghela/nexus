"""
Performance Regression Tests

Tests to detect performance regressions across releases.
"""

import pytest
import time
import torch
import torch.nn as nn
import numpy as np
from typing import Callable, List, Dict, Any
from dataclasses import dataclass
import statistics
import json
import os


@dataclass
class BenchmarkResult:
    """Performance benchmark result."""
    name: str
    avg_time: float
    min_time: float
    max_time: float
    std_dev: float
    iterations: int
    metadata: Dict[str, Any]


class PerformanceBenchmark:
    """Performance benchmarking utility."""
    
    def __init__(self, baseline_file: str = "performance_baselines.json"):
        self.baseline_file = baseline_file
        self.baselines = self._load_baselines()
    
    def _load_baselines(self) -> Dict[str, float]:
        """Load performance baselines from file."""
        if os.path.exists(self.baseline_file):
            with open(self.baseline_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_baselines(self):
        """Save performance baselines to file."""
        with open(self.baseline_file, 'w') as f:
            json.dump(self.baselines, f, indent=2)
    
    def benchmark(
        self,
        name: str,
        func: Callable,
        iterations: int = 10,
        warmup: int = 3,
        tolerance: float = 0.2  # 20% tolerance
    ) -> BenchmarkResult:
        """
        Benchmark a function.
        
        Args:
            name: Benchmark name
            func: Function to benchmark
            iterations: Number of iterations
            warmup: Warmup iterations
            tolerance: Allowed regression tolerance (0.2 = 20%)
            
        Returns:
            BenchmarkResult with statistics
        """
        # Warmup
        for _ in range(warmup):
            func()
        
        # Benchmark
        times = []
        for _ in range(iterations):
            start = time.perf_counter()
            func()
            end = time.perf_counter()
            times.append(end - start)
        
        result = BenchmarkResult(
            name=name,
            avg_time=statistics.mean(times),
            min_time=min(times),
            max_time=max(times),
            std_dev=statistics.stdev(times) if len(times) > 1 else 0,
            iterations=iterations,
            metadata={}
        )
        
        # Check for regression
        if name in self.baselines:
            baseline = self.baselines[name]
            regression = (result.avg_time - baseline) / baseline
            
            if regression > tolerance:
                raise AssertionError(
                    f"Performance regression detected in {name}: "
                    f"{regression:.1%} slower than baseline"
                )
        else:
            # Set baseline
            self.baselines[name] = result.avg_time
            self._save_baselines()
        
        return result


class TestInferencePerformance:
    """Test inference performance."""
    
    @pytest.fixture
    def sample_model(self):
        """Create a sample model for testing."""
        return nn.Sequential(
            nn.Linear(1000, 500),
            nn.ReLU(),
            nn.Linear(500, 250),
            nn.ReLU(),
            nn.Linear(250, 100)
        )
    
    def test_inference_latency(self, sample_model):
        """Test inference latency hasn't regressed."""
        model = sample_model
        model.eval()
        
        input_tensor = torch.randn(32, 1000)
        
        def inference():
            with torch.no_grad():
                _ = model(input_tensor)
        
        benchmark = PerformanceBenchmark()
        result = benchmark.benchmark(
            name="inference_latency",
            func=inference,
            iterations=20,
            tolerance=0.3
        )
        
        # Should complete in reasonable time
        assert result.avg_time < 1.0  # Less than 1 second
    
    def test_batch_inference_scaling(self, sample_model):
        """Test inference scales properly with batch size."""
        model = sample_model
        model.eval()
        
        batch_sizes = [1, 8, 16, 32]
        times = []
        
        for batch_size in batch_sizes:
            input_tensor = torch.randn(batch_size, 1000)
            
            def inference():
                with torch.no_grad():
                    _ = model(input_tensor)
            
            # Time the inference
            start = time.perf_counter()
            for _ in range(10):
                inference()
            elapsed = time.perf_counter() - start
            
            times.append((batch_size, elapsed))
        
        # Time should scale roughly linearly with batch size
        for i in range(1, len(times)):
            prev_size, prev_time = times[i-1]
            curr_size, curr_time = times[i]
            
            ratio = (curr_time / prev_time) / (curr_size / prev_size)
            # Allow 2x variance in scaling
            assert 0.5 < ratio < 2.0


class TestTrainingPerformance:
    """Test training performance."""
    
    def test_training_step_latency(self):
        """Test training step latency."""
        model = nn.Linear(1000, 100)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        
        def training_step():
            optimizer.zero_grad()
            input_tensor = torch.randn(32, 1000)
            output = model(input_tensor)
            loss = output.sum()
            loss.backward()
            optimizer.step()
        
        benchmark = PerformanceBenchmark()
        result = benchmark.benchmark(
            name="training_step_latency",
            func=training_step,
            iterations=10,
            tolerance=0.3
        )
        
        assert result.avg_time < 2.0
    
    def test_backward_pass_performance(self):
        """Test backward pass performance."""
        model = nn.Sequential(
            nn.Linear(1000, 500),
            nn.ReLU(),
            nn.Linear(500, 100)
        )
        
        input_tensor = torch.randn(64, 1000, requires_grad=True)
        
        def backward_pass():
            output = model(input_tensor)
            loss = output.sum()
            loss.backward()
        
        benchmark = PerformanceBenchmark()
        result = benchmark.benchmark(
            name="backward_pass_latency",
            func=backward_pass,
            iterations=10,
            tolerance=0.3
        )
        
        assert result.avg_time < 1.0


class TestDataLoadingPerformance:
    """Test data loading performance."""
    
    def test_tensor_creation_speed(self):
        """Test tensor creation hasn't slowed down."""
        def create_tensors():
            for _ in range(100):
                _ = torch.randn(100, 100)
        
        benchmark = PerformanceBenchmark()
        result = benchmark.benchmark(
            name="tensor_creation",
            func=create_tensors,
            iterations=5,
            tolerance=0.3
        )
        
        assert result.avg_time < 1.0
    
    def test_data_transfer_speed(self):
        """Test CPU to GPU transfer speed (if CUDA available)."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        data = torch.randn(1000, 1000)
        
        def transfer():
            _ = data.cuda()
            torch.cuda.synchronize()
        
        benchmark = PerformanceBenchmark()
        result = benchmark.benchmark(
            name="cpu_to_gpu_transfer",
            func=transfer,
            iterations=10,
            tolerance=0.3
        )
        
        # Transfer should be reasonably fast
        assert result.avg_time < 0.1


class TestMemoryPerformance:
    """Test memory performance."""
    
    def test_memory_allocation_speed(self):
        """Test memory allocation performance."""
        def allocate_memory():
            tensors = []
            for _ in range(10):
                tensor = torch.randn(1000, 1000)
                tensors.append(tensor)
            return tensors
        
        benchmark = PerformanceBenchmark()
        result = benchmark.benchmark(
            name="memory_allocation",
            func=allocate_memory,
            iterations=10,
            tolerance=0.3
        )
        
        assert result.avg_time < 1.0
    
    def test_memory_deallocation_speed(self):
        """Test memory deallocation performance."""
        tensors = [torch.randn(1000, 1000) for _ in range(100)]
        
        def deallocate():
            nonlocal tensors
            tensors = []
            import gc
            gc.collect()
        
        benchmark = PerformanceBenchmark()
        result = benchmark.benchmark(
            name="memory_deallocation",
            func=deallocate,
            iterations=5,
            tolerance=0.3
        )
        
        assert result.avg_time < 0.5


class TestSerializationPerformance:
    """Test model serialization performance."""
    
    def test_model_save_speed(self):
        """Test model saving performance."""
        model = nn.Sequential(
            nn.Linear(1000, 500),
            nn.ReLU(),
            nn.Linear(500, 100)
        )
        
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False) as f:
            temp_path = f.name
        
        def save_model():
            torch.save(model.state_dict(), temp_path)
        
        try:
            benchmark = PerformanceBenchmark()
            result = benchmark.benchmark(
                name="model_save",
                func=save_model,
                iterations=5,
                tolerance=0.3
            )
            
            assert result.avg_time < 1.0
        finally:
            os.unlink(temp_path)
    
    def test_model_load_speed(self):
        """Test model loading performance."""
        model = nn.Sequential(
            nn.Linear(1000, 500),
            nn.ReLU(),
            nn.Linear(500, 100)
        )
        
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False) as f:
            temp_path = f.name
        
        torch.save(model.state_dict(), temp_path)
        
        def load_model():
            _ = torch.load(temp_path)
        
        try:
            benchmark = PerformanceBenchmark()
            result = benchmark.benchmark(
                name="model_load",
                func=load_model,
                iterations=5,
                tolerance=0.3
            )
            
            assert result.avg_time < 1.0
        finally:
            os.unlink(temp_path)


class TestThroughput:
    """Test system throughput."""
    
    def test_inference_throughput(self):
        """Test inference throughput."""
        model = nn.Linear(1000, 100)
        model.eval()
        
        batch_size = 64
        num_batches = 100
        
        input_tensor = torch.randn(batch_size, 1000)
        
        start = time.perf_counter()
        with torch.no_grad():
            for _ in range(num_batches):
                _ = model(input_tensor)
        elapsed = time.perf_counter() - start
        
        throughput = (batch_size * num_batches) / elapsed
        
        # Should process at least 1000 items per second
        assert throughput > 1000
    
    def test_concurrent_throughput(self):
        """Test throughput with concurrent operations."""
        import threading
        
        model = nn.Linear(1000, 100)
        model.eval()
        
        results = []
        lock = threading.Lock()
        
        def worker():
            input_tensor = torch.randn(32, 1000)
            with torch.no_grad():
                _ = model(input_tensor)
            with lock:
                results.append(1)
        
        num_threads = 10
        threads = [threading.Thread(target=worker) for _ in range(num_threads)]
        
        start = time.perf_counter()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        elapsed = time.perf_counter() - start
        
        assert len(results) == num_threads
        # Should complete in reasonable time
        assert elapsed < 5.0


class TestLatencyConsistency:
    """Test latency consistency."""
    
    def test_latency_consistency(self):
        """Test that latency is consistent (low variance)."""
        model = nn.Linear(1000, 100)
        model.eval()
        
        input_tensor = torch.randn(32, 1000)
        
        times = []
        for _ in range(50):
            start = time.perf_counter()
            with torch.no_grad():
                _ = model(input_tensor)
            times.append(time.perf_counter() - start)
        
        # Calculate coefficient of variation (CV)
        mean = statistics.mean(times)
        std_dev = statistics.stdev(times)
        cv = std_dev / mean
        
        # CV should be low (consistent latency)
        assert cv < 0.5  # Less than 50% variation


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
