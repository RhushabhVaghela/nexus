#!/usr/bin/env python3
"""
Multimodal Architecture Benchmark Suite

Comprehensive benchmarks for multimodal embedding injection performance:
- Embedding projection latency (different dimensions)
- Multimodal fusion throughput (images/sec)
- Memory usage during injection
- Attention mask computation time
- Label shifting performance
- Compare with/without optimization flags

Usage:
    python benchmarks/test_multimodal_architect_benchmark.py --all
    python benchmarks/test_multimodal_architect_benchmark.py --category projection
    python benchmarks/test_multimodal_architect_benchmark.py --dimensions 512 1024 2048
    python benchmarks/test_multimodal_architect_benchmark.py --memory-profiling
    python benchmarks/test_multimodal_architect_benchmark.py --output results/multimodal_benchmark.json

Environment Variables:
    BENCHMARK_ITERATIONS: Number of iterations (default: 100)
    BENCHMARK_WARMUP: Number of warmup iterations (default: 10)
    ENABLE_GPU_BENCHMARKS: Set to '1' to include GPU benchmarks
"""

import os
import sys
import json
import time
import argparse
import tempfile
import shutil
import statistics
import tracemalloc
import gc
import random
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any, Callable
from contextlib import contextmanager

import numpy as np

# Ensure src is in path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️  PyTorch not available. Running in mock mode.")


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""
    name: str
    category: str
    iterations: int
    total_time: float
    mean_time: float
    median_time: float
    std_dev: float
    min_time: float
    max_time: float
    p95_time: float
    p99_time: float
    memory_delta_mb: Optional[float] = None
    throughput: Optional[float] = None  # ops/sec or items/sec
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BenchmarkSuite:
    """Collection of benchmark results."""
    name: str
    timestamp: str
    python_version: str
    platform: str
    torch_version: Optional[str] = None
    results: List[BenchmarkResult] = field(default_factory=list)
    
    def add_result(self, result: BenchmarkResult):
        self.results.append(result)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "timestamp": self.timestamp,
            "python_version": self.python_version,
            "platform": self.platform,
            "torch_version": self.torch_version,
            "results": [r.to_dict() for r in self.results],
            "summary": {
                "total_benchmarks": len(self.results),
                "categories": list(set(r.category for r in self.results))
            }
        }
    
    def save(self, path: str):
        """Save benchmark results to JSON file."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"📊 Benchmark results saved to {path}")


class BenchmarkRunner:
    """Runner for executing benchmarks with timing and memory profiling."""
    
    def __init__(self, iterations: int = 100, warmup: int = 10):
        self.iterations = iterations
        self.warmup = warmup
        self.results: List[BenchmarkResult] = []
    
    def run(
        self,
        name: str,
        category: str,
        func: Callable,
        *args,
        memory_profile: bool = False,
        calculate_throughput: bool = False,
        throughput_items: int = 1,
        metadata: Optional[Dict] = None,
        **kwargs
    ) -> BenchmarkResult:
        """
        Run a benchmark function and collect timing statistics.
        
        Args:
            name: Benchmark name
            category: Benchmark category
            func: Function to benchmark
            args: Positional arguments for func
            memory_profile: Whether to profile memory usage
            calculate_throughput: Whether to calculate throughput
            throughput_items: Number of items processed per iteration
            metadata: Additional metadata to store
            kwargs: Keyword arguments for func
        """
        times = []
        memory_delta = None
        
        # Warmup iterations
        for _ in range(self.warmup):
            result = func(*args, **kwargs)
            if TORCH_AVAILABLE and torch.is_tensor(result):
                torch.cuda.synchronize() if torch.cuda.is_available() else None
        
        # Actual benchmark iterations
        if memory_profile:
            gc.collect()
            tracemalloc.start()
            start_mem = tracemalloc.get_traced_memory()[0]
        
        for _ in range(self.iterations):
            start = time.perf_counter()
            result = func(*args, **kwargs)
            if TORCH_AVAILABLE and torch.is_tensor(result):
                torch.cuda.synchronize() if torch.cuda.is_available() else None
            end = time.perf_counter()
            times.append(end - start)
        
        if memory_profile:
            end_mem = tracemalloc.get_traced_memory()[0]
            tracemalloc.stop()
            memory_delta = (end_mem - start_mem) / (1024 * 1024)  # MB
        
        # Calculate statistics
        times.sort()
        mean_time = statistics.mean(times)
        throughput = (throughput_items / mean_time) if calculate_throughput else None
        
        result = BenchmarkResult(
            name=name,
            category=category,
            iterations=self.iterations,
            total_time=sum(times),
            mean_time=mean_time,
            median_time=statistics.median(times),
            std_dev=statistics.stdev(times) if len(times) > 1 else 0.0,
            min_time=min(times),
            max_time=max(times),
            p95_time=times[int(len(times) * 0.95)],
            p99_time=times[int(len(times) * 0.99)],
            memory_delta_mb=memory_delta,
            throughput=throughput,
            metadata=metadata or {}
        )
        
        self.results.append(result)
        return result
    
    def print_result(self, result: BenchmarkResult):
        """Print a benchmark result in a formatted way."""
        print(f"\n{'='*60}")
        print(f"📊 {result.name}")
        print(f"{'='*60}")
        print(f"Category: {result.category}")
        print(f"Iterations: {result.iterations}")
        print(f"Total Time: {result.total_time:.4f}s")
        print(f"Mean: {result.mean_time*1000:.4f}ms")
        print(f"Median: {result.median_time*1000:.4f}ms")
        print(f"Std Dev: {result.std_dev*1000:.4f}ms")
        print(f"Min: {result.min_time*1000:.4f}ms")
        print(f"Max: {result.max_time*1000:.4f}ms")
        print(f"P95: {result.p95_time*1000:.4f}ms")
        print(f"P99: {result.p99_time*1000:.4f}ms")
        if result.memory_delta_mb is not None:
            print(f"Memory Delta: {result.memory_delta_mb:.4f} MB")
        if result.throughput is not None:
            print(f"Throughput: {result.throughput:.2f} ops/sec")


class MockMultimodalComponents:
    """Mock implementations for benchmarking without actual dependencies."""
    
    def __init__(self):
        self.projections = {}
        self.fusion_layers = {}
        self.attention_masks = {}
        
    def create_projection_layer(self, input_dim: int, output_dim: int):
        """Create a mock projection layer."""
        if TORCH_AVAILABLE:
            return nn.Linear(input_dim, output_dim)
        else:
            return MockProjectionLayer(input_dim, output_dim)
    
    def create_fusion_layer(self, hidden_dim: int, num_modalities: int = 2):
        """Create a mock fusion layer."""
        if TORCH_AVAILABLE:
            return MockFusionModule(hidden_dim, num_modalities)
        else:
            return MockFusionModule(hidden_dim, num_modalities)
    
    def generate_random_embeddings(self, batch_size: int, seq_len: int, dim: int):
        """Generate random embeddings for benchmarking."""
        if TORCH_AVAILABLE:
            return torch.randn(batch_size, seq_len, dim)
        else:
            return np.random.randn(batch_size, seq_len, dim).astype(np.float32)


class MockProjectionLayer:
    """Mock projection layer for benchmarking without PyTorch."""
    
    def __init__(self, input_dim: int, output_dim: int):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weights = np.random.randn(input_dim, output_dim).astype(np.float32) * 0.02
        self.bias = np.zeros(output_dim, dtype=np.float32)
    
    def __call__(self, x):
        if isinstance(x, np.ndarray):
            return np.dot(x, self.weights) + self.bias
        return x  # Fallback
    
    def to(self, device):
        return self


class MockFusionModule:
    """Mock multimodal fusion module."""
    
    def __init__(self, hidden_dim: int, num_modalities: int = 2):
        self.hidden_dim = hidden_dim
        self.num_modalities = num_modalities
        if TORCH_AVAILABLE:
            self.attention = nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)
            self.norm = nn.LayerNorm(hidden_dim)
        self.weights = np.random.randn(num_modalities).astype(np.float32)
        self.weights = self.weights / np.sum(self.weights)
    
    def __call__(self, *modalities):
        if TORCH_AVAILABLE and len(modalities) > 0 and torch.is_tensor(modalities[0]):
            # Simple weighted combination with attention
            combined = sum(w * m for w, m in zip(self.weights, modalities))
            attn_out, _ = self.attention(combined, combined, combined)
            return self.norm(attn_out + combined)
        else:
            # NumPy fallback
            return sum(w * m for w, m in zip(self.weights, modalities))


class EmbeddingProjectionBenchmarks:
    """Benchmarks for embedding projection performance."""
    
    def __init__(self, runner: BenchmarkRunner, mock_components: MockMultimodalComponents):
        self.runner = runner
        self.mock = mock_components
    
    def benchmark_projection_latency(self, dimensions: List[Tuple[int, int]]):
        """Benchmark embedding projection latency for different dimensions."""
        print("\n" + "="*60)
        print("📐 EMBEDDING PROJECTION LATENCY")
        print("="*60)
        
        for input_dim, output_dim in dimensions:
            projection = self.mock.create_projection_layer(input_dim, output_dim)
            embeddings = self.mock.generate_random_embeddings(1, 1, input_dim)
            
            if TORCH_AVAILABLE:
                projection = projection.to(torch.float32)
                embeddings = embeddings.to(torch.float32)
                if torch.cuda.is_available():
                    projection = projection.cuda()
                    embeddings = embeddings.cuda()
            
            def project():
                with torch.no_grad() if TORCH_AVAILABLE else contextmanager(lambda: (yield))():
                    return projection(embeddings)
            
            result = self.runner.run(
                f"projection_{input_dim}_to_{output_dim}",
                "projection",
                project,
                metadata={"input_dim": input_dim, "output_dim": output_dim}
            )
            self.runner.print_result(result)
    
    def benchmark_projection_batch_sizes(self, dim: int = 768):
        """Benchmark projection with different batch sizes."""
        print("\n" + "="*60)
        print("📦 PROJECTION BATCH SIZE SCALING")
        print("="*60)
        
        projection = self.mock.create_projection_layer(dim, dim)
        batch_sizes = [1, 4, 8, 16, 32, 64, 128]
        
        for batch_size in batch_sizes:
            embeddings = self.mock.generate_random_embeddings(batch_size, 1, dim)
            
            if TORCH_AVAILABLE:
                projection = projection.to(torch.float32)
                embeddings = embeddings.to(torch.float32)
            
            def project():
                with torch.no_grad() if TORCH_AVAILABLE else contextmanager(lambda: (yield))():
                    return projection(embeddings)
            
            result = self.runner.run(
                f"projection_batch_{batch_size}",
                "projection",
                project,
                metadata={"batch_size": batch_size, "dim": dim}
            )
            self.runner.print_result(result)
    
    def benchmark_projection_with_memory(self):
        """Benchmark projection with memory profiling."""
        print("\n" + "="*60)
        print("🧠 PROJECTION MEMORY USAGE")
        print("="*60)
        
        dimensions = [(512, 768), (768, 1024), (1024, 2048), (2048, 4096)]
        
        for input_dim, output_dim in dimensions:
            projection = self.mock.create_projection_layer(input_dim, output_dim)
            embeddings = self.mock.generate_random_embeddings(32, 196, input_dim)  # 32 images, 196 patches
            
            if TORCH_AVAILABLE:
                projection = projection.to(torch.float32)
                embeddings = embeddings.to(torch.float32)
            
            def project():
                with torch.no_grad() if TORCH_AVAILABLE else contextmanager(lambda: (yield))():
                    return projection(embeddings)
            
            result = self.runner.run(
                f"projection_memory_{input_dim}_to_{output_dim}",
                "projection_memory",
                project,
                memory_profile=True,
                metadata={"input_dim": input_dim, "output_dim": output_dim, "batch_size": 32}
            )
            self.runner.print_result(result)


class MultimodalFusionBenchmarks:
    """Benchmarks for multimodal fusion performance."""
    
    def __init__(self, runner: BenchmarkRunner, mock_components: MockMultimodalComponents):
        self.runner = runner
        self.mock = mock_components
    
    def benchmark_fusion_throughput(self):
        """Benchmark multimodal fusion throughput (images/sec equivalent)."""
        print("\n" + "="*60)
        print("🔄 MULTIMODAL FUSION THROUGHPUT")
        print("="*60)
        
        hidden_dims = [512, 768, 1024, 2048]
        
        for hidden_dim in hidden_dims:
            fusion = self.mock.create_fusion_layer(hidden_dim, num_modalities=2)
            text_emb = self.mock.generate_random_embeddings(1, 77, hidden_dim)  # Text
            image_emb = self.mock.generate_random_embeddings(1, 196, hidden_dim)  # Image patches
            
            if TORCH_AVAILABLE:
                fusion = fusion.to(torch.float32) if hasattr(fusion, 'to') else fusion
                text_emb = text_emb.to(torch.float32)
                image_emb = image_emb.to(torch.float32)
            
            def fuse():
                with torch.no_grad() if TORCH_AVAILABLE else contextmanager(lambda: (yield))():
                    return fusion(text_emb, image_emb)
            
            result = self.runner.run(
                f"fusion_throughput_{hidden_dim}",
                "fusion",
                fuse,
                calculate_throughput=True,
                throughput_items=1,
                metadata={"hidden_dim": hidden_dim, "num_modalities": 2}
            )
            self.runner.print_result(result)
    
    def benchmark_fusion_modalities_scaling(self):
        """Benchmark fusion with different numbers of modalities."""
        print("\n" + "="*60)
        print("🔗 FUSION MODALITY SCALING")
        print("="*60)
        
        hidden_dim = 768
        modality_counts = [2, 3, 4, 5]
        
        for num_modalities in modality_counts:
            fusion = self.mock.create_fusion_layer(hidden_dim, num_modalities)
            embeddings = [
                self.mock.generate_random_embeddings(1, 50, hidden_dim)
                for _ in range(num_modalities)
            ]
            
            if TORCH_AVAILABLE:
                embeddings = [e.to(torch.float32) for e in embeddings]
            
            def fuse():
                with torch.no_grad() if TORCH_AVAILABLE else contextmanager(lambda: (yield))():
                    return fusion(*embeddings)
            
            result = self.runner.run(
                f"fusion_{num_modalities}_modalities",
                "fusion",
                fuse,
                metadata={"hidden_dim": hidden_dim, "num_modalities": num_modalities}
            )
            self.runner.print_result(result)
    
    def benchmark_fusion_sequence_lengths(self):
        """Benchmark fusion with different sequence lengths."""
        print("\n" + "="*60)
        print("📏 FUSION SEQUENCE LENGTH SCALING")
        print("="*60)
        
        hidden_dim = 768
        seq_lengths = [50, 100, 200, 500, 1000]
        
        for seq_len in seq_lengths:
            fusion = self.mock.create_fusion_layer(hidden_dim, 2)
            text_emb = self.mock.generate_random_embeddings(1, seq_len, hidden_dim)
            image_emb = self.mock.generate_random_embeddings(1, 196, hidden_dim)
            
            if TORCH_AVAILABLE:
                text_emb = text_emb.to(torch.float32)
                image_emb = image_emb.to(torch.float32)
            
            def fuse():
                with torch.no_grad() if TORCH_AVAILABLE else contextmanager(lambda: (yield))():
                    return fusion(text_emb, image_emb)
            
            result = self.runner.run(
                f"fusion_seq_len_{seq_len}",
                "fusion",
                fuse,
                metadata={"hidden_dim": hidden_dim, "text_seq_len": seq_len, "image_patches": 196}
            )
            self.runner.print_result(result)


class AttentionMaskBenchmarks:
    """Benchmarks for attention mask computation."""
    
    def __init__(self, runner: BenchmarkRunner, mock_components: MockMultimodalComponents):
        self.runner = runner
        self.mock = mock_components
    
    def benchmark_attention_mask_computation(self):
        """Benchmark attention mask computation time."""
        print("\n" + "="*60)
        print("🎯 ATTENTION MASK COMPUTATION")
        print("="*60)
        
        seq_lengths = [128, 256, 512, 1024, 2048, 4096]
        
        for seq_len in seq_lengths:
            # Simulate different mask patterns
            if TORCH_AVAILABLE:
                mask_patterns = self._generate_mask_patterns_torch(seq_len)
            else:
                mask_patterns = self._generate_mask_patterns_numpy(seq_len)
            
            def compute_masks():
                for pattern in mask_patterns:
                    if TORCH_AVAILABLE:
                        _ = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
                    else:
                        _ = np.triu(np.ones((seq_len, seq_len)), k=1).astype(bool)
                return mask_patterns
            
            result = self.runner.run(
                f"attention_mask_{seq_len}",
                "attention_mask",
                compute_masks,
                metadata={"seq_len": seq_len, "num_patterns": len(mask_patterns)}
            )
            self.runner.print_result(result)
    
    def _generate_mask_patterns_torch(self, seq_len: int) -> List:
        """Generate different attention mask patterns using PyTorch."""
        patterns = [
            torch.ones(seq_len, seq_len),  # Full attention
            torch.tril(torch.ones(seq_len, seq_len)),  # Causal
            torch.block_diag(*[torch.ones(64, 64) for _ in range(seq_len // 64)]),  # Block
        ]
        return patterns
    
    def _generate_mask_patterns_numpy(self, seq_len: int) -> List:
        """Generate different attention mask patterns using NumPy."""
        patterns = [
            np.ones((seq_len, seq_len)),
            np.tril(np.ones((seq_len, seq_len))),
        ]
        return patterns
    
    def benchmark_multimodal_attention_mask(self):
        """Benchmark attention mask computation for multimodal inputs."""
        print("\n" + "="*60)
        print("🔀 MULTIMODAL ATTENTION MASK")
        print("="*60)
        
        configs = [
            ("text_only", {"text": 77}, 77),
            ("text_image", {"text": 77, "image": 196}, 273),
            ("text_image_audio", {"text": 77, "image": 196, "audio": 300}, 573),
            ("long_context", {"text": 2048, "image": 196}, 2244),
        ]
        
        for name, modality_lengths, total_len in configs:
            if TORCH_AVAILABLE:
                mask = torch.ones(total_len, total_len)
            else:
                mask = np.ones((total_len, total_len))
            
            def compute_multimodal_mask():
                offset = 0
                for modality, length in modality_lengths.items():
                    if TORCH_AVAILABLE:
                        mask[offset:offset+length, offset:offset+length] = 1
                    else:
                        mask[offset:offset+length, offset:offset+length] = 1
                    offset += length
                return mask
            
            result = self.runner.run(
                f"mm_attention_mask_{name}",
                "attention_mask",
                compute_multimodal_mask,
                metadata={
                    "config": name,
                    "modalities": list(modality_lengths.keys()),
                    "total_length": total_len
                }
            )
            self.runner.print_result(result)


class LabelShiftingBenchmarks:
    """Benchmarks for label shifting performance."""
    
    def __init__(self, runner: BenchmarkRunner, mock_components: MockMultimodalComponents):
        self.runner = runner
        self.mock = mock_components
    
    def benchmark_label_shifting(self):
        """Benchmark label shifting for different sequence lengths."""
        print("\n" + "="*60)
        print("🏷️  LABEL SHIFTING PERFORMANCE")
        print("="*60)
        
        seq_lengths = [128, 256, 512, 1024, 2048]
        
        for seq_len in seq_lengths:
            if TORCH_AVAILABLE:
                labels = torch.randint(0, 50000, (1, seq_len))
            else:
                labels = np.random.randint(0, 50000, (1, seq_len))
            
            def shift_labels():
                # Shift labels for next-token prediction
                if TORCH_AVAILABLE:
                    shifted = labels.clone()
                    shifted[:, :-1] = labels[:, 1:]
                    shifted[:, -1] = -100  # Ignore index
                else:
                    shifted = labels.copy()
                    shifted[:, :-1] = labels[:, 1:]
                    shifted[:, -1] = -100
                return shifted
            
            result = self.runner.run(
                f"label_shift_{seq_len}",
                "label_shifting",
                shift_labels,
                metadata={"seq_len": seq_len, "vocab_size": 50000}
            )
            self.runner.print_result(result)
    
    def benchmark_label_shifting_batch_sizes(self):
        """Benchmark label shifting with different batch sizes."""
        print("\n" + "="*60)
        print("📚 LABEL SHIFTING BATCH SCALING")
        print("="*60)
        
        seq_len = 512
        batch_sizes = [1, 4, 8, 16, 32]
        
        for batch_size in batch_sizes:
            if TORCH_AVAILABLE:
                labels = torch.randint(0, 50000, (batch_size, seq_len))
            else:
                labels = np.random.randint(0, 50000, (batch_size, seq_len))
            
            def shift_labels():
                if TORCH_AVAILABLE:
                    shifted = labels.clone()
                    shifted[:, :-1] = labels[:, 1:]
                    shifted[:, -1] = -100
                else:
                    shifted = labels.copy()
                    shifted[:, :-1] = labels[:, 1:]
                    shifted[:, -1] = -100
                return shifted
            
            result = self.runner.run(
                f"label_shift_batch_{batch_size}",
                "label_shifting",
                shift_labels,
                metadata={"batch_size": batch_size, "seq_len": seq_len}
            )
            self.runner.print_result(result)


class OptimizationComparisonBenchmarks:
    """Benchmarks comparing with/without optimization flags."""
    
    def __init__(self, runner: BenchmarkRunner, mock_components: MockMultimodalComponents):
        self.runner = runner
        self.mock = mock_components
    
    def benchmark_optimization_flags(self):
        """Compare performance with different optimization flags."""
        print("\n" + "="*60)
        print("⚡ OPTIMIZATION FLAG COMPARISON")
        print("="*60)
        
        if not TORCH_AVAILABLE:
            print("⚠️  PyTorch not available, skipping optimization benchmarks")
            return
        
        hidden_dim = 768
        fusion = self.mock.create_fusion_layer(hidden_dim, 2)
        text_emb = self.mock.generate_random_embeddings(16, 77, hidden_dim)
        image_emb = self.mock.generate_random_embeddings(16, 196, hidden_dim)
        
        fusion = fusion.to(torch.float32)
        text_emb = text_emb.to(torch.float32)
        image_emb = image_emb.to(torch.float32)
        
        # Benchmark with different optimization settings
        configs = [
            ("baseline", {}),
            ("no_grad", {"grad_enabled": False}),
            ("inference_mode", {"inference_mode": True}),
            ("torch_compile", {"compile": True}),
        ]
        
        for config_name, config in configs:
            def fuse_optimized():
                if config.get("grad_enabled") is False:
                    with torch.no_grad():
                        return fusion(text_emb, image_emb)
                elif config.get("inference_mode"):
                    with torch.inference_mode():
                        return fusion(text_emb, image_emb)
                elif config.get("compile"):
                    # Note: torch.compile would be applied to fusion module
                    return fusion(text_emb, image_emb)
                else:
                    return fusion(text_emb, image_emb)
            
            result = self.runner.run(
                f"optimization_{config_name}",
                "optimization",
                fuse_optimized,
                metadata={"optimization": config_name, "config": config}
            )
            self.runner.print_result(result)
    
    def benchmark_precision_modes(self):
        """Compare different precision modes (FP32, FP16, BF16)."""
        print("\n" + "="*60)
        print("🎯 PRECISION MODE COMPARISON")
        print("="*60)
        
        if not TORCH_AVAILABLE:
            print("⚠️  PyTorch not available, skipping precision benchmarks")
            return
        
        hidden_dim = 768
        projection = self.mock.create_projection_layer(hidden_dim, hidden_dim)
        embeddings = self.mock.generate_random_embeddings(32, 196, hidden_dim)
        
        precision_modes = [
            ("fp32", torch.float32),
        ]
        
        if torch.cuda.is_available():
            precision_modes.extend([
                ("fp16", torch.float16),
                ("bf16", torch.bfloat16),
            ])
        
        for mode_name, dtype in precision_modes:
            proj = projection.to(dtype)
            emb = embeddings.to(dtype)
            if torch.cuda.is_available():
                proj = proj.cuda()
                emb = emb.cuda()
            
            def project():
                with torch.no_grad():
                    return proj(emb)
            
            result = self.runner.run(
                f"precision_{mode_name}",
                "precision",
                project,
                metadata={"precision": mode_name, "dtype": str(dtype)}
            )
            self.runner.print_result(result)


class MemoryUsageBenchmarks:
    """Benchmarks for memory usage during multimodal operations."""
    
    def __init__(self, runner: BenchmarkRunner, mock_components: MockMultimodalComponents):
        self.runner = runner
        self.mock = mock_components
    
    def benchmark_injection_memory(self):
        """Benchmark memory usage during multimodal embedding injection."""
        print("\n" + "="*60)
        print("💾 MULTIMODAL INJECTION MEMORY PROFILE")
        print("="*60)
        
        configs = [
            ("small", {"text": 77, "image": 196}, 768),
            ("medium", {"text": 512, "image": 196, "audio": 300}, 1024),
            ("large", {"text": 2048, "image": 196, "audio": 300, "video": 400}, 2048),
        ]
        
        for name, modality_lengths, hidden_dim in configs:
            # Create embeddings for each modality
            embeddings = {}
            for modality, length in modality_lengths.items():
                embeddings[modality] = self.mock.generate_random_embeddings(1, length, hidden_dim)
            
            if TORCH_AVAILABLE:
                embeddings = {k: v.to(torch.float32) for k, v in embeddings.items()}
            
            def inject_embeddings():
                # Simulate injection process
                combined = []
                for emb in embeddings.values():
                    combined.append(emb)
                if TORCH_AVAILABLE:
                    return torch.cat(combined, dim=1)
                else:
                    return np.concatenate(combined, axis=1)
            
            result = self.runner.run(
                f"injection_memory_{name}",
                "memory",
                inject_embeddings,
                memory_profile=True,
                metadata={
                    "config": name,
                    "hidden_dim": hidden_dim,
                    "modalities": list(modality_lengths.keys()),
                    "total_tokens": sum(modality_lengths.values())
                }
            )
            self.runner.print_result(result)


class RegressionBenchmarks:
    """Benchmarks for detecting performance regressions."""
    
    def __init__(self, runner: BenchmarkRunner, mock_components: MockMultimodalComponents):
        self.runner = runner
        self.mock = mock_components
    
    def run_regression_suite(self):
        """Run core regression benchmarks."""
        print("\n" + "="*60)
        print("🔄 REGRESSION BENCHMARK SUITE")
        print("="*60)
        
        # Core multimodal operations
        baseline_ops = [
            ("projection_512_768", lambda: self._benchmark_projection(512, 768)),
            ("projection_768_1024", lambda: self._benchmark_projection(768, 1024)),
            ("fusion_2modal", lambda: self._benchmark_fusion(768, 2)),
            ("fusion_3modal", lambda: self._benchmark_fusion(768, 3)),
            ("attention_mask_512", lambda: self._benchmark_attention_mask(512)),
            ("label_shift_512", lambda: self._benchmark_label_shift(512)),
        ]
        
        for name, func in baseline_ops:
            result = self.runner.run(
                f"regression_{name}",
                "regression",
                func,
                metadata={"baseline_op": name}
            )
            self.runner.print_result(result)
    
    def _benchmark_projection(self, input_dim: int, output_dim: int):
        """Helper for projection benchmark."""
        projection = self.mock.create_projection_layer(input_dim, output_dim)
        embeddings = self.mock.generate_random_embeddings(1, 1, input_dim)
        if TORCH_AVAILABLE:
            with torch.no_grad():
                return projection(embeddings.to(torch.float32))
        return projection(embeddings)
    
    def _benchmark_fusion(self, hidden_dim: int, num_modalities: int):
        """Helper for fusion benchmark."""
        fusion = self.mock.create_fusion_layer(hidden_dim, num_modalities)
        embeddings = [self.mock.generate_random_embeddings(1, 50, hidden_dim) for _ in range(num_modalities)]
        if TORCH_AVAILABLE:
            with torch.no_grad():
                return fusion(*[e.to(torch.float32) for e in embeddings])
        return fusion(*embeddings)
    
    def _benchmark_attention_mask(self, seq_len: int):
        """Helper for attention mask benchmark."""
        if TORCH_AVAILABLE:
            return torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
        else:
            return np.triu(np.ones((seq_len, seq_len)), k=1)
    
    def _benchmark_label_shift(self, seq_len: int):
        """Helper for label shift benchmark."""
        if TORCH_AVAILABLE:
            labels = torch.randint(0, 50000, (1, seq_len))
            shifted = labels.clone()
            shifted[:, :-1] = labels[:, 1:]
            return shifted
        else:
            labels = np.random.randint(0, 50000, (1, seq_len))
            shifted = labels.copy()
            shifted[:, :-1] = labels[:, 1:]
            return shifted


def run_all_benchmarks(args) -> BenchmarkSuite:
    """Run all benchmark suites."""
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")
    suite = BenchmarkSuite(
        name="Multimodal Architecture Benchmark Suite",
        timestamp=timestamp,
        python_version=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        platform=os.name,
        torch_version=torch.__version__ if TORCH_AVAILABLE else None
    )
    
    # Setup
    iterations = int(os.environ.get("BENCHMARK_ITERATIONS", args.iterations))
    warmup = int(os.environ.get("BENCHMARK_WARMUP", args.warmup))
    runner = BenchmarkRunner(iterations=iterations, warmup=warmup)
    mock_components = MockMultimodalComponents()
    
    # Run benchmarks by category
    if args.category in ["all", "projection"]:
        projection_bench = EmbeddingProjectionBenchmarks(runner, mock_components)
        
        dimensions = [(d, d * 2) for d in args.dimensions]
        projection_bench.benchmark_projection_latency(dimensions)
        projection_bench.benchmark_projection_batch_sizes()
        
        if args.memory_profiling:
            projection_bench.benchmark_projection_with_memory()
    
    if args.category in ["all", "fusion"]:
        fusion_bench = MultimodalFusionBenchmarks(runner, mock_components)
        fusion_bench.benchmark_fusion_throughput()
        fusion_bench.benchmark_fusion_modalities_scaling()
        fusion_bench.benchmark_fusion_sequence_lengths()
    
    if args.category in ["all", "attention"]:
        attention_bench = AttentionMaskBenchmarks(runner, mock_components)
        attention_bench.benchmark_attention_mask_computation()
        attention_bench.benchmark_multimodal_attention_mask()
    
    if args.category in ["all", "label_shifting"]:
        label_bench = LabelShiftingBenchmarks(runner, mock_components)
        label_bench.benchmark_label_shifting()
        label_bench.benchmark_label_shifting_batch_sizes()
    
    if args.category in ["all", "optimization"]:
        opt_bench = OptimizationComparisonBenchmarks(runner, mock_components)
        opt_bench.benchmark_optimization_flags()
        opt_bench.benchmark_precision_modes()
    
    if args.memory_profiling:
        memory_bench = MemoryUsageBenchmarks(runner, mock_components)
        memory_bench.benchmark_injection_memory()
    
    if args.category in ["all", "regression"]:
        regression_bench = RegressionBenchmarks(runner, mock_components)
        regression_bench.run_regression_suite()
    
    # Collect all results
    for result in runner.results:
        suite.add_result(result)
    
    return suite


def print_summary(suite: BenchmarkSuite):
    """Print a summary of all benchmark results."""
    print("\n" + "="*70)
    print("📊 BENCHMARK SUMMARY")
    print("="*70)
    
    # Group by category
    by_category = {}
    for result in suite.results:
        cat = result.category
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append(result)
    
    for category, results in sorted(by_category.items()):
        print(f"\n{category.upper()}:")
        print("-" * 70)
        for result in results:
            throughput_str = f" ({result.throughput:.1f} ops/sec)" if result.throughput else ""
            print(f"  {result.name:50s} {result.mean_time*1000:>8.4f}ms{throughput_str}")
    
    print("\n" + "="*70)
    print(f"Total Benchmarks: {len(suite.results)}")
    print(f"Categories: {len(by_category)}")
    print("="*70)


def main():
    parser = argparse.ArgumentParser(
        description="Multimodal Architecture Benchmark Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python benchmarks/test_multimodal_architect_benchmark.py --all
  python benchmarks/test_multimodal_architect_benchmark.py --category projection
  python benchmarks/test_multimodal_architect_benchmark.py --dimensions 512 1024
  python benchmarks/test_multimodal_architect_benchmark.py --memory-profiling --output results.json
        """
    )
    
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all benchmarks"
    )
    parser.add_argument(
        "--category",
        choices=["all", "projection", "fusion", "attention", "label_shifting", "optimization", "memory", "regression"],
        default="all",
        help="Benchmark category to run"
    )
    parser.add_argument(
        "--dimensions",
        type=int,
        nargs="+",
        default=[512, 768, 1024, 2048],
        help="Embedding dimensions to benchmark (default: 512 768 1024 2048)"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="Number of iterations per benchmark (default: 100)"
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="Number of warmup iterations (default: 10)"
    )
    parser.add_argument(
        "--memory-profiling",
        action="store_true",
        help="Enable memory profiling"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output file for benchmark results (JSON)"
    )
    parser.add_argument(
        "--baseline",
        type=str,
        help="Compare against baseline results file"
    )
    
    args = parser.parse_args()
    
    if args.all:
        args.category = "all"
        args.memory_profiling = True
    
    print("="*70)
    print("🚀 Multimodal Architecture Benchmark Suite")
    print("="*70)
    print(f"PyTorch Available: {TORCH_AVAILABLE}")
    if TORCH_AVAILABLE:
        print(f"PyTorch Version: {torch.__version__}")
        print(f"CUDA Available: {torch.cuda.is_available()}")
    print(f"Iterations: {args.iterations}")
    print(f"Warmup: {args.warmup}")
    print(f"Memory Profiling: {args.memory_profiling}")
    print(f"Category: {args.category}")
    print("="*70)
    
    # Run benchmarks
    suite = run_all_benchmarks(args)
    
    # Print summary
    print_summary(suite)
    
    # Save results
    if args.output:
        suite.save(args.output)
    
    # Compare against baseline if provided
    if args.baseline and os.path.exists(args.baseline):
        print("\n" + "="*70)
        print("📊 BASELINE COMPARISON")
        print("="*70)
        with open(args.baseline, 'r') as f:
            baseline = json.load(f)
        
        baseline_results = {r['name']: r for r in baseline.get('results', [])}
        
        for result in suite.results:
            if result.name in baseline_results:
                baseline_time = baseline_results[result.name]['mean_time']
                current_time = result.mean_time
                delta = ((current_time - baseline_time) / baseline_time) * 100
                
                if abs(delta) > 5:  # >5% change
                    indicator = "🔴" if delta > 0 else "🟢"
                    print(f"{indicator} {result.name:50s} {delta:+.1f}% change")


if __name__ == "__main__":
    main()
