#!/usr/bin/env python3
"""
OmniModelLoader Benchmark Suite

Comprehensive benchmarks for model loading performance, covering:
- Model loading performance across different architectures
- SAE model detection overhead
- Tokenizer loading with fallback mechanisms
- Memory usage during model loading
- Architecture detection performance
- Comparison benchmarks for different loading strategies

Usage:
    python benchmarks/test_omni_loader_benchmark.py --all
    python benchmarks/test_omni_loader_benchmark.py --category detection
    python benchmarks/test_omni_loader_benchmark.py --architectures llama qwen mistral
    python benchmarks/test_omni_loader_benchmark.py --memory-profiling
    python benchmarks/test_omni_loader_benchmark.py --output results/loader_benchmark.json

Environment Variables:
    TEST_MODEL_PATH: Path to real model for integration benchmarks
    BENCHMARK_ITERATIONS: Number of iterations for each benchmark (default: 100)
    BENCHMARK_WARMUP: Number of warmup iterations (default: 10)
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
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any, Callable
from unittest.mock import Mock, patch, MagicMock
from contextlib import contextmanager

# Ensure src is in path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from omni.loader import OmniModelLoader, load_omni_model, OmniModelConfig


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
    results: List[BenchmarkResult] = field(default_factory=list)
    
    def add_result(self, result: BenchmarkResult):
        self.results.append(result)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "timestamp": self.timestamp,
            "python_version": self.python_version,
            "platform": self.platform,
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
            metadata: Additional metadata to store
            kwargs: Keyword arguments for func
        """
        times = []
        memory_delta = None
        
        # Warmup iterations
        for _ in range(self.warmup):
            func(*args, **kwargs)
        
        # Actual benchmark iterations
        if memory_profile:
            gc.collect()
            tracemalloc.start()
            start_mem = tracemalloc.get_traced_memory()[0]
        
        for _ in range(self.iterations):
            start = time.perf_counter()
            func(*args, **kwargs)
            end = time.perf_counter()
            times.append(end - start)
        
        if memory_profile:
            end_mem = tracemalloc.get_traced_memory()[0]
            tracemalloc.stop()
            memory_delta = (end_mem - start_mem) / (1024 * 1024)  # MB
        
        # Calculate statistics
        times.sort()
        result = BenchmarkResult(
            name=name,
            category=category,
            iterations=self.iterations,
            total_time=sum(times),
            mean_time=statistics.mean(times),
            median_time=statistics.median(times),
            std_dev=statistics.stdev(times) if len(times) > 1 else 0.0,
            min_time=min(times),
            max_time=max(times),
            p95_time=times[int(len(times) * 0.95)],
            p99_time=times[int(len(times) * 0.99)],
            memory_delta_mb=memory_delta,
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


class ModelLoadingBenchmarks:
    """Benchmarks for model loading performance."""
    
    def __init__(self, runner: BenchmarkRunner, temp_dir: Path):
        self.runner = runner
        self.temp_dir = temp_dir
    
    def setup_mock_model(self, config: Dict, files: Optional[Dict] = None) -> Path:
        """Create a mock model directory for benchmarking."""
        model_path = self.temp_dir / f"mock_model_{time.time_ns()}"
        model_path.mkdir(exist_ok=True)
        
        with open(model_path / "config.json", "w") as f:
            json.dump(config, f)
        
        if files:
            for filename, content in files.items():
                filepath = model_path / filename
                filepath.parent.mkdir(parents=True, exist_ok=True)
                if isinstance(content, dict):
                    with open(filepath, "w") as f:
                        json.dump(content, f)
                else:
                    with open(filepath, "w") as f:
                        f.write(content)
        
        return model_path
    
    def benchmark_architecture_detection(self):
        """Benchmark architecture detection performance."""
        configs = {
            "llama": {"architectures": ["LlamaForCausalLM"], "model_type": "llama"},
            "qwen": {"architectures": ["Qwen2ForCausalLM"], "model_type": "qwen2"},
            "mistral": {"architectures": ["MistralForCausalLM"], "model_type": "mistral"},
            "gemma": {"architectures": ["GemmaForCausalLM"], "model_type": "gemma"},
            "phi": {"architectures": ["Phi3ForCausalLM"], "model_type": "phi3"},
            "deepseek": {"architectures": ["DeepseekForCausalLM"], "model_type": "deepseek"},
            "falcon": {"architectures": ["FalconForCausalLM"], "model_type": "falcon"},
        }
        
        model_paths = {}
        for name, config in configs.items():
            model_paths[name] = self.setup_mock_model(config)
        
        for name, model_path in model_paths.items():
            result = self.runner.run(
                f"architecture_detection_{name}",
                "detection",
                OmniModelLoader.get_model_info,
                model_path,
                metadata={"architecture": name}
            )
            self.runner.print_result(result)
    
    def benchmark_model_category_detection(self):
        """Benchmark model category detection for different types."""
        # Create different model types
        models = {
            "transformers_llm": ({"architectures": ["LlamaForCausalLM"]}, {}),
            "vision_encoder": ({"architectures": ["SigLIPModel"]}, {}),
            "asr_model": ({"architectures": ["WhisperForConditionalGeneration"]}, {}),
            "diffusers": ({}, {"model_index.json": {"_class_name": "StableDiffusionPipeline"}}),
            "sae_model": ({}, {"resid_post": {}, "attn_out": {}}),
        }
        
        model_paths = {}
        for name, (config, files) in models.items():
            model_paths[name] = self.setup_mock_model(config, files)
        
        for name, model_path in model_paths.items():
            result = self.runner.run(
                f"category_detection_{name}",
                "detection",
                OmniModelLoader._detect_model_category,
                model_path,
                metadata={"category": name}
            )
            self.runner.print_result(result)
    
    def benchmark_sae_detection(self):
        """Benchmark SAE model detection performance."""
        # Create SAE model with different configurations
        sae_configs = [
            ("sae_resid_post", {"resid_post": {}}),
            ("sae_mlp_out", {"mlp_out": {}}),
            ("sae_attn_out", {"attn_out": {}}),
            ("sae_transcoder", {"transcoder": {}}),
            ("sae_resid_post_all", {"resid_post_all": {}}),
            ("sae_multi", {"resid_post": {}, "mlp_out": {}, "attn_out": {}}),
        ]
        
        sae_paths = {}
        for name, files in sae_configs:
            sae_paths[name] = self.setup_mock_model({}, files)
        
        for name, model_path in sae_paths.items():
            result = self.runner.run(
                f"sae_detection_{name}",
                "sae",
                OmniModelLoader._is_sae_model,
                model_path,
                metadata={"sae_type": name}
            )
            self.runner.print_result(result)
    
    def benchmark_sae_base_model_extraction(self):
        """Benchmark SAE base model extraction performance."""
        # Create SAE model with base model info
        sae_path = self.setup_mock_model({}, {
            "resid_post": {
                "layer_0": {
                    "config.json": json.dumps({"model_name": "google/gemma-2b-it"})
                }
            }
        })
        
        result = self.runner.run(
            "sae_base_model_extraction",
            "sae",
            OmniModelLoader._get_sae_base_model,
            sae_path,
            metadata={"base_model": "google/gemma-2b-it"}
        )
        self.runner.print_result(result)
    
    def benchmark_tokenizer_loading(self):
        """Benchmark tokenizer loading with various scenarios."""
        with patch('transformers.AutoTokenizer.from_pretrained') as mock_tokenizer:
            mock_instance = Mock()
            mock_instance.pad_token = "<pad>"
            mock_tokenizer.return_value = mock_instance
            
            # Normal model
            normal_model = self.setup_mock_model({"model_type": "gpt2"})
            result = self.runner.run(
                "tokenizer_load_normal",
                "tokenizer",
                self._load_tokenizer_wrapper,
                normal_model,
                metadata={"scenario": "normal"}
            )
            self.runner.print_result(result)
            
            # SAE model
            sae_model = self.setup_mock_model({}, {"resid_post": {}})
            with patch.object(OmniModelLoader, '_get_sae_base_model', return_value="gpt2"):
                result = self.runner.run(
                    "tokenizer_load_sae_fallback",
                    "tokenizer",
                    self._load_tokenizer_wrapper,
                    sae_model,
                    metadata={"scenario": "sae_fallback"}
                )
                self.runner.print_result(result)
    
    def _load_tokenizer_wrapper(self, model_path):
        """Wrapper for tokenizer loading that creates a fresh loader each time."""
        loader = OmniModelLoader(model_path)
        try:
            return loader._load_tokenizer(model_path, trust_remote_code=True)
        except:
            pass  # Expected in benchmark without real tokenizer files
    
    def benchmark_support_checking(self):
        """Benchmark model support checking performance."""
        configs = {
            "supported_llama": {"architectures": ["LlamaForCausalLM"], "model_type": "llama"},
            "supported_qwen": {"architectures": ["Qwen2ForCausalLM"], "model_type": "qwen2"},
            "supported_mistral": {"architectures": ["MistralForCausalLM"], "model_type": "mistral"},
            "supported_gemma": {"architectures": ["GemmaForCausalLM"], "model_type": "gemma"},
            "unsupported_custom": {"architectures": ["UnknownArch"], "model_type": "unknown"},
            "glm4_moe_lite": {"architectures": ["Glm4MoeLiteForCausalLM"], "model_type": "glm4_moe_lite"},
            "step_robotics": {"architectures": ["Step3VL10BForCausalLM"], "model_type": "step_robotics"},
            "qwen3": {"architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3"},
        }
        
        model_paths = {}
        for name, config in configs.items():
            model_paths[name] = self.setup_mock_model(config)
        
        for name, model_path in model_paths.items():
            result = self.runner.run(
                f"support_check_{name}",
                "support",
                OmniModelLoader.is_model_supported,
                model_path,
                metadata={"model": name}
            )
            self.runner.print_result(result)
    
    def benchmark_omni_model_detection(self):
        """Benchmark Omni model detection performance."""
        configs = {
            "omni_qwen": {"model_type": "qwen2_5_omni", "architectures": ["Qwen2_5OmniForConditionalGeneration"]},
            "omni_any2any": {"model_type": "any-to-any", "architectures": ["SomeModel"]},
            "omni_qwen3": {"model_type": "qwen3omni", "architectures": ["Qwen3OmniForConditionalGeneration"]},
            "regular_llama": {"architectures": ["LlamaForCausalLM"], "model_type": "llama"},
            "regular_qwen": {"architectures": ["Qwen2ForCausalLM"], "model_type": "qwen2"},
            "regular_mistral": {"architectures": ["MistralForCausalLM"], "model_type": "mistral"},
        }
        
        model_paths = {}
        for name, config in configs.items():
            model_paths[name] = self.setup_mock_model(config)
        
        for name, model_path in model_paths.items():
            result = self.runner.run(
                f"omni_detection_{name}",
                "detection",
                OmniModelLoader.is_omni_model,
                model_path,
                metadata={"model_type": name}
            )
            self.runner.print_result(result)
    
    def benchmark_with_memory_profiling(self):
        """Run benchmarks with memory profiling enabled."""
        print("\n" + "="*60)
        print("🧠 MEMORY PROFILING BENCHMARKS")
        print("="*60)
        
        # Model info retrieval with memory
        model_path = self.setup_mock_model({
            "architectures": ["LlamaForCausalLM"],
            "model_type": "llama",
            "quantization_config": {"load_in_4bit": True}
        })
        
        result = self.runner.run(
            "model_info_with_memory",
            "memory",
            OmniModelLoader.get_model_info,
            model_path,
            memory_profile=True,
            metadata={"operation": "model_info"}
        )
        self.runner.print_result(result)
        
        # Category detection with memory
        result = self.runner.run(
            "category_detection_with_memory",
            "memory",
            OmniModelLoader._detect_model_category,
            model_path,
            memory_profile=True,
            metadata={"operation": "category_detection"}
        )
        self.runner.print_result(result)
        
        # Support checking with memory
        result = self.runner.run(
            "support_check_with_memory",
            "memory",
            OmniModelLoader.is_model_supported,
            model_path,
            memory_profile=True,
            metadata={"operation": "support_check"}
        )
        self.runner.print_result(result)
        
        # SAE detection with memory
        sae_path = self.setup_mock_model({}, {"resid_post": {}, "mlp_out": {}})
        result = self.runner.run(
            "sae_detection_with_memory",
            "memory",
            OmniModelLoader._is_sae_model,
            sae_path,
            memory_profile=True,
            metadata={"operation": "sae_detection"}
        )
        self.runner.print_result(result)


class ComparisonBenchmarks:
    """Benchmarks comparing different loading strategies."""
    
    def __init__(self, runner: BenchmarkRunner, temp_dir: Path):
        self.runner = runner
        self.temp_dir = temp_dir
    
    def setup_mock_model(self, config: Dict, files: Optional[Dict] = None) -> Path:
        """Create a mock model directory."""
        model_path = self.temp_dir / f"mock_model_{time.time_ns()}"
        model_path.mkdir(exist_ok=True)
        
        with open(model_path / "config.json", "w") as f:
            json.dump(config, f)
        
        if files:
            for filename, content in files.items():
                filepath = model_path / filename
                filepath.parent.mkdir(parents=True, exist_ok=True)
                if isinstance(content, dict):
                    with open(filepath, "w") as f:
                        json.dump(content, f)
                else:
                    with open(filepath, "w") as f:
                        f.write(content)
        
        return model_path
    
    def benchmark_loading_strategies(self):
        """Compare different model loading strategies."""
        print("\n" + "="*60)
        print("⚡ LOADING STRATEGY COMPARISON")
        print("="*60)
        
        # Create models of different categories
        models = {
            "transformers": ({"architectures": ["LlamaForCausalLM"]}, {}),
            "vision": ({"architectures": ["SigLIPModel"]}, {}),
            "asr": ({"architectures": ["WhisperForConditionalGeneration"]}, {}),
        }
        
        for model_name, (config, files) in models.items():
            model_path = self.setup_mock_model(config, files)
            
            # Time category detection
            result = self.runner.run(
                f"strategy_detection_{model_name}",
                "comparison",
                OmniModelLoader._detect_model_category,
                model_path,
                metadata={"strategy": "detection", "model_type": model_name}
            )
            self.runner.print_result(result)
            
            # Time support check
            result = self.runner.run(
                f"strategy_support_{model_name}",
                "comparison",
                OmniModelLoader.is_model_supported,
                model_path,
                metadata={"strategy": "support", "model_type": model_name}
            )
            self.runner.print_result(result)
            
            # Time full info retrieval
            result = self.runner.run(
                f"strategy_full_info_{model_name}",
                "comparison",
                OmniModelLoader.get_model_info,
                model_path,
                metadata={"strategy": "full_info", "model_type": model_name}
            )
            self.runner.print_result(result)
    
    def benchmark_architecture_variations(self):
        """Benchmark different architecture configurations."""
        print("\n" + "="*60)
        print("🏗️  ARCHITECTURE VARIATION BENCHMARKS")
        print("="*60)
        
        architectures = [
            ("llama_simple", {"architectures": ["LlamaForCausalLM"]}),
            ("llama_with_quant", {"architectures": ["LlamaForCausalLM"], "quantization_config": {}}),
            ("llama_with_talker", {"architectures": ["LlamaForCausalLM"], "talker_config": {}, "audio_config": {}}),
            ("qwen_omni", {"architectures": ["Qwen2_5OmniForConditionalGeneration"], "talker_config": {}}),
            ("complex_config", {
                "architectures": ["LlamaForCausalLM"],
                "model_type": "llama",
                "quantization_config": {"load_in_4bit": True},
                "talker_config": {"hidden_size": 1536},
                "audio_config": {"sample_rate": 16000}
            }),
        ]
        
        for name, config in architectures:
            model_path = self.setup_mock_model(config)
            
            result = self.runner.run(
                f"arch_variation_{name}",
                "architecture",
                OmniModelLoader.get_model_info,
                model_path,
                metadata={"config_type": name, "config_size": len(json.dumps(config))}
            )
            self.runner.print_result(result)


class RegressionBenchmarks:
    """Benchmarks for detecting performance regressions."""
    
    def __init__(self, runner: BenchmarkRunner, temp_dir: Path):
        self.runner = runner
        self.temp_dir = temp_dir
    
    def setup_mock_model(self, config: Dict, files: Optional[Dict] = None) -> Path:
        """Create a mock model directory."""
        model_path = self.temp_dir / f"mock_model_{time.time_ns()}"
        model_path.mkdir(exist_ok=True)
        
        with open(model_path / "config.json", "w") as f:
            json.dump(config, f)
        
        if files:
            for filename, content in files.items():
                filepath = model_path / filename
                filepath.parent.mkdir(parents=True, exist_ok=True)
                if isinstance(content, dict):
                    with open(filepath, "w") as f:
                        json.dump(content, f)
                else:
                    with open(filepath, "w") as f:
                        f.write(content)
        
        return model_path
    
    def run_regression_suite(self):
        """Run the complete regression benchmark suite."""
        print("\n" + "="*60)
        print("🔄 REGRESSION TEST BENCHMARKS")
        print("="*60)
        
        # Core operations that should have consistent performance
        baseline_ops = [
            ("get_model_info_llama", lambda: OmniModelLoader.get_model_info(
                self.setup_mock_model({"architectures": ["LlamaForCausalLM"]})
            )),
            ("get_model_info_qwen", lambda: OmniModelLoader.get_model_info(
                self.setup_mock_model({"architectures": ["Qwen2ForCausalLM"]})
            )),
            ("is_supported_llama", lambda: OmniModelLoader.is_model_supported(
                self.setup_mock_model({"architectures": ["LlamaForCausalLM"]})
            )),
            ("detect_category_llama", lambda: OmniModelLoader._detect_model_category(
                self.setup_mock_model({"architectures": ["LlamaForCausalLM"]})
            )),
            ("detect_category_vision", lambda: OmniModelLoader._detect_model_category(
                self.setup_mock_model({"architectures": ["SigLIPModel"]})
            )),
            ("is_omni_model_false", lambda: OmniModelLoader.is_omni_model(
                self.setup_mock_model({"architectures": ["LlamaForCausalLM"], "model_type": "llama"})
            )),
            ("is_omni_model_true", lambda: OmniModelLoader.is_omni_model(
                self.setup_mock_model({"model_type": "qwen2_5_omni"})
            )),
            ("is_sae_model", lambda: OmniModelLoader._is_sae_model(
                self.setup_mock_model({}, {"resid_post": {}})
            )),
            ("is_sae_model_false", lambda: OmniModelLoader._is_sae_model(
                self.setup_mock_model({"architectures": ["LlamaForCausalLM"]})
            )),
        ]
        
        for name, func in baseline_ops:
            result = self.runner.run(
                f"regression_{name}",
                "regression",
                func,
                metadata={"baseline_op": name}
            )
            self.runner.print_result(result)


def run_all_benchmarks(args) -> BenchmarkSuite:
    """Run all benchmark suites."""
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")
    suite = BenchmarkSuite(
        name="OmniModelLoader Benchmark Suite",
        timestamp=timestamp,
        python_version=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        platform=os.name
    )
    
    # Setup
    iterations = int(os.environ.get("BENCHMARK_ITERATIONS", args.iterations))
    warmup = int(os.environ.get("BENCHMARK_WARMUP", args.warmup))
    runner = BenchmarkRunner(iterations=iterations, warmup=warmup)
    temp_dir = Path(tempfile.mkdtemp())
    
    try:
        # Model Loading Benchmarks
        if args.category in ["all", "detection"]:
            print("\n" + "="*60)
            print("🔍 ARCHITECTURE DETECTION BENCHMARKS")
            print("="*60)
            loading_bench = ModelLoadingBenchmarks(runner, temp_dir)
            loading_bench.benchmark_architecture_detection()
            loading_bench.benchmark_model_category_detection()
            loading_bench.benchmark_omni_model_detection()
        
        # SAE Benchmarks
        if args.category in ["all", "sae"]:
            print("\n" + "="*60)
            print("🧬 SAE MODEL BENCHMARKS")
            print("="*60)
            loading_bench = ModelLoadingBenchmarks(runner, temp_dir)
            loading_bench.benchmark_sae_detection()
            loading_bench.benchmark_sae_base_model_extraction()
        
        # Tokenizer Benchmarks
        if args.category in ["all", "tokenizer"]:
            print("\n" + "="*60)
            print("🔤 TOKENIZER BENCHMARKS")
            print("="*60)
            loading_bench = ModelLoadingBenchmarks(runner, temp_dir)
            loading_bench.benchmark_tokenizer_loading()
        
        # Support Checking Benchmarks
        if args.category in ["all", "support"]:
            print("\n" + "="*60)
            print("✅ SUPPORT CHECK BENCHMARKS")
            print("="*60)
            loading_bench = ModelLoadingBenchmarks(runner, temp_dir)
            loading_bench.benchmark_support_checking()
        
        # Memory Profiling
        if args.memory_profiling:
            loading_bench = ModelLoadingBenchmarks(runner, temp_dir)
            loading_bench.benchmark_with_memory_profiling()
        
        # Comparison Benchmarks
        if args.category in ["all", "comparison"]:
            comparison_bench = ComparisonBenchmarks(runner, temp_dir)
            comparison_bench.benchmark_loading_strategies()
            comparison_bench.benchmark_architecture_variations()
        
        # Regression Benchmarks
        if args.category in ["all", "regression"]:
            regression_bench = RegressionBenchmarks(runner, temp_dir)
            regression_bench.run_regression_suite()
        
        # Collect all results
        for result in runner.results:
            suite.add_result(result)
        
    finally:
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)
    
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
            print(f"  {result.name:50s} {result.mean_time*1000:>8.4f}ms")
    
    print("\n" + "="*70)
    print(f"Total Benchmarks: {len(suite.results)}")
    print(f"Categories: {len(by_category)}")
    print("="*70)


def main():
    parser = argparse.ArgumentParser(
        description="OmniModelLoader Benchmark Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python benchmarks/test_omni_loader_benchmark.py --all
  python benchmarks/test_omni_loader_benchmark.py --category detection
  python benchmarks/test_omni_loader_benchmark.py --memory-profiling
  python benchmarks/test_omni_loader_benchmark.py --output results.json
        """
    )
    
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all benchmarks"
    )
    parser.add_argument(
        "--category",
        choices=["all", "detection", "sae", "tokenizer", "support", "comparison", "regression"],
        default="all",
        help="Benchmark category to run"
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
    print("🚀 OmniModelLoader Benchmark Suite")
    print("="*70)
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
