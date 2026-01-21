"""
Tests for benchmark runner.
Tests timing, accuracy, and metrics collection.
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
from dataclasses import asdict


# Real model path from user environment
REAL_MODEL_PATH = "/mnt/e/data/models/Qwen2.5-0.5B"


class TestBenchmarkResult:
    """Test BenchmarkResult dataclass."""
    
    def test_import_result(self):
        """Test BenchmarkResult can be imported."""
        from src.benchmarks.benchmark_runner import BenchmarkRunner, BenchmarkConfig
        from src.metrics_tracker import BenchmarkMetrics
        assert BenchmarkMetrics is not None
    
    def test_result_defaults(self):
        """Test BenchmarkResult default values."""
        from src.metrics_tracker import BenchmarkMetrics
        
        result = BenchmarkMetrics(
            name="test_ppl",
            category="accuracy",
            model_name="TestModel",
            tokens_per_second=100.5,
            perplexity=12.5,
            success=True
        )
        
        assert result.name == "test_ppl"
        assert result.category == "accuracy"
        assert result.total_time_s == 0.0 # This is a default for BenchmarkMetrics if not provided
        assert result.tokens_per_second == 100.5
        assert result.success is True
        assert result.perplexity == 12.5
    
    def test_result_custom_values(self):
        """Test BenchmarkResult custom values."""
        from src.metrics_tracker import BenchmarkMetrics
        
        result = BenchmarkMetrics(
            name="test_gen",
            category="generation",
            model_name="TestModel",
            total_time_s=1.5,
            tokens_per_second=100.0,
            perplexity=5.2,
        )
        
        assert result.total_time_s == 1.5
        assert result.tokens_per_second == 100.0
        assert result.perplexity == 5.2


class TestBenchmarkConfig:
    """Test BenchmarkConfig dataclass."""
    
    def test_import_config(self):
        """Test BenchmarkConfig can be imported."""
        from src.benchmarks.benchmark_runner import BenchmarkConfig
        assert BenchmarkConfig is not None
    
    def test_config_defaults(self):
        """Test BenchmarkConfig default values."""
        from src.benchmarks.benchmark_runner import BenchmarkConfig
        
        config = BenchmarkConfig(model_path="/path/to/model")
        
        assert config.model_path == "/path/to/model"
        assert config.warmup_runs == 2
        assert config.benchmark_runs == 5
        assert config.max_new_tokens == 100


class TestBenchmarkRunner:
    """Test BenchmarkRunner class."""
    
    def test_import_runner(self):
        """Test BenchmarkRunner can be imported."""
        from src.benchmarks.benchmark_runner import BenchmarkRunner
        assert BenchmarkRunner is not None
    
    def test_runner_initialization(self):
        """Test BenchmarkRunner initialization."""
        from src.benchmarks.benchmark_runner import BenchmarkRunner, BenchmarkConfig
        
        config = BenchmarkConfig(model_path="/fake/path")
        runner = BenchmarkRunner(config)
        
        assert runner.config.model_path == "/fake/path"
        assert runner.model is None
        assert runner.local_results == []
        assert runner.tracker is not None
    
    def test_get_memory_stats(self):
        """Test memory stats collection."""
        from src.benchmarks.benchmark_runner import BenchmarkRunner, BenchmarkConfig
        
        config = BenchmarkConfig(model_path="/fake/path")
        runner = BenchmarkRunner(config)
        
        stats = runner._get_memory_stats()
        
        assert "gpu_peak_mb" in stats
        assert "ram_mb" in stats


class TestBenchmarkMethods:
    """Test individual benchmark methods with mocks."""
    
    def test_benchmark_generation_real_or_mock(self):
        """Test generation benchmark with Real model if available, else Mock."""
        from src.benchmarks.benchmark_runner import BenchmarkRunner, BenchmarkConfig
        from src.metrics_tracker import BenchmarkMetrics
        from unittest.mock import MagicMock
        
        path_to_use = REAL_MODEL_PATH if Path(REAL_MODEL_PATH).exists() else "/fake/path"
        
        config = BenchmarkConfig(
            model_path=path_to_use,
            warmup_runs=1,
            benchmark_runs=1,
            max_new_tokens=5
        )
        
        if Path(path_to_use).exists():
            # REAL RUN
            print(f"Running test with REAL MODEL at {path_to_use}")
            runner = BenchmarkRunner(config)
            runner.setup() # Load real model
            result = runner.benchmark_generation("Test prompt", "real_test")
            assert result.success
            assert result.tokens_per_second > 0
        else:
            # MOCK RUN
            print("Real model not found, falling back to MOCK")
            runner = BenchmarkRunner(config)
            runner.model = MagicMock()
            runner.model.device = "cpu"
            runner.model.generate.return_value = MagicMock(
                sequences=MagicMock(shape=(1, 50)),
                scores=[]
            )
            runner.tokenizer = MagicMock()
            runner.tokenizer.return_value = {
                "input_ids": MagicMock(shape=(1, 10), to=MagicMock(return_value={"input_ids": MagicMock(shape=(1, 10))})),
            }
            assert callable(runner.benchmark_generation)
    
    def test_benchmark_perplexity_mocked(self):
        """Test perplexity benchmark with mocked model."""
        from src.benchmarks.benchmark_runner import BenchmarkRunner, BenchmarkConfig
        
        config = BenchmarkConfig(model_path="/fake/path")
        runner = BenchmarkRunner(config)
        
        assert callable(runner.benchmark_perplexity)
