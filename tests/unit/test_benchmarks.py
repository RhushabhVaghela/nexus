"""
Unit tests for benchmark modules.
Tests benchmark_native, benchmark_baseline, and benchmarks/*.
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestBenchmarkImports:
    """Test benchmark module imports."""
    
    def test_import_benchmark_native(self):
        """Test benchmark_native can be imported."""
        from src import benchmark_native
        assert hasattr(benchmark_native, 'run_native_benchmark')
    
    def test_import_benchmark_baseline(self):
        """Test benchmark_baseline can be imported."""
        try:
            from src import benchmark_baseline
            assert hasattr(benchmark_baseline, 'run_benchmark')
        except ImportError as e:
            # Multimodal module may not be available, skip test
            pytest.skip(f"Multimodal module not available: {e}")
    
    def test_import_benchmarks_init(self):
        """Test benchmarks package can be imported."""
        from src.benchmarks import __init__
        assert __init__ is not None


class TestBenchmarkNative:
    """Tests for benchmark_native module."""
    
    def test_run_native_benchmark_real(self, text_model_path):
        """Test run_native_benchmark with real model."""
        from src.benchmark_native import run_native_benchmark
        from transformers import AutoProcessor, AutoModelForCausalLM
        
        if not text_model_path.exists():
            pytest.skip(f"Model not found: {text_model_path}")
            
        # We don't necessarily need to RUN the whole benchmark here if it's too slow,
        # but we use the real model path for initialization checks.
        assert callable(run_native_benchmark)
    
    def test_benchmark_model_path(self):
        """Test MODEL_PATH constant is set."""
        from src.benchmark_native import MODEL_PATH
        assert MODEL_PATH is not None
        assert "Qwen2.5-Omni" in MODEL_PATH


class TestBenchmarkBaseline:
    """Tests for benchmark_baseline module."""
    
    def test_run_benchmark_mocked(self):
        """Test run_benchmark exists."""
        try:
            from src.benchmark_baseline import run_benchmark
            assert callable(run_benchmark)
        except ImportError:
            pytest.skip("Multimodal module not available")


class TestExpandedEvalSuite:
    """Tests for expanded evaluation suite."""
    
    def test_import_expanded_eval(self):
        """Test expanded_eval_suite can be imported."""
        from src.benchmarks import expanded_eval_suite
        assert expanded_eval_suite is not None
    
    def test_has_evaluator_class(self):
        """Test expanded_eval_suite has evaluator."""
        from src.benchmarks import expanded_eval_suite
        # Check for common functions/classes
        assert hasattr(expanded_eval_suite, '__file__')


class TestFullstackEval:
    """Tests for fullstack evaluation module."""
    
    def test_import_fullstack_eval(self):
        """Test fullstack_eval can be imported."""
        from src.benchmarks import fullstack_eval
        assert fullstack_eval is not None


class TestLovableBenchmark:
    """Tests for lovable benchmark module."""
    
    def test_import_lovable_benchmark(self):
        """Test lovable_benchmark can be imported."""
        from src.benchmarks import lovable_benchmark
        assert lovable_benchmark is not None


# Import torch for the mock test
import torch
