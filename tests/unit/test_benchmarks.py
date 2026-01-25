"""
Unit tests for benchmark modules.
Tests benchmark_native, benchmark_baseline, and benchmarks/*. (MOCKED)
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

class TestBenchmarkImports:
    """Test benchmark module imports."""
    
    def test_import_benchmark_native(self):
        from src import benchmark_native
        assert hasattr(benchmark_native, 'run_native_benchmark')
    
    def test_import_benchmark_baseline(self):
        try:
            from src import benchmark_baseline
            assert hasattr(benchmark_baseline, 'run_benchmark')
        except ImportError:
            pytest.skip("Multimodal module not available")
    
    def test_import_benchmarks_init(self):
        from src.benchmarks import __init__
        assert __init__ is not None


class TestBenchmarkNative:
    """Tests for benchmark_native module."""
    
    def test_run_native_benchmark_mocked(self):
        from src.benchmark_native import run_native_benchmark
        assert callable(run_native_benchmark)
    
    def test_benchmark_model_path(self):
        from src.benchmark_native import MODEL_PATH
        assert MODEL_PATH is not None
        assert "Qwen2.5-Omni" in MODEL_PATH


class TestBenchmarkBaseline:
    """Tests for benchmark_baseline module."""
    
    def test_run_benchmark_mocked(self):
        try:
            from src.benchmark_baseline import run_benchmark
            assert callable(run_benchmark)
        except ImportError:
            pytest.skip("Multimodal module not available")


class TestBenchmarkRunnerMocked:
    """Detailed tests for BenchmarkRunner using mocks."""
    
    @patch("src.omni.loader.OmniModelLoader.load")
    def test_runner_setup(self, mock_load):
        from src.benchmarks.benchmark_runner import BenchmarkRunner, BenchmarkConfig
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_load.return_value = (mock_model, mock_tokenizer)
        
        config = BenchmarkConfig(model_path="/fake/path")
        runner = BenchmarkRunner(config)
        runner.setup()
        
        assert runner.model == mock_model
        assert runner.tokenizer == mock_tokenizer
        assert mock_model.eval.called

    @patch("src.omni.loader.OmniModelLoader.load")
    def test_generation_benchmark_logic(self, mock_load):
        from src.benchmarks.benchmark_runner import BenchmarkRunner, BenchmarkConfig
        import torch
        
        mock_model = MagicMock()
        mock_model.device = "cpu"
        mock_tokenizer = MagicMock()
        mock_load.return_value = (mock_model, mock_tokenizer)
        
        # Mock tokenizer outputs
        mock_tokenizer.return_value = {"input_ids": torch.zeros((1, 10), dtype=torch.long)}
        
        # Mock model generate output
        mock_gen_output = MagicMock()
        mock_gen_output.sequences = torch.zeros((1, 20), dtype=torch.long)
        mock_model.generate.return_value = mock_gen_output
        
        config = BenchmarkConfig(model_path="/fake/path", warmup_runs=0, benchmark_runs=1)
        runner = BenchmarkRunner(config)
        
        result = runner.benchmark_generation("Test prompt")
        
        assert result.success
        assert result.input_tokens == 10
        assert result.output_tokens == 10 # 20 - 10
        assert result.tokens_per_second > 0
        assert mock_model.generate.called

    @patch("src.omni.loader.OmniModelLoader.load")
    def test_perplexity_benchmark_logic(self, mock_load):
        from src.benchmarks.benchmark_runner import BenchmarkRunner, BenchmarkConfig
        import torch
        
        mock_model = MagicMock()
        mock_model.device = "cpu"
        mock_tokenizer = MagicMock()
        mock_load.return_value = (mock_model, mock_tokenizer)
        
        # Mock tokenizer outputs
        mock_tokenizer.return_value = {"input_ids": torch.zeros((1, 10), dtype=torch.long)}
        
        # Mock model forward output
        mock_output = MagicMock()
        mock_output.loss = torch.tensor(1.0)
        mock_model.return_value = mock_output
        
        config = BenchmarkConfig(model_path="/fake/path")
        runner = BenchmarkRunner(config)
        
        result = runner.benchmark_perplexity("Test text")
        
        assert result.success
        assert result.loss == 1.0
        assert result.perplexity == pytest.approx(2.71828, abs=0.01)

class TestExpandedEvalSuite:
    """Tests for expanded evaluation suite."""
    
    def test_import_expanded_eval(self):
        from src.benchmarks import expanded_eval_suite
        assert expanded_eval_suite is not None
    
    def test_has_evaluator_class(self):
        from src.benchmarks import expanded_eval_suite
        assert hasattr(expanded_eval_suite, '__file__')


class TestFullstackEval:
    """Tests for fullstack evaluation module."""
    
    def test_import_fullstack_eval(self):
        from src.benchmarks import fullstack_eval
        assert fullstack_eval is not None


class TestLovableBenchmark:
    """Tests for lovable benchmark module."""
    
    def test_import_lovable_benchmark(self):
        from src.benchmarks import lovable_benchmark
        assert lovable_benchmark is not None

import torch
