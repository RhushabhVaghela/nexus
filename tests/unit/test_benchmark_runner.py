"""
Tests for benchmark runner.
Tests timing, accuracy, and metrics collection.
"""

import pytest
import sys
import torch
import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch
from src.benchmarks.benchmark_runner import BenchmarkRunner, BenchmarkConfig
from src.metrics_tracker import BenchmarkMetrics

class TestBenchmarkResult:
    """Test BenchmarkMetrics dataclass."""
    
    def test_import_result(self):
        """Test BenchmarkMetrics can be imported."""
        assert BenchmarkMetrics is not None
    
    def test_result_defaults(self):
        """Test BenchmarkMetrics default values."""
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
        assert result.tokens_per_second == 100.5
        assert result.success is True
        assert result.perplexity == 12.5

class TestBenchmarkConfig:
    """Test BenchmarkConfig dataclass."""
    
    def test_config_defaults(self):
        """Test BenchmarkConfig default values."""
        config = BenchmarkConfig(model_path="/path/to/model")
        assert config.model_path == "/path/to/model"
        assert config.warmup_runs == 2
        assert config.benchmark_runs == 5
        assert config.max_new_tokens == 100

class TestBenchmarkMethods:
    """Detailed tests for BenchmarkRunner using mocks."""
    
    @pytest.fixture
    def mock_runner(self, tmp_path):
        config = BenchmarkConfig(
            model_path="/fake/path",
            output_dir=str(tmp_path),
            warmup_runs=1,
            benchmark_runs=1,
            max_new_tokens=5
        )
        runner = BenchmarkRunner(config)
        runner.model = MagicMock()
        runner.model.device = "cpu"
        runner.tokenizer = MagicMock()
        return runner

    def test_benchmark_generation_success(self, mock_runner):
        # Mock tokenizer to return a dict of tensors
        mock_ids = MagicMock()
        mock_ids.shape = (1, 10)
        mock_ids.to.return_value = mock_ids
        mock_runner.tokenizer.return_value = {"input_ids": mock_ids}
        
        # Mock model.generate
        mock_gen_out = MagicMock()
        mock_gen_out.sequences = torch.zeros((1, 20))
        mock_runner.model.generate.return_value = mock_gen_out
        
        result = mock_runner.benchmark_generation("Test prompt")
        assert result.success
        assert result.input_tokens == 10
        assert result.output_tokens == 10
        assert result.tokens_per_second > 0

    def test_benchmark_generation_error(self, mock_runner):
        mock_runner.tokenizer.side_effect = Exception("Tokenize failed")
        result = mock_runner.benchmark_generation("Test prompt")
        assert not result.success
        assert "Tokenize failed" in result.error

    def test_benchmark_perplexity_success(self, mock_runner):
        mock_ids = MagicMock()
        mock_ids.shape = (1, 10)
        mock_ids.to.return_value = mock_ids
        mock_runner.tokenizer.return_value = {"input_ids": mock_ids}
        
        mock_out = MagicMock()
        mock_out.loss = torch.tensor(1.0)
        mock_runner.model.return_value = mock_out
        
        result = mock_runner.benchmark_perplexity("Test text")
        assert result.success
        assert result.loss == 1.0
        assert result.perplexity == pytest.approx(2.718, abs=0.01)

    def test_benchmark_perplexity_error(self, mock_runner):
        mock_runner.tokenizer.side_effect = Exception("Failed")
        result = mock_runner.benchmark_perplexity("Test text")
        assert not result.success

    def test_get_sample_prompts_fallback(self, mock_runner):
        # Force exception in real sampling to hit fallbacks
        with patch("src.data.universal_loader.load_dataset_universal", side_effect=Exception("No data")):
            prompts = mock_runner.get_sample_prompts()
            assert len(prompts) >= 5
            assert "Explain quantum computing" in prompts[0]

    def test_runner_setup(self, mock_runner):
        with patch("src.omni.loader.OmniModelLoader.load", return_value=(MagicMock(), MagicMock())):
            mock_runner.setup()
            assert mock_runner.model is not None
            assert mock_runner.tokenizer is not None

    def test_benchmark_generation_auto_setup(self, mock_runner):
        mock_runner.model = None # Force setup()
        mock_runner.tokenizer = None
        with patch.object(mock_runner, 'setup') as m_setup:
            # We must set model/tokenizer after setup or mock setup to do it
            def side_effect():
                mock_runner.model = MagicMock()
                mock_runner.model.device = "cpu"
                mock_runner.tokenizer = MagicMock()
            m_setup.side_effect = side_effect
            
            mock_ids = MagicMock()
            mock_ids.shape = (1, 1)
            mock_ids.to.return_value = mock_ids
            # wait, side_effect needs to ensure tokenizer returns a dict
            mock_runner.benchmark_generation("prompt")
            assert m_setup.called

    def test_benchmark_generation_first_token(self, mock_runner):
        mock_ids = MagicMock(); mock_ids.shape = (1, 1); mock_ids.to.return_value = mock_ids
        mock_runner.tokenizer.return_value = {"input_ids": mock_ids}
        
        mock_gen_out = MagicMock()
        mock_gen_out.sequences = torch.zeros((1, 5))
        mock_gen_out.scores = [torch.zeros(1, 10)] # Hit line 140-141
        mock_runner.model.generate.return_value = mock_gen_out
        
        res = mock_runner.benchmark_generation("P")
        assert res.first_token_time_s > 0

    def test_get_sample_prompts_formats(self, mock_runner, tmp_path):
        # Test line 230-243 (text, instruction, query, etc.)
        formats = [
            {"text": "Text prompt"},
            {"instruction": "Instr prompt"},
            {"query": "Query prompt"},
            {"other": "Other prompt more than 5 chars"},
            "Raw string prompt"
        ]
        
        for i, fmt in enumerate(formats):
            p = tmp_path / f"d_{i}.jsonl"
            with open(p, 'w') as f: f.write(json.dumps(fmt) + "\n")
            
            with patch("src.benchmarks.benchmark_runner.ALL_DATASETS", {"t": [str(p)]}), \
                 patch("pathlib.Path.exists", return_value=True):
                prompts = mock_runner.get_sample_prompts()
                # Check if it was picked up
                assert any("prompt" in pr for pr in prompts)

    def test_run_all(self, mock_runner):
        with patch.object(mock_runner, 'setup'), \
             patch.object(mock_runner, 'get_sample_prompts', return_value=["P1"]), \
             patch.object(mock_runner, 'benchmark_generation') as m_gen, \
             patch.object(mock_runner, 'benchmark_perplexity') as m_ppl:
            
            m_gen.return_value = BenchmarkMetrics(name="g", category="generation", model_name="m", success=True)
            m_ppl.return_value = BenchmarkMetrics(name="p", category="accuracy", model_name="m", success=True)
            
            mock_runner.run_all()
            assert m_gen.called
            assert m_ppl.called

    def test_print_summary(self, mock_runner, capsys):
        mock_runner.local_results = [
            BenchmarkMetrics(name="g", category="generation", model_name="m", success=True, tokens_per_second=10.0, latency_ms=100.0)
        ]
        mock_runner.print_summary()
        captured = capsys.readouterr()
        assert "Generation Performance" in captured.out
        assert "10.0 tokens/sec" in captured.out

    def test_main_execution(self):
        with patch("sys.argv", ["runner.py", "--model", "/fake", "--runs", "1"]), \
             patch("src.benchmarks.benchmark_runner.BenchmarkRunner.run_all"), \
             patch("src.benchmarks.benchmark_runner.BenchmarkRunner.print_summary"):
            from src.benchmarks.benchmark_runner import main
            assert main() == 0

    def test_memory_stats(self, mock_runner):
        with patch("torch.cuda.is_available", return_value=True), \
             patch("torch.cuda.max_memory_allocated", return_value=1024*1024), \
             patch("torch.cuda.memory_reserved", return_value=2048*1024):
            stats = mock_runner._get_memory_stats()
            assert stats["gpu_peak_mb"] == 1.0
            assert stats["gpu_reserved_mb"] == 2.0

    def test_clear_memory(self, mock_runner):
        with patch("torch.cuda.is_available", return_value=True), \
             patch("torch.cuda.empty_cache") as m_empty:
            mock_runner._clear_memory()
            assert m_empty.called
