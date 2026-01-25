"""
Unit tests for RULER benchmark runner.
(MOCKED)
"""

import pytest
import torch
import sys
import json
from pathlib import Path
from unittest.mock import MagicMock, patch
from src.benchmarks.ruler_benchmark import RULERBenchmark, RULERConfig, TaskResult, RULERResult

class TestRULERConfig:
    def test_config_defaults(self):
        config = RULERConfig()
        assert config.samples_per_task == 20
        assert 4096 in config.context_lengths

class TestRULERBenchmark:
    @pytest.fixture
    def mock_benchmark(self):
        config = RULERConfig(model_path="/fake", context_lengths=[1024], samples_per_task=1)
        bench = RULERBenchmark(config)
        bench.model = MagicMock()
        bench.model.device = "cpu"
        bench.model.config.max_position_embeddings = 2048
        bench.tokenizer = MagicMock()
        bench.tokenizer.pad_token_id = 0
        return bench

    def test_generate_response(self, mock_benchmark):
        mock_benchmark.tokenizer.return_value = MagicMock(input_ids=torch.zeros((1, 5)))
        mock_benchmark.tokenizer.decode.return_value = "Answer"
        mock_benchmark.model.generate.return_value = torch.zeros((1, 10))
        
        resp, latency = mock_benchmark.generate_response("Test prompt")
        assert resp == "Answer"
        assert latency >= 0

    def test_format_prompt(self, mock_benchmark):
        from src.benchmarks.ruler_tasks import TaskSample
        sample = TaskSample(context="Ctx", question="Q?", expected_answer="A")
        prompt = mock_benchmark.format_prompt(sample)
        assert "Ctx" in prompt
        assert "Q?" in prompt

    @patch("src.benchmarks.ruler_benchmark.get_task")
    def test_evaluate_task(self, mock_get_task, mock_benchmark):
        mock_task = MagicMock()
        mock_task_instance = mock_task.return_value
        mock_task_instance.generate_samples.return_value = [
            MagicMock(context="C", question="Q", expected_answer="A")
        ]
        mock_task_instance.evaluate_response.return_value = (True, 1.0)
        # Fix: RULERBenchmark.evaluate_task uses type(task)(config)
        # So we should pass the CLASS or a mock that behaves like a class
        
        with patch.object(mock_benchmark, 'generate_response', return_value=("A", 100.0)):
            # We need to ensure type(task)(config) works
            class FakeTask:
                def __init__(self, cfg): 
                    self.name = "fake"
                    self.category = MagicMock(value="retrieval")
                def generate_samples(self, n): 
                    return [MagicMock(context="C", question="Q", expected_answer="A")]
                def evaluate_response(self, r, e): return (True, 1.0)
            
            task = FakeTask(None)
            result = mock_benchmark.evaluate_task(task, 1024, 1)
            assert result.accuracy == 1.0
            assert result.correct_count == 1

    @patch("src.benchmarks.ruler_benchmark.get_task")
    def test_run(self, mock_get_task, mock_benchmark):
        class FakeTask:
            def __init__(self, cfg): 
                self.name = "fake"
                self.category = MagicMock(value="retrieval")
            def generate_samples(self, n): return []
        
        mock_get_task.return_value = FakeTask(None)
        
        with patch.object(mock_benchmark, 'evaluate_task') as m_eval:
            m_eval.return_value = TaskResult("fake", 1024, 0.8, 100.0, 1, 1, "retrieval")
            result = mock_benchmark.run()
            assert result.effective_context == 1024
            assert result.overall_scores[1024] == pytest.approx(0.8)

    def test_print_results(self, mock_benchmark, capsys):
        res = RULERResult(
            model_path="/fake",
            task_results=[],
            effective_context=1024,
            overall_scores={1024: 0.9},
            category_scores={"retrieval": {1024: 0.9}}
        )
        mock_benchmark.print_results(res)
        captured = capsys.readouterr()
        assert "RULER BENCHMARK RESULTS" in captured.out

    @patch("transformers.AutoModelForCausalLM.from_pretrained")
    @patch("transformers.AutoTokenizer.from_pretrained")
    def test_setup(self, mock_tok, mock_model, mock_benchmark):
        mock_benchmark.setup()
        assert mock_model.called
        assert mock_tok.called

    def test_main_execution(self, tmp_path):
        out = tmp_path / "res.json"
        with patch("sys.argv", ["ruler.py", "--model", "/fake", "--lengths", "1024", "--samples", "1", "--output", str(out)]), \
             patch("src.benchmarks.ruler_benchmark.RULERBenchmark.setup"), \
             patch("src.benchmarks.ruler_benchmark.RULERBenchmark.run") as m_run:
            
            m_run.return_value = RULERResult("/fake", [], 1024, {1024: 0.9}, {})
            from src.benchmarks.ruler_benchmark import main
            main()
            assert out.exists()

    @patch("src.benchmarks.ruler_benchmark.RULERBenchmark.setup", side_effect=Exception("Fail"))
    def test_main_fail(self, mock_setup):
        with patch("sys.argv", ["ruler.py", "--model", "/fake"]):
            from src.benchmarks.ruler_benchmark import main
            main() # Should handle exception and print usage
