"""
Unit tests for Expanded Evaluation Suite.
(MOCKED)
"""

import pytest
import sys
import json
from pathlib import Path
from unittest.mock import MagicMock, patch
from src.benchmarks.expanded_eval_suite import (
    ExpandedEvalSuite, MultipleChoiceEvaluator, MathEvaluator,
    CodeEvaluator, SWEBenchEvaluator, BenchmarkSample, get_evaluator
)

class TestExpandedEvalSuite:
    @pytest.fixture
    def mock_model(self):
        return lambda x: "A" # Dummy MC answer

    def test_mc_evaluator(self):
        evaluator = MultipleChoiceEvaluator("mmlu")
        sample = BenchmarkSample("1", "Q", ["Paris", "London", "Berlin"], "Paris", "subj", "mmlu")
        assert evaluator.evaluate_sample(sample, "A") is True
        assert evaluator.evaluate_sample(sample, "Paris") is True
        assert evaluator.evaluate_sample(sample, "B") is False

    def test_math_evaluator(self):
        evaluator = MathEvaluator("gsm8k")
        sample = BenchmarkSample("1", "1+1?", [], "2", "math", "gsm8k")
        assert evaluator.evaluate_sample(sample, "The answer is 2.") is True
        assert evaluator.evaluate_sample(sample, "It is 2.0") is True
        assert evaluator.evaluate_sample(sample, "3") is False

    @patch("subprocess.run")
    def test_code_evaluator(self, mock_run):
        evaluator = CodeEvaluator("humaneval")
        sample = BenchmarkSample("1", "def f():", [], "return 1", "assert f()==1", "humaneval")
        
        mock_run.return_value = MagicMock(return_code=0) # Success
        # Note: evaluate_sample returns result.returncode == 0
        # Wait, the code has result.returncode == 0. 
        # My mock should have returncode (no underscore).
        mock_run.return_value.returncode = 0
        
        assert evaluator.evaluate_sample(sample, " return 1") is True
        
        mock_run.return_value.returncode = 1
        assert evaluator.evaluate_sample(sample, " return 2") is False

    def test_swe_evaluator(self):
        evaluator = SWEBenchEvaluator("swe_bench_lite")
        sample = BenchmarkSample("1", "prob", [], "patch", "repo", "swe")
        assert evaluator.evaluate_sample(sample, "diff --git a/file b/file") is True
        assert evaluator.evaluate_sample(sample, "@@ -1,1 +1,1 @@") is True
        assert evaluator.evaluate_sample(sample, "not a patch") is False

    @patch("src.benchmarks.expanded_eval_suite.load_dataset")
    def test_load_samples_mc(self, mock_load):
        mock_load.return_value = [
            {"question": "Q", "choices": ["A", "B"], "answer": 0}
        ]
        evaluator = MultipleChoiceEvaluator("mmlu")
        samples = evaluator.load_samples(limit=1)
        assert len(samples) == 1
        assert samples[0].correct_answer == "A"

    @patch("src.benchmarks.expanded_eval_suite.load_dataset")
    def test_load_samples_math(self, mock_load):
        mock_load.return_value = [
            {"question": "1+1", "answer": "The answer is #### 2"}
        ]
        evaluator = MathEvaluator("gsm8k")
        samples = evaluator.load_samples(limit=1)
        assert samples[0].correct_answer == "2"

    def test_suite_run_all(self):
        suite = ExpandedEvalSuite(model_fn=lambda x: "A")
        # Mock evaluators to avoid HF calls
        with patch("src.benchmarks.expanded_eval_suite.get_evaluator") as mock_get:
            mock_eval = MagicMock()
            mock_eval.run_evaluation.return_value = MagicMock(benchmark="mmlu", score=100.0, metric="acc", num_samples=1, correct=1)
            mock_get.return_value = mock_eval
            
            summary = suite.run_all(benchmarks=["mmlu"], limit=1)
            assert summary["total_benchmarks"] == 1
            assert summary["aggregate"]["knowledge"] == 100.0

    def test_main_list(self):
        with patch("sys.argv", ["suite.py", "--list"]):
            from src.benchmarks.expanded_eval_suite import main
            main()

    def test_main_execution(self, tmp_path):
        out = tmp_path / "results.json"
        with patch("sys.argv", ["suite.py", "--benchmarks", "mmlu", "--limit", "1", "--output", str(out)]), \
             patch("src.benchmarks.expanded_eval_suite.ExpandedEvalSuite.run_all", return_value={"aggregate": {}, "benchmarks": {}}), \
             patch("src.benchmarks.expanded_eval_suite.ExpandedEvalSuite.save_results"):
            from src.benchmarks.expanded_eval_suite import main
            main()
