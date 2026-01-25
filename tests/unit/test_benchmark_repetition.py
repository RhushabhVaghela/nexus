"""
Unit tests for Repetition benchmark.
(MOCKED)
"""

import pytest
import sys
import json
from pathlib import Path
from unittest.mock import MagicMock, patch
from src.benchmarks.benchmark_repetition import RepetitionBenchmark

class TestRepetitionBenchmark:
    @pytest.fixture
    def repetition_benchmark(self):
        with patch("src.benchmarks.benchmark_repetition.OmniMultimodalLM"):
            return RepetitionBenchmark("/fake/path", device="cpu")

    def test_gen_text_task(self, repetition_benchmark):
        q, c, t = repetition_benchmark.gen_text_task(num_names=10)
        assert "What is the" in q
        assert "List of names" in c
        assert t in c

    def test_gen_vision_task(self, repetition_benchmark):
        img, q, t = repetition_benchmark.gen_vision_task()
        assert img is not None
        assert "image" in q

    def test_run_inference_mock(self, repetition_benchmark):
        # When model is None, it uses mock logic
        repetition_benchmark.model = None
        res, lat = repetition_benchmark.run_inference("test", factor=1)
        assert res == "MOCK_RESULT"
        assert lat >= 100 # Simulated 0.1s

    def test_run_inference_real_mocked(self, repetition_benchmark):
        repetition_benchmark.model = MagicMock()
        repetition_benchmark.tokenizer = MagicMock()
        repetition_benchmark.tokenizer.return_value = MagicMock()
        repetition_benchmark.tokenizer.decode.return_value = "Result"
        repetition_benchmark.model.wrapper.llm.generate.return_value = [MagicMock()]
        
        res, lat = repetition_benchmark.run_inference("test", factor=1)
        assert res == "Result"

    def test_run_suite(self, repetition_benchmark):
        repetition_benchmark.model = None # Use mock mode
        repetition_benchmark.run_suite(iterations=1)
        assert Path("repetition_benchmark_results.json").exists()

    def test_main_execution(self):
        from src.benchmarks.benchmark_repetition import main
        with patch("sys.argv", ["bench.py", "--model-path", "/fake", "--iterations", "1"]), \
             patch("src.benchmarks.benchmark_repetition.RepetitionBenchmark.run_suite"):
            main()
