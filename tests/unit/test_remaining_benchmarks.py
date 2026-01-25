"""
Unit tests for remaining benchmark modules.
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

def test_benchmark_omni_inference_main():
    with patch("src.benchmarks.benchmark_omni_inference.OmniInference"), \
         patch("src.benchmarks.benchmark_omni_inference.MetricsTracker"), \
         patch("src.benchmarks.benchmark_omni_inference.patch") as mock_patch:
        
        # mock_patch(...) is called
        # it returns a context manager
        # when we do "with patch(...) as mock_load:"
        # mock_load is the result of __enter__
        mock_load = MagicMock()
        mock_ctx = MagicMock()
        mock_ctx.__enter__.return_value = mock_load
        mock_patch.return_value = mock_ctx
        
        from src.benchmarks.benchmark_omni_inference import benchmark_omni_inference
        benchmark_omni_inference()

def test_benchmark_data_processing_main():
    from src.benchmarks.benchmark_data_processing import main
    with patch("sys.argv", ["bench.py"]):
        # benchmark_normalization is called
        main()

def test_benchmark_repetition_main():
    from src.benchmarks.benchmark_repetition import main
    # We need to mock things inside benchmark_repetition
    with patch("sys.argv", ["bench.py", "--model-path", "/fake", "--iterations", "1"]), \
         patch("src.benchmarks.benchmark_repetition.OmniMultimodalLM"), \
         patch("src.benchmarks.benchmark_repetition.PromptRepetitionEngine"):
        main()
