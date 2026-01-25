"""
Unit tests for baseline and native benchmark scripts.
"""

import pytest
import sys
from unittest.mock import MagicMock, patch

def test_benchmark_baseline_main():
    from src import benchmark_baseline
    with patch("sys.argv", ["baseline.py"]), \
         patch("src.benchmark_baseline.OmniMultimodalLM"), \
         patch("src.benchmark_baseline.get_test_prompts"):
        benchmark_baseline.run_benchmark()

def test_benchmark_native_main():
    from src import benchmark_native
    with patch("sys.argv", ["native.py"]), \
         patch("src.benchmark_native.AutoModelForCausalLM"), \
         patch("src.benchmark_native.AutoProcessor"):
        benchmark_native.run_native_benchmark()
