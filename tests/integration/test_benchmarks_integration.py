import pytest
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# Locate the benchmark runner script
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.benchmarks.benchmark_runner import BenchmarkRunner, BenchmarkConfig, main as runner_main

@pytest.mark.benchmark
def test_generation_benchmark(omni_model_path, real_text_model, real_text_tokenizer):
    """Run generation benchmark via pytest wrapper."""
    # We use a mocked config or minimal run for the test suite integration
    # unless --full-benchmarks is passed, which invokes the real heavy runner.
    
    config = BenchmarkConfig(
        model_path=str(omni_model_path), 
        output_dir="/tmp/benchmark_test",
        benchmark_runs=1,
        max_new_tokens=10
    )
    
    try:
        with patch("src.omni.loader.OmniModelLoader.load", return_value=(real_text_model, real_text_tokenizer)):
            runner = BenchmarkRunner(config)
            # We assume prompts loading works (fallback exists)
            prompt = "Hello world"
            result = runner.benchmark_generation(prompt, "test_gen")
            assert result.success, f"Generation benchmark failed: {result.error}"
            assert result.tokens_per_second >= 0
    except Exception as e:
        pytest.fail(f"Benchmark Runner failed: {e}")

@pytest.mark.benchmark
def test_perplexity_benchmark(omni_model_path, real_text_model, real_text_tokenizer):
    """Run perplexity benchmark via pytest wrapper."""
    config = BenchmarkConfig(
        model_path=str(omni_model_path),
        benchmark_runs=1
    )
    
    try:
        with patch("src.omni.loader.OmniModelLoader.load", return_value=(real_text_model, real_text_tokenizer)):
            runner = BenchmarkRunner(config)
            result = runner.benchmark_perplexity("Test text", "test_ppl")
            assert result.success, f"Perplexity benchmark failed: {result.error}"
    except Exception as e:
        pytest.fail(f"Benchmark Runner failed: {e}")
