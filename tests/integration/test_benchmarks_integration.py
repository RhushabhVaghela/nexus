import pytest
import sys
from pathlib import Path
from unittest.mock import MagicMock

# Locate the benchmark runner script
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.benchmarks.benchmark_runner import BenchmarkRunner, BenchmarkConfig, main as runner_main

@pytest.mark.benchmark
def test_generation_benchmark():
    """Run generation benchmark via pytest wrapper."""
    # We use a mocked config or minimal run for the test suite integration
    # unless --full-benchmarks is passed, which invokes the real heavy runner.
    
    # Check if we should run real benchmarks
    # Note: Logic inside validation/production would run the script directly.
    # Here we ensure the runner class itself works.
    
    config = BenchmarkConfig(
        model_path="/mnt/e/data/models/Qwen2.5-0.5B", # Default test model
        output_dir="/tmp/benchmark_test",
        benchmark_runs=1,
        max_new_tokens=10
    )
    
    # Check if model exists, else skip
    if not Path(config.model_path).exists():
        pytest.skip(f"Test model not found at {config.model_path}")

    try:
        runner = BenchmarkRunner(config)
        # We assume prompts loading works (fallback exists)
        prompt = "Hello world"
        result = runner.benchmark_generation(prompt, "test_gen")
        assert result.success, "Generation benchmark failed"
        assert result.tokens_per_second >= 0
    except Exception as e:
        pytest.fail(f"Benchmark Runner failed: {e}")

@pytest.mark.benchmark
def test_perplexity_benchmark():
    """Run perplexity benchmark via pytest wrapper."""
    config = BenchmarkConfig(
        model_path="/mnt/e/data/models/Qwen2.5-0.5B",
        benchmark_runs=1
    )
    
    if not Path(config.model_path).exists():
        pytest.skip("Test model not found")

    try:
        runner = BenchmarkRunner(config)
        result = runner.benchmark_perplexity("Test text", "test_ppl")
        assert result.success, "Perplexity benchmark failed"
    except Exception as e:
        pytest.fail(f"Benchmark Runner failed: {e}")
