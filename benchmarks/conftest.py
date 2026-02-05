"""
Pytest Configuration for Benchmarks
Shared fixtures and configuration for the benchmark suite.
"""

import pytest
import torch
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "benchmark: mark test as a performance benchmark"
    )
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line("markers", "gpu: mark test as requiring GPU")


def pytest_collection_modifyitems(config, items):
    """Modify test collection based on markers."""
    for item in items:
        # Add benchmark marker if not present
        if "benchmark" not in item.keywords:
            item.add_marker(pytest.mark.benchmark)

        # Add GPU marker for GPU-specific tests
        if "cuda" in str(item.fspath) or "gpu" in str(item.fspath).lower():
            item.add_marker(pytest.mark.gpu)


@pytest.fixture(scope="session")
def project_root():
    """Return project root path."""
    return Path(__file__).parent.parent.parent


@pytest.fixture(scope="session")
def device():
    """Return appropriate device for testing."""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch, "mps") and torch.mps.is_available():
        return "mps"
    else:
        return "cpu"


@pytest.fixture
def cleanup_cuda():
    """Clean up CUDA memory after test."""
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@pytest.fixture
def sample_prompts():
    """Return sample prompts for testing."""
    return [
        "What is the capital of France?",
        "Explain quantum computing in simple terms.",
        "Write a Python function to calculate factorial.",
        "Describe the history of artificial intelligence.",
        "What are the benefits of exercise?",
    ]


@pytest.fixture
def short_prompt():
    """Return a short prompt for quick tests."""
    return "Hello, how are you?"


@pytest.fixture
def medium_prompt():
    """Return a medium-length prompt."""
    return """
    Write a detailed analysis of the following topic:
    The impact of artificial intelligence on modern healthcare systems,
    including diagnosis, treatment planning, and patient care.
    """


@pytest.fixture
def long_prompt():
    """Return a long prompt for extensive testing."""
    return """
    Write a comprehensive essay covering the following aspects:
    1. Introduction to machine learning and its applications
    2. Deep learning architectures including CNNs, RNNs, and Transformers
    3. Natural language processing and its evolution
    4. Computer vision and image recognition
    5. Reinforcement learning and decision making
    6. Ethical considerations in AI development
    7. Future prospects and challenges
    """


@pytest.fixture
def sample_batch():
    """Create sample training batch."""
    return {
        "input_ids": torch.randint(0, 1000, (4, 512)),
        "attention_mask": torch.ones(4, 512),
        "teacher_logits": torch.randn(4, 512, 32000),
        "teacher_hidden_states": torch.randn(4, 512, 4096),
    }


@pytest.fixture
def inference_config():
    """Return sample generation configuration."""
    from src.models.omni.inference import GenerationConfig

    return GenerationConfig(
        max_new_tokens=100,
        temperature=0.7,
        top_p=0.9,
        top_k=50,
        do_sample=True,
        repetition_penalty=1.1,
    )


def pytest_benchmark_compare(config, benchmarks, column, row):
    """Custom comparison function for benchmark results."""
    if column == "mean":
        # Compare mean execution time
        return benchmarks[0][column] <= benchmarks[1][column] * 1.1


def pytest_benchmark_update_stats(config, benchmarks, machine_info, **kwargs):
    """Update benchmark statistics."""
    for benchmark in benchmarks:
        # Add machine info to stats
        benchmark["machine_info"] = machine_info


# Skipping logic for optional dependencies
def pytest_runtest_setup(item):
    """Set up test, skip if dependencies not available."""
    # Check for optional dependencies
    optional_deps = {
        "tensorrt": "tensorrt",
        "deepspeed": "deepspeed",
        "accelerate": "accelerate",
    }

    for marker_name, module_name in optional_deps.items():
        if item.get_closest_marker(marker_name):
            try:
                __import__(module_name)
            except ImportError:
                pytest.skip(f"{module_name} not installed")


# Performance assertions helpers
class PerformanceAssertions:
    """Helper class for performance assertions."""

    @staticmethod
    def assert_throughput(tokens_per_second, minimum=1.0):
        """Assert minimum token throughput."""
        assert tokens_per_second >= minimum, (
            f"Throughput {tokens_per_second:.2f} below minimum {minimum}"
        )

    @staticmethod
    def assert_latency_ms(latency_ms, maximum=1000.0):
        """Assert maximum latency."""
        assert latency_ms <= maximum, (
            f"Latency {latency_ms:.2f}ms exceeds maximum {maximum}ms"
        )

    @staticmethod
    def assert_memory_mb(memory_mb, maximum=16384.0):
        """Assert maximum memory usage."""
        assert memory_mb <= maximum, (
            f"Memory {memory_mb:.2f}MB exceeds maximum {maximum}MB"
        )

    @staticmethod
    def assert_speedup(baseline, optimized, minimum=1.0):
        """Assert minimum speedup."""
        speedup = baseline / optimized if optimized > 0 else 0
        assert speedup >= minimum, f"Speedup {speedup:.2f}x below minimum {minimum}x"

    @staticmethod
    def assert_savings_percent(baseline, optimized, minimum=0.0):
        """Assert minimum savings percentage."""
        savings = ((baseline - optimized) / baseline) * 100 if baseline > 0 else 0
        assert savings >= minimum, f"Savings {savings:.1f}% below minimum {minimum}%"


# Make assertions available in tests
@pytest.fixture
def assert_perf():
    """Return performance assertion helper."""
    return PerformanceAssertions


# Benchmark result helpers
class BenchmarkResultHelpers:
    """Helper methods for benchmark results."""

    @staticmethod
    def calculate_speedup(baseline_time, optimized_time):
        """Calculate speedup ratio."""
        return baseline_time / optimized_time if optimized_time > 0 else 0

    @staticmethod
    def calculate_savings_percent(baseline, optimized):
        """Calculate memory savings percentage."""
        return ((baseline - optimized) / baseline) * 100 if baseline > 0 else 0

    @staticmethod
    def format_throughput(tokens_per_second):
        """Format throughput for display."""
        if tokens_per_second > 1000:
            return f"{tokens_per_second / 1000:.2f}K tokens/sec"
        elif tokens_per_second > 1:
            return f"{tokens_per_second:.2f} tokens/sec"
        else:
            return f"{tokens_per_second * 60:.2f} tokens/min"

    @staticmethod
    def format_memory(memory_mb):
        """Format memory for display."""
        if memory_mb > 1024:
            return f"{memory_mb / 1024:.2f}GB"
        else:
            return f"{memory_mb:.2f}MB"

    @staticmethod
    def format_latency(latency_ms):
        """Format latency for display."""
        if latency_ms < 1:
            return f"{latency_ms * 1000:.2f}μs"
        elif latency_ms < 1000:
            return f"{latency_ms:.2f}ms"
        else:
            return f"{latency_ms / 1000:.2f}s"


@pytest.fixture
def benchmark_helpers():
    """Return benchmark result helper methods."""
    return BenchmarkResultHelpers
