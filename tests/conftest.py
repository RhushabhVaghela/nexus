"""
conftest.py
Shared pytest fixtures with comprehensive metrics collection.

Metrics tracked:
- Test identification (id, name, file, class, module)
- Timing (seconds, milliseconds)
- GPU memory (bytes, kb, mb, peak)
- RAM memory (bytes, kb, mb, percent)
- System info (hostname, gpu_name, cuda_version, python_version)
- Environment (base_model, device, torch_version)
"""

import os
import sys
import torch
import importlib.util

# Patch torch.__spec__ if missing to fix ValueError in transformers/datasets
# Patch torch.__spec__ if missing to fix ValueError in transformers/datasets
try:
    if torch.__spec__ is None:
        from importlib.machinery import ModuleSpec
        # Create a dummy spec if file not found
        torch.__spec__ = ModuleSpec(name="torch", loader=None, origin=getattr(torch, '__file__', None))
except Exception:
    pass

import time
import csv
import gc
import platform
import socket
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import pytest

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def get_system_info() -> Dict[str, str]:
    """Get system information."""
    info = {
        "hostname": socket.gethostname(),
        "platform": platform.system(),
        "python_version": platform.python_version(),
        "torch_version": "",
        "cuda_version": "",
        "gpu_name": "",
        "gpu_count": 0,
        "cpu_count": os.cpu_count() or 0,
    }
    
    try:
        import torch
        info["torch_version"] = torch.__version__
        if torch.cuda.is_available():
            info["cuda_version"] = torch.version.cuda or ""
            info["gpu_count"] = torch.cuda.device_count()
            info["gpu_name"] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    
    return info


# Import unified tracker
from src.metrics_tracker import MetricsTracker, TestDetailMetrics, ValidationMetrics

class TestMetricsCollector:
    """Collect comprehensive metrics from all tests and export to CSV using MetricsTracker."""
    
    def __init__(self):
        self.tracker = MetricsTracker()
        self.results = []
        self.start_time = None
        self.system_info = get_system_info()
    
    def start_session(self):
        self.start_time = datetime.now()
        self.results = []
        self.system_info = get_system_info()
    
    def record_test(self, nodeid: str, outcome: str, duration: float, 
                    memory_stats: Dict[str, float] = None, error: str = ""):
        # Parse output for logging
        parts = nodeid.split("::")
        test_file = parts[0] if len(parts) > 0 else ""
        test_name = parts[-1] if parts else nodeid
        
        # Create metrics object
        metrics = TestDetailMetrics(
            test_id=nodeid,
            name=test_name,
            outcome=outcome,
            duration_s=round(duration, 6),
            duration_ms=round(duration * 1000, 3),
            gpu_memory_mb=round(memory_stats.get("gpu_memory_mb", 0) if memory_stats else 0, 2),
            ram_mb=round(memory_stats.get("ram_mb", 0) if memory_stats else 0, 2),
            error=error[:500] if error else "",
            file=test_file
        )
        
        # Log to global CSV
        self.tracker.log_test_detail(metrics)
        self.results.append(metrics)
    
    def get_summary(self) -> Dict[str, Any]:
        if not self.results:
            return {}
        
        total = len(self.results)
        passed = sum(1 for r in self.results if r.outcome == "passed")
        failed = sum(1 for r in self.results if r.outcome == "failed")
        skipped = sum(1 for r in self.results if r.outcome == "skipped")
        total_duration = sum(r.duration_s for r in self.results)
        
        # Log session summary to validation_metrics.csv
        val_metrics = ValidationMetrics(
            test_type="pytest_session",
            total_tests=total,
            passed=passed,
            failed=failed,
            skipped=skipped,
            duration_seconds=total_duration,
            details=f"Session ending {datetime.now().strftime('%H:%M:%S')}",
            errors=f"{failed} failures" if failed > 0 else ""
        )
        self.tracker.log_validation(val_metrics)
        
        return {
            "total": total,
            "passed": passed,
            "failed": failed,
            "pass_rate": f"{100*passed/total:.1f}%" if total > 0 else "0%",
            "duration": total_duration,
            "gpu_name": self.system_info.get("gpu_name", ""),
        }
    
    def _get_memory_stats(self) -> Dict[str, float]:
        """Get current memory stats (Legacy helper, kept for hook compatibility)."""
        stats = {}
        try:
            import torch
            if torch.cuda.is_available():
                stats["gpu_memory_mb"] = torch.cuda.memory_allocated() / (1024**2)
        except: pass
        try:
            import psutil
            stats["ram_mb"] = psutil.Process().memory_info().rss / (1024**2)
        except: pass
        return stats

# Global collector instance
_metrics_collector = TestMetricsCollector()


# ============== CONSTANTS ==============

TEXT_MODEL_PATH = Path("/mnt/e/data/models/Qwen2.5-0.5B")
OMNI_MODEL_PATH = Path("/mnt/e/data/models/Qwen2.5-Omni-7B-GPTQ-Int4")
VISION_ENCODER_PATH = Path("/mnt/e/data/encoders/vision-encoders/siglip2-so400m-patch16-512")
AUDIO_ENCODER_PATH = Path("/mnt/e/data/encoders/audio-encoders/whisper-large-v3-turbo")
DATASETS_PATH = Path("/mnt/e/data/datasets")


# ============== PYTEST HOOKS ==============

def pytest_addoption(parser):
    """Add custom CLI options for Nexus Model test suite."""
    parser.addoption(
        "--full-tests", "-F", 
        action="store_true", 
        default=False,
        help="Run entire test suite including slow integration/e2e tests. "
             "Uses Qwen2.5-0.5B model at /mnt/e/data/models/Qwen2.5-0.5B. "
             "Duration: ~90s. Default: False (slow tests skipped)"
    )
    parser.addoption(
        "--full-benchmarks", "-G", 
        action="store_true",
        default=False,
        help="Run ALL benchmarks (Global). Measures tokens/sec, latency, perplexity. "
             "Uses real dataset samples from ALL_DATASETS. "
             "Results exported to results/benchmark_metrics.csv. Default: False"
    )
    parser.addoption(
        "--test", "-T", 
        action="store",
        default=None,
        help="Filter for specific tests by name pattern. Supports comma-separated values. "
             "Example: --test 'model_load,tokenizer' or -T cot. Default: None (all tests)"
    )
    parser.addoption(
        "--benchmark", "-B", 
        action="store",
        default=None,
        help="Filter for specific benchmarks by name pattern. Comma-separated. "
             "Only selects @pytest.mark.benchmark tests. Example: -B generation. Default: None"
    )
    parser.addoption(
        "--no-skip", "-N",
        action="store_true",
        default=False,
        help="Force run ALL tests and benchmarks, including those that would normally be "
             "skipped due to missing models/files. Implies --full-tests and --full-benchmarks. "
             "Use for 100% coverage testing with real models. Default: False"
    )


def pytest_configure(config):
    """Initialize metrics collection and register markers."""
    _metrics_collector.start_session()
    
    # Register markers
    config.addinivalue_line("markers", "benchmark: mark test as a benchmark")
    config.addinivalue_line("markers", "slow: mark test as slow (skipped unless --full-tests)")


def pytest_collection_modifyitems(config, items):
    """Handle custom options for filtering."""
    
    # 1. Handle --test <names> (Alias for -k, supports comma-sep)
    specific_tests = config.getoption("--test")
    if specific_tests:
        selected = []
        target_names = [t.strip() for t in specific_tests.split(",") if t.strip()]
        for item in items:
            # Check if ANY of the target names appear in the nodeid
            if any(target in item.nodeid for target in target_names):
                selected.append(item)
        items[:] = selected
        return

    # 2. Handle --benchmark <names>
    specific_benchmarks = config.getoption("--benchmark")
    if specific_benchmarks:
        selected = []
        target_names = [b.strip() for b in specific_benchmarks.split(",") if b.strip()]
        for item in items:
            if "benchmark" in item.keywords:
                if any(target in item.nodeid for target in target_names):
                    selected.append(item)
        items[:] = selected
        return

    # 3. Handle --no-skip (force run everything, no skips)
    no_skip = config.getoption("--no-skip")
    if no_skip:
        # --no-skip implies --full-tests and --full-benchmarks
        # Don't add any skip markers - run everything
        return
    
    # 4. Handle --full-benchmarks
    # If set, we run EVERYTHING including benchmarks.
    # If NOT set, we skip benchmarks unless explicitly requested via --benchmark
    run_benchmarks = config.getoption("--full-benchmarks")
    if not run_benchmarks and not specific_benchmarks:
        skip_benchmark = pytest.mark.skip(reason="Use --full-benchmarks or --no-skip to run")
        for item in items:
            if "benchmark" in item.keywords:
                item.add_marker(skip_benchmark)
    
    # 5. Handle --full-tests (vs slow tests)
    run_full = config.getoption("--full-tests")
    if not run_full:
        skip_slow = pytest.mark.skip(reason="Use --full-tests or --no-skip to run slow tests")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Collect metrics for each test."""
    outcome = yield
    report = outcome.get_result()
    
    if report.when == "call":
        memory_stats = _metrics_collector._get_memory_stats()
        
        error = ""
        if report.failed and hasattr(report, "longreprtext"):
            error = str(report.longreprtext)
        
        _metrics_collector.record_test(
            nodeid=item.nodeid,
            outcome=report.outcome,
            duration=report.duration,
            memory_stats=memory_stats,
            error=error,
        )


def pytest_sessionfinish(session, exitstatus):
    """Export metrics and print summary at end of test session."""
    # CSV export is handled per-test now
    summary = _metrics_collector.get_summary()
    
    if summary:
        print("\n" + "="*70)
        print("TEST METRICS SUMMARY")
        print("="*70)
        print(f"Total: {summary['total']} | Passed: {summary['passed']} | Failed: {summary['failed']}")
        print(f"Pass Rate: {summary['pass_rate']}")
        print(f"Duration: {summary['duration']:.2f}s")
        if summary.get('gpu_name'):
            print(f"GPU: {summary['gpu_name']}")
        print(f"\nDetailed Results: results/test_details.csv")
        print("="*70)


# ============== FIXTURES ==============

@pytest.fixture(scope="session")
def no_skip(request):
    """Check if --no-skip flag is set. Tests can use this to avoid skipping."""
    return request.config.getoption("--no-skip")

@pytest.fixture(scope="session")
def device():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"

@pytest.fixture(scope="session")
def has_gpu():
    import torch
    return torch.cuda.is_available()

@pytest.fixture(scope="session")
def text_model_path():
    return TEXT_MODEL_PATH

@pytest.fixture(scope="session")
def omni_model_path():
    return OMNI_MODEL_PATH

@pytest.fixture(scope="session")
def real_text_model(text_model_path, device):
    import torch
    from transformers import AutoModelForCausalLM
    if not Path(text_model_path).exists():
        pytest.skip(f"Model not found: {text_model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        text_model_path,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
    )
    model.eval()
    yield model
    del model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

@pytest.fixture(scope="session")
def real_text_tokenizer(text_model_path):
    from transformers import AutoTokenizer
    if not Path(text_model_path).exists():
        pytest.skip(f"Tokenizer not found: {text_model_path}")
    tokenizer = AutoTokenizer.from_pretrained(text_model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

@pytest.fixture(scope="function")
def sample_text_prompt():
    return "What is 2 + 2?"

@pytest.fixture(scope="function")
def temp_output_dir(tmp_path):
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    return output_dir

# ============== CONFIG FIXTURES ==============

@pytest.fixture(scope="session")
def encoders_config():
    """Load encoders.yaml configuration."""
    import yaml
    config_path = PROJECT_ROOT / "configs" / "encoders.yaml"
    if not config_path.exists():
        pytest.skip("encoders.yaml not found")
    with open(config_path) as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="session")
def decoders_config():
    """Load decoders.yaml configuration."""
    import yaml
    config_path = PROJECT_ROOT / "configs" / "decoders.yaml"
    if not config_path.exists():
        pytest.skip("decoders.yaml not found")
    with open(config_path) as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="session")
def datasets_config():
    """Load datasets.yaml configuration."""
    import yaml
    config_path = PROJECT_ROOT / "configs" / "datasets.yaml"
    if not config_path.exists():
        pytest.skip("datasets.yaml not found")
    with open(config_path) as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="session")
def outputs_config():
    """Load outputs.yaml configuration."""
    import yaml
    config_path = PROJECT_ROOT / "configs" / "outputs.yaml"
    if not config_path.exists():
        pytest.skip("outputs.yaml not found")
    with open(config_path) as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="session")
def vision_encoder_path(encoders_config):
    """Get vision encoder path from config."""
    return encoders_config.get("encoders", {}).get("vision", {}).get("default", "")


@pytest.fixture(scope="session")
def audio_encoder_path(encoders_config):
    """Get audio encoder path from config."""
    return encoders_config.get("encoders", {}).get("audio_input", {}).get("default", "")
