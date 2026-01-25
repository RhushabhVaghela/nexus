"""
conftest.py
Shared pytest fixtures with comprehensive metrics collection and MOCKING.
"""

import os
import sys
import torch
import importlib.util
from unittest.mock import MagicMock, patch

# Patch torch.__spec__ if missing to fix ValueError in transformers/datasets
try:
    if torch.__spec__ is None:
        from importlib.machinery import ModuleSpec
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
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def get_system_info() -> Dict[str, str]:
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
    except Exception: pass
    return info

from src.metrics_tracker import MetricsTracker, ExecutionDetailMetrics, ValidationMetrics

class TestMetricsCollector:
    def __init__(self):
        self.tracker = MetricsTracker()
        self.results = []
        self.system_info = get_system_info()
    
    def start_session(self):
        self.results = []
    
    def record_test(self, nodeid: str, outcome: str, duration: float, 
                    memory_stats: Dict[str, float] = None, error: str = ""):
        parts = nodeid.split("::")
        test_file = parts[0] if len(parts) > 0 else ""
        test_name = parts[-1] if parts else nodeid
        metrics = ExecutionDetailMetrics(
            test_id=nodeid, name=test_name, outcome=outcome,
            duration_s=round(duration, 6), duration_ms=round(duration * 1000, 3),
            gpu_memory_mb=round(memory_stats.get("gpu_memory_mb", 0) if memory_stats else 0, 2),
            ram_mb=round(memory_stats.get("ram_mb", 0) if memory_stats else 0, 2),
            error=error[:500] if error else "", file=test_file
        )
        self.tracker.log_test_detail(metrics)
        self.results.append(metrics)
    
    def get_summary(self) -> Dict[str, Any]:
        if not self.results: return {}
        total = len(self.results); passed = sum(1 for r in self.results if r.outcome == "passed")
        failed = sum(1 for r in self.results if r.outcome == "failed"); skipped = sum(1 for r in self.results if r.outcome == "skipped")
        total_duration = sum(r.duration_s for r in self.results)
        val_metrics = ValidationMetrics(
            test_type="pytest_session", total_tests=total, passed=passed, failed=failed, skipped=skipped,
            duration_seconds=total_duration, details=f"Session ending {datetime.now().strftime('%H:%M:%S')}",
            errors=f"{failed} failures" if failed > 0 else ""
        )
        self.tracker.log_validation(val_metrics)
        return {"total": total, "passed": passed, "failed": failed, "pass_rate": f"{100*passed/total:.1f}%", "duration": total_duration, "gpu_name": self.system_info.get("gpu_name", "")}
    
    def _get_memory_stats(self) -> Dict[str, float]:
        stats = {}
        try:
            import torch
            if torch.cuda.is_available(): stats["gpu_memory_mb"] = torch.cuda.memory_allocated() / (1024**2)
        except: pass
        try:
            import psutil
            stats["ram_mb"] = psutil.Process().memory_info().rss / (1024**2)
        except: pass
        return stats

_metrics_collector = TestMetricsCollector()

# ============== CONSTANTS ==============
TEXT_MODEL_PATH = Path("/mnt/e/data/models/Qwen2.5-0.5B")
OMNI_MODEL_PATH = Path("/mnt/e/data/models/Qwen2.5-Omni-7B-GPTQ-Int4")

@pytest.fixture(scope="session")
def fake_omni_model_path(tmp_path_factory):
    """Create a fake omni model directory with config.json."""
    model_dir = tmp_path_factory.mktemp("fake_omni")
    config = {
        "model_type": "qwen2_5_omni",
        "architectures": ["Qwen2_5OmniForConditionalGeneration"],
        "vocab_size": 151936,
        "hidden_size": 4096,
        "vision_config": {"hidden_size": 1024},
        "audio_config": {"hidden_size": 1024},
        "token2wav_config": {},
        "video_config": {},
    }
    import json
    with open(model_dir / "config.json", "w") as f:
        json.dump(config, f)
    return str(model_dir)

@pytest.fixture(scope="session")
def omni_model_path(request, fake_omni_model_path, fake_model_path):
    if request.config.getoption("--small-model"):
        # Small model is text-only, so use the text-only fake
        if not request.config.getoption("--use-real-models"):
            return Path(fake_model_path)
        return Path("/mnt/e/data/models/Qwen2.5-0.5B")

    if not request.config.getoption("--use-real-models"):
        return Path(fake_omni_model_path)
    return OMNI_MODEL_PATH

def pytest_addoption(parser):
    parser.addoption("--small-model", "-S", action="store_true", default=False)
    parser.addoption("--full-tests", "-F", action="store_true", default=False)
    parser.addoption("--full-benchmarks", "-G", action="store_true", default=False)
    parser.addoption("--test", "-T", action="store", default=None)
    parser.addoption("--benchmark", "-B", action="store", default=None)
    parser.addoption("--no-skip", "-N", action="store_true", default=False)
    parser.addoption("--use-real-models", action="store_true", default=False)

def pytest_configure(config):
    _metrics_collector.start_session()
    config.addinivalue_line("markers", "benchmark: mark test as a benchmark")
    config.addinivalue_line("markers", "slow: mark test as slow")

def pytest_collection_modifyitems(config, items):
    specific_tests = config.getoption("--test")
    if specific_tests:
        items[:] = [item for item in items if any(t.strip() in item.nodeid for t in specific_tests.split(","))]
    specific_benchmarks = config.getoption("--benchmark")
    if specific_benchmarks:
        items[:] = [item for item in items if "benchmark" in item.keywords and any(b.strip() in item.nodeid for b in specific_benchmarks.split(","))]
    if config.getoption("--no-skip"): return
    if not config.getoption("--full-benchmarks") and not specific_benchmarks:
        for item in items:
            if "benchmark" in item.keywords: item.add_marker(pytest.mark.skip(reason="Use --full-benchmarks"))
    if not config.getoption("--full-tests"):
        for item in items:
            if "slow" in item.keywords: item.add_marker(pytest.mark.skip(reason="Use --full-tests"))

@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if report.when == "call":
        _metrics_collector.record_test(item.nodeid, report.outcome, report.duration, _metrics_collector._get_memory_stats(), str(report.longreprtext) if report.failed else "")

def pytest_sessionfinish(session, exitstatus):
    summary = _metrics_collector.get_summary()
    if summary:
        print("\n" + "="*70 + "\nTEST METRICS SUMMARY\n" + "="*70)
        print(f"Total: {summary['total']} | Passed: {summary['passed']} | Failed: {summary['failed']}\nPass Rate: {summary['pass_rate']}\nDuration: {summary['duration']:.2f}s")
        if summary.get('gpu_name'): print(f"GPU: {summary['gpu_name']}")
        print(f"\nDetailed Results: results/test_details.csv\n" + "="*70)

# ============== FIXTURES ==============
@pytest.fixture(scope="session")
def device(request): 
    if not request.config.getoption("--use-real-models"):
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"

@pytest.fixture(scope="session")
def fake_model_path(tmp_path_factory):
    """Create a fake model directory with config.json for testing."""
    model_dir = tmp_path_factory.mktemp("fake_model")
    config = {
        "model_type": "qwen2",
        "architectures": ["Qwen2ForCausalLM"],
        "vocab_size": 151936,
        "hidden_size": 4096,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
    }
    import json
    with open(model_dir / "config.json", "w") as f:
        json.dump(config, f)
    return str(model_dir)

@pytest.fixture(scope="session")
def text_model_path(request, fake_model_path):
    if request.config.getoption("--small-model"): 
        return Path("/mnt/e/data/models/Qwen2.5-0.5B")
    # Return fake path by default to avoid accidental usage of real path
    if not request.config.getoption("--use-real-models"):
        return Path(fake_model_path)
    return TEXT_MODEL_PATH

@pytest.fixture(scope="session")
def real_text_model(text_model_path, device, request):
    if not request.config.getoption("--use-real-models"):
        mock_model = MagicMock()
        mock_model.device = torch.device(device)
        mock_model.config = MagicMock(vocab_size=151936, hidden_size=4096, model_type="qwen2")
        mock_model.return_value = MagicMock(loss=torch.tensor(0.5), logits=torch.randn(1, 10, 151936))
        # Return an object with sequences attribute to mimic transformers output
        mock_output = MagicMock()
        mock_output.sequences = torch.randint(0, 1000, (1, 20))
        mock_model.generate.return_value = mock_output
        mock_model.training = False
        mock_model.eval.return_value = mock_model
        mock_model.parameters.return_value = [torch.nn.Parameter(torch.randn(1))]
        yield mock_model
        return

    if not Path(text_model_path).exists(): pytest.skip(f"Model not found: {text_model_path}")
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(str(text_model_path), torch_dtype=torch.float16 if device == "cuda" else torch.float32, device_map="auto" if device == "cuda" else None, trust_remote_code=True)
    model.eval()
    yield model
    del model; gc.collect()
    if device == "cuda": torch.cuda.empty_cache()

@pytest.fixture(scope="session")
def real_text_tokenizer(text_model_path, request):
    if not request.config.getoption("--use-real-models"):
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = "[PAD]"; mock_tokenizer.eos_token = "[EOS]"
        mock_tokenizer.pad_token_id = 0; mock_tokenizer.eos_token_id = 1
        mock_tokenizer.return_value = {"input_ids": torch.zeros((1, 10), dtype=torch.long), "attention_mask": torch.ones((1, 10), dtype=torch.long)}
        mock_tokenizer.encode.return_value = [1, 2, 3]
        mock_tokenizer.decode.return_value = "Mocked Paris"
        return mock_tokenizer

    if not Path(text_model_path).exists(): pytest.skip(f"Tokenizer not found: {text_model_path}")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(text_model_path, trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    return tokenizer
