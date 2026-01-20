"""
conftest.py
Shared pytest fixtures with automatic metrics collection.

Metrics tracked:
- Test name, duration, status, memory usage
- Automatic CSV export after test run
- Progress bars for test execution
"""

import os
import sys
import time
import csv
import gc
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import pytest

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ============== METRICS COLLECTION ==============

class TestMetricsCollector:
    """Collect metrics from all tests and export to CSV."""
    
    def __init__(self):
        self.results = []
        self.start_time = None
        self.output_dir = PROJECT_ROOT / "results"
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def start_session(self):
        self.start_time = datetime.now()
        self.results = []
    
    def record_test(self, nodeid: str, outcome: str, duration: float, 
                    memory_mb: float = 0, error: str = ""):
        self.results.append({
            "timestamp": datetime.now().isoformat(),
            "test_id": nodeid,
            "test_name": nodeid.split("::")[-1] if "::" in nodeid else nodeid,
            "test_file": nodeid.split("::")[0] if "::" in nodeid else "",
            "test_class": nodeid.split("::")[1] if nodeid.count("::") > 1 else "",
            "outcome": outcome,
            "duration_seconds": round(duration, 4),
            "memory_mb": round(memory_mb, 2),
            "error": error[:200] if error else "",
        })
    
    def export_csv(self):
        if not self.results:
            return
        
        timestamp = self.start_time.strftime("%Y%m%d_%H%M%S") if self.start_time else "unknown"
        csv_path = self.output_dir / f"test_results_{timestamp}.csv"
        
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.results[0].keys())
            writer.writeheader()
            writer.writerows(self.results)
        
        return csv_path
    
    def get_summary(self) -> Dict[str, Any]:
        if not self.results:
            return {}
        
        total = len(self.results)
        passed = sum(1 for r in self.results if r["outcome"] == "passed")
        failed = sum(1 for r in self.results if r["outcome"] == "failed")
        skipped = sum(1 for r in self.results if r["outcome"] == "skipped")
        total_duration = sum(r["duration_seconds"] for r in self.results)
        
        return {
            "total": total,
            "passed": passed,
            "failed": failed,
            "skipped": skipped,
            "pass_rate": f"{100*passed/total:.1f}%" if total > 0 else "0%",
            "total_duration_seconds": round(total_duration, 2),
            "avg_duration_seconds": round(total_duration/total, 4) if total > 0 else 0,
            "slowest_tests": sorted(self.results, key=lambda x: x["duration_seconds"], reverse=True)[:5],
        }


# Global collector instance
_metrics_collector = TestMetricsCollector()


# ============== PYTEST HOOKS ==============

def pytest_configure(config):
    """Initialize metrics collection at start of test session."""
    _metrics_collector.start_session()


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Collect metrics for each test."""
    outcome = yield
    report = outcome.get_result()
    
    if report.when == "call":
        # Get memory usage
        memory_mb = 0
        try:
            import torch
            if torch.cuda.is_available():
                memory_mb = torch.cuda.memory_allocated() / (1024**2)
        except Exception:
            pass
        
        error = ""
        if report.failed and hasattr(report, "longreprtext"):
            error = str(report.longreprtext)
        
        _metrics_collector.record_test(
            nodeid=item.nodeid,
            outcome=report.outcome,
            duration=report.duration,
            memory_mb=memory_mb,
            error=error,
        )


def pytest_sessionfinish(session, exitstatus):
    """Export metrics and print summary at end of test session."""
    csv_path = _metrics_collector.export_csv()
    summary = _metrics_collector.get_summary()
    
    if summary:
        print("\n" + "="*70)
        print("TEST METRICS SUMMARY")
        print("="*70)
        print(f"Total: {summary['total']} | Passed: {summary['passed']} | "
              f"Failed: {summary['failed']} | Skipped: {summary['skipped']}")
        print(f"Pass Rate: {summary['pass_rate']}")
        print(f"Total Duration: {summary['total_duration_seconds']:.2f}s")
        print(f"Avg Duration: {summary['avg_duration_seconds']:.4f}s per test")
        
        if summary.get('slowest_tests'):
            print("\nSlowest Tests:")
            for t in summary['slowest_tests'][:5]:
                print(f"  {t['duration_seconds']:.3f}s - {t['test_name']}")
        
        if csv_path:
            print(f"\nMetrics exported to: {csv_path}")
        print("="*70)


# ============== CONSTANTS ==============

# Model paths
TEXT_MODEL_PATH = "/mnt/e/data/models/Qwen2.5-0.5B"
OMNI_MODEL_PATH = "/mnt/e/data/models/Qwen2.5-Omni-7B-GPTQ-Int4"

# Encoder paths
VISION_ENCODER_PATH = "/mnt/e/data/encoders/vision-encoders/siglip2-so400m-patch16-512"
AUDIO_ENCODER_PATH = "/mnt/e/data/encoders/audio-encoders/whisper-large-v3-turbo"

# Dataset paths
DATASETS_PATH = "/mnt/e/data/datasets"


# ============== FIXTURES ==============

@pytest.fixture(scope="session")
def device():
    """Get available device."""
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="session")
def has_gpu():
    """Check if GPU is available."""
    import torch
    return torch.cuda.is_available()


@pytest.fixture(scope="session")
def text_model_path():
    """Path to text-only test model."""
    return TEXT_MODEL_PATH


@pytest.fixture(scope="session")
def omni_model_path():
    """Path to full Omni model."""
    return OMNI_MODEL_PATH


@pytest.fixture(scope="session")
def vision_encoder_path():
    """Path to vision encoder."""
    return VISION_ENCODER_PATH


@pytest.fixture(scope="session")
def audio_encoder_path():
    """Path to audio encoder."""
    return AUDIO_ENCODER_PATH


@pytest.fixture(scope="session")
def real_text_model(text_model_path, device):
    """Load real Qwen2.5-0.5B model for testing."""
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
    
    # Cleanup
    del model
    gc.collect()
    if device == "cuda":
        import torch
        torch.cuda.empty_cache()


@pytest.fixture(scope="session")
def real_text_tokenizer(text_model_path):
    """Load real tokenizer for testing."""
    from transformers import AutoTokenizer
    
    if not Path(text_model_path).exists():
        pytest.skip(f"Tokenizer not found: {text_model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        text_model_path,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return tokenizer


@pytest.fixture(scope="session")
def real_model_and_tokenizer(real_text_model, real_text_tokenizer):
    """Combined model and tokenizer fixture."""
    return {
        "model": real_text_model,
        "tokenizer": real_text_tokenizer,
    }


@pytest.fixture(scope="session")
def encoders_config():
    """Load encoders.yaml configuration."""
    import yaml
    
    config_path = PROJECT_ROOT / "configs" / "encoders.yaml"
    if not config_path.exists():
        pytest.skip("encoders.yaml not found")
    
    with open(config_path) as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="function")
def sample_text_prompt():
    """Sample prompt for testing."""
    return "What is 2 + 2? Please explain your reasoning step by step."


@pytest.fixture(scope="function")
def sample_messages():
    """Sample chat messages for testing."""
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"},
    ]


@pytest.fixture(scope="function")
def temp_output_dir(tmp_path):
    """Temporary output directory for tests."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    return output_dir


@pytest.fixture(scope="function")
def temp_checkpoint_dir(tmp_path):
    """Temporary checkpoint directory for tests."""
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    return checkpoint_dir


@pytest.fixture(scope="session")
def metrics_tracker():
    """Get the global metrics tracker for tests."""
    from src.metrics_tracker import MetricsTracker
    return MetricsTracker(output_dir=str(PROJECT_ROOT / "results"))
