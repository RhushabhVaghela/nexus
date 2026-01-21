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


class TestMetricsCollector:
    """Collect comprehensive metrics from all tests and export to CSV."""
    
    def __init__(self):
        self.results = []
        self.start_time = None
        self.output_dir = PROJECT_ROOT / "results"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.system_info = get_system_info()
        self._reset_gpu_peak()
    
    def _reset_gpu_peak(self):
        """Reset GPU peak memory tracking."""
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass
    
    def _get_memory_stats(self) -> Dict[str, float]:
        """Get current GPU and RAM memory stats."""
        stats = {
            "gpu_memory_bytes": 0,
            "gpu_memory_kb": 0.0,
            "gpu_memory_mb": 0.0,
            "gpu_peak_bytes": 0,
            "gpu_peak_kb": 0.0,
            "gpu_peak_mb": 0.0,
            "gpu_reserved_mb": 0.0,
            "ram_bytes": 0,
            "ram_kb": 0.0,
            "ram_mb": 0.0,
            "ram_percent": 0.0,
            "ram_available_mb": 0.0,
            "ram_total_mb": 0.0,
        }
        
        # GPU memory
        try:
            import torch
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated()
                peak = torch.cuda.max_memory_allocated()
                reserved = torch.cuda.memory_reserved()
                stats.update({
                    "gpu_memory_bytes": allocated,
                    "gpu_memory_kb": allocated / 1024,
                    "gpu_memory_mb": allocated / (1024**2),
                    "gpu_peak_bytes": peak,
                    "gpu_peak_kb": peak / 1024,
                    "gpu_peak_mb": peak / (1024**2),
                    "gpu_reserved_mb": reserved / (1024**2),
                })
        except Exception:
            pass
        
        # RAM memory
        try:
            import psutil
            process = psutil.Process()
            mem_info = process.memory_info()
            vm = psutil.virtual_memory()
            stats.update({
                "ram_bytes": mem_info.rss,
                "ram_kb": mem_info.rss / 1024,
                "ram_mb": mem_info.rss / (1024**2),
                "ram_percent": process.memory_percent(),
                "ram_available_mb": vm.available / (1024**2),
                "ram_total_mb": vm.total / (1024**2),
            })
        except Exception:
            pass
        
        return stats
    
    def start_session(self):
        self.start_time = datetime.now()
        self.results = []
        self.system_info = get_system_info()
        self._reset_gpu_peak()
    
    def record_test(self, nodeid: str, outcome: str, duration: float, 
                    memory_stats: Dict[str, float] = None, error: str = ""):
        if memory_stats is None:
            memory_stats = self._get_memory_stats()
        
        # Parse node ID components
        parts = nodeid.split("::")
        test_file = parts[0] if len(parts) > 0 else ""
        test_class = parts[1] if len(parts) > 2 else ""
        test_name = parts[-1] if parts else nodeid
        test_module = Path(test_file).stem if test_file else ""
        
        self.results.append({
            # Identification
            "timestamp": datetime.now().isoformat(),
            "test_id": nodeid,
            "test_name": test_name,
            "test_class": test_class,
            "test_module": test_module,
            "test_file": test_file,
            
            # Status
            "outcome": outcome,
            "errors_count": 1 if outcome == "failed" else 0,
            "error": error[:500] if error else "",
            
            # Timing
            "duration_seconds": round(duration, 6),
            "duration_ms": round(duration * 1000, 3),
            "duration_us": round(duration * 1000000, 1),
            
            # GPU Memory
            "gpu_memory_bytes": int(memory_stats.get("gpu_memory_bytes", 0)),
            "gpu_memory_kb": round(memory_stats.get("gpu_memory_kb", 0), 3),
            "gpu_memory_mb": round(memory_stats.get("gpu_memory_mb", 0), 3),
            "gpu_peak_bytes": int(memory_stats.get("gpu_peak_bytes", 0)),
            "gpu_peak_mb": round(memory_stats.get("gpu_peak_mb", 0), 3),
            "gpu_reserved_mb": round(memory_stats.get("gpu_reserved_mb", 0), 3),
            
            # RAM Memory
            "ram_bytes": int(memory_stats.get("ram_bytes", 0)),
            "ram_kb": round(memory_stats.get("ram_kb", 0), 3),
            "ram_mb": round(memory_stats.get("ram_mb", 0), 3),
            "ram_percent": round(memory_stats.get("ram_percent", 0), 2),
            "ram_available_mb": round(memory_stats.get("ram_available_mb", 0), 1),
            "ram_total_mb": round(memory_stats.get("ram_total_mb", 0), 1),
            
            # System Info
            "hostname": self.system_info.get("hostname", ""),
            "platform": self.system_info.get("platform", ""),
            "python_version": self.system_info.get("python_version", ""),
            "torch_version": self.system_info.get("torch_version", ""),
            "cuda_version": self.system_info.get("cuda_version", ""),
            "gpu_name": self.system_info.get("gpu_name", ""),
            "gpu_count": self.system_info.get("gpu_count", 0),
            "cpu_count": self.system_info.get("cpu_count", 0),
            
            # Environment
            "base_model": TEXT_MODEL_PATH,
            "device": "cuda" if self.system_info.get("gpu_count", 0) > 0 else "cpu",
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
        total_duration_ms = sum(r["duration_ms"] for r in self.results)
        peak_gpu = max((r["gpu_peak_mb"] for r in self.results), default=0)
        peak_ram = max((r["ram_mb"] for r in self.results), default=0)
        
        return {
            "total": total,
            "passed": passed,
            "failed": failed,
            "skipped": skipped,
            "pass_rate": f"{100*passed/total:.1f}%" if total > 0 else "0%",
            "total_duration_seconds": round(total_duration, 2),
            "total_duration_ms": round(total_duration_ms, 2),
            "avg_duration_ms": round(total_duration_ms/total, 3) if total > 0 else 0,
            "peak_gpu_mb": round(peak_gpu, 2),
            "peak_ram_mb": round(peak_ram, 2),
            "gpu_name": self.system_info.get("gpu_name", ""),
            "slowest_tests": sorted(self.results, key=lambda x: x["duration_seconds"], reverse=True)[:5],
        }


# Global collector instance
_metrics_collector = TestMetricsCollector()


# ============== CONSTANTS ==============

TEXT_MODEL_PATH = "/mnt/e/data/models/Qwen2.5-0.5B"
OMNI_MODEL_PATH = "/mnt/e/data/models/Qwen2.5-Omni-7B-GPTQ-Int4"
VISION_ENCODER_PATH = "/mnt/e/data/encoders/vision-encoders/siglip2-so400m-patch16-512"
AUDIO_ENCODER_PATH = "/mnt/e/data/encoders/audio-encoders/whisper-large-v3-turbo"
DATASETS_PATH = "/mnt/e/data/datasets"


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
    csv_path = _metrics_collector.export_csv()
    summary = _metrics_collector.get_summary()
    
    if summary:
        print("\n" + "="*70)
        print("TEST METRICS SUMMARY")
        print("="*70)
        print(f"Total: {summary['total']} | Passed: {summary['passed']} | "
              f"Failed: {summary['failed']} | Skipped: {summary['skipped']}")
        print(f"Pass Rate: {summary['pass_rate']}")
        print(f"Duration: {summary['total_duration_seconds']:.2f}s ({summary['avg_duration_ms']:.2f}ms avg)")
        print(f"Peak GPU: {summary['peak_gpu_mb']:.2f} MB | Peak RAM: {summary['peak_ram_mb']:.2f} MB")
        if summary.get('gpu_name'):
            print(f"GPU: {summary['gpu_name']}")
        
        if summary.get('slowest_tests'):
            print("\nSlowest Tests:")
            for t in summary['slowest_tests'][:5]:
                print(f"  {t['duration_ms']:>8.1f}ms - {t['test_name']}")
        
        if csv_path:
            print(f"\nCSV: {csv_path}")
        print("="*70)


# ============== FIXTURES ==============

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
