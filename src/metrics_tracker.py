#!/usr/bin/env python3
"""
metrics_tracker.py
Unified metrics tracking for training, testing, and validation.

Features:
- Progress bars with ETA (tqdm)
- CSV results export
- GPU/memory monitoring
- Per-dataset and per-capability tracking
"""

import os
import csv
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict, field
import logging

try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm not installed
    def tqdm(iterable, **kwargs):
        desc = kwargs.get('desc', '')
        total = kwargs.get('total', len(iterable) if hasattr(iterable, '__len__') else None)
        for i, item in enumerate(iterable):
            if total:
                print(f"\r{desc}: {i+1}/{total}", end='')
            yield item
        print()

logger = logging.getLogger(__name__)


@dataclass
class TrainingMetrics:
    """Metrics for a single training run."""
    timestamp: str = ""
    capability: str = ""
    dataset: str = ""
    base_model: str = ""
    
    # Training metrics
    samples: int = 0
    steps: int = 0
    epochs: int = 0
    batch_size: int = 1
    
    # Performance
    duration_seconds: float = 0.0
    samples_per_second: float = 0.0
    
    # Loss
    initial_loss: float = 0.0
    final_loss: float = 0.0
    avg_loss: float = 0.0
    
    # Memory
    gpu_memory_peak_gb: float = 0.0
    gpu_memory_used_gb: float = 0.0
    
    # Output
    checkpoint_path: str = ""
    output_format: str = "safetensors"
    
    # Status
    success: bool = False
    error: str = ""


@dataclass
class ValidationMetrics:
    """Metrics for validation run."""
    timestamp: str = ""
    test_type: str = ""  # unit, integration, e2e, real_training
    
    # Test counts
    total_tests: int = 0
    passed: int = 0
    failed: int = 0
    skipped: int = 0
    
    # Performance
    duration_seconds: float = 0.0
    
    # Details
    details: str = ""
    errors: str = ""


class MetricsTracker:
    """Track and export metrics to CSV."""
    
    def __init__(self, output_dir: str = "./results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.training_file = self.output_dir / "training_metrics.csv"
        self.validation_file = self.output_dir / "validation_metrics.csv"
        
        self._init_csv_files()
    
    def _init_csv_files(self):
        """Initialize CSV files with headers if they don't exist."""
        if not self.training_file.exists():
            with open(self.training_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(TrainingMetrics.__annotations__.keys()))
                writer.writeheader()
        
        if not self.validation_file.exists():
            with open(self.validation_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(ValidationMetrics.__annotations__.keys()))
                writer.writeheader()
    
    def log_training(self, metrics: TrainingMetrics):
        """Append training metrics to CSV."""
        metrics.timestamp = datetime.now().isoformat()
        
        with open(self.training_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(TrainingMetrics.__annotations__.keys()))
            writer.writerow(asdict(metrics))
        
        logger.info(f"Logged training metrics for {metrics.capability}/{metrics.dataset}")
    
    def log_validation(self, metrics: ValidationMetrics):
        """Append validation metrics to CSV."""
        metrics.timestamp = datetime.now().isoformat()
        
        with open(self.validation_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(ValidationMetrics.__annotations__.keys()))
            writer.writerow(asdict(metrics))
        
        logger.info(f"Logged validation metrics: {metrics.test_type}")
    
    def get_gpu_metrics(self) -> Dict[str, float]:
        """Get current GPU metrics."""
        try:
            import torch
            if torch.cuda.is_available():
                return {
                    "gpu_memory_used_gb": torch.cuda.memory_allocated() / (1024**3),
                    "gpu_memory_peak_gb": torch.cuda.max_memory_allocated() / (1024**3),
                }
        except Exception:
            pass
        return {"gpu_memory_used_gb": 0.0, "gpu_memory_peak_gb": 0.0}


class ProgressTracker:
    """Track progress with ETA for training loops."""
    
    def __init__(self, total: int, desc: str = "Training"):
        self.total = total
        self.desc = desc
        self.start_time = time.time()
        self.current = 0
        self.losses = []
    
    def __enter__(self):
        self.pbar = tqdm(total=self.total, desc=self.desc, 
                         bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        return self
    
    def __exit__(self, *args):
        self.pbar.close()
    
    def update(self, loss: float = None):
        """Update progress and optionally record loss."""
        self.current += 1
        if loss is not None:
            self.losses.append(loss)
        self.pbar.update(1)
        if loss is not None:
            self.pbar.set_postfix({"loss": f"{loss:.4f}"})
    
    def get_eta_seconds(self) -> float:
        """Get estimated time remaining in seconds."""
        if self.current == 0:
            return 0.0
        elapsed = time.time() - self.start_time
        rate = self.current / elapsed
        remaining = self.total - self.current
        return remaining / rate if rate > 0 else 0.0
    
    def get_metrics(self) -> Dict[str, float]:
        """Get summary metrics."""
        elapsed = time.time() - self.start_time
        return {
            "duration_seconds": elapsed,
            "samples_per_second": self.current / elapsed if elapsed > 0 else 0,
            "initial_loss": self.losses[0] if self.losses else 0.0,
            "final_loss": self.losses[-1] if self.losses else 0.0,
            "avg_loss": sum(self.losses) / len(self.losses) if self.losses else 0.0,
        }


def run_with_progress(iterable, desc: str = "Processing", callback=None):
    """Run an iterable with progress bar."""
    items = list(iterable)
    with tqdm(items, desc=desc) as pbar:
        for item in pbar:
            result = callback(item) if callback else item
            yield result


# ============== ALL AVAILABLE DATASETS ==============

ALL_DATASETS = {
    "cot": [
        "/mnt/e/data/datasets/kaist-ai_CoT-Collection/data/CoT_collection_en.json",
        "/mnt/e/data/datasets/O1-OPEN_OpenO1-SFT-Pro",
        "/mnt/e/data/datasets/O1-OPEN_OpenO1-SFT-Ultra",
    ],
    "reasoning": [
        "/mnt/e/data/datasets/openai_gsm8k",
        "/mnt/e/data/datasets/O1-OPEN_OpenO1-SFT-Pro",
    ],
    "thinking": [
        "/mnt/e/data/datasets/O1-OPEN_OpenO1-SFT-Pro",
        "/mnt/e/data/datasets/O1-OPEN_OpenO1-SFT-Ultra",
    ],
    "tools": [
        "/mnt/e/data/datasets/Salesforce_xlam-function-calling-60k/xlam_function_calling_60k.json",
        "/mnt/e/data/datasets/NousResearch_hermes-function-calling-v1",
        "/mnt/e/data/datasets/argilla_apigen-function-calling",
        "/mnt/e/data/datasets/hiyouga_glaive-function-calling-v2-sharegpt",
        "/mnt/e/data/datasets/gorilla-llm_gorilla-openfunctions-v2",
    ],
    "streaming": [],  # Runtime feature
    "podcast": [
        "/mnt/e/data/datasets/spawn99_CornellMovieDialogCorpus",
        "/mnt/e/data/datasets/nlpdata_dialogre",
    ],
    "vision-qa": [
        "/mnt/e/data/datasets/AI4Math_MathVista",
        "/mnt/e/data/datasets/AI4Math_MathVerse",
    ],
    "video-understanding": [
        "/mnt/e/data/datasets/VLM2Vec_MSR-VTT",
        "/mnt/e/data/datasets/qingy2024_VaTeX",
        "/mnt/e/data/datasets/XiangpengYang_VideoCoF-50k",
    ],
    "image-generation": [
        "/mnt/e/data/datasets/LucasFang_Laion-Aesthetics-High-Resolution-GoT",
        "/mnt/e/data/datasets/LucasFang_OmniEdit-GoT",
    ],
    "video-generation": [
        "/mnt/e/data/datasets/fullstack__stargate_s04e01_100topkdiverse_text2vid",
    ],
}


def print_summary_table(metrics_list: List[TrainingMetrics]):
    """Print a summary table of training metrics."""
    print("\n" + "="*100)
    print("TRAINING RESULTS SUMMARY")
    print("="*100)
    print(f"{'Capability':<15} {'Dataset':<30} {'Steps':<8} {'Loss':<10} {'Time':<10} {'Status'}")
    print("-"*100)
    
    for m in metrics_list:
        dataset_name = Path(m.dataset).name[:28] if m.dataset else "N/A"
        status = "✅ PASS" if m.success else "❌ FAIL"
        loss_str = f"{m.final_loss:.4f}" if m.final_loss else "N/A"
        print(f"{m.capability:<15} {dataset_name:<30} {m.steps:<8} {loss_str:<10} {m.duration_seconds:>6.1f}s   {status}")
    
    print("="*100)
    passed = sum(1 for m in metrics_list if m.success)
    print(f"\nTotal: {passed}/{len(metrics_list)} passed")
