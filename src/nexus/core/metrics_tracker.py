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


@dataclass
class BenchmarkMetrics:
    """Metrics for benchmark run."""
    timestamp: str = ""
    name: str = ""
    category: str = ""  # "generation", "accuracy", "perplexity"
    model_name: str = ""
    
    # Timing
    total_time_s: float = 0.0
    tokens_per_second: float = 0.0
    latency_ms: float = 0.0
    first_token_time_s: float = 0.0
    
    # Counts
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    
    # Accuracy
    token_accuracy: float = 0.0
    perplexity: float = 0.0
    loss: float = 0.0
    
    # Memory
    gpu_peak_mb: float = 0.0
    gpu_reserved_mb: float = 0.0
    ram_mb: float = 0.0
    
    # Status
    success: bool = True
    error: str = ""


@dataclass
class ExecutionDetailMetrics:
    """Metrics for a single test execution."""
    timestamp: str = ""
    test_id: str = ""
    name: str = ""
    outcome: str = ""  # passed, failed, skipped
    
    # Timing
    duration_s: float = 0.0
    duration_ms: float = 0.0
    
    # Memory
    gpu_memory_mb: float = 0.0
    ram_mb: float = 0.0
    
    # Details
    error: str = ""
    file: str = ""


class MetricsTracker:
    """Track and export metrics to CSV."""
    
    def __init__(self, output_dir: str = "./results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.training_file = self.output_dir / "training_metrics.csv"
        self.validation_file = self.output_dir / "validation_metrics.csv"
        self.benchmark_file = self.output_dir / "benchmark_metrics.csv"
        self.test_details_file = self.output_dir / "test_details.csv"
        
        self._init_csv_files()
    
    def _init_csv_files(self):
        """Initialize CSV files with headers if they don't exist."""
        files_map = {
            self.training_file: TrainingMetrics,
            self.validation_file: ValidationMetrics,
            self.benchmark_file: BenchmarkMetrics,
            self.test_details_file: ExecutionDetailMetrics,
        }
        
        for path, dataclass_type in files_map.items():
            if not path.exists():
                with open(path, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=list(dataclass_type.__annotations__.keys()))
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

    def log_benchmark(self, metrics: BenchmarkMetrics):
        """Append benchmark metrics to CSV."""
        metrics.timestamp = datetime.now().isoformat()
        
        with open(self.benchmark_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(BenchmarkMetrics.__annotations__.keys()))
            writer.writerow(asdict(metrics))
        
        logger.info(f"Logged benchmark metrics: {metrics.name}")
    
    def log_test_detail(self, metrics: ExecutionDetailMetrics):
        """Append test detail metrics to CSV."""
        metrics.timestamp = datetime.now().isoformat()
        
        with open(self.test_details_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(ExecutionDetailMetrics.__annotations__.keys()))
            writer.writerow(asdict(metrics))
    
    def get_gpu_metrics(self) -> Dict[str, float]:
        """Get current GPU metrics."""
        try:
            import torch
            if torch.cuda.is_available():
                return {
                    "gpu_memory_used_gb": torch.cuda.memory_allocated() / (1024**3),
                    "gpu_memory_peak_gb": torch.cuda.max_memory_allocated() / (1024**3),
                }
        except Exception as e:
            logger.warning(f"Failed to get GPU metrics: {e}")
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

def discover_datasets(base_path: str = "/mnt/e/data/datasets") -> Dict[str, List[str]]:
    """
    Dynamically discover datasets by scanning a base directory.
    Subdirectories are treated as dataset folders.
    If a subdirectory contains files like 'MathVista', it gets tagged appropriately.
    """
    datasets = {
        "cot": [], "reasoning": [], "thinking": [], "tools": [],
        "vision-qa": [], "video-understanding": [], "image-generation": [],
        "video-generation": [], "podcast": [], "streaming": [],
        "remotion-explainer": []
    }
    
    base = Path(base_path)
    if not base.exists():
        return datasets
        
    # Map keywords to categories
    KEYWORD_MAP = {
        "cot": "cot", "reasoning": "reasoning", "thought": "thinking", "thinking": "thinking",
        "tool": "tools", "xlam": "tools", "hermes": "tools", "apigen": "tools",
        "vision": "vision-qa", "math": "vision-qa", "mm": "vision-qa",
        "video": "video-understanding", "msr-vtt": "video-understanding",
        "image": "image-generation", "generation": "image-generation",
        "vid-gen": "video-generation", "text2vid": "video-generation",
        "podcast": "podcast", "dialog": "podcast",
        "tri-streaming": "tri-streaming", "streaming": "streaming",
        "remotion": "remotion-explainer", "explainer": "remotion-explainer"
    }
    
    try:
        # Optimized recursive search with depth limit
        data_extensions = {".json", ".jsonl", ".parquet", ".arrow"}
        max_depth = 3
        base_path_str = str(base_path)
        base_sep_count = base_path_str.count(os.sep)
        
        for root, dirs, files in os.walk(base_path):
            current_depth = root.count(os.sep) - base_sep_count
            
            # Check for HF dataset
            if "dataset_info.json" in files:
                _categorize(root, datasets, KEYWORD_MAP)
                dirs[:] = [] # Skip subdirs of HF dataset
                continue
                
            # Check for data files
            has_data = any(f.lower().endswith(tuple(data_extensions)) for f in files)
            
            if has_data:
                _categorize(root, datasets, KEYWORD_MAP)
                # Once we identify a folder as a dataset, we stop descending
                dirs[:] = [] 
            elif current_depth >= max_depth:
                # Stop descending if we reached max depth without finding data
                dirs[:] = []
                
    except Exception as e:
        logger.warning(f"Error during optimized dataset discovery: {e}")
        
    # Remove duplicates within categories
    for cat in datasets:
        datasets[cat] = list(set(datasets[cat]))
        
    return datasets

def _categorize(path: str, datasets: Dict[str, List[str]], keyword_map: Dict[str, str]):
    """Helper to categorize a path based on keywords."""
    path_lower = path.lower()
    matched = False
    for kw, cat in keyword_map.items():
        if kw in path_lower:
            datasets[cat].append(path)
            matched = True
            break
    
    if not matched and ("gsm" in path_lower or "math" in path_lower):
        datasets["reasoning"].append(path)


# Default static list for fallback or explicit legacy support
_STATIC_DATASETS = {
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
    "remotion-explainer": [
        "/mnt/e/data/datasets/remotion/remotion_explainer_dataset.jsonl",
    ],
}

# Final ALL_DATASETS combines Static + Dynamically discovered
# Lazy-loaded to avoid overhead on every import
_ALL_DATASETS_CACHE = None

def get_all_datasets() -> Dict[str, List[str]]:
    """Combine static dataset list with dynamic discovery."""
    global _ALL_DATASETS_CACHE
    if _ALL_DATASETS_CACHE is not None:
        return _ALL_DATASETS_CACHE
        
    dynamic = discover_datasets()
    combined = _STATIC_DATASETS.copy()
    
    for cat, paths in dynamic.items():
        if cat not in combined:
            combined[cat] = []
        # Add paths while avoiding duplicates
        existing = set(combined[cat])
        for p in paths:
            if p not in existing:
                combined[cat].append(p)
                existing.add(p)
                
    _ALL_DATASETS_CACHE = combined
    return combined

# We keep this for backward compatibility but it will be empty or minimal until called
# Actually, better to just export the function and let callers call it.
# To avoid breaking other files, we'll keep the variable but make it a proxy or just call it.
# For now, let's just make it call the function once.
ALL_DATASETS = {} # Will be populated by callers who need it, or we can use a property trick

def get_capability_datasets(capability: str) -> List[str]:
    """Get datasets for a specific capability."""
    return get_all_datasets().get(capability, [])


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
