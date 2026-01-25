import pytest
import os
import csv
import shutil
from pathlib import Path
from src.metrics_tracker import MetricsTracker, TrainingMetrics, ValidationMetrics, BenchmarkMetrics, ExecutionDetailMetrics, ProgressTracker

@pytest.fixture
def temp_results_dir(tmp_path):
    return tmp_path / "results"

def test_metrics_tracker_init(temp_results_dir):
    tracker = MetricsTracker(output_dir=str(temp_results_dir))
    assert temp_results_dir.exists()
    assert (temp_results_dir / "training_metrics.csv").exists()
    assert (temp_results_dir / "validation_metrics.csv").exists()
    assert (temp_results_dir / "benchmark_metrics.csv").exists()
    assert (temp_results_dir / "test_details.csv").exists()

def test_log_training(temp_results_dir):
    tracker = MetricsTracker(output_dir=str(temp_results_dir))
    metrics = TrainingMetrics(
        capability="test_cap",
        dataset="test_ds",
        steps=100,
        final_loss=0.5,
        success=True
    )
    tracker.log_training(metrics)
    
    with open(temp_results_dir / "training_metrics.csv", "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        assert len(rows) == 1
        assert rows[0]["capability"] == "test_cap"
        assert rows[0]["dataset"] == "test_ds"
        assert float(rows[0]["final_loss"]) == 0.5
        assert rows[0]["success"] == "True"

def test_log_validation(temp_results_dir):
    tracker = MetricsTracker(output_dir=str(temp_results_dir))
    metrics = ValidationMetrics(
        test_type="unit",
        total_tests=10,
        passed=9,
        failed=1
    )
    tracker.log_validation(metrics)
    
    with open(temp_results_dir / "validation_metrics.csv", "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        assert len(rows) == 1
        assert rows[0]["test_type"] == "unit"
        assert int(rows[0]["total_tests"]) == 10

def test_log_benchmark(temp_results_dir):
    tracker = MetricsTracker(output_dir=str(temp_results_dir))
    metrics = BenchmarkMetrics(
        name="speed_test",
        tokens_per_second=50.5
    )
    tracker.log_benchmark(metrics)
    
    with open(temp_results_dir / "benchmark_metrics.csv", "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        assert len(rows) == 1
        assert rows[0]["name"] == "speed_test"
        assert float(rows[0]["tokens_per_second"]) == 50.5

def test_progress_tracker():
    with ProgressTracker(total=10, desc="Testing") as tracker:
        for i in range(10):
            tracker.update(loss=1.0 - i/10)
    
    metrics = tracker.get_metrics()
    assert metrics["initial_loss"] == 1.0
    assert metrics["final_loss"] == pytest.approx(0.1)
    assert metrics["avg_loss"] == pytest.approx(0.55)
    assert metrics["duration_seconds"] > 0

def test_gpu_metrics_fallback():
    tracker = MetricsTracker()
    metrics = tracker.get_gpu_metrics()
    assert "gpu_memory_used_gb" in metrics
    assert "gpu_memory_peak_gb" in metrics
