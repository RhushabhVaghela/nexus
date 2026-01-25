import pytest
import os
import json
import torch
from pathlib import Path
from src.utils.data_mixer import normalize_to_messages, mix_datasets
from src.utils.diversity_enforcement import create_zipfian_weights, EntropyTracker, DiversityEnforcedGenerator
from src.utils.hardware_optimizer import optimize_for_hardware
from src.utils.quality_metrics import QualityMetrics, QualityScoringPipeline

# --- Data Mixer Tests ---

def test_normalize_to_messages_alpaca():
    sample = {"instruction": "Hi", "output": "Hello"}
    norm = normalize_to_messages(sample)
    assert norm["messages"][0]["content"] == "Hi"
    assert norm["messages"][1]["content"] == "Hello"

def test_normalize_to_messages_sharegpt():
    sample = {"conversations": [{"from": "human", "value": "Hi"}, {"from": "gpt", "value": "Hello"}]}
    norm = normalize_to_messages(sample)
    assert norm["messages"][0]["content"] == "Hi"
    assert norm["messages"][1]["content"] == "Hello"

def test_mix_datasets():
    real = [{"id": f"r{i}", "messages": [{"role": "u", "content": "r"}]} for i in range(100)]
    synth = [{"id": f"s{i}", "messages": [{"role": "u", "content": "s"}]} for i in range(100)]
    
    mixed, stats = mix_datasets(real, synth, real_ratio=0.3)
    # 70% synth = 100, so real should be ~42 (42 / 142 ~= 0.295)
    assert stats["synthetic_samples"] == 100
    assert stats["real_samples"] <= 100
    assert len(mixed) == stats["real_samples"] + 100

# --- Diversity Enforcement Tests ---

def test_zipfian_weights():
    blueprints = ["A", "B", "C"]
    weights = create_zipfian_weights(blueprints)
    assert weights["A"] > weights["B"] > weights["C"]
    assert sum(weights.values()) == pytest.approx(1.0)

def test_entropy_tracker():
    tracker = EntropyTracker()
    tracker.record("A")
    tracker.record("A")
    tracker.record("B")
    entropy = tracker.compute_entropy()
    assert 0 < entropy < 1.0

def test_diversity_generator():
    gen = DiversityEnforcedGenerator(blueprint_weights={"A": 0.9, "B": 0.1})
    selected = gen.select_with_diversity_bias()
    assert selected in ["A", "B"]
    gen.record_generation(selected)
    assert gen.template_usage[selected] == 1

# --- Hardware Optimizer Tests ---

def test_hardware_optimization():
    profile = optimize_for_hardware()
    assert "device" in profile
    assert "flash_attention" in profile

# --- Quality Metrics Tests ---

def test_quality_metrics_syntax():
    sample = {"messages": [{"role": "assistant", "content": "```python\ndef foo(): return 42\n```"}]}
    assert QualityMetrics.syntax_validity(sample) == 1.0
    
    bad_sample = {"messages": [{"role": "assistant", "content": "```python\ndef foo(\n```"}]}
    assert QualityMetrics.syntax_validity(bad_sample) < 1.0

def test_quality_metrics_length():
    sample = {"messages": [{"role": "user", "content": "a" * 1000}]}
    assert QualityMetrics.content_length_score(sample) == 1.0

def test_quality_pipeline():
    pipeline = QualityScoringPipeline(threshold=0.1)
    batch = [
        {"messages": [{"role": "assistant", "content": "short"}]},
        {"messages": [{"role": "assistant", "content": "```python\nprint('hello')\n```"}]}
    ]
    filtered = pipeline.filter_batch(batch)
    assert len(filtered) > 0
