import pytest
import json
from unittest.mock import patch, MagicMock
from src.utils.generate_remotion_dataset import generate_sample, CATEGORIES

def test_generate_sample_basic():
    # Test without weights
    sample = generate_sample(1)
    assert sample["id"] == "remotion_1"
    assert "instruction" in sample
    assert "output" in sample
    assert sample["category"] == "remotion-explainer"

def test_generate_sample_with_weights():
    # Force 'math' category
    weights = [0] * len(CATEGORIES)
    math_idx = CATEGORIES.index("math")
    weights[math_idx] = 1.0
    
    sample = generate_sample(1, weights=weights)
    assert sample["instruction"] == "Explain Euler's Identity."
    assert "NexusMath" in sample["output"]

@patch("src.utils.generate_remotion_dataset.KB", {
    "history": {"timelines": [{"topic": "WW2", "events": []}]},
    "business": {
        "charts": [{"topic": "Sales", "type": "bar", "data": []}],
        "funnels": [{"topic": "Sales Pipeline", "steps": ["Leads", "Sales"]}]
    },
    "lifestyle": {"recipes": [{"topic": "Pasta", "steps": [{"title": "Boil"}]}]}
})
def test_generate_sample_kb_content():
    # Force 'timeline'
    weights = [0] * len(CATEGORIES)
    weights[CATEGORIES.index("timeline")] = 1.0
    sample = generate_sample(1, weights=weights)
    assert "WW2" in sample["instruction"]
    
    # Force 'chart'
    weights = [0] * len(CATEGORIES)
    weights[CATEGORIES.index("chart")] = 1.0
    sample = generate_sample(2, weights=weights)
    assert "Sales" in sample["instruction"]

def test_generate_sample_map():
    weights = [0] * len(CATEGORIES)
    weights[CATEGORIES.index("map")] = 1.0
    sample = generate_sample(3, weights=weights)
    assert "global locations" in sample["instruction"]
    assert "NexusMap" in sample["output"]
