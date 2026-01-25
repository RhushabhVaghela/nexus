"""
Unit tests for data_mixer.py
"""

import pytest
import sys
import json
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open

from src.utils.data_mixer import normalize_to_messages, mix_datasets, split_and_save

class TestDataMixer:
    
    def test_normalize_alpaca(self):
        sample = {
            "instruction": "Hello",
            "input": "World",
            "output": "Hi there"
        }
        normalized = normalize_to_messages(sample, source="test")
        assert normalized is not None
        assert normalized["messages"][0]["content"] == "Hello\n\nWorld"
        assert normalized["messages"][1]["content"] == "Hi there"
        
    def test_normalize_sharegpt(self):
        sample = {
            "conversations": [
                {"from": "human", "value": "Hi"},
                {"from": "gpt", "value": "Hello"}
            ]
        }
        normalized = normalize_to_messages(sample, source="test")
        assert normalized is not None
        assert normalized["messages"][0]["role"] == "user"
        assert normalized["messages"][1]["role"] == "assistant"

    def test_normalize_multimodal(self):
        sample = {
            "messages": [
                {"role": "user", "content": "Look"},
                {"role": "assistant", "content": "I see."}
            ],
            "modalities": {"image": [{"path": "img.png"}]}
        }
        normalized = normalize_to_messages(sample, source="test")
        assert normalized is not None
        assert "modalities" in normalized
        assert normalized["modalities"]["image"][0]["path"] == "img.png"

    def test_mix_datasets(self):
        real = [{"id": i, "source": "real"} for i in range(10)]
        synth = [{"id": i, "source": "synth"} for i in range(20)]
        
        # Target 30% real
        mixed, stats = mix_datasets(real, synth, real_ratio=0.3, seed=42)
        
        # Total synthetic is 20. Target real = 20 * (0.3/0.7) = 8.57 -> 8
        # Total mixed = 20 + 8 = 28
        assert len(mixed) == 28
        real_count = sum(1 for s in mixed if s["source"] == "real")
        assert real_count == 8
        assert stats["real_samples"] == 8
        
    def test_mix_datasets_not_enough_real(self):
        real = [{"id": i, "source": "real"} for i in range(2)] # Only 2 real
        synth = [{"id": i, "source": "synth"} for i in range(20)]
        
        # Target 30% real -> needs 8 real, but we only have 2
        mixed, stats = mix_datasets(real, synth, real_ratio=0.3, seed=42)
        
        # Should take all 2 real
        assert len(mixed) == 22
        real_count = sum(1 for s in mixed if s["source"] == "real")
        assert real_count == 2

    def test_split_and_save(self):
        samples = [{"id": i} for i in range(100)]
        with patch("src.utils.data_mixer.Path.mkdir"), \
             patch("builtins.open", mock_open()) as mock_file:
            
            splits = split_and_save(samples, "output", train_ratio=0.8, val_ratio=0.1)
            
            assert len(splits["train"]) == 80
            assert len(splits["val"]) == 10
            assert len(splits["test"]) == 10
            assert mock_file.call_count >= 3
