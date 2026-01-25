import pytest
import torch
from pathlib import Path
from unittest.mock import MagicMock, patch
from src.voice_engine.vibe_modulator import VibeModulator

class TestVibeModulator:
    def test_init(self):
        modulator = VibeModulator(model_path="/fake/path")
        assert modulator.model_path == Path("/fake/path")

    @patch("transformers.pipeline")
    def test_detect_vibe(self, mock_pipeline):
        mock_analyzer = MagicMock()
        mock_analyzer.return_value = [{'label': 'POSITIVE', 'score': 0.95}]
        mock_pipeline.return_value = mock_analyzer
        
        modulator = VibeModulator()
        # Priority check
        assert modulator.detect_vibe("This is great!") == "excited"
        assert modulator.detect_vibe("Is it?") == "curious"
        
        # Sentiment check
        assert modulator.detect_vibe("I love this") == "happy"

    def test_get_vibe_params(self):
        modulator = VibeModulator()
        params = modulator.get_vibe_params("excited")
        assert params["pitch"] == 1.1
        assert params["emotion_id"] == 1
        
        # Fallback
        params = modulator.get_vibe_params("unknown")
        assert params == VibeModulator.VIBE_MAP["neutral"]

    def test_apply_vibe(self):
        modulator = VibeModulator()
        audio = torch.randn(1, 16000)
        # apply_vibe currently returns audio as is (placeholder)
        out = modulator.apply_vibe(audio, "excited")
        assert torch.equal(out, audio)
