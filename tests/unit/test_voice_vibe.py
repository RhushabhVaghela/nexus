import pytest
import torch
from pathlib import Path
from unittest.mock import MagicMock, patch
from src.nexus.voice_engine.vibe_modulator import VibeModulator, VibeParams, EmotionID


class TestVibeModulator:
    def test_init(self):
        """Test modulator initialization."""
        modulator = VibeModulator(config_path="/fake/path")
        assert modulator.config_path == "/fake/path"
        assert modulator.custom_vibes == {}

    def test_get_vibe_params(self):
        """Test getting vibe parameters."""
        modulator = VibeModulator()
        
        # Test excited vibe
        params = modulator.get_vibe_params("excited")
        assert params["pitch"] == 1.1
        assert params["emotion_id"] == EmotionID.EXCITED
        
        # Test neutral fallback for unknown vibe
        params = modulator.get_vibe_params("unknown")
        assert params["pitch"] == 1.0  # neutral defaults
        
    def test_get_vibe_params_default(self):
        """Test default vibe is neutral."""
        modulator = VibeModulator()
        params = modulator.get_vibe_params()
        assert params["pitch"] == 1.0
        assert params["energy"] == 1.0
        assert params["speed"] == 1.0

    def test_list_vibes(self):
        """Test listing available vibes."""
        modulator = VibeModulator()
        vibes = modulator.list_vibes()
        
        assert "neutral" in vibes
        assert "happy" in vibes
        assert "excited" in vibes
        assert "professional" in vibes

    def test_modulate_audio_params(self):
        """Test audio parameter modulation."""
        modulator = VibeModulator()
        
        base_params = {"pitch": 1.0, "energy": 1.0, "speed": 1.0}
        modulated = modulator.modulate_audio_params(base_params, "excited")
        
        assert modulated["pitch"] == 1.0 * 1.1  # base * excited pitch
        assert modulated["energy"] == 1.0 * 1.3  # base * excited energy

    def test_get_emotion_id(self):
        """Test getting emotion ID."""
        modulator = VibeModulator()
        
        assert modulator.get_emotion_id("neutral") == EmotionID.NEUTRAL
        assert modulator.get_emotion_id("happy") == EmotionID.HAPPY
        assert modulator.get_emotion_id("excited") == EmotionID.EXCITED

    def test_blend_vibes(self):
        """Test blending two vibes."""
        modulator = VibeModulator()
        
        blended = modulator.blend_vibes("neutral", "excited", ratio=0.5)
        
        # Should be halfway between neutral (1.0) and excited (1.1/1.3/1.15)
        assert blended.pitch == pytest.approx(1.05, 0.01)
        assert blended.energy == pytest.approx(1.15, 0.01)

    def test_register_custom_vibe(self):
        """Test registering a custom vibe."""
        modulator = VibeModulator()
        
        custom_params = VibeParams(pitch=1.5, energy=0.5, speed=1.2)
        modulator.register_vibe("custom", custom_params)
        
        assert "custom" in modulator.list_vibes()
        params = modulator.get_vibe_params("custom")
        assert params["pitch"] == 1.5


class TestVibeParams:
    def test_default_values(self):
        """Test default parameter values."""
        params = VibeParams()
        assert params.pitch == 1.0
        assert params.energy == 1.0
        assert params.speed == 1.0
        assert params.emotion_id == 0

    def test_to_dict(self):
        """Test conversion to dictionary."""
        params = VibeParams(pitch=1.2, energy=0.8)
        data = params.to_dict()
        
        assert data["pitch"] == 1.2
        assert data["energy"] == 0.8

    def test_from_dict(self):
        """Test creation from dictionary."""
        data = {"pitch": 1.3, "energy": 1.1, "speed": 0.9, "emotion_id": 5,
                "warmth": 1.0, "brightness": 1.0, "clarity": 1.0, "expressiveness": 1.0}
        params = VibeParams.from_dict(data)
        
        assert params.pitch == 1.3
        assert params.emotion_id == 5


class TestEmotionID:
    def test_enum_values(self):
        """Test emotion ID enum values."""
        assert EmotionID.NEUTRAL == 0
        assert EmotionID.HAPPY == 2
        assert EmotionID.EXCITED == 8
