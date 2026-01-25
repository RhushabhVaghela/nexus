"""
Unit tests for detect_modalities.py

Tests:
- Modality detection with real text-only model
- Modality detection with Omni model
- Format output (human-readable and JSON)
- Edge cases and error handling
"""

import pytest
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.detect_modalities import detect_modalities, format_report, _analyze_config


class TestDetectModalitiesWithRealModel:
    """Test modality detection using real models."""
    
    def test_detect_text_only_model(self, text_model_path):
        """Test detection on text-only Qwen2.5-0.5B."""
        result = detect_modalities(text_model_path)
        
        # Basic structure
        assert "modalities" in result
        assert "model_type" in result
        
        # Text-only model should have text=True, others=False
        mods = result["modalities"]
        assert mods["text"] is True
        assert mods["vision"] is False
        assert mods["audio_input"] is False
        assert mods["audio_output"] is False
        assert mods["video"] is False
        
        # Should not be Omni
        assert result.get("is_omni", False) is False
    
    def test_detect_omni_model(self, omni_model_path, request):
        """Test detection on Qwen2.5-Omni model (or small model if requested)."""
        if not Path(omni_model_path).exists():
            pytest.skip("Omni model not found")
        
        result = detect_modalities(omni_model_path)
        
        is_small = request.config.getoption("--small-model")
        
        mods = result["modalities"]
        assert mods["text"] is True
        
        if is_small:
            # If using small model for omni fixture, it shouldn't be omni
            assert mods["vision"] is False
            assert result.get("is_omni", False) is False
        else:
            assert mods["vision"] is True
            assert mods["audio_input"] is True
            assert mods["audio_output"] is True
            assert mods["video"] is True
            assert result.get("is_omni") is True


class TestAnalyzeConfig:
    """Test config analysis functions."""
    
    def test_text_only_config(self):
        """Test analyzing a text-only model config."""
        config = {
            "model_type": "qwen2",
            "architectures": ["Qwen2ForCausalLM"],
        }
        
        result = {
            "modalities": {"text": True, "vision": False, "vision_output": False,
                          "audio_input": False, "audio_output": False, 
                          "video": False, "video_output": False},
            "native_encoders": [],
            "native_decoders": [],
        }
        
        result = _analyze_config(config, result)
        
        assert result["modalities"]["text"] is True
        assert result["modalities"]["vision"] is False
    
    def test_vision_config_detection(self):
        """Test that vision_config is properly detected."""
        config = {
            "model_type": "llava",
            "architectures": ["LlavaForConditionalGeneration"],
            "vision_config": {"hidden_size": 1024},
        }
        
        result = {
            "modalities": {"text": True, "vision": False, "vision_output": False,
                          "audio_input": False, "audio_output": False,
                          "video": False, "video_output": False},
            "native_encoders": [],
            "native_decoders": [],
        }
        
        result = _analyze_config(config, result)
        
        assert result["modalities"]["vision"] is True
        assert "vision" in result["native_encoders"]
    
    def test_omni_model_detection(self):
        """Test Omni model type detection."""
        config = {
            "model_type": "qwen2_5_omni",
            "architectures": ["Qwen2_5OmniForConditionalGeneration"],
        }
        
        result = {
            "modalities": {"text": True, "vision": False, "vision_output": False,
                          "audio_input": False, "audio_output": False,
                          "video": False, "video_output": False},
            "native_encoders": [],
            "native_decoders": [],
        }
        
        result = _analyze_config(config, result)
        
        assert result.get("is_omni") is True
        assert result["modalities"]["vision"] is True
        assert result["modalities"]["audio_input"] is True
        assert result["modalities"]["audio_output"] is True
        assert result["modalities"]["video"] is True


class TestFormatReport:
    """Test report formatting."""
    
    def test_json_format(self):
        """Test JSON output format."""
        result = {
            "model_path": "/test/model",
            "model_type": "qwen2",
            "modalities": {"text": True, "vision": False, "vision_output": False,
                          "audio_input": False, "audio_output": False,
                          "video": False, "video_output": False},
            "native_encoders": [],
            "native_decoders": [],
        }
        
        output = format_report(result, use_json=True)
        
        # Should be valid JSON
        parsed = json.loads(output)
        assert parsed["model_type"] == "qwen2"
        assert parsed["modalities"]["text"] is True
    
    def test_human_readable_format(self):
        """Test human-readable output format."""
        result = {
            "model_path": "/test/model",
            "model_type": "qwen2",
            "modalities": {"text": True, "vision": False, "vision_output": False,
                          "audio_input": False, "audio_output": False,
                          "video": False, "video_output": False},
            "native_encoders": [],
            "native_decoders": [],
        }
        
        output = format_report(result, use_json=False)
        
        # Should contain key info
        assert "text" in output.lower()
        assert "vision" in output.lower()
        assert "✅" in output or "Yes" in output  # Positive indicator
        assert "❌" in output or "No" in output   # Negative indicator


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_nonexistent_path(self):
        """Test with non-existent path."""
        result = detect_modalities("/nonexistent/path/to/model")
        
        # Should have error but not crash
        assert "error" in result or result["modalities"]["text"] is True
    
    def test_empty_config(self):
        """Test with empty config."""
        result = {
            "modalities": {"text": True, "vision": False, "vision_output": False,
                          "audio_input": False, "audio_output": False,
                          "video": False, "video_output": False},
            "native_encoders": [],
            "native_decoders": [],
        }
        
        result = _analyze_config({}, result)
        
        # Should still have text=True (default for LLMs)
        assert result["modalities"]["text"] is True
