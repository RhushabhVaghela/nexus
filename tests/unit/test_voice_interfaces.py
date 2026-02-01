import pytest
import torch
from unittest.mock import MagicMock
from src.nexus.voice_engine.interfaces import (
    UniversalVoicePipeline,
    BaseReasoningEngine,
    BaseVoiceIdentity,
    BaseAcousticEngine,
    AudioSegment,
    BrainOutput,
)

def test_universal_voice_pipeline():
    # Mock implementations
    mock_brain = MagicMock(spec=BaseReasoningEngine)
    mock_identity = MagicMock(spec=BaseVoiceIdentity)
    mock_acoustic = MagicMock(spec=BaseAcousticEngine)
    
    pipeline = UniversalVoicePipeline(mock_brain, mock_identity, mock_acoustic)
    
    # Set up mock return values
    mock_brain.process.return_value = BrainOutput(
        text="Hello world",
        sentiment="positive",
        intent="greeting"
    )
    mock_acoustic.synthesize.return_value = AudioSegment(
        waveform=torch.zeros(16000),
        sample_rate=16000,
        duration_seconds=1.0,
        text="Hello world",
        voice_id="voice_1"
    )
    
    audio, brain_output = pipeline.process_turn("Hi", "voice_1")
    
    assert brain_output.text == "Hello world"
    assert audio.duration_seconds == 1.0
    assert mock_brain.process.called
    assert mock_acoustic.synthesize.called


def test_pipeline_without_engines():
    """Test pipeline works without optional engines."""
    pipeline = UniversalVoicePipeline()
    
    audio, brain_output = pipeline.process_turn("Hi")
    
    assert brain_output.text == "You said: Hi"
    assert audio.voice_id == pipeline.active_voice


def test_voice_switching():
    """Test voice switching functionality."""
    pipeline = UniversalVoicePipeline()
    
    assert pipeline.active_voice == "NATM1"
    
    pipeline.switch_voice("NATF0")
    assert pipeline.active_voice == "NATF0"


def test_vibe_setting():
    """Test vibe setting functionality."""
    pipeline = UniversalVoicePipeline()
    
    assert pipeline.active_vibe == "neutral"
    
    pipeline.set_vibe("excited")
    assert pipeline.active_vibe == "excited"


def test_conversation_history():
    """Test conversation history tracking."""
    pipeline = UniversalVoicePipeline()
    
    pipeline.process_turn("Hello")
    pipeline.process_turn("How are you?")
    
    assert len(pipeline.conversation_history) == 2
    assert pipeline.conversation_history[0]["user"] == "Hello"
    assert pipeline.conversation_history[1]["user"] == "How are you?"
    
    pipeline.clear_history()
    assert len(pipeline.conversation_history) == 0


def test_pipeline_stats():
    """Test pipeline statistics."""
    pipeline = UniversalVoicePipeline()
    
    stats = pipeline.get_stats()
    
    assert "active_voice" in stats
    assert "active_vibe" in stats
    assert stats["has_reasoning"] is False
    assert stats["has_acoustic"] is False
