import pytest
import torch
from unittest.mock import MagicMock
from src.voice_engine.interfaces import UniversalVoicePipeline, BaseReasoningEngine, BaseVoiceIdentity, BaseAcousticEngine

def test_universal_voice_pipeline():
    # Mock implementations
    mock_brain = MagicMock(spec=BaseReasoningEngine)
    mock_identity = MagicMock(spec=BaseVoiceIdentity)
    mock_acoustic = MagicMock(spec=BaseAcousticEngine)
    
    pipeline = UniversalVoicePipeline(mock_brain, mock_identity, mock_acoustic)
    
    mock_brain.generate_response.return_value = {
        "text": "Hello world",
        "metadata": {"vibe": "happy"}
    }
    mock_identity.get_embedding.return_value = torch.zeros(1, 256)
    mock_acoustic.synthesize.return_value = torch.zeros(1, 16000)
    
    audio, brain_output = pipeline.process_turn("Hi", "voice_1")
    
    assert brain_output["text"] == "Hello world"
    assert audio.shape[1] == 16000
    assert mock_brain.generate_response.called
    assert mock_identity.get_embedding.called
    assert mock_acoustic.synthesize.called
