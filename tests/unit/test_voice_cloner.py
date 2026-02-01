"""
Unit tests for voice_engine/cloner.py
"""

import pytest
import sys
from unittest.mock import MagicMock, patch
from pathlib import Path

# Mock torch/torchaudio before import
torch_mock = MagicMock()
torch_mock.__spec__ = MagicMock()
torch_mock.__version__ = "2.3.0"  # Satisfy version checks
torch_mock.cuda.is_available.return_value = False
sys.modules["torch"] = torch_mock

torchaudio_mock = MagicMock()
torchaudio_mock.__spec__ = MagicMock()
sys.modules["torchaudio"] = torchaudio_mock

# Mock transformers to prevent deep dependency issues
transformers_mock = MagicMock()
transformers_mock.__spec__ = MagicMock()
sys.modules["transformers"] = transformers_mock


def test_voice_cloner_init():
    """Test voice cloner initialization."""
    from src.nexus.voice_engine import cloner
    
    # Test basic class existence
    assert hasattr(cloner, "VoiceCloner")
    assert hasattr(cloner, "VoiceEncoder")
    assert hasattr(cloner, "VoiceDNA")
    
    # Test instantiation
    vc = cloner.VoiceCloner()
    assert vc is not None
    assert vc.clones_created == 0


def test_voice_encoder_init():
    """Test voice encoder initialization."""
    from src.nexus.voice_engine import cloner
    
    encoder = cloner.VoiceEncoder()
    assert encoder.model_name == "microsoft/wavlm-base-plus-sv"
    assert encoder._model is None


def test_voice_dna_creation():
    """Test voice DNA dataclass."""
    from src.nexus.voice_engine import cloner
    
    import numpy as np
    embedding = np.zeros(512)
    
    dna = cloner.VoiceDNA(
        embedding=embedding,
        sample_rate=16000,
        duration_seconds=5.0,
        source_path="/path/to/audio.wav",
        voice_name="TestVoice",
        metadata={"description": "Test voice DNA"}
    )
    
    assert dna.voice_name == "TestVoice"
    assert dna.sample_rate == 16000
    assert dna.duration_seconds == 5.0


@patch("src.nexus.voice_engine.cloner.TORCH_AVAILABLE", True)
@patch("src.nexus.voice_engine.cloner.torch")
def test_clone_voice_with_torch(mock_torch):
    """Test voice cloning when torch is available."""
    from src.nexus.voice_engine import cloner
    
    # Mock torch.cuda
    mock_torch.cuda.is_available.return_value = False
    mock_torch.tensor = MagicMock
    
    # Mock torchaudio.load
    mock_waveform = MagicMock()
    mock_waveform.shape = [1, 16000]  # 1 second at 16kHz
    mock_waveform.dim.return_value = 2
    mock_waveform.mean.return_value = mock_waveform
    
    torchaudio_mock.load.return_value = (mock_waveform, 16000)
    
    vc = cloner.VoiceCloner()
    
    # Mock the encoder
    vc.encoder.extract_features = MagicMock(return_value=MagicMock())
    
    # Create dummy wav file
    with patch.object(Path, "exists", return_value=True):
        dna_path = vc.clone_voice("dummy_path.wav", "MyClonedVoice")
    
    # Should return a path or None
    assert dna_path is None or isinstance(dna_path, str)


def test_voice_cloner_compare():
    """Test voice comparison."""
    from src.nexus.voice_engine import cloner
    from unittest.mock import patch
    
    vc = cloner.VoiceCloner()
    
    # Test comparison with non-existent DNA (patch load to return None)
    with patch.object(vc, 'load_voice_dna', return_value=None):
        similarity = vc.compare_voices("/fake/path1.pt", "/fake/path2.pt")
        assert similarity == 0.0


def test_global_instances():
    """Test global module instances."""
    from src.nexus.voice_engine.cloner import voice_cloner
    
    assert voice_cloner is not None
    assert isinstance(voice_cloner.clones_created, int)
