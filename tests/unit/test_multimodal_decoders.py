import pytest
import torch
from unittest.mock import MagicMock, patch
from pathlib import Path
from src.multimodal.decoders import ImageDecoder, AudioDecoder, VideoDecoder, OmniDecoder, ContentDecoder

@pytest.fixture
def mock_image(tmp_path):
    img_path = tmp_path / "test.jpg"
    img_path.write_text("dummy")
    return str(img_path)

@pytest.fixture
def mock_audio(tmp_path):
    aud_path = tmp_path / "test.wav"
    aud_path.write_text("dummy")
    return str(aud_path)

def test_base_decoder():
    decoder = ContentDecoder()
    with pytest.raises(NotImplementedError):
        decoder.decode("test")

def test_image_decoder_init():
    decoder = ImageDecoder(model_id="mock-id")
    assert decoder.model_id == "mock-id"

def test_image_decoder_file_not_found():
    decoder = ImageDecoder()
    result = decoder.decode("non_existent.jpg")
    assert "warning" in result
    assert result["modality"] == "image"

@patch("PIL.Image.open")
def test_image_decoder_success(mock_open, mock_image):
    mock_img_obj = MagicMock()
    mock_open.return_value = mock_img_obj
    
    decoder = ImageDecoder()
    decoder.processor = MagicMock()
    decoder.processor.return_value = {"pixel_values": "tensors"}
    
    result = decoder.decode(mock_image)
    assert result["pixel_values"] == "tensors"
    assert result["modality"] == "image"

@patch("PIL.Image.open")
def test_image_decoder_error(mock_open, mock_image):
    mock_open.side_effect = Exception("Open failed")
    decoder = ImageDecoder()
    with pytest.raises(RuntimeError, match="Failed to process image"):
        decoder.decode(mock_image)

def test_image_decoder_no_processor(mock_image):
    with patch("PIL.Image.open") as mock_open:
        mock_open.return_value = MagicMock()
        decoder = ImageDecoder()
        decoder.processor = None
        result = decoder.decode(mock_image)
        assert result["modality"] == "image"
        assert "pixel_values" not in result

def test_audio_decoder_file_not_found():
    decoder = AudioDecoder()
    result = decoder.decode("non_existent.wav")
    assert "warning" in result
    assert result["modality"] == "audio"

@patch("torchaudio.load")
def test_audio_decoder_success(mock_load, mock_audio):
    # Test stereo to mono path (shape[0] > 1) and resampling (sample_rate != 16000)
    mock_waveform = torch.zeros(2, 16000) 
    mock_load.return_value = (mock_waveform, 44100)
    
    decoder = AudioDecoder()
    decoder.processor = MagicMock()
    decoder.processor.return_value = {"input_features": torch.zeros(1, 80, 3000)}
    
    with patch("torchaudio.transforms.Resample") as mock_resample_cls:
        mock_resampler = MagicMock()
        mock_resampler.return_value = torch.zeros(2, 16000)
        mock_resample_cls.return_value = mock_resampler
        
        result = decoder.decode(mock_audio)
        assert "input_features" in result
        assert result["modality"] == "audio"
        assert mock_resample_cls.called

@patch("torchaudio.load")
def test_audio_decoder_error(mock_load, mock_audio):
    mock_load.side_effect = Exception("Load failed")
    decoder = AudioDecoder()
    with pytest.raises(RuntimeError, match="Failed to process audio"):
        decoder.decode(mock_audio)

def test_audio_decoder_no_processor(mock_audio):
    with patch("torchaudio.load") as mock_load:
        mock_load.return_value = (torch.zeros(1, 16000), 16000)
        decoder = AudioDecoder()
        decoder.processor = None
        result = decoder.decode(mock_audio)
        assert result["modality"] == "audio"
        assert "input_features" not in result

@patch("transformers.AutoProcessor.from_pretrained")
def test_audio_decoder_init_error(mock_from_pretrained):
    mock_from_pretrained.side_effect = Exception("Init failed")
    # This should print warning but not raise
    decoder = AudioDecoder()
    assert decoder.processor is None

def test_video_decoder():
    decoder = VideoDecoder()
    result = decoder.decode("test.mp4")
    assert result["modality"] == "video"
    assert result["strategy"] == "temporal_pooling"

def test_omni_decoder():
    omni = OmniDecoder()
    omni.image = MagicMock()
    omni.audio = MagicMock()
    omni.video = MagicMock()
    
    omni.decode("f.jpg", "vision")
    assert omni.image.decode.called
    
    omni.decode("f.wav", "audio")
    assert omni.audio.decode.called
    
    omni.decode("f.mp4", "video")
    assert omni.video.decode.called
    
    omni.decode("f.jpg", "image")
    assert omni.image.decode.call_count == 2
    
    with pytest.raises(ValueError):
        omni.decode("f.txt", "text")
