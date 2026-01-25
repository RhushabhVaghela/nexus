import pytest
from unittest.mock import MagicMock, patch
from src.utils.model_utils import check_modality

@patch("src.utils.model_utils.AutoConfig.from_pretrained")
def test_check_modality_text(mock_config_fn):
    # Mock text model config
    mock_config = MagicMock()
    mock_config.vision_config = None
    mock_config.audio_config = None
    mock_config.architectures = ["Qwen2ForCausalLM"]
    mock_config.model_type = "qwen2"
    mock_config.to_dict.return_value = {}
    mock_config_fn.return_value = mock_config
    
    assert check_modality("path", "text") is True
    assert check_modality("path", "multimodal") is False

@patch("src.utils.model_utils.AutoConfig.from_pretrained")
def test_check_modality_multimodal_vision(mock_config_fn):
    # Mock multimodal model (has vision_config)
    mock_config = MagicMock()
    mock_config.vision_config = MagicMock()
    mock_config.audio_config = None
    mock_config.architectures = ["LlavaForConditionalGeneration"]
    mock_config.model_type = "llava"
    mock_config_fn.return_value = mock_config
    
    assert check_modality("path", "multimodal") is True
    assert check_modality("path", "text") is False

@patch("src.utils.model_utils.AutoConfig.from_pretrained")
def test_check_modality_multimodal_arch(mock_config_fn):
    # Mock multimodal model (detected by arch string)
    mock_config = MagicMock()
    mock_config.vision_config = None
    mock_config.audio_config = None
    mock_config.architectures = ["Qwen2VLForConditionalGeneration"]
    mock_config.model_type = "qwen2_vl"
    mock_config_fn.return_value = mock_config
    
    assert check_modality("path", "multimodal") is True

def test_check_modality_error():
    with patch("src.utils.model_utils.AutoConfig.from_pretrained", side_effect=Exception("Fail")):
        assert check_modality("path") is False
