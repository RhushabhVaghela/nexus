"""
Integration tests for multimodal encoders.

Tests:
- Encoder initialization
- Projector shapes
- Wrapper integration
- Decoder interfaces
"""

import pytest
import torch
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestVisionEncoderIntegration:
    """Test VisionEncoder with real/mocked components."""
    
    def test_vision_encoder_import(self):
        """Test VisionEncoder can be imported."""
        from src.multimodal.model import VisionEncoder
        assert VisionEncoder is not None
    
    def test_vision_encoder_default_path(self):
        """Test VisionEncoder uses correct default path."""
        from src.multimodal.model import VisionEncoder
        
        # Check default parameter
        import inspect
        sig = inspect.signature(VisionEncoder.__init__)
        model_name_param = sig.parameters.get('model_name')
        
        assert model_name_param is not None
        assert "/mnt/e/data/encoders" in str(model_name_param.default)
    
    @pytest.mark.slow
    @pytest.mark.gpu
    def test_vision_encoder_loads_if_exists(self, vision_encoder_path):
        """Test VisionEncoder loads if path exists."""
        if not Path(vision_encoder_path).exists():
            pytest.skip("Vision encoder not found")
        
        from src.multimodal.model import VisionEncoder
        
        encoder = VisionEncoder(
            model_name=vision_encoder_path,
            load_in_8bit=True,  # Save memory
        )
        
        assert encoder is not None
        assert hasattr(encoder, 'forward')


class TestAudioEncoderIntegration:
    """Test AudioEncoder with real/mocked components."""
    
    def test_audio_encoder_import(self):
        """Test AudioEncoder can be imported."""
        from src.multimodal.model import AudioEncoder
        assert AudioEncoder is not None
    
    def test_audio_encoder_default_path(self):
        """Test AudioEncoder uses correct default path."""
        from src.multimodal.model import AudioEncoder
        
        import inspect
        sig = inspect.signature(AudioEncoder.__init__)
        model_name_param = sig.parameters.get('model_name')
        
        assert model_name_param is not None
        assert "/mnt/e/data/encoders" in str(model_name_param.default)


class TestProjectorShapes:
    """Test projector network shapes."""
    
    @pytest.mark.skip(reason="MLP/Projector class not implemented in current model")
    def test_projector_import(self):
        """Test MLP can be imported."""
        pass
    
    @pytest.mark.skip(reason="MLP/Projector class not implemented in current model")  
    def test_projector_initialization(self):
        """Test projector initializes with correct dimensions."""
        pass
    
    @pytest.mark.skip(reason="MLP/Projector class not implemented in current model")
    def test_projector_forward_shape(self):
        """Test projector produces correct output shape."""
        pass


class TestPerceiverResampler:
    """Test PerceiverResampler for vision/audio token reduction."""
    
    def test_resampler_import(self):
        """Test PerceiverResampler can be imported."""
        from src.multimodal.model import PerceiverResampler
        assert PerceiverResampler is not None
    
    def test_resampler_initialization(self):
        """Test resampler initializes correctly."""
        from src.multimodal.model import PerceiverResampler
        
        dim = 1024
        num_latents = 64
        
        resampler = PerceiverResampler(dim=dim, num_latents=num_latents)
        
        assert resampler is not None
    
    def test_resampler_reduces_sequence(self):
        """Test resampler reduces sequence length."""
        from src.multimodal.model import PerceiverResampler
        
        dim = 1024
        num_latents = 64
        batch_size = 2
        input_seq_len = 256  # Many vision tokens
        
        resampler = PerceiverResampler(dim=dim, num_latents=num_latents)
        
        # Create dummy input
        x = torch.randn(batch_size, input_seq_len, dim)
        
        # Forward pass
        out = resampler(x)
        
        # Should reduce to num_latents
        assert out.shape[1] == num_latents
        assert out.shape[0] == batch_size
        assert out.shape[2] == dim


class TestModularMultimodalWrapper:
    """Test ModularMultimodalWrapper integration."""
    
    def test_wrapper_import(self):
        """Test wrapper can be imported."""
        from src.multimodal.model import ModularMultimodalWrapper
        assert ModularMultimodalWrapper is not None
    
    def test_wrapper_default_paths(self):
        """Test wrapper uses correct default encoder paths."""
        from src.multimodal.model import ModularMultimodalWrapper
        
        import inspect
        sig = inspect.signature(ModularMultimodalWrapper.__init__)
        
        vision_param = sig.parameters.get('vision_name')
        audio_param = sig.parameters.get('audio_name')
        
        if vision_param:
            assert "/mnt/e/data/encoders" in str(vision_param.default)
        if audio_param:
            assert "/mnt/e/data/encoders" in str(audio_param.default)


class TestDecoderInterfaces:
    """Test decoder interface implementations."""
    
    def test_image_decoder_import(self):
        """Test ImageDecoder can be imported."""
        from src.multimodal.decoders import ImageDecoder
        assert ImageDecoder is not None
    
    def test_audio_decoder_import(self):
        """Test AudioDecoder can be imported."""
        from src.multimodal.decoders import AudioDecoder
        assert AudioDecoder is not None
    
    def test_video_decoder_import(self):
        """Test VideoDecoder can be imported."""
        from src.multimodal.decoders import VideoDecoder
        assert VideoDecoder is not None
    
    def test_omni_decoder_import(self):
        """Test OmniDecoder can be imported."""
        from src.multimodal.decoders import OmniDecoder
        assert OmniDecoder is not None
    
    def test_image_decoder_decode(self):
        """Test ImageDecoder decode method."""
        from src.multimodal.decoders import ImageDecoder
        
        decoder = ImageDecoder()
        result = decoder.decode("test.png")
        
        assert "modality" in result
        assert result["modality"] == "image"
        assert "tensor_type" in result
    
    def test_audio_decoder_decode(self):
        """Test AudioDecoder decode method."""
        from src.multimodal.decoders import AudioDecoder
        
        decoder = AudioDecoder()
        result = decoder.decode("test.mp3")
        
        assert "modality" in result
        assert result["modality"] == "audio"
    
    def test_video_decoder_decode(self):
        """Test VideoDecoder decode method."""
        from src.multimodal.decoders import VideoDecoder
        
        decoder = VideoDecoder()
        result = decoder.decode("test.mp4")
        
        assert "modality" in result
        assert result["modality"] == "video"
    
    def test_omni_decoder_routes_correctly(self):
        """Test OmniDecoder routes to correct sub-decoder."""
        from src.multimodal.decoders import OmniDecoder
        
        decoder = OmniDecoder()
        
        # Test vision routing
        result = decoder.decode("test.png", modality="vision")
        assert result["modality"] == "image"
        
        # Test audio routing
        result = decoder.decode("test.mp3", modality="audio")
        assert result["modality"] == "audio"
        
        # Test video routing
        result = decoder.decode("test.mp4", modality="video")
        assert result["modality"] == "video"


class TestEncodersConfig:
    """Test encoders.yaml configuration."""
    
    def test_config_loads(self, encoders_config):
        """Test config loads successfully."""
        assert encoders_config is not None
    
    def test_config_has_encoders(self, encoders_config):
        """Test config has encoders section."""
        assert "encoders" in encoders_config
        assert "vision" in encoders_config["encoders"]
        assert "audio_input" in encoders_config["encoders"]
    
    def test_config_has_decoders(self, encoders_config):
        """Test config has decoders section."""
        assert "decoders" in encoders_config
        assert "vision_output" in encoders_config["decoders"]
        assert "video_output" in encoders_config["decoders"]
    
    def test_config_has_datasets(self, encoders_config):
        """Test config has datasets section."""
        assert "datasets" in encoders_config
    
    def test_encoder_paths_are_local(self, encoders_config):
        """Test encoder paths use local paths."""
        vision_path = encoders_config["encoders"]["vision"]["default"]
        assert "/mnt/e/data" in vision_path
        
        audio_path = encoders_config["encoders"]["audio_input"]["default"]
        assert "/mnt/e/data" in audio_path
