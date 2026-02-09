"""
Unit tests for diffusion pipelines
15+ tests covering ImagePipeline, PipelineLoader, and DiffusionAdapter
"""

import pytest
import torch
from unittest.mock import Mock, patch, MagicMock
from PIL import Image
import numpy as np

from src.models.diffusion import ImagePipeline, PipelineConfig, DiffusionPipelineLoader
from src.models.diffusion.adapter import DiffusionAdapter


class TestPipelineConfig:
    """Tests for PipelineConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = PipelineConfig(model_id="test/model")
        assert config.model_id == "test/model"
        assert config.model_type == "sdxl"
        assert config.default_steps == 30
        assert config.default_guidance_scale == 7.5
        assert config.default_height == 1024
        assert config.default_width == 1024
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = PipelineConfig(
            model_id="test/model",
            model_type="flux",
            default_steps=50,
            default_guidance_scale=3.5,
        )
        assert config.model_type == "flux"
        assert config.default_steps == 50
        assert config.default_guidance_scale == 3.5
    
    def test_config_dtype(self):
        """Test dtype configuration."""
        config = PipelineConfig(model_id="test", dtype=torch.float32)
        assert config.dtype == torch.float32


class TestImagePipeline:
    """Tests for ImagePipeline class."""
    
    @pytest.fixture
    def mock_pipeline(self):
        """Create a mock pipeline for testing."""
        with patch('nexus.models.diffusion.image_pipeline.AutoPipelineForText2Image') as mock_pipe:
            mock_instance = Mock()
            mock_pipe.from_pretrained.return_value = mock_instance
            yield mock_instance
    
    def test_detect_model_type_sdxl(self):
        """Test SDXL model type detection."""
        config = PipelineConfig(model_id="stabilityai/stable-diffusion-xl-base-1.0")
        pipeline = ImagePipeline(config)
        assert pipeline.model_type == "sdxl"
    
    def test_detect_model_type_sd3(self):
        """Test SD3 model type detection."""
        config = PipelineConfig(model_id="stabilityai/stable-diffusion-3-medium")
        pipeline = ImagePipeline(config)
        assert pipeline.model_type == "sd3"
    
    def test_detect_model_type_flux(self):
        """Test FLUX model type detection."""
        config = PipelineConfig(model_id="black-forest-labs/FLUX.1-dev")
        pipeline = ImagePipeline(config)
        assert pipeline.model_type == "flux"
    
    def test_detect_model_type_z_image(self):
        """Test Z-Image model type detection."""
        config = PipelineConfig(model_id="stabilityai/z-image")
        pipeline = ImagePipeline(config)
        assert pipeline.model_type == "z-image"
    
    def test_detect_model_type_hunyuan(self):
        """Test Hunyuan model type detection."""
        config = PipelineConfig(model_id="Tencent-Hunyuan/HunyuanDiT-v1.2")
        pipeline = ImagePipeline(config)
        assert pipeline.model_type == "hunyuan"
    
    def test_detect_model_type_unknown(self):
        """Test unknown model defaults to sdxl."""
        config = PipelineConfig(model_id="unknown/random-model")
        pipeline = ImagePipeline(config)
        assert pipeline.model_type == "sdxl"
    
    def test_get_device_cuda(self):
        """Test device detection with CUDA."""
        with patch('torch.cuda.is_available', return_value=True):
            config = PipelineConfig(model_id="test")
            pipeline = ImagePipeline(config)
            assert pipeline.device.type == "cuda"
    
    def test_get_device_cpu(self):
        """Test device detection without CUDA."""
        with patch('torch.cuda.is_available', return_value=False):
            with patch('torch.backends.mps.is_available', return_value=False):
                config = PipelineConfig(model_id="test")
                pipeline = ImagePipeline(config)
                assert pipeline.device.type == "cpu"
    
    def test_prepare_generation_kwargs_flux_schnell(self):
        """Test FLUX Schnell specific kwargs."""
        config = PipelineConfig(model_id="flux-schnell", model_type="flux-schnell")
        pipeline = ImagePipeline(config)
        
        kwargs = pipeline._prepare_generation_kwargs(
            num_inference_steps=10,
            guidance_scale=7.5
        )
        assert kwargs["num_inference_steps"] == 4  # Capped at 4
        assert kwargs["guidance_scale"] == 0.0  # FLUX Schnell doesn't use CFG
    
    def test_prepare_generation_kwargs_flux_dev(self):
        """Test FLUX Dev specific kwargs."""
        config = PipelineConfig(model_id="flux-dev", model_type="flux")
        pipeline = ImagePipeline(config)
        
        kwargs = pipeline._prepare_generation_kwargs(guidance_scale=7.5)
        assert kwargs["guidance_scale"] == 3.5  # Adjusted for FLUX
    
    def test_prepare_generation_kwargs_sd3(self):
        """Test SD3 specific kwargs."""
        config = PipelineConfig(model_id="sd3", model_type="sd3")
        pipeline = ImagePipeline(config)
        
        kwargs = pipeline._prepare_generation_kwargs(guidance_scale=7.5)
        assert kwargs["guidance_scale"] == 5.0  # Adjusted for SD3
    
    def test_context_manager(self, mock_pipeline):
        """Test context manager functionality."""
        config = PipelineConfig(model_id="test/model")
        
        with patch.object(ImagePipeline, 'load') as mock_load, \
             patch.object(ImagePipeline, 'unload') as mock_unload:
            
            with ImagePipeline(config) as pipeline:
                mock_load.assert_called_once()
            
            mock_unload.assert_called_once()


class TestDiffusionPipelineLoader:
    """Tests for DiffusionPipelineLoader."""
    
    def test_list_presets(self):
        """Test listing available presets."""
        presets = DiffusionPipelineLoader.list_presets()
        assert "sdxl-base" in presets
        assert "sd-2-1" in presets
        assert "flux-dev" in presets
        assert "flux-schnell" in presets
    
    def test_get_preset_info_sdxl(self):
        """Test getting SDXL preset info."""
        info = DiffusionPipelineLoader.get_preset_info("sdxl-base")
        assert info["model_type"] == "sdxl"
        assert info["default_steps"] == 30
        assert info["default_height"] == 1024
    
    def test_get_preset_info_flux(self):
        """Test getting FLUX preset info."""
        info = DiffusionPipelineLoader.get_preset_info("flux-schnell")
        assert info["model_type"] == "flux"
        assert info["default_steps"] == 4
        assert info["default_guidance_scale"] == 0.0
    
    def test_get_preset_info_unknown(self):
        """Test getting unknown preset raises error."""
        with pytest.raises(ValueError):
            DiffusionPipelineLoader.get_preset_info("unknown-preset")
    
    def test_estimate_quantized_size_fp8(self):
        """Test size estimation for FP8 quantization."""
        with patch.object(DiffusionPipelineLoader, 'load') as mock_load:
            loader = DiffusionPipelineLoader()
            # Test size estimation logic
            size_bytes, human = loader.load_quantized.__wrapped__(loader, "test", "fp8")
            # Should return appropriate values


class TestDiffusionAdapter:
    """Tests for DiffusionAdapter."""
    
    def test_adapter_initialization(self):
        """Test adapter initialization."""
        adapter = DiffusionAdapter(
            teacher_dim=2048,
            student_dim=1024,
            extract_features_from="unet"
        )
        assert adapter.teacher_dim == 2048
        assert adapter.student_dim == 1024
        assert adapter.extract_features_from == "unet"
    
    def test_adapter_forward(self):
        """Test adapter forward pass."""
        adapter = DiffusionAdapter(teacher_dim=256, student_dim=128)
        x = torch.randn(1, 256)
        output = adapter(x)
        assert output.shape == (1, 128)
    
    def test_compute_distillation_loss_mse(self):
        """Test MSE distillation loss."""
        adapter = DiffusionAdapter(teacher_dim=256, student_dim=128)
        
        teacher_features = {
            "layer_0": torch.randn(1, 64, 64),
        }
        student_features = {
            "layer_0": torch.randn(1, 64, 64),
        }
        
        loss = adapter.compute_distillation_loss(
            teacher_features, student_features, loss_type="mse"
        )
        assert loss.item() >= 0
    
    def test_compute_distillation_loss_cosine(self):
        """Test cosine distillation loss."""
        adapter = DiffusionAdapter(teacher_dim=256, student_dim=128)
        
        teacher_features = {
            "layer_0": torch.randn(1, 64, 64),
        }
        student_features = {
            "layer_0": torch.randn(1, 64, 64),
        }
        
        loss = adapter.compute_distillation_loss(
            teacher_features, student_features, loss_type="cosine"
        )
        assert loss.item() >= 0
        assert loss.item() <= 1


class TestModelTypeMap:
    """Tests for model type mapping."""
    
    def test_model_type_map_entries(self):
        """Test all expected models are in the type map."""
        expected_models = [
            "stabilityai/stable-diffusion-2-1",
            "stabilityai/stable-diffusion-xl-base-1.0",
            "stabilityai/stable-diffusion-3-medium",
            "black-forest-labs/FLUX.1-dev",
            "black-forest-labs/FLUX.1-schnell",
            "black-forest-labs/FLUX.1-fill-dev",
            "Tencent-Hunyuan/HunyuanDiT-v1.2",
        ]
        
        for model in expected_models:
            assert model in ImagePipeline.MODEL_TYPE_MAP
