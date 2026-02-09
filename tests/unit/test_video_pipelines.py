"""
Unit tests for video pipelines
12+ tests covering VideoPipeline, FrameGenerator, and TemporalConsistency
"""

import pytest
import torch
from unittest.mock import Mock, patch
from PIL import Image
import numpy as np

from src.models.video import VideoPipeline, VideoConfig, FrameGenerator
from src.models.video.frame_generator import FrameGenerationConfig
from src.models.video.temporal_consistency import (
    TemporalConsistencyProcessor, TemporalAttention, MotionEstimator
)


class TestVideoConfig:
    """Tests for VideoConfig dataclass."""
    
    def test_default_config(self):
        """Test default video configuration."""
        config = VideoConfig(model_id="test/model")
        assert config.model_id == "test/model"
        assert config.model_type == "ltx-video"
        assert config.num_frames == 49
        assert config.fps == 24
        assert config.height == 512
        assert config.width == 512
    
    def test_custom_config(self):
        """Test custom video configuration."""
        config = VideoConfig(
            model_id="test",
            num_frames=16,
            fps=30,
            height=1024,
            width=1024,
        )
        assert config.num_frames == 16
        assert config.fps == 30
        assert config.height == 1024


class TestVideoPipeline:
    """Tests for VideoPipeline class."""
    
    def test_detect_model_type_ltx(self):
        """Test LTX-Video model type detection."""
        config = VideoConfig(model_id="Lightricks/LTX-Video")
        pipeline = VideoPipeline(config)
        assert pipeline.model_type == "ltx-video"
    
    def test_detect_model_type_svd(self):
        """Test SVD model type detection."""
        config = VideoConfig(model_id="stabilityai/stable-video-diffusion-img2vid")
        pipeline = VideoPipeline(config)
        assert pipeline.model_type == "svd"
    
    def test_detect_model_type_cogvideo(self):
        """Test CogVideoX model type detection."""
        config = VideoConfig(model_id="THUDM/CogVideoX-2b")
        pipeline = VideoPipeline(config)
        assert pipeline.model_type == "cogvideo"
    
    def test_detect_model_type_hunyuan(self):
        """Test HunyuanVideo model type detection."""
        config = VideoConfig(model_id="Tencent-Hunyuan/HunyuanVideo")
        pipeline = VideoPipeline(config)
        assert pipeline.model_type == "hunyuan-video"
    
    def test_get_device_cuda(self):
        """Test device detection with CUDA."""
        with patch('torch.cuda.is_available', return_value=True):
            config = VideoConfig(model_id="test")
            pipeline = VideoPipeline(config)
            assert pipeline.device.type == "cuda"
    
    def test_get_device_mps(self):
        """Test device detection with MPS."""
        with patch('torch.cuda.is_available', return_value=False):
            with patch('torch.backends.mps.is_available', return_value=True):
                config = VideoConfig(model_id="test")
                pipeline = VideoPipeline(config)
                assert pipeline.device.type == "mps"
    
    def test_model_type_map(self):
        """Test all video models in type map."""
        expected = [
            "Lightricks/LTX-Video",
            "stabilityai/stable-video-diffusion-img2vid",
            "stabilityai/stable-video-diffusion-img2vid-xt",
            "THUDM/CogVideoX-2b",
            "Tencent-Hunyuan/HunyuanVideo",
        ]
        for model in expected:
            assert model in VideoPipeline.MODEL_TYPE_MAP


class TestFrameGenerator:
    """Tests for FrameGenerator class."""
    
    @pytest.fixture
    def mock_pipeline(self):
        """Create a mock pipeline."""
        pipeline = Mock()
        pipeline.generate.return_value = {
            "images": [Image.new("RGB", (64, 64), color="red")]
        }
        pipeline.generate_variations.return_value = {
            "images": [Image.new("RGB", (64, 64), color="blue")]
        }
        return pipeline
    
    def test_frame_generator_init(self, mock_pipeline):
        """Test frame generator initialization."""
        config = FrameGenerationConfig()
        generator = FrameGenerator(mock_pipeline, config)
        assert generator.pipeline == mock_pipeline
        assert generator.config == config
    
    def test_blend_frames(self, mock_pipeline):
        """Test frame blending."""
        config = FrameGenerationConfig()
        generator = FrameGenerator(mock_pipeline, config)
        
        frames_a = [Image.new("RGB", (64, 64), color="red") for _ in range(3)]
        frames_b = [Image.new("RGB", (64, 64), color="blue") for _ in range(3)]
        
        blended = generator._blend_frames(frames_a, frames_b)
        assert len(blended) == 3
        assert all(isinstance(f, Image.Image) for f in blended)
    
    def test_blend_images(self, mock_pipeline):
        """Test image blending."""
        config = FrameGenerationConfig()
        generator = FrameGenerator(mock_pipeline, config)
        
        img_a = Image.new("RGB", (64, 64), color=(255, 0, 0))
        img_b = Image.new("RGB", (64, 64), color=(0, 0, 255))
        
        blended = generator._blend_images(img_a, img_b, 0.5, 0.5)
        assert isinstance(blended, Image.Image)
        assert blended.size == (64, 64)
    
    def test_clear_buffer(self, mock_pipeline):
        """Test clearing frame buffer."""
        config = FrameGenerationConfig()
        generator = FrameGenerator(mock_pipeline, config)
        generator._frame_buffer = [Image.new("RGB", (64, 64))]
        
        generator.clear_buffer()
        assert len(generator._frame_buffer) == 0


class TestTemporalConsistency:
    """Tests for TemporalConsistencyProcessor."""
    
    @pytest.fixture
    def sample_frames(self):
        """Create sample frames for testing."""
        return [Image.new("RGB", (64, 64), color=(i*10, i*10, i*10)) for i in range(5)]
    
    def test_processor_init(self):
        """Test processor initialization."""
        processor = TemporalConsistencyProcessor(
            consistency_weight=0.8,
            flow_weight=0.5,
            use_optical_flow=True
        )
        assert processor.consistency_weight == 0.8
        assert processor.flow_weight == 0.5
        assert processor.use_optical_flow is True
    
    def test_temporal_smoothing(self, sample_frames):
        """Test temporal smoothing."""
        processor = TemporalConsistencyProcessor(use_optical_flow=False)
        smoothed = processor._apply_temporal_smoothing(sample_frames, strength=0.5)
        assert len(smoothed) == len(sample_frames)
        assert all(isinstance(f, Image.Image) for f in smoothed)
    
    def test_stabilization(self, sample_frames):
        """Test motion stabilization."""
        processor = TemporalConsistencyProcessor(use_optical_flow=False)
        stabilized = processor._apply_stabilization(sample_frames, strength=0.3)
        assert len(stabilized) == len(sample_frames)
        assert all(isinstance(f, Image.Image) for f in stabilized)
    
    def test_compute_temporal_loss_mse(self):
        """Test temporal loss computation with MSE."""
        processor = TemporalConsistencyProcessor()
        features = [
            torch.randn(1, 64, 32, 32),
            torch.randn(1, 64, 32, 32),
            torch.randn(1, 64, 32, 32),
        ]
        
        loss = processor.compute_temporal_loss(features, loss_type="mse")
        assert loss.item() >= 0
    
    def test_interpolate_frames(self):
        """Test frame interpolation."""
        processor = TemporalConsistencyProcessor()
        frame1 = Image.new("RGB", (64, 64), color=(255, 0, 0))
        frame2 = Image.new("RGB", (64, 64), color=(0, 0, 255))
        
        interpolated = processor.interpolate_frames(frame1, frame2, num_interpolated=3)
        assert len(interpolated) == 3
        assert all(isinstance(f, Image.Image) for f in interpolated)


class TestTemporalAttention:
    """Tests for TemporalAttention module."""
    
    def test_attention_init(self):
        """Test attention module initialization."""
        attn = TemporalAttention(channels=64, num_frames=8, num_heads=8)
        assert attn.channels == 64
        assert attn.num_frames == 8
        assert attn.num_heads == 8
    
    def test_attention_forward(self):
        """Test attention forward pass."""
        attn = TemporalAttention(channels=64, num_frames=4, num_heads=8)
        x = torch.randn(1, 64, 4, 32, 32)  # batch, channels, frames, height, width
        output = attn(x)
        assert output.shape == x.shape


class TestMotionEstimator:
    """Tests for MotionEstimator."""
    
    def test_estimator_init(self):
        """Test estimator initialization."""
        estimator = MotionEstimator(in_channels=3, hidden_dim=64)
        assert estimator is not None
    
    def test_motion_estimation(self):
        """Test motion estimation."""
        estimator = MotionEstimator(in_channels=3, hidden_dim=32)
        frame1 = torch.randn(1, 3, 64, 64)
        frame2 = torch.randn(1, 3, 64, 64)
        
        flow = estimator(frame1, frame2)
        assert flow.shape == (1, 2, 64, 64)
    
    def test_warp(self):
        """Test frame warping."""
        estimator = MotionEstimator(in_channels=3, hidden_dim=32)
        frame = torch.randn(1, 3, 64, 64)
        flow = torch.zeros(1, 2, 64, 64)
        
        warped = estimator.warp(frame, flow)
        assert warped.shape == frame.shape
