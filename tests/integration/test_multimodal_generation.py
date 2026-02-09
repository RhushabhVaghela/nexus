"""
Integration tests for multimodal generation
8+ tests covering end-to-end workflows
"""

import pytest
import torch
from unittest.mock import Mock, patch, MagicMock
from PIL import Image
import numpy as np

from src.models.diffusion import ImagePipeline, PipelineConfig
from src.models.video import VideoPipeline, VideoConfig
from src.models.diffusion.adapter import DiffusionAdapter
from src.models.video.frame_generator import FrameGenerator
from src.models.video.temporal_consistency import TemporalConsistencyProcessor


@pytest.mark.integration
def test_image_generation_workflow():
    """Test complete image generation workflow."""
    with patch('nexus.models.diffusion.image_pipeline.AutoPipelineForText2Image') as mock_pipe_class:
        # Setup mock
        mock_pipe = Mock()
        mock_pipe_class.from_pretrained.return_value = mock_pipe
        mock_pipe.to.return_value = mock_pipe
        
        # Mock generation output
        mock_image = Image.new("RGB", (1024, 1024), color="blue")
        mock_output = Mock()
        mock_output.images = [mock_image]
        mock_output.nsfw_content_detected = [False]
        mock_pipe.return_value = mock_output
        
        # Create pipeline
        config = PipelineConfig(
            model_id="stabilityai/stable-diffusion-xl-base-1.0",
            model_type="sdxl",
            enable_vae_slicing=False,
            enable_cpu_offload=False
        )
        
        with ImagePipeline(config) as pipeline:
            result = pipeline.generate(
                prompt="a beautiful landscape",
                num_images_per_prompt=1
            )
            
            assert "images" in result
            assert len(result["images"]) == 1


@pytest.mark.integration
def test_video_generation_workflow():
    """Test complete video generation workflow."""
    with patch('nexus.models.video.video_pipeline.LTXPipeline') as mock_pipe_class:
        # Setup mock
        mock_pipe = Mock()
        mock_pipe_class.from_pretrained.return_value = mock_pipe
        
        # Mock generation output
        mock_frames = [Image.new("RGB", (512, 512), color="red") for _ in range(8)]
        mock_output = Mock()
        mock_output.frames = mock_frames
        mock_pipe.return_value = mock_output
        
        # Create pipeline
        config = VideoConfig(
            model_id="Lightricks/LTX-Video",
            num_frames=8,
            enable_vae_slicing=False,
            enable_cpu_offload=False
        )
        
        with VideoPipeline(config) as pipeline:
            result = pipeline.generate(
                prompt="a car driving through a city",
                num_frames=8
            )
            
            assert "frames" in result
            assert result["num_frames"] == 8


@pytest.mark.integration
def test_image_to_video_workflow():
    """Test image-to-video workflow."""
    with patch('nexus.models.video.video_pipeline.StableVideoDiffusionPipeline') as mock_pipe_class:
        mock_pipe = Mock()
        mock_pipe_class.from_pretrained.return_value = mock_pipe
        
        mock_frames = [Image.new("RGB", (512, 512)) for _ in range(5)]
        mock_output = Mock()
        mock_output.frames = mock_frames
        mock_pipe.return_value = mock_output
        
        config = VideoConfig(
            model_id="stabilityai/stable-video-diffusion-img2vid",
            model_type="svd"
        )
        
        with VideoPipeline(config) as pipeline:
            input_image = Image.new("RGB", (512, 512), color="green")
            result = pipeline.generate_from_image(
                image=input_image,
                num_frames=5
            )
            
            assert "frames" in result
            assert len(result["frames"]) == 5


@pytest.mark.integration
def test_diffusion_adapter_integration():
    """Test DiffusionAdapter with actual pipeline."""
    with patch('nexus.models.diffusion.image_pipeline.AutoPipelineForText2Image') as mock_pipe_class:
        # Setup mock pipeline
        mock_pipe = Mock()
        mock_pipe_class.from_pretrained.return_value = mock_pipe
        mock_pipe.to.return_value = mock_pipe
        
        mock_image = Image.new("RGB", (512, 512))
        mock_output = Mock()
        mock_output.images = [mock_image]
        mock_output.nsfw_content_detected = [False]
        mock_pipe.return_value = mock_output
        
        # Create adapter
        adapter = DiffusionAdapter(
            teacher_dim=1024,
            student_dim=512,
            extract_features_from="unet"
        )
        
        # Load pipeline through adapter
        config = PipelineConfig(model_id="test/model")
        image_pipeline = ImagePipeline(config)
        image_pipeline.load()
        
        adapter.attach_pipeline(image_pipeline)
        
        # Verify adapter setup
        assert adapter.pipeline is not None
        assert adapter.teacher_dim == 1024


@pytest.mark.integration
def test_frame_generator_with_pipeline():
    """Test FrameGenerator integration with pipeline."""
    mock_pipeline = Mock()
    mock_images = [Image.new("RGB", (64, 64)) for _ in range(4)]
    mock_pipeline.generate.return_value = {"images": mock_images}
    
    from src.models.video.frame_generator import FrameGenerationConfig
    
    config = FrameGenerationConfig(
        num_frames=4,
        overlap_frames=1,
        mode="overlap"
    )
    
    generator = FrameGenerator(mock_pipeline, config)
    
    frames = generator.generate_sequence(
        prompt="animation test",
        num_frames=4
    )
    
    assert len(frames) == 4


@pytest.mark.integration
def test_temporal_consistency_with_frames():
    """Test TemporalConsistencyProcessor with frame sequences."""
    processor = TemporalConsistencyProcessor(
        consistency_weight=0.8,
        use_optical_flow=False
    )
    
    # Create test frame sequence
    frames = [
        Image.new("RGB", (64, 64), color=(i*50, i*50, i*50))
        for i in range(5)
    ]
    
    # Process frames
    smoothed = processor.process_sequence(frames, mode="smooth")
    
    assert len(smoothed) == len(frames)
    assert all(isinstance(f, Image.Image) for f in smoothed)


@pytest.mark.integration
def test_end_to_end_multimodal_pipeline():
    """Test end-to-end multimodal generation pipeline."""
    # This test simulates a complete workflow:
    # Text -> Image -> Video
    
    with patch('nexus.models.diffusion.image_pipeline.AutoPipelineForText2Image') as mock_img_pipe, \
         patch('nexus.models.video.video_pipeline.LTXPipeline') as mock_vid_pipe:
        
        # Setup image pipeline mock
        mock_image_output = Mock()
        mock_image = Image.new("RGB", (512, 512), color="purple")
        mock_image_output.images = [mock_image]
        mock_image_output.nsfw_content_detected = [False]
        
        mock_img_instance = Mock()
        mock_img_instance.to.return_value = mock_img_instance
        mock_img_instance.return_value = mock_image_output
        mock_img_pipe.from_pretrained.return_value = mock_img_instance
        
        # Setup video pipeline mock
        mock_frames = [Image.new("RGB", (512, 512)) for _ in range(4)]
        mock_vid_output = Mock()
        mock_vid_output.frames = mock_frames
        
        mock_vid_instance = Mock()
        mock_vid_instance.return_value = mock_vid_output
        mock_vid_pipe.from_pretrained.return_value = mock_vid_instance
        
        # Step 1: Generate image
        img_config = PipelineConfig(
            model_id="test/image-model",
            enable_vae_slicing=False
        )
        
        with ImagePipeline(img_config) as img_pipeline:
            img_result = img_pipeline.generate("a scenic mountain")
            generated_image = img_result["images"][0]
        
        # Step 2: Generate video from image
        vid_config = VideoConfig(
            model_id="test/video-model",
            num_frames=4
        )
        
        with VideoPipeline(vid_config) as vid_pipeline:
            vid_result = vid_pipeline.generate_from_image(
                image=generated_image,
                prompt="mountain landscape with moving clouds"
            )
        
        assert len(vid_result["frames"]) == 4


@pytest.mark.integration
def test_registry_with_unknown_model():
    """Test registry auto-detection with unknown models."""
    from src.core.towers.registry import detect_architecture, register_unknown_model
    
    # Test detection of unknown model types
    assert detect_architecture("some/vision-model") == "vision"
    assert detect_architecture("some/audio-model") == "audio"
    assert detect_architecture("some/llama-custom") == "causal"
    
    # Test registration
    info = register_unknown_model(
        "my_custom_model",
        "custom/model-id",
        model_type="auto",
        tags=["custom", "experimental"]
    )
    
    assert info["auto_detected"] is True
    assert "auto-registered" in info["tags"]


@pytest.mark.integration
def test_error_handling_and_recovery():
    """Test error handling and graceful recovery."""
    config = PipelineConfig(model_id="nonexistent/model")
    pipeline = ImagePipeline(config)
    
    # Test that pipeline handles errors gracefully
    with pytest.raises(Exception):
        with patch('nexus.models.diffusion.image_pipeline.AutoPipelineForText2Image') as mock_pipe:
            mock_pipe.from_pretrained.side_effect = Exception("Model not found")
            pipeline.load()
    
    # Verify cleanup
    assert pipeline.pipeline is None
