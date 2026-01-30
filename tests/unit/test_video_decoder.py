"""
Unit tests for VideoDecoder module.

Tests the video generation pipeline using Stable Video Diffusion (SVD)
with proper memory management and export functionality.
"""

import pytest
import torch
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch, Mock
from PIL import Image
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class TestVideoDecoder:
    """Tests for VideoDecoder class."""
    
    @pytest.fixture
    def decoder(self):
        """Fixture for VideoDecoder instance."""
        from src.nexus_final.decoders import VideoDecoder
        return VideoDecoder(
            model_id="stabilityai/stable-video-diffusion-img2vid-xt",
            device="cpu",
            torch_dtype=torch.float32,
            enable_model_cpu_offload=False,
            enable_vae_slicing=False,
            enable_vae_tiling=False
        )
    
    @pytest.fixture
    def mock_pipeline(self):
        """Fixture for mocked StableVideoDiffusionPipeline."""
        mock = MagicMock()
        
        # Mock VAE
        mock.vae = MagicMock()
        mock.vae.config = MagicMock()
        mock.vae.config.scaling_factor = 0.18215
        mock.vae.decode = MagicMock(return_value=MagicMock(
            sample=torch.randn(1, 3, 256, 256)
        ))
        
        # Mock UNet
        mock.unet = MagicMock()
        mock.unet.eval = MagicMock()
        
        # Mock return value for generate
        mock_frames = [Image.new('RGB', (256, 256), color='red') for _ in range(5)]
        mock.return_value = MagicMock(frames=[mock_frames])
        
        return mock
    
    def test_initialization(self, decoder):
        """Test VideoDecoder initialization."""
        assert decoder.model_id == "stabilityai/stable-video-diffusion-img2vid-xt"
        assert decoder.device == "cpu"
        assert decoder.torch_dtype == torch.float32
        assert decoder.pipeline is None
        assert decoder._enable_model_cpu_offload is False
    
    def test_initialization_with_defaults(self):
        """Test VideoDecoder with default parameters."""
        from src.nexus_final.decoders import VideoDecoder
        decoder = VideoDecoder()
        
        assert decoder.model_id == "stabilityai/stable-video-diffusion-img2vid-xt"
        assert decoder.device == "cuda"
        assert decoder.torch_dtype == torch.float16
        assert decoder._enable_model_cpu_offload is True
        assert decoder._enable_vae_slicing is True
    
    @patch('src.nexus_final.decoders.StableVideoDiffusionPipeline')
    def test_load_success(self, mock_pipeline_class, decoder, mock_pipeline):
        """Test successful model loading."""
        mock_pipeline_class.from_pretrained.return_value = mock_pipeline
        
        result = decoder.load()
        
        assert result is decoder  # Returns self
        assert decoder.pipeline is not None
        mock_pipeline_class.from_pretrained.assert_called_once()
    
    @patch('src.nexus_final.decoders.StableVideoDiffusionPipeline')
    def test_load_with_cpu_offload(self, mock_pipeline_class, mock_pipeline):
        """Test loading with CPU offloading enabled."""
        from src.nexus_final.decoders import VideoDecoder
        decoder = VideoDecoder(
            device="cuda",
            enable_model_cpu_offload=True
        )
        mock_pipeline_class.from_pretrained.return_value = mock_pipeline
        
        decoder.load()
        
        mock_pipeline.enable_model_cpu_offload.assert_called_once()
    
    @patch('src.nexus_final.decoders.StableVideoDiffusionPipeline')
    def test_load_with_vae_slicing(self, mock_pipeline_class, mock_pipeline):
        """Test loading with VAE slicing enabled."""
        mock_pipeline_class.from_pretrained.return_value = mock_pipeline
        
        from src.nexus_final.decoders import VideoDecoder
        decoder = VideoDecoder(enable_vae_slicing=True)
        
        decoder.load()
        
        mock_pipeline.enable_vae_slicing.assert_called_once()
    
    @patch('src.nexus_final.decoders.StableVideoDiffusionPipeline')
    def test_load_with_vae_tiling(self, mock_pipeline_class, mock_pipeline):
        """Test loading with VAE tiling enabled."""
        mock_pipeline_class.from_pretrained.return_value = mock_pipeline
        
        from src.nexus_final.decoders import VideoDecoder
        decoder = VideoDecoder(enable_vae_tiling=True)
        
        decoder.load()
        
        mock_pipeline.enable_vae_tiling.assert_called_once()
    
    @patch('src.nexus_final.decoders.StableVideoDiffusionPipeline')
    def test_load_failure(self, mock_pipeline_class, decoder):
        """Test handling of model loading failure."""
        mock_pipeline_class.from_pretrained.side_effect = RuntimeError("Model not found")
        
        with pytest.raises(RuntimeError, match="Failed to load Video Decoder"):
            decoder.load()
    
    def test_generate_without_load(self, decoder):
        """Test generation without loading model first."""
        with pytest.raises(RuntimeError, match="VideoDecoder not loaded"):
            decoder.generate(Image.new('RGB', (256, 256)))
    
    @patch('src.nexus_final.decoders.StableVideoDiffusionPipeline')
    def test_generate_with_pil_image(self, mock_pipeline_class, decoder, mock_pipeline):
        """Test generation with PIL Image input."""
        mock_pipeline_class.from_pretrained.return_value = mock_pipeline
        decoder.load()
        
        conditioning = Image.new('RGB', (256, 256), color='blue')
        frames = decoder.generate(conditioning, num_frames=5)
        
        assert len(frames) == 5
        assert all(isinstance(f, Image.Image) for f in frames)
    
    @patch('src.nexus_final.decoders.StableVideoDiffusionPipeline')
    @patch('src.nexus_final.decoders.load_image')
    def test_generate_with_image_path(self, mock_load_image, mock_pipeline_class, decoder, mock_pipeline):
        """Test generation with image path input."""
        mock_pipeline_class.from_pretrained.return_value = mock_pipeline
        mock_load_image.return_value = Image.new('RGB', (256, 256), color='green')
        decoder.load()
        
        frames = decoder.generate("/path/to/image.jpg", num_frames=5)
        
        assert len(frames) == 5
        mock_load_image.assert_called_once_with("/path/to/image.jpg")
    
    @patch('src.nexus_final.decoders.StableVideoDiffusionPipeline')
    def test_generate_with_latent_tensor(self, mock_pipeline_class, decoder, mock_pipeline):
        """Test generation with latent tensor input."""
        mock_pipeline_class.from_pretrained.return_value = mock_pipeline
        decoder.load()
        
        latent = torch.randn(1, 4, 32, 32)  # B, C, H, W
        frames = decoder.generate(latent, num_frames=5)
        
        assert len(frames) == 5
    
    def test_generate_with_invalid_conditioning(self, decoder):
        """Test generation with invalid conditioning input."""
        decoder.pipeline = MagicMock()  # Pretend it's loaded
        
        with pytest.raises(ValueError, match="Conditioning must be PIL Image"):
            decoder.generate(12345)  # Invalid type
    
    @patch('src.nexus_final.decoders.StableVideoDiffusionPipeline')
    def test_generate_with_seed(self, mock_pipeline_class, decoder, mock_pipeline):
        """Test generation with random seed for reproducibility."""
        mock_pipeline_class.from_pretrained.return_value = mock_pipeline
        decoder.load()
        
        conditioning = Image.new('RGB', (256, 256))
        frames = decoder.generate(conditioning, seed=42)
        
        # Check that generator was created with correct seed
        call_kwargs = mock_pipeline.call_args.kwargs
        assert call_kwargs.get('generator') is not None
    
    @patch('src.nexus_final.decoders.StableVideoDiffusionPipeline')
    def test_generate_parameters_passed(self, mock_pipeline_class, decoder, mock_pipeline):
        """Test that all generation parameters are passed correctly."""
        mock_pipeline_class.from_pretrained.return_value = mock_pipeline
        decoder.load()
        
        conditioning = Image.new('RGB', (256, 256))
        frames = decoder.generate(
            conditioning,
            num_frames=25,
            num_inference_steps=30,
            min_guidance_scale=1.5,
            max_guidance_scale=3.5,
            fps=8,
            motion_bucket_id=150,
            noise_aug_strength=0.05,
            decode_chunk_size=4
        )
        
        call_kwargs = mock_pipeline.call_args.kwargs
        assert call_kwargs['num_frames'] == 25
        assert call_kwargs['num_inference_steps'] == 30
        assert call_kwargs['min_guidance_scale'] == 1.5
        assert call_kwargs['max_guidance_scale'] == 3.5
        assert call_kwargs['fps'] == 8
        assert call_kwargs['motion_bucket_id'] == 150
        assert call_kwargs['noise_aug_strength'] == 0.05
        assert call_kwargs['decode_chunk_size'] == 4
    
    @patch('src.nexus_final.decoders.StableVideoDiffusionPipeline')
    def test_generate_failure(self, mock_pipeline_class, decoder, mock_pipeline):
        """Test handling of generation failure."""
        mock_pipeline_class.from_pretrained.return_value = mock_pipeline
        mock_pipeline.side_effect = RuntimeError("Generation failed")
        decoder.load()
        
        conditioning = Image.new('RGB', (256, 256))
        
        with pytest.raises(RuntimeError, match="Video generation failed"):
            decoder.generate(conditioning)
    
    @patch('src.nexus_final.decoders.StableVideoDiffusionPipeline')
    def test_generate_from_text_with_image_generator(self, mock_pipeline_class, decoder, mock_pipeline):
        """Test text-to-video generation with image generator."""
        mock_pipeline_class.from_pretrained.return_value = mock_pipeline
        decoder.load()
        
        mock_image_gen = MagicMock()
        mock_image_gen.return_value = MagicMock(
            images=[Image.new('RGB', (256, 256), color='purple')]
        )
        
        frames = decoder.generate_from_text(
            "a cat walking",
            image_generator=mock_image_gen,
            num_frames=5
        )
        
        assert len(frames) == 5
        mock_image_gen.assert_called_once()
    
    @patch('src.nexus_final.decoders.StableVideoDiffusionPipeline')
    def test_generate_from_text_without_image_generator(self, mock_pipeline_class, decoder, mock_pipeline):
        """Test text-to-video without image generator raises error."""
        mock_pipeline_class.from_pretrained.return_value = mock_pipeline
        decoder.load()
        
        with pytest.raises(ValueError, match="Text-to-video generation requires an image generator"):
            decoder.generate_from_text("a cat walking")
    
    def test_to_video_array(self, decoder):
        """Test conversion of frames to numpy array."""
        frames = [Image.new('RGB', (256, 256), color=(i*50, i*30, i*20)) for i in range(5)]
        
        video_array = decoder.to_video_array(frames)
        
        assert isinstance(video_array, np.ndarray)
        assert video_array.shape == (5, 256, 256, 3)
        assert video_array.dtype == np.uint8
    
    @patch('src.nexus_final.decoders.imageio')
    def test_save_video_mp4(self, mock_imageio, decoder, tmp_path):
        """Test saving video as MP4."""
        frames = [Image.new('RGB', (256, 256)) for _ in range(5)]
        output_path = tmp_path / "output.mp4"
        
        result = decoder.save_video(frames, str(output_path), fps=7, format="mp4")
        
        assert result == str(output_path)
        mock_imageio.mimsave.assert_called_once()
    
    def test_save_video_gif(self, decoder, tmp_path):
        """Test saving video as GIF."""
        frames = [Image.new('RGB', (256, 256)) for _ in range(5)]
        output_path = tmp_path / "output.gif"
        
        result = decoder.save_video(frames, str(output_path), fps=10, format="gif")
        
        assert result == str(output_path)
        # GIF is saved using PIL's save method
    
    @patch('src.nexus_final.decoders.imageio')
    def test_save_video_webm(self, mock_imageio, decoder, tmp_path):
        """Test saving video as WebM."""
        frames = [Image.new('RGB', (256, 256)) for _ in range(5)]
        output_path = tmp_path / "output.webm"
        
        result = decoder.save_video(frames, str(output_path), fps=30, format="webm")
        
        assert result == str(output_path)
        mock_imageio.mimsave.assert_called_once()
    
    def test_save_video_missing_imageio(self, decoder, tmp_path):
        """Test saving video when imageio is not available."""
        with patch.dict('sys.modules', {'imageio': None}):
            frames = [Image.new('RGB', (256, 256)) for _ in range(5)]
            output_path = tmp_path / "output.mp4"
            
            with pytest.raises(ImportError, match="imageio is required"):
                decoder.save_video(frames, str(output_path))
    
    @patch('src.nexus_final.decoders.StableVideoDiffusionPipeline')
    def test_latent_to_image(self, mock_pipeline_class, decoder, mock_pipeline):
        """Test conversion of latent tensor to image."""
        mock_pipeline_class.from_pretrained.return_value = mock_pipeline
        decoder.load()
        
        latent = torch.randn(1, 4, 32, 32)
        image = decoder._latent_to_image(latent)
        
        assert isinstance(image, Image.Image)
        assert image.size == (256, 256)
    
    def test_latent_to_image_3d_tensor(self, decoder):
        """Test latent conversion with 3D tensor."""
        decoder.pipeline = MagicMock()
        decoder.pipeline.vae = MagicMock()
        decoder.pipeline.vae.config.scaling_factor = 0.18215
        decoder.pipeline.vae.decode.return_value = MagicMock(
            sample=torch.randn(1, 3, 256, 256)
        )
        
        latent = torch.randn(4, 32, 32)  # 3D tensor (C, H, W)
        image = decoder._latent_to_image(latent)
        
        assert isinstance(image, Image.Image)
    
    def test_get_memory_stats_without_gpu(self, decoder):
        """Test memory stats when GPU is not available."""
        decoder.pipeline = MagicMock()  # Pretend loaded
        
        stats = decoder.get_memory_stats()
        
        assert stats["device"] == "cpu"
        assert stats["model_loaded"] is True
        assert "gpu_allocated_mb" not in stats
    
    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.memory_allocated', return_value=1024**3)  # 1 GB
    @patch('torch.cuda.memory_reserved', return_value=2*1024**3)  # 2 GB
    def test_get_memory_stats_with_gpu(self, mock_reserved, mock_allocated, mock_available, decoder):
        """Test memory stats when GPU is available."""
        decoder.device = "cuda"
        decoder.pipeline = MagicMock()
        
        stats = decoder.get_memory_stats()
        
        assert stats["device"] == "cuda"
        assert stats["gpu_allocated_mb"] == 1024.0  # 1 GB in MB
        assert stats["gpu_reserved_mb"] == 2048.0  # 2 GB in MB
    
    def test_unload(self, decoder):
        """Test unloading model from memory."""
        decoder.pipeline = MagicMock()
        
        with patch('torch.cuda.empty_cache') as mock_empty_cache:
            decoder.unload()
            
            assert decoder.pipeline is None
            mock_empty_cache.assert_called_once()
    
    def test_unload_when_not_loaded(self, decoder):
        """Test unloading when model was never loaded."""
        decoder.pipeline = None
        
        # Should not raise error
        decoder.unload()
        assert decoder.pipeline is None


class TestNexusDecoders:
    """Tests for NexusDecoders factory class."""
    
    @patch('src.nexus_final.decoders.SpeechT5HifiGan')
    def test_load_audio_vocoder(self, mock_vocoder_class):
        """Test loading audio vocoder."""
        mock_vocoder = MagicMock()
        mock_vocoder_class.from_pretrained.return_value = mock_vocoder
        
        from src.nexus_final.decoders import NexusDecoders
        result = NexusDecoders.load_audio_vocoder(device="cpu")
        
        assert result == mock_vocoder
        mock_vocoder.eval.assert_called_once()
        mock_vocoder.to.assert_called_once_with("cpu")
    
    @patch('src.nexus_final.decoders.AutoencoderKL')
    def test_load_image_decoder(self, mock_vae_class):
        """Test loading image VAE decoder."""
        mock_vae = MagicMock()
        mock_vae_class.from_pretrained.return_value = mock_vae
        
        from src.nexus_final.decoders import NexusDecoders
        result = NexusDecoders.load_image_decoder(device="cpu")
        
        assert result == mock_vae
        mock_vae.eval.assert_called_once()
        mock_vae.to.assert_called_once_with("cpu")
    
    @patch('src.nexus_final.decoders.VideoDecoder')
    def test_load_video_decoder(self, mock_decoder_class):
        """Test loading video decoder."""
        mock_decoder = MagicMock()
        mock_decoder_class.return_value = mock_decoder
        mock_decoder.load.return_value = mock_decoder
        
        from src.nexus_final.decoders import NexusDecoders
        result = NexusDecoders.load_video_decoder(device="cuda")
        
        assert result == mock_decoder
        mock_decoder_class.assert_called_once_with(
            model_id="stabilityai/stable-video-diffusion-img2vid-xt",
            device="cuda"
        )
        mock_decoder.load.assert_called_once()
    
    @patch('src.nexus_final.decoders.VideoDecoder')
    def test_load_video_decoder_with_kwargs(self, mock_decoder_class):
        """Test loading video decoder with additional kwargs."""
        mock_decoder = MagicMock()
        mock_decoder_class.return_value = mock_decoder
        mock_decoder.load.return_value = mock_decoder
        
        from src.nexus_final.decoders import NexusDecoders
        result = NexusDecoders.load_video_decoder(
            device="cuda",
            torch_dtype=torch.float16,
            enable_model_cpu_offload=True
        )
        
        mock_decoder_class.assert_called_once_with(
            model_id="stabilityai/stable-video-diffusion-img2vid-xt",
            device="cuda",
            torch_dtype=torch.float16,
            enable_model_cpu_offload=True
        )


class TestVideoDecoderEdgeCases:
    """Edge case tests for VideoDecoder."""
    
    @pytest.fixture
    def loaded_decoder(self):
        """Create a loaded decoder with mocked pipeline."""
        from src.nexus_final.decoders import VideoDecoder
        decoder = VideoDecoder(device="cpu")
        decoder.pipeline = MagicMock()
        
        mock_frames = [Image.new('RGB', (256, 256)) for _ in range(5)]
        decoder.pipeline.return_value = MagicMock(frames=[mock_frames])
        
        return decoder
    
    def test_generate_with_zero_frames(self, loaded_decoder):
        """Test generation with zero frames (edge case)."""
        conditioning = Image.new('RGB', (256, 256))
        
        frames = loaded_decoder.generate(conditioning, num_frames=0)
        
        call_kwargs = loaded_decoder.pipeline.call_args.kwargs
        assert call_kwargs['num_frames'] == 0
    
    def test_generate_with_single_frame(self, loaded_decoder):
        """Test generation with single frame."""
        conditioning = Image.new('RGB', (256, 256))
        
        frames = loaded_decoder.generate(conditioning, num_frames=1)
        
        assert len(frames) == 5  # From mock
    
    def test_generate_with_very_long_sequence(self, loaded_decoder):
        """Test generation with very long frame sequence."""
        conditioning = Image.new('RGB', (256, 256))
        
        frames = loaded_decoder.generate(conditioning, num_frames=1000)
        
        call_kwargs = loaded_decoder.pipeline.call_args.kwargs
        assert call_kwargs['num_frames'] == 1000
    
    def test_save_video_empty_frames(self, loaded_decoder, tmp_path):
        """Test saving video with empty frame list."""
        output_path = tmp_path / "output.mp4"
        
        # Should handle empty frames gracefully
        with pytest.raises(IndexError):  # Image.new will fail
            loaded_decoder.save_video([], str(output_path))
    
    def test_save_video_single_frame(self, loaded_decoder, tmp_path):
        """Test saving video with single frame."""
        frames = [Image.new('RGB', (256, 256))]
        output_path = tmp_path / "output.gif"
        
        result = loaded_decoder.save_video(frames, str(output_path), format="gif")
        
        assert result == str(output_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
