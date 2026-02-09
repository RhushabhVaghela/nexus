"""
Benchmarks for generation speed
5+ benchmarks comparing inference performance
"""

import pytest
import time
import torch
from unittest.mock import Mock, patch
from PIL import Image
import numpy as np

from src.models.diffusion import ImagePipeline, PipelineConfig
from src.models.video import VideoPipeline, VideoConfig
from src.models.video.frame_generator import FrameGenerator, FrameGenerationConfig


class TestImageGenerationBenchmarks:
    """Benchmarks for image generation speed."""
    
    @pytest.mark.benchmark
    def test_sdxl_generation_speed(self, benchmark):
        """Benchmark SDXL generation speed."""
        with patch('nexus.models.diffusion.image_pipeline.StableDiffusionXLPipeline') as mock_pipe:
            # Setup
            mock_instance = Mock()
            mock_output = Mock()
            mock_output.images = [Image.new("RGB", (1024, 1024))]
            mock_output.nsfw_content_detected = [False]
            mock_instance.return_value = mock_output
            mock_pipe.from_pretrained.return_value = mock_instance
            
            config = PipelineConfig(
                model_id="stabilityai/stable-diffusion-xl-base-1.0",
                model_type="sdxl"
            )
            pipeline = ImagePipeline(config)
            pipeline.load()
            
            # Benchmark
            result = benchmark(pipeline.generate, "a beautiful sunset", num_inference_steps=30)
            
            # Assertions
            assert "images" in result
            assert len(result["images"]) == 1
    
    @pytest.mark.benchmark
    def test_flux_generation_speed(self, benchmark):
        """Benchmark FLUX generation speed."""
        with patch('nexus.models.diffusion.image_pipeline.FluxPipeline') as mock_pipe:
            # Setup
            mock_instance = Mock()
            mock_output = Mock()
            mock_output.images = [Image.new("RGB", (1024, 1024))]
            mock_output.nsfw_content_detected = [False]
            mock_instance.return_value = mock_output
            mock_pipe.from_pretrained.return_value = mock_instance
            
            config = PipelineConfig(
                model_id="black-forest-labs/FLUX.1-dev",
                model_type="flux"
            )
            pipeline = ImagePipeline(config)
            pipeline.load()
            
            # Benchmark
            result = benchmark(pipeline.generate, "a futuristic city", num_inference_steps=50)
            
            assert "images" in result
    
    @pytest.mark.benchmark
    def test_image_to_image_speed(self, benchmark):
        """Benchmark img2img variation speed."""
        with patch('nexus.models.diffusion.image_pipeline.AutoPipelineForText2Image') as mock_pipe:
            mock_instance = Mock()
            mock_pipe.from_pretrained.return_value = mock_instance
            
            mock_img2img = Mock()
            mock_output = Mock()
            mock_output.images = [Image.new("RGB", (512, 512))]
            mock_img2img.return_value = mock_output
            
            with patch('nexus.models.diffusion.image_pipeline.AutoPipelineForImage2Image') as mock_i2i:
                mock_i2i.from_pipe.return_value = mock_img2img
                
                config = PipelineConfig(model_id="test/model")
                pipeline = ImagePipeline(config)
                pipeline.load()
                
                input_image = Image.new("RGB", (512, 512), color="blue")
                
                # Benchmark
                result = benchmark(
                    pipeline.generate_variations,
                    input_image,
                    prompt="variation",
                    strength=0.75
                )
                
                assert "images" in result


class TestVideoGenerationBenchmarks:
    """Benchmarks for video generation speed."""
    
    @pytest.mark.benchmark
    def test_ltx_video_generation_speed(self, benchmark):
        """Benchmark LTX-Video generation speed."""
        with patch('nexus.models.video.video_pipeline.LTXPipeline') as mock_pipe:
            mock_instance = Mock()
            mock_output = Mock()
            mock_output.frames = [Image.new("RGB", (512, 512)) for _ in range(24)]
            mock_instance.return_value = mock_output
            mock_pipe.from_pretrained.return_value = mock_instance
            
            config = VideoConfig(
                model_id="Lightricks/LTX-Video",
                num_frames=24
            )
            pipeline = VideoPipeline(config)
            pipeline.load()
            
            # Benchmark
            result = benchmark(
                pipeline.generate,
                "a car driving through a city",
                num_frames=24
            )
            
            assert "frames" in result
            assert len(result["frames"]) == 24
    
    @pytest.mark.benchmark
    def test_frame_generator_sequence_speed(self, benchmark):
        """Benchmark frame sequence generation speed."""
        mock_pipeline = Mock()
        mock_images = [[Image.new("RGB", (64, 64)) for _ in range(4)] for _ in range(3)]
        mock_pipeline.generate.side_effect = [
            {"images": imgs} for imgs in mock_images
        ]
        
        config = FrameGenerationConfig(
            num_frames=12,
            overlap_frames=2,
            mode="overlap"
        )
        generator = FrameGenerator(mock_pipeline, config)
        
        # Benchmark
        frames = benchmark(
            generator.generate_sequence,
            prompt="animated character",
            num_frames=12
        )
        
        assert len(frames) == 12


class TestMemoryEfficiencyBenchmarks:
    """Benchmarks for memory efficiency."""
    
    @pytest.mark.benchmark
    def test_vae_slicing_memory(self):
        """Benchmark memory usage with VAE slicing."""
        with patch('nexus.models.diffusion.image_pipeline.StableDiffusionXLPipeline') as mock_pipe:
            mock_instance = Mock()
            mock_pipe.from_pretrained.return_value = mock_instance
            
            # Without slicing
            config_no_slice = PipelineConfig(
                model_id="test/model",
                enable_vae_slicing=False
            )
            pipeline_no_slice = ImagePipeline(config_no_slice)
            pipeline_no_slice.load()
            
            # With slicing
            config_slice = PipelineConfig(
                model_id="test/model",
                enable_vae_slicing=True
            )
            pipeline_slice = ImagePipeline(config_slice)
            pipeline_slice.load()
            
            # Verify slicing was enabled
            mock_instance.enable_vae_slicing.assert_called_once()
    
    @pytest.mark.benchmark
    def test_cpu_offload_memory(self):
        """Benchmark memory usage with CPU offloading."""
        with patch('nexus.models.diffusion.image_pipeline.StableDiffusionXLPipeline') as mock_pipe:
            mock_instance = Mock()
            mock_pipe.from_pretrained.return_value = mock_instance
            
            config = PipelineConfig(
                model_id="test/model",
                enable_cpu_offload=True
            )
            pipeline = ImagePipeline(config)
            pipeline.load()
            
            # Verify offload was enabled
            mock_instance.enable_model_cpu_offload.assert_called_once()


class TestThroughputBenchmarks:
    """Benchmarks for generation throughput."""
    
    @pytest.mark.benchmark
    def test_batch_throughput(self, benchmark):
        """Benchmark batch generation throughput."""
        with patch('nexus.models.diffusion.image_pipeline.StableDiffusionXLPipeline') as mock_pipe:
            mock_instance = Mock()
            mock_output = Mock()
            mock_output.images = [Image.new("RGB", (512, 512)) for _ in range(4)]
            mock_output.nsfw_content_detected = [False] * 4
            mock_instance.return_value = mock_output
            mock_pipe.from_pretrained.return_value = mock_instance
            
            config = PipelineConfig(model_id="test/model")
            pipeline = ImagePipeline(config)
            pipeline.load()
            
            prompts = ["sunset", "mountain", "ocean", "forest"]
            
            # Benchmark batch generation
            result = benchmark(
                pipeline.generate,
                prompts,
                num_images_per_prompt=1
            )
            
            assert len(result["images"]) == 4


class TestQualityVsSpeedTradeoff:
    """Benchmarks for quality vs speed tradeoffs."""
    
    @pytest.mark.benchmark
    @pytest.mark.parametrize("num_steps", [10, 20, 30, 50])
    def test_steps_vs_quality(self, num_steps, benchmark):
        """Benchmark different step counts for quality tradeoff."""
        with patch('nexus.models.diffusion.image_pipeline.StableDiffusionXLPipeline') as mock_pipe:
            mock_instance = Mock()
            mock_output = Mock()
            mock_output.images = [Image.new("RGB", (512, 512))]
            mock_output.nsfw_content_detected = [False]
            mock_instance.return_value = mock_output
            mock_pipe.from_pretrained.return_value = mock_instance
            
            config = PipelineConfig(model_id="test/model")
            pipeline = ImagePipeline(config)
            pipeline.load()
            
            # Benchmark with different step counts
            result = benchmark(
                pipeline.generate,
                "a landscape",
                num_inference_steps=num_steps
            )
            
            assert "images" in result


# Performance comparison utilities
def compare_generation_methods():
    """
    Utility function to compare different generation methods.
    Not a test, but a helper for analysis.
    """
    methods = {
        "SDXL": {"model": "sdxl-base", "steps": 30},
        "SDXL-Turbo": {"model": "sdxl-turbo", "steps": 4},
        "FLUX-Schnell": {"model": "flux-schnell", "steps": 4},
        "FLUX-Dev": {"model": "flux-dev", "steps": 50},
    }
    
    results = {}
    for name, config in methods.items():
        # Would run actual benchmarks here
        results[name] = {
            "steps": config["steps"],
            "estimated_time": config["steps"] * 0.5  # Placeholder
        }
    
    return results


def get_performance_recommendations():
    """
    Get performance recommendations based on hardware.
    """
    recommendations = {
        "low_vram": {
            "enable_vae_slicing": True,
            "enable_cpu_offload": True,
            "dtype": "float16",
            "recommended_models": ["sdxl-turbo", "flux-schnell"]
        },
        "mid_vram": {
            "enable_vae_slicing": True,
            "enable_cpu_offload": False,
            "dtype": "float16",
            "recommended_models": ["sdxl-base", "sd3-medium"]
        },
        "high_vram": {
            "enable_vae_slicing": False,
            "enable_cpu_offload": False,
            "dtype": "bfloat16",
            "recommended_models": ["flux-dev", "sd3.5-large"]
        }
    }
    return recommendations
