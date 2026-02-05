"""
Unified Pipeline Loader for Diffusion Models

Provides a single entry point for loading all supported diffusion pipelines
with automatic model type detection and optimized configuration.
"""

import torch
from typing import Optional, Dict, Any, Union, List
from pathlib import Path
import logging

from .image_pipeline import ImagePipeline, PipelineConfig

logger = logging.getLogger(__name__)


class DiffusionPipelineLoader:
    """
    Unified loader for diffusion models.
    
    Features:
    - Automatic model type detection
    - Memory-efficient loading with quantization options
    - Batch loading of multiple models
    - Model caching and reuse
    """
    
    # Pre-configured model presets
    PRESETS = {
        # Stable Diffusion variants
        "sd-2-1": {
            "model_id": "stabilityai/stable-diffusion-2-1",
            "model_type": "sd",
            "default_steps": 50,
            "default_height": 768,
            "default_width": 768,
        },
        "sdxl-base": {
            "model_id": "stabilityai/stable-diffusion-xl-base-1.0",
            "model_type": "sdxl",
            "default_steps": 30,
            "default_height": 1024,
            "default_width": 1024,
        },
        "sdxl-turbo": {
            "model_id": "stabilityai/sdxl-turbo",
            "model_type": "sdxl",
            "default_steps": 4,
            "default_guidance_scale": 0.0,
            "default_height": 512,
            "default_width": 512,
        },
        "sd3-medium": {
            "model_id": "stabilityai/stable-diffusion-3-medium-diffusers",
            "model_type": "sd3",
            "default_steps": 28,
            "default_guidance_scale": 5.0,
            "default_height": 1024,
            "default_width": 1024,
        },
        "sd3.5-medium": {
            "model_id": "stabilityai/stable-diffusion-3.5-medium",
            "model_type": "sd3",
            "default_steps": 28,
            "default_guidance_scale": 5.0,
        },
        "sd3.5-large": {
            "model_id": "stabilityai/stable-diffusion-3.5-large",
            "model_type": "sd3",
            "default_steps": 28,
            "default_guidance_scale": 5.0,
        },
        
        # FLUX variants
        "flux-dev": {
            "model_id": "black-forest-labs/FLUX.1-dev",
            "model_type": "flux",
            "default_steps": 50,
            "default_guidance_scale": 3.5,
            "default_height": 1024,
            "default_width": 1024,
        },
        "flux-schnell": {
            "model_id": "black-forest-labs/FLUX.1-schnell",
            "model_type": "flux",
            "default_steps": 4,
            "default_guidance_scale": 0.0,
            "default_height": 1024,
            "default_width": 1024,
        },
        "flux-fill": {
            "model_id": "black-forest-labs/FLUX.1-fill-dev",
            "model_type": "flux-fill",
            "default_steps": 50,
            "default_guidance_scale": 3.5,
        },
        
        # Z-Image variants
        "z-image": {
            "model_id": "stabilityai/z-image",
            "model_type": "z-image",
            "default_steps": 30,
            "default_height": 1024,
            "default_width": 1024,
        },
        "z-image-turbo": {
            "model_id": "stabilityai/z-image-turbo",
            "model_type": "z-image-turbo",
            "default_steps": 4,
            "default_guidance_scale": 1.0,
        },
        
        # Hunyuan variants
        "hunyuan-dit": {
            "model_id": "Tencent-Hunyuan/HunyuanDiT-v1.2",
            "model_type": "hunyuan",
            "default_steps": 50,
            "default_height": 1024,
            "default_width": 1024,
        },
    }
    
    def __init__(
        self,
        cache_dir: Optional[str] = None,
        device: str = "auto",
        default_dtype: torch.dtype = torch.float16,
    ):
        self.cache_dir = cache_dir
        self.device = device
        self.default_dtype = default_dtype
        self._loaded_pipelines: Dict[str, ImagePipeline] = {}
    
    @classmethod
    def list_presets(cls) -> List[str]:
        """List available model presets."""
        return list(cls.PRESETS.keys())
    
    @classmethod
    def get_preset_info(cls, preset_name: str) -> Dict[str, Any]:
        """Get information about a preset."""
        if preset_name not in cls.PRESETS:
            raise ValueError(f"Unknown preset: {preset_name}. Available: {list(cls.PRESETS.keys())}")
        return cls.PRESETS[preset_name].copy()
    
    def load(
        self,
        model_id_or_preset: str,
        model_type: str = "auto",
        **override_kwargs
    ) -> ImagePipeline:
        """
        Load a diffusion pipeline.
        
        Args:
            model_id_or_preset: Model ID (e.g., 'stabilityai/sdxl-base') or preset name
            model_type: Model type override ('auto' for detection)
            **override_kwargs: Override any PipelineConfig parameters
        
        Returns:
            Loaded ImagePipeline instance
        """
        # Check if it's a preset
        preset_config = {}
        if model_id_or_preset in self.PRESETS:
            preset_config = self.PRESETS[model_id_or_preset].copy()
            model_id = preset_config.pop("model_id")
        else:
            model_id = model_id_or_preset
        
        # Build config
        config_dict = {
            "model_id": model_id,
            "model_type": model_type if model_type != "auto" else preset_config.get("model_type", "auto"),
            "device": self.device,
            "dtype": self.default_dtype,
            "cache_dir": self.cache_dir,
        }
        
        # Apply preset defaults
        for key in ["default_steps", "default_guidance_scale", "default_height", "default_width"]:
            if key in preset_config:
                config_dict[key] = preset_config[key]
        
        # Apply overrides
        config_dict.update(override_kwargs)
        
        # Create and load pipeline
        config = PipelineConfig(**config_dict)
        pipeline = ImagePipeline(config)
        pipeline.load()
        
        # Cache if named
        cache_key = override_kwargs.get("cache_key")
        if cache_key:
            self._loaded_pipelines[cache_key] = pipeline
        
        return pipeline
    
    def load_quantized(
        self,
        model_id_or_preset: str,
        quantization: str = "fp8",
        **kwargs
    ) -> ImagePipeline:
        """
        Load a quantized diffusion pipeline for memory efficiency.
        
        Args:
            model_id_or_preset: Model ID or preset name
            quantization: Quantization type ('fp8', 'int8', 'int4')
            **kwargs: Additional load arguments
        """
        logger.info(f"Loading {model_id_or_preset} with {quantization} quantization")
        
        # Set appropriate dtype based on quantization
        if quantization == "fp8":
            # FP8 requires specific hardware support (Hopper/Ada)
            dtype = torch.float8_e4m3fn if hasattr(torch, 'float8_e4m3fn') else torch.float16
        elif quantization in ["int8", "int4"]:
            # Use bitsandbytes for int8/int4
            dtype = torch.float16
            kwargs["load_in_8bit"] = quantization == "int8"
            kwargs["load_in_4bit"] = quantization == "int4"
        else:
            dtype = self.default_dtype
        
        kwargs["dtype"] = dtype
        kwargs["quantization"] = quantization
        
        return self.load(model_id_or_preset, **kwargs)
    
    def get_cached(self, cache_key: str) -> Optional[ImagePipeline]:
        """Get a cached pipeline by key."""
        return self._loaded_pipelines.get(cache_key)
    
    def unload(self, cache_key: str):
        """Unload a cached pipeline."""
        if cache_key in self._loaded_pipelines:
            self._loaded_pipelines[cache_key].unload()
            del self._loaded_pipelines[cache_key]
            logger.info(f"Unloaded pipeline: {cache_key}")
    
    def unload_all(self):
        """Unload all cached pipelines."""
        for key, pipeline in list(self._loaded_pipelines.items()):
            pipeline.unload()
        self._loaded_pipelines.clear()
        logger.info("All pipelines unloaded")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - unload all pipelines."""
        self.unload_all()


def quick_load(
    model_id_or_preset: str,
    device: str = "auto",
    **kwargs
) -> ImagePipeline:
    """
    Quick load a diffusion pipeline without creating a loader instance.
    
    Args:
        model_id_or_preset: Model ID or preset name
        device: Device to load on
        **kwargs: Additional arguments for PipelineConfig
    
    Returns:
        Loaded ImagePipeline
    
    Example:
        >>> pipeline = quick_load("sdxl-base")
        >>> result = pipeline.generate("a cat in a hat")
    """
    loader = DiffusionPipelineLoader(device=device)
    return loader.load(model_id_or_preset, **kwargs)
