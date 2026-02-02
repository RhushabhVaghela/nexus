"""
Image Pipeline for Stable Diffusion, SDXL, FLUX, Z-Image, and HunyuanImage
"""

import torch
from typing import Optional, Union, List, Dict, Any
from dataclasses import dataclass
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for image generation pipelines."""
    model_id: str
    model_type: str = "sdxl"  # sd, sdxl, flux, z-image, z-image-turbo, hunyuan
    device: str = "auto"
    dtype: torch.dtype = torch.float16
    enable_vae_slicing: bool = True
    enable_vae_tiling: bool = False
    enable_cpu_offload: bool = False
    use_karras_sigmas: bool = True
    safety_checker: bool = False
    cache_dir: Optional[str] = None
    local_files_only: bool = False
    
    # Generation parameters
    default_steps: int = 30
    default_guidance_scale: float = 7.5
    default_height: int = 1024
    default_width: int = 1024
    
    # Memory optimization
    quantization: Optional[str] = None  # "fp8", "int8", None


class ImagePipeline:
    """
    Unified image generation pipeline supporting multiple diffusion models.
    
    Supported Models:
    - Stable Diffusion 1.5/2.1
    - Stable Diffusion XL (SDXL)
    - Stable Diffusion 3
    - FLUX (FLUX.1-dev, FLUX.1-schnell)
    - Z-Image, Z-Image-Turbo
    - HunyuanImage
    """
    
    MODEL_TYPE_MAP = {
        # Stable Diffusion variants
        "stabilityai/stable-diffusion-2-1": "sd",
        "stabilityai/stable-diffusion-xl-base-1.0": "sdxl",
        "stabilityai/stable-diffusion-3-medium": "sd3",
        "stabilityai/stable-diffusion-3.5-medium": "sd3",
        "stabilityai/stable-diffusion-3.5-large": "sd3",
        
        # FLUX variants
        "black-forest-labs/FLUX.1-dev": "flux",
        "black-forest-labs/FLUX.1-schnell": "flux",
        "black-forest-labs/FLUX.1-fill-dev": "flux-fill",
        
        # Z-Image variants
        "stabilityai/z-image": "z-image",
        "stabilityai/z-image-turbo": "z-image-turbo",
        
        # Hunyuan variants
        "Tencent-Hunyuan/HunyuanDiT-v1.2": "hunyuan",
        "Tencent-Hunyuan/HunyuanDiT-v1.1": "hunyuan",
    }
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.pipeline = None
        self.model_type = self._detect_model_type()
        self.device = self._get_device()
        
    def _detect_model_type(self) -> str:
        """Auto-detect model type from model ID."""
        if self.config.model_type != "auto":
            return self.config.model_type
            
        model_id_lower = self.config.model_id.lower()
        
        # Check exact matches first
        if self.config.model_id in self.MODEL_TYPE_MAP:
            return self.MODEL_TYPE_MAP[self.config.model_id]
        
        # Pattern matching
        if "flux" in model_id_lower:
            return "flux"
        elif "z-image-turbo" in model_id_lower:
            return "z-image-turbo"
        elif "z-image" in model_id_lower:
            return "z-image"
        elif "hunyuan" in model_id_lower:
            return "hunyuan"
        elif "stable-diffusion-3" in model_id_lower or "sd3" in model_id_lower:
            return "sd3"
        elif "xl" in model_id_lower or "sdxl" in model_id_lower:
            return "sdxl"
        elif "stable-diffusion" in model_id_lower:
            return "sd"
        
        logger.warning(f"Unknown model type for {self.config.model_id}, defaulting to sdxl")
        return "sdxl"
    
    def _get_device(self) -> torch.device:
        """Determine the appropriate device."""
        if self.config.device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        return torch.device(self.config.device)
    
    def load(self) -> "ImagePipeline":
        """Load the diffusion pipeline."""
        from diffusers import (
            StableDiffusionPipeline,
            StableDiffusionXLPipeline,
            StableDiffusion3Pipeline,
            FluxPipeline,
            FluxFillPipeline,
            AutoPipelineForText2Image,
        )
        
        logger.info(f"Loading {self.config.model_id} (type: {self.model_type})")
        
        load_kwargs = {
            "torch_dtype": self.config.dtype,
            "cache_dir": self.config.cache_dir,
            "local_files_only": self.config.local_files_only,
        }
        
        # Add safety checker setting
        if not self.config.safety_checker:
            load_kwargs["safety_checker"] = None
        
        try:
            if self.model_type == "sd":
                self.pipeline = StableDiffusionPipeline.from_pretrained(
                    self.config.model_id,
                    **load_kwargs
                )
            elif self.model_type == "sdxl":
                self.pipeline = StableDiffusionXLPipeline.from_pretrained(
                    self.config.model_id,
                    **load_kwargs
                )
            elif self.model_type == "sd3":
                self.pipeline = StableDiffusion3Pipeline.from_pretrained(
                    self.config.model_id,
                    **load_kwargs
                )
            elif self.model_type in ["flux", "flux-schnell"]:
                self.pipeline = FluxPipeline.from_pretrained(
                    self.config.model_id,
                    **load_kwargs
                )
            elif self.model_type == "flux-fill":
                self.pipeline = FluxFillPipeline.from_pretrained(
                    self.config.model_id,
                    **load_kwargs
                )
            elif self.model_type in ["z-image", "z-image-turbo", "hunyuan"]:
                # Use auto pipeline for custom architectures
                self.pipeline = AutoPipelineForText2Image.from_pretrained(
                    self.config.model_id,
                    **load_kwargs
                )
            else:
                # Fallback to auto-detection
                self.pipeline = AutoPipelineForText2Image.from_pretrained(
                    self.config.model_id,
                    **load_kwargs
                )
            
            # Move to device and optimize
            self._optimize_pipeline()
            
            logger.info(f"Pipeline loaded successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load pipeline: {e}")
            raise
        
        return self
    
    def _optimize_pipeline(self):
        """Apply memory optimizations."""
        # Move to device
        if self.device.type == "cuda":
            self.pipeline = self.pipeline.to(self.device)
        
        # Enable VAE slicing for large images
        if self.config.enable_vae_slicing and hasattr(self.pipeline, "enable_vae_slicing"):
            self.pipeline.enable_vae_slicing()
            logger.info("VAE slicing enabled")
        
        # Enable VAE tiling for very large images
        if self.config.enable_vae_tiling and hasattr(self.pipeline, "enable_vae_tiling"):
            self.pipeline.enable_vae_tiling()
            logger.info("VAE tiling enabled")
        
        # Enable CPU offload for low VRAM
        if self.config.enable_cpu_offload and hasattr(self.pipeline, "enable_model_cpu_offload"):
            self.pipeline.enable_model_cpu_offload()
            logger.info("Model CPU offload enabled")
        
        # Enable attention slicing
        if hasattr(self.pipeline, "enable_attention_slicing"):
            self.pipeline.enable_attention_slicing(1)
    
    def generate(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        num_images_per_prompt: int = 1,
        seed: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate images from text prompts.
        
        Args:
            prompt: Text prompt(s) for image generation
            negative_prompt: Negative prompt(s) to avoid
            height: Image height (default from config)
            width: Image width (default from config)
            num_inference_steps: Number of denoising steps
            guidance_scale: CFG scale
            num_images_per_prompt: Number of images to generate per prompt
            seed: Random seed for reproducibility
            **kwargs: Additional pipeline-specific arguments
        
        Returns:
            Dictionary with 'images' (PIL images) and 'nsfw_detected' flags
        """
        if self.pipeline is None:
            raise RuntimeError("Pipeline not loaded. Call load() first.")
        
        # Set defaults from config
        height = height or self.config.default_height
        width = width or self.config.default_width
        num_inference_steps = num_inference_steps or self.config.default_steps
        guidance_scale = guidance_scale or self.config.default_guidance_scale
        
        # Set generator for reproducibility
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        
        # Adjust parameters for specific model types
        gen_kwargs = self._prepare_generation_kwargs(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            num_images_per_prompt=num_images_per_prompt,
            generator=generator,
            **kwargs
        )
        
        # Generate
        logger.info(f"Generating {num_images_per_prompt} image(s) at {width}x{height}")
        with torch.inference_mode():
            output = self.pipeline(**gen_kwargs)
        
        result = {
            "images": output.images,
            "nsfw_detected": getattr(output, "nsfw_content_detected", [False] * len(output.images)),
        }
        
        return result
    
    def _prepare_generation_kwargs(self, **kwargs) -> Dict[str, Any]:
        """Prepare generation kwargs based on model type."""
        gen_kwargs = kwargs.copy()
        
        # Model-specific adjustments
        if self.model_type == "flux-schnell":
            # FLUX Schnell uses fewer steps and different guidance
            gen_kwargs["num_inference_steps"] = min(gen_kwargs.get("num_inference_steps", 4), 4)
            gen_kwargs["guidance_scale"] = 0.0  # FLUX Schnell doesn't use CFG
        elif self.model_type == "flux":
            # FLUX dev uses guidance scale of 3.5
            if gen_kwargs.get("guidance_scale") == 7.5:
                gen_kwargs["guidance_scale"] = 3.5
        elif self.model_type == "sd3":
            # SD3 works well with lower guidance
            if gen_kwargs.get("guidance_scale") == 7.5:
                gen_kwargs["guidance_scale"] = 5.0
        
        # Remove None values
        gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}
        
        return gen_kwargs
    
    def generate_variations(
        self,
        image,
        prompt: Optional[str] = None,
        strength: float = 0.75,
        num_images: int = 1,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate variations of an input image (img2img).
        
        Args:
            image: Input PIL image
            prompt: Optional text prompt to guide variation
            strength: How much to change the image (0-1)
            num_images: Number of variations to generate
            **kwargs: Additional generation arguments
        """
        from diffusers import AutoPipelineForImage2Image
        
        # Create img2img pipeline
        pipe_img2img = AutoPipelineForImage2Image.from_pipe(self.pipeline)
        
        if self.device.type == "cuda":
            pipe_img2img = pipe_img2img.to(self.device)
        
        if prompt is None:
            prompt = "variation of the image"
        
        generator = None
        if "seed" in kwargs:
            generator = torch.Generator(device=self.device).manual_seed(kwargs.pop("seed"))
        
        with torch.inference_mode():
            output = pipe_img2img(
                prompt=prompt,
                image=image,
                strength=strength,
                num_inference_steps=kwargs.get("num_inference_steps", self.config.default_steps),
                guidance_scale=kwargs.get("guidance_scale", self.config.default_guidance_scale),
                num_images_per_prompt=num_images,
                generator=generator,
            )
        
        return {"images": output.images}
    
    def inpaint(
        self,
        image,
        mask,
        prompt: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Inpaint masked regions of an image.
        
        Args:
            image: Input PIL image
            mask: Mask PIL image (white = inpaint, black = keep)
            prompt: Text prompt describing the inpainting
            **kwargs: Additional generation arguments
        """
        from diffusers import AutoPipelineForInpainting
        
        # Create inpainting pipeline
        pipe_inpaint = AutoPipelineForInpainting.from_pipe(self.pipeline)
        
        if self.device.type == "cuda":
            pipe_inpaint = pipe_inpaint.to(self.device)
        
        generator = None
        if "seed" in kwargs:
            generator = torch.Generator(device=self.device).manual_seed(kwargs.pop("seed"))
        
        with torch.inference_mode():
            output = pipe_inpaint(
                prompt=prompt,
                image=image,
                mask_image=mask,
                num_inference_steps=kwargs.get("num_inference_steps", self.config.default_steps),
                guidance_scale=kwargs.get("guidance_scale", self.config.default_guidance_scale),
                generator=generator,
            )
        
        return {"images": output.images}
    
    def unload(self):
        """Unload the pipeline to free memory."""
        if self.pipeline is not None:
            del self.pipeline
            self.pipeline = None
            torch.cuda.empty_cache()
            logger.info("Pipeline unloaded")
    
    def __enter__(self):
        """Context manager entry."""
        self.load()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.unload()
