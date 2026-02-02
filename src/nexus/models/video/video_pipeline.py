"""
Video Generation Pipeline for LTX-2, SVD, and other video models
"""

import torch
from typing import Optional, Union, List, Dict, Any, Tuple
from dataclasses import dataclass
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class VideoConfig:
    """Configuration for video generation pipelines."""
    model_id: str
    model_type: str = "ltx-video"  # ltx-video, svd, cogvideo, hunyuan-video
    device: str = "auto"
    dtype: torch.dtype = torch.bfloat16
    
    # Video parameters
    num_frames: int = 49
    fps: int = 24
    height: int = 512
    width: int = 512
    
    # Generation parameters
    num_inference_steps: int = 50
    guidance_scale: float = 6.0
    min_guidance_scale: float = 1.0
    max_guidance_scale: float = 3.0
    
    # Memory optimization
    enable_vae_slicing: bool = True
    enable_vae_tiling: bool = True
    enable_cpu_offload: bool = True
    
    # Model-specific
    motion_bucket_id: int = 127  # For SVD
    noise_aug_strength: float = 0.02
    
    cache_dir: Optional[str] = None


class VideoPipeline:
    """
    Unified video generation pipeline supporting multiple video models.
    
    Supported Models:
    - LTX-Video (Lightricks)
    - Stable Video Diffusion (SVD)
    - CogVideoX
    - HunyuanVideo
    """
    
    MODEL_TYPE_MAP = {
        # LTX-Video variants
        "Lightricks/LTX-Video": "ltx-video",
        "Lightricks/LTX-Video-2B": "ltx-video",
        
        # SVD variants
        "stabilityai/stable-video-diffusion-img2vid": "svd",
        "stabilityai/stable-video-diffusion-img2vid-xt": "svd-xt",
        "stabilityai/stable-video-diffusion-img2vid-xt-1-1": "svd-xt",
        
        # CogVideoX variants
        "THUDM/CogVideoX-2b": "cogvideo",
        "THUDM/CogVideoX-5b": "cogvideo",
        "THUDM/CogVideoX-5b-I2V": "cogvideo-i2v",
        
        # HunyuanVideo
        "Tencent-Hunyuan/HunyuanVideo": "hunyuan-video",
    }
    
    def __init__(self, config: VideoConfig):
        self.config = config
        self.pipeline = None
        self.model_type = self._detect_model_type()
        self.device = self._get_device()
    
    def _detect_model_type(self) -> str:
        """Auto-detect model type from model ID."""
        if self.config.model_type != "auto":
            return self.config.model_type
        
        model_id_lower = self.config.model_id.lower()
        
        # Check exact matches
        if self.config.model_id in self.MODEL_TYPE_MAP:
            return self.MODEL_TYPE_MAP[self.config.model_id]
        
        # Pattern matching
        if "ltx" in model_id_lower:
            return "ltx-video"
        elif "svd" in model_id_lower or "stable-video" in model_id_lower:
            return "svd"
        elif "cogvideo" in model_id_lower:
            return "cogvideo"
        elif "hunyuan" in model_id_lower and "video" in model_id_lower:
            return "hunyuan-video"
        
        logger.warning(f"Unknown model type for {self.config.model_id}, defaulting to ltx-video")
        return "ltx-video"
    
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
    
    def load(self) -> "VideoPipeline":
        """Load the video generation pipeline."""
        logger.info(f"Loading {self.config.model_id} (type: {self.model_type})")
        
        load_kwargs = {
            "torch_dtype": self.config.dtype,
            "cache_dir": self.config.cache_dir,
        }
        
        try:
            if self.model_type == "ltx-video":
                self._load_ltx_video(load_kwargs)
            elif self.model_type in ["svd", "svd-xt"]:
                self._load_svd(load_kwargs)
            elif self.model_type in ["cogvideo", "cogvideo-i2v"]:
                self._load_cogvideo(load_kwargs)
            elif self.model_type == "hunyuan-video":
                self._load_hunyuan_video(load_kwargs)
            else:
                raise ValueError(f"Unsupported model type: {self.model_type}")
            
            self._optimize_pipeline()
            logger.info(f"Video pipeline loaded successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load video pipeline: {e}")
            raise
        
        return self
    
    def _load_ltx_video(self, load_kwargs: Dict):
        """Load LTX-Video pipeline."""
        from diffusers import LTXPipeline
        
        self.pipeline = LTXPipeline.from_pretrained(
            self.config.model_id,
            **load_kwargs
        )
    
    def _load_svd(self, load_kwargs: Dict):
        """Load Stable Video Diffusion pipeline."""
        from diffusers import StableVideoDiffusionPipeline
        
        self.pipeline = StableVideoDiffusionPipeline.from_pretrained(
            self.config.model_id,
            **load_kwargs
        )
    
    def _load_cogvideo(self, load_kwargs: Dict):
        """Load CogVideoX pipeline."""
        from diffusers import CogVideoXPipeline
        
        self.pipeline = CogVideoXPipeline.from_pretrained(
            self.config.model_id,
            **load_kwargs
        )
    
    def _load_hunyuan_video(self, load_kwargs: Dict):
        """Load HunyuanVideo pipeline."""
        from diffusers import HunyuanVideoPipeline
        
        self.pipeline = HunyuanVideoPipeline.from_pretrained(
            self.config.model_id,
            **load_kwargs
        )
    
    def _optimize_pipeline(self):
        """Apply memory optimizations."""
        if self.config.enable_vae_slicing and hasattr(self.pipeline, "enable_vae_slicing"):
            self.pipeline.enable_vae_slicing()
            logger.info("VAE slicing enabled")
        
        if self.config.enable_vae_tiling and hasattr(self.pipeline, "enable_vae_tiling"):
            self.pipeline.enable_vae_tiling()
            logger.info("VAE tiling enabled")
        
        if self.config.enable_cpu_offload and hasattr(self.pipeline, "enable_model_cpu_offload"):
            self.pipeline.enable_model_cpu_offload()
            logger.info("Model CPU offload enabled")
        
        if hasattr(self.pipeline, "enable_attention_slicing"):
            self.pipeline.enable_attention_slicing(1)
    
    def generate(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        num_frames: Optional[int] = None,
        fps: Optional[int] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        seed: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate video from text prompt.
        
        Args:
            prompt: Text prompt describing the video
            negative_prompt: What to avoid in the video
            num_frames: Number of frames to generate
            fps: Frames per second
            height: Video height
            width: Video width
            num_inference_steps: Number of denoising steps
            guidance_scale: CFG scale
            seed: Random seed
            **kwargs: Additional model-specific arguments
        
        Returns:
            Dictionary with 'frames' (list of PIL images) and 'fps'
        """
        if self.pipeline is None:
            raise RuntimeError("Pipeline not loaded. Call load() first.")
        
        # Set defaults
        num_frames = num_frames or self.config.num_frames
        fps = fps or self.config.fps
        height = height or self.config.height
        width = width or self.config.width
        num_inference_steps = num_inference_steps or self.config.num_inference_steps
        guidance_scale = guidance_scale or self.config.guidance_scale
        
        # Set generator
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        
        # Prepare generation arguments
        gen_kwargs = {
            "prompt": prompt,
            "num_frames": num_frames,
            "num_inference_steps": num_inference_steps,
            "generator": generator,
        }
        
        # Add model-specific parameters
        if self.model_type == "ltx-video":
            gen_kwargs.update({
                "width": width,
                "height": height,
                "guidance_scale": guidance_scale,
            })
        elif self.model_type in ["svd", "svd-xt"]:
            # SVD requires an image, not text
            raise ValueError("SVD requires an image input. Use generate_from_image() instead.")
        elif self.model_type == "cogvideo":
            gen_kwargs.update({
                "guidance_scale": guidance_scale,
                "height": height,
                "width": width,
            })
        elif self.model_type == "hunyuan-video":
            gen_kwargs.update({
                "guidance_scale": guidance_scale,
                "height": height,
                "width": width,
            })
        
        if negative_prompt:
            gen_kwargs["negative_prompt"] = negative_prompt
        
        logger.info(f"Generating {num_frames} frames at {width}x{height}")
        
        with torch.inference_mode():
            output = self.pipeline(**gen_kwargs)
        
        return {
            "frames": output.frames[0] if isinstance(output.frames, list) else output.frames,
            "fps": fps,
            "num_frames": num_frames,
        }
    
    def generate_from_image(
        self,
        image,
        prompt: Optional[str] = None,
        num_frames: int = 25,
        fps: int = 7,
        motion_bucket_id: Optional[int] = None,
        noise_aug_strength: Optional[float] = None,
        seed: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate video from an image (Image-to-Video).
        
        Args:
            image: Input PIL image
            prompt: Optional text prompt (for models that support it)
            num_frames: Number of frames to generate
            fps: Frames per second
            motion_bucket_id: Motion intensity (SVD specific)
            noise_aug_strength: Noise augmentation strength
            seed: Random seed
            **kwargs: Additional arguments
        
        Returns:
            Dictionary with frames and metadata
        """
        if self.pipeline is None:
            raise RuntimeError("Pipeline not loaded. Call load() first.")
        
        motion_bucket_id = motion_bucket_id or self.config.motion_bucket_id
        noise_aug_strength = noise_aug_strength or self.config.noise_aug_strength
        
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        
        gen_kwargs = {
            "image": image,
            "num_frames": num_frames,
            "generator": generator,
        }
        
        if self.model_type in ["svd", "svd-xt"]:
            gen_kwargs.update({
                "fps": fps,
                "motion_bucket_id": motion_bucket_id,
                "noise_aug_strength": noise_aug_strength,
            })
        
        if prompt and self.model_type in ["cogvideo-i2v", "ltx-video"]:
            gen_kwargs["prompt"] = prompt
        
        logger.info(f"Generating video from image: {num_frames} frames")
        
        with torch.inference_mode():
            output = self.pipeline(**gen_kwargs)
        
        return {
            "frames": output.frames[0] if isinstance(output.frames, list) else output.frames,
            "fps": fps,
            "num_frames": num_frames,
        }
    
    def interpolate_frames(
        self,
        start_image,
        end_image,
        num_intermediate_frames: int = 5,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate intermediate frames between two images.
        
        Args:
            start_image: Starting frame
            end_image: Ending frame
            num_intermediate_frames: Number of frames to generate
            **kwargs: Additional generation arguments
        
        Returns:
            Dictionary with interpolated frames
        """
        # This is a placeholder for frame interpolation
        # Implementation would use a dedicated interpolation model
        logger.info(f"Interpolating {num_intermediate_frames} frames between images")
        
        # For now, use img2vid with both images as conditioning
        frames = []
        
        # Generate video from start image
        result = self.generate_from_image(
            start_image,
            num_frames=num_intermediate_frames + 2,
            **kwargs
        )
        
        return {
            "frames": result["frames"],
            "fps": result["fps"],
            "interpolated": True,
        }
    
    def unload(self):
        """Unload the pipeline to free memory."""
        if self.pipeline is not None:
            del self.pipeline
            self.pipeline = None
            torch.cuda.empty_cache()
            logger.info("Video pipeline unloaded")
    
    def __enter__(self):
        """Context manager entry."""
        self.load()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.unload()
