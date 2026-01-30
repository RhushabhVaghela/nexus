import torch
import torch.nn as nn
from typing import Optional, Union, List, Tuple, Dict, Any
from PIL import Image
import numpy as np
from transformers import SpeechT5HifiGan, AutoModelForCausalLM
from diffusers import AutoencoderKL, StableVideoDiffusionPipeline
from diffusers.utils import load_image


class VideoDecoder:
    """
    Video decoder using Stable Video Diffusion (SVD) for generating video frames
    from conditioning inputs (images or text prompts).
    
    This class wraps the StableVideoDiffusionPipeline from diffusers and provides
    a unified interface for video generation with proper memory management.
    
    Attributes:
        pipeline: The StableVideoDiffusionPipeline instance
        device: The device to run inference on
        model_id: The model identifier used for loading
    """
    
    def __init__(
        self,
        model_id: str = "stabilityai/stable-video-diffusion-img2vid-xt",
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.float16,
        enable_model_cpu_offload: bool = True,
        enable_vae_slicing: bool = True,
        enable_vae_tiling: bool = False
    ):
        """
        Initialize the VideoDecoder.
        
        Args:
            model_id: HuggingFace model ID for SVD (default: SVD-XT for 25 frames)
            device: Device to load the model on ('cuda', 'cpu', etc.)
            torch_dtype: Data type for model weights (fp16 recommended for memory)
            enable_model_cpu_offload: Offload model to CPU when not in use
            enable_vae_slicing: Enable VAE slicing for lower memory usage
            enable_vae_tiling: Enable VAE tiling for very large videos
        """
        self.model_id = model_id
        self.device = device
        self.torch_dtype = torch_dtype
        self.pipeline = None
        self._enable_model_cpu_offload = enable_model_cpu_offload
        self._enable_vae_slicing = enable_vae_slicing
        self._enable_vae_tiling = enable_vae_tiling
        
    def load(self) -> "VideoDecoder":
        """
        Load the Stable Video Diffusion pipeline.
        
        Returns:
            self for method chaining
            
        Raises:
            RuntimeError: If model loading fails
        """
        try:
            print(f"Loading Video Decoder: {self.model_id}")
            
            # Load the SVD pipeline
            self.pipeline = StableVideoDiffusionPipeline.from_pretrained(
                self.model_id,
                torch_dtype=self.torch_dtype,
                variant="fp16" if self.torch_dtype == torch.float16 else None
            )
            
            # Enable memory optimizations
            if self._enable_vae_slicing:
                self.pipeline.enable_vae_slicing()
                
            if self._enable_vae_tiling:
                self.pipeline.enable_vae_tiling()
                
            if self._enable_model_cpu_offload:
                self.pipeline.enable_model_cpu_offload()
            else:
                self.pipeline = self.pipeline.to(self.device)
                
            # Set to evaluation mode
            self.pipeline.unet.eval()
            self.pipeline.vae.eval()
            
            print(f"Video Decoder loaded successfully on {self.device}")
            return self
            
        except Exception as e:
            raise RuntimeError(f"Failed to load Video Decoder: {str(e)}") from e
    
    def generate(
        self,
        conditioning: Union[Image.Image, str, torch.Tensor],
        num_frames: int = 25,
        num_inference_steps: int = 25,
        min_guidance_scale: float = 1.0,
        max_guidance_scale: float = 3.0,
        fps: int = 7,
        motion_bucket_id: int = 127,
        noise_aug_strength: float = 0.02,
        decode_chunk_size: int = 8,
        seed: Optional[int] = None,
        output_type: str = "pil"
    ) -> List[Image.Image]:
        """
        Generate video frames from conditioning input.
        
        Args:
            conditioning: Input image (PIL Image), path to image (str), or latent tensor
            num_frames: Number of video frames to generate (default: 25 for SVD-XT)
            num_inference_steps: Number of denoising steps (default: 25)
            min_guidance_scale: Minimum guidance scale for classifier-free guidance
            max_guidance_scale: Maximum guidance scale for classifier-free guidance
            fps: Frames per second for the output video
            motion_bucket_id: Motion bucket ID (0-255), higher = more motion
            noise_aug_strength: Noise augmentation strength for conditioning image
            decode_chunk_size: Number of frames to decode at once (for memory management)
            seed: Random seed for reproducibility
            output_type: Output format - "pil" for PIL Images, "np" for numpy arrays
            
        Returns:
            List of PIL Images representing video frames
            
        Raises:
            RuntimeError: If pipeline is not loaded or generation fails
            ValueError: If invalid conditioning input provided
        """
        if self.pipeline is None:
            raise RuntimeError("VideoDecoder not loaded. Call load() first.")
        
        # Set random seed if provided
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        
        # Process conditioning input
        if isinstance(conditioning, str):
            # Load image from path
            try:
                conditioning = load_image(conditioning)
            except Exception as e:
                raise ValueError(f"Failed to load image from path: {conditioning}") from e
        elif isinstance(conditioning, torch.Tensor):
            # Convert latent tensor to image if needed
            if conditioning.dim() == 4:  # (B, C, H, W)
                conditioning = self._latent_to_image(conditioning)
            else:
                raise ValueError(f"Unexpected tensor shape: {conditioning.shape}")
        elif not isinstance(conditioning, Image.Image):
            raise ValueError(
                f"Conditioning must be PIL Image, path string, or tensor. Got {type(conditioning)}"
            )
        
        try:
            # Generate video frames
            with torch.no_grad():
                frames = self.pipeline(
                    image=conditioning,
                    height=conditioning.height if hasattr(conditioning, 'height') else 576,
                    width=conditioning.width if hasattr(conditioning, 'width') else 1024,
                    num_frames=num_frames,
                    num_inference_steps=num_inference_steps,
                    min_guidance_scale=min_guidance_scale,
                    max_guidance_scale=max_guidance_scale,
                    fps=fps,
                    motion_bucket_id=motion_bucket_id,
                    noise_aug_strength=noise_aug_strength,
                    decode_chunk_size=decode_chunk_size,
                    generator=generator,
                    output_type=output_type
                ).frames[0]
            
            return frames
            
        except Exception as e:
            raise RuntimeError(f"Video generation failed: {str(e)}") from e
    
    def generate_from_text(
        self,
        prompt: str,
        image_generator: Optional[Any] = None,
        num_frames: int = 25,
        num_inference_steps: int = 25,
        seed: Optional[int] = None,
        **kwargs
    ) -> List[Image.Image]:
        """
        Generate video from text prompt by first generating a conditioning image.
        
        Args:
            prompt: Text description of the desired video
            image_generator: Optional image generation pipeline (e.g., StableDiffusionPipeline)
                           If None, raises ValueError (text-to-video requires image first)
            num_frames: Number of video frames to generate
            num_inference_steps: Number of denoising steps
            seed: Random seed for reproducibility
            **kwargs: Additional arguments passed to generate()
            
        Returns:
            List of PIL Images representing video frames
            
        Raises:
            ValueError: If no image_generator is provided
        """
        if image_generator is None:
            raise ValueError(
                "Text-to-video generation requires an image generator. "
                "Please provide an image generation pipeline (e.g., StableDiffusionPipeline) "
                "or first generate an image and pass it to generate()."
            )
        
        # Generate conditioning image from text
        print(f"Generating conditioning image for prompt: {prompt[:50]}...")
        with torch.no_grad():
            conditioning_image = image_generator(
                prompt,
                num_inference_steps=25,
                generator=torch.Generator(device=self.device).manual_seed(seed) if seed else None
            ).images[0]
        
        # Generate video from the conditioning image
        return self.generate(
            conditioning=conditioning_image,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            seed=seed,
            **kwargs
        )
    
    def to_video_array(self, frames: List[Image.Image]) -> np.ndarray:
        """
        Convert list of PIL frames to numpy video array.
        
        Args:
            frames: List of PIL Image frames
            
        Returns:
            Numpy array of shape (num_frames, height, width, 3) with values 0-255
        """
        return np.stack([np.array(frame) for frame in frames])
    
    def save_video(
        self,
        frames: List[Image.Image],
        output_path: str,
        fps: int = 7,
        format: str = "mp4"
    ) -> str:
        """
        Save video frames to file.
        
        Args:
            frames: List of PIL Image frames
            output_path: Path to save the video
            fps: Frames per second
            format: Output format (mp4, gif, webm)
            
        Returns:
            Path to saved video file
        """
        try:
            import imageio
        except ImportError:
            raise ImportError(
                "imageio is required for video saving. "
                "Install with: pip install imageio[ffmpeg]"
            )
        
        video_array = self.to_video_array(frames)
        
        if format == "gif":
            frames[0].save(
                output_path,
                save_all=True,
                append_images=frames[1:],
                duration=int(1000 / fps),
                loop=0
            )
        else:
            imageio.mimsave(output_path, video_array, fps=fps)
        
        return output_path
    
    def _latent_to_image(self, latent: torch.Tensor) -> Image.Image:
        """
        Convert latent tensor to PIL Image using the VAE decoder.
        
        Args:
            latent: Latent tensor of shape (B, C, H, W) or (C, H, W)
            
        Returns:
            PIL Image
        """
        if latent.dim() == 3:
            latent = latent.unsqueeze(0)
        
        with torch.no_grad():
            # Decode latent to image
            image = self.pipeline.vae.decode(latent / self.pipeline.vae.config.scaling_factor).sample
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).numpy()
            image = (image * 255).astype(np.uint8)[0]
        
        return Image.fromarray(image)
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get memory usage statistics.
        
        Returns:
            Dictionary with memory statistics
        """
        stats = {
            "device": self.device,
            "model_loaded": self.pipeline is not None
        }
        
        if self.device == "cuda" and torch.cuda.is_available():
            stats["gpu_allocated_mb"] = torch.cuda.memory_allocated() / 1024**2
            stats["gpu_reserved_mb"] = torch.cuda.memory_reserved() / 1024**2
            
        return stats
    
    def unload(self):
        """
        Unload the pipeline from memory to free GPU resources.
        """
        if self.pipeline is not None:
            del self.pipeline
            self.pipeline = None
            torch.cuda.empty_cache()
            print("Video Decoder unloaded from memory")


class NexusDecoders:
    """
    Factory for initializing multimodal decoders at inference time.
    These are standard, lightweight components included in the bundle.
    """
    
    @staticmethod
    def load_audio_vocoder(model_id="microsoft/speecht5_hifigan", device="cuda"):
        """
        Loads the HiFi-GAN Vocoder for Text-to-Speech synthesis.
        Converts Mel Spectrograms (from Audio Adapter) -> Waveform.
        """
        print(f"Loading Audio Vocoder: {model_id}")
        vocoder = SpeechT5HifiGan.from_pretrained(model_id).to(device)
        vocoder.eval()
        return vocoder

    @staticmethod
    def load_image_decoder(model_id="stabilityai/sd-vae-ft-mse", device="cuda"):
        """
        Loads the VAE Decoder for Image Generation.
        Converts Latent Images (from Vision Adapter) -> RGB Pixels.
        """
        print(f"Loading Image VAE: {model_id}")
        vae = AutoencoderKL.from_pretrained(model_id).to(device)
        vae.eval()
        return vae

    @staticmethod
    def load_video_decoder(
        model_id="stabilityai/stable-video-diffusion-img2vid-xt",
        device="cuda",
        **kwargs
    ) -> VideoDecoder:
        """
        Loads and initializes the Video Decoder using Stable Video Diffusion.
        
        This method creates and returns a VideoDecoder instance that can generate
        video frames from conditioning inputs (images or text prompts).
        
        Args:
            model_id: HuggingFace model ID for SVD
                     (default: stabilityai/stable-video-diffusion-img2vid-xt for 25 frames)
                     Alternative: stabilityai/stable-video-diffusion-img2vid for 14 frames
            device: Device to load the model on
            **kwargs: Additional arguments passed to VideoDecoder constructor
                     (torch_dtype, enable_model_cpu_offload, enable_vae_slicing, etc.)
        
        Returns:
            VideoDecoder: Initialized and loaded video decoder ready for generation
            
        Example:
            >>> decoder = NexusDecoders.load_video_decoder()
            >>> frames = decoder.generate("path/to/image.jpg", num_frames=25)
            >>> decoder.save_video(frames, "output.mp4")
        """
        decoder = VideoDecoder(model_id=model_id, device=device, **kwargs)
        return decoder.load()  # Load and return the initialized decoder 
