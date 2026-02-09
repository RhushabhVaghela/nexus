"""
Frame Generator - Frame-by-frame video generation with temporal coherence
"""

import torch
import torch.nn as nn
from typing import Optional, List, Dict, Any, Tuple, Callable
from dataclasses import dataclass
import numpy as np

try:
    from PIL import Image

    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
import logging

if not PIL_AVAILABLE:
    raise ImportError(
        "Pillow is required for the video frame_generator module. "
        "Install with: pip install Pillow"
    )

logger = logging.getLogger(__name__)


@dataclass
class FrameGenerationConfig:
    """Configuration for frame generation."""

    num_frames: int = 24
    overlap_frames: int = 4  # Overlapping frames for temporal consistency
    guidance_scale: float = 7.5
    num_inference_steps: int = 25
    seed: Optional[int] = None

    # Temporal consistency
    temporal_weight: float = 0.8
    motion_consistency_weight: float = 0.5

    # Generation mode
    mode: str = "overlap"  # 'overlap', 'keyframe', 'autoregressive'


class FrameGenerator:
    """
    Generates video frames with temporal consistency.

    Supports multiple generation modes:
    - Overlap: Generates frames with overlapping windows
    - Keyframe: Generates keyframes and interpolates
    - Autoregressive: Uses previous frames as conditioning
    """

    def __init__(
        self,
        pipeline,
        config: Optional[FrameGenerationConfig] = None,
    ):
        """
        Args:
            pipeline: VideoPipeline or ImagePipeline instance
            config: Frame generation configuration
        """
        self.pipeline = pipeline
        self.config = config or FrameGenerationConfig()
        self._frame_buffer: List[Image.Image] = []

    def generate_sequence(
        self,
        prompt: str,
        num_frames: Optional[int] = None,
        keyframes: Optional[List[Image.Image]] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        **kwargs,
    ) -> List[Image.Image]:
        """
        Generate a sequence of frames.

        Args:
            prompt: Text prompt for generation
            num_frames: Total number of frames
            keyframes: Optional list of keyframe images
            progress_callback: Callback function(frame_idx, total_frames)
            **kwargs: Additional generation arguments

        Returns:
            List of generated frames
        """
        num_frames = num_frames or self.config.num_frames

        if self.config.mode == "overlap":
            return self._generate_overlap_mode(
                prompt, num_frames, progress_callback, **kwargs
            )
        elif self.config.mode == "keyframe":
            return self._generate_keyframe_mode(
                prompt, num_frames, keyframes, progress_callback, **kwargs
            )
        elif self.config.mode == "autoregressive":
            return self._generate_autoregressive_mode(
                prompt, num_frames, progress_callback, **kwargs
            )
        else:
            raise ValueError(f"Unknown generation mode: {self.config.mode}")

    def _generate_overlap_mode(
        self,
        prompt: str,
        num_frames: int,
        progress_callback: Optional[Callable] = None,
        **kwargs,
    ) -> List[Image.Image]:
        """Generate frames using overlapping windows for smooth transitions."""
        frames = []
        window_size = self.config.num_frames
        overlap = self.config.overlap_frames
        step = window_size - overlap

        num_windows = (num_frames - overlap) // step + 1

        for window_idx in range(num_windows):
            start_frame = window_idx * step
            end_frame = min(start_frame + window_size, num_frames)
            actual_window_size = end_frame - start_frame

            if window_idx == 0:
                # First window - full generation
                window_frames = self._generate_window(
                    prompt, actual_window_size, seed=self.config.seed, **kwargs
                )
                frames.extend(window_frames)
            else:
                # Subsequent windows - blend with overlap
                window_frames = self._generate_window(
                    prompt,
                    actual_window_size,
                    seed=self.config.seed + window_idx if self.config.seed else None,
                    **kwargs,
                )

                # Blend overlapping region
                if len(frames) >= overlap:
                    blended = self._blend_frames(
                        frames[-overlap:], window_frames[:overlap]
                    )
                    # Replace overlap in frames
                    frames = frames[:-overlap] + blended
                    # Add remaining new frames
                    frames.extend(window_frames[overlap:])
                else:
                    frames.extend(window_frames)

            if progress_callback:
                progress_callback(min(end_frame, num_frames), num_frames)

        return frames[:num_frames]

    def _generate_keyframe_mode(
        self,
        prompt: str,
        num_frames: int,
        keyframes: Optional[List[Image.Image]],
        progress_callback: Optional[Callable] = None,
        **kwargs,
    ) -> List[Image.Image]:
        """Generate frames from keyframes."""
        if keyframes is None or len(keyframes) < 2:
            logger.warning(
                "Keyframe mode requires at least 2 keyframes. Using overlap mode."
            )
            return self._generate_overlap_mode(
                prompt, num_frames, progress_callback, **kwargs
            )

        frames = []
        keyframe_interval = num_frames // (len(keyframes) - 1)

        for i in range(len(keyframes) - 1):
            start_keyframe = keyframes[i]
            end_keyframe = keyframes[i + 1]

            # Generate interpolation frames
            interp_frames = self._interpolate_keyframes(
                start_keyframe, end_keyframe, keyframe_interval - 1, prompt, **kwargs
            )

            if i == 0:
                frames.append(start_keyframe)

            frames.extend(interp_frames)
            frames.append(end_keyframe)

            if progress_callback:
                progress_callback(
                    min((i + 1) * keyframe_interval, num_frames), num_frames
                )

        return frames[:num_frames]

    def _generate_autoregressive_mode(
        self,
        prompt: str,
        num_frames: int,
        progress_callback: Optional[Callable] = None,
        **kwargs,
    ) -> List[Image.Image]:
        """Generate frames autoregressively using previous frames as context."""
        frames = []
        context_frames = []

        for frame_idx in range(num_frames):
            if frame_idx == 0:
                # First frame - regular generation
                frame = self._generate_single_frame(
                    prompt, seed=self.config.seed, **kwargs
                )
            else:
                # Use last few frames as context
                context_frames = frames[-self.config.overlap_frames :]
                frame = self._generate_with_context(
                    prompt,
                    context_frames,
                    seed=self.config.seed + frame_idx if self.config.seed else None,
                    **kwargs,
                )

            frames.append(frame)

            if progress_callback:
                progress_callback(frame_idx + 1, num_frames)

        return frames

    def _generate_window(
        self, prompt: str, num_frames: int, seed: Optional[int] = None, **kwargs
    ) -> List[Image.Image]:
        """Generate a window of frames."""
        # Check if pipeline has video generation capability
        if hasattr(self.pipeline, "generate"):
            result = self.pipeline.generate(
                prompt=prompt,
                num_frames=num_frames,
                seed=seed,
                num_inference_steps=self.config.num_inference_steps,
                guidance_scale=self.config.guidance_scale,
                **kwargs,
            )
            return result.get("frames", [])
        else:
            # Fallback to image generation
            frames = []
            for i in range(num_frames):
                result = self.pipeline.generate(
                    prompt=prompt,
                    seed=seed + i if seed else None,
                    num_inference_steps=self.config.num_inference_steps,
                    guidance_scale=self.config.guidance_scale,
                    **kwargs,
                )
                frames.append(result["images"][0])
            return frames

    def _generate_single_frame(
        self, prompt: str, seed: Optional[int] = None, **kwargs
    ) -> Image.Image:
        """Generate a single frame."""
        result = self.pipeline.generate(
            prompt=prompt,
            num_images_per_prompt=1,
            seed=seed,
            num_inference_steps=self.config.num_inference_steps,
            guidance_scale=self.config.guidance_scale,
            **kwargs,
        )
        return result["images"][0]

    def _generate_with_context(
        self,
        prompt: str,
        context_frames: List[Image.Image],
        seed: Optional[int] = None,
        **kwargs,
    ) -> Image.Image:
        """Generate a frame conditioned on previous frames."""
        # Use img2img with temporal conditioning
        if len(context_frames) > 0:
            # Use last frame as base
            base_image = context_frames[-1]
            result = self.pipeline.generate_variations(
                image=base_image,
                prompt=prompt,
                strength=0.3,  # Low strength for temporal consistency
                seed=seed,
                **kwargs,
            )
            return result["images"][0]
        else:
            return self._generate_single_frame(prompt, seed, **kwargs)

    def _interpolate_keyframes(
        self,
        start_frame: Image.Image,
        end_frame: Image.Image,
        num_intermediate: int,
        prompt: str,
        **kwargs,
    ) -> List[Image.Image]:
        """Generate intermediate frames between keyframes."""
        frames = []

        for i in range(num_intermediate):
            alpha = (i + 1) / (num_intermediate + 1)

            # Blend prompts or use same prompt
            blended_prompt = prompt

            # Use img2img with varying strength for interpolation
            # Start closer to start_frame, end closer to end_frame
            strength = alpha

            # Generate from start_frame
            result = self.pipeline.generate_variations(
                image=start_frame,
                prompt=blended_prompt,
                strength=strength * 0.5,  # Keep it relatively close
                **kwargs,
            )
            frames.append(result["images"][0])

        return frames

    def _blend_frames(
        self,
        frames_a: List[Image.Image],
        frames_b: List[Image.Image],
    ) -> List[Image.Image]:
        """Blend two sets of overlapping frames."""
        blended = []

        for i, (frame_a, frame_b) in enumerate(zip(frames_a, frames_b)):
            # Calculate blend weight
            alpha = i / (len(frames_a) - 1) if len(frames_a) > 1 else 0.5
            weight_a = 1 - alpha
            weight_b = alpha

            # Blend images
            blended_frame = self._blend_images(frame_a, frame_b, weight_a, weight_b)
            blended.append(blended_frame)

        return blended

    def _blend_images(
        self,
        img_a: Image.Image,
        img_b: Image.Image,
        weight_a: float,
        weight_b: float,
    ) -> Image.Image:
        """Blend two PIL images."""
        # Convert to numpy arrays
        arr_a = np.array(img_a).astype(np.float32)
        arr_b = np.array(img_b).astype(np.float32)

        # Blend
        blended = weight_a * arr_a + weight_b * arr_b
        blended = np.clip(blended, 0, 255).astype(np.uint8)

        return Image.fromarray(blended)

    def clear_buffer(self):
        """Clear the frame buffer."""
        self._frame_buffer.clear()
