"""
Temporal Consistency Processor - Ensure smooth video transitions

Implements various techniques for maintaining temporal consistency:
- Optical flow warping
- Temporal attention
- Consistency losses
- Frame interpolation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Dict, Any

try:
    from PIL import Image

    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
import numpy as np

if not PIL_AVAILABLE:
    raise ImportError(
        "Pillow is required for the video temporal_consistency module. "
        "Install with: pip install Pillow"
    )
import logging

logger = logging.getLogger(__name__)


class TemporalConsistencyProcessor:
    """
    Ensures temporal consistency across video frames.

    Methods:
    - Flow warping: Uses optical flow to warp frames
    - Feature consistency: Enforces consistency in feature space
    - Latent blending: Blends latents for smooth transitions
    """

    def __init__(
        self,
        consistency_weight: float = 0.8,
        flow_weight: float = 0.5,
        use_optical_flow: bool = True,
    ):
        """
        Args:
            consistency_weight: Weight for temporal consistency loss
            flow_weight: Weight for optical flow guidance
            use_optical_flow: Whether to compute and use optical flow
        """
        self.consistency_weight = consistency_weight
        self.flow_weight = flow_weight
        self.use_optical_flow = use_optical_flow
        self._flow_model = None

    def process_sequence(
        self,
        frames: List[Image.Image],
        mode: str = "smooth",
        strength: float = 0.5,
    ) -> List[Image.Image]:
        """
        Process a sequence of frames for temporal consistency.

        Args:
            frames: List of PIL images
            mode: Processing mode ('smooth', 'stabilize', 'flow')
            strength: Strength of the consistency correction

        Returns:
            List of processed frames
        """
        if len(frames) < 2:
            return frames

        if mode == "smooth":
            return self._apply_temporal_smoothing(frames, strength)
        elif mode == "stabilize":
            return self._apply_stabilization(frames, strength)
        elif mode == "flow":
            return self._apply_flow_warping(frames, strength)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def _apply_temporal_smoothing(
        self,
        frames: List[Image.Image],
        strength: float,
    ) -> List[Image.Image]:
        """Apply temporal smoothing using moving average."""
        smoothed = []

        for i, frame in enumerate(frames):
            # Get neighboring frames
            neighbors = []
            for j in range(max(0, i - 1), min(len(frames), i + 2)):
                if j != i:
                    neighbors.append(frames[j])

            if not neighbors:
                smoothed.append(frame)
                continue

            # Convert to numpy
            frame_arr = np.array(frame).astype(np.float32)
            neighbor_arrs = [np.array(n).astype(np.float32) for n in neighbors]

            # Weighted average
            current_weight = 1.0 - strength
            neighbor_weight = strength / len(neighbors)

            smoothed_arr = current_weight * frame_arr
            for neighbor_arr in neighbor_arrs:
                smoothed_arr += neighbor_weight * neighbor_arr

            smoothed_arr = np.clip(smoothed_arr, 0, 255).astype(np.uint8)
            smoothed.append(Image.fromarray(smoothed_arr))

        return smoothed

    def _apply_stabilization(
        self,
        frames: List[Image.Image],
        strength: float,
    ) -> List[Image.Image]:
        """Apply motion stabilization."""
        # Simplified stabilization - maintain consistent frame-to-frame changes
        stabilized = [frames[0]]

        for i in range(1, len(frames)):
            prev_frame = stabilized[-1]
            curr_frame = frames[i]

            # Blend based on difference
            prev_arr = np.array(prev_frame).astype(np.float32)
            curr_arr = np.array(curr_frame).astype(np.float32)

            # Calculate difference
            diff = np.abs(curr_arr - prev_arr).mean()

            # Adaptive blending
            blend_strength = min(strength * (diff / 255.0), 0.5)
            blended = (1 - blend_strength) * curr_arr + blend_strength * prev_arr

            blended = np.clip(blended, 0, 255).astype(np.uint8)
            stabilized.append(Image.fromarray(blended))

        return stabilized

    def _apply_flow_warping(
        self,
        frames: List[Image.Image],
        strength: float,
    ) -> List[Image.Image]:
        """Apply optical flow warping for consistency."""
        if not self.use_optical_flow:
            return self._apply_temporal_smoothing(frames, strength)

        try:
            import cv2
        except ImportError:
            logger.warning("OpenCV not available, falling back to smoothing")
            return self._apply_temporal_smoothing(frames, strength)

        warped = [frames[0]]

        for i in range(1, len(frames)):
            prev_frame = np.array(frames[i - 1])
            curr_frame = np.array(frames[i])

            # Convert to grayscale for flow
            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_RGB2GRAY)
            curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_RGB2GRAY)

            # Calculate optical flow
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray,
                curr_gray,
                None,
                pyr_scale=0.5,
                levels=3,
                winsize=15,
                iterations=3,
                poly_n=5,
                poly_sigma=1.2,
                flags=0,
            )

            # Warp previous frame using flow
            h, w = prev_frame.shape[:2]
            flow_map = (
                np.column_stack((np.repeat(np.arange(h), w), np.tile(np.arange(w), h)))
                .reshape(h, w, 2)
                .astype(np.float32)
            )
            flow_map += flow

            warped_prev = cv2.remap(
                prev_frame, flow_map[..., 1], flow_map[..., 0], cv2.INTER_LINEAR
            )

            # Blend warped frame with current frame
            blended = (1 - strength) * curr_frame + strength * warped_prev
            blended = np.clip(blended, 0, 255).astype(np.uint8)

            warped.append(Image.fromarray(blended))

        return warped

    def compute_temporal_loss(
        self,
        features: List[torch.Tensor],
        loss_type: str = "mse",
    ) -> torch.Tensor:
        """
        Compute temporal consistency loss between frame features.

        Args:
            features: List of feature tensors for each frame
            loss_type: Type of loss ('mse', 'cosine', 'smooth_l1')

        Returns:
            Temporal loss tensor
        """
        if len(features) < 2:
            return torch.tensor(0.0, device=features[0].device)

        losses = []

        for i in range(1, len(features)):
            prev_feat = features[i - 1]
            curr_feat = features[i]

            # Ensure same shape
            if prev_feat.shape != curr_feat.shape:
                curr_feat = (
                    F.interpolate(
                        curr_feat.unsqueeze(0) if curr_feat.dim() == 3 else curr_feat,
                        size=prev_feat.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    ).squeeze(0)
                    if curr_feat.dim() == 3
                    else curr_feat
                )

            if loss_type == "mse":
                loss = F.mse_loss(curr_feat, prev_feat)
            elif loss_type == "cosine":
                loss = (
                    1
                    - F.cosine_similarity(
                        curr_feat.flatten(1), prev_feat.flatten(1), dim=1
                    ).mean()
                )
            elif loss_type == "smooth_l1":
                loss = F.smooth_l1_loss(curr_feat, prev_feat)
            else:
                raise ValueError(f"Unknown loss type: {loss_type}")

            losses.append(loss)

        return torch.stack(losses).mean() * self.consistency_weight

    def blend_latents_temporally(
        self,
        latents: torch.Tensor,
        num_overlap_frames: int = 4,
    ) -> torch.Tensor:
        """
        Blend latents for temporal consistency in diffusion.

        Args:
            latents: Latent tensor of shape (batch, channels, frames, height, width)
            num_overlap_frames: Number of frames to blend at boundaries

        Returns:
            Blended latents
        """
        if latents.dim() != 5:
            return latents

        batch, channels, num_frames, height, width = latents.shape

        if num_frames < num_overlap_frames * 2:
            return latents

        blended = latents.clone()

        # Create blending weights
        weights = torch.linspace(0, 1, num_overlap_frames, device=latents.device)

        # Blend middle frames with neighbors
        for i in range(1, num_frames - 1):
            weight_current = 0.7
            weight_neighbors = 0.15

            blended[:, :, i] = (
                weight_current * latents[:, :, i]
                + weight_neighbors * latents[:, :, i - 1]
                + weight_neighbors * latents[:, :, i + 1]
            )

        return blended

    def interpolate_frames(
        self,
        frame1: Image.Image,
        frame2: Image.Image,
        num_interpolated: int = 1,
    ) -> List[Image.Image]:
        """
        Generate interpolated frames between two frames.

        Args:
            frame1: Starting frame
            frame2: Ending frame
            num_interpolated: Number of frames to interpolate

        Returns:
            List of interpolated frames
        """
        arr1 = np.array(frame1).astype(np.float32)
        arr2 = np.array(frame2).astype(np.float32)

        interpolated = []

        for i in range(1, num_interpolated + 1):
            alpha = i / (num_interpolated + 1)
            blended = (1 - alpha) * arr1 + alpha * arr2
            blended = np.clip(blended, 0, 255).astype(np.uint8)
            interpolated.append(Image.fromarray(blended))

        return interpolated


class TemporalAttention(nn.Module):
    """
    Temporal attention module for video processing.

    Applies attention across the temporal dimension to maintain consistency.
    """

    def __init__(
        self,
        channels: int,
        num_frames: int = 8,
        num_heads: int = 8,
    ):
        super().__init__()
        self.channels = channels
        self.num_frames = num_frames
        self.num_heads = num_heads

        self.qkv_proj = nn.Conv3d(channels, channels * 3, kernel_size=1)
        self.out_proj = nn.Conv3d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply temporal attention.

        Args:
            x: Input tensor of shape (batch, channels, frames, height, width)

        Returns:
            Output tensor with same shape
        """
        batch, channels, frames, height, width = x.shape

        # Generate Q, K, V
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=1)

        # Reshape for multi-head attention
        head_dim = channels // self.num_heads
        q = q.view(batch, self.num_heads, head_dim, frames, height * width)
        k = k.view(batch, self.num_heads, head_dim, frames, height * width)
        v = v.view(batch, self.num_heads, head_dim, frames, height * width)

        # Compute attention across temporal dimension
        scale = head_dim**-0.5
        attn = torch.einsum("bhctn,bhcTn->bhtT", q, k) * scale
        attn = F.softmax(attn, dim=-1)

        # Apply attention to values
        out = torch.einsum("bhtT,bhcTn->bhctn", attn, v)

        # Reshape back
        out = out.view(batch, channels, frames, height, width)
        out = self.out_proj(out)

        return x + out  # Residual connection


class MotionEstimator(nn.Module):
    """
    Estimates motion between consecutive frames for temporal consistency.
    """

    def __init__(self, in_channels: int = 3, hidden_dim: int = 64):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels * 2, hidden_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.flow_head = nn.Conv2d(hidden_dim, 2, 3, padding=1)

    def forward(
        self,
        frame1: torch.Tensor,
        frame2: torch.Tensor,
    ) -> torch.Tensor:
        """
        Estimate optical flow between two frames.

        Args:
            frame1: First frame (batch, channels, height, width)
            frame2: Second frame (batch, channels, height, width)

        Returns:
            Flow tensor (batch, 2, height, width)
        """
        x = torch.cat([frame1, frame2], dim=1)
        features = self.encoder(x)
        flow = self.flow_head(features)
        return flow

    def warp(
        self,
        frame: torch.Tensor,
        flow: torch.Tensor,
    ) -> torch.Tensor:
        """
        Warp frame using flow.

        Args:
            frame: Frame to warp (batch, channels, height, width)
            flow: Flow field (batch, 2, height, width)

        Returns:
            Warped frame
        """
        batch, channels, height, width = frame.shape

        # Create sampling grid
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, height, device=frame.device),
            torch.linspace(-1, 1, width, device=frame.device),
            indexing="ij",
        )
        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).repeat(batch, 1, 1, 1)

        # Normalize flow to [-1, 1]
        flow_norm = flow.clone()
        flow_norm[:, 0] = flow[:, 0] / (width / 2)
        flow_norm[:, 1] = flow[:, 1] / (height / 2)

        # Add flow to grid
        sampling_grid = grid + flow_norm.permute(0, 2, 3, 1)

        # Sample
        warped = F.grid_sample(
            frame,
            sampling_grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )

        return warped
