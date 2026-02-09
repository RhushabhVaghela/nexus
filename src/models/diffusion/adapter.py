"""
Diffusion Adapter - Integration with Nexus Tower Architecture

Provides adapter interfaces between diffusion models and the Nexus
student training framework, enabling knowledge distillation from
image generation teachers.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, List, Tuple
import logging

from src.core.adapters.base import BaseAdapter
from .image_pipeline import ImagePipeline, PipelineConfig

logger = logging.getLogger(__name__)


class DiffusionAdapter(BaseAdapter):
    """
    Adapter for integrating diffusion models into Nexus training.
    
    This adapter bridges the gap between diffusion teacher models and
    the Nexus student architecture, enabling:
    - Feature extraction from diffusion UNet/VAE
    - Knowledge distillation from image generation
    - Multi-modal training with vision-language alignment
    """
    
    def __init__(
        self,
        teacher_dim: int,
        student_dim: int,
        pipeline: Optional[ImagePipeline] = None,
        extract_features_from: str = "unet",
        feature_layers: Optional[List[int]] = None,
    ):
        """
        Args:
            teacher_dim: Dimension of teacher model features
            student_dim: Dimension of student model features
            pipeline: Pre-loaded ImagePipeline (optional)
            extract_features_from: Which component to extract features from ('unet', 'vae', 'text_encoder')
            feature_layers: Specific layers to extract features from
        """
        super().__init__(teacher_dim, student_dim)
        self.pipeline = pipeline
        self.extract_features_from = extract_features_from
        self.feature_layers = feature_layers or [-1, -2]
        self._feature_hooks = []
        self._extracted_features = {}
    
    def attach_pipeline(self, pipeline: ImagePipeline):
        """Attach a diffusion pipeline for feature extraction."""
        self.pipeline = pipeline
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward hooks for feature extraction."""
        if self.pipeline is None or self.pipeline.pipeline is None:
            return
        
        self._remove_hooks()  # Clear existing hooks
        
        pipe = self.pipeline.pipeline
        target_module = None
        
        if self.extract_features_from == "unet":
            target_module = getattr(pipe, "unet", None)
        elif self.extract_features_from == "vae":
            target_module = getattr(pipe, "vae", None)
        elif self.extract_features_from == "text_encoder":
            target_module = getattr(pipe, "text_encoder", None) or getattr(pipe, "text_encoder_2", None)
        
        if target_module is None:
            logger.warning(f"Could not find {self.extract_features_from} in pipeline")
            return
        
        # Register hooks on specific layers
        layers = self._get_layers(target_module)
        for idx in self.feature_layers:
            if abs(idx) <= len(layers):
                layer = layers[idx]
                hook = layer.register_forward_hook(self._make_hook(idx))
                self._feature_hooks.append(hook)
    
    def _get_layers(self, module: nn.Module) -> List[nn.Module]:
        """Extract layers from a module for hooking."""
        layers = []
        
        # Try common attribute names
        for attr in ["blocks", "layers", "down_blocks", "up_blocks", "mid_block"]:
            if hasattr(module, attr):
                sub = getattr(module, attr)
                if isinstance(sub, (list, nn.ModuleList)):
                    layers.extend(sub)
        
        # If no layers found, use all children
        if not layers:
            layers = list(module.children())
        
        return layers
    
    def _make_hook(self, layer_idx: int):
        """Create a hook function for a specific layer."""
        def hook_fn(module, input, output):
            # Store the output features
            if isinstance(output, tuple):
                output = output[0]
            self._extracted_features[f"layer_{layer_idx}"] = output.detach()
        return hook_fn
    
    def _remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self._feature_hooks:
            hook.remove()
        self._feature_hooks.clear()
        self._extracted_features.clear()
    
    def extract_features(
        self,
        prompt: str,
        num_inference_steps: int = 1,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Extract features from the diffusion model.
        
        Args:
            prompt: Text prompt to process
            num_inference_steps: Number of denoising steps (1 for fast feature extraction)
            **kwargs: Additional generation arguments
        
        Returns:
            Dictionary of extracted features by layer name
        """
        if self.pipeline is None:
            raise RuntimeError("No pipeline attached. Call attach_pipeline() first.")
        
        self._extracted_features.clear()
        
        # Run generation with hooks active
        with torch.no_grad():
            # Reduce steps for faster feature extraction
            self.pipeline.generate(
                prompt=prompt,
                num_inference_steps=num_inference_steps,
                **kwargs
            )
        
        return self._extracted_features.copy()
    
    def compute_distillation_loss(
        self,
        teacher_features: Dict[str, torch.Tensor],
        student_features: Dict[str, torch.Tensor],
        loss_type: str = "mse"
    ) -> torch.Tensor:
        """
        Compute distillation loss between teacher and student features.
        
        Args:
            teacher_features: Features from teacher model
            student_features: Features from student model
            loss_type: Type of loss ('mse', 'cosine', 'kl')
        
        Returns:
            Loss tensor
        """
        losses = []
        
        for layer_name in teacher_features.keys():
            if layer_name not in student_features:
                continue
            
            t_feat = teacher_features[layer_name]
            s_feat = student_features[layer_name]
            
            # Ensure same shape
            if t_feat.shape != s_feat.shape:
                t_feat = self._adapt_feature_shape(t_feat, s_feat.shape)
            
            if loss_type == "mse":
                loss = nn.functional.mse_loss(s_feat, t_feat)
            elif loss_type == "cosine":
                loss = 1 - nn.functional.cosine_similarity(
                    s_feat.flatten(1), t_feat.flatten(1), dim=1
                ).mean()
            elif loss_type == "kl":
                loss = nn.functional.kl_div(
                    nn.functional.log_softmax(s_feat.flatten(1), dim=-1),
                    nn.functional.softmax(t_feat.flatten(1), dim=-1),
                    reduction="batchmean"
                )
            else:
                raise ValueError(f"Unknown loss type: {loss_type}")
            
            losses.append(loss)
        
        return torch.stack(losses).mean() if losses else torch.tensor(0.0)
    
    def _adapt_feature_shape(
        self,
        feature: torch.Tensor,
        target_shape: Tuple[int, ...]
    ) -> torch.Tensor:
        """Adapt feature tensor to target shape."""
        # Project through adapter
        flat_feature = feature.flatten(1)
        adapted = self.forward(flat_feature)
        
        # Reshape to target
        return adapted.reshape(target_shape)
    
    def prepare_training_batch(
        self,
        prompts: List[str],
        images: Optional[List] = None,
    ) -> Dict[str, Any]:
        """
        Prepare a batch for training with the diffusion adapter.
        
        Args:
            prompts: List of text prompts
            images: Optional list of images for img2img training
        
        Returns:
            Training batch dictionary
        """
        batch = {
            "prompts": prompts,
            "images": images,
        }
        
        # Extract teacher features for each prompt
        teacher_features_list = []
        for prompt in prompts:
            features = self.extract_features(prompt, num_inference_steps=1)
            teacher_features_list.append(features)
        
        batch["teacher_features"] = teacher_features_list
        return batch
    
    def __del__(self):
        """Cleanup hooks on deletion."""
        self._remove_hooks()


class MultiScaleDiffusionAdapter(nn.Module):
    """
    Multi-scale adapter that extracts features at different resolutions
    from the diffusion UNet.
    """
    
    def __init__(
        self,
        teacher_dims: List[int],
        student_dim: int,
        num_scales: int = 3,
    ):
        super().__init__()
        self.num_scales = num_scales
        self.adapters = nn.ModuleList([
            DiffusionAdapter(t_dim, student_dim)
            for t_dim in teacher_dims[:num_scales]
        ])
    
    def forward(self, features_by_scale: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Process features at multiple scales.
        
        Args:
            features_by_scale: List of feature tensors at different scales
        
        Returns:
            List of adapted features
        """
        adapted = []
        for i, feat in enumerate(features_by_scale[:self.num_scales]):
            adapted_feat = self.adapters[i](feat)
            adapted.append(adapted_feat)
        return adapted
