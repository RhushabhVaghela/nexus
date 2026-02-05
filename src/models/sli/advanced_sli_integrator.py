"""
Advanced SLI Integrator for Nexus

Combines all advanced SLI components into a unified integration module:
- NVFP4 Streaming Loader for quantized layer loading
- QAD Distillation Loss for knowledge transfer
- Nested Update Scheduler for efficient training
- Hierarchical Layer Cache for optimal memory management

This module provides a production-ready API for streaming layer inference
with quantization, distillation, and nested learning.

Author: Nexus Team
"""

import logging
import os
from typing import Dict, Optional, Any, List, Tuple, Union, Callable
from dataclasses import dataclass, field
from pathlib import Path
import threading
import json

import torch
import torch.nn as nn
from tqdm import tqdm

from .nvfp4_loader import (
    NVFP4StreamingLoader,
    NVFP4Config,
    NVFP4Mode,
    QuantizedTensor,
)
from .qad_loss import (
    QADDistillationLoss,
    QADLossConfig,
    PerLayerQADLoss,
)
from .nested_scheduler import (
    NestedUpdateScheduler,
    NestedUpdateConfig,
    UpdateGroup,
)
from .hierarchical_cache import (
    HierarchicalLayerCache,
    HierarchicalCacheConfig,
    CacheTier,
)
from .layer_cache import LayerCache
from .io_optimizer import IOOptimizer, IOPriority, EnhancedPrefetchBuffer
from .sliding_window_buffer import (
    SlidingWindowBuffer,
    AdaptiveSlidingWindow,
    SlidingWindowConfig,
)
from .compressed_storage import (
    CompressedLayerStorage,
    LayerCompressor,
    CompressionConfig,
    CompressionAlgorithm,
)
from .storage_tier_manager import (
    StorageTierManager,
    StorageTierConfig,
    StorageTier,
)
from .exceptions import SLIError

logger = logging.getLogger(__name__)


class AdvancedSLIError(SLIError):
    """Raised when advanced SLI integration fails."""
    pass


@dataclass
class AdvancedSLIConfig:
    """Configuration for Advanced SLI Integrator.
    
    Attributes:
        nvfp4_config: NVFP4 quantization configuration
        qad_config: QAD distillation configuration
        nested_config: Nested update scheduler configuration
        cache_config: Hierarchical cache configuration
        sliding_window_config: Sliding window buffer configuration
        compression_config: Layer compression configuration
        storage_tier_config: Storage tier manager configuration
        enable_quantization: Enable NVFP4 quantization
        enable_distillation: Enable QAD distillation
        enable_nested_updates: Enable nested update scheduling
        enable_hierarchical_cache: Enable hierarchical caching
        enable_sliding_window: Enable sliding window buffer
        enable_compression: Enable layer compression
        enable_storage_tiering: Enable hot/cold storage tiering
        enable_enhanced_prefetch: Enable enhanced prefetch buffer
        sliding_window_size: int = 5
        device: Target device
        output_dir: Output directory for profiles
    """
    nvfp4_config: Optional[NVFP4Config] = None
    qad_config: Optional[QADLossConfig] = None
    nested_config: Optional[NestedUpdateConfig] = None
    cache_config: Optional[HierarchicalCacheConfig] = None
    sliding_window_config: Optional[SlidingWindowConfig] = None
    compression_config: Optional[CompressionConfig] = None
    storage_tier_config: Optional[StorageTierConfig] = None
    enable_quantization: bool = True
    enable_distillation: bool = True
    enable_nested_updates: bool = True
    enable_hierarchical_cache: bool = True
    enable_sliding_window: bool = True
    enable_compression: bool = True
    enable_storage_tiering: bool = True
    enable_enhanced_prefetch: bool = True
    sliding_window_size: int = 5
    device: str = "cuda"
    output_dir: str = "./advanced_sli_output"
    
    def __post_init__(self):
        """Initialize default configs if not provided."""
        if self.nvfp4_config is None:
            self.nvfp4_config = NVFP4Config(mode=NVFP4Mode.MIXED)
        if self.qad_config is None:
            self.qad_config = QADLossConfig()
        if self.nested_config is None:
            self.nested_config = NestedUpdateConfig()
        if self.cache_config is None:
            self.cache_config = HierarchicalCacheConfig()
        if self.sliding_window_config is None:
            self.sliding_window_config = SlidingWindowConfig()
        if self.compression_config is None:
            self.compression_config = CompressionConfig()
        if self.storage_tier_config is None:
            self.storage_tier_config = StorageTierConfig()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'nvfp4_config': self.nvfp4_config.to_dict() if self.nvfp4_config else None,
            'qad_config': self.qad_config.to_dict() if self.qad_config else None,
            'nested_config': self.nested_config.to_dict() if self.nested_config else None,
            'cache_config': self.cache_config.to_dict() if self.cache_config else None,
            'sliding_window_config': self.sliding_window_config.to_dict() if hasattr(self.sliding_window_config, 'to_dict') else None,
            'enable_quantization': self.enable_quantization,
            'enable_distillation': self.enable_distillation,
            'enable_nested_updates': self.enable_nested_updates,
            'enable_hierarchical_cache': self.enable_hierarchical_cache,
            'enable_sliding_window': self.enable_sliding_window,
            'enable_compression': self.enable_compression,
            'enable_storage_tiering': self.enable_storage_tiering,
            'enable_enhanced_prefetch': self.enable_enhanced_prefetch,
            'sliding_window_size': self.sliding_window_size,
            'device': self.device,
            'output_dir': self.output_dir,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AdvancedSLIConfig':
        """Create config from dictionary."""
        config = cls()
        if data.get('nvfp4_config'):
            config.nvfp4_config = NVFP4Config.from_dict(data['nvfp4_config'])
        if data.get('qad_config'):
            config.qad_config = QADLossConfig.from_dict(data['qad_config'])
        if data.get('nested_config'):
            config.nested_config = NestedUpdateConfig.from_dict(data['nested_config'])
        if data.get('cache_config'):
            config.cache_config = HierarchicalCacheConfig.from_dict(data['cache_config'])
        config.enable_quantization = data.get('enable_quantization', True)
        config.enable_distillation = data.get('enable_distillation', True)
        config.enable_nested_updates = data.get('enable_nested_updates', True)
        config.enable_hierarchical_cache = data.get('enable_hierarchical_cache', True)
        config.enable_sliding_window = data.get('enable_sliding_window', True)
        config.enable_compression = data.get('enable_compression', True)
        config.enable_storage_tiering = data.get('enable_storage_tiering', True)
        config.enable_enhanced_prefetch = data.get('enable_enhanced_prefetch', True)
        config.sliding_window_size = data.get('sliding_window_size', 5)
        config.device = data.get('device', 'cuda')
        config.output_dir = data.get('output_dir', './advanced_sli_output')
        return config


@dataclass
class LayerInfo:
    """Information about a cached/loaded layer."""
    layer_idx: int
    is_quantized: bool
    tier: Optional[str]
    size_bytes: int
    load_time_ms: float


class AdvancedSLIIntegrator:
    """Advanced SLI Integrator combining NVFP4, QAD, and Nested Learning.
    
    This integrator provides a unified interface for:
    1. Streaming layer loading with NVFP4 quantization
    2. Knowledge distillation from FP32 teacher to NVFP4 student
    3. Nested update scheduling for efficient training
    4. Hierarchical caching for optimal memory usage
    
    Example:
        >>> config = AdvancedSLIConfig(
        ...     enable_quantization=True,
        ...     enable_distillation=True,
        ...     enable_nested_updates=True,
        ... )
        >>> integrator = AdvancedSLIIntegrator(config)
        >>> 
        >>> # Load and quantize layers
        >>> for i in range(num_layers):
        ...     layer = integrator.load_layer(model_id, i)
        ...     output = layer(input_tensor)
        >>> 
        >>> # Training with distillation
        >>> loss = integrator.compute_distillation_loss(
        ...     student_logits, teacher_logits, labels
        ... )
        >>> 
        >>> # Check update schedule
        >>> if integrator.should_update(layer_idx, step):
        ...     update_layer(layer_idx)
    """
    
    def __init__(self, config: Optional[AdvancedSLIConfig] = None):
        """Initialize Advanced SLI Integrator.
        
        Args:
            config: Integration configuration
        """
        self.config = config or AdvancedSLIConfig()
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.nvfp4_loader: Optional[NVFP4StreamingLoader] = None
        self.qad_loss: Optional[QADDistillationLoss] = None
        self.nested_scheduler: Optional[NestedUpdateScheduler] = None
        self.hierarchical_cache: Optional[HierarchicalLayerCache] = None
        
        # New I/O optimization components
        self.sliding_window: Optional[SlidingWindowBuffer] = None
        self.compressed_storage: Optional[CompressedLayerStorage] = None
        self.storage_tier_manager: Optional[StorageTierManager] = None
        
        # Standard layer cache for compatibility
        self.layer_cache: Optional[LayerCache] = None
        self.io_optimizer: Optional[IOOptimizer] = None
        
        # Initialize if enabled
        if self.config.enable_quantization:
            self._init_nvfp4_loader()
        
        if self.config.enable_distillation:
            self._init_qad_loss()
        
        if self.config.enable_nested_updates:
            self._init_nested_scheduler()
        
        if self.config.enable_hierarchical_cache:
            self._init_hierarchical_cache()
        
        # Initialize new I/O optimization components
        if self.config.enable_sliding_window:
            self._init_sliding_window()
        
        if self.config.enable_compression:
            self._init_compressed_storage()
        
        if self.config.enable_storage_tiering:
            self._init_storage_tier_manager()
        
        if self.config.enable_enhanced_prefetch:
            self._init_enhanced_io_optimizer()
        
        # Statistics
        self._stats = {
            'layers_loaded': 0,
            'layers_quantized': 0,
            'layers_cached': 0,
            'training_steps': 0,
        }
        
        self._lock = threading.RLock()
        
        logger.info("AdvancedSLIIntegrator initialized")
        logger.info(f"  Quantization: {self.config.enable_quantization}")
        logger.info(f"  Distillation: {self.config.enable_distillation}")
        logger.info(f"  Nested Updates: {self.config.enable_nested_updates}")
        logger.info(f"  Hierarchical Cache: {self.config.enable_hierarchical_cache}")
        logger.info(f"  Sliding Window: {self.config.enable_sliding_window}")
        logger.info(f"  Compression: {self.config.enable_compression}")
        logger.info(f"  Storage Tiering: {self.config.enable_storage_tiering}")
        logger.info(f"  Enhanced Prefetch: {self.config.enable_enhanced_prefetch}")
    
    def _init_nvfp4_loader(self):
        """Initialize NVFP4 streaming loader."""
        cache_dir = str(self.output_dir / "nvfp4_cache")
        self.nvfp4_loader = NVFP4StreamingLoader(
            config=self.config.nvfp4_config,
            cache_dir=cache_dir,
            device=self.config.device
        )
        logger.info("NVFP4 loader initialized")
    
    def _init_qad_loss(self):
        """Initialize QAD distillation loss."""
        self.qad_loss = QADDistillationLoss(self.config.qad_config)
        logger.info("QAD loss initialized")
    
    def _init_nested_scheduler(self):
        """Initialize nested update scheduler."""
        self.nested_scheduler = NestedUpdateScheduler(self.config.nested_config)
        logger.info("Nested scheduler initialized")
    
    def _init_hierarchical_cache(self):
        """Initialize hierarchical cache."""
        self.hierarchical_cache = HierarchicalLayerCache(self.config.cache_config)
        logger.info("Hierarchical cache initialized")
    
    def _init_sliding_window(self):
        """Initialize sliding window buffer."""
        self.sliding_window = AdaptiveSlidingWindow(
            window_size=self.config.sliding_window_size,
            config=self.config.sliding_window_config,
        )
        logger.info(f"Sliding window initialized (size={self.config.sliding_window_size})")
    
    def _init_compressed_storage(self):
        """Initialize compressed layer storage."""
        storage_dir = str(self.output_dir / "compressed_layers")
        self.compressed_storage = CompressedLayerStorage(
            storage_dir=storage_dir,
            config=self.config.compression_config,
        )
        logger.info("Compressed storage initialized")
    
    def _init_storage_tier_manager(self):
        """Initialize storage tier manager."""
        self.storage_tier_manager = StorageTierManager(
            config=self.config.storage_tier_config,
        )
        logger.info("Storage tier manager initialized")
    
    def _init_enhanced_io_optimizer(self):
        """Initialize I/O optimizer with enhanced prefetch."""
        if self.hierarchical_cache is not None:
            self.io_optimizer = IOOptimizer(
                layer_cache=self.hierarchical_cache,
                enable_prefetch=True,
                use_enhanced_prefetch=True,
                prefetch_lookahead=5,
                max_concurrent_downloads=8,
                io_thread_count=8,
            )
            logger.info("Enhanced I/O optimizer initialized")
    
    def load_layer(
        self,
        model_id: str,
        layer_idx: int,
        layer_weights: Optional[Dict[str, torch.Tensor]] = None,
        is_attention: bool = False
    ) -> nn.Module:
        """Load a layer with full pipeline.
        
        Args:
            model_id: Model identifier
            layer_idx: Layer index
            layer_weights: Optional pre-loaded weights
            is_attention: Whether this is an attention layer
            
        Returns:
            Loaded layer module
        """
        layer_id = f"{model_id}_layer_{layer_idx}"
        
        # Try hierarchical cache first
        if self.hierarchical_cache is not None:
            layer = self.hierarchical_cache.get_layer(layer_id, self.config.device)
            if layer is not None:
                logger.debug(f"Layer {layer_id} loaded from hierarchical cache")
                return layer
        
        # Try NVFP4 loader
        if self.nvfp4_loader is not None:
            layer = self.nvfp4_loader.load_layer(
                model_id, layer_idx, layer_weights
            )
            
            # Quantize if needed
            if self.config.enable_quantization and layer is not None:
                layer = self.nvfp4_loader.quantize_layer(
                    layer,
                    is_attention=is_attention,
                    layer_name=layer_id
                )
                self._stats['layers_quantized'] += 1
            
            # Cache if hierarchical cache is enabled
            if layer is not None and self.hierarchical_cache is not None:
                priority = 8 if is_attention else 5
                self.hierarchical_cache.cache_layer(
                    layer_id, layer, priority=priority
                )
        else:
            # Fallback: build layer from weights
            layer = self._build_layer_from_weights(layer_weights or {})
        
        with self._lock:
            self._stats['layers_loaded'] += 1
        
        return layer.to(self.config.device) if layer is not None else None
    
    def _build_layer_from_weights(
        self,
        weights: Dict[str, torch.Tensor]
    ) -> nn.Module:
        """Build layer from weight dictionary."""
        layer = nn.Module()
        for name, tensor in weights.items():
            if tensor.requires_grad:
                setattr(layer, name, nn.Parameter(tensor))
            else:
                layer.register_buffer(name, tensor)
        return layer
    
    def quantize_layer(self, layer: nn.Module, is_attention: bool = False) -> nn.Module:
        """Quantize a layer using NVFP4.
        
        Args:
            layer: Layer to quantize
            is_attention: Whether this is an attention layer
            
        Returns:
            Quantized layer
        """
        if self.nvfp4_loader is None:
            raise AdvancedSLIError("NVFP4 loader not initialized")
        
        return self.nvfp4_loader.quantize_layer(layer, is_attention)
    
    def dequantize_layer(self, layer: nn.Module) -> nn.Module:
        """Dequantize a layer from NVFP4.
        
        Args:
            layer: Quantized layer
            
        Returns:
            Dequantized layer
        """
        if self.nvfp4_loader is None:
            raise AdvancedSLIError("NVFP4 loader not initialized")
        
        return self.nvfp4_loader.dequantize_layer(layer)
    
    def compute_distillation_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        hidden_student: Optional[torch.Tensor] = None,
        hidden_teacher: Optional[torch.Tensor] = None,
        attention_student: Optional[torch.Tensor] = None,
        attention_teacher: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute QAD distillation loss.
        
        Args:
            student_logits: Student model logits
            teacher_logits: Teacher model logits
            labels: Ground truth labels
            hidden_student: Student hidden states
            hidden_teacher: Teacher hidden states
            attention_student: Student attention outputs
            attention_teacher: Teacher attention outputs
            mask: Attention mask
            
        Returns:
            Distillation loss
        """
        if self.qad_loss is None:
            raise AdvancedSLIError("QAD loss not initialized")
        
        return self.qad_loss(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            labels=labels,
            hidden_student=hidden_student,
            hidden_teacher=hidden_teacher,
            attention_student=attention_student,
            attention_teacher=attention_teacher,
            mask=mask
        )
    
    def should_update(self, layer_idx: int, step: Optional[int] = None) -> bool:
        """Check if layer should be updated at current step.
        
        Args:
            layer_idx: Layer index
            step: Current step (uses internal counter if None)
            
        Returns:
            True if layer should be updated
        """
        if not self.config.enable_nested_updates:
            return True
        
        if self.nested_scheduler is None:
            return True
        
        return self.nested_scheduler.should_update(layer_idx, step)
    
    def get_update_layers(self, step: Optional[int] = None) -> List[int]:
        """Get list of layers to update at current step.
        
        Args:
            step: Current step
            
        Returns:
            List of layer indices
        """
        if not self.config.enable_nested_updates:
            return []
        
        if self.nested_scheduler is None:
            return []
        
        return self.nested_scheduler.get_update_layers(step)
    
    def step_scheduler(self):
        """Advance nested update scheduler."""
        if self.nested_scheduler is not None:
            self.nested_scheduler.step()
        
        with self._lock:
            self._stats['training_steps'] += 1
    
    def prefetch_layers(self, model_id: str, layer_indices: List[int]):
        """Prefetch layers into cache.
        
        Args:
            model_id: Model identifier
            layer_indices: List of layer indices to prefetch
        """
        if self.hierarchical_cache is None:
            return
        
        layer_ids = [f"{model_id}_layer_{idx}" for idx in layer_indices]
        self.hierarchical_cache.prefetch_layers(layer_ids)
    
    def get_layer_info(self, layer_id: str) -> Optional[LayerInfo]:
        """Get information about a cached layer.
        
        Args:
            layer_id: Layer identifier
            
        Returns:
            Layer information or None
        """
        # Check hierarchical cache
        if self.hierarchical_cache is not None:
            entry = self.hierarchical_cache._entries.get(layer_id)
            if entry is not None:
                return LayerInfo(
                    layer_idx=int(layer_id.split('_')[-1]),
                    is_quantized=True,
                    tier=entry.tier.value,
                    size_bytes=entry.size_bytes,
                    load_time_ms=0.0
                )
        
        return None
    
    def load_layer_with_sliding_window(
        self,
        model_id: str,
        layer_idx: int,
        total_layers: int,
        auto_slide: bool = True
    ) -> Optional[nn.Module]:
        """Load a layer using sliding window buffer.
        
        Args:
            model_id: Model identifier
            layer_idx: Layer index
            total_layers: Total number of layers
            auto_slide: Whether to auto-advance window
            
        Returns:
            Layer module or None
        """
        if self.sliding_window is None:
            return None
        
        # Initialize window if needed
        if self.sliding_window._current_model_id != model_id:
            self.sliding_window.initialize_window(model_id, start_layer=layer_idx, total_layers=total_layers)
        
        # Get layer from window
        layer = self.sliding_window.get_layer(model_id, layer_idx, auto_advance=auto_slide)
        
        return layer
    
    def get_stats(self) -> Dict[str, Any]:
        """Get integration statistics."""
        stats = self._stats.copy()
        
        if self.nvfp4_loader is not None:
            stats['nvfp4'] = self.nvfp4_loader.get_stats()
        
        if self.qad_loss is not None:
            stats['qad'] = self.qad_loss.get_stats()
        
        if self.nested_scheduler is not None:
            stats['nested'] = self.nested_scheduler.get_stats()
        
        if self.hierarchical_cache is not None:
            stats['cache'] = self.hierarchical_cache.get_stats()
        
        if self.sliding_window is not None:
            stats['sliding_window'] = self.sliding_window.get_stats()
        
        if self.compressed_storage is not None:
            stats['compression'] = self.compressed_storage.get_compression_stats()
        
        if self.storage_tier_manager is not None:
            stats['storage_tiers'] = self.storage_tier_manager.get_stats()
        
        if self.io_optimizer is not None:
            stats['io_optimizer'] = self.io_optimizer.get_stats()
        
        return stats
    
    def save_config(self, path: Optional[str] = None):
        """Save configuration to file.
        
        Args:
            path: Output path (default: output_dir/config.json)
        """
        if path is None:
            path = self.output_dir / "config.json"
        
        with open(path, 'w') as f:
            json.dump(self.config.to_dict(), f, indent=2)
        
        logger.info(f"Config saved to {path}")
    
    def clear_cache(self):
        """Clear all caches."""
        if self.hierarchical_cache is not None:
            self.hierarchical_cache.clear()
        
        if self.nvfp4_loader is not None:
            self.nvfp4_loader.clear_cache()
        
        logger.info("All caches cleared")
    
    def export_model_profile(self, model_id: str, num_layers: int) -> Dict[str, Any]:
        """Export model profile for efficient loading.
        
        Args:
            model_id: Model identifier
            num_layers: Number of layers
            
        Returns:
            Model profile dictionary
        """
        profile = {
            'model_id': model_id,
            'num_layers': num_layers,
            'config': self.config.to_dict(),
            'layer_groups': {},
        }
        
        if self.nested_scheduler is not None:
            profile['layer_groups'] = {
                str(i): self.nested_scheduler.get_group(i).value
                for i in range(num_layers)
            }
        
        return profile
    
    def run_inference_pipeline(
        self,
        model_id: str,
        input_tensor: torch.Tensor,
        num_layers: int,
        layer_factory: Callable[[int], nn.Module]
    ) -> torch.Tensor:
        """Run full inference pipeline.
        
        Args:
            model_id: Model identifier
            input_tensor: Input tensor
            num_layers: Number of layers
            layer_factory: Factory function to create layers
            
        Returns:
            Output tensor
        """
        x = input_tensor
        
        # Prefetch first few layers
        prefetch_indices = list(range(min(3, num_layers)))
        self.prefetch_layers(model_id, prefetch_indices)
        
        for layer_idx in tqdm(range(num_layers), desc="Processing layers"):
            # Load layer
            layer = self.load_layer(model_id, layer_idx)
            
            if layer is None:
                # Fallback to factory
                layer = layer_factory(layer_idx).to(self.config.device)
            
            # Forward pass
            with torch.no_grad():
                x = layer(x)
            
            # Prefetch next layers
            next_indices = list(range(layer_idx + 1, min(layer_idx + 4, num_layers)))
            self.prefetch_layers(model_id, next_indices)
        
        return x


# Factory functions
def create_advanced_integrator(
    mode: str = "balanced",
    device: str = "cuda",
    **kwargs
) -> AdvancedSLIIntegrator:
    """Create AdvancedSLIIntegrator with preset configurations.
    
    Args:
        mode: Preset mode ("fast", "balanced", "quality")
        device: Target device
        **kwargs: Additional config overrides
        
    Returns:
        Configured AdvancedSLIIntegrator
    """
    if mode == "fast":
        config = AdvancedSLIConfig(
            nvfp4_config=NVFP4Config(mode=NVFP4Mode.SOFTWARE),
            qad_config=QADLossConfig(temperature=2.0, alpha=0.5),
            nested_config=NestedUpdateConfig(
                medium_interval=20,
                slow_interval=200
            ),
            device=device,
            **kwargs
        )
    elif mode == "quality":
        config = AdvancedSLIConfig(
            nvfp4_config=NVFP4Config(mode=NVFP4Mode.MIXED),
            qad_config=QADLossConfig(temperature=1.0, alpha=0.9),
            nested_config=NestedUpdateConfig(
                medium_interval=5,
                slow_interval=50
            ),
            device=device,
            **kwargs
        )
    else:  # balanced
        config = AdvancedSLIConfig(
            nvfp4_config=NVFP4Config(mode=NVFP4Mode.MIXED),
            qad_config=QADLossConfig(temperature=1.5, alpha=0.7),
            nested_config=NestedUpdateConfig(),
            device=device,
            **kwargs
        )
    
    return AdvancedSLIIntegrator(config)


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("Testing Advanced SLI Integrator")
    print("=" * 60)
    
    # Create integrator in balanced mode
    integrator = create_advanced_integrator(mode="balanced")
    
    # Show config
    print(f"\nConfiguration:")
    config_dict = integrator.config.to_dict()
    print(f"  Quantization: {config_dict['enable_quantization']}")
    print(f"  Distillation: {config_dict['enable_distillation']}")
    print(f"  Nested Updates: {config_dict['enable_nested_updates']}")
    print(f"  Hierarchical Cache: {config_dict['enable_hierarchical_cache']}")
    
    # Create test layers
    print("\nLoading test layers...")
    for i in range(3):
        layer = nn.Linear(1024, 1024)
        quantized = integrator.quantize_layer(layer, is_attention=(i == 0))
        print(f"  Layer {i}: quantized={quantized is not None}")
    
    # Test distillation
    print("\nTesting distillation...")
    student_logits = torch.randn(2, 1000)
    teacher_logits = torch.randn(2, 1000)
    labels = torch.randint(0, 1000, (2,))
    
    loss = integrator.compute_distillation_loss(
        student_logits, teacher_logits, labels
    )
    print(f"  Loss: {loss.item():.4f}")
    
    # Test nested scheduler
    print("\nTesting nested scheduler...")
    integrator.nested_scheduler = NestedUpdateScheduler(
        NestedUpdateConfig(fast_layers={0, 1}, medium_layers={2, 3}, slow_layers={4}),
        num_layers=5
    )
    
    for step in range(15):
        update_layers = integrator.get_update_layers(step)
        integrator.step_scheduler()
        if step < 5:
            print(f"  Step {step:2d}: Update layers {update_layers}")
    
    # Show stats
    print(f"\nStats: {integrator.get_stats()}")
    
    print("\n" + "=" * 60)
