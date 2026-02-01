"""
src/nexus_core/towers/base_tower.py

Base Tower implementation with full layer orchestration capabilities.
Provides the foundation for all tower types (Static, Dynamic, MoE, Router).

Features:
- Layer-wise activation tracking and caching
- Adapter management with hot-swapping
- Teacher model integration with frozen weights
- Gradient checkpointing for memory efficiency
- Activation checkpointing for debugging
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Callable, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
from collections import OrderedDict

logger = logging.getLogger(__name__)


class TowerMode(Enum):
    """Operational modes for towers."""
    INFERENCE = "inference"
    TRAINING = "training"
    DISTILLATION = "distillation"  # Student learning from teacher
    EVALUATION = "evaluation"


@dataclass
class TowerConfig:
    """Configuration for tower initialization."""
    name: str = "base_tower"
    hidden_size: int = 4096
    num_layers: int = 32
    num_attention_heads: int = 32
    intermediate_size: int = 14336
    max_position_embeddings: int = 32768
    vocab_size: int = 128256
    rms_norm_eps: float = 1e-5
    rope_theta: float = 500000.0
    use_cache: bool = True
    
    # Memory optimization settings
    gradient_checkpointing: bool = False
    activation_checkpointing: bool = False
    use_flash_attention: bool = True
    
    # Adapter settings
    adapter_dim: int = 64
    num_adapters: int = 4
    
    # Quantization settings
    load_in_4bit: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True


@dataclass
class LayerState:
    """State tracking for a single layer."""
    layer_idx: int
    activated: bool = False
    activation_count: int = 0
    last_activation_time: Optional[float] = None
    output_cache: Optional[torch.Tensor] = None
    gradient_norm: Optional[float] = None
    
    def record_activation(self, output: torch.Tensor):
        """Record layer activation."""
        import time
        self.activated = True
        self.activation_count += 1
        self.last_activation_time = time.time()
        self.output_cache = output.detach().clone()
    
    def clear_cache(self):
        """Clear cached output."""
        self.output_cache = None


class BaseTower(nn.Module, ABC):
    """
    Base Tower class with layer orchestration capabilities.
    
    This class provides:
    1. Layer-wise tracking and management
    2. Adapter integration and hot-swapping
    3. Teacher model loading and frozen inference
    4. Activation caching for debugging
    5. Gradient checkpointing for memory efficiency
    
    All specialized towers (Static, Dynamic, MoE, Router) inherit from this base.
    """
    
    def __init__(self, config: Union[TowerConfig, Dict[str, Any]]):
        """
        Initialize the base tower.
        
        Args:
            config: Tower configuration (TowerConfig object or dict)
        """
        super().__init__()
        
        # Convert dict to TowerConfig if needed
        if isinstance(config, dict):
            self.config = TowerConfig(**config)
        else:
            self.config = config
        
        # Layer tracking
        self.layer_states: Dict[int, LayerState] = {}
        self._layer_modules: nn.ModuleDict = nn.ModuleDict()
        
        # Adapter management
        self.adapters: nn.ModuleDict = nn.ModuleDict()
        self.active_adapter: Optional[str] = None
        
        # Teacher model (frozen)
        self.frozen_teacher: Optional[nn.Module] = None
        self.teacher_outputs: Optional[Dict[str, torch.Tensor]] = None
        
        # Operational mode
        self.mode: TowerMode = TowerMode.INFERENCE
        
        # Activation cache for debugging/analysis
        self._activation_cache: Dict[str, List[torch.Tensor]] = {}
        self._cache_enabled: bool = False
        
        # Gradient checkpointing
        self._gradient_checkpointing: bool = self.config.gradient_checkpointing
        
        # Statistics
        self._stats = {
            "forward_passes": 0,
            "adapter_switches": 0,
            "teacher_calls": 0,
            "cache_hits": 0,
        }
        
        # Device management
        self._device: Optional[torch.device] = None
        
        logger.info(f"Initialized {self.config.name} tower with {self.config.num_layers} layers")
    
    @property
    def device(self) -> torch.device:
        """Get the device of the tower."""
        if self._device is None:
            # Try to infer from parameters
            try:
                self._device = next(self.parameters()).device
            except StopIteration:
                self._device = torch.device("cpu")
        return self._device
    
    @abstractmethod
    def forward(self, x: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None,
                **kwargs) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the tower.
        
        Args:
            x: Input tensor
            attention_mask: Optional attention mask
            **kwargs: Additional arguments
            
        Returns:
            Dictionary containing outputs (e.g., hidden_states, logits)
        """
        pass
    
    def load_teacher(self, teacher_model: nn.Module, 
                     freeze: bool = True,
                     device_map: str = "auto") -> None:
        """
        Load a frozen teacher model for distillation.
        
        Args:
            teacher_model: The teacher model to load
            freeze: Whether to freeze teacher parameters
            device_map: Device mapping strategy
        """
        self.frozen_teacher = teacher_model
        
        if freeze:
            for param in self.frozen_teacher.parameters():
                param.requires_grad = False
            self.frozen_teacher.eval()
        
        logger.info(f"Loaded teacher model with {sum(1 for _ in self.frozen_teacher.parameters())} parameters")
        self._stats["teacher_calls"] = 0
    
    def get_teacher_output(self, x: torch.Tensor, 
                          **kwargs) -> Optional[Dict[str, torch.Tensor]]:
        """
        Get output from teacher model.
        
        Args:
            x: Input tensor
            **kwargs: Additional arguments for teacher
            
        Returns:
            Teacher output dictionary or None if no teacher loaded
        """
        if self.frozen_teacher is None:
            return None
        
        with torch.no_grad():
            output = self.frozen_teacher(x, **kwargs)
            self._stats["teacher_calls"] += 1
            
            # Convert to dict if needed
            if not isinstance(output, dict):
                output = {"hidden_states": output}
            
            return output
    
    def add_adapter(self, adapter_name: str, 
                    adapter_module: nn.Module,
                    set_active: bool = False) -> None:
        """
        Add an adapter module to the tower.
        
        Args:
            adapter_name: Unique name for the adapter
            adapter_module: The adapter module to add
            set_active: Whether to activate this adapter immediately
        """
        self.adapters[adapter_name] = adapter_module
        logger.info(f"Added adapter '{adapter_name}'")
        
        if set_active or self.active_adapter is None:
            self.set_active_adapter(adapter_name)
    
    def remove_adapter(self, adapter_name: str) -> bool:
        """
        Remove an adapter from the tower.
        
        Args:
            adapter_name: Name of the adapter to remove
            
        Returns:
            True if adapter was removed, False if not found
        """
        if adapter_name not in self.adapters:
            logger.warning(f"Adapter '{adapter_name}' not found")
            return False
        
        del self.adapters[adapter_name]
        
        if self.active_adapter == adapter_name:
            self.active_adapter = next(iter(self.adapters.keys())) if self.adapters else None
        
        logger.info(f"Removed adapter '{adapter_name}'")
        return True
    
    def set_active_adapter(self, adapter_name: str) -> bool:
        """
        Set the active adapter for forward passes.
        
        Args:
            adapter_name: Name of the adapter to activate
            
        Returns:
            True if adapter was activated, False if not found
        """
        if adapter_name not in self.adapters:
            logger.warning(f"Cannot activate adapter '{adapter_name}': not found")
            return False
        
        self.active_adapter = adapter_name
        self._stats["adapter_switches"] += 1
        logger.debug(f"Activated adapter '{adapter_name}'")
        return True
    
    def get_adapter(self, adapter_name: Optional[str] = None) -> Optional[nn.Module]:
        """
        Get an adapter module.
        
        Args:
            adapter_name: Name of the adapter (defaults to active adapter)
            
        Returns:
            The adapter module or None if not found
        """
        name = adapter_name or self.active_adapter
        if name is None:
            return None
        return self.adapters.get(name)
    
    def apply_adapter(self, x: torch.Tensor, 
                      adapter_name: Optional[str] = None) -> torch.Tensor:
        """
        Apply an adapter to the input.
        
        Args:
            x: Input tensor
            adapter_name: Adapter to apply (defaults to active)
            
        Returns:
            Transformed tensor
        """
        adapter = self.get_adapter(adapter_name)
        if adapter is None:
            return x
        return adapter(x)
    
    def list_adapters(self) -> List[str]:
        """List all registered adapter names."""
        return list(self.adapters.keys())
    
    def register_layer(self, layer_idx: int, layer_module: nn.Module) -> None:
        """
        Register a layer module for tracking.
        
        Args:
            layer_idx: Index of the layer
            layer_module: The layer module
        """
        self._layer_modules[str(layer_idx)] = layer_module
        self.layer_states[layer_idx] = LayerState(layer_idx=layer_idx)
    
    def get_layer(self, layer_idx: int) -> Optional[nn.Module]:
        """Get a layer module by index."""
        return self._layer_modules.get(str(layer_idx))
    
    def track_layer_activation(self, layer_idx: int, 
                               output: torch.Tensor) -> None:
        """
        Track activation of a specific layer.
        
        Args:
            layer_idx: Index of the activated layer
            output: Output tensor from the layer
        """
        if layer_idx in self.layer_states:
            self.layer_states[layer_idx].record_activation(output)
        
        # Cache for debugging if enabled
        if self._cache_enabled:
            key = f"layer_{layer_idx}"
            if key not in self._activation_cache:
                self._activation_cache[key] = []
            self._activation_cache[key].append(output.detach().cpu())
    
    def get_layer_stats(self, layer_idx: Optional[int] = None) -> Dict[str, Any]:
        """
        Get statistics for layer activations.
        
        Args:
            layer_idx: Specific layer index (None for all layers)
            
        Returns:
            Dictionary of layer statistics
        """
        if layer_idx is not None:
            state = self.layer_states.get(layer_idx)
            if state:
                return {
                    "layer_idx": state.layer_idx,
                    "activated": state.activated,
                    "activation_count": state.activation_count,
                    "last_activation_time": state.last_activation_time
                }
            return {}
        
        # Return stats for all layers
        return {
            idx: {
                "activated": state.activated,
                "activation_count": state.activation_count
            }
            for idx, state in self.layer_states.items()
        }
    
    def clear_layer_caches(self) -> None:
        """Clear all layer activation caches."""
        for state in self.layer_states.values():
            state.clear_cache()
        self._activation_cache.clear()
        logger.debug("Cleared all layer caches")
    
    def set_mode(self, mode: Union[TowerMode, str]) -> None:
        """
        Set the operational mode of the tower.
        
        Args:
            mode: Mode to set (TowerMode enum or string)
        """
        if isinstance(mode, str):
            mode = TowerMode(mode)
        
        self.mode = mode
        
        if mode == TowerMode.TRAINING:
            self.train()
        else:
            self.eval()
        
        logger.info(f"Tower mode set to: {mode.value}")
    
    def enable_activation_caching(self) -> None:
        """Enable activation caching for debugging."""
        self._cache_enabled = True
        logger.info("Activation caching enabled")
    
    def disable_activation_caching(self) -> None:
        """Disable activation caching."""
        self._cache_enabled = False
        self.clear_layer_caches()
        logger.info("Activation caching disabled")
    
    def get_activation_cache(self, layer_idx: Optional[int] = None) -> Dict[str, List[torch.Tensor]]:
        """
        Get cached activations.
        
        Args:
            layer_idx: Specific layer (None for all)
            
        Returns:
            Dictionary of cached activations
        """
        if layer_idx is not None:
            key = f"layer_{layer_idx}"
            return {key: self._activation_cache.get(key, [])}
        
        return dict(self._activation_cache)
    
    def enable_gradient_checkpointing(self) -> None:
        """Enable gradient checkpointing for memory efficiency."""
        self._gradient_checkpointing = True
        logger.info("Gradient checkpointing enabled")
    
    def disable_gradient_checkpointing(self) -> None:
        """Disable gradient checkpointing."""
        self._gradient_checkpointing = False
        logger.info("Gradient checkpointing disabled")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get tower statistics."""
        return {
            **self._stats,
            "num_adapters": len(self.adapters),
            "active_adapter": self.active_adapter,
            "mode": self.mode.value,
            "num_layers_tracked": len(self.layer_states),
            "teacher_loaded": self.frozen_teacher is not None
        }
    
    def reset_stats(self) -> None:
        """Reset all statistics."""
        self._stats = {
            "forward_passes": 0,
            "adapter_switches": 0,
            "teacher_calls": 0,
            "cache_hits": 0,
        }
        for state in self.layer_states.values():
            state.activation_count = 0
    
    def save_checkpoint(self, path: str, 
                        additional_data: Optional[Dict[str, Any]] = None) -> None:
        """
        Save tower checkpoint.
        
        Args:
            path: Path to save checkpoint
            additional_data: Additional data to save
        """
        checkpoint = {
            "config": self.config.__dict__,
            "state_dict": self.state_dict(),
            "adapters": {name: adapter.state_dict() 
                        for name, adapter in self.adapters.items()},
            "active_adapter": self.active_adapter,
            "stats": self._stats,
            "layer_states": {
                idx: {
                    "activation_count": state.activation_count,
                    "activated": state.activated
                }
                for idx, state in self.layer_states.items()
            }
        }
        
        if additional_data:
            checkpoint.update(additional_data)
        
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved to: {path}")
    
    def load_checkpoint(self, path: str, 
                        load_adapters: bool = True,
                        strict: bool = True) -> Dict[str, Any]:
        """
        Load tower checkpoint.
        
        Args:
            path: Path to checkpoint
            load_adapters: Whether to load adapter weights
            strict: Strict state dict loading
            
        Returns:
            Checkpoint dictionary
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        # Load main state dict
        self.load_state_dict(checkpoint["state_dict"], strict=strict)
        
        # Load adapters
        if load_adapters and "adapters" in checkpoint:
            for name, adapter_state in checkpoint["adapters"].items():
                if name in self.adapters:
                    self.adapters[name].load_state_dict(adapter_state)
            
            if "active_adapter" in checkpoint:
                self.set_active_adapter(checkpoint["active_adapter"])
        
        # Load stats
        if "stats" in checkpoint:
            self._stats.update(checkpoint["stats"])
        
        logger.info(f"Checkpoint loaded from: {path}")
        return checkpoint
    
    def get_memory_footprint(self) -> Dict[str, float]:
        """
        Get memory usage statistics.
        
        Returns:
            Dictionary with memory usage in MB
        """
        memory_stats = {
            "model_parameters_mb": 0.0,
            "adapters_mb": 0.0,
            "activation_cache_mb": 0.0,
            "total_mb": 0.0
        }
        
        # Model parameters
        param_bytes = sum(p.numel() * p.element_size() 
                         for p in self.parameters())
        memory_stats["model_parameters_mb"] = param_bytes / (1024 ** 2)
        
        # Adapters
        adapter_bytes = sum(p.numel() * p.element_size() 
                           for adapter in self.adapters.values() 
                           for p in adapter.parameters())
        memory_stats["adapters_mb"] = adapter_bytes / (1024 ** 2)
        
        # Activation cache
        cache_bytes = sum(
            t.numel() * t.element_size()
            for cache_list in self._activation_cache.values()
            for t in cache_list
        )
        memory_stats["activation_cache_mb"] = cache_bytes / (1024 ** 2)
        
        memory_stats["total_mb"] = (
            memory_stats["model_parameters_mb"] +
            memory_stats["adapters_mb"] +
            memory_stats["activation_cache_mb"]
        )
        
        return memory_stats
    
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"name='{self.config.name}', "
            f"layers={self.config.num_layers}, "
            f"adapters={len(self.adapters)}, "
            f"mode={self.mode.value}"
            f")"
        )


__all__ = [
    'BaseTower',
    'TowerConfig',
    'TowerMode',
    'LayerState'
]
