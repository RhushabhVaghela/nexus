"""
src/nexus_core/towers/dynamic_tower.py

Dynamic Tower with architecture loading and adaptive layer selection.
Supports dynamic architecture modification based on task requirements.

Features:
- Dynamic layer activation/deactivation
- Architecture search and optimization
- Task-specific layer routing
- Progressive layer unfreezing for fine-tuning
- Adaptive depth based on input complexity
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List, Tuple, Callable, Set
from dataclasses import dataclass
from pathlib import Path
import json
import logging

from .base_tower import BaseTower, TowerConfig, TowerMode

logger = logging.getLogger(__name__)


@dataclass
class DynamicTowerConfig(TowerConfig):
    """Configuration for Dynamic Tower."""
    # Dynamic architecture settings
    min_active_layers: int = 4
    max_active_layers: int = 32
    layer_selection_strategy: str = "adaptive"  # adaptive, uniform, entropy_based
    
    # Architecture loading
    architecture_config_path: Optional[str] = None
    pretrained_weights_path: Optional[str] = None
    
    # Adaptive settings
    complexity_threshold_low: float = 0.3
    complexity_threshold_high: float = 0.7
    adaptive_depth_enabled: bool = True
    
    # Progressive unfreezing
    progressive_unfreeze: bool = False
    unfreeze_rate: int = 4  # Layers per epoch
    
    # Load balancing
    balance_layer_usage: bool = True


class DynamicTower(BaseTower):
    """
    Dynamic Tower with adaptive architecture loading and layer management.
    
    This tower can:
    1. Load different architectures based on task requirements
    2. Dynamically activate/deactivate layers
    3. Adjust depth based on input complexity
    4. Support progressive unfreezing for fine-tuning
    
    Use Cases:
    - Multi-task learning with varying complexity
    - Resource-constrained inference
    - Progressive training from shallow to deep
    """
    
    def __init__(self, config: DynamicTowerConfig):
        """
        Initialize Dynamic Tower.
        
        Args:
            config: DynamicTowerConfig instance
        """
        super().__init__(config)
        self.config: DynamicTowerConfig = config
        
        # Layer activation state
        self._active_layers: Set[int] = set(range(config.num_layers))
        self._layer_gates: nn.ParameterDict = nn.ParameterDict()
        
        # Complexity estimator
        self._complexity_estimator: Optional[nn.Module] = None
        
        # Architecture cache
        self._architecture_history: List[Dict[str, Any]] = []
        
        # Layer usage statistics for load balancing
        self._layer_usage_count: Dict[int, int] = {i: 0 for i in range(config.num_layers)}
        
        # Progressive unfreezing state
        self._unfrozen_layers: Set[int] = set()
        self._current_unfreeze_epoch: int = 0
        
        # Build dynamic components
        self._build_complexity_estimator()
        self._initialize_layer_gates()
        
        logger.info(f"DynamicTower initialized with {config.num_layers} layers")
    
    def _build_complexity_estimator(self):
        """Build a lightweight complexity estimation network."""
        if not self.config.adaptive_depth_enabled:
            return
        
        # Simple MLP for complexity estimation
        self._complexity_estimator = nn.Sequential(
            nn.Linear(self.config.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def _initialize_layer_gates(self):
        """Initialize learnable layer gates."""
        for i in range(self.config.num_layers):
            # Gate value between 0 and 1
            gate = nn.Parameter(torch.tensor(1.0))
            self._layer_gates[str(i)] = gate
    
    def load_architecture(self, config_path: str, 
                          weights_path: Optional[str] = None) -> None:
        """
        Load architecture configuration from file.
        
        Args:
            config_path: Path to architecture config (JSON)
            weights_path: Optional path to pretrained weights
        """
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Architecture config not found: {config_path}")
        
        with open(config_file, 'r') as f:
            arch_config = json.load(f)
        
        # Update configuration
        if "num_layers" in arch_config:
            self.config.num_layers = arch_config["num_layers"]
        if "hidden_size" in arch_config:
            self.config.hidden_size = arch_config["hidden_size"]
        
        # Load layer activation pattern if specified
        if "active_layers" in arch_config:
            self._active_layers = set(arch_config["active_layers"])
        
        # Record in history
        self._architecture_history.append({
            "config_path": config_path,
            "weights_path": weights_path,
            "timestamp": self._get_timestamp(),
            "active_layers": list(self._active_layers)
        })
        
        # Load weights if provided
        if weights_path:
            self.load_pretrained_weights(weights_path)
        
        logger.info(f"Loaded architecture from {config_path}")
    
    def load_pretrained_weights(self, weights_path: str, 
                                strict: bool = False) -> None:
        """
        Load pretrained weights into the tower.
        
        Args:
            weights_path: Path to weights file
            strict: Whether to strictly enforce key matching
        """
        try:
            checkpoint = torch.load(weights_path, map_location=self.device)
            
            if "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint
            
            # Filter state dict for compatible layers
            model_dict = self.state_dict()
            filtered_dict = {}
            
            for k, v in state_dict.items():
                if k in model_dict and model_dict[k].shape == v.shape:
                    filtered_dict[k] = v
                elif not strict:
                    logger.warning(f"Skipping incompatible key: {k}")
            
            model_dict.update(filtered_dict)
            self.load_state_dict(model_dict, strict=False)
            
            logger.info(f"Loaded pretrained weights from {weights_path}")
            
        except Exception as e:
            logger.error(f"Failed to load weights: {e}")
            raise
    
    def estimate_complexity(self, x: torch.Tensor) -> float:
        """Estimate input complexity to determine required depth."""
        if self._complexity_estimator is None:
            return 0.5  # Default medium complexity
        
        with torch.no_grad():
            # Use mean pooling for sequence inputs
            if x.dim() > 2:
                pooled = x.mean(dim=1)
            else:
                pooled = x
            
            complexity = self._complexity_estimator(pooled)
            return complexity.mean().item()
    
    def select_active_layers(self, complexity: float) -> Set[int]:
        """
        Select which layers to activate based on complexity.
        
        Args:
            complexity: Complexity score (0-1)
            
        Returns:
            Set of active layer indices
        """
        if not self.config.adaptive_depth_enabled:
            return self._active_layers
        
        num_layers = self.config.num_layers
        min_layers = self.config.min_active_layers
        max_layers = min(self.config.max_active_layers, num_layers)
        
        # Determine number of layers based on complexity
        if complexity < self.config.complexity_threshold_low:
            target_layers = min_layers
        elif complexity > self.config.complexity_threshold_high:
            target_layers = max_layers
        else:
            # Linear interpolation
            ratio = (complexity - self.config.complexity_threshold_low) / \
                   (self.config.complexity_threshold_high - self.config.complexity_threshold_low)
            target_layers = int(min_layers + ratio * (max_layers - min_layers))
        
        # Select layers based on strategy
        if self.config.layer_selection_strategy == "uniform":
            # Evenly spaced layers
            if target_layers >= num_layers:
                return set(range(num_layers))
            step = num_layers / target_layers
            selected = set(int(i * step) for i in range(target_layers))
            
        elif self.config.layer_selection_strategy == "entropy_based":
            # Select layers with highest activation entropy
            layer_scores = {
                i: self._layer_usage_count[i] 
                for i in range(num_layers)
            }
            # Sort by usage (prefer less used for balancing)
            sorted_layers = sorted(layer_scores.keys(), 
                                  key=lambda x: layer_scores[x])
            selected = set(sorted_layers[:target_layers])
            
        else:  # adaptive
            # Use first N layers (common pattern in transformers)
            selected = set(range(target_layers))
        
        return selected
    
    def set_active_layers(self, layer_indices: Set[int]) -> None:
        """
        Manually set which layers are active.
        
        Args:
            layer_indices: Set of layer indices to activate
        """
        self._active_layers = layer_indices & set(range(self.config.num_layers))
        logger.info(f"Active layers set to: {sorted(self._active_layers)}")
    
    def enable_layer(self, layer_idx: int) -> None:
        """Enable a specific layer."""
        if 0 <= layer_idx < self.config.num_layers:
            self._active_layers.add(layer_idx)
    
    def disable_layer(self, layer_idx: int) -> None:
        """Disable a specific layer."""
        self._active_layers.discard(layer_idx)
    
    def progressive_unfreeze_step(self, epoch: Optional[int] = None) -> Set[int]:
        """
        Perform one step of progressive layer unfreezing.
        
        Args:
            epoch: Current training epoch (auto-incremented if None)
            
        Returns:
            Set of newly unfrozen layers
        """
        if not self.config.progressive_unfreeze:
            return set()
        
        if epoch is not None:
            self._current_unfreeze_epoch = epoch
        else:
            self._current_unfreeze_epoch += 1
        
        # Calculate how many layers to unfreeze
        layers_to_unfreeze = self.config.unfreeze_rate * self._current_unfreeze_epoch
        layers_to_unfreeze = min(layers_to_unfreeze, self.config.num_layers)
        
        # Unfreeze from top down (last layers first)
        newly_unfrozen = set()
        for i in range(self.config.num_layers - 1, 
                      self.config.num_layers - layers_to_unfreeze - 1, -1):
            if i not in self._unfrozen_layers:
                layer = self.get_layer(i)
                if layer is not None:
                    for param in layer.parameters():
                        param.requires_grad = True
                    self._unfrozen_layers.add(i)
                    newly_unfrozen.add(i)
        
        if newly_unfrozen:
            logger.info(f"Progressive unfreeze: layers {sorted(newly_unfrozen)}")
        
        return newly_unfrozen
    
    def forward(self, x: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None,
                **kwargs) -> Dict[str, torch.Tensor]:
        """
        Forward pass with dynamic layer selection.
        
        Args:
            x: Input tensor
            attention_mask: Optional attention mask
            **kwargs: Additional arguments
            
        Returns:
            Dictionary containing outputs
        """
        self._stats["forward_passes"] += 1
        
        # Estimate complexity if adaptive depth is enabled
        if self.config.adaptive_depth_enabled and self.training:
            complexity = self.estimate_complexity(x)
            active_layers = self.select_active_layers(complexity)
        else:
            active_layers = self._active_layers
        
        # Forward through active layers
        hidden_states = x
        all_hidden_states = []
        
        for i in sorted(active_layers):
            layer = self.get_layer(i)
            if layer is None:
                continue
            
            # Apply layer gate
            gate_value = torch.sigmoid(self._layer_gates[str(i)])
            
            # Forward through layer
            if self._gradient_checkpointing and self.training:
                layer_output = torch.utils.checkpoint.checkpoint(
                    layer, hidden_states, attention_mask
                )
            else:
                layer_output = layer(hidden_states, attention_mask=attention_mask)
            
            # Apply gating
            if isinstance(layer_output, dict):
                layer_hidden = layer_output.get("hidden_states", layer_output)
            else:
                layer_hidden = layer_output
            
            hidden_states = gate_value * layer_hidden + (1 - gate_value) * hidden_states
            
            # Track activation
            self.track_layer_activation(i, hidden_states)
            self._layer_usage_count[i] += 1
            all_hidden_states.append(hidden_states)
        
        # Apply adapter if active
        if self.active_adapter:
            hidden_states = self.apply_adapter(hidden_states)
        
        return {
            "hidden_states": hidden_states,
            "all_hidden_states": all_hidden_states,
            "active_layers": active_layers,
            "num_active_layers": len(active_layers)
        }
    
    def get_layer_usage_stats(self) -> Dict[str, Any]:
        """Get statistics on layer usage."""
        total_uses = sum(self._layer_usage_count.values())
        if total_uses == 0:
            return {"total_uses": 0, "distribution": {}}
        
        distribution = {
            i: count / total_uses 
            for i, count in self._layer_usage_count.items()
        }
        
        return {
            "total_uses": total_uses,
            "distribution": distribution,
            "most_used": max(self._layer_usage_count, key=self._layer_usage_count.get),
            "least_used": min(self._layer_usage_count, key=self._layer_usage_count.get),
        }
    
    def reset_layer_usage_stats(self) -> None:
        """Reset layer usage statistics."""
        self._layer_usage_count = {i: 0 for i in range(self.config.num_layers)}
    
    def save_architecture(self, path: str) -> None:
        """
        Save current architecture configuration.
        
        Args:
            path: Path to save configuration
        """
        config = {
            "num_layers": self.config.num_layers,
            "hidden_size": self.config.hidden_size,
            "active_layers": list(self._active_layers),
            "unfrozen_layers": list(self._unfrozen_layers),
            "layer_gates": {
                k: v.item() for k, v in self._layer_gates.items()
            },
            "layer_selection_strategy": self.config.layer_selection_strategy,
            "adaptive_depth_enabled": self.config.adaptive_depth_enabled,
        }
        
        with open(path, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Architecture saved to {path}")
    
    def _get_timestamp(self) -> str:
        """Get current timestamp string."""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get DynamicTower statistics."""
        base_stats = super().get_stats()
        dynamic_stats = {
            "active_layers": len(self._active_layers),
            "unfrozen_layers": len(self._unfrozen_layers),
            "layer_usage": self.get_layer_usage_stats(),
            "architecture_history_count": len(self._architecture_history),
        }
        return {**base_stats, **dynamic_stats}


__all__ = [
    'DynamicTower',
    'DynamicTowerConfig'
]