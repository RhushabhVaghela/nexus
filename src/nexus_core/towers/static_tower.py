"""
src/nexus_core/towers/static_tower.py

Static Tower with frozen weights support for efficient inference.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
import logging

from .base_tower import BaseTower, TowerConfig

logger = logging.getLogger(__name__)


@dataclass
class StaticTowerConfig(TowerConfig):
    """Configuration for Static Tower."""
    freeze_all: bool = True
    use_kvcache: bool = True
    quantize_weights: bool = True


class StaticTower(BaseTower):
    """Static Tower optimized for inference with frozen weights."""
    
    def __init__(self, config: StaticTowerConfig):
        super().__init__(config)
        self.config: StaticTowerConfig = config
        
        if config.freeze_all:
            for param in self.parameters():
                param.requires_grad = False
            self.eval()
            logger.info("All parameters frozen for inference")
    
    def forward(self, x: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                **kwargs) -> Dict[str, torch.Tensor]:
        """Forward pass optimized for inference."""
        self._stats["forward_passes"] += 1
        
        hidden_states = x
        all_hidden_states = []
        
        for i in range(self.config.num_layers):
            layer = self.get_layer(i)
            if layer is None:
                continue
            
            layer_output = layer(hidden_states, attention_mask=attention_mask)
            
            if isinstance(layer_output, dict):
                hidden_states = layer_output.get("hidden_states")
                if hidden_states is None:
                    hidden_states = layer_output
            else:
                hidden_states = layer_output
            
            self.track_layer_activation(i, hidden_states)
            all_hidden_states.append(hidden_states)
        
        if self.active_adapter:
            hidden_states = self.apply_adapter(hidden_states)
        
        return {
            "hidden_states": hidden_states,
            "all_hidden_states": all_hidden_states
        }


__all__ = ['StaticTower', 'StaticTowerConfig']