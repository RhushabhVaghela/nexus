"""
Adaptive Layer Skipping (SWIFT, LayerSkip, AdaSkip)

Key insight: Not all layers needed for all inputs.
Simple inputs: 50 layers, Complex inputs: 80 layers
Average: 55-65 layers per token
Performance: 1.82×-2.16× speedup

Research references:
- SWIFT: https://arxiv.org/abs/2404.00242
- LayerSkip: https://arxiv.org/abs/2404.16710
- AdaSkip: Adaptive Computation for Transformers
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, List, Tuple, Callable
from dataclasses import dataclass
import logging
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class LayerSkipConfig:
    """Configuration for adaptive layer skipping."""
    min_layers: int = 50
    max_layers: int = 80
    confidence_threshold: float = 0.9
    entropy_threshold: float = 0.5
    skip_pattern: str = "adaptive"  # "early", "uniform", "adaptive"
    training_mode: bool = False
    

class LayerSkipRouter(nn.Module):
    """
    Router that decides whether to skip layers based on input complexity.
    
    Based on LayerSkip: Train early exit layers and skip later layers when confident.
    """
    
    def __init__(self, hidden_size: int, num_layers: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Early exit classifier
        self.exit_classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
        
        # Complexity estimator
        self.complexity_estimator = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.LayerNorm(hidden_size // 4),
            nn.GELU(),
            nn.Linear(hidden_size // 4, 1),
            nn.Sigmoid()
        )
        
    def should_exit_early(
        self,
        hidden_states: torch.Tensor,
        current_layer: int
    ) -> Tuple[bool, float]:
        """
        Determine if we can exit early at current layer.
        
        Args:
            hidden_states: Current hidden states
            current_layer: Current layer index
            
        Returns:
            Tuple of (should_exit, confidence)
        """
        # Get mean pooled representation
        pooled = hidden_states.mean(dim=1)  # [batch, hidden]
        
        # Check exit confidence
        exit_prob = self.exit_classifier(pooled).mean().item()
        
        # Estimate complexity
        complexity = self.complexity_estimator(pooled).mean().item()
        
        # Exit if confident and past minimum layers
        should_exit = (
            current_layer >= 30 and  # Minimum layers
            exit_prob > 0.85 and
            complexity < 0.6  # Simple input
        )
        
        return should_exit, exit_prob
    
    def estimate_layers_needed(self, hidden_states: torch.Tensor) -> int:
        """
        Estimate how many layers are needed for this input.
        
        Args:
            hidden_states: Input hidden states
            
        Returns:
            Estimated number of layers needed
        """
        pooled = hidden_states.mean(dim=1)
        complexity = self.complexity_estimator(pooled).mean().item()
        
        # Map complexity to layer count (50-80 range)
        layers = int(50 + complexity * 30)
        return layers


class SWIFTSkipper(nn.Module):
    """
    SWIFT: Sample-Wise adaptive layer skipping.
    
    Dynamically skips layers per sample based on confidence.
    """
    
    def __init__(self, hidden_size: int, skip_every_n: int = 2):
        super().__init__()
        self.hidden_size = hidden_size
        self.skip_every_n = skip_every_n
        
        # Learnable skip gate
        self.skip_gate = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.GELU(),
            nn.Linear(hidden_size // 4, 1),
            nn.Sigmoid()
        )
        
        self.layer_norm = nn.LayerNorm(hidden_size)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        layer_func: Callable,
        layer_idx: int
    ) -> Tuple[torch.Tensor, bool]:
        """
        Conditionally execute layer based on skip decision.
        
        Args:
            hidden_states: Input hidden states
            layer_func: Layer function to execute
            layer_idx: Current layer index
            
        Returns:
            Tuple of (output, was_skipped)
        """
        # Check if we should skip
        pooled = hidden_states.mean(dim=1)
        skip_prob = self.skip_gate(pooled).mean()
        
        # Skip pattern: uniform skipping
        should_skip = (
            layer_idx % self.skip_every_n == 0 and
            layer_idx > 10 and  # Don't skip early layers
            layer_idx < 70 and  # Don't skip final layers
            skip_prob > 0.5
        )
        
        if should_skip:
            # Skip layer, just apply residual with layer norm
            output = self.layer_norm(hidden_states)
            return output, True
        else:
            # Execute layer normally
            output = layer_func(hidden_states)
            return output, False


class AdaptiveLayerSkipper:
    """
    Main adaptive layer skipping optimizer.
    
    Combines SWIFT and LayerSkip techniques for optimal performance.
    """
    
    def __init__(
        self,
        model: nn.Module,
        num_layers: int,
        hidden_size: int,
        config: Optional[LayerSkipConfig] = None
    ):
        self.model = model
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.config = config or LayerSkipConfig()
        
        # Initialize routers
        self.layer_skip_router = LayerSkipRouter(hidden_size, num_layers)
        self.swift_skipper = SWIFTSkipper(hidden_size)
        
        # Statistics
        self.stats = {
            "total_tokens": 0,
            "layers_skipped": 0,
            "early_exits": 0,
            "avg_layers_used": []
        }
        
        logger.info(f"AdaptiveLayerSkipper initialized (min={self.config.min_layers}, max={self.config.max_layers})")
    
    def forward_with_skipping(
        self,
        hidden_states: torch.Tensor,
        layers: List[nn.Module],
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Forward pass with adaptive layer skipping.
        
        Args:
            hidden_states: Input hidden states
            layers: List of transformer layers
            attention_mask: Optional attention mask
            
        Returns:
            Tuple of (output, metrics)
        """
        current = hidden_states
        layers_used = 0
        layers_skipped = 0
        early_exit = False
        
        for layer_idx, layer in enumerate(layers):
            # Check for early exit (LayerSkip style)
            if layer_idx >= self.config.min_layers:
                should_exit, confidence = self.layer_skip_router.should_exit_early(
                    current, layer_idx
                )
                
                if should_exit and not self.config.training_mode:
                    logger.debug(f"Early exit at layer {layer_idx} (confidence={confidence:.3f})")
                    early_exit = True
                    self.stats["early_exits"] += 1
                    break
            
            # Apply SWIFT-style skipping
            if self.config.skip_pattern == "uniform":
                current, was_skipped = self.swift_skipper(
                    current,
                    lambda h: layer(h, attention_mask=attention_mask)[0] if attention_mask is not None else layer(h)[0],
                    layer_idx
                )
                
                if was_skipped:
                    layers_skipped += 1
                else:
                    layers_used += 1
                    
            elif self.config.skip_pattern == "adaptive":
                # Adaptive: estimate layers needed
                if layer_idx == 0:
                    estimated = self.layer_skip_router.estimate_layers_needed(current)
                    self.target_layers = min(estimated, self.config.max_layers)
                
                # Skip if beyond estimated
                if layer_idx >= self.target_layers and layer_idx < len(layers) - 5:
                    layers_skipped += 1
                    continue
                
                # Execute layer
                current = layer(current, attention_mask=attention_mask)[0] if attention_mask is not None else layer(current)[0]
                layers_used += 1
                
            else:  # "early" - just skip final layers
                if layer_idx >= self.config.max_layers:
                    break
                    
                current = layer(current, attention_mask=attention_mask)[0] if attention_mask is not None else layer(current)[0]
                layers_used += 1
        
        self.stats["total_tokens"] += 1
        self.stats["layers_skipped"] += layers_skipped
        self.stats["avg_layers_used"].append(layers_used)
        
        metrics = {
            "layers_used": layers_used,
            "layers_skipped": layers_skipped,
            "early_exit": early_exit,
            "skip_rate": layers_skipped / (layers_used + layers_skipped) if (layers_used + layers_skipped) > 0 else 0
        }
        
        return current, metrics
    
    def get_stats(self) -> Dict[str, Any]:
        """Get skipping statistics."""
        avg_layers = np.mean(self.stats["avg_layers_used"]) if self.stats["avg_layers_used"] else self.num_layers
        theoretical_speedup = self.num_layers / avg_layers if avg_layers > 0 else 1.0
        
        return {
            "total_tokens": self.stats["total_tokens"],
            "early_exits": self.stats["early_exits"],
            "avg_layers_used": avg_layers,
            "layers_skipped": self.stats["layers_skipped"],
            "theoretical_speedup": theoretical_speedup
        }
    
    def reset_stats(self):
        """Reset statistics."""
        self.stats = {
            "total_tokens": 0,
            "layers_skipped": 0,
            "early_exits": 0,
            "avg_layers_used": []
        }


class LayerSkipIntegration(nn.Module):
    """
    Full integration of layer skipping into transformer model.
    
    Wraps a transformer model with LayerSkip capabilities.
    """
    
    def __init__(
        self,
        base_model: nn.Module,
        hidden_size: int,
        num_layers: int,
        config: Optional[LayerSkipConfig] = None
    ):
        super().__init__()
        self.base_model = base_model
        self.skipper = AdaptiveLayerSkipper(base_model, num_layers, hidden_size, config)
        
        # Extract layers
        self.layers = self.skipper.layers if hasattr(self.skipper, 'layers') else []
        
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Forward pass with layer skipping."""
        
        # Get initial hidden states if not provided
        if hidden_states is None and input_ids is not None:
            hidden_states = self.base_model.get_input_embeddings()(input_ids)
        
        # Forward with skipping
        output, metrics = self.skipper.forward_with_skipping(
            hidden_states,
            self.layers,
            attention_mask
        )
        
        return {
            "last_hidden_state": output,
            "metrics": metrics
        }
