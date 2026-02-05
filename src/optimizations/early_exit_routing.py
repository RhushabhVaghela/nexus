"""
Early Exit + Dynamic Routing (LayerSkip, DASH)

Key insight: Simple inputs exit early.
Router decides per-token which layers to skip.
Average: 60 layers instead of 80 = 1.33× faster

Research references:
- LayerSkip: https://arxiv.org/abs/2404.16710
- DASH: Dynamic Activation Shaping
- Early Exit: https://arxiv.org/abs/2004.12998
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
class DynamicRoutingConfig:
    """Configuration for dynamic routing."""
    min_layers: int = 30
    max_layers: int = 80
    confidence_threshold: float = 0.85
    entropy_threshold: float = 0.5
    use_token_routing: bool = True
    use_layer_routing: bool = True
    training_mode: bool = False


class TokenRouter(nn.Module):
    """
    Per-token router that decides exit point.
    
    Different tokens may need different computation depths.
    """
    
    def __init__(self, hidden_size: int, num_layers: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Token complexity estimator
        self.complexity_estimator = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
        
        # Exit confidence predictor
        self.exit_predictor = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 4),
                nn.GELU(),
                nn.Linear(hidden_size // 4, 1),
                nn.Sigmoid()
            )
            for _ in range(num_layers)
        ])
        
    def estimate_exit_layer(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Estimate optimal exit layer for each token.
        
        Args:
            hidden_states: [batch, seq_len, hidden_size]
            
        Returns:
            Exit layer indices [batch, seq_len]
        """
        # Complexity per token
        complexity = self.complexity_estimator(hidden_states).squeeze(-1)  # [batch, seq_len]
        
        # Map complexity to layer: low complexity -> early exit
        # Complexity 0.0-0.3: exit at 40% of layers
        # Complexity 0.3-0.7: exit at 70% of layers
        # Complexity 0.7-1.0: exit at 100% of layers
        
        exit_layers = torch.zeros_like(complexity, dtype=torch.long)
        
        exit_layers = torch.where(
            complexity < 0.3,
            torch.tensor(int(self.num_layers * 0.4), device=complexity.device),
            exit_layers
        )
        exit_layers = torch.where(
            (complexity >= 0.3) & (complexity < 0.7),
            torch.tensor(int(self.num_layers * 0.7), device=complexity.device),
            exit_layers
        )
        exit_layers = torch.where(
            complexity >= 0.7,
            torch.tensor(self.num_layers, device=complexity.device),
            exit_layers
        )
        
        return exit_layers
    
    def should_exit_at_layer(
        self,
        hidden_states: torch.Tensor,
        layer_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Determine which tokens should exit at current layer.
        
        Args:
            hidden_states: [batch, seq_len, hidden_size]
            layer_idx: Current layer index
            
        Returns:
            Tuple of (exit_mask [batch, seq_len], confidence [batch, seq_len])
        """
        if layer_idx >= len(self.exit_predictor):
            # All tokens must exit at final layer
            batch_size, seq_len = hidden_states.shape[:2]
            return torch.ones(batch_size, seq_len, device=hidden_states.device, dtype=torch.bool), \
                   torch.ones(batch_size, seq_len, device=hidden_states.device)
        
        # Get exit confidence for this layer
        confidence = self.exit_predictor[layer_idx](hidden_states).squeeze(-1)
        
        # Exit if confident enough
        exit_mask = confidence > 0.85
        
        return exit_mask, confidence


class DynamicLayerRouter(nn.Module):
    """
    Router that dynamically selects which layers to execute.
    
    Based on DASH: Skip less important layers for each input.
    """
    
    def __init__(self, hidden_size: int, num_layers: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Layer importance predictor
        self.layer_importance = nn.Sequential(
            nn.Linear(hidden_size, num_layers),
            nn.Sigmoid()
        )
        
        # Learnable layer gates
        self.layer_gates = nn.Parameter(torch.ones(num_layers) * 0.5)
        
    def compute_layer_mask(
        self,
        hidden_states: torch.Tensor,
        min_layers: int = 30
    ) -> torch.Tensor:
        """
        Compute which layers to execute.
        
        Args:
            hidden_states: [batch, seq_len, hidden_size]
            min_layers: Minimum layers to always execute
            
        Returns:
            Layer mask [num_layers] (1=execute, 0=skip)
        """
        # Get pooled representation
        pooled = hidden_states.mean(dim=1)  # [batch, hidden]
        
        # Predict importance per layer
        importance = self.layer_importance(pooled).mean(dim=0)  # [num_layers]
        
        # Combine with learnable gates
        combined_score = importance * torch.sigmoid(self.layer_gates)
        
        # Always keep minimum layers
        layer_mask = torch.zeros(self.num_layers, device=hidden_states.device)
        layer_mask[:min_layers] = 1.0
        
        # Select additional layers based on score
        remaining_budget = self.num_layers - min_layers
        if remaining_budget > 0:
            _, top_indices = torch.topk(combined_score[min_layers:], remaining_budget // 2)
            layer_mask[min_layers + top_indices] = 1.0
        
        return layer_mask


class EarlyExitRouter:
    """
    Main early exit and dynamic routing optimizer.
    
    Combines LayerSkip and DASH techniques.
    """
    
    def __init__(
        self,
        model: nn.Module,
        num_layers: int,
        hidden_size: int,
        config: Optional[DynamicRoutingConfig] = None
    ):
        self.model = model
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.config = config or DynamicRoutingConfig()
        
        # Initialize routers
        self.token_router = TokenRouter(hidden_size, num_layers)
        self.layer_router = DynamicLayerRouter(hidden_size, num_layers)
        
        # Statistics
        self.stats = {
            "total_tokens": 0,
            "early_exits": 0,
            "layers_skipped": 0,
            "avg_exit_layer": []
        }
        
        # Cache for exit decisions
        self.exit_cache: Dict[int, torch.Tensor] = {}
        
        logger.info(f"EarlyExitRouter initialized (min={self.config.min_layers})")
    
    def forward_with_routing(
        self,
        hidden_states: torch.Tensor,
        layers: List[nn.Module],
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Forward pass with early exit and dynamic routing.
        
        Args:
            hidden_states: Input hidden states
            layers: List of transformer layers
            attention_mask: Optional attention mask
            
        Returns:
            Tuple of (output, metrics)
        """
        batch_size, seq_len, _ = hidden_states.shape
        device = hidden_states.device
        
        # Compute layer mask (DASH-style)
        layer_mask = self.layer_router.compute_layer_mask(
            hidden_states,
            min_layers=self.config.min_layers
        )
        
        # Track which tokens have exited
        active_mask = torch.ones(batch_size, seq_len, device=device, dtype=torch.bool)
        exit_layers = torch.full((batch_size, seq_len), self.num_layers, device=device, dtype=torch.long)
        
        current = hidden_states
        layers_executed = 0
        
        for layer_idx, layer in enumerate(layers):
            # Skip if layer mask says so
            if layer_mask[layer_idx] < 0.5:
                continue
            
            # Only process active tokens
            if not active_mask.any():
                break
            
            # Execute layer for active tokens
            current_active = current.clone()
            layer_output = layer(current_active, attention_mask=attention_mask)[0] if attention_mask is not None else layer(current_active)[0]
            current = torch.where(active_mask.unsqueeze(-1), layer_output, current)
            
            layers_executed += 1
            
            # Check for early exit (LayerSkip-style)
            if layer_idx >= self.config.min_layers and self.config.use_token_routing:
                should_exit, confidence = self.token_router.should_exit_at_layer(current, layer_idx)
                
                # Mark tokens that should exit
                new_exits = should_exit & active_mask
                exit_layers = torch.where(new_exits, layer_idx, exit_layers)
                active_mask = active_mask & ~should_exit
                
                if new_exits.any():
                    self.stats["early_exits"] += new_exits.sum().item()
        
        # Update statistics
        avg_exit = exit_layers.float().mean().item()
        self.stats["total_tokens"] += batch_size * seq_len
        self.stats["avg_exit_layer"].append(avg_exit)
        self.stats["layers_skipped"] += (self.num_layers - layers_executed) * batch_size * seq_len
        
        metrics = {
            "layers_executed": layers_executed,
            "avg_exit_layer": avg_exit,
            "early_exit_rate": (exit_layers < self.num_layers).float().mean().item(),
            "theoretical_speedup": self.num_layers / avg_exit if avg_exit > 0 else 1.0
        }
        
        return current, metrics
    
    def get_stats(self) -> Dict[str, Any]:
        """Get routing statistics."""
        avg_exit = np.mean(self.stats["avg_exit_layer"]) if self.stats["avg_exit_layer"] else self.num_layers
        
        return {
            "total_tokens": self.stats["total_tokens"],
            "early_exits": self.stats["early_exits"],
            "avg_exit_layer": avg_exit,
            "theoretical_speedup": self.num_layers / avg_exit if avg_exit > 0 else 1.0
        }
    
    def reset_stats(self):
        """Reset statistics."""
        self.stats = {
            "total_tokens": 0,
            "early_exits": 0,
            "layers_skipped": 0,
            "avg_exit_layer": []
        }


class AdaptiveExitLayer(nn.Module):
    """
    Wrapper that adds early exit capability to any layer.
    """
    
    def __init__(self, base_layer: nn.Module, exit_predictor: nn.Module, layer_idx: int):
        super().__init__()
        self.base_layer = base_layer
        self.exit_predictor = exit_predictor
        self.layer_idx = layer_idx
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward with exit prediction.
        
        Returns:
            Tuple of (output, should_exit)
        """
        output = self.base_layer(hidden_states, attention_mask=attention_mask)[0] if attention_mask is not None else self.base_layer(hidden_states)[0]
        
        # Predict exit confidence
        confidence = self.exit_predictor(output).squeeze(-1)
        should_exit = confidence > 0.85
        
        return output, should_exit
