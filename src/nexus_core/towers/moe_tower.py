"""
src/nexus_core/towers/moe_tower.py

Mixture of Experts (MoE) Tower with sparse expert routing.
Implements efficient sparse activation for large-scale models.

Features:
- Top-k expert routing with load balancing
- Sparse expert activation (only k experts active per token)
- Expert capacity management
- Load balancing auxiliary loss
- Expert dropout for regularization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
import logging
import math

from .base_tower import BaseTower, TowerConfig, TowerMode

logger = logging.getLogger(__name__)


@dataclass
class MoETowerConfig(TowerConfig):
    """Configuration for MoE Tower."""
    # MoE settings
    num_experts: int = 8
    num_experts_per_token: int = 2  # Top-k
    expert_hidden_size: int = 4096
    
    # Load balancing
    load_balance_loss_coef: float = 0.01
    router_z_loss_coef: float = 0.001
    
    # Capacity management
    capacity_factor: float = 1.25  # Buffer capacity for uneven routing
    
    # Expert dropout
    expert_dropout_rate: float = 0.0
    
    # Switch Transformer style
    use_switch_style: bool = False  # If True, use top-1 instead of top-k
    
    # Expert parallelism
    expert_parallel_size: int = 1
    
    # Jitter
    router_jitter_noise: float = 0.0  # Add noise for exploration


class Expert(nn.Module):
    """
    Individual expert network (FFN).
    """
    
    def __init__(self, hidden_size: int, intermediate_size: int, 
                 dropout_rate: float = 0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        
        # FFN layers
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(intermediate_size, hidden_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward through expert."""
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class Router(nn.Module):
    """
    Sparse router for expert selection.
    
    Routes tokens to top-k experts based on learned routing weights.
    """
    
    def __init__(self, hidden_size: int, num_experts: int, 
                 top_k: int = 2, jitter_noise: float = 0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.jitter_noise = jitter_noise
        
        # Routing weights
        self.weight = nn.Linear(hidden_size, num_experts, bias=False)
        
        # Statistics
        self.reset_stats()
    
    def reset_stats(self):
        """Reset routing statistics."""
        self._stats = {
            "total_tokens": 0,
            "expert_selections": [0] * self.num_experts,
            "average_topk_prob": 0.0,
        }
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Route tokens to experts.
        
        Args:
            x: Input tensor (batch_size * seq_len, hidden_size)
            
        Returns:
            expert_indices: (batch_size * seq_len, top_k) - selected experts
            expert_weights: (batch_size * seq_len, top_k) - routing weights
            aux_loss: Dictionary with auxiliary losses
        """
        batch_size = x.shape[0]
        
        # Compute router logits
        router_logits = self.weight(x)  # (batch_size, num_experts)
        
        # Add jitter noise for exploration during training
        if self.training and self.jitter_noise > 0:
            noise = torch.randn_like(router_logits) * self.jitter_noise
            router_logits = router_logits + noise
        
        # Compute routing probabilities
        router_probs = F.softmax(router_logits, dim=-1)
        
        # Select top-k experts
        if self.top_k == 1:
            expert_weights, expert_indices = torch.max(router_probs, dim=-1, keepdim=True)
        else:
            expert_weights, expert_indices = torch.topk(router_probs, self.top_k, dim=-1)
            # Normalize weights
            expert_weights = expert_weights / expert_weights.sum(dim=-1, keepdim=True)
        
        # Compute auxiliary losses
        aux_losses = self._compute_aux_losses(router_logits, router_probs, expert_indices)
        
        # Update statistics
        self._update_stats(expert_indices, expert_weights)
        
        return expert_indices, expert_weights, aux_losses
    
    def _compute_aux_losses(self, router_logits: torch.Tensor,
                           router_probs: torch.Tensor,
                           expert_indices: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute auxiliary losses for load balancing."""
        num_tokens = router_logits.shape[0]
        
        # Load balancing loss (encourage uniform expert usage)
        # f_i = fraction of tokens routed to expert i
        router_prob_per_expert = torch.mean(router_probs, dim=0)
        
        # Count tokens per expert
        expert_mask = F.one_hot(expert_indices, self.num_experts).sum(dim=1)  # (batch, num_experts)
        tokens_per_expert = torch.mean(expert_mask.float(), dim=0)
        
        # Load balancing loss: want all experts to get equal load
        load_balance_loss = torch.sum(tokens_per_expert * router_prob_per_expert) * self.num_experts
        
        # Router z-loss (encourage router logits to stay small for stability)
        log_z = torch.logsumexp(router_logits, dim=-1)
        z_loss = torch.mean(log_z ** 2)
        
        return {
            "load_balance_loss": load_balance_loss,
            "router_z_loss": z_loss
        }
    
    def _update_stats(self, expert_indices: torch.Tensor, 
                     expert_weights: torch.Tensor):
        """Update routing statistics."""
        self._stats["total_tokens"] += expert_indices.shape[0]
        self._stats["average_topk_prob"] = expert_weights.mean().item()
        
        for idx in expert_indices.flatten():
            self._stats["expert_selections"][idx.item()] += 1


class MoETower(BaseTower):
    """
    Mixture of Experts Tower with sparse expert routing.
    
    This tower implements:
    1. Top-k sparse routing to expert networks
    2. Load balancing across experts
    3. Efficient batched expert computation
    4. Expert capacity management
    
    Inspired by "Outrageously Large Neural Networks: The Sparsely-Gated 
    Mixture-of-Experts Layer" and "Switch Transformers".
    """
    
    def __init__(self, config: MoETowerConfig):
        """
        Initialize MoE Tower.
        
        Args:
            config: MoETowerConfig instance
        """
        super().__init__(config)
        self.config: MoETowerConfig = config
        
        # Create router
        self.router = Router(
            hidden_size=config.hidden_size,
            num_experts=config.num_experts,
            top_k=1 if config.use_switch_style else config.num_experts_per_token,
            jitter_noise=config.router_jitter_noise
        )
        
        # Create experts
        self.experts = nn.ModuleList([
            Expert(
                hidden_size=config.hidden_size,
                intermediate_size=config.expert_hidden_size,
                dropout_rate=config.expert_dropout_rate
            )
            for _ in range(config.num_experts)
        ])
        
        # Expert capacity (max tokens per expert)
        self._expert_capacity = None
        
        # Statistics
        self._forward_count = 0
        
        logger.info(
            f"MoETower initialized: {config.num_experts} experts, "
            f"top-{config.num_experts_per_token} routing"
        )
    
    def _compute_expert_capacity(self, num_tokens: int) -> int:
        """Compute capacity per expert with buffer factor."""
        tokens_per_expert = num_tokens * self.config.num_experts_per_token / self.config.num_experts
        capacity = int(tokens_per_expert * self.config.capacity_factor)
        return max(capacity, 1)  # At least 1
    
    def forward(self, x: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                **kwargs) -> Dict[str, torch.Tensor]:
        """
        Forward pass with sparse expert routing.
        
        Args:
            x: Input tensor (batch_size, seq_len, hidden_size)
            attention_mask: Optional attention mask
            **kwargs: Additional arguments
            
        Returns:
            Dictionary with output and auxiliary losses
        """
        self._stats["forward_passes"] += 1
        self._forward_count += 1
        
        batch_size, seq_len, hidden_size = x.shape
        num_tokens = batch_size * seq_len
        
        # Reshape for routing
        x_flat = x.view(-1, hidden_size)  # (batch_size * seq_len, hidden_size)
        
        # Route tokens to experts
        expert_indices, expert_weights, aux_losses = self.router(x_flat)
        
        # Compute expert capacity
        expert_capacity = self._compute_expert_capacity(num_tokens)
        
        # Process through experts
        if self.config.use_switch_style:
            # Switch Transformer style (top-1)
            output = self._switch_forward(
                x_flat, expert_indices.squeeze(-1), expert_weights.squeeze(-1)
            )
        else:
            # Standard MoE (top-k)
            output = self._moe_forward(
                x_flat, expert_indices, expert_weights, expert_capacity
            )
        
        # Reshape back
        output = output.view(batch_size, seq_len, hidden_size)
        
        # Apply layer norm (common practice in MoE)
        if hasattr(self, 'layer_norm'):
            output = self.layer_norm(output)
        
        # Scale auxiliary losses
        aux_losses["load_balance_loss"] *= self.config.load_balance_loss_coef
        aux_losses["router_z_loss"] *= self.config.router_z_loss_coef
        aux_losses["total_aux_loss"] = (
            aux_losses["load_balance_loss"] + aux_losses["router_z_loss"]
        )
        
        # Track layer activations
        for i in range(self.config.num_layers):
            if i in expert_indices:
                self.track_layer_activation(i, output.view(-1, hidden_size))
        
        return {
            "hidden_states": output,
            "aux_losses": aux_losses,
            "expert_indices": expert_indices,
            "expert_weights": expert_weights
        }
    
    def _switch_forward(self, x: torch.Tensor, 
                       expert_indices: torch.Tensor,
                       expert_weights: torch.Tensor) -> torch.Tensor:
        """
        Switch Transformer style forward (top-1 routing).
        
        Args:
            x: Input tensor (num_tokens, hidden_size)
            expert_indices: Selected expert for each token (num_tokens,)
            expert_weights: Routing weights (num_tokens,)
            
        Returns:
            Output tensor (num_tokens, hidden_size)
        """
        num_tokens = x.shape[0]
        output = torch.zeros_like(x)
        
        # Group tokens by expert
        for expert_idx in range(self.config.num_experts):
            mask = expert_indices == expert_idx
            if not mask.any():
                continue
            
            expert_input = x[mask]
            expert_output = self.experts[expert_idx](expert_input)
            
            # Apply routing weight
            weights = expert_weights[mask].unsqueeze(-1)
            output[mask] = weights * expert_output
        
        return output
    
    def _moe_forward(self, x: torch.Tensor,
                    expert_indices: torch.Tensor,
                    expert_weights: torch.Tensor,
                    expert_capacity: int) -> torch.Tensor:
        """
        Standard MoE forward (top-k routing).
        
        Args:
            x: Input tensor (num_tokens, hidden_size)
            expert_indices: Selected experts (num_tokens, top_k)
            expert_weights: Routing weights (num_tokens, top_k)
            expert_capacity: Max tokens per expert
            
        Returns:
            Output tensor (num_tokens, hidden_size)
        """
        num_tokens, hidden_size = x.shape
        top_k = expert_indices.shape[1]
        output = torch.zeros_like(x)
        
        # Process each expert
        for expert_idx in range(self.config.num_experts):
            # Find tokens routed to this expert
            token_indices = []
            weight_values = []
            k_positions = []
            
            for k in range(top_k):
                mask = expert_indices[:, k] == expert_idx
                if mask.any():
                    token_indices.extend(torch.where(mask)[0].tolist())
                    weight_values.extend(expert_weights[mask, k].tolist())
                    k_positions.extend([k] * mask.sum().item())
            
            if not token_indices:
                continue
            
            # Limit to capacity (drop excess tokens if over capacity)
            if len(token_indices) > expert_capacity:
                # Sort by weight and keep top ones
                sorted_indices = sorted(
                    range(len(token_indices)),
                    key=lambda i: weight_values[i],
                    reverse=True
                )[:expert_capacity]
                token_indices = [token_indices[i] for i in sorted_indices]
                weight_values = [weight_values[i] for i in sorted_indices]
            
            # Gather inputs for this expert
            expert_input = x[token_indices]
            
            # Forward through expert
            expert_output = self.experts[expert_idx](expert_input)
            
            # Scatter outputs back with weights
            for i, (token_idx, weight) in enumerate(zip(token_indices, weight_values)):
                output[token_idx] += weight * expert_output[i]
        
        return output
    
    def get_expert_stats(self) -> Dict[str, Any]:
        """Get expert usage statistics."""
        router_stats = self.router._stats
        total_selections = sum(router_stats["expert_selections"])
        
        if total_selections == 0:
            return {"error": "No routing has occurred"}
        
        expert_utilization = [
            count / total_selections 
            for count in router_stats["expert_selections"]
        ]
        
        return {
            "total_tokens_routed": router_stats["total_tokens"],
            "average_topk_probability": router_stats["average_topk_prob"],
            "expert_selections": router_stats["expert_selections"],
            "expert_utilization": expert_utilization,
            "utilization_std": torch.tensor(expert_utilization).std().item(),
            "most_used_expert": max(range(self.config.num_experts),
                                   key=lambda i: router_stats["expert_selections"][i]),
            "least_used_expert": min(range(self.config.num_experts),
                                    key=lambda i: router_stats["expert_selections"][i]),
        }
    
    def reset_expert_stats(self):
        """Reset expert statistics."""
        self.router.reset_stats()
        self._forward_count = 0
    
    def load_expert_weights(self, expert_idx: int, 
                           weights_path: str) -> None:
        """
        Load weights for a specific expert.
        
        Args:
            expert_idx: Index of expert to load
            weights_path: Path to weights file
        """
        if expert_idx >= self.config.num_experts:
            raise ValueError(f"Expert index {expert_idx} out of range")
        
        try:
            state_dict = torch.load(weights_path, map_location=self.device)
            self.experts[expert_idx].load_state_dict(state_dict)
            logger.info(f"Loaded weights for expert {expert_idx}")
        except Exception as e:
            logger.error(f"Failed to load expert weights: {e}")
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        """Get MoE tower statistics."""
        base_stats = super().get_stats()
        moe_stats = {
            "num_experts": self.config.num_experts,
            "top_k": self.config.num_experts_per_token,
            "expert_stats": self.get_expert_stats(),
            "total_forwards": self._forward_count
        }
        return {**base_stats, **moe_stats}


__all__ = [
    'MoETower',
    'MoETowerConfig',
    'Expert',
    'Router'
]