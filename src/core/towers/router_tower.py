"""
src/nexus_core/towers/router_tower.py

Router Tower for multi-model routing and ensemble.
Routes inputs to appropriate specialist towers based on content.

Features:
- Content-based routing to specialist towers
- Ensemble aggregation from multiple towers
- Confidence-weighted predictions
- Load balancing across towers
- Fallback mechanisms
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, List, Tuple, Callable, Union
from dataclasses import dataclass
from enum import Enum
import logging

from .base_tower import BaseTower, TowerConfig

logger = logging.getLogger(__name__)


class RoutingStrategy(Enum):
    """Routing strategies for multi-tower selection."""
    SINGLE = "single"           # Route to single best tower
    TOP_K = "top_k"             # Route to top-k towers
    WEIGHTED = "weighted"       # Weighted combination of all towers
    CONFIDENCE = "confidence"   # Confidence-based routing
    CASCADE = "cascade"         # Cascade through towers


@dataclass
class RouterTowerConfig(TowerConfig):
    """Configuration for Router Tower."""
    # Routing settings
    strategy: RoutingStrategy = RoutingStrategy.SINGLE
    top_k: int = 2
    confidence_threshold: float = 0.7
    
    # Ensemble settings
    ensemble_method: str = "mean"  # mean, weighted_mean, max
    
    # Load balancing
    balance_load: bool = True
    tower_capacity: Optional[Dict[str, int]] = None
    
    # Training
    train_router: bool = True
    router_lr: float = 1e-4
    
    # Fallback
    fallback_tower: Optional[str] = None
    fallback_on_error: bool = True
    
    # Latency optimization
    parallel_inference: bool = False
    async_routing: bool = False


class TowerRouter(nn.Module):
    """
    Router network for selecting appropriate towers.
    
    Learns to route inputs to the most appropriate tower(s)
    based on input content.
    """
    
    def __init__(self, hidden_size: int, num_towers: int, 
                 strategy: RoutingStrategy = RoutingStrategy.SINGLE):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_towers = num_towers
        self.strategy = strategy
        
        # Router network
        self.router = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, num_towers)
        )
        
        # Statistics
        self.reset_stats()
    
    def reset_stats(self):
        """Reset routing statistics."""
        self._stats = {
            "total_routes": 0,
            "tower_selections": [0] * self.num_towers,
            "confidence_scores": [],
        }
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Route input to towers.
        
        Args:
            x: Input features (batch_size, hidden_size)
            
        Returns:
            tower_weights: (batch_size, num_towers) - routing weights
            tower_indices: (batch_size, top_k) - selected tower indices
            confidence: (batch_size,) - routing confidence
        """
        # Get routing logits
        logits = self.router(x)  # (batch_size, num_towers)
        
        # Compute routing probabilities
        weights = F.softmax(logits, dim=-1)
        
        # Compute confidence (max probability)
        confidence, selected = torch.max(weights, dim=-1)
        
        # Select top-k towers
        if self.strategy == RoutingStrategy.SINGLE:
            top_k = 1
        elif self.strategy == RoutingStrategy.TOP_K:
            top_k = min(self.strategy.value, self.num_towers)
        else:
            top_k = self.num_towers
        
        top_weights, top_indices = torch.topk(weights, top_k, dim=-1)
        
        # Update statistics
        self._update_stats(selected, confidence)
        
        return weights, top_indices, confidence
    
    def _update_stats(self, selections: torch.Tensor, 
                     confidence: torch.Tensor):
        """Update routing statistics."""
        self._stats["total_routes"] += selections.shape[0]
        self._stats["confidence_scores"].extend(confidence.tolist())
        
        for idx in selections:
            self._stats["tower_selections"][idx.item()] += 1


class RouterTower(BaseTower):
    """
    Router Tower for multi-model routing and ensemble.
    
    This tower manages multiple specialist towers and routes inputs
    to the most appropriate one(s) based on content.
    
    Use cases:
    - Multi-task learning with task-specific towers
    - Ensemble methods for improved accuracy
    - Load balancing across compute resources
    - Domain-specific routing (e.g., code vs. natural language)
    """
    
    def __init__(self, config: RouterTowerConfig):
        """
        Initialize Router Tower.
        
        Args:
            config: RouterTowerConfig instance
        """
        super().__init__(config)
        self.config: RouterTowerConfig = config
        
        # Registered towers
        self.towers: Dict[str, BaseTower] = {}
        self._tower_order: List[str] = []  # For consistent indexing
        
        # Router network
        self.router: Optional[TowerRouter] = None
        
        # Tower weights for weighted ensemble
        self._tower_weights: Optional[torch.Tensor] = None
        
        # Statistics
        self._inference_stats = {
            "tower_usage": {},
            "latency_ms": [],
            "errors": []
        }
        
        logger.info("RouterTower initialized")
    
    def register_tower(self, name: str, tower: BaseTower,
                       weight: float = 1.0) -> None:
        """
        Register a specialist tower.
        
        Args:
            name: Unique name for the tower
            tower: Tower instance
            weight: Weight for ensemble methods
        """
        if name in self.towers:
            logger.warning(f"Tower '{name}' already registered, overwriting")
        
        self.towers[name] = tower
        if name not in self._tower_order:
            self._tower_order.append(name)
        
        # Update router if needed
        if self.router is not None and len(self.towers) != self.router.num_towers:
            logger.info(f"Reinitializing router for {len(self.towers)} towers")
            self._init_router()
        
        logger.info(f"Registered tower: {name}")
    
    def unregister_tower(self, name: str) -> bool:
        """Unregister a tower."""
        if name not in self.towers:
            return False
        
        del self.towers[name]
        self._tower_order.remove(name)
        
        # Reinitialize router
        if self.towers:
            self._init_router()
        
        logger.info(f"Unregistered tower: {name}")
        return True
    
    def _init_router(self):
        """Initialize the router network."""
        if not self.towers:
            return
        
        self.router = TowerRouter(
            hidden_size=self.config.hidden_size,
            num_towers=len(self.towers),
            strategy=self.config.strategy
        )
        
        # Initialize tower weights
        self._tower_weights = torch.ones(len(self.towers))
    
    def set_tower_weights(self, weights: Dict[str, float]) -> None:
        """
        Set weights for ensemble methods.
        
        Args:
            weights: Dictionary mapping tower names to weights
        """
        if self._tower_weights is None:
            self._tower_weights = torch.ones(len(self.towers))
        
        for name, weight in weights.items():
            if name in self._tower_order:
                idx = self._tower_order.index(name)
                self._tower_weights[idx] = weight
        
        # Normalize
        self._tower_weights = self._tower_weights / self._tower_weights.sum()
    
    def forward(self, x: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                **kwargs) -> Dict[str, torch.Tensor]:
        """
        Forward pass with routing to specialist towers.
        
        Args:
            x: Input tensor
            attention_mask: Optional attention mask
            **kwargs: Additional arguments
            
        Returns:
            Dictionary with outputs and routing information
        """
        self._stats["forward_passes"] += 1
        
        if not self.towers:
            raise RuntimeError("No towers registered with RouterTower")
        
        # Initialize router if needed
        if self.router is None:
            self._init_router()
        
        # Get routing decisions
        batch_size = x.shape[0]
        
        # Compute routing based on input features
        # Use mean pooled features for routing
        router_input = x.mean(dim=1) if x.dim() > 2 else x
        tower_weights, tower_indices, confidence = self.router(router_input)
        
        # Route based on strategy
        if self.config.strategy == RoutingStrategy.SINGLE:
            output = self._single_route(x, tower_indices, confidence, 
                                       attention_mask, **kwargs)
        
        elif self.config.strategy == RoutingStrategy.TOP_K:
            output = self._topk_route(x, tower_indices, tower_weights,
                                     attention_mask, **kwargs)
        
        elif self.config.strategy == RoutingStrategy.WEIGHTED:
            output = self._weighted_route(x, tower_weights, attention_mask, **kwargs)
        
        elif self.config.strategy == RoutingStrategy.CONFIDENCE:
            output = self._confidence_route(x, tower_weights, confidence,
                                           attention_mask, **kwargs)
        
        elif self.config.strategy == RoutingStrategy.CASCADE:
            output = self._cascade_route(x, attention_mask, **kwargs)
        
        else:
            raise ValueError(f"Unknown routing strategy: {self.config.strategy}")
        
        # Add routing metadata
        output["routing"] = {
            "tower_weights": tower_weights,
            "tower_indices": tower_indices,
            "confidence": confidence,
            "strategy": self.config.strategy.value
        }
        
        return output
    
    def _single_route(self, x: torch.Tensor, tower_indices: torch.Tensor,
                     confidence: torch.Tensor,
                     attention_mask: Optional[torch.Tensor],
                     **kwargs) -> Dict[str, torch.Tensor]:
        """Route to single best tower."""
        # Use the most selected tower
        selected_idx = tower_indices[:, 0].mode()[0].item()
        tower_name = self._tower_order[selected_idx]
        tower = self.towers[tower_name]
        
        try:
            output = tower(x, attention_mask=attention_mask, **kwargs)
            output["selected_tower"] = tower_name
            output["routing_confidence"] = confidence.mean().item()
            
            # Update stats
            self._inference_stats["tower_usage"][tower_name] = \
                self._inference_stats["tower_usage"].get(tower_name, 0) + 1
            
        except Exception as e:
            logger.error(f"Tower '{tower_name}' failed: {e}")
            if self.config.fallback_on_error and self.config.fallback_tower:
                fallback = self.towers.get(self.config.fallback_tower)
                if fallback:
                    output = fallback(x, attention_mask=attention_mask, **kwargs)
                    output["selected_tower"] = self.config.fallback_tower
                    output["fallback_used"] = True
                else:
                    raise
            else:
                raise
        
        return output
    
    def _topk_route(self, x: torch.Tensor, tower_indices: torch.Tensor,
                   tower_weights: torch.Tensor,
                   attention_mask: Optional[torch.Tensor],
                   **kwargs) -> Dict[str, torch.Tensor]:
        """Route to top-k towers and ensemble."""
        outputs = []
        weights = []
        
        for i in range(min(self.config.top_k, tower_indices.shape[1])):
            idx = tower_indices[:, i].mode()[0].item()
            tower_name = self._tower_order[idx]
            tower = self.towers[tower_name]
            
            try:
                output = tower(x, attention_mask=attention_mask, **kwargs)
                outputs.append(output.get("hidden_states", output))
                weights.append(tower_weights[:, idx].mean().item())
            except Exception as e:
                logger.error(f"Tower '{tower_name}' failed: {e}")
                continue
        
        if not outputs:
            raise RuntimeError("All towers failed")
        
        # Ensemble outputs
        weights = torch.tensor(weights, device=x.device)
        weights = weights / weights.sum()
        
        ensemble_output = self._ensemble_outputs(outputs, weights)
        
        return {
            "hidden_states": ensemble_output,
            "selected_towers": [self._tower_order[tower_indices[0, i].item()] 
                               for i in range(tower_indices.shape[1])],
            "ensemble_weights": weights.tolist()
        }
    
    def _weighted_route(self, x: torch.Tensor, tower_weights: torch.Tensor,
                       attention_mask: Optional[torch.Tensor],
                       **kwargs) -> Dict[str, torch.Tensor]:
        """Weighted combination of all towers."""
        outputs = []
        
        for name, tower in self.towers.items():
            try:
                output = tower(x, attention_mask=attention_mask, **kwargs)
                outputs.append(output.get("hidden_states", output))
            except Exception as e:
                logger.error(f"Tower '{name}' failed: {e}")
                continue
        
        if not outputs:
            raise RuntimeError("All towers failed")
        
        # Use tower weights or learned weights
        weights = self._tower_weights.to(x.device) if self._tower_weights is not None else tower_weights[0]
        weights = weights / weights.sum()
        
        ensemble_output = self._ensemble_outputs(outputs, weights)
        
        return {
            "hidden_states": ensemble_output,
            "tower_weights": weights.tolist()
        }
    
    def _confidence_route(self, x: torch.Tensor, tower_weights: torch.Tensor,
                         confidence: torch.Tensor,
                         attention_mask: Optional[torch.Tensor],
                         **kwargs) -> Dict[str, torch.Tensor]:
        """Confidence-based routing."""
        if confidence.mean() > self.config.confidence_threshold:
            # High confidence, use single tower
            return self._single_route(x, tower_weights.argmax(dim=1, keepdim=True),
                                     confidence, attention_mask, **kwargs)
        else:
            # Low confidence, use ensemble
            return self._weighted_route(x, tower_weights, attention_mask, **kwargs)
    
    def _cascade_route(self, x: torch.Tensor,
                      attention_mask: Optional[torch.Tensor],
                      **kwargs) -> Dict[str, torch.Tensor]:
        """Cascade through towers until confident."""
        # Order by speed (assume registration order)
        for name, tower in self.towers.items():
            try:
                output = tower(x, attention_mask=attention_mask, **kwargs)
                # Could add confidence check here to continue cascading
                output["selected_tower"] = name
                return output
            except Exception as e:
                logger.warning(f"Tower '{name}' failed in cascade: {e}")
                continue
        
        raise RuntimeError("All towers failed in cascade")
    
    def _ensemble_outputs(self, outputs: List[torch.Tensor],
                         weights: torch.Tensor) -> torch.Tensor:
        """Ensemble multiple outputs."""
        if self.config.ensemble_method == "mean":
            return torch.stack(outputs).mean(dim=0)
        
        elif self.config.ensemble_method == "weighted_mean":
            stacked = torch.stack(outputs)  # (num_towers, batch, seq, hidden)
            weights = weights.view(-1, 1, 1, 1)
            return (stacked * weights).sum(dim=0)
        
        elif self.config.ensemble_method == "max":
            return torch.stack(outputs).max(dim=0)[0]
        
        else:
            raise ValueError(f"Unknown ensemble method: {self.config.ensemble_method}")
    
    def list_towers(self) -> List[str]:
        """List registered tower names."""
        return list(self.towers.keys())
    
    def get_tower_stats(self) -> Dict[str, Any]:
        """Get statistics for all towers."""
        tower_stats = {}
        for name, tower in self.towers.items():
            tower_stats[name] = tower.get_stats()
        
        routing_stats = {}
        if self.router is not None:
            routing_stats = {
                "total_routes": self.router._stats["total_routes"],
                "tower_selections": {
                    self._tower_order[i]: count
                    for i, count in enumerate(self.router._stats["tower_selections"])
                },
                "average_confidence": (
                    sum(self.router._stats["confidence_scores"]) /
                    len(self.router._stats["confidence_scores"])
                    if self.router._stats["confidence_scores"] else 0
                )
            }
        
        return {
            "tower_stats": tower_stats,
            "routing_stats": routing_stats,
            "inference_stats": self._inference_stats
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get RouterTower statistics."""
        base_stats = super().get_stats()
        router_stats = {
            "num_towers": len(self.towers),
            "strategy": self.config.strategy.value,
            "tower_names": self.list_towers(),
        }
        return {**base_stats, **router_stats, **self.get_tower_stats()}


__all__ = [
    'RouterTower',
    'RouterTowerConfig',
    'TowerRouter',
    'RoutingStrategy'
]