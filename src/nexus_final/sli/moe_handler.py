"""
MoE (Mixture of Experts) Handler for Universal SLI

Provides specialized handling for MoE architectures including:
- Expert routing and selection
- Weight sharding for sparse expert loading
- Support for Mixtral, Qwen2-MoE, DeepSeek-MoE, and other MoE variants
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Set
import torch
import torch.nn as nn
from transformers import PretrainedConfig

from .exceptions import MoEConfigurationError


@dataclass
class MoEConfig:
    """Configuration for MoE models."""
    num_experts: int = 8
    top_k: int = 2
    moe_type: str = "mixtral"  # mixtral, qwen2_moe, deepseek, grok
    has_shared_experts: bool = False
    num_shared_experts: int = 0
    expert_capacity: Optional[int] = None
    router_aux_loss_coef: float = 0.001
    
    def __post_init__(self):
        """Validate configuration."""
        if self.top_k > self.num_experts:
            raise MoEConfigurationError(
                self.moe_type,
                f"top_k ({self.top_k}) cannot be greater than num_experts ({self.num_experts})"
            )


class MoEHandler:
    """
    Handler for MoE-specific operations.
    
    Provides functionality for:
    - Detecting MoE configuration from model config
    - Getting expert weight patterns
    - Determining which experts to load
    - Computing routing decisions
    """
    
    # MoE type detection patterns
    MOE_PATTERNS = {
        "mixtral": {
            "model_types": ["mixtral"],
            "architectures": ["MixtralForCausalLM"],
            "expert_attr": "num_local_experts",
            "top_k_attr": "num_experts_per_tok",
        },
        "qwen2_moe": {
            "model_types": ["qwen2_moe"],
            "architectures": ["Qwen2MoeForCausalLM"],
            "expert_attr": "num_experts",
            "top_k_attr": "num_experts_per_tok",
        },
        "deepseek": {
            "model_types": ["deepseek"],
            "architectures": ["DeepseekMoeForCausalLM", "DeepseekForCausalLM"],
            "expert_attr": "n_routed_experts",
            "top_k_attr": "num_experts_per_tok",
            "shared_attr": "n_shared_experts",
        },
        "grok": {
            "model_types": ["grok"],
            "architectures": ["GrokForCausalLM", "Grok1ForCausalLM"],
            "expert_attr": "num_experts",
            "top_k_attr": "top_k",
        },
        "glm4_moe": {
            "model_types": ["glm4_moe", "glm4_moe_lite"],
            "architectures": ["Glm4MoeForCausalLM", "Glm4MoeLiteForCausalLM"],
            "expert_attr": "num_experts",
            "top_k_attr": "top_k",
        },
    }
    
    def __init__(self, config: PretrainedConfig):
        """
        Initialize MoE handler from model config.
        
        Args:
            config: Model configuration
        """
        self.config = config
        self.moe_config = self._parse_config(config)
    
    def _parse_config(self, config: PretrainedConfig) -> MoEConfig:
        """
        Parse MoE configuration from model config.
        
        Args:
            config: Model configuration
            
        Returns:
            MoEConfig instance
        """
        model_type = getattr(config, "model_type", "").lower()
        architectures = getattr(config, "architectures", [])
        
        # Detect MoE type
        moe_type = self._detect_moe_type(model_type, architectures)
        pattern = self.MOE_PATTERNS.get(moe_type, self.MOE_PATTERNS["mixtral"])
        
        # Extract expert count
        expert_attr = pattern.get("expert_attr", "num_experts")
        num_experts = getattr(config, expert_attr, 8)
        
        # Extract top-k
        top_k_attr = pattern.get("top_k_attr", "top_k")
        top_k = getattr(config, top_k_attr, 2)
        
        # Check for shared experts (DeepSeek style)
        has_shared = False
        num_shared = 0
        if "shared_attr" in pattern:
            shared_attr = pattern["shared_attr"]
            if hasattr(config, shared_attr):
                has_shared = True
                num_shared = getattr(config, shared_attr)
        
        return MoEConfig(
            num_experts=num_experts,
            top_k=top_k,
            moe_type=moe_type,
            has_shared_experts=has_shared,
            num_shared_experts=num_shared,
        )
    
    def _detect_moe_type(self, model_type: str, architectures: List[str]) -> str:
        """
        Detect the specific MoE type.
        
        Args:
            model_type: Model type from config
            architectures: List of architecture names
            
        Returns:
            MoE type identifier
        """
        for moe_type, pattern in self.MOE_PATTERNS.items():
            # Check model types
            if any(mt in model_type for mt in pattern["model_types"]):
                return moe_type
            # Check architectures
            for arch in architectures:
                if any(pa in arch for pa in pattern["architectures"]):
                    return moe_type
        
        # Default to mixtral if MoE attributes are present
        return "mixtral"
    
    def is_moe_layer(self, layer_idx: int) -> bool:
        """
        Check if a specific layer is an MoE layer.
        
        In many MoE architectures, not all layers are MoE layers.
        For example, Mixtral uses MoE in every other layer.
        
        Args:
            layer_idx: Layer index
            
        Returns:
            True if layer is an MoE layer
        """
        config = self.config
        
        # Check for MoE layer configuration
        if hasattr(config, "moe_layer_interval"):
            interval = getattr(config, "moe_layer_interval", 1)
            return layer_idx % interval == 0
        
        # DeepSeek: specific layers are MoE
        if hasattr(config, "moe_layer_indices"):
            moe_indices = getattr(config, "moe_layer_indices", [])
            return layer_idx in moe_indices
        
        # Default: all layers are MoE (Mixtral style)
        return True
    
    def get_expert_weight_pattern(self, layer_idx: int, expert_idx: int) -> str:
        """
        Get weight pattern for a specific expert.
        
        Args:
            layer_idx: Layer index
            expert_idx: Expert index
            
        Returns:
            Weight name prefix for the expert
        """
        moe_type = self.moe_config.moe_type
        
        if moe_type == "mixtral":
            return f"model.layers.{layer_idx}.block_sparse_moe.experts.{expert_idx}."
        elif moe_type == "qwen2_moe":
            return f"model.layers.{layer_idx}.mlp.experts.{expert_idx}."
        elif moe_type == "deepseek":
            return f"model.layers.{layer_idx}.mlp.experts.{expert_idx}."
        elif moe_type == "grok":
            return f"model.layers.{layer_idx}.moe_block.experts.{expert_idx}."
        else:
            # Default pattern
            return f"model.layers.{layer_idx}.mlp.experts.{expert_idx}."
    
    def get_router_weight_pattern(self, layer_idx: int) -> str:
        """
        Get weight pattern for the router/gate.
        
        Args:
            layer_idx: Layer index
            
        Returns:
            Weight name prefix for the router
        """
        moe_type = self.moe_config.moe_type
        
        if moe_type == "mixtral":
            return f"model.layers.{layer_idx}.block_sparse_moe.gate."
        elif moe_type in ["qwen2_moe", "deepseek"]:
            return f"model.layers.{layer_idx}.mlp.gate."
        elif moe_type == "grok":
            return f"model.layers.{layer_idx}.moe_block.gate."
        else:
            return f"model.layers.{layer_idx}.mlp.gate."
    
    def get_expert_indices_to_load(
        self,
        layer_idx: int,
        routing_weights: Optional[torch.Tensor] = None
    ) -> Set[int]:
        """
        Determine which experts to load for a layer.
        
        Args:
            layer_idx: Layer index
            routing_weights: Optional pre-computed routing weights
            
        Returns:
            Set of expert indices to load
        """
        if not self.is_moe_layer(layer_idx):
            return set()
        
        num_experts = self.moe_config.num_experts
        top_k = self.moe_config.top_k
        
        # If routing weights provided, select top-k
        if routing_weights is not None:
            top_indices = torch.topk(routing_weights, top_k).indices
            return set(top_indices.tolist())
        
        # Otherwise, load all experts (for training/analysis)
        # Or load top-k based on some heuristic
        return set(range(num_experts))
    
    def compute_routing_weights(
        self,
        hidden_states: torch.Tensor,
        router_weights: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute routing weights for experts.
        
        Args:
            hidden_states: Input hidden states [batch_size, seq_len, hidden_dim]
            router_weights: Router weight matrix [hidden_dim, num_experts]
            
        Returns:
            Tuple of (expert_indices, expert_weights)
        """
        # Simple routing: linear projection + softmax
        router_logits = torch.matmul(hidden_states, router_weights)
        
        # Select top-k experts
        top_k = self.moe_config.top_k
        expert_weights, expert_indices = torch.topk(
            torch.softmax(router_logits, dim=-1),
            top_k,
            dim=-1
        )
        
        # Normalize weights
        expert_weights = expert_weights / expert_weights.sum(dim=-1, keepdim=True)
        
        return expert_indices, expert_weights
    
    def load_expert_weights(
        self,
        layer_idx: int,
        expert_indices: Set[int],
        weight_loader: callable
    ) -> Dict[int, Dict[str, torch.Tensor]]:
        """
        Load weights for specific experts.
        
        Args:
            layer_idx: Layer index
            expert_indices: Set of expert indices to load
            weight_loader: Function to load weights by name
            
        Returns:
            Dictionary mapping expert index to weights
        """
        expert_weights = {}
        
        for expert_idx in expert_indices:
            pattern = self.get_expert_weight_pattern(layer_idx, expert_idx)
            
            # Load weights matching this pattern
            weights = weight_loader(pattern)
            expert_weights[expert_idx] = weights
        
        return expert_weights
    
    def get_model_info(self) -> Dict[str, any]:
        """Get MoE model information."""
        return {
            "moe_type": self.moe_config.moe_type,
            "num_experts": self.moe_config.num_experts,
            "top_k": self.moe_config.top_k,
            "has_shared_experts": self.moe_config.has_shared_experts,
            "num_shared_experts": self.moe_config.num_shared_experts,
        }
