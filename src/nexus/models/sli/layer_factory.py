"""
Universal Layer Factory for SLI

Provides factory methods for creating layer instances from any supported
architecture family.
"""

from typing import Dict, Optional, Any
import torch.nn as nn
from transformers import PretrainedConfig

from .architecture_registry import (
    ArchitectureRegistry, 
    ArchitectureFamily,
    get_registry
)
from .exceptions import LayerCreationError


class UniversalLayerFactory:
    """
    Factory for creating architecture-agnostic layers.
    
    This factory uses the ArchitectureRegistry to detect the appropriate
    architecture family and create layer instances accordingly.
    """
    
    def __init__(self, registry: Optional[ArchitectureRegistry] = None):
        """
        Initialize the layer factory.
        
        Args:
            registry: ArchitectureRegistry instance (uses global if None)
        """
        self.registry = registry or get_registry()
        self._layer_cache: Dict[str, nn.Module] = {}
    
    def create_layer(
        self,
        config: PretrainedConfig,
        layer_idx: int,
        layer_type: str = "decoder"
    ) -> nn.Module:
        """
        Create a layer instance from config.
        
        Args:
            config: Model configuration
            layer_idx: Layer index
            layer_type: Type of layer ("decoder" or "encoder")
            
        Returns:
            Instantiated layer module
            
        Raises:
            LayerCreationError: If layer creation fails
        """
        family = self.registry.detect_family(config)
        return family.create_layer(config, layer_idx, layer_type)
    
    def get_weight_prefix(
        self,
        config: PretrainedConfig,
        layer_idx: int,
        layer_type: str = "decoder"
    ) -> str:
        """
        Get weight prefix for layer.
        
        Args:
            config: Model configuration
            layer_idx: Layer index
            layer_type: Type of layer ("decoder" or "encoder")
            
        Returns:
            Weight name prefix
        """
        family = self.registry.detect_family(config)
        return family.get_layer_prefix(layer_idx, layer_type)
    
    def get_embedding_info(self, config: PretrainedConfig) -> Dict[str, str]:
        """
        Get embedding-related weight names.
        
        Args:
            config: Model configuration
            
        Returns:
            Dictionary with embedding and lm_head names
        """
        family = self.registry.detect_family(config)
        return {
            "embedding": family.get_embedding_name(),
            "lm_head": family.get_lm_head_name(),
        }
    
    def get_model_info(self, config: PretrainedConfig) -> Dict[str, Any]:
        """
        Get general model information from config.
        
        Args:
            config: Model configuration
            
        Returns:
            Dictionary with model metadata
        """
        family = self.registry.detect_family(config)
        
        return {
            "family_id": family.family_id,
            "family_name": family.family_name,
            "num_layers": family.get_num_layers(config),
            "hidden_size": family.get_hidden_size(config),
            "vocab_size": family.get_vocab_size(config),
            "trust_remote_code": family.trust_remote_code,
        }
    
    def is_moe_model(self, config: PretrainedConfig) -> bool:
        """
        Check if config represents an MoE model.
        
        Args:
            config: Model configuration
            
        Returns:
            True if MoE model
        """
        try:
            family = self.registry.detect_family(config)
            return family.family_id == "moe"
        except Exception:
            # Check directly for MoE attributes
            moe_attrs = ["num_local_experts", "n_routed_experts"]
            return any(hasattr(config, attr) for attr in moe_attrs)
