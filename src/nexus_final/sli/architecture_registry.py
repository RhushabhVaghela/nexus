"""
Architecture Registry for Universal SLI

This module provides a registry system for handling 130+ model architectures
categorized by family. It auto-detects architecture families from model configs
and provides family-specific handlers for layer creation and weight mapping.
"""

import json
import os
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any, Type
from pathlib import Path

import torch
import torch.nn as nn
from transformers import PretrainedConfig

from .exceptions import UnsupportedArchitectureError, LayerCreationError


class ArchitectureFamily(ABC):
    """Abstract base class for architecture families."""
    
    family_id: str = ""
    family_name: str = ""
    model_types: List[str] = []
    architectures: List[str] = []
    trust_remote_code: bool = False
    
    def matches(self, model_type: str, architectures: List[str]) -> bool:
        """
        Check if config matches this family.
        
        Args:
            model_type: The model_type from config
            architectures: List of architecture names from config
            
        Returns:
            True if this family handles the given config
        """
        model_type_lower = model_type.lower() if model_type else ""
        architectures_lower = [a.lower() for a in architectures] if architectures else []
        
        # Check model type
        if model_type_lower in [mt.lower() for mt in self.model_types]:
            return True
            
        # Check architectures
        for arch in architectures_lower:
            # Direct match
            if arch in [a.lower() for a in self.architectures]:
                return True
            # Partial match (e.g., "LlamaForCausalLM" matches "llama" family)
            if self.family_id.lower() in arch:
                return True
                
        return False
    
    @abstractmethod
    def get_layer_prefix(self, layer_idx: int, layer_type: str = "decoder") -> str:
        """
        Get weight prefix for layer.
        
        Args:
            layer_idx: Layer index
            layer_type: Type of layer ("decoder", "encoder")
            
        Returns:
            Weight name prefix for the layer
        """
        pass
    
    @abstractmethod
    def create_layer(self, config: PretrainedConfig, layer_idx: int, 
                     layer_type: str = "decoder") -> nn.Module:
        """
        Create a layer instance for this architecture family.
        
        Args:
            config: Model configuration
            layer_idx: Layer index
            layer_type: Type of layer ("decoder", "encoder")
            
        Returns:
            Instantiated layer module
        """
        pass
    
    def get_embedding_name(self) -> str:
        """Get the embedding weight name."""
        return "embed_tokens"
    
    def get_lm_head_name(self) -> str:
        """Get the LM head weight name."""
        return "lm_head"
    
    def get_num_layers(self, config: PretrainedConfig) -> int:
        """Get number of layers from config."""
        # Common attribute names for layer count
        for attr in ["num_hidden_layers", "n_layer", "num_layers", "n_layers"]:
            if hasattr(config, attr):
                return getattr(config, attr)
        raise ValueError(f"Cannot determine number of layers from config: {config}")
    
    def get_hidden_size(self, config: PretrainedConfig) -> int:
        """Get hidden size from config."""
        for attr in ["hidden_size", "d_model", "n_embd", "embed_dim"]:
            if hasattr(config, attr):
                return getattr(config, attr)
        raise ValueError(f"Cannot determine hidden size from config: {config}")
    
    def get_vocab_size(self, config: PretrainedConfig) -> int:
        """Get vocabulary size from config."""
        for attr in ["vocab_size", "n_vocab", "vocab_size_original"]:
            if hasattr(config, attr):
                return getattr(config, attr)
        raise ValueError(f"Cannot determine vocab size from config: {config}")


class LlamaFamilyHandler(ArchitectureFamily):
    """Handler for Llama-based architectures (Llama, Mistral, Mixtral, Qwen2, etc.)."""
    
    family_id = "llama"
    family_name = "Llama-Based Architectures"
    model_types = [
        "llama", "llama2", "llama3", "llama4", "mistral", "mixtral",
        "yi", "deepseek", "codellama", "vicuna", "alpaca", "wizardlm",
        "openchat", "zephyr", "starling", "neural-chat"
    ]
    architectures = [
        "LlamaForCausalLM", "Llama4ForCausalLM", "MistralForCausalLM",
        "MixtralForCausalLM", "YiForCausalLM", "DeepseekForCausalLM",
        "CodellamaForCausalLM", "LlamaModel", "MistralModel"
    ]
    
    def get_layer_prefix(self, layer_idx: int, layer_type: str = "decoder") -> str:
        return f"model.layers.{layer_idx}."
    
    def create_layer(self, config: PretrainedConfig, layer_idx: int,
                     layer_type: str = "decoder") -> nn.Module:
        try:
            from transformers.models.llama.modeling_llama import LlamaDecoderLayer
            return LlamaDecoderLayer(config, layer_idx=layer_idx)
        except ImportError as e:
            raise LayerCreationError(layer_idx, self.family_id, e)
    
    def get_embedding_name(self) -> str:
        return "model.embed_tokens"
    
    def get_lm_head_name(self) -> str:
        return "lm_head"


class QwenFamilyHandler(ArchitectureFamily):
    """Handler for Qwen-based architectures."""
    
    family_id = "qwen"
    family_name = "Qwen-Based Architectures"
    model_types = [
        "qwen", "qwen2", "qwen2_5", "qwen2_vl", "qwen2_omni",
        "qwen2_5_omni", "qwen2_5_vl", "qwen3"
    ]
    architectures = [
        "Qwen2ForCausalLM", "Qwen2VLForConditionalGeneration",
        "Qwen2OmniTalkerForConditionalGeneration", "Qwen2_5OmniForConditionalGeneration",
        "Qwen2_5_VLForConditionalGeneration", "Qwen3ForCausalLM",
        "Qwen3VLForConditionalGeneration", "Qwen3OmniForConditionalGeneration"
    ]
    
    def get_layer_prefix(self, layer_idx: int, layer_type: str = "decoder") -> str:
        return f"model.layers.{layer_idx}."
    
    def create_layer(self, config: PretrainedConfig, layer_idx: int,
                     layer_type: str = "decoder") -> nn.Module:
        try:
            from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer
            return Qwen2DecoderLayer(config, layer_idx=layer_idx)
        except ImportError as e:
            raise LayerCreationError(layer_idx, self.family_id, e)
    
    def get_embedding_name(self) -> str:
        return "model.embed_tokens"


class GPTFamilyHandler(ArchitectureFamily):
    """Handler for GPT-based architectures (GPT-2, GPT-J, GPT-NeoX, etc.)."""
    
    family_id = "gpt"
    family_name = "GPT-Based Architectures"
    model_types = [
        "gpt2", "gptj", "gpt_neo", "gpt_neox", "gpt_bigcode",
        "pythia", "falcon", "starcoder", "santacoder"
    ]
    architectures = [
        "GPT2LMHeadModel", "GPTJForCausalLM", "GPTNeoForCausalLM",
        "GPTNeoXForCausalLM", "GPTBigCodeForCausalLM", "FalconForCausalLM"
    ]
    
    def _detect_subtype(self, config: PretrainedConfig) -> str:
        """Detect the specific GPT subtype from config."""
        model_type = getattr(config, "model_type", "").lower()
        architectures = getattr(config, "architectures", [])
        
        if "gpt2" in model_type or any("GPT2" in a for a in architectures):
            return "gpt2"
        elif "gptj" in model_type or any("GPTJ" in a for a in architectures):
            return "gptj"
        elif "gpt_neox" in model_type or any("GPTNeoX" in a for a in architectures):
            return "gpt_neox"
        elif "gpt_neo" in model_type or any("GPTNeo" in a for a in architectures):
            return "gpt_neo"
        elif "gpt_bigcode" in model_type or any("GPTBigCode" in a for a in architectures):
            return "gpt_bigcode"
        elif "falcon" in model_type or any("Falcon" in a for a in architectures):
            return "falcon"
        else:
            return "gpt2"  # Default fallback
    
    def get_layer_prefix(self, layer_idx: int, layer_type: str = "decoder") -> str:
        return f"transformer.h.{layer_idx}."
    
    def create_layer(self, config: PretrainedConfig, layer_idx: int,
                     layer_type: str = "decoder") -> nn.Module:
        subtype = self._detect_subtype(config)
        
        try:
            if subtype == "gpt2":
                from transformers.models.gpt2.modeling_gpt2 import GPT2Block
                return GPT2Block(config, layer_idx=layer_idx)
            elif subtype == "gptj":
                from transformers.models.gptj.modeling_gptj import GPTJBlock
                return GPTJBlock(config, layer_idx=layer_idx)
            elif subtype == "gpt_neox":
                from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXLayer
                return GPTNeoXLayer(config, layer_idx=layer_idx)
            elif subtype == "gpt_neo":
                from transformers.models.gpt_neo.modeling_gpt_neo import GPTNeoBlock
                return GPTNeoBlock(config, layer_idx=layer_idx)
            elif subtype == "gpt_bigcode":
                from transformers.models.gpt_bigcode.modeling_gpt_bigcode import GPTBigCodeBlock
                return GPTBigCodeBlock(config, layer_idx=layer_idx)
            elif subtype == "falcon":
                from transformers.models.falcon.modeling_falcon import FalconDecoderLayer
                return FalconDecoderLayer(config, layer_idx=layer_idx)
            else:
                from transformers.models.gpt2.modeling_gpt2 import GPT2Block
                return GPT2Block(config, layer_idx=layer_idx)
        except ImportError as e:
            raise LayerCreationError(layer_idx, self.family_id, e)
    
    def get_embedding_name(self) -> str:
        return "transformer.wte"
    
    def get_lm_head_name(self) -> str:
        return "lm_head"


class ChatGLMFamilyHandler(ArchitectureFamily):
    """Handler for ChatGLM-based architectures."""
    
    family_id = "chatglm"
    family_name = "ChatGLM-Based Architectures"
    model_types = ["chatglm", "chatglm2", "chatglm3", "glm4", "glm4_moe"]
    architectures = [
        "ChatGLMForConditionalGeneration", "ChatGLMModel",
        "Glm4ForCausalLM", "Glm4MoeForCausalLM", "Glm4MoeLiteForCausalLM"
    ]
    trust_remote_code = True
    
    def get_layer_prefix(self, layer_idx: int, layer_type: str = "decoder") -> str:
        return f"transformer.encoder.layers.{layer_idx}."
    
    def create_layer(self, config: PretrainedConfig, layer_idx: int,
                     layer_type: str = "decoder") -> nn.Module:
        try:
            # ChatGLM models often require trust_remote_code
            # Try to import from transformers or from modeling file
            try:
                from transformers.models.chatglm.modeling_chatglm import GLMBlock
                return GLMBlock(config, layer_idx=layer_idx)
            except ImportError:
                # Fallback: try loading with custom modeling
                import importlib.util
                spec = importlib.util.spec_from_file_location(
                    "modeling_chatglm", 
                    Path(config.name_or_path) / "modeling_chatglm.py"
                )
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    return module.GLMBlock(config, layer_idx=layer_idx)
                else:
                    raise ImportError("Cannot find ChatGLM modeling file")
        except Exception as e:
            raise LayerCreationError(layer_idx, self.family_id, e)
    
    def get_embedding_name(self) -> str:
        return "transformer.embedding.word_embeddings"


class T5FamilyHandler(ArchitectureFamily):
    """Handler for T5-based architectures (T5, FLAN-T5, UL2, etc.)."""
    
    family_id = "t5"
    family_name = "T5-Based Architectures"
    model_types = ["t5", "t5v1_1", "flan_t5", "ul2", "longt5", "byt5", "umt5", "mt5"]
    architectures = [
        "T5ForConditionalGeneration", "T5EncoderModel", "T5Model",
        "LongT5ForConditionalGeneration", "UMT5ForConditionalGeneration"
    ]
    
    def get_layer_prefix(self, layer_idx: int, layer_type: str = "decoder") -> str:
        if layer_type == "encoder":
            return f"encoder.block.{layer_idx}."
        else:
            return f"decoder.block.{layer_idx}."
    
    def create_layer(self, config: PretrainedConfig, layer_idx: int,
                     layer_type: str = "decoder") -> nn.Module:
        try:
            from transformers.models.t5.modeling_t5 import T5Block
            return T5Block(config, layer_idx=layer_idx, has_relative_attention_bias=(layer_idx == 0))
        except ImportError as e:
            raise LayerCreationError(layer_idx, self.family_id, e)
    
    def get_embedding_name(self) -> str:
        return "shared"


class BLOOMFamilyHandler(ArchitectureFamily):
    """Handler for BLOOM-based architectures."""
    
    family_id = "bloom"
    family_name = "BLOOM-Based Architectures"
    model_types = ["bloom", "bloomz"]
    architectures = ["BloomForCausalLM", "BloomModel"]
    
    def get_layer_prefix(self, layer_idx: int, layer_type: str = "decoder") -> str:
        return f"transformer.h.{layer_idx}."
    
    def create_layer(self, config: PretrainedConfig, layer_idx: int,
                     layer_type: str = "decoder") -> nn.Module:
        try:
            from transformers.models.bloom.modeling_bloom import BloomBlock
            return BloomBlock(config, layer_idx=layer_idx)
        except ImportError as e:
            raise LayerCreationError(layer_idx, self.family_id, e)
    
    def get_embedding_name(self) -> str:
        return "transformer.word_embeddings"


class OPTFamilyHandler(ArchitectureFamily):
    """Handler for OPT-based architectures."""
    
    family_id = "opt"
    family_name = "OPT-Based Architectures"
    model_types = ["opt", "opt_iml"]
    architectures = ["OPTForCausalLM", "OPTModel"]
    
    def get_layer_prefix(self, layer_idx: int, layer_type: str = "decoder") -> str:
        return f"model.decoder.layers.{layer_idx}."
    
    def create_layer(self, config: PretrainedConfig, layer_idx: int,
                     layer_type: str = "decoder") -> nn.Module:
        try:
            from transformers.models.opt.modeling_opt import OPTDecoderLayer
            return OPTDecoderLayer(config, layer_idx=layer_idx)
        except ImportError as e:
            raise LayerCreationError(layer_idx, self.family_id, e)
    
    def get_embedding_name(self) -> str:
        return "model.decoder.embed_tokens"


class MambaFamilyHandler(ArchitectureFamily):
    """Handler for Mamba/State Space Model architectures."""
    
    family_id = "mamba"
    family_name = "Mamba/State Space Models"
    model_types = ["mamba", "mamba2", "falcon_mamba", "jamba", "zamba", "rwkv"]
    architectures = [
        "MambaForCausalLM", "Mamba2ForCausalLM", "MambaLMHeadModel",
        "FalconMambaForCausalLM", "JambaForCausalLM", "ZambaForCausalLM"
    ]
    
    def _detect_subtype(self, config: PretrainedConfig) -> str:
        """Detect Mamba subtype from config."""
        model_type = getattr(config, "model_type", "").lower()
        if "mamba2" in model_type:
            return "mamba2"
        elif "mamba" in model_type:
            return "mamba"
        elif "jamba" in model_type:
            return "jamba"
        elif "zamba" in model_type:
            return "zamba"
        else:
            return "mamba"
    
    def get_layer_prefix(self, layer_idx: int, layer_type: str = "decoder") -> str:
        return f"backbone.layers.{layer_idx}."
    
    def create_layer(self, config: PretrainedConfig, layer_idx: int,
                     layer_type: str = "decoder") -> nn.Module:
        subtype = self._detect_subtype(config)
        
        try:
            if subtype == "mamba2":
                from transformers.models.mamba2.modeling_mamba2 import Mamba2Block
                return Mamba2Block(config, layer_idx=layer_idx)
            elif subtype == "mamba":
                from transformers.models.mamba.modeling_mamba import MambaBlock
                return MambaBlock(config, layer_idx=layer_idx)
            elif subtype == "jamba":
                from transformers.models.jamba.modeling_jamba import JambaMambaLayer
                return JambaMambaLayer(config, layer_idx=layer_idx)
            elif subtype == "zamba":
                from transformers.models.zamba.modeling_zamba import ZambaBlock
                return ZambaBlock(config, layer_idx=layer_idx)
            else:
                from transformers.models.mamba.modeling_mamba import MambaBlock
                return MambaBlock(config, layer_idx=layer_idx)
        except ImportError as e:
            raise LayerCreationError(layer_idx, self.family_id, e)
    
    def get_embedding_name(self) -> str:
        return "backbone.embeddings"


class MoEFamilyHandler(ArchitectureFamily):
    """
    Handler for Mixture of Experts architectures.
    
    Note: MoE handlers extend base architectures (usually Llama-like)
    with expert-specific handling.
    """
    
    family_id = "moe"
    family_name = "Mixture of Experts Architectures"
    model_types = [
        "mixtral", "qwen2_moe", "deepseek_moe", "grok", "glm4_moe"
    ]
    architectures = [
        "MixtralForCausalLM", "Qwen2MoeForCausalLM", "DeepseekMoeForCausalLM",
        "GrokForCausalLM", "Glm4MoeForCausalLM", "PhiMoEForCausalLM"
    ]
    
    def _detect_subtype(self, config: PretrainedConfig) -> str:
        """Detect MoE subtype from config."""
        model_type = getattr(config, "model_type", "").lower()
        architectures = getattr(config, "architectures", [])
        
        if "mixtral" in model_type:
            return "mixtral"
        elif "qwen2_moe" in model_type:
            return "qwen2_moe"
        elif "deepseek" in model_type:
            return "deepseek"
        elif "grok" in model_type:
            return "grok"
        elif "glm4_moe" in model_type:
            return "glm4_moe"
        elif any("Mixtral" in a for a in architectures):
            return "mixtral"
        elif any("Qwen2Moe" in a for a in architectures):
            return "qwen2_moe"
        else:
            return "mixtral"  # Default
    
    def get_layer_prefix(self, layer_idx: int, layer_type: str = "decoder") -> str:
        return f"model.layers.{layer_idx}."
    
    def get_expert_prefix(self, layer_idx: int, expert_idx: int) -> str:
        """Get weight prefix for a specific expert."""
        return f"model.layers.{layer_idx}.block_sparse_moe.experts.{expert_idx}."
    
    def create_layer(self, config: PretrainedConfig, layer_idx: int,
                     layer_type: str = "decoder") -> nn.Module:
        subtype = self._detect_subtype(config)
        
        try:
            if subtype == "mixtral":
                from transformers.models.mixtral.modeling_mixtral import MixtralDecoderLayer
                return MixtralDecoderLayer(config, layer_idx=layer_idx)
            elif subtype == "qwen2_moe":
                from transformers.models.qwen2_moe.modeling_qwen2_moe import Qwen2MoeDecoderLayer
                return Qwen2MoeDecoderLayer(config, layer_idx=layer_idx)
            elif subtype == "deepseek":
                from transformers.models.deepseek.modeling_deepseek import DeepseekDecoderLayer
                return DeepseekDecoderLayer(config, layer_idx=layer_idx)
            else:
                from transformers.models.mixtral.modeling_mixtral import MixtralDecoderLayer
                return MixtralDecoderLayer(config, layer_idx=layer_idx)
        except ImportError as e:
            raise LayerCreationError(layer_idx, self.family_id, e)
    
    def get_num_experts(self, config: PretrainedConfig) -> int:
        """Get number of experts from config."""
        for attr in ["num_local_experts", "num_experts", "n_routed_experts"]:
            if hasattr(config, attr):
                return getattr(config, attr)
        return 8  # Default
    
    def get_top_k(self, config: PretrainedConfig) -> int:
        """Get top-k routing value from config."""
        for attr in ["num_experts_per_tok", "top_k", "topk"]:
            if hasattr(config, attr):
                return getattr(config, attr)
        return 2  # Default


class PhiFamilyHandler(ArchitectureFamily):
    """Handler for Phi-based architectures."""
    
    family_id = "phi"
    family_name = "Phi-Based Architectures"
    model_types = ["phi", "phi2", "phi3", "phi4"]
    architectures = ["PhiForCausalLM", "Phi3ForCausalLM", "PhiMoEForCausalLM"]
    
    def get_layer_prefix(self, layer_idx: int, layer_type: str = "decoder") -> str:
        return f"model.layers.{layer_idx}."
    
    def create_layer(self, config: PretrainedConfig, layer_idx: int,
                     layer_type: str = "decoder") -> nn.Module:
        try:
            from transformers.models.phi.modeling_phi import PhiDecoderLayer
            return PhiDecoderLayer(config, layer_idx=layer_idx)
        except ImportError as e:
            raise LayerCreationError(layer_idx, self.family_id, e)
    
    def get_embedding_name(self) -> str:
        return "model.embed_tokens"


class GemmaFamilyHandler(ArchitectureFamily):
    """Handler for Gemma-based architectures."""
    
    family_id = "gemma"
    family_name = "Gemma-Based Architectures"
    model_types = ["gemma", "gemma2", "gemma3", "gemma3_text"]
    architectures = [
        "GemmaForCausalLM", "Gemma2ForCausalLM", "Gemma3ForCausalLM"
    ]
    
    def get_layer_prefix(self, layer_idx: int, layer_type: str = "decoder") -> str:
        return f"model.layers.{layer_idx}."
    
    def create_layer(self, config: PretrainedConfig, layer_idx: int,
                     layer_type: str = "decoder") -> nn.Module:
        try:
            from transformers.models.gemma.modeling_gemma import GemmaDecoderLayer
            return GemmaDecoderLayer(config, layer_idx=layer_idx)
        except ImportError as e:
            raise LayerCreationError(layer_idx, self.family_id, e)
    
    def get_embedding_name(self) -> str:
        return "model.embed_tokens"


class ArchitectureRegistry:
    """
    Registry for all supported architecture families.
    
    This class provides:
    - Registration of architecture families
    - Auto-detection of architecture family from config
    - Access to family-specific handlers
    """
    
    _instance = None
    _families: Dict[str, ArchitectureFamily] = {}
    _initialized = False
    
    def __new__(cls):
        """Singleton pattern to ensure single registry instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self._register_default_families()
            ArchitectureRegistry._initialized = True
    
    def _register_default_families(self):
        """Register all built-in architecture families."""
        families = [
            LlamaFamilyHandler(),
            QwenFamilyHandler(),
            GPTFamilyHandler(),
            ChatGLMFamilyHandler(),
            T5FamilyHandler(),
            BLOOMFamilyHandler(),
            OPTFamilyHandler(),
            MambaFamilyHandler(),
            MoEFamilyHandler(),
            PhiFamilyHandler(),
            GemmaFamilyHandler(),
        ]
        
        for family in families:
            self.register(family.family_id, family)
    
    def register(self, family_id: str, family: ArchitectureFamily):
        """
        Register an architecture family.
        
        Args:
            family_id: Unique identifier for the family
            family: ArchitectureFamily instance
        """
        self._families[family_id] = family
    
    def get_family(self, family_id: str) -> Optional[ArchitectureFamily]:
        """
        Get a family handler by ID.
        
        Args:
            family_id: Family identifier
            
        Returns:
            ArchitectureFamily instance or None
        """
        return self._families.get(family_id)
    
    def detect_family(self, config: PretrainedConfig) -> ArchitectureFamily:
        """
        Auto-detect architecture family from config.
        
        Args:
            config: Model configuration object
            
        Returns:
            ArchitectureFamily instance
            
        Raises:
            UnsupportedArchitectureError: If family cannot be detected
        """
        model_type = getattr(config, "model_type", "").lower()
        architectures = getattr(config, "architectures", [])
        
        # Try each registered family
        for family in self._families.values():
            if family.matches(model_type, architectures):
                return family
        
        # Special handling: Check if it's an MoE model by attributes
        if self._is_moe_model(config):
            return self._families.get("moe") or self._families.get("llama")
        
        # If no match found, raise error
        raise UnsupportedArchitectureError(model_type, architectures)
    
    def _is_moe_model(self, config: PretrainedConfig) -> bool:
        """Check if config indicates an MoE model."""
        moe_attrs = ["num_local_experts", "n_routed_experts", "moe_intermediate_size"]
        return any(hasattr(config, attr) for attr in moe_attrs)
    
    def list_families(self) -> List[str]:
        """List all registered family IDs."""
        return list(self._families.keys())
    
    def get_family_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all registered families."""
        info = {}
        for family_id, family in self._families.items():
            info[family_id] = {
                "name": family.family_name,
                "model_types": family.model_types,
                "architectures": family.architectures,
                "trust_remote_code": family.trust_remote_code,
            }
        return info


# Global registry instance
_registry = None


def get_registry() -> ArchitectureRegistry:
    """Get the global architecture registry instance."""
    global _registry
    if _registry is None:
        _registry = ArchitectureRegistry()
    return _registry
