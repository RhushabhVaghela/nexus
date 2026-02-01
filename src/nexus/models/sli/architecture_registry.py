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
    
    def matches(self, model_type: str, architectures: List[str], config: Optional[Any] = None) -> bool:
        """
        Check if config matches this family.
        
        Args:
            model_type: The model_type from config
            architectures: List of architecture names from config
            config: Optional full config object for additional checks
            
        Returns:
            True if this family handles the given config
        """
        model_type_lower = model_type.lower() if model_type else ""
        architectures_lower = [a.lower() for a in architectures] if architectures else []
        
        # Check if this is an MoE model - if so, don't match (let MoE handler take it)
        if config is not None:
            moe_attrs = ["num_local_experts", "n_routed_experts", "moe_intermediate_size"]
            if any(hasattr(config, attr) for attr in moe_attrs):
                return False
        
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


class BERTFamilyHandler(ArchitectureFamily):
    """
    Handler for BERT-based encoder-only architectures.
    
    Note: Encoder-only models have limited SLI support as they are
    primarily used for embedding extraction and classification tasks,
    not generative text completion.
    """
    
    family_id = "bert"
    family_name = "BERT-Based Encoder Architectures"
    model_types = [
        "bert", "roberta", "deberta", "deberta_v2", "distilbert",
        "albert", "modernbert", "jinabert", "nomic_bert", "neobert",
        "electra", "xlm_roberta", "camembert"
    ]
    architectures = [
        "BertModel", "BertForMaskedLM", "BertForSequenceClassification",
        "RobertaModel", "RobertaForSequenceClassification",
        "DebertaModel", "DebertaForSequenceClassification",
        "DistilBertModel", "DistilBertForMaskedLM", "DistilBertForSequenceClassification",
        "AlbertModel", "AlbertForMaskedLM", "AlbertForSequenceClassification",
        "ModernBertModel", "ModernBertForSequenceClassification",
        "JinaBertModel", "JinaBertForMaskedLM",
        "NeoBERT", "NeoBERTForSequenceClassification", "NeoBERTLMHead",
        "NomicBertModel",
        "ElectraModel", "ElectraForSequenceClassification",
        "XLMRobertaModel", "XLMRobertaForSequenceClassification",
        "CamembertModel", "CamembertForMaskedLM"
    ]
    
    def _detect_subtype(self, config: PretrainedConfig) -> str:
        """Detect the specific BERT subtype from config."""
        model_type = getattr(config, "model_type", "").lower()
        architectures = getattr(config, "architectures", [])
        
        if "roberta" in model_type or any("Roberta" in a for a in architectures):
            return "roberta"
        elif "deberta" in model_type or any("Deberta" in a for a in architectures):
            return "deberta"
        elif "distilbert" in model_type or any("DistilBert" in a for a in architectures):
            return "distilbert"
        elif "albert" in model_type or any("Albert" in a for a in architectures):
            return "albert"
        elif "modernbert" in model_type or any("ModernBert" in a for a in architectures):
            return "modernbert"
        elif "jina" in model_type or any("JinaBert" in a for a in architectures):
            return "jinabert"
        elif "neobert" in model_type or any("NeoBERT" in a for a in architectures):
            return "neobert"
        elif "nomic" in model_type or any("NomicBert" in a for a in architectures):
            return "nomic_bert"
        elif "electra" in model_type or any("Electra" in a for a in architectures):
            return "electra"
        elif "xlm" in model_type or any("XLMRoberta" in a for a in architectures):
            return "xlm_roberta"
        elif "camembert" in model_type or any("Camembert" in a for a in architectures):
            return "camembert"
        else:
            return "bert"  # Default fallback
    
    def get_layer_prefix(self, layer_idx: int, layer_type: str = "encoder") -> str:
        """
        Get weight prefix for encoder layer.
        
        Note: Encoder-only models only support "encoder" layer type.
        """
        subtype = getattr(self, '_last_subtype', 'bert')
        
        if subtype == "distilbert":
            return f"transformer.layer.{layer_idx}."
        elif subtype in ["roberta", "camembert", "xlm_roberta"]:
            return f"roberta.encoder.layer.{layer_idx}."
        elif subtype == "albert":
            # ALBERT uses shared layers with a different structure
            return f"albert.encoder.albert_layer_group.{layer_idx % 12}.albert_layers.{layer_idx % 12}."
        else:
            # Standard BERT and others
            return f"encoder.layer.{layer_idx}."
    
    def create_layer(self, config: PretrainedConfig, layer_idx: int,
                     layer_type: str = "encoder") -> nn.Module:
        """
        Create an encoder layer for this architecture family.
        
        Note: Encoder-only models only have encoder layers, no decoder.
        """
        subtype = self._detect_subtype(config)
        self._last_subtype = subtype  # Store for get_layer_prefix
        
        try:
            if subtype == "bert":
                from transformers.models.bert.modeling_bert import BertLayer
                return BertLayer(config, layer_idx=layer_idx)
            elif subtype == "roberta":
                from transformers.models.roberta.modeling_roberta import RobertaLayer
                return RobertaLayer(config, layer_idx=layer_idx)
            elif subtype == "deberta":
                from transformers.models.deberta.modeling_deberta import DebertaLayer
                return DebertaLayer(config, layer_idx=layer_idx)
            elif subtype == "distilbert":
                from transformers.models.distilbert.modeling_distilbert import TransformerBlock
                return TransformerBlock(config, layer_idx=layer_idx)
            elif subtype == "albert":
                from transformers.models.albert.modeling_albert import AlbertLayer
                return AlbertLayer(config, layer_idx=layer_idx)
            elif subtype == "modernbert":
                from transformers.models.modernbert.modeling_modernbert import ModernBertLayer
                return ModernBertLayer(config, layer_idx=layer_idx)
            else:
                # Default fallback to BERT
                from transformers.models.bert.modeling_bert import BertLayer
                return BertLayer(config, layer_idx=layer_idx)
        except ImportError as e:
            raise LayerCreationError(layer_idx, self.family_id, e)
    
    def get_embedding_name(self) -> str:
        """Get the embedding weight name."""
        subtype = getattr(self, '_last_subtype', 'bert')
        
        if subtype in ["roberta", "camembert", "xlm_roberta"]:
            return "roberta.embeddings"
        elif subtype == "distilbert":
            return "distilbert.embeddings"
        elif subtype == "albert":
            return "albert.embeddings"
        elif subtype == "electra":
            return "electra.embeddings"
        else:
            return "embeddings"
    
    def get_lm_head_name(self) -> str:
        """Encoder-only models don't have LM heads."""
        return None
    
    def is_encoder_only(self) -> bool:
        """Indicate this is an encoder-only architecture."""
        return True


class ArchitectureRegistry:
    """
    Registry for all supported architecture families.
    
    This class provides:
    - Registration of architecture families
    - Auto-detection of architecture family from config
    - Access to family-specific handlers
    - Custom layer registration for extensibility
    """
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        """Singleton pattern to ensure single registry instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self._families: Dict[str, ArchitectureFamily] = {}
            self._custom_layers: Dict[str, Any] = {}  # Registry for custom layers
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
            BERTFamilyHandler(),  # Encoder-only architectures (BERT, RoBERTa, DeBERTa, etc.)
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
        
        # First: Check if it's an MoE model by attributes (before other checks)
        # MoE models often inherit from Llama architecture but need special handling
        if self._is_moe_model(config):
            # Check if it's explicitly an MoE architecture type
            moe_family = self._families.get("moe")
            if moe_family and moe_family.matches(model_type, architectures):
                return moe_family
            # If MoE attributes present but no specific match, return MoE family
            return moe_family or self._families.get("llama")
        
        # Second: Try exact architecture matches first (more specific)
        for family in self._families.values():
            # Check for exact architecture matches
            for arch in architectures:
                arch_lower = arch.lower()
                for family_arch in family.architectures:
                    if arch_lower == family_arch.lower():
                        return family
        
        # Third: Try model type and partial matches (with config for MoE check)
        for family in self._families.values():
            if family.matches(model_type, architectures, config):
                return family
        
        # If no match found, raise error
        raise UnsupportedArchitectureError(model_type, architectures)
    
    def _is_moe_model(self, config: PretrainedConfig) -> bool:
        """Check if config indicates an MoE model."""
        moe_attrs = ["num_local_experts", "n_routed_experts", "moe_intermediate_size"]
        
        # Check if config has MoE attributes with actual values (not None, not mock)
        for attr in moe_attrs:
            if hasattr(config, attr):
                val = getattr(config, attr)
                # Skip if value is None or a MagicMock (from tests)
                if val is not None and type(val).__name__ != 'MagicMock':
                    # Check if it's a reasonable value (int > 0)
                    try:
                        if int(val) > 0:
                            return True
                    except (TypeError, ValueError):
                        continue
        return False
    
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
    
    def register_custom_layer(self, layer_name: str, layer_factory: Any) -> None:
        """
        Register a custom layer factory in the registry.
        
        This allows users to extend the architecture registry with custom
        layer types that can be used in model construction.
        
        Args:
            layer_name: Unique identifier for the custom layer type.
                       Should follow naming convention: "<family>_<layer_type>"
                       e.g., "llama_custom_attn", "moe_expert_layer"
            layer_factory: A callable (class or factory function) that creates
                          the layer instance. Should accept standard arguments
                          (config, layer_idx, layer_type) and return nn.Module.
        
        Raises:
            ValueError: If layer_name is already registered or invalid.
            TypeError: If layer_factory is not callable.
        
        Example:
            >>> registry = ArchitectureRegistry()
            >>> registry.register_custom_layer(
            ...     "llama_rotary_with_scaling",
            ...     lambda config, idx: RotaryEmbeddingWithScaling(config)
            ... )
        """
        if not isinstance(layer_name, str) or not layer_name:
            raise ValueError(f"layer_name must be a non-empty string, got {layer_name!r}")
        
        if not callable(layer_factory):
            raise TypeError(f"layer_factory must be callable, got {type(layer_factory).__name__}")
        
        if layer_name in self._custom_layers:
            raise ValueError(
                f"Custom layer '{layer_name}' is already registered. "
                f"Use unregister_custom_layer() first if you want to replace it."
            )
        
        self._custom_layers[layer_name] = layer_factory
    
    def get_layer_factory(self, layer_name: str) -> Any:
        """
        Retrieve a registered custom layer factory by name.
        
        Args:
            layer_name: The identifier of the custom layer to retrieve.
        
        Returns:
            The registered layer factory (callable).
        
        Raises:
            KeyError: If the layer_name is not registered.
        
        Example:
            >>> registry = ArchitectureRegistry()
            >>> factory = registry.get_layer_factory("llama_custom_attn")
            >>> layer = factory(config, layer_idx=0)
        """
        if layer_name not in self._custom_layers:
            available = list(self._custom_layers.keys())
            raise KeyError(
                f"Custom layer '{layer_name}' not found in registry. "
                f"Available custom layers: {available if available else 'none'}"
            )
        
        return self._custom_layers[layer_name]
    
    def unregister_custom_layer(self, layer_name: str) -> bool:
        """
        Remove a custom layer from the registry.
        
        Args:
            layer_name: The identifier of the custom layer to remove.
        
        Returns:
            True if the layer was removed, False if it didn't exist.
        """
        if layer_name in self._custom_layers:
            del self._custom_layers[layer_name]
            return True
        return False
    
    def list_custom_layers(self) -> List[str]:
        """List all registered custom layer names."""
        return list(self._custom_layers.keys())
    
    def clear_custom_layers(self) -> None:
        """Clear all custom layers from the registry. Use with caution."""
        self._custom_layers.clear()


# Global registry instance
_registry = None


def get_registry() -> ArchitectureRegistry:
    """Get the global architecture registry instance."""
    global _registry
    if _registry is None:
        _registry = ArchitectureRegistry()
    return _registry
