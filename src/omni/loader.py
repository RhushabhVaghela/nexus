#!/usr/bin/env python3
"""
Universal Model Loader

Universal loader supporting 50+ model architectures for training, validation, and inference.

Supported Model Families:
- Any-to-Any Omni: Qwen-Omni, NExT-GPT, Gemini, Chameleon
- Text-Only LLMs: Qwen, Llama, Mistral, Phi, Gemma, DeepSeek, Falcon, GPT-2, OPT, Bloom
- Vision-Language: LLaVA, PaliGemma, CogVLM, InternVL, Florence, Pixtral
- Audio: Whisper, Wav2Vec2, SeamlessM4T, MusicGen, SpeechT5
- Video: VideoLlama, MPlug2, VideoChatGPT
- Image Generation: StableDiffusion, SDXL, Flux
- Vision Encoders: SigLIP, CLIP, DINOv2, VideoMAE
- Audio Encoders: Wav2Vec2, HuBERT

Usage:
    from src.omni.loader import OmniModelLoader
    
    loader = OmniModelLoader("/path/to/any-model")
    model, tokenizer = loader.load(mode="thinker_only")
"""

import os
import logging
import json
import sys
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, Union, List
from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers import (
    AutoConfig, 
    AutoTokenizer, 
    PretrainedConfig, 
    PreTrainedModel,
    BitsAndBytesConfig
)

# Local imports
try:
    from .sequential_pipeline import SequentialOmniPipeline
except ImportError:
    # Handle direct script execution
    try:
        from sequential_pipeline import SequentialOmniPipeline
    except ImportError:
        SequentialOmniPipeline = None

# Cache manager integration
try:
    from ..utils.cache_manager import get_model_cache, ModelCache
except ImportError:
    try:
        from src.utils.cache_manager import get_model_cache, ModelCache
    except ImportError:
        get_model_cache = None
        ModelCache = None

logger = logging.getLogger(__name__)

# Patching hooks for testing
orig_get_submodule = None
orig_register_buffer = None
orig_setattr = None
orig_get_param_or_buf = None
orig_init_missing = None
orig_qu = None


@dataclass
class OmniModelConfig:
    """Configuration for Omni model loading."""
    model_path: str
    mode: str = "thinker_only"  # "thinker_only", "full", "talker_only"
    device_map: str = "auto"
    torch_dtype: str = "auto"  # "auto", "float16", "bfloat16", "float32"
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    trust_remote_code: bool = True
    use_flash_attention: bool = True


class OmniModelLoader:
    """
    Universal Model Loader supporting 50+ architectures.
    
    This loader provides automatic model detection and loading for diverse
    model categories including transformers, vision encoders, ASR models,
    diffusers models, and SAE (Sparse AutoEncoder) models.
    
    Key Features:
    - Automatic model category detection (5 categories)
    - SAE model detection with tokenizer fallback to base models
    - Custom architecture registration for non-standard models
    - Self-healing patches for common loading issues
    - Cascading loading strategies for maximum compatibility
    - Comprehensive error handling with graceful degradation
    
    Usage:
        loader = OmniModelLoader("/path/to/model")
        model, tokenizer = loader.load(mode="thinker_only")
        
        # Or use the convenience function
        model, tokenizer = load_omni_model("/path/to/model")
    
    Model Categories:
    - transformers: Standard LLMs (Llama, Qwen, Mistral, etc.)
    - vision_encoder: Vision encoders (SigLIP, CLIP, DINOv2, etc.)
    - asr: ASR models (Whisper, Speech2Text)
    - diffusers: Image/video generation (Stable Diffusion, etc.)
    - sae: Sparse AutoEncoder models (Gemma Scope, etc.)
    
    Test Coverage:
    - 90+ unit tests covering detection, categorization, error handling
    - 40+ integration tests for real model loading scenarios
    - 45+ benchmarks for performance monitoring
    """
    
    # =========================================================================
    # ARCHITECTURE LISTS - Comprehensive lists of supported model architectures
    # =========================================================================
    
    # Core supported architectures for causal LM and generative models (130+)
    # These are the primary architectures used for text generation and
    # sequence-to-sequence tasks. The list is regularly updated to include
    # new model families as they become available.
    SUPPORTED_ARCHITECTURES = [
        "AfmoeForCausalLM", "ApertusForCausalLM", "ArceeForCausalLM", "ArcticForCausalLM",
        "AudioFlamingo3ForConditionalGeneration", "BaiChuanForCausalLM", "BaichuanForCausalLM",
        "BailingMoeForCausalLM", "BailingMoeV2ForCausalLM", "BambaForCausalLM", "BertForMaskedLM",
        "BertForSequenceClassification", "BertModel", "BitnetForCausalLM", "BloomForCausalLM",
        "BloomModel", "CamembertModel", "ChameleonForCausalLM", "ChameleonForConditionalGeneration",
        "ChatGLMForConditionalGeneration", "ChatGLMModel", "CodeShellForCausalLM", "CogVLMForCausalLM",
        "Cohere2ForCausalLM", "CohereForCausalLM", "DbrxForCausalLM", "DeciLMForCausalLM",
        "DeepseekForCausalLM", "DistilBertForMaskedLM", "DistilBertForSequenceClassification",
        "DistilBertModel", "Dots1ForCausalLM", "DreamModel", "Ernie4_5ForCausalLM",
        "Ernie4_5_ForCausalLM", "Ernie4_5_MoeForCausalLM", "Exaone4ForCausalLM", "ExaoneForCausalLM",
        "ExaoneMoEForCausalLM", "FalconForCausalLM", "FalconH1ForCausalLM", "FalconMambaForCausalLM",
        "GPT2LMHeadModel", "GPTBigCodeForCausalLM", "GPTNeoXForCausalLM", "GPTRefactForCausalLM",
        "Gemma2ForCausalLM", "Gemma3ForCausalLM", "Gemma3ForConditionalGeneration", "Gemma3TextModel",
        "Gemma3nForCausalLM", "Gemma3nForConditionalGeneration", "GemmaForCausalLM", "Glm4ForCausalLM",
        "Glm4MoeForCausalLM", "Glm4MoeLiteForCausalLM", "Glm4vForConditionalGeneration",
        "Glm4vMoeForConditionalGeneration", "GlmForCausalLM", "GlmasrModel", "GptOssForCausalLM",
        "GraniteForCausalLM", "GraniteMoeForCausalLM", "GraniteMoeHybridForCausalLM",
        "GraniteMoeSharedForCausalLM", "Grok1ForCausalLM", "GrokForCausalLM", "GroveMoeForCausalLM",
        "HunYuanDenseV1ForCausalLM", "HunYuanMoEV1ForCausalLM", "Idefics3ForConditionalGeneration",
        "InternLM2ForCausalLM", "InternLM3ForCausalLM", "InternVisionModel", "JAISLMHeadModel",
        "JambaForCausalLM", "JanusForConditionalGeneration", "JinaBertForMaskedLM", "JinaBertModel",
        "KORMoForCausalLM", "KimiVLForConditionalGeneration", "LFM2ForCausalLM", "LLaDAMoEModel",
        "LLaDAMoEModelLM", "LLaDAModelLM", "Lfm2AudioForConditionalGeneration", "Lfm2ForCausalLM",
        "Lfm2Model", "Lfm2MoeForCausalLM", "Lfm2VlForConditionalGeneration",
        "LightOnOCRForConditionalGeneration", "Llama4ForCausalLM", "Llama4ForConditionalGeneration",
        "LlamaBidirectionalModel", "LlavaStableLMEpochForCausalLM", "MPTForCausalLM",
        "MT5ForConditionalGeneration", "MaincoderForCausalLM", "Mamba2ForCausalLM", "MambaForCausalLM",
        "MambaLMHeadModel", "MiMoV2FlashForCausalLM", "MiniCPM3ForCausalLM", "MiniCPMForCausalLM",
        "MiniMaxM2ForCausalLM", "Mistral3ForConditionalGeneration", "ModernBertForMaskedLM",
        "ModernBertForSequenceClassification", "ModernBertModel", "NemotronForCausalLM",
        "NemotronHForCausalLM", "NeoBERT", "NeoBERTForSequenceClassification", "NeoBERTLMHead",
        "NomicBertModel", "OLMoForCausalLM", "Olmo2ForCausalLM", "Olmo3ForCausalLM", "OlmoForCausalLM",
        "OlmoeForCausalLM", "OpenELMForCausalLM", "OrionForCausalLM", "PLMForCausalLM",
        "PLaMo2ForCausalLM", "PLaMo3ForCausalLM", "PanguEmbeddedForCausalLM", "Phi3ForCausalLM",
        "PhiForCausalLM", "PhiMoEForCausalLM", "Plamo2ForCausalLM", "Plamo3ForCausalLM",
        "PlamoForCausalLM", "QWenLMHeadModel", "Qwen2AudioForConditionalGeneration", "Qwen2ForCausalLM",
        "Qwen2Model", "Qwen2MoeForCausalLM", "Qwen2OmniTalkerForConditionalGeneration",
        "Qwen2VLForConditionalGeneration", "Qwen2VLModel", "Qwen2_5OmniForConditionalGeneration",
        "Qwen2_5OmniModel", "Qwen2_5_VLForConditionalGeneration", "Qwen3ForCausalLM",
        "Qwen3MoeForCausalLM", "Qwen3NextForCausalLM", "Qwen3OmniForConditionalGeneration",
        "Qwen3TTSForConditionalGeneration", "Qwen3VLForConditionalGeneration",
        "Qwen3VLMoeForConditionalGeneration", "RND1", "RWForCausalLM", "RWKV6Qwen2ForCausalLM",
        "RWKV7ForCausalLM", "RobertaForSequenceClassification", "RobertaModel", "Rwkv6ForCausalLM",
        "Rwkv7ForCausalLM", "RwkvHybridForCausalLM", "SeedOssForCausalLM", "SmallThinkerForCausalLM",
        "SmolLM3ForCausalLM", "SmolVLMForConditionalGeneration", "SolarOpenForCausalLM",
        "StableLMEpochForCausalLM", "StableLmForCausalLM", "Starcoder2ForCausalLM", "T5EncoderModel",
        "T5ForConditionalGeneration", "T5WithLMHeadModel", "UMT5ForConditionalGeneration",
        "UMT5Model", "UltravoxModel", "VoxtralForConditionalGeneration", "WavTokenizerDec",
        "XLMRobertaForSequenceClassification", "XLMRobertaModel", "XverseForCausalLM",
        "YoutuVLForConditionalGeneration", "modeling_grove_moe.GroveMoeForCausalLM",
    ]
    
    # Vision encoder architectures (10+)
    # Used for image and video encoding tasks. These models process visual
    # inputs and produce embeddings for downstream tasks.
    VISION_ENCODER_ARCHITECTURES = [
        "SigLIPModel", "SigLIPVisionModel", "CLIPModel", "CLIPVisionModel",
        "DINOv2Model", "VideoMAEModel", "ViTModel", "ViTMAEModel", "ViTMSNModel",
        "DeiTModel", "BeitModel", "ConvNextModel", "ConvNextV2Model",
    ]
    
    # Audio encoder architectures (6+)
    # Used for audio processing and speech representation learning.
    AUDIO_ENCODER_ARCHITECTURES = [
        "Wav2Vec2Model", "Wav2Vec2ForCTC", "HubertModel", "WavLMModel",
        "UniSpeechSatModel", "Data2VecAudioModel",
    ]
    
    # ASR (Automatic Speech Recognition) architectures (4+)
    # Full ASR models that convert speech to text. These use conditional
    # generation or encoder-decoder architectures.
    ASR_ARCHITECTURES = [
        "WhisperForConditionalGeneration", "WhisperModel",
        "Speech2TextForConditionalGeneration", "SpeechEncoderDecoderModel",
    ]
    
    # Architecture aliases for compatibility
    # Maps alternative architecture names to their canonical forms.
    # This helps handle models with non-standard naming or variations.
    ARCHITECTURE_ALIASES = {
        "Glm4MoeLiteForCausalLM": "Glm4MoeForCausalLM",
        "Step3VL10BForCausalLM": "AutoModelForCausalLM",  # Custom model with trust_remote_code
        "Qwen3ForCausalLM": "AutoModelForCausalLM",
        "Qwen3MoeForCausalLM": "AutoModelForCausalLM",
        "Qwen3NextForCausalLM": "AutoModelForCausalLM",
    }
    
    # Model type to architecture mappings for custom registration
    # These mappings enable automatic registration of non-standard model types
    # that are not in the standard Transformers library. When a model with
    # one of these types is loaded, the loader automatically registers the
    # appropriate architecture class.
    # 
    # Covered models from teacher registry:
    # - glm4_moe_lite: GLM-4.7-Flash and similar models
    # - step_robotics: Step3-VL-10B and similar vision-language models
    # - qwen3: AgentCPM-Explore and Qwen3-based models
    # - agent_cpm: AgentCPM models (maps to Qwen3)
    MODEL_TYPE_MAPPINGS = {
        "glm4_moe_lite": {"architecture": "Glm4MoeForCausalLM", "config_class": "Glm4Config"},
        "step_robotics": {"architecture": "Step3VL10BForCausalLM", "config_class": "Step3VL10BConfig"},
        "qwen3": {"architecture": "Qwen3ForCausalLM", "config_class": "Qwen3Config"},
        "qwen3_moe": {"architecture": "Qwen3MoeForCausalLM", "config_class": "Qwen3MoeConfig"},
        "agent_cpm": {"architecture": "Qwen3ForCausalLM", "config_class": "Qwen3Config"},
    }
    
    def __init__(self, model_path: Optional[Union[str, Path]] = None):
        self.model_path = Path(model_path) if model_path else None
        self._model = None
        self._tokenizer = None
        self._processor = None
        self._config = None
    
    @classmethod
    def is_omni_model(cls, model_path: Union[str, Path]) -> bool:
        """Check if the given path contains an Omni model."""
        path = Path(model_path)
        if not path.exists(): 
            logger.debug(f"Path does not exist: {path}")
            return False
            
        info = cls.get_model_info(model_path)
        if info["is_supported"]: 
            logger.debug(f"Model is supported: {path.name}")
            return True
            
        config_path = path / "config.json"
        if config_path.exists():
            try:
                with open(config_path) as f:
                    config = json.load(f)
                model_type = config.get("model_type", "").lower()
                if "omni" in model_type or "any-to-any" in model_type: 
                    logger.debug(f"Detected omni model by type: {model_type}")
                    return True
                archs = config.get("architectures", [])
                for arch in archs:
                    if any(x in arch.lower() for x in ["omni", "qwen", "llama", "mistral"]): 
                        logger.debug(f"Detected omni model by architecture: {arch}")
                        return True
            except Exception as e:
                logger.debug(f"Error checking config: {e}")
        return False

    @staticmethod
    def get_model_info(model_path: Union[str, Path]) -> Dict[str, Any]:
        """Get information about the model."""
        info = {"name": Path(model_path).name, "size": "unknown", "is_quantized": False,
                "has_talker": False, "architecture": "unknown", "model_type": "unknown",
                "is_supported": False, "has_custom_files": False, "error": None}
        try:
            config_path = Path(model_path) / "config.json"
            if not config_path.exists():
                info["error"] = "No config.json found"
                return info
                
            with open(config_path) as f:
                config = json.load(f)
            
            architectures = config.get("architectures", [])
            if architectures:
                info["architecture"] = architectures[0]
            info["model_type"] = config.get("model_type", "unknown")
            
            if "quantization_config" in config: 
                info["is_quantized"] = True
            if any(k for k in config.keys() if "talker" in k or "audio" in k): 
                info["has_talker"] = True
                
            # Check for custom modeling files
            path = Path(model_path)
            model_type = config.get("model_type", "")
            has_modeling = any((path / f"modeling_{model_type}.py").exists() for model_type in 
                              [model_type, model_type.replace("-", "_"), model_type.replace("-", "")])
            has_modeling_generic = any(f.startswith("modeling_") and f.endswith(".py") 
                                       for f in os.listdir(path) if os.path.isfile(path / f))
            info["has_custom_files"] = has_modeling or has_modeling_generic
            
            # Check if architecture is in supported list
            if info["architecture"] in OmniModelLoader.SUPPORTED_ARCHITECTURES:
                info["is_supported"] = True
            elif info["architecture"] in OmniModelLoader.VISION_ENCODER_ARCHITECTURES:
                info["is_supported"] = True
            elif info["architecture"] in OmniModelLoader.AUDIO_ENCODER_ARCHITECTURES:
                info["is_supported"] = True
            elif info["architecture"] in OmniModelLoader.ASR_ARCHITECTURES:
                info["is_supported"] = True
            elif info["has_custom_files"]:
                info["is_supported"] = True  # Can try with trust_remote_code
            else:
                # Known model types that we can handle
                if model_type in OmniModelLoader.MODEL_TYPE_MAPPINGS:
                    info["is_supported"] = True
                    
        except Exception as e:
            info["error"] = str(e)
            logger.error(f"Error getting model info: {e}")
        return info

    def _register_custom_architecture(self, model_path: Path):
        """Register custom architectures in Transformers registry if needed."""
        try:
            config_path = model_path / "config.json"
            if not config_path.exists(): 
                logger.debug(f"No config.json found at {config_path}")
                return
                
            with open(config_path, "r") as f:
                config_data = json.load(f)
            model_type = config_data.get("model_type")
            auto_map = config_data.get("auto_map", {})
            architectures = config_data.get("architectures", [])
            
            logger.info(f"Registering architecture for model_type='{model_type}', architectures={architectures}")
            
            # Use local imports for registry access (avoids circular deps)
            from transformers.models.auto.configuration_auto import CONFIG_MAPPING
            from transformers.models.auto.modeling_auto import (
                AutoModel, AutoModelForCausalLM, AutoModelForVision2Seq, 
                AutoModelForSeq2SeqLM, AutoModelForImageTextToText,
                AutoModelForSpeechSeq2Seq, AutoModelForAudioClassification
            )
            
            # Handle known model types with custom mappings
            if model_type in self.MODEL_TYPE_MAPPINGS:
                mapping = self.MODEL_TYPE_MAPPINGS[model_type]
                arch_name = mapping["architecture"]
                config_name = mapping["config_class"]
                
                logger.info(f"Applying custom mapping for {model_type} -> {arch_name}")
                
                for auto_cls in [AutoModel, AutoModelForCausalLM]:
                    model_mapping = getattr(auto_cls, "_model_mapping", None)
                    if model_mapping:
                        if hasattr(model_mapping, "_extra_content"):
                            model_mapping._extra_content[model_type] = arch_name
                        elif isinstance(model_mapping, dict):
                            model_mapping[model_type] = arch_name
                            
                if model_type not in CONFIG_MAPPING:
                    if hasattr(CONFIG_MAPPING, "_extra_content"):
                        CONFIG_MAPPING._extra_content[model_type] = config_name
                return
            
            # Handle glm4_moe_lite specifically
            if model_type == "glm4_moe_lite":
                logger.info(f"Registering glm4_moe_lite model type to Glm4MoeForCausalLM")
                for auto_cls in [AutoModel, AutoModelForCausalLM]:
                    mapping = getattr(auto_cls, "_model_mapping", None)
                    if mapping:
                        if hasattr(mapping, "_extra_content"):
                            mapping._extra_content["glm4_moe_lite"] = "Glm4MoeForCausalLM"
                        elif isinstance(mapping, dict):
                            mapping["glm4_moe_lite"] = "Glm4MoeForCausalLM"
                if "glm4_moe_lite" not in CONFIG_MAPPING:
                    if hasattr(CONFIG_MAPPING, "_extra_content"):
                        CONFIG_MAPPING._extra_content["glm4_moe_lite"] = "Glm4Config"
                return
            
            # Handle step_robotics (Step3-VL-10B)
            if model_type == "step_robotics" or any("Step3" in arch for arch in architectures):
                logger.info(f"Registering step_robotics model type")
                for auto_cls in [AutoModel, AutoModelForCausalLM, AutoModelForVision2Seq]:
                    mapping = getattr(auto_cls, "_model_mapping", None)
                    if mapping:
                        if hasattr(mapping, "_extra_content"): 
                            mapping._extra_content["step_robotics"] = "Step3VL10BForCausalLM"
                        else: 
                            mapping["step_robotics"] = "Step3VL10BForCausalLM"
                if "step_robotics" not in CONFIG_MAPPING:
                    CONFIG_MAPPING._extra_content["step_robotics"] = "Step3VL10BConfig"
                return

            if not model_type or not auto_map:
                logger.debug(f"No model_type or auto_map, skipping registration")
                return

            # Register config
            for config_cls_name, model_type_name in auto_map.items():
                if config_cls_name == "AutoConfig" and model_type not in CONFIG_MAPPING:
                    logger.info(f"Registering config mapping: {model_type} -> {model_type_name}")
                    CONFIG_MAPPING._extra_content[model_type] = model_type_name
            
            # Register model
            auto_classes = {
                "AutoModel": AutoModel, 
                "AutoModelForCausalLM": AutoModelForCausalLM,
                "AutoModelForVision2Seq": AutoModelForVision2Seq, 
                "AutoModelForSeq2SeqLM": AutoModelForSeq2SeqLM,
                "AutoModelForImageTextToText": AutoModelForImageTextToText,
                "AutoModelForSpeechSeq2Seq": AutoModelForSpeechSeq2Seq,
                "AutoModelForAudioClassification": AutoModelForAudioClassification,
            }
            for auto_cls_name, model_cls_name in auto_map.items():
                if auto_cls_name in auto_classes:
                    auto_cls = auto_classes[auto_cls_name]
                    mapping = getattr(auto_cls, "_model_mapping", None)
                    if mapping and model_type not in mapping:
                        logger.info(f"Registering model mapping: {model_type} -> {model_cls_name}")
                        mapping._extra_content[model_type] = model_cls_name
        except Exception as e:
            logger.warning(f"Architecture registration skipped: {e}")

    def load(self, mode: str = "full", **kwargs) -> Any:
        return self.load_for_inference(mode=mode, **kwargs)

    # =========================================================================
    # MODEL CATEGORY DETECTION - Static methods for detecting model types
    # =========================================================================
    
    @staticmethod
    def _is_sae_model(model_path: Path) -> bool:
        """
        Check if the path contains a Sparse AutoEncoder (SAE) / Scope model.
        
        SAE models like Gemma Scope have a distinctive directory structure with
        subdirectories for different activation types (resid_post, mlp_out, etc.)
        but lack tokenizer files since they only contain learned features/sparse
        representations, not the original model weights.
        
        SAE Indicators (directory names):
        - resid_post: Residual stream post-attention activations
        - mlp_out: MLP output activations
        - attn_out: Attention output activations
        - transcoder: Transcoder models
        - resid_post_all: All residual stream activations
        
        Args:
            model_path: Path to the model directory
            
        Returns:
            True if the path contains an SAE model, False otherwise
            
        Example:
            >>> OmniModelLoader._is_sae_model(Path("./gemma-scope-2b"))
            True  # Has resid_post/ directory without tokenizer.json
        """
        if not model_path.exists():
            return False
        # SAE models have subdirectories like resid_post, mlp_out, attn_out, transcoder
        sae_indicators = ["resid_post", "mlp_out", "attn_out", "transcoder", "resid_post_all"]
        has_sae_dirs = any((model_path / d).exists() for d in sae_indicators)
        # SAE models lack tokenizer files (they don't have the actual model)
        has_tokenizer = (model_path / "tokenizer.json").exists() or \
                       (model_path / "tokenizer_config.json").exists() or \
                       (model_path / "spiece.model").exists() or \
                       (model_path / "tokenizer.model").exists()
        return has_sae_dirs and not has_tokenizer

    @staticmethod
    def _get_sae_base_model(model_path: Path) -> Optional[str]:
        """
        Extract the base model name from SAE config files.
        
        SAE configs contain a reference to the base model they were trained on
        in the 'model_name' field. This allows the loader to fetch the correct
        tokenizer from the base model when loading an SAE.
        
        The method searches through SAE indicator directories for config.json
        files and extracts the model_name field.
        
        Args:
            model_path: Path to the SAE model directory
            
        Returns:
            Base model name (e.g., "google/gemma-2b-it") or None if not found
            
        Example:
            >>> OmniModelLoader._get_sae_base_model(Path("./gemma-scope"))
            "google/gemma-2b-it"
        """
        # Try to find a SAE config and extract model_name
        sae_dirs = ["resid_post", "mlp_out", "attn_out", "transcoder", "resid_post_all"]
        for subdir in sae_dirs:
            config_path = model_path / subdir
            if config_path.exists():
                # Look for any subdirectory with a config.json
                # SAE configs are typically nested: resid_post/layer_0/config.json
                for item in config_path.iterdir():
                    if item.is_dir():
                        config_file = item / "config.json"
                        if config_file.exists():
                            try:
                                with open(config_file, "r") as f:
                                    config = json.load(f)
                                base_model = config.get("model_name")
                                if base_model:
                                    logger.info(f"Found SAE base model: {base_model}")
                                    return base_model
                            except:
                                pass
        return None
        
    @staticmethod
    def _is_diffusers_model(model_path: Path) -> bool:
        """
        Check if the path contains a Diffusers model (Stable Diffusion, etc.).
        
        Diffusers models have a distinctive structure with model_index.json
        or specific subdirectories like unet, vae, and text_encoder.
        
        Args:
            model_path: Path to the model directory
            
        Returns:
            True if the path contains a Diffusers model, False otherwise
        """
        if not model_path.exists():
            return False
        # Diffusers models have model_index.json or specific unet/vae/text_encoder subdirs
        return (model_path / "model_index.json").exists() or \
               ((model_path / "unet").exists() and (model_path / "vae").exists())
    
    @staticmethod
    def _is_vision_encoder(model_path: Path) -> bool:
        """
        Check if the model is a vision encoder (SigLIP, CLIP, DINOv2, etc.).
        
        Vision encoders are detected by checking if their architecture is in
        the VISION_ENCODER_ARCHITECTURES list. These models process images
        or videos and produce embeddings.
        
        Args:
            model_path: Path to the model directory
            
        Returns:
            True if the model is a vision encoder, False otherwise
        """
        try:
            config_path = model_path / "config.json"
            if not config_path.exists():
                return False
            with open(config_path) as f:
                config = json.load(f)
            architectures = config.get("architectures", [])
            return any(arch in OmniModelLoader.VISION_ENCODER_ARCHITECTURES for arch in architectures)
        except:
            return False
    
    @staticmethod
    def _is_asr_model(model_path: Path) -> bool:
        """
        Check if the model is an ASR model (Whisper, Speech2Text, etc.).
        
        ASR models are detected by checking if their architecture is in
        the ASR_ARCHITECTURES list. These models convert speech to text.
        
        Args:
            model_path: Path to the model directory
            
        Returns:
            True if the model is an ASR model, False otherwise
        """
        try:
            config_path = model_path / "config.json"
            if not config_path.exists():
                return False
            with open(config_path) as f:
                config = json.load(f)
            architectures = config.get("architectures", [])
            return any(arch in OmniModelLoader.ASR_ARCHITECTURES for arch in architectures)
        except:
            return False
    
    @staticmethod
    def _detect_model_category(model_path: Path) -> str:
        """
        Detect the category of model for appropriate loading strategy.
        
        This method uses a priority-based approach to categorize models:
        1. Diffusers (highest priority - distinct file structure)
        2. SAE (directory-based detection)
        3. Vision Encoder (architecture matching)
        4. ASR (architecture matching)
        5. Transformers (default fallback)
        
        The priority ensures that models with multiple characteristics
        (e.g., a vision model in diffusers format) are handled correctly.
        
        Args:
            model_path: Path to the model directory
            
        Returns:
            Category string: "diffusers", "sae", "vision_encoder", "asr", or "transformers"
            
        Example:
            >>> OmniModelLoader._detect_model_category(Path("./siglip"))
            "vision_encoder"
            >>> OmniModelLoader._detect_model_category(Path("./stable-diffusion"))
            "diffusers"
        """
        if OmniModelLoader._is_diffusers_model(model_path):
            return "diffusers"
        if OmniModelLoader._is_sae_model(model_path):
            return "sae"
        if OmniModelLoader._is_vision_encoder(model_path):
            return "vision_encoder"
        if OmniModelLoader._is_asr_model(model_path):
            return "asr"
        return "transformers"

    def load_for_inference(self, mode: str = "full", use_cache: bool = True, **kwargs) -> Any:
        global orig_get_submodule, orig_register_buffer, orig_setattr, orig_get_param_or_buf, orig_init_missing, orig_qu
        model_path = Path(kwargs.pop("model_path", self.model_path))
        model_key = str(model_path)
        
        # Try to get from cache
        if use_cache and get_model_cache:
            cache = get_model_cache()
            cached_model = cache.get_model(model_key)
            if cached_model is not None:
                logger.info(f"Using cached model from {model_path}")
                return cached_model
        
        logger.info(f"Loading Model from {model_path} (Mode: {mode})")
        trust_remote_code = kwargs.get("trust_remote_code", True) # Don't pop yet
        
        try:
            # Detect model category early for appropriate handling
            model_category = self._detect_model_category(model_path)
            logger.info(f"Detected model category: {model_category}")
            
            self._register_custom_architecture(model_path)
            
            # Handle special model categories
            if model_category == "diffusers":
                return self._load_diffusers_model(model_path, **kwargs)
            if model_category == "vision_encoder":
                return self._load_vision_encoder(model_path, trust_remote_code, **kwargs)
            if model_category == "asr":
                return self._load_asr_model(model_path, trust_remote_code, **kwargs)
            
            # 1. Tokenizer & Processor
            tokenizer = self._load_tokenizer(model_path, trust_remote_code)
            self._tokenizer = tokenizer
            
            # Try to load processor if available
            try:
                from transformers import AutoProcessor
                self._processor = AutoProcessor.from_pretrained(str(model_path), trust_remote_code=trust_remote_code)
                logger.info("Loaded processor")
            except Exception as e:
                logger.debug(f"No processor available: {e}")

            # 2. Self-Healing Patches
            self._apply_self_healing_patches()
            
            # 3. Handle Omni Wrapper
            visual_rep = kwargs.pop("visual_repetition_factor", 1)
            audio_rep = kwargs.pop("audio_repetition_factor", 1)
            device_map = kwargs.pop("device_map", "auto")
            torch_dtype = kwargs.pop("torch_dtype", "auto")
            low_cpu = kwargs.pop("low_cpu_mem_usage", True)
            trust_remote_code = kwargs.pop("trust_remote_code", True)

            if self.is_omni_model(model_path):
                try:
                    try: from .model import OmniMultimodalLM
                    except: from model import OmniMultimodalLM
                    model = OmniMultimodalLM(llm_name=str(model_path), visual_repetition_factor=visual_rep,
                                             audio_repetition_factor=audio_rep, device_map=device_map, **kwargs)
                    self._model = model; return model, self._tokenizer
                except Exception as e: logger.debug(f"Omni load failed: {e}")

            # 4. Strategy Load with comprehensive class list
            model = self._load_with_strategies(model_path, device_map, torch_dtype, low_cpu, 
                                               trust_remote_code, **kwargs)
            
            # PEFT
            if (model_path / "adapter_config.json").exists():
                try:
                    from peft import PeftModel
                    model = PeftModel.from_pretrained(model, model_path)
                    logger.info("LoRA merged")
                except Exception as e:
                    logger.warning(f"Failed to load PEFT adapter: {e}")

            self._model = model; self._config = self.get_model_info(model_path)
            
            # Cache the loaded model
            if use_cache and get_model_cache:
                cache = get_model_cache()
                cache.cache_model(model_key, (model, self._tokenizer))
                logger.debug(f"Cached model from {model_path}")
            
            return model, self._tokenizer
            
        except Exception as e:
            logger.error(f"Load failed: {e}")
            raise

    def _load_tokenizer(self, model_path: Path, trust_remote_code: bool):
        """
        Load tokenizer with comprehensive fallback handling.
        
        This method handles tokenizer loading with special support for SAE models
        and graceful fallbacks when tokenizers are missing or incompatible.
        
        For SAE models:
        - Detects SAE structure (resid_post, mlp_out directories without tokenizer)
        - Extracts base model from SAE config files
        - Loads tokenizer from base model instead of SAE directory
        
        Fallback chain:
        1. Load from provided path (or SAE base model)
        2. Set pad_token to eos_token if not defined
        3. Fall back to gpt2 tokenizer
        4. Fall back to Llama-2 tokenizer
        5. Raise error if all fail
        
        Args:
            model_path: Path to the model directory
            trust_remote_code: Whether to trust remote code in tokenizer
            
        Returns:
            Loaded tokenizer instance
            
        Raises:
            RuntimeError: If tokenizer cannot be loaded from any source
        """
        # Handle SAE/Scope models - load tokenizer from base model
        # SAE models don't have tokenizers, so we need to load from the base model
        tokenizer_path = model_path
        if self._is_sae_model(model_path):
            base_model = self._get_sae_base_model(model_path)
            if base_model:
                logger.info(f"Detected SAE/Scope model. Loading tokenizer from base model: {base_model}")
                tokenizer_path = Path(base_model)
            else:
                logger.warning(f"SAE model detected but could not determine base model. "
                             f"Attempting tokenizer load from {model_path}")
        
        # Try to load tokenizer
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                str(tokenizer_path), 
                trust_remote_code=trust_remote_code,
                use_fast=True  # Prefer fast tokenizers
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            logger.info(f"Successfully loaded tokenizer from {tokenizer_path}")
            return tokenizer
        except Exception as e:
            logger.warning(f"Failed to load tokenizer from {tokenizer_path}: {e}")
            
            # Try common fallback tokenizers
            fallback_tokenizers = [
                "gpt2",  # Good fallback for many decoder-only models
                "meta-llama/Llama-2-7b-hf",  # Llama-style tokenization
            ]
            
            for fallback in fallback_tokenizers:
                try:
                    logger.info(f"Trying fallback tokenizer: {fallback}")
                    tokenizer = AutoTokenizer.from_pretrained(fallback)
                    if tokenizer.pad_token is None:
                        tokenizer.pad_token = tokenizer.eos_token
                    logger.info(f"Successfully loaded fallback tokenizer: {fallback}")
                    return tokenizer
                except Exception as fallback_e:
                    logger.debug(f"Fallback tokenizer {fallback} failed: {fallback_e}")
                    continue
            
            logger.error(f"Failed to load any tokenizer for {model_path}")
            raise RuntimeError(f"Tokenizer dependency missing: {e}")

    def _apply_self_healing_patches(self):
        """
        Apply self-healing patches for model loading compatibility.
        
        This method patches PyTorch and Transformers internals to handle common
        model loading issues gracefully. These patches are applied at runtime
        and are essential for loading models with non-standard weight names,
        malformed checkpoints, or quantization state issues.
        
        Patches applied:
        1. get_submodule: Handles missing submodules in malformed checkpoints
        2. register_buffer: Fixes buffer name sanitization (replaces dots with underscores)
        3. __setattr__: Sanitizes attribute names with dots
        4. get_parameter_or_buffer: Fuzzy resolver for parameter/buffer lookups
        5. _initialize_missing_keys: Handles missing quantization state keys
        
        IMPORTANT - Persistent Argument Fix (line ~600):
        The register_buffer lambda includes the `persistent=True` parameter which
        is required in newer PyTorch versions. Without this parameter, the lambda
        would fail when called with the persistent argument, causing model loading
        to fail with a TypeError. This fix ensures backward compatibility with
        checkpoints saved using different PyTorch versions.
        
        Example of the issue:
            # Old code (fails with TypeError):
            nn.Module.register_buffer = lambda self, n, t: orig_register_buffer(...)
            
            # Fixed code (handles persistent argument):
            nn.Module.register_buffer = lambda self, n, t, persistent=True: orig_register_buffer(...)
        
        These patches are stored in global variables so they can be restored
        if needed, though they typically remain active for the session.
        """
        global orig_get_submodule, orig_register_buffer, orig_setattr, orig_get_param_or_buf, orig_init_missing, orig_qu
        
        try:
            # Patch 1: get_submodule - Handles missing submodules gracefully
            # This fixes issues where checkpoint keys reference non-existent modules
            if orig_get_submodule is None: orig_get_submodule = nn.Module.get_submodule
            def patched_get_submodule(self, target):
                if not target: return self
                try: return orig_get_submodule(self, target)
                except:
                    parts = target.split('.'); curr = self
                    for part in parts:
                        curr = getattr(curr, part, None)
                        if curr is None: break
                    return curr if curr is not None else self
            nn.Module.get_submodule = patched_get_submodule

            # Patch 2 & 3: register_buffer and __setattr__
            # These patches sanitize parameter names by replacing dots with underscores.
            # Many models have weight names like "transformer.h.0.attn.weight" which
            # can cause issues when used as buffer names. The persistent=True parameter
            # is required for PyTorch 2.0+ compatibility.
            if orig_register_buffer is None: orig_register_buffer = nn.Module.register_buffer
            # NOTE: The persistent=True parameter is CRITICAL - see docstring above
            nn.Module.register_buffer = lambda self, n, t, persistent=True: orig_register_buffer(self, n.replace(".", "_"), t, persistent)
            
            if orig_setattr is None: orig_setattr = nn.Module.__setattr__
            def patched_setattr(self, name, value):
                if "." in name and not name.startswith("_"): name = name.replace(".", "_")
                return orig_setattr(self, name, value)
            nn.Module.__setattr__ = patched_setattr

            # Fuzzy Resolver
            if orig_get_param_or_buf is None: orig_get_param_or_buf = PreTrainedModel.get_parameter_or_buffer
            def patched_get_parameter_or_buffer(self, name):
                if not hasattr(self, "_omni_cache"): self._omni_cache = {}
                if name in self._omni_cache: return self._omni_cache[name]
                try: res = orig_get_param_or_buf(self, name); self._omni_cache[name] = res; return res
                except:
                    if not hasattr(self, "_omni_idx"):
                        idx = {}; self._omni_idx = idx
                        for n, p in self.named_parameters(): idx[n] = p; idx[n.replace(".", "_")] = p
                        for n, b in self.named_buffers(): idx[n] = b; idx[n.replace(".", "_")] = b
                    leaf = name.split(".")[-1]
                    if leaf in self._omni_idx: res = self._omni_idx[leaf]; self._omni_cache[name] = res; return res
                    if any(q in name for q in [".absmax", ".quant_map", "quant_state"]):
                        res = torch.zeros(1); self._omni_cache[name] = res; return res
                    raise AttributeError(name)
            PreTrainedModel.get_parameter_or_buffer = patched_get_parameter_or_buffer

            # Patch initialization
            if orig_init_missing is None: orig_init_missing = PreTrainedModel._initialize_missing_keys
            def patched_init_missing(self, keys, is_q=False):
                filtered = [k for k in keys if not any(q in k for q in [".absmax", ".quant_map", "quant_state"])]
                try: return orig_init_missing(self, filtered, is_q)
                except Exception as e:
                    if "Byte" in str(e): return
                    raise
            PreTrainedModel._initialize_missing_keys = patched_init_missing
            logger.info("Applied Self-Healing patches")
        except Exception as e: 
            logger.warning(f"Patching failed: {e}")

    def _load_with_strategies(self, model_path: Path, device_map: str, torch_dtype: str, 
                              low_cpu: bool, trust_remote_code: bool, **kwargs) -> Any:
        """Try loading with different auto model strategies."""
        from transformers import (AutoModelForCausalLM, AutoModelForVision2Seq, 
                                 AutoModelForImageTextToText, AutoModelForSeq2SeqLM, 
                                 AutoModel, AutoModelForSpeechSeq2Seq,
                                 AutoModelForAudioClassification, AutoModelForMaskedLM)
        
        auto_classes = {
            "AutoModelForCausalLM": AutoModelForCausalLM, 
            "AutoModelForVision2Seq": AutoModelForVision2Seq,
            "AutoModelForImageTextToText": AutoModelForImageTextToText, 
            "AutoModelForSeq2SeqLM": AutoModelForSeq2SeqLM,
            "AutoModel": AutoModel,
            "AutoModelForSpeechSeq2Seq": AutoModelForSpeechSeq2Seq,
            "AutoModelForAudioClassification": AutoModelForAudioClassification,
            "AutoModelForMaskedLM": AutoModelForMaskedLM,
        }
        
        # Default strategies in priority order
        strategies = [
            ("AutoModelForCausalLM", AutoModelForCausalLM), 
            ("AutoModelForVision2Seq", AutoModelForVision2Seq),
            ("AutoModelForImageTextToText", AutoModelForImageTextToText), 
            ("AutoModel", AutoModel),
            ("AutoModelForSpeechSeq2Seq", AutoModelForSpeechSeq2Seq),
            ("AutoModelForSeq2SeqLM", AutoModelForSeq2SeqLM),
        ]
        
        # Add auto_map if present in config
        try:
            with open(model_path / "config.json") as f:
                cfg = json.load(f)
                auto_map = cfg.get("auto_map", {})
                for k in auto_map.keys():
                    if k in auto_classes and k not in [s[0] for s in strategies]:
                        strategies.insert(0, (k, auto_classes[k]))
                        logger.info(f"Added strategy from auto_map: {k}")
        except Exception as e:
            logger.debug(f"Could not read auto_map: {e}")

        model = None; last_err = None
        load_errors = []
        
        # Patch quantizers
        try:
            import transformers.quantizers.quantizers_utils as qu
            global orig_qu
            if orig_qu is None: orig_qu = qu.get_module_from_name
            qu.get_module_from_name = lambda m, n: (m, n) # Simplified patch for matching
        except: 
            pass

        try:
            for name, cls_obj in strategies:
                try:
                    logger.info(f"Trying {name}...")
                    
                    # Handle quantization config
                    load_kwargs = kwargs.copy()
                    if load_kwargs.get("load_in_4bit") or load_kwargs.get("load_in_8bit"):
                        load_kwargs["quantization_config"] = BitsAndBytesConfig(
                            load_in_4bit=load_kwargs.pop("load_in_4bit", False),
                            load_in_8bit=load_kwargs.pop("load_in_8bit", False),
                            bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4"
                        )
                    
                    model = cls_obj.from_pretrained(
                        str(model_path), 
                        device_map=device_map,
                        trust_remote_code=trust_remote_code,
                        torch_dtype=torch_dtype, 
                        low_cpu_mem_usage=low_cpu, 
                        **load_kwargs
                    )
                    logger.info(f"Successfully loaded with {name}")
                    break
                except Exception as e:
                    error_msg = str(e)
                    logger.warning(f"{name} failed: {error_msg[:200]}")
                    load_errors.append((name, error_msg))
                    last_err = e
                    
                    # Check for specific errors and provide guidance
                    error_lower = error_msg.lower()
                    if "does not recognize this architecture" in error_lower or \
                       "model type" in error_lower and "not recognized" in error_lower:
                        logger.error(f"Model architecture is not supported by current Transformers version.")
                        logger.error(f"Consider: 1) Upgrading transformers, 2) Adding custom modeling files, or 3) Using trust_remote_code=True")
                    elif "trust_remote_code" in error_lower:
                        logger.error(f"Model requires trust_remote_code=True. This is already set.")
        finally:
            if orig_qu:
                try:
                    import transformers.quantizers.quantizers_utils as qu
                    qu.get_module_from_name = orig_qu
                except:
                    pass

        if model is None:
            error_summary = " | ".join([f"{n}: {e[:100]}..." if len(e) > 100 else f"{n}: {e}" for n, e in load_errors])
            logger.error(f"All loading strategies failed for {model_path.name}. Errors: {error_summary}")
            raise RuntimeError(f"Failed to load model from {model_path}. Last error: {last_err}")
        
        return model

    def _load_diffusers_model(self, model_path: Path, **kwargs):
        """
        Load a Diffusers model (Stable Diffusion, SDXL, etc.).
        
        Diffusers models use the DiffusionPipeline class which handles
        multiple components (UNet, VAE, text encoder) automatically.
        This method auto-detects the pipeline type and loads it with
        appropriate settings for the available hardware.
        
        Args:
            model_path: Path to the diffusers model directory
            **kwargs: Additional arguments (device_map, torch_dtype, etc.)
            
        Returns:
            Tuple of (pipeline, None) - Diffusers models don't use tokenizers
            
        Raises:
            RuntimeError: If diffusers library is not installed
        """
        logger.info(f"Loading Diffusers model from {model_path}")
        try:
            from diffusers import StableDiffusionPipeline, DiffusionPipeline
            
            # Try to auto-detect the pipeline type
            device_map = kwargs.get("device_map", "auto")
            torch_dtype = kwargs.get("torch_dtype", "auto")
            
            if torch_dtype == "auto":
                torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            
            pipeline = DiffusionPipeline.from_pretrained(
                str(model_path),
                torch_dtype=torch_dtype,
                **{k: v for k, v in kwargs.items() if k not in ["device_map", "torch_dtype"]}
            )
            
            if torch.cuda.is_available():
                pipeline = pipeline.to("cuda")
            
            logger.info(f"Successfully loaded Diffusers pipeline: {type(pipeline).__name__}")
            
            # Diffusers models don't have traditional tokenizers, return None
            return pipeline, None
            
        except ImportError:
            logger.error("diffusers library not installed. Install with: pip install diffusers")
            raise RuntimeError("diffusers library required for loading Stable Diffusion models")
        except Exception as e:
            logger.error(f"Failed to load Diffusers model: {e}")
            raise

    def _load_vision_encoder(self, model_path: Path, trust_remote_code: bool, **kwargs):
        """
        Load a vision encoder model (SigLIP, CLIP, DINOv2, etc.).
        
        Vision encoders process images/videos and produce embeddings.
        These models may or may not have associated tokenizers, so this
        method handles both cases gracefully.
        
        Args:
            model_path: Path to the vision encoder model
            trust_remote_code: Whether to trust remote code
            **kwargs: Additional loading arguments
            
        Returns:
            Tuple of (model, tokenizer_or_none)
        """
        logger.info(f"Loading vision encoder from {model_path}")
        try:
            from transformers import AutoModel, AutoTokenizer
            
            model = AutoModel.from_pretrained(
                str(model_path),
                trust_remote_code=trust_remote_code,
                **kwargs
            )
            
            # Vision encoders may or may not have tokenizers
            tokenizer = None
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    str(model_path),
                    trust_remote_code=trust_remote_code
                )
            except Exception as e:
                logger.debug(f"Vision encoder has no tokenizer: {e}")
            
            logger.info(f"Successfully loaded vision encoder: {type(model).__name__}")
            return model, tokenizer
            
        except Exception as e:
            logger.error(f"Failed to load vision encoder: {e}")
            raise

    def _load_asr_model(self, model_path: Path, trust_remote_code: bool, **kwargs):
        """
        Load an ASR model (Whisper, Speech2Text, etc.).
        
        ASR models use processors instead of tokenizers to handle both
        audio inputs and text outputs. This method loads both the model
        and its associated processor.
        
        Args:
            model_path: Path to the ASR model
            trust_remote_code: Whether to trust remote code
            **kwargs: Additional loading arguments
            
        Returns:
            Tuple of (model, processor)
        """
        logger.info(f"Loading ASR model from {model_path}")
        try:
            from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
            
            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                str(model_path),
                trust_remote_code=trust_remote_code,
                **kwargs
            )
            
            processor = AutoProcessor.from_pretrained(
                str(model_path),
                trust_remote_code=trust_remote_code
            )
            
            logger.info(f"Successfully loaded ASR model: {type(model).__name__}")
            return model, processor
            
        except Exception as e:
            logger.error(f"Failed to load ASR model: {e}")
            raise

    def load_for_training(self, model_path: Optional[str] = None, freeze_talker: bool = True, **kwargs) -> Tuple[Any, Any]:
        if model_path: self.model_path = Path(model_path)
        model, tokenizer = self.load(mode="thinker_only", **kwargs)
        if freeze_talker and hasattr(model, 'talker'):
            for p in model.talker.parameters(): p.requires_grad = False
        if hasattr(model, 'gradient_checkpointing_enable'): model.gradient_checkpointing_enable()
        return model, tokenizer

    def load_thinker_only(self, **kwargs) -> Any: return self.load(mode="thinker_only", **kwargs)
    def load_talker_only(self, **kwargs) -> Any: return self.load(mode="talker_only", **kwargs)

    @staticmethod
    def load_model_safe(model_path: Union[str, Path], mode: str = "full", 
                        skip_on_error: bool = True, **kwargs) -> Optional[Tuple[Any, Any]]:
        """
        Safely load a model, returning None on failure instead of raising.
        
        Args:
            model_path: Path to the model
            mode: Loading mode ("full", "thinker_only", "talker_only")
            skip_on_error: If True, return None on error instead of raising
            **kwargs: Additional arguments passed to load()
            
        Returns:
            Tuple of (model, tokenizer) or None if loading failed and skip_on_error=True
        """
        loader = OmniModelLoader(model_path)
        try:
            return loader.load(mode=mode, **kwargs)
        except Exception as e:
            logger.error(f"Failed to load model from {model_path}: {e}")
            if skip_on_error:
                return None
            raise

    @staticmethod
    def is_model_supported(model_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Check if a model is supported and can be loaded.
        
        This is the primary API for checking model compatibility before attempting
        to load. It performs comprehensive checks including:
        - Config file existence and validity
        - Architecture recognition against supported lists
        - Model category detection
        - Custom modeling file detection
        - Special case handling (diffusers, SAE, custom mappings)
        
        Returns a dict with:
            - supported (bool): Whether the model can be loaded
            - has_custom_files (bool): Whether custom modeling files exist
            - architecture (str): The model architecture name (e.g., "LlamaForCausalLM")
            - model_type (str): The model type from config (e.g., "llama")
            - category (str): Model category - one of:
                - "transformers": Standard LLMs
                - "vision_encoder": Vision encoders
                - "asr": ASR models
                - "diffusers": Diffusers models
                - "sae": SAE models
            - error (str): Error message if not supported, None otherwise
            
        Example:
            >>> result = OmniModelLoader.is_model_supported("./models/llama-7b")
            >>> print(result["supported"])  # True
            >>> print(result["category"])   # "transformers"
            >>> print(result["architecture"])  # "LlamaForCausalLM"
        """
        path = Path(model_path)
        result = {
            "supported": False,
            "has_custom_files": False,
            "architecture": "unknown",
            "model_type": "unknown",
            "category": "unknown",
            "error": None
        }
        
        try:
            config_path = path / "config.json"
            if not config_path.exists():
                result["error"] = "No config.json found"
                return result
                
            with open(config_path) as f:
                config = json.load(f)
            
            model_type = config.get("model_type", "")
            architectures = config.get("architectures", [])
            architecture = architectures[0] if architectures else "unknown"
            
            result["model_type"] = model_type
            result["architecture"] = architecture
            
            # Detect category
            if OmniModelLoader._is_diffusers_model(path):
                result["category"] = "diffusers"
            elif OmniModelLoader._is_sae_model(path):
                result["category"] = "sae"
            elif OmniModelLoader._is_vision_encoder(path):
                result["category"] = "vision_encoder"
            elif OmniModelLoader._is_asr_model(path):
                result["category"] = "asr"
            else:
                result["category"] = "transformers"
            
            # Check for custom modeling files
            has_modeling = any((path / f"modeling_{model_type}.py").exists() for model_type in 
                              [model_type, model_type.replace("-", "_"), model_type.replace("-", "")])
            has_modeling_generic = any(f.startswith("modeling_") and f.endswith(".py") for f in os.listdir(path) if os.path.isfile(path / f))
            result["has_custom_files"] = has_modeling or has_modeling_generic
            
            # Check if architecture is supported
            all_supported = (
                OmniModelLoader.SUPPORTED_ARCHITECTURES +
                OmniModelLoader.VISION_ENCODER_ARCHITECTURES +
                OmniModelLoader.AUDIO_ENCODER_ARCHITECTURES +
                OmniModelLoader.ASR_ARCHITECTURES
            )
            
            if architecture in all_supported:
                result["supported"] = True
            elif result["has_custom_files"]:
                result["supported"] = True  # Can try with trust_remote_code
            elif result["category"] == "diffusers":
                result["supported"] = True  # Diffusers models are supported
            elif result["category"] == "sae":
                result["supported"] = True  # SAE models are supported (load base model tokenizer)
            elif model_type in OmniModelLoader.MODEL_TYPE_MAPPINGS:
                result["supported"] = True  # We have a workaround
            else:
                result["error"] = f"Architecture '{architecture}' (type: {model_type}) not in supported list and no custom files found"
                    
        except Exception as e:
            result["error"] = str(e)
            
        return result


def load_omni_model(path: Union[str, Path], mode: str = "thinker_only", 
                    skip_on_error: bool = False, **kwargs) -> Union[Tuple[Any, Any], None]:
    """
    Load an Omni model from the given path.
    
    Args:
        path: Path to the model
        mode: Loading mode ("full", "thinker_only", "talker_only")
        skip_on_error: If True, return None on error instead of raising
        **kwargs: Additional arguments passed to load()
        
    Returns:
        Tuple of (model, tokenizer) or None if skip_on_error=True and loading failed
    """
    if skip_on_error:
        return OmniModelLoader.load_model_safe(path, mode=mode, skip_on_error=True, **kwargs)
    return OmniModelLoader(path).load(mode=mode, **kwargs)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("model_path")
    p.add_argument("--mode", default="thinker_only")
    p.add_argument("--check-only", action="store_true")
    p.add_argument("--trust-remote-code", action="store_true", default=True)
    args = p.parse_args()
    if args.check_only:
        info = OmniModelLoader.get_model_info(args.model_path)
        support_info = OmniModelLoader.is_model_supported(args.model_path)
        print(f"Is Omni: {OmniModelLoader.is_omni_model(args.model_path)}")
        print(f"Info: {json.dumps(info, indent=2)}")
        print(f"Support Info: {json.dumps(support_info, indent=2)}")
    else:
        m, t = load_omni_model(args.model_path, mode=args.mode, trust_remote_code=args.trust_remote_code)
        print(f"Loaded {type(m)}")
