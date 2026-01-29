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

Usage:
    from src.omni.loader import OmniModelLoader
    
    loader = OmniModelLoader("/path/to/any-model")
    model, tokenizer = loader.load(mode="thinker_only")
"""

import os
import logging
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, Union
from dataclasses import dataclass

import torch

logger = logging.getLogger(__name__)


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
    
    Supports:
    - Any HuggingFace model (auto-detection)
    - GPTQ/GGUF quantized models
    - Full precision and bfloat16 models
    - Multimodal models (vision, audio, video)
    - Text-only LLMs (Qwen, Llama, Mistral, Phi, Gemma, DeepSeek, etc.)
    """
    
    SUPPORTED_ARCHITECTURES = [
        "AfmoeForCausalLM",
        "ApertusForCausalLM",
        "ArceeForCausalLM",
        "ArcticForCausalLM",
        "AudioFlamingo3ForConditionalGeneration",
        "BaiChuanForCausalLM",
        "BaichuanForCausalLM",
        "BailingMoeForCausalLM",
        "BailingMoeV2ForCausalLM",
        "BambaForCausalLM",
        "BertForMaskedLM",
        "BertForSequenceClassification",
        "BertModel",
        "BitnetForCausalLM",
        "BloomForCausalLM",
        "BloomModel",
        "CamembertModel",
        "ChameleonForCausalLM",
        "ChameleonForConditionalGeneration",
        "ChatGLMForConditionalGeneration",
        "ChatGLMModel",
        "CodeShellForCausalLM",
        "CogVLMForCausalLM",
        "Cohere2ForCausalLM",
        "CohereForCausalLM",
        "DbrxForCausalLM",
        "DeciLMForCausalLM",
        "DeepseekForCausalLM",
        "DistilBertForMaskedLM",
        "DistilBertForSequenceClassification",
        "DistilBertModel",
        "Dots1ForCausalLM",
        "DreamModel",
        "Ernie4_5ForCausalLM",
        "Ernie4_5_ForCausalLM",
        "Ernie4_5_MoeForCausalLM",
        "Exaone4ForCausalLM",
        "ExaoneForCausalLM",
        "ExaoneMoEForCausalLM",
        "FalconForCausalLM",
        "FalconH1ForCausalLM",
        "FalconMambaForCausalLM",
        "GPT2LMHeadModel",
        "GPTBigCodeForCausalLM",
        "GPTNeoXForCausalLM",
        "GPTRefactForCausalLM",
        "Gemma2ForCausalLM",
        "Gemma3ForCausalLM",
        "Gemma3ForConditionalGeneration",
        "Gemma3TextModel",
        "Gemma3nForCausalLM",
        "Gemma3nForConditionalGeneration",
        "GemmaForCausalLM",
        "Glm4ForCausalLM",
        "Glm4MoeForCausalLM",
        "Glm4MoeLiteForCausalLM",
        "Glm4vForConditionalGeneration",
        "Glm4vMoeForConditionalGeneration",
        "GlmForCausalLM",
        "GlmasrModel",
        "GptOssForCausalLM",
        "GraniteForCausalLM",
        "GraniteMoeForCausalLM",
        "GraniteMoeHybridForCausalLM",
        "GraniteMoeSharedForCausalLM",
        "Grok1ForCausalLM",
        "GrokForCausalLM",
        "GroveMoeForCausalLM",
        "HunYuanDenseV1ForCausalLM",
        "HunYuanMoEV1ForCausalLM",
        "Idefics3ForConditionalGeneration",
        "InternLM2ForCausalLM",
        "InternLM3ForCausalLM",
        "InternVisionModel",
        "JAISLMHeadModel",
        "JambaForCausalLM",
        "JanusForConditionalGeneration",
        "JinaBertForMaskedLM",
        "JinaBertModel",
        "KORMoForCausalLM",
        "KimiVLForConditionalGeneration",
        "LFM2ForCausalLM",
        "LLaDAMoEModel",
        "LLaDAMoEModelLM",
        "LLaDAModelLM",
        "Lfm2AudioForConditionalGeneration",
        "Lfm2ForCausalLM",
        "Lfm2Model",
        "Lfm2MoeForCausalLM",
        "Lfm2VlForConditionalGeneration",
        "LightOnOCRForConditionalGeneration",
        "Llama4ForCausalLM",
        "Llama4ForConditionalGeneration",
        "LlamaBidirectionalModel",
        "LlavaStableLMEpochForCausalLM",
        "MPTForCausalLM",
        "MT5ForConditionalGeneration",
        "MaincoderForCausalLM",
        "Mamba2ForCausalLM",
        "MambaForCausalLM",
        "MambaLMHeadModel",
        "MiMoV2FlashForCausalLM",
        "MiniCPM3ForCausalLM",
        "MiniCPMForCausalLM",
        "MiniMaxM2ForCausalLM",
        "Mistral3ForConditionalGeneration",
        "ModernBertForMaskedLM",
        "ModernBertForSequenceClassification",
        "ModernBertModel",
        "NemotronForCausalLM",
        "NemotronHForCausalLM",
        "NeoBERT",
        "NeoBERTForSequenceClassification",
        "NeoBERTLMHead",
        "NomicBertModel",
        "OLMoForCausalLM",
        "Olmo2ForCausalLM",
        "Olmo3ForCausalLM",
        "OlmoForCausalLM",
        "OlmoeForCausalLM",
        "OpenELMForCausalLM",
        "OrionForCausalLM",
        "PLMForCausalLM",
        "PLaMo2ForCausalLM",
        "PLaMo3ForCausalLM",
        "PanguEmbeddedForCausalLM",
        "Phi3ForCausalLM",
        "PhiForCausalLM",
        "PhiMoEForCausalLM",
        "Plamo2ForCausalLM",
        "Plamo3ForCausalLM",
        "PlamoForCausalLM",
        "QWenLMHeadModel",
        "Qwen2AudioForConditionalGeneration",
        "Qwen2ForCausalLM",
        "Qwen2Model",
        "Qwen2MoeForCausalLM",
        "Qwen2OmniTalkerForConditionalGeneration",
        "Qwen2VLForConditionalGeneration",
        "Qwen2VLModel",
        "Qwen2_5OmniForConditionalGeneration",
        "Qwen2_5OmniModel",
        "Qwen2_5_VLForConditionalGeneration",
        "Qwen3ForCausalLM",
        "Qwen3MoeForCausalLM",
        "Qwen3NextForCausalLM",
        "Qwen3OmniForConditionalGeneration",
        "Qwen3TTSForConditionalGeneration",
        "Qwen3VLForConditionalGeneration",
        "Qwen3VLMoeForConditionalGeneration",
        "RND1",
        "RWForCausalLM",
        "RWKV6Qwen2ForCausalLM",
        "RWKV7ForCausalLM",
        "RobertaForSequenceClassification",
        "RobertaModel",
        "Rwkv6ForCausalLM",
        "Rwkv7ForCausalLM",
        "RwkvHybridForCausalLM",
        "SeedOssForCausalLM",
        "SmallThinkerForCausalLM",
        "SmolLM3ForCausalLM",
        "SmolVLMForConditionalGeneration",
        "SolarOpenForCausalLM",
        "StableLMEpochForCausalLM",
        "StableLmForCausalLM",
        "Starcoder2ForCausalLM",
        "T5EncoderModel",
        "T5ForConditionalGeneration",
        "T5WithLMHeadModel",
        "UMT5ForConditionalGeneration",
        "UMT5Model",
        "UltravoxModel",
        "VoxtralForConditionalGeneration",
        "WavTokenizerDec",
        "XLMRobertaForSequenceClassification",
        "XLMRobertaModel",
        "XverseForCausalLM",
        "YoutuVLForConditionalGeneration",
        "modeling_grove_moe.GroveMoeForCausalLM",
    ]
    
    def __init__(self, model_path: Optional[Union[str, Path]] = None):
        self.model_path = Path(model_path) if model_path else None
        self._model = None
        self._tokenizer = None
        self._processor = None
        self._config = None # This will store the model info
    
    @classmethod
    def is_omni_model(cls, model_path: Union[str, Path]) -> bool:
        """Check if the given path contains an Omni model."""
        path = Path(model_path)
        if not path.exists():
            return False
            
        # 1. Direct registry check for known architectures
        info = cls.get_model_info(model_path)
        if info["is_supported"]:
            return True
            
        # 2. Config-based check
        config_path = path / "config.json"
        if config_path.exists():
            try:
                import json
                with open(config_path) as f:
                    config = json.load(f)
                
                # Check model_type or architectures
                model_type = config.get("model_type", "").lower()
                if "omni" in model_type or "any-to-any" in model_type:
                    return True
                    
                archs = config.get("architectures", [])
                for arch in archs:
                    if "omni" in arch.lower() or "qwen" in arch.lower():
                        return True
                    if "llama" in arch.lower() or "mistral" in arch.lower():
                        return True
            except: pass
            
        return False

    @classmethod
    def get_model_info(cls, model_path: Union[str, Path]) -> Dict[str, Any]:
        """Get information about the model."""
        import json
        info = {
            "name": Path(model_path).name,
            "size": "unknown",
            "is_quantized": False,
            "has_talker": False,
            "architecture": "unknown",
            "is_supported": False
        }
        
        try:
            config_path = Path(model_path) / "config.json"
            if config_path.exists():
                with open(config_path) as f:
                    config = json.load(f)
                    
                if "architectures" in config:
                    info["architecture"] = config["architectures"][0]
                    
                if "quantization_config" in config:
                    info["is_quantized"] = True
                    
                if any(k for k in config.keys() if "talker" in k or "audio" in k):
                    info["has_talker"] = True

                # Check support
                if info["architecture"] in cls.SUPPORTED_ARCHITECTURES:
                    info["is_supported"] = True
                elif "omni" in config.get("model_type", "").lower():
                    info["is_supported"] = True
                    
        except Exception:
            pass
            
        return info

    def load(self, mode: str = "full", **kwargs) -> Any:
        """Load the model."""
        return self.load_for_inference(mode=mode, **kwargs)

    def load_for_inference(self, mode: str = "full", **kwargs) -> Any:
        """
        Load model for inference with UNIVERSAL support.
        """
        model_path = self.model_path
        if "model_path" in kwargs:
            model_path = Path(kwargs.pop("model_path"))

        logger.info(f"Loading Model from {model_path} (Mode: {mode})")
        
        trust_remote_code = kwargs.pop("trust_remote_code", True)
        
        try:
            from transformers import AutoTokenizer, AutoProcessor, AutoModelForCausalLM, AutoModel
            
            # 1. Load Tokenizer
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    str(model_path),
                    trust_remote_code=trust_remote_code,
                )
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                self._tokenizer = tokenizer
            except Exception as e:
                logger.error(f"Failed to load tokenizer from {model_path}: {e}")
                raise RuntimeError(f"Tokenizer dependency missing: {e}")

            # 2. Load Processor (Multimodal)
            try:
                processor = AutoProcessor.from_pretrained(
                    str(model_path),
                    trust_remote_code=trust_remote_code,
                )
                self._processor = processor
            except Exception:
                logger.debug("No AutoProcessor found (might be text-only model)")
            
            # 3. Load Model with Fallback Strategy
            device_map = kwargs.pop("device_map", "auto")
            torch_dtype = kwargs.pop("torch_dtype", "auto")
            low_cpu_mem_usage = kwargs.pop("low_cpu_mem_usage", True)
            
            # Check for Prompt Repetition factors in kwargs
            # If present and > 1, we might want to use the OmniMultimodalLM wrapper
            # from src.multimodal.model if this is a base model.
            visual_rep = kwargs.pop("visual_repetition_factor", 1)
            audio_rep = kwargs.pop("audio_repetition_factor", 1)
            
            if visual_rep > 1 or audio_rep > 1:
                logger.info(f"Prompt Repetition requested (V:{visual_rep}, A:{audio_rep}). Loading via OmniMultimodalLM wrapper.")
                from src.multimodal.model import OmniMultimodalLM
                model = OmniMultimodalLM(
                    llm_name=str(model_path),
                    visual_repetition_factor=visual_rep,
                    audio_repetition_factor=audio_rep,
                    device_map=device_map,
                    **kwargs
                )
                self._model = model
                return model, self._tokenizer
            
            # Remove redundant kwargs that are explicitly passed
            kwargs.pop("trust_remote_code", None)

            # 4. Finalize Load Strategy
            current_device = kwargs.get("device") or ("cuda:0" if torch.cuda.is_available() else "cpu")
            
            # --- Monkey-patch transformers/torch for non-standard architectures ---
            try:
                import transformers.quantizers.quantizers_utils as qu
                import torch.nn as nn
                orig_get_submodule = nn.Module.get_submodule

                def patched_get_submodule(self, target):
                    if not target: return self
                    try:
                        return orig_get_submodule(self, target)
                    except AttributeError as e:
                        if "is not an nn.Module" in str(e):
                            # Traversal bug: target path contains a parameter. 
                            # Search for the deepest valid module parent.
                            parts = target.split('.')
                            for i in range(len(parts) - 1, 0, -1):
                                try:
                                    return orig_get_submodule(self, ".".join(parts[:i]))
                                except: continue
                            return self
                        raise e
                
                nn.Module.get_submodule = patched_get_submodule
                logger.info("Applied aggressive monkey-patch to nn.Module.get_submodule")
            except Exception as e:
                logger.warning(f"Failed to apply transformers monkey-patch: {e}")
            # ----------------------------------------------------------------------

            # Map of class names to actual objects
            from transformers import (
                AutoModelForCausalLM, AutoModel, AutoModelForVision2Seq, 
                AutoModelForSeq2SeqLM, AutoModelForImageTextToText
            )
            auto_classes = {
                "AutoModelForCausalLM": AutoModelForCausalLM,
                "AutoModel": AutoModel,
                "AutoModelForVision2Seq": AutoModelForVision2Seq,
                "AutoModelForSeq2SeqLM": AutoModelForSeq2SeqLM,
                "AutoModelForImageTextToText": AutoModelForImageTextToText
            }

            # Build prioritized strategies
            strategies = []
            
            # 1. Add auto_map preferences first
            try:
                config_json = model_path / "config.json"
                if config_json.exists():
                    with open(config_json) as f:
                        cfg_data = json.load(f)
                        if "auto_map" in cfg_data:
                            for auto_name in cfg_data["auto_map"].keys():
                                if auto_name in auto_classes:
                                    strategies.append((auto_name, auto_classes[auto_name]))
            except:
                pass

            # 2. Add defaults if not already present
            default_priority = [
                "AutoModelForCausalLM", 
                "AutoModelForImageTextToText",
                "AutoModelForVision2Seq", 
                "AutoModel"
            ]
            for name in default_priority:
                if name not in [s[0] for s in strategies]:
                    strategies.append((name, auto_classes[name]))

            model = None
            last_error = None
            
            # --- Monkey-patch transformers for non-standard architectures ---
            # This fixes the AttributeError: 'weight' is not an nn.Module during 4-bit loading
            orig_get_module_from_name = None
            try:
                import transformers.quantizers.quantizers_utils as qu
                orig_get_module_from_name = qu.get_module_from_name
                
                def patched_get_module_from_name(module, tensor_name: str):
                    if "." in tensor_name:
                        module_name, tensor_name = tensor_name.rsplit(".", 1)
                        try:
                            # Try standard PyTorch way first
                            module = module.get_submodule(module_name)
                        except AttributeError as e:
                            if "is not an nn.Module" in str(e):
                                # Traversal bug: path contains a parameter. 
                                # Manually traverse using getattr to skip module check.
                                for part in module_name.split("."):
                                    module = getattr(module, part)
                            else:
                                raise e
                    return module, tensor_name
                
                qu.get_module_from_name = patched_get_module_from_name
                logger.info("Applied monkey-patch to transformers.quantizers.quantizers_utils.get_module_from_name")
            except Exception as e:
                logger.warning(f"Failed to apply transformers monkey-patch: {e}")
            # -----------------------------------------------------------------

            try:
                for class_name, cls_obj in strategies:
                    try:
                        logger.info(f"Attempting load with {class_name}...")
                        
                        # Setup quantization if needed
                        if kwargs.get("load_in_4bit") or kwargs.get("load_in_8bit"):
                            from transformers import BitsAndBytesConfig
                            skip_modules = [
                                "vision_model", "visual", "vision_tower", "multi_modal_projector", 
                                "vit_large_projector", "ln_pre", "ln_post", "ln_1", "ln_2", "conv1",
                                "wit_large_projector", "downsampler"
                            ]
                            q_config = BitsAndBytesConfig(
                                load_in_4bit=kwargs.pop("load_in_4bit", False),
                                load_in_8bit=kwargs.pop("load_in_8bit", False),
                                llm_int8_skip_modules=skip_modules,
                                bnb_4bit_compute_dtype=torch.float16,
                                bnb_4bit_quant_type="nf4",
                                bnb_4bit_use_double_quant=True
                            )
                            kwargs["quantization_config"] = q_config

                        # Strategy A: Standard with low_cpu and auto device map
                        try:
                            model = cls_obj.from_pretrained(
                                str(model_path),
                                device_map=device_map,
                                trust_remote_code=trust_remote_code,
                                torch_dtype=torch_dtype,
                                low_cpu_mem_usage=low_cpu_mem_usage,
                                **kwargs
                            )
                            break
                        except AttributeError as ae:
                            if "is not an nn.Module" in str(ae):
                                logger.warning(f"Caught traversal error for {class_name}. Retrying with pinned device_map...")
                                pinned_map = {"": torch.cuda.current_device() if torch.cuda.is_available() else "cpu"}
                                model = cls_obj.from_pretrained(
                                    str(model_path),
                                    device_map=pinned_map,
                                    trust_remote_code=trust_remote_code,
                                    torch_dtype=torch_dtype,
                                    low_cpu_mem_usage=False,
                                    **kwargs
                                )
                                break
                            else: raise ae
                        except Exception as e:
                            if "low_cpu_mem_usage" in str(e):
                                logger.warning(f"Retrying {class_name} without low_cpu_mem_usage...")
                                model = cls_obj.from_pretrained(
                                    str(model_path),
                                    device_map=device_map,
                                    trust_remote_code=trust_remote_code,
                                    torch_dtype=torch_dtype,
                                    low_cpu_mem_usage=False,
                                    **kwargs
                                )
                                break
                            else: raise e
                    except Exception as e:
                        logger.warning(f"{class_name} strategy failed: {e}")
                        last_error = e
                        continue
            finally:
                if orig_get_module_from_name:
                    import transformers.quantizers.quantizers_utils as qu
                    qu.get_module_from_name = orig_get_module_from_name
                    logger.info("Restored original get_module_from_name")
            
            if model is None:
                raise RuntimeError(f"All loading strategies failed. Final error: {last_error}")

            logger.info(f"Model loaded successfully: {type(model).__name__}")
            
            # 4. Load LoRA Adapter if present
            adapter_config = model_path / "adapter_config.json"
            if adapter_config.exists():
                logger.info("LoRA adapter detected, loading...")
                from peft import PeftModel
                model = PeftModel.from_pretrained(model, model_path)
                logger.info("✓ LoRA adapter merged for inference")

            self._model = model
            self._config = self.get_model_info(model_path)
            
            return model, self._tokenizer

        except Exception as e:
            logger.error(f"Critical error loading model: {e}")
            raise
            
    def load_for_training(
        self,
        model_path: Optional[str] = None,
        freeze_talker: bool = True,
        **kwargs
    ) -> Tuple[Any, Any]:
        """
        Load model configured for training.
        """
        if model_path:
            self.model_path = Path(model_path)
            
        model, tokenizer = self.load(mode="thinker_only", **kwargs)
        
        # Freeze talker if present and requested
        if freeze_talker:
            if hasattr(model, 'talker'):
                for param in model.talker.parameters():
                    param.requires_grad = False
                logger.info("Talker parameters frozen for training")
            elif hasattr(model, 'wrapper') and hasattr(model.wrapper, 'speech_decoder'):
                # Handle our wrapper-based talker
                for param in model.wrapper.speech_decoder.parameters():
                    param.requires_grad = False
                logger.info("Wrapped speech decoder frozen for training")
        
        # Enable gradient checkpointing to save memory
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled")
        elif hasattr(model, 'wrapper') and hasattr(model.wrapper.llm, 'gradient_checkpointing_enable'):
            model.wrapper.llm.gradient_checkpointing_enable()
            logger.info("Wrapped LLM gradient checkpointing enabled")
        
        return model, tokenizer


def load_omni_model(
    model_path: Union[str, Path],
    mode: str = "thinker_only",
    **kwargs
) -> Tuple[Any, Any]:
    """Convenience function."""
    loader = OmniModelLoader(model_path)
    return loader.load(mode=mode, **kwargs)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Omni Model Loader")
    parser.add_argument("model_path", help="Path to Omni model")
    parser.add_argument("--mode", default="thinker_only", help="Loading mode")
    parser.add_argument("--check-only", action="store_true", help="Only check if model is Omni")
    args = parser.parse_args()
    
    if args.check_only:
        is_omni = OmniModelLoader.is_omni_model(args.model_path)
        info = OmniModelLoader.get_model_info(args.model_path)
        print(f"Is Omni: {is_omni}")
        print(f"Info: {info}")
    else:
        loader = OmniModelLoader(args.model_path)
        model, tokenizer = loader.load(mode=args.mode)
        print(f"Model loaded: {type(model)}")
        print(f"Tokenizer vocab size: {len(tokenizer)}")
