#!/usr/bin/env python3
"""
Omni Model Loader
Universal loader for Qwen2.5-Omni models supporting training, validation, and inference.

The Qwen2.5-Omni model has two main components:
- Thinker: Main LLM for text reasoning and multimodal understanding
- Talker: Audio output generation model

Usage:
    from src.omni.loader import OmniModelLoader
    
    loader = OmniModelLoader()
    model, tokenizer = loader.load("/path/to/omni-model", mode="thinker_only")
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
    Universal loader for Qwen2.5-Omni models.
    
    Supports:
    - GPTQ quantized models
    - Full precision models
    - Thinker-only mode for text training
    - Full mode for audio output
    """
    
    SUPPORTED_ARCHITECTURES = [
        "Qwen2_5OmniForConditionalGeneration",
        "Qwen2OmniTalkerForConditionalGeneration",
    ]
    
    def __init__(self, model_path: Union[str, Path]):
        self.model_path = Path(model_path)
        self._model = None
        self._tokenizer = None
        self._processor = None
        self._config = None # This will store the model info
    
    @staticmethod
    def is_omni_model(model_path: Union[str, Path]) -> bool:
        """
        Check if the model at model_path is an Omni-compatible model.
        
        Now supports UNIVERSAL detection - checks for specific Omni markers 
        but defaults to treating any valid model path as potentially compatible 
        if explicitly requested via the Omni loader.
        """
        path = Path(model_path)
        if not path.exists():
            return False
            
        # If it contains "omni" in the name, we treat it as an Omni model
        if "omni" in path.name.lower():
            return True
            
        # Otherwise, check config for specifics if needed, but be permissive
        # We assume if the user is using OmniModelLoader, they intend to load it as such
        # or we will fallback to standard AutoModel in load().
        try:
             import json
             config_path = path / "config.json"
             if config_path.exists():
                 with open(config_path) as f:
                     config = json.load(f)
                 # Check for explicit architecture or just 'auto_map' presence which implies custom code
                 if "architectures" in config:
                     archs = config["architectures"]
                     # Permissive check: if it's ANY known Omni variant OR just a standard LLM we can extend
                     for arch in archs:
                         if "omni" in arch.lower() or "qwen" in arch.lower():
                             return True
                         # Also support standard models that we might want to "Omni-fy" via adapter
                         if "llama" in arch.lower() or "mistral" in arch.lower():
                             return True
        except Exception:
            pass
            
        # Default: If we can't disprove it, and the folder exists, we allow it.
        # The loader will fail gracefully later if it's truly incompatible.
        return True

    @staticmethod
    def get_model_info(model_path: Union[str, Path]) -> Dict[str, Any]:
        """Get information about the model."""
        import json
        info = {
            "name": Path(model_path).name,
            "size": "unknown",
            "is_quantized": False,
            "has_talker": False,
            "architecture": "unknown"
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
                    
                # Check for talker keys
                if any(k for k in config.keys() if "talker" in k or "audio" in k):
                    info["has_talker"] = True
                    
        except Exception:
            pass
            
        return info

    def load(self, mode: str = "full", **kwargs) -> Any:
        """Load the model (wrapper for load_for_inference)."""
        return self.load_for_inference(mode=mode, **kwargs)

    def load_for_inference(self, mode: str = "full", **kwargs) -> Any:
        """
        Load model for inference with UNIVERSAL support.
        """
        logger.info(f"Loading Model from {self.model_path} (Mode: {mode})")
        
        trust_remote_code = kwargs.get("trust_remote_code", True)
        
        try:
            from transformers import AutoTokenizer, AutoProcessor, AutoModelForCausalLM, AutoModel
            
            # 1. Load Tokenizer
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    str(self.model_path),
                    trust_remote_code=trust_remote_code,
                )
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                self._tokenizer = tokenizer
            except Exception as e:
                logger.error(f"Failed to load tokenizer from {self.model_path}: {e}")
                raise RuntimeError(f"Tokenizer dependency missing: {e}")

            # 2. Load Processor (Multimodal)
            try:
                processor = AutoProcessor.from_pretrained(
                    str(self.model_path),
                    trust_remote_code=trust_remote_code,
                )
                self._processor = processor
            except Exception:
                logger.debug("No AutoProcessor found (might be text-only model)")
            
            # 3. Load Model with Fallback Strategy
            device_map = kwargs.get("device_map", "auto")
            torch_dtype = kwargs.get("torch_dtype", "auto")
            
            logger.info("Attempting load with AutoModelForCausalLM...")
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    device_map=device_map,
                    trust_remote_code=trust_remote_code,
                    torch_dtype=torch_dtype,
                    low_cpu_mem_usage=True
                )
            except Exception as e1:
                logger.warning(f"AutoModelForCausalLM failed ({e1}), falling back to AutoModel...")
                try:
                    model = AutoModel.from_pretrained(
                        self.model_path,
                        device_map=device_map,
                        trust_remote_code=trust_remote_code,
                        torch_dtype=torch_dtype,
                        low_cpu_mem_usage=True
                    )
                except Exception as e2:
                    raise RuntimeError(f"All loading strategies failed. AutoModel error: {e2}")

            logger.info(f"Model loaded successfully: {type(model).__name__}")
            self._model = model
            
            # Store config info
            self._config = self.get_model_info(self.model_path)
            
            return model, self._tokenizer

        except Exception as e:
            logger.error(f"Critical error loading model: {e}")
            raise
            

    
    def _load_thinker_only(
        self,
        model_path: Path,
        info: Dict[str, Any],
        device_map: str,
        torch_dtype: torch.dtype,
        trust_remote_code: bool,
    ):
        """Load only the thinker component for text training."""
        from transformers import AutoModel, AutoModelForCausalLM
        
        # For GPTQ models, use AutoModel with trust_remote_code
        if info["is_quantized"] and info["quantization_method"] == "gptq":
            logger.info("Loading GPTQ quantized Omni model...")
            try:
                # Try AutoModel first (handles Omni architecture)
                model = AutoModel.from_pretrained(
                    str(model_path),
                    device_map=device_map,
                    torch_dtype=torch_dtype,
                    trust_remote_code=trust_remote_code,
                )
            except Exception as e:
                logger.warning(f"AutoModel failed: {e}, trying direct import...")
                # Fall back to direct import
                model = self._load_with_direct_import(model_path, device_map, torch_dtype)
        else:
            # Standard loading
            model = AutoModel.from_pretrained(
                str(model_path),
                device_map=device_map,
                torch_dtype=torch_dtype,
                trust_remote_code=trust_remote_code,
            )
        
        return model
    
    def _load_full_model(
        self,
        model_path: Path,
        info: Dict[str, Any],
        device_map: str,
        torch_dtype: torch.dtype,
        trust_remote_code: bool,
    ):
        """Load full model with thinker and talker."""
        from transformers import AutoModel
        
        try:
            model = AutoModel.from_pretrained(
                str(model_path),
                device_map=device_map,
                torch_dtype=torch_dtype,
                trust_remote_code=trust_remote_code,
            )
        except Exception as e:
            logger.warning(f"AutoModel failed: {e}, trying direct import...")
            # Fall back to direct import
            model = self._load_with_direct_import(model_path, device_map, torch_dtype)
        
        return model
    
    def _load_with_direct_import(
        self,
        model_path: Path,
        device_map: str,
        torch_dtype: torch.dtype,
    ):
        """Load model using direct import or AutoModelForCausalLM fallback."""
        # Method 1: Try specific architecture class directly (matches config.json)
        try:
            from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import Qwen2_5OmniForConditionalGeneration
            logger.info("Directly loading Qwen2_5OmniForConditionalGeneration...")
            model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
                str(model_path),
                device_map=device_map,
                torch_dtype=torch_dtype,
                trust_remote_code=True,
            )
            return model
        except ImportError:
            logger.debug("Could not direct import Qwen2_5OmniForConditionalGeneration")
        except Exception as e:
            logger.warning(f"Failed loading via Qwen2_5OmniForConditionalGeneration: {e}")

        # Method 2: Try specific base model class
        try:
            from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import Qwen2_5OmniModel
            logger.info("Directly loading Qwen2_5OmniModel...")
            model = Qwen2_5OmniModel.from_pretrained(
                str(model_path),
                device_map=device_map,
                torch_dtype=torch_dtype,
                trust_remote_code=True,
            )
            return model
        except ImportError:
            logger.debug("Could not direct import Qwen2_5OmniModel")
        except Exception as e:
             logger.warning(f"Failed loading via Qwen2_5OmniModel: {e}")

        # Method 3: AutoModel/AutoModelForCausalLM with explicit config
        try:
            from transformers import AutoConfig, AutoModel, AutoModelForCausalLM
            
            logger.info("Falling back to AutoConfig loading...")
            config = AutoConfig.from_pretrained(str(model_path), trust_remote_code=True)
            
            # Try AutoModelForCausalLM first
            try:
                return AutoModelForCausalLM.from_pretrained(
                    str(model_path),
                    config=config,
                    device_map=device_map,
                    torch_dtype=torch_dtype,
                    trust_remote_code=True,
                )
            except Exception as e:
                logger.warning(f"AutoModelForCausalLM failed: {e}")

            # Try AutoModel
            logger.warning("Falling back to AutoModel...")
            return AutoModel.from_pretrained(
                str(model_path),
                config=config,
                device_map=device_map,
                torch_dtype=torch_dtype,
                trust_remote_code=True,
            )
            
        except Exception as e:
            logger.warning(f"AutoModel fallback failed: {e}")
            
        # Method 4: Standard CAUSAL LM fallback (Most robust for 7B)
        try:
            logger.info("Falling back to standard AutoModelForCausalLM...")
            return AutoModelForCausalLM.from_pretrained(
                str(model_path),
                device_map=device_map,
                torch_dtype=torch_dtype,
                trust_remote_code=True,
            )
        except Exception as e:
             logger.error(f"Critical error loading Omni model: {e}")
             raise
    
    def _load_standard_model(
        self,
        model_path: Path,
        device_map: str,
        torch_dtype: Optional[torch.dtype],
        trust_remote_code: bool,
    ):
        """Load a standard (non-Omni) model."""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        if torch_dtype is None:
            torch_dtype = torch.float16
        
        tokenizer = AutoTokenizer.from_pretrained(
            str(model_path),
            trust_remote_code=trust_remote_code,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            device_map=device_map,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
        )
        
        return model, tokenizer
    
    def load_for_training(
        self,
        model_path: Union[str, Path],
        freeze_talker: bool = True,
    ) -> Tuple[Any, Any]:
        """
        Load model configured for training.
        
        Args:
            model_path: Path to the model
            freeze_talker: Whether to freeze talker weights (recommended)
        
        Returns:
            Tuple of (model, tokenizer)
        """
        model, tokenizer = self.load(model_path, mode="thinker_only")
        
        # Freeze talker if present and requested
        if freeze_talker and hasattr(model, 'talker'):
            for param in model.talker.parameters():
                param.requires_grad = False
            logger.info("Talker parameters frozen for training")
        
        # Enable gradient checkpointing to save memory
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled")
        
        return model, tokenizer


def load_omni_model(
    model_path: Union[str, Path],
    mode: str = "thinker_only",
    **kwargs
) -> Tuple[Any, Any]:
    """
    Convenience function to load Omni model.
    
    Args:
        model_path: Path to model
        mode: Loading mode ("thinker_only", "full")
        **kwargs: Additional arguments for loader
    
    Returns:
        Tuple of (model, tokenizer)
    """
    loader = OmniModelLoader()
    return loader.load(model_path, mode=mode, **kwargs)


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
        loader = OmniModelLoader()
        model, tokenizer = loader.load(args.model_path, mode=args.mode)
        print(f"Model loaded: {type(model)}")
        print(f"Tokenizer vocab size: {len(tokenizer)}")
