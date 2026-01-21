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
    
    def __init__(self):
        self._model = None
        self._tokenizer = None
        self._processor = None
        self._config = None
    
    @staticmethod
    def is_omni_model(model_path: Union[str, Path]) -> bool:
        """Check if the model path contains an Omni model."""
        model_path = Path(model_path)
        config_path = model_path / "config.json"
        
        if not config_path.exists():
            return False
        
        try:
            import json
            with open(config_path) as f:
                config = json.load(f)
            
            model_type = config.get("model_type", "")
            architectures = config.get("architectures", [])
            
            # Check for Omni indicators
            if "omni" in model_type.lower():
                return True
            
            for arch in architectures:
                if "Omni" in arch:
                    return True
            
            return False
        except Exception:
            return False
    
    @staticmethod
    def get_model_info(model_path: Union[str, Path]) -> Dict[str, Any]:
        """Get information about the Omni model."""
        model_path = Path(model_path)
        config_path = model_path / "config.json"
        
        info = {
            "is_omni": False,
            "has_talker": False,
            "is_quantized": False,
            "quantization_method": None,
            "architectures": [],
            "model_type": "",
        }
        
        if not config_path.exists():
            return info
        
        try:
            import json
            with open(config_path) as f:
                config = json.load(f)
            
            info["model_type"] = config.get("model_type", "")
            info["architectures"] = config.get("architectures", [])
            info["is_omni"] = "omni" in info["model_type"].lower()
            info["has_talker"] = "talker_config" in config
            
            # Check quantization
            if "quantization_config" in config:
                info["is_quantized"] = True
                info["quantization_method"] = config["quantization_config"].get("quant_method", "unknown")
            
            # Check for quantize_config.json (GPTQ)
            quant_config_path = model_path / "quantize_config.json"
            if quant_config_path.exists():
                info["is_quantized"] = True
                with open(quant_config_path) as f:
                    quant_config = json.load(f)
                    info["quantization_method"] = quant_config.get("quant_method", "gptq")
            
            return info
        except Exception as e:
            logger.warning(f"Error reading model info: {e}")
            return info
    
    def load(
        self,
        model_path: Union[str, Path],
        mode: str = "thinker_only",
        device_map: str = "auto",
        torch_dtype: Optional[torch.dtype] = None,
        trust_remote_code: bool = True,
    ) -> Tuple[Any, Any]:
        """
        Load Omni model and tokenizer.
        
        Args:
            model_path: Path to the Omni model
            mode: Loading mode:
                - "thinker_only": Load only the thinker (LLM) for text training
                - "full": Load complete model with talker for audio output
                - "talker_only": Load only the talker component
            device_map: Device mapping strategy
            torch_dtype: Torch data type (default: auto-detect)
            trust_remote_code: Whether to trust remote code
        
        Returns:
            Tuple of (model, tokenizer)
        """
        model_path = Path(model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        info = self.get_model_info(model_path)
        
        if not info["is_omni"]:
            logger.warning(f"Model at {model_path} is not an Omni model, using standard loader")
            return self._load_standard_model(model_path, device_map, torch_dtype, trust_remote_code)
        
        logger.info(f"Loading Omni model from {model_path}")
        logger.info(f"Mode: {mode}, Quantized: {info['is_quantized']}, Has Talker: {info['has_talker']}")
        
        # Determine dtype
        if torch_dtype is None:
            if info["is_quantized"]:
                torch_dtype = torch.float16
            else:
                torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        
        try:
            from transformers import AutoTokenizer, AutoProcessor
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                str(model_path),
                trust_remote_code=trust_remote_code,
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Try to load processor (for multimodal)
            try:
                processor = AutoProcessor.from_pretrained(
                    str(model_path),
                    trust_remote_code=trust_remote_code,
                )
            except Exception:
                processor = None
            
            # Load model based on mode
            if mode == "thinker_only":
                model = self._load_thinker_only(model_path, info, device_map, torch_dtype, trust_remote_code)
            elif mode == "full":
                model = self._load_full_model(model_path, info, device_map, torch_dtype, trust_remote_code)
            else:
                raise ValueError(f"Unknown mode: {mode}")
            
            self._model = model
            self._tokenizer = tokenizer
            self._processor = processor
            self._config = info
            
            return model, tokenizer
            
        except Exception as e:
            logger.error(f"Error loading Omni model: {e}")
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
        try:
            # Try specific Omni model class first
            from transformers import Qwen2_5OmniModel
            
            model = Qwen2_5OmniModel.from_pretrained(
                str(model_path),
                device_map=device_map,
                torch_dtype=torch_dtype,
                trust_remote_code=True,
            )
            return model
        except ImportError:
            pass

        # Fallback to AutoModelForCausalLM (handles "Qwen2_5OmniForConditionalGeneration")
        try:
            from transformers import AutoModelForCausalLM
            logger.info("Falling back to AutoModelForCausalLM...")
            model = AutoModelForCausalLM.from_pretrained(
                str(model_path),
                device_map=device_map,
                torch_dtype=torch_dtype,
                trust_remote_code=True,
            )
            return model
        except Exception:
            # Final fallback to AutoModel
            logger.warning(
                "AutoModelForCausalLM failed. Using AutoModel with trust_remote_code=True."
            )
            from transformers import AutoModel
            
            model = AutoModel.from_pretrained(
                str(model_path),
                device_map=device_map,
                torch_dtype=torch_dtype,
                trust_remote_code=True,
            )
            return model
    
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
    
    def load_for_inference(
        self,
        model_path: Union[str, Path],
        enable_audio: bool = True,
    ) -> Tuple[Any, Any]:
        """
        Load model configured for inference.
        
        Args:
            model_path: Path to the model
            enable_audio: Whether to enable audio output
        
        Returns:
            Tuple of (model, tokenizer)
        """
        mode = "full" if enable_audio else "thinker_only"
        model, tokenizer = self.load(model_path, mode=mode)
        
        model.eval()
        
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
