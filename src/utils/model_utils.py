
import os
import logging
from typing import Optional
from transformers import AutoConfig

logger = logging.getLogger(__name__)

def check_modality(model_path: str, expected_type: str = "text") -> bool:
    """
    Checks the modality of a model based on its configuration attributes.
    This is stricter and more universal than name matching.
    
    Args:
        model_path: Path to the model.
        expected_type: 'text' or 'multimodal'.
        
    Returns:
        True if matches, False otherwise.
    """
    try:
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        
        # Heuristics for Multimodality
        # 1. Existence of vision/audio specific configs
        has_vision_config = hasattr(config, "vision_config") and config.vision_config is not None
        has_audio_config = hasattr(config, "audio_config") and config.audio_config is not None
        
        # 2. Check architecture strings for known multimodal patterns (fallback but robust)
        archs = config.architectures if config.architectures else []
        arch_str = str(archs).lower()
        is_multimodal_arch = any(x in arch_str for x in [
            'llava', 'blip', 'calyx', 'chameleon', 'flamingomodel', 'idefics', 
            'mllama', 'paligemma', 'qwen2_vl', 'vipallava', 'visual'
        ])
        
        # 3. Check for specific multimodal keys in config dict (if generic model)
        # e.g. "mm_vision_tower", "projector_hidden_act"
        config_dict = config.to_dict()
        has_mm_keys = any(k for k in config_dict.keys() if k.startswith("mm_") or "vision" in k or "audio" in k)
        
        is_multimodal = has_vision_config or has_audio_config or is_multimodal_arch # or has_mm_keys (removed has_mm_keys as it might be too aggressive for simple text models with 'revision' etc)
        
        # Re-check has_mm_keys more carefully
        if not is_multimodal:
             # Some VLM configs just put 'model_type': 'llava'
             if config.model_type in ['llava', 'llava_next', 'qwen2_vl', 'videollama']:
                 is_multimodal = True

        logger.info(f"Model Architecture: {archs}, Type: {config.model_type}")
        
        if expected_type == "text":
            if is_multimodal:
                logger.warning(f"⚠️ Modality Mismatch: Expected Text model, detected Multimodal signal (Arch: {archs}, Type: {config.model_type})")
                return False
            logger.info("✅ Modality Check Passed: Text Model")
            return True
            
        elif expected_type == "multimodal":
            if not is_multimodal:
                logger.warning(f"⚠️ Modality Mismatch: Expected Multimodal model, detected Text architecture (Arch: {archs}, Type: {config.model_type})")
                return False
            logger.info("✅ Modality Check Passed: Multimodal Model")
            return True
            
        return True
        
    except Exception as e:
        logger.error(f"Failed to check modality for {model_path}: {e}")
        # If we can't load config, we can't verify. Fail safe? Or allow?
        # User wants verification, so fail safe (False) is safer.
        return False
