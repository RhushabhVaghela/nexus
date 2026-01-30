#!/usr/bin/env python3
"""
Comprehensive test for OmniModelLoader architecture support.

Tests all model types from the teacher registry:
- AgentCPM-Explore (Qwen3-based)
- GLM-4.7-Flash (glm4_moe_lite)
- Step3-VL-10B (Vision-language)
- Gemma Scope (SAE model)
- Stable Diffusion (Diffusers)
- SigLIP (Vision encoder)
- Whisper (ASR)
- etc.
"""

import sys
import json
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from omni.loader import OmniModelLoader, load_omni_model


def test_model_info_detection():
    """Test model info detection for various model types."""
    print("\n" + "="*60)
    print("TEST 1: Model Info Detection")
    print("="*60)
    
    test_cases = [
        ("./models/AgentCPM-Explore", "AgentCPM-Explore (Qwen3-based agent)"),
        ("./models/zai-org_GLM-4.7-Flash", "GLM-4.7-Flash (glm4_moe_lite)"),
        ("./models/stepfun-ai_Step3-VL-10B", "Step3-VL-10B (Vision-language)"),
        ("./models/google_gemma-scope-2-27b-pt", "Gemma Scope (SAE)"),
        ("/mnt/e/data/decoders/image-decoders/stabilityai_stable-diffusion-3-medium-diffusers", "Stable Diffusion (Diffusers)"),
        ("/mnt/e/data/encoders/image-encoders/siglip2-so400m-patch16-512", "SigLIP (Vision encoder)"),
        ("/mnt/e/data/encoders/vision-encoders/MCG-NJU_videomae-large", "VideoMAE (Video encoder)"),
        ("./models/microsoft_VibeVoice-ASR", "VibeVoice (ASR)"),
    ]
    
    for path, description in test_cases:
        print(f"\nTesting: {description}")
        print(f"  Path: {path}")
        try:
            info = OmniModelLoader.get_model_info(path)
            support = OmniModelLoader.is_model_supported(path)
            print(f"  Architecture: {info.get('architecture', 'unknown')}")
            print(f"  Model Type: {info.get('model_type', 'unknown')}")
            print(f"  Category: {support.get('category', 'unknown')}")
            print(f"  Supported: {support.get('supported', False)}")
            if info.get('error'):
                print(f"  Error: {info['error']}")
        except Exception as e:
            print(f"  ERROR: {e}")


def test_category_detection():
    """Test model category detection."""
    print("\n" + "="*60)
    print("TEST 2: Model Category Detection")
    print("="*60)
    
    # Test the static methods for category detection
    test_paths = [
        "./models/google_gemma-scope-2-27b-pt",
        "/mnt/e/data/decoders/image-decoders/stabilityai_stable-diffusion-3-medium-diffusers",
        "/mnt/e/data/encoders/image-encoders/siglip2-so400m-patch16-512",
        "./models/microsoft_VibeVoice-ASR",
    ]
    
    for path in test_paths:
        print(f"\nPath: {path}")
        try:
            category = OmniModelLoader._detect_model_category(Path(path))
            is_sae = OmniModelLoader._is_sae_model(Path(path))
            is_diffusers = OmniModelLoader._is_diffusers_model(Path(path))
            is_vision = OmniModelLoader._is_vision_encoder(Path(path))
            is_asr = OmniModelLoader._is_asr_model(Path(path))
            
            print(f"  Category: {category}")
            print(f"  Is SAE: {is_sae}")
            print(f"  Is Diffusers: {is_diffusers}")
            print(f"  Is Vision Encoder: {is_vision}")
            print(f"  Is ASR: {is_asr}")
        except Exception as e:
            print(f"  ERROR: {e}")


def test_architecture_registration():
    """Test architecture registration for custom models."""
    print("\n" + "="*60)
    print("TEST 3: Architecture Registration")
    print("="*60)
    
    # Test model type mappings
    mappings = OmniModelLoader.MODEL_TYPE_MAPPINGS
    print("\nModel Type Mappings:")
    for model_type, mapping in mappings.items():
        print(f"  {model_type} -> {mapping['architecture']} (config: {mapping['config_class']})")
    
    # Test supported architectures count
    print(f"\nSupported Architectures: {len(OmniModelLoader.SUPPORTED_ARCHITECTURES)}")
    print(f"Vision Encoder Architectures: {len(OmniModelLoader.VISION_ENCODER_ARCHITECTURES)}")
    print(f"Audio Encoder Architectures: {len(OmniModelLoader.AUDIO_ENCODER_ARCHITECTURES)}")
    print(f"ASR Architectures: {len(OmniModelLoader.ASR_ARCHITECTURES)}")


def test_safe_loading():
    """Test safe loading with skip_on_error."""
    print("\n" + "="*60)
    print("TEST 4: Safe Loading")
    print("="*60)
    
    test_paths = [
        "./models/AgentCPM-Explore",
        "./models/zai-org_GLM-4.7-Flash",
        "./models/nonexistent_model",
    ]
    
    for path in test_paths:
        print(f"\nTesting safe load: {path}")
        try:
            result = OmniModelLoader.load_model_safe(
                path, 
                mode="thinker_only",
                skip_on_error=True,
                trust_remote_code=True
            )
            if result is None:
                print(f"  Result: None (model skipped gracefully)")
            else:
                model, tokenizer = result
                print(f"  Result: Loaded {type(model).__name__}")
        except Exception as e:
            print(f"  ERROR: {e}")


def test_error_handling():
    """Test error handling for unsupported models."""
    print("\n" + "="*60)
    print("TEST 5: Error Handling")
    print("="*60)
    
    # Test with a non-existent model
    print("\nTesting with non-existent model:")
    result = OmniModelLoader.is_model_supported("./models/nonexistent_model")
    print(f"  Supported: {result['supported']}")
    print(f"  Error: {result['error']}")
    
    # Test with path that has no config
    print("\nTesting support check output format:")
    for key, value in result.items():
        print(f"  {key}: {value}")


def test_specific_architectures():
    """Test specific architecture handling."""
    print("\n" + "="*60)
    print("TEST 6: Specific Architecture Checks")
    print("="*60)
    
    # Check for specific architectures from teacher registry
    key_architectures = [
        "Glm4MoeForCausalLM",  # For GLM-4.7-Flash
        "Qwen3ForCausalLM",    # For AgentCPM-Explore
        "Step3VL10BForCausalLM",  # For Step3-VL-10B
        "WhisperForConditionalGeneration",  # For VibeVoice
        "SigLIPModel",         # For SigLIP
    ]
    
    all_supported = (
        OmniModelLoader.SUPPORTED_ARCHITECTURES +
        OmniModelLoader.VISION_ENCODER_ARCHITECTURES +
        OmniModelLoader.AUDIO_ENCODER_ARCHITECTURES +
        OmniModelLoader.ASR_ARCHITECTURES
    )
    
    for arch in key_architectures:
        supported = arch in all_supported
        status = "✓" if supported else "✗"
        print(f"  {status} {arch}: {'Supported' if supported else 'NOT SUPPORTED'}")


def main():
    """Run all tests."""
    print("="*60)
    print("OMNI MODEL LOADER - COMPREHENSIVE ARCHITECTURE TEST")
    print("="*60)
    
    try:
        test_model_info_detection()
        test_category_detection()
        test_architecture_registration()
        test_error_handling()
        test_specific_architectures()
        # test_safe_loading()  # Commented out as it requires actual models
        
        print("\n" + "="*60)
        print("ALL TESTS COMPLETED")
        print("="*60)
        
    except Exception as e:
        logger.exception("Test failed with error")
        print(f"\nTest failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
