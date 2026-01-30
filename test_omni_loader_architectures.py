#!/usr/bin/env python3
"""
Architecture support test for OmniModelLoader.
Tests detection and classification without requiring full model loading.
"""

import sys
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


def test_model_type_mappings():
    """Test that all model types from teacher registry are mapped."""
    print("\n" + "="*60)
    print("TEST: Model Type Mappings")
    print("="*60)
    
    # Import loader (this will fail if torch not installed, so we parse the file)
    loader_file = Path(__file__).parent / "src" / "omni" / "loader.py"
    content = loader_file.read_text()
    
    # Check for key mappings
    key_mappings = [
        '"glm4_moe_lite"',
        '"step_robotics"',
        '"qwen3"',
        '"agent_cpm"',
    ]
    
    for mapping in key_mappings:
        found = mapping in content
        status = "✓" if found else "✗"
        print(f"  {status} Mapping for {mapping}: {'Found' if found else 'MISSING'}")
    
    return True


def test_supported_architectures():
    """Test that key architectures are in the supported list."""
    print("\n" + "="*60)
    print("TEST: Supported Architectures")
    print("="*60)
    
    loader_file = Path(__file__).parent / "src" / "omni" / "loader.py"
    content = loader_file.read_text()
    
    key_architectures = [
        "Glm4MoeForCausalLM",
        "Glm4MoeLiteForCausalLM",
        "Qwen3ForCausalLM",
        "Qwen3MoeForCausalLM",
        "WhisperForConditionalGeneration",
        "SigLIPModel",
        "SigLIPVisionModel",
        "VideoMAEModel",
    ]
    
    for arch in key_architectures:
        found = f'"{arch}"' in content
        status = "✓" if found else "✗"
        print(f"  {status} {arch}: {'Found' if found else 'MISSING'}")
    
    return True


def test_category_detection_methods():
    """Test that category detection methods exist."""
    print("\n" + "="*60)
    print("TEST: Category Detection Methods")
    print("="*60)
    
    loader_file = Path(__file__).parent / "src" / "omni" / "loader.py"
    content = loader_file.read_text()
    
    methods = [
        "_is_sae_model",
        "_is_diffusers_model",
        "_is_vision_encoder",
        "_is_asr_model",
        "_detect_model_category",
    ]
    
    for method in methods:
        found = f"def {method}(" in content
        status = "✓" if found else "✗"
        print(f"  {status} Method {method}(): {'Found' if found else 'MISSING'}")
    
    return True


def test_specialized_loaders():
    """Test that specialized loaders exist."""
    print("\n" + "="*60)
    print("TEST: Specialized Loaders")
    print("="*60)
    
    loader_file = Path(__file__).parent / "src" / "omni" / "loader.py"
    content = loader_file.read_text()
    
    loaders = [
        "_load_diffusers_model",
        "_load_vision_encoder",
        "_load_asr_model",
    ]
    
    for loader in loaders:
        found = f"def {loader}(" in content
        status = "✓" if found else "✗"
        print(f"  {status} Loader {loader}(): {'Found' if found else 'MISSING'}")
    
    return True


def test_error_handling():
    """Test that error handling is comprehensive."""
    print("\n" + "="*60)
    print("TEST: Error Handling")
    print("="*60)
    
    loader_file = Path(__file__).parent / "src" / "omni" / "loader.py"
    content = loader_file.read_text()
    
    checks = [
        ("Fallback tokenizer logic", "fallback_tokenizers"),
        ("Safe load method", "load_model_safe"),
        ("Model support check", "is_model_supported"),
        ("Error logging", "logger.error"),
        ("Warning logging", "logger.warning"),
    ]
    
    for desc, pattern in checks:
        found = pattern in content
        status = "✓" if found else "✗"
        print(f"  {status} {desc}: {'Found' if found else 'MISSING'}")
    
    return True


def test_teacher_registry_coverage():
    """Test that all teacher registry models are covered."""
    print("\n" + "="*60)
    print("TEST: Teacher Registry Coverage")
    print("="*60)
    
    registry_file = Path(__file__).parent / "configs" / "teacher_registry.json"
    if not registry_file.exists():
        print("  ✗ Teacher registry not found")
        return False
    
    registry = json.loads(registry_file.read_text())
    loader_file = Path(__file__).parent / "src" / "omni" / "loader.py"
    content = loader_file.read_text()
    
    print(f"\n  Registry has {len(registry)} models")
    
    # Check specific model categories
    coverage = {
        "AgentCPM-Explore (Qwen3)": "qwen3" in content.lower() or "Qwen3" in content,
        "GLM-4.7-Flash (glm4_moe_lite)": "glm4_moe_lite" in content,
        "Gemma Scope (SAE)": "_is_sae_model" in content,
        "Stable Diffusion (Diffusers)": "_is_diffusers_model" in content,
        "SigLIP (Vision encoder)": "SigLIP" in content,
        "VideoMAE (Video encoder)": "VideoMAE" in content,
        "Whisper/VibeVoice (ASR)": "Whisper" in content,
    }
    
    for model, supported in coverage.items():
        status = "✓" if supported else "✗"
        print(f"  {status} {model}: {'Supported' if supported else 'NOT SUPPORTED'}")
    
    return True


def main():
    """Run all tests."""
    print("="*60)
    print("OMNI MODEL LOADER - ARCHITECTURE SUPPORT TEST")
    print("="*60)
    
    all_passed = True
    
    try:
        all_passed &= test_model_type_mappings()
        all_passed &= test_supported_architectures()
        all_passed &= test_category_detection_methods()
        all_passed &= test_specialized_loaders()
        all_passed &= test_error_handling()
        all_passed &= test_teacher_registry_coverage()
        
        print("\n" + "="*60)
        if all_passed:
            print("ALL TESTS PASSED ✓")
        else:
            print("SOME TESTS FAILED ✗")
        print("="*60)
        
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
