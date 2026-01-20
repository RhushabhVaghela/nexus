#!/usr/bin/env python3
"""
detect_modalities.py
Probes any base model to detect its native modalities.

Returns a structured report of:
- text: Always True for LLMs
- vision: True if model has vision encoder/config
- audio_input: True if model has audio encoder (STT)
- audio_output: True if model has audio decoder (TTS)
- video: True if model has video/temporal processing

Usage:
    python src/detect_modalities.py /path/to/model
    python src/detect_modalities.py --json /path/to/model
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any

# Try importing transformers, fallback to config-only mode
try:
    from transformers import AutoConfig
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


def detect_modalities(model_path: str) -> Dict[str, Any]:
    """
    Probe a model to detect its native modalities.
    
    Args:
        model_path: Path to model directory or HuggingFace model ID
        
    Returns:
        Dictionary with modality flags and metadata
    """
    result = {
        "model_path": model_path,
        "model_type": "unknown",
        "modalities": {
            "text": True,  # All LLMs have text
            "vision": False,           # See images
            "vision_output": False,    # Generate images
            "audio_input": False,      # Hear audio (STT)
            "audio_output": False,     # Speak audio (TTS)
            "video": False,            # Understand video
            "video_output": False,     # Generate video
        },
        "native_encoders": [],
        "native_decoders": [],
        "detection_method": "config_analysis",
    }
    
    # Try config-based detection first
    config_path = Path(model_path) / "config.json"
    
    if config_path.exists():
        with open(config_path, "r") as f:
            config = json.load(f)
        result = _analyze_config(config, result)
    elif HAS_TRANSFORMERS:
        # Fallback to AutoConfig for Hub models
        try:
            config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
            result = _analyze_autoconfig(config, result)
        except Exception as e:
            result["error"] = f"Failed to load config: {e}"
    else:
        result["error"] = "No config.json found and transformers not installed"
    
    return result


def _analyze_config(config: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze raw config.json for modality indicators."""
    
    model_type = config.get("model_type", "").lower()
    architectures = config.get("architectures", [])
    
    result["model_type"] = model_type
    result["architectures"] = architectures
    
    # Vision detection patterns
    vision_indicators = [
        "vision_config" in config,
        "visual_encoder" in config,
        "image_size" in config,
        "patch_size" in config,
        any("vision" in arch.lower() for arch in architectures),
        any("vl" in arch.lower() for arch in architectures),  # Vision-Language
        model_type in ["qwen2_vl", "llava", "idefics", "paligemma", "qwen2_5_omni"],
    ]
    
    if any(vision_indicators):
        result["modalities"]["vision"] = True
        result["native_encoders"].append("vision")
    
    # Audio Input (STT/ASR) detection patterns
    audio_input_indicators = [
        "audio_config" in config,
        "audio_encoder" in config,
        "whisper" in model_type,
        "speech" in model_type and "encoder" in str(config),
        any("audio" in arch.lower() for arch in architectures),
        model_type in ["qwen2_audio", "qwen2_5_omni"],
    ]
    
    if any(audio_input_indicators):
        result["modalities"]["audio_input"] = True
        result["native_encoders"].append("audio")
    
    # Audio Output (TTS) detection patterns
    audio_output_indicators = [
        "token2wav_config" in config,
        "audio_decoder" in config,
        "vocoder" in str(config).lower(),
        "tts" in model_type,
        model_type in ["qwen2_5_omni"],  # Known TTS-capable models
        "thinker_config" in config and "token2wav_config" in config,  # Omni pattern
    ]
    
    if any(audio_output_indicators):
        result["modalities"]["audio_output"] = True
        result["native_decoders"].append("audio")
    
    # Video detection patterns
    video_indicators = [
        "video_config" in config,
        "temporal" in str(config).lower(),
        "frame" in str(config).lower() and "vision" in str(config).lower(),
        model_type in ["video_llava", "qwen2_5_omni"],
    ]
    
    if any(video_indicators):
        result["modalities"]["video"] = True
        result["native_encoders"].append("video")
    
    # Vision Output (Image Generation) detection patterns
    vision_output_indicators = [
        "image_decoder" in config,
        "diffusion" in str(config).lower(),
        "vae_decoder" in str(config).lower(),
        any("generation" in arch.lower() and "image" in arch.lower() for arch in architectures),
    ]
    
    if any(vision_output_indicators):
        result["modalities"]["vision_output"] = True
        result["native_decoders"].append("vision")
    
    # Video Output (Video Generation) detection patterns
    video_output_indicators = [
        "video_decoder" in config,
        "video_generation" in str(config).lower(),
        any("video" in arch.lower() and "generation" in arch.lower() for arch in architectures),
    ]
    
    if any(video_output_indicators):
        result["modalities"]["video_output"] = True
        result["native_decoders"].append("video")
    
    # Special case: Omni models (like Qwen2.5-Omni)
    if model_type == "qwen2_5_omni" or "omni" in model_type.lower():
        result["modalities"]["vision"] = True
        result["modalities"]["audio_input"] = True
        result["modalities"]["audio_output"] = True
        result["modalities"]["video"] = True
        # Note: Qwen2.5-Omni does NOT have vision/video generation
        result["native_encoders"] = ["vision", "audio", "video"]
        result["native_decoders"] = ["audio"]
        result["is_omni"] = True
    
    return result


def _analyze_autoconfig(config, result: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze transformers AutoConfig object."""
    
    model_type = getattr(config, "model_type", "unknown")
    result["model_type"] = model_type
    
    # Check for vision config
    if hasattr(config, "vision_config") or hasattr(config, "visual_encoder_config"):
        result["modalities"]["vision"] = True
        result["native_encoders"].append("vision")
    
    # Check for audio config
    if hasattr(config, "audio_config") or hasattr(config, "audio_encoder_config"):
        result["modalities"]["audio_input"] = True
        result["native_encoders"].append("audio")
    
    # Check for TTS/audio output
    if hasattr(config, "token2wav_config") or hasattr(config, "audio_decoder_config"):
        result["modalities"]["audio_output"] = True
        result["native_decoders"].append("audio")
    
    # Check for video
    if hasattr(config, "video_config"):
        result["modalities"]["video"] = True
        result["native_encoders"].append("video")
    
    # Omni detection
    if "omni" in model_type.lower():
        result["is_omni"] = True
        result["modalities"] = {
            "text": True,
            "vision": True,
            "audio_input": True,
            "audio_output": True,
            "video": True,
        }
    
    return result


def format_report(result: Dict[str, Any], use_json: bool = False) -> str:
    """Format detection results for display."""
    
    if use_json:
        return json.dumps(result, indent=2)
    
    lines = [
        "=" * 60,
        "🔍 MODALITY DETECTION REPORT",
        "=" * 60,
        f"Model: {result['model_path']}",
        f"Type:  {result['model_type']}",
        "",
        "📊 Detected Modalities:",
    ]
    
    icons = {
        "text": "📝",
        "vision": "👁️",
        "vision_output": "🎨",
        "audio_input": "🎤",
        "audio_output": "🔊",
        "video": "📹",
        "video_output": "🎬",
    }
    
    for modality, present in result["modalities"].items():
        icon = icons.get(modality, "•")
        status = "✅ Yes" if present else "❌ No"
        lines.append(f"  {icon} {modality}: {status}")
    
    if result.get("native_encoders"):
        lines.append(f"\n🔧 Native Encoders: {', '.join(result['native_encoders'])}")
    
    if result.get("native_decoders"):
        lines.append(f"🔧 Native Decoders: {', '.join(result['native_decoders'])}")
    
    if result.get("is_omni"):
        lines.append("\n🌟 This is a FULL OMNI model (all modalities native)")
    
    if result.get("error"):
        lines.append(f"\n⚠️ Warning: {result['error']}")
    
    lines.append("=" * 60)
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Detect native modalities in a base model"
    )
    parser.add_argument(
        "model_path",
        help="Path to model directory or HuggingFace model ID"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON"
    )
    
    args = parser.parse_args()
    
    result = detect_modalities(args.model_path)
    print(format_report(result, args.json))
    
    # Exit with error if detection failed
    if result.get("error"):
        sys.exit(1)
    
    return result


if __name__ == "__main__":
    main()
