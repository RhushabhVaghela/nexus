"""
Memory Optimization Configuration for Nexus Training

Now dynamically detects actual system RAM and VRAM instead of
hardcoding 16GB/32GB. Falls back to conservative defaults if
detection fails.

Strategy:
1. CPU Offloading: Keep large frozen models on CPU, active parts on GPU
2. Quantization: 8-bit/4-bit inference for frozen encoders
3. Gradient Checkpointing: Reduce memory during backprop
4. Mixed Precision: FP16 training
"""

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False

try:
    from transformers import BitsAndBytesConfig

    BITSANDBYTES_AVAILABLE = True
except ImportError:
    BitsAndBytesConfig = None
    BITSANDBYTES_AVAILABLE = False

import psutil
import logging

logger = logging.getLogger(__name__)

# Import centralized guard for WSL-aware decisions
try:
    from src.utils.memory_guard import guard

    GUARD_AVAILABLE = True
except ImportError:
    guard = None
    GUARD_AVAILABLE = False


def _detect_system_resources():
    """
    Dynamically detect actual system RAM and VRAM.
    Returns (vram_gb, ram_gb) rounded to nearest integer.
    """
    # RAM
    try:
        ram_gb = round(psutil.virtual_memory().total / (1024**3))
    except Exception:
        ram_gb = 32  # Conservative fallback

    # VRAM
    vram_gb = 16  # Default fallback
    if TORCH_AVAILABLE and torch is not None:
        try:
            if torch.cuda.is_available():
                # Use the primary GPU
                props = torch.cuda.get_device_properties(0)
                vram_gb = round(props.total_memory / (1024**3))
        except Exception:
            pass

    return vram_gb, ram_gb


def _get_quantization_config():
    """Create BitsAndBytesConfig lazily to avoid import-time crashes."""
    if not BITSANDBYTES_AVAILABLE:
        return None
    return BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False,
    )


def _build_memory_config():
    """
    Build memory config dynamically based on actual hardware.
    """
    vram_gb, ram_gb = _detect_system_resources()
    is_wsl = guard.is_wsl if GUARD_AVAILABLE else False

    # Adjust estimated usage based on actual resources
    # If VRAM < 12GB, we can't fit video decoder on GPU
    if vram_gb < 12:
        vram_estimate = f"~{vram_gb - 2}GB"
        ram_estimate = f"~{min(ram_gb - 4, 24)}GB"
    else:
        vram_estimate = f"~{min(vram_gb - 2, 14)}GB"
        ram_estimate = f"~{min(ram_gb - 8, 20)}GB"

    config = {
        # Device map for hybrid CPU/GPU execution
        "device_map": {
            # Stage 1: Training DFM connectors + decoders, LLM frozen
            "stage1": {
                "llm": "cpu",  # Frozen LLM on CPU
                "vision_encoder": "cpu",  # Frozen on CPU
                "audio_encoder": "cpu",  # Frozen on CPU
                "vision_connector": "cuda:0",  # Training on GPU
                "audio_connector": "cuda:0",  # Training on GPU
                "video_decoder": "cuda:0" if vram_gb >= 12 else "cpu",
                "speech_decoder": "cpu",  # Move to CPU if needed
                "projections": "cuda:0",  # Small, keep on GPU
            },
            # Stage 2: Full model training
            "stage2": {
                "llm": "auto",  # Let accelerate decide
                "encoders": "cpu",
                "connectors": "cuda:0",
                "decoders": "auto",
            },
        },
        # Quantization config is generated lazily via _get_quantization_config()
        "quantization": None,
        # Training optimizations
        "training": {
            "gradient_checkpointing": True,
            "mixed_precision": "fp16",
            "batch_size": 1,
            "gradient_accumulation_steps": 8,
            "max_grad_norm": 1.0,
            "cpu_offload_optimizer": True,
        },
        # Detected system resources
        "vram": f"{vram_gb}GB",
        "ram": f"{ram_gb}GB",
        "is_wsl": is_wsl,
        "estimated_usage": {
            "vram": vram_estimate,
            "ram": ram_estimate,
        },
    }

    logger.info(
        f"Memory config built: VRAM={vram_gb}GB RAM={ram_gb}GB "
        f"WSL={'yes' if is_wsl else 'no'}"
    )

    return config


# Build config on import (lazy quantization, dynamic detection)
MEMORY_CONFIG = _build_memory_config()


def get_memory_config() -> dict:
    """Get memory config with lazily-initialized quantization."""
    config = dict(MEMORY_CONFIG)
    if config["quantization"] is None:
        config["quantization"] = _get_quantization_config()
    return config


def get_device_map_stage1():
    """Get optimized device map for Stage 1 training."""
    vram_gb, _ = _detect_system_resources()

    device_map = {
        # Frozen encoders on CPU
        "llm": "cpu",
        "vision_encoder.encoder": "cpu",
        "audio_encoder.encoder": "cpu",
        # Training components on GPU
        "vision_connector": 0,
        "audio_connector": 0,
        # Projections (small) on GPU
        "vision_proj": 0,
        "audio_proj": 0,
        "video_proj_out": 0,
        "speech_proj_out": 0,
    }

    # Video decoder needs ~8GB — only put on GPU if we have enough VRAM
    if vram_gb >= 12:
        device_map["video_decoder"] = 0
        device_map["speech_decoder"] = "cpu"
    else:
        device_map["video_decoder"] = "cpu"
        device_map["speech_decoder"] = "cpu"
        logger.warning(
            f"VRAM ({vram_gb}GB) < 12GB: video_decoder moved to CPU. "
            f"Training will be slower but won't OOM."
        )

    return device_map


def print_memory_plan():
    """Print memory allocation plan with detected resources."""
    vram_gb, ram_gb = _detect_system_resources()
    is_wsl = guard.is_wsl if GUARD_AVAILABLE else False
    env = "WSL2" if is_wsl else "Native"

    print("💾 Memory Optimization Plan:")
    print("=" * 60)
    print(f"System:     {env}")
    print(f"VRAM (GPU): {vram_gb}GB (detected)")
    print(f"RAM (CPU):  {ram_gb}GB (detected)")

    if is_wsl:
        print(f"⚠️  WSL2: Tighter memory thresholds active")

    print()
    print("Stage 1 Allocation:")
    print("  GPU (Trainable):")
    print("    - DFM Connectors (2x):  ~1.5GB")

    if vram_gb >= 12:
        print("    - Video Decoder:        ~8GB")
        print("    - Activations + Grads:  ~4GB")
        print(f"    Total GPU:              ~{min(vram_gb - 2, 14)}GB ✓")
    else:
        print("    - Video Decoder:        CPU (VRAM too small)")
        print("    - Activations + Grads:  ~2GB")
        print(f"    Total GPU:              ~{min(vram_gb - 2, 4)}GB ✓")

    print()
    print("  CPU (Frozen):")
    print("    - LLM (8-bit):          ~7GB")
    print("    - Vision Encoder:       ~2GB")
    print("    - Audio Encoder:        ~1GB")
    print("    - Speech Decoder:       ~2GB")
    if vram_gb < 12:
        print("    - Video Decoder:        ~8GB (offloaded from GPU)")
    print("    - Optimizer States:     ~6GB")
    print(f"    Total RAM:              ~{min(ram_gb - 8, 20)}GB ✓")
    print("=" * 60)


if __name__ == "__main__":
    print_memory_plan()
