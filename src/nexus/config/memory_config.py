"""
Memory Optimization Configuration for 16GB VRAM + 32GB RAM

Strategy:
1. CPU Offloading: Keep large frozen models on CPU, active parts on GPU
2. Quantization: 8-bit/4-bit inference for frozen encoders
3. Gradient Checkpointing: Reduce memory during backprop
4. Mixed Precision: FP16 training
"""

import torch
from transformers import BitsAndBytesConfig

# Memory-optimized model loading config
MEMORY_CONFIG = {
    # Device map for hybrid CPU/GPU execution
    "device_map": {
        # Stage 1: Training DFM connectors + decoders, LLM frozen
        "stage1": {
            "llm": "cpu",  # Frozen LLM on CPU (13GB)
            "vision_encoder": "cpu",  # Frozen on CPU (4.3GB) 
            "audio_encoder": "cpu",  # Frozen on CPU (1.6GB)
            "vision_connector": "cuda:0",  # Training on GPU (398M)
            "audio_connector": "cuda:0",  # Training on GPU (398M)
            "video_decoder": "cuda:0",  # Training on GPU (7.2GB)
            "speech_decoder": "cpu",  # Move to CPU if needed (2.4GB)
            "projections": "cuda:0"  # Small, keep on GPU
        },
        
        # Stage 2: Full model training
        "stage2": {
            "llm": "auto",  # Let accelerate decide
            "encoders": "cpu",
            "connectors": "cuda:0",
            "decoders": "auto"
        }
    },
    
    # Quantization config for frozen models (reduce memory)
    "quantization": BitsAndBytesConfig(
        load_in_8bit=True,  # 8-bit quantization
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False,
    ),
    
    # Training optimizations
    "training": {
        "gradient_checkpointing": True,  # Trade compute for memory
        "mixed_precision": "fp16",  # FP16 training
        "batch_size": 1,  # Start with 1, increase if possible
        "gradient_accumulation_steps": 8,  # Simulate larger batches
        "max_grad_norm": 1.0,
        "cpu_offload_optimizer": True  # Offload optimizer states to CPU
    },
    
    # Memory limits
    "vram": "16GB",
    "ram": "32GB",
    "estimated_usage": {
        "vram": "~14GB",  # Connectors + decoders + activations
        "ram": "~20GB"  # Frozen models + optimizer states
    }
}


def get_device_map_stage1():
    """Get optimized device map for Stage 1 training."""
    return {
        # Frozen encoders on CPU
        "llm": "cpu",
        "vision_encoder.encoder": "cpu",
        "audio_encoder.encoder": "cpu",
        
        # Training components on GPU
        "vision_connector": 0,
        "audio_connector": 0,
        "video_decoder": 0,
        
        # Offload speech decoder if GPU full
        "speech_decoder": "cpu",
        
        # Projections (small) on GPU
        "vision_proj": 0,
        "audio_proj": 0,
        "video_proj_out": 0,
        "speech_proj_out": 0,
    }


def print_memory_plan():
    """Print memory allocation plan."""
    print("💾 Memory Optimization Plan:")
    print("=" * 60)
    print(f"VRAM (GPU): {MEMORY_CONFIG['vram']}")
    print(f"RAM (CPU):  {MEMORY_CONFIG['ram']}")
    print()
    print("Stage 1 Allocation:")
    print("  GPU (Traina ble):")
    print("    - DFM Connectors (2x):  ~1.5GB")
    print("    - Video Decoder:        ~8GB")
    print("    - Activations + Grads:  ~4GB")
    print("    Total GPU:              ~14GB ✓")
    print()
    print("  CPU (Frozen):")
    print("    - LLM (8-bit):          ~7GB")
    print("    - Vision Encoder:       ~2GB")
    print("    - Audio Encoder:        ~1GB")
    print("    - Speech Decoder:       ~2GB")
    print("    - Optimizer States:     ~6GB")
    print("    Total RAM:              ~18GB ✓")
    print("=" * 60)


if __name__ == "__main__":
    print_memory_plan()
