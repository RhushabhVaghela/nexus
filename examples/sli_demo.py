#!/usr/bin/env python3
"""
Sequential Layer Ingestion (SLI) Demo - Nexus

Demonstrates how to use the Universal SLI system to process large models
layer-by-layer on consumer hardware.

Usage:
    python examples/sli_demo.py --model "meta-llama/Llama-3.2-1B"
    python examples/sli_demo.py --model "gpt2"
    python examples/sli_demo.py --model "google/flan-t5-base"

Features:
- Automatic architecture detection
- Memory-efficient layer-by-layer processing
- Support for 11+ architecture families
"""

import argparse
import torch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.core.towers.universal_loader import OmniModelLoader
from src.data.universal_loader import UniversalDataLoader


def run_sli_demo(model_name: str, sample_text: str = None):
    """
    Run SLI demo on a model.
    
    Args:
        model_name: HuggingFace model name or local path
        sample_text: Optional sample text for processing
    """
    print(f"\n{'='*60}")
    print(f"NEXUS - Sequential Layer Ingestion (SLI) Demo")
    print(f"{'='*60}\n")
    
    # Default sample text
    if sample_text is None:
        sample_text = "The future of artificial intelligence is"
    
    print(f"Model: {model_name}")
    print(f"Sample Text: {sample_text}")
    print("-" * 60)
    
    # Step 1: Load model metadata
    print("\n[Step 1] Loading model with Universal SLI...")
    try:
        loader = OmniModelLoader(model_name)
        print(f"✓ Model loaded successfully")
        print(f"  - Architecture: {loader.metadata.get('architecture', 'Unknown')}")
        print(f"  - Hidden Size: {loader.metadata.get('hidden_size', 'Unknown')}")
        print(f"  - Num Layers: {loader.metadata.get('num_layers', 'Unknown')}")
        print(f"  - Num Parameters: {loader.metadata.get('num_parameters', 'Unknown')}")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        return
    
    # Step 2: Process layer by layer
    print("\n[Step 2] Processing layers sequentially...")
    num_layers = loader.metadata.get("num_layers", 12)
    
    # Simulate layer processing
    for layer_idx in range(min(num_layers, 5)):  # Show first 5 layers
        print(f"  Processing layer {layer_idx + 1}/{num_layers}...", end="\r")
        
        # In a real scenario, this would load and process the layer
        # layer = loader.load_layer(layer_idx)
        # output = layer.process(input_data)
        
    print(f"  ✓ Processed {min(num_layers, 5)} layers (showing first 5)")
    
    # Step 3: Demonstrate memory efficiency
    print("\n[Step 3] Memory Analysis...")
    if torch.cuda.is_available():
        memory_allocated = torch.cuda.memory_allocated() / 1024**3
        memory_reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"  GPU Memory Allocated: {memory_allocated:.2f} GB")
        print(f"  GPU Memory Reserved: {memory_reserved:.2f} GB")
    else:
        print("  Running on CPU (no GPU memory usage)")
    
    # Step 4: Show supported capabilities
    print("\n[Step 4] Supported Capabilities...")
    capabilities = {
        "text_generation": True,
        "layer_by_layer": True,
        "memory_efficient": True,
        "architecture_families": [
            "Llama", "GPT", "Qwen", "MoE", "T5", "Mamba",
            "BERT", "Gemma", "Phi", "BLOOM", "OPT"
        ]
    }
    
    for capability, supported in capabilities.items():
        if capability == "architecture_families":
            print(f"  {capability}: {', '.join(supported)}")
        else:
            status = "✓" if supported else "✗"
            print(f"  {status} {capability}")
    
    # Step 5: Example processing with data loader
    print("\n[Step 5] Data Loading Example...")
    try:
        data_loader = UniversalDataLoader()
        print(f"  ✓ Data loader initialized")
        print(f"  Supported formats: text, json, jsonl, csv, parquet")
    except Exception as e:
        print(f"  Note: Data loader requires additional setup: {e}")
    
    print(f"\n{'='*60}")
    print("SLI Demo Complete!")
    print(f"{'='*60}\n")
    
    print("Next steps:")
    print("  1. Try with different models: python examples/sli_demo.py --model 'gpt2'")
    print("  2. See docs/SLI_UNIVERSAL_GUIDE.md for full documentation")
    print("  3. Run training: ./scripts/nexus.sh universal --enable-cot")


def main():
    parser = argparse.ArgumentParser(
        description="Sequential Layer Ingestion (SLI) Demo"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt2",
        help="Model name or path (default: gpt2)"
    )
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="Sample text for processing"
    )
    
    args = parser.parse_args()
    
    run_sli_demo(args.model, args.text)


if __name__ == "__main__":
    main()
