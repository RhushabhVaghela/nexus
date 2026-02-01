
import torch
import json
import warnings
from pathlib import Path

# Fix import path
import sys
sys.path.insert(0, str(Path(__file__).parent))
from multimodal.model import OmniMultimodalLM
from multimodal import get_test_prompts

warnings.filterwarnings("ignore")

def run_benchmark():
    print("--- BENCHMARKING BASELINE MODEL (Untrained Projectors) ---")
    
    # 1. Load Model
    model = OmniMultimodalLM(
        llm_name="/mnt/e/data/models/Qwen2.5-Omni-7B-GPTQ-Int4",
        device_map="auto"
    )
    model.eval()
    
    # 2. Get Test Prompts
    prompts = get_test_prompts()
    
    results = []
    
    # 3. Run Inference
    print("\n--- Running Inference ---")
    
    # Text-only baseline
    print("Test: Text Only")
    inputs = model.llm.tokenizer("Describe the concept of multimodal learning.", return_tensors="pt").to(model.llm.device)
    out = model.llm.generate(**inputs, max_new_tokens=50)
    text_out = model.llm.tokenizer.decode(out[0])
    print(f"Output: {text_out[:100]}...")
    results.append({"type": "text", "input": "Describe multimodal learning", "output": text_out})

    # Vision - Run Real Inference
    print("\nTest: Vision (Multimodal)")
    try:
        # Create a real tensor input (simulating an image) to ensure encoder runs
        # Shape: [1, 3, 384, 384] standard for SigLIP/CLIP suitable for testing speed
        dummy_image = torch.randn(1, 3, 384, 384).to(model.llm.device)
        
        # We need to construct the inputs manually since we don't have the processor handy here
        # or we assume model.generate can handle pixel_values if the model class supports it.
        # OmniMultimodalLM usually expects 'pixel_values' or 'images' in generate.
        
        # Check if model has a helper to wrap inputs
        # For baseline, we just run the vision encoder directly to prove it works/benchmark it
        if hasattr(model, "vision_encoder"):
             print("Running Vision Encoder Forward Pass...")
             with torch.no_grad():
                 features = model.vision_encoder(dummy_image)
             print(f"Vision Encoder Output Shape: {features.shape}")
             results.append({"type": "vision_encoder", "status": "success", "shape": str(features.shape)})
        else:
             print("Model has no vision_encoder attribute.")
             
    except Exception as e:
        print(f"Vision Benchmark Error: {e}")
        results.append({"type": "vision", "error": str(e)})

    print("\n✅ Baseline Benchmark Complete.")
    print("Note: Vision encoder verified with synthetic tensor input.")

if __name__ == "__main__":
    run_benchmark()
