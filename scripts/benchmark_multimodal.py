import sys
import os
import torch
import json
import time
import argparse
from tqdm import tqdm

# Ensure we can import from scripts/ and src/
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from scripts.inference_multimodal import load_multimodal_nexus

def run_benchmark(release_path, num_samples=10):
    print(f"🚀 Starting Multimodal Benchmark on {release_path}...")
    model, tokenizer = load_multimodal_nexus(release_path)
    model.eval()
    
    test_cases = [
        {"prompt": "What is in this image?", "type": "vision"},
        {"prompt": "Describe the sound in this clip.", "type": "audio"},
        {"prompt": "Explain the relationship between the visual and the text.", "type": "mixed"}
    ]
    
    results = []
    
    for i in tqdm(range(num_samples)):
        case = test_cases[i % len(test_cases)]
        
        # Prepare mock experts
        adapter_states = {}
        if case["type"] in ["vision", "mixed"]:
            adapter_states["vision"] = torch.randn(1, 16, model.config.hidden_size).to(model.device)
        if case["type"] in ["audio", "mixed"]:
            adapter_states["audio"] = torch.randn(1, 16, model.config.hidden_size).to(model.device)
            
        inputs = tokenizer(case["prompt"], return_tensors="pt").to(model.device)
        
        start_time = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                adapter_hidden_states=adapter_states if adapter_states else None
            )
        end_time = time.time()
        
        duration = end_time - start_time
        tokens = outputs.shape[1] - inputs["input_ids"].shape[1]
        
        results.append({
            "id": i,
            "type": case["type"],
            "tps": tokens / max(duration, 0.0001),
            "tokens": tokens,
            "duration": duration,
            "success": True
        })

    # Summary
    avg_tps = sum(r['tps'] for r in results) / len(results)
    print(f"\n✅ Benchmark Complete. Avg Throughput: {avg_tps:.2f} tokens/s")
    
    output_path = "results/multimodal_bench.json"
    os.makedirs("results", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"📊 Results saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="nexus-release-v1")
    parser.add_argument("--samples", type=int, default=10)
    args = parser.parse_args()
    
    run_benchmark(args.model, args.samples)
