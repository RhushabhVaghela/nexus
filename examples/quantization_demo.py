#!/usr/bin/env python3
"""
NVFP4 Quantization Demo - Nexus

Demonstrates how to use 4-bit quantization for memory-efficient inference.

Usage:
    python examples/quantization_demo.py --model "meta-llama/Llama-3.2-1B" --compare
    python examples/quantization_demo.py --model "gpt2"

Features:
- 4-bit quantization with NF4/FP4
- Memory usage comparison
- Inference speed benchmarking
"""

import argparse
import torch
import time
import psutil
import os
from typing import Dict, Tuple
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def get_memory_usage() -> Dict[str, float]:
    """Get current memory usage."""
    process = psutil.Process(os.getpid())
    return {
        "rss_mb": process.memory_info().rss / 1024 / 1024,
        "vms_mb": process.memory_info().vms / 1024 / 1024,
    }


def load_model_fp16(model_name: str):
    """Load model in FP16 (baseline)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    print(f"Loading {model_name} in FP16...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    return model, tokenizer


def load_model_4bit(model_name: str):
    """Load model in 4-bit quantized mode."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    
    print(f"Loading {model_name} in 4-bit...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    return model, tokenizer


def benchmark_inference(model, tokenizer, prompt: str, num_runs: int = 3) -> Tuple[float, str]:
    """Benchmark inference speed."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Warmup
    with torch.no_grad():
        _ = model.generate(**inputs, max_new_tokens=50)
    
    # Benchmark
    times = []
    for _ in range(num_runs):
        start = time.time()
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=50)
        times.append(time.time() - start)
    
    avg_time = sum(times) / len(times)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return avg_time, generated_text


def run_quantization_demo(model_name: str, compare: bool = True):
    """Run quantization demo."""
    print(f"\n{'='*60}")
    print(f"NEXUS - Quantization Demo")
    print(f"{'='*60}\n")
    
    prompt = "The quick brown fox jumps over the lazy dog. In the future, AI will"
    
    if compare:
        # FP16 Baseline
        print("[FP16 Baseline]")
        print("-" * 40)
        mem_before = get_memory_usage()
        model_fp16, tokenizer = load_model_fp16(model_name)
        mem_after = get_memory_usage()
        
        fp16_memory = mem_after["rss_mb"] - mem_before["rss_mb"]
        print(f"Memory used: {fp16_memory:.1f} MB")
        
        fp16_time, fp16_output = benchmark_inference(model_fp16, tokenizer, prompt)
        print(f"Inference time: {fp16_time:.2f}s")
        
        # Cleanup
        del model_fp16
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # 4-bit Quantized
        print("\n[4-bit Quantized]")
        print("-" * 40)
        mem_before = get_memory_usage()
        model_4bit, tokenizer = load_model_4bit(model_name)
        mem_after = get_memory_usage()
        
        quant_memory = mem_after["rss_mb"] - mem_before["rss_mb"]
        print(f"Memory used: {quant_memory:.1f} MB")
        
        quant_time, quant_output = benchmark_inference(model_4bit, tokenizer, prompt)
        print(f"Inference time: {quant_time:.2f}s")
        
        # Summary
        print(f"\n{'='*60}")
        print("Comparison Summary")
        print(f"{'='*60}")
        print(f"Memory Savings: {fp16_memory/quant_memory:.1f}x reduction")
        print(f"Speed Change: {quant_time/fp16_time:.2f}x ({'faster' if quant_time < fp16_time else 'slower'})")
    else:
        # Just show 4-bit
        print("[4-bit Quantized Mode]")
        print("-" * 40)
        model_4bit, tokenizer = load_model_4bit(model_name)
        quant_time, output = benchmark_inference(model_4bit, tokenizer, prompt)
        print(f"Inference time: {quant_time:.2f}s")
        print(f"Output: {output[:100]}...")
    
    print(f"\n{'='*60}")
    print("Quantization Demo Complete!")
    print(f"{'='*60}\n")
    
    print("Next steps:")
    print("  1. Try different models")
    print("  2. Use in training: ./scripts/nexus.sh universal --enable-cot")
    print("  3. See docs/ for NVFP4 hardware acceleration")


def main():
    parser = argparse.ArgumentParser(description="Quantization Demo")
    parser.add_argument("--model", type=str, default="gpt2", help="Model name")
    parser.add_argument("--compare", action="store_true", help="Compare FP16 vs 4-bit")
    
    args = parser.parse_args()
    run_quantization_demo(args.model, args.compare)


if __name__ == "__main__":
    main()
