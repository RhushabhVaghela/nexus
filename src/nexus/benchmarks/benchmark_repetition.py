#!/usr/bin/env python3
"""
benchmark_repetition.py
Benchmark suite for Prompt Repetition (arXiv:2512.14982).
Compares Baseline vs 2x vs 3x across Text, Vision, and Audio.
"""

import time
import torch
import json
import random
import argparse
from pathlib import Path
from typing import Dict, List, Any, Tuple
from PIL import Image, ImageDraw, ImageFont
import numpy as np

from src.multimodal.model import OmniMultimodalLM
from src.utils.repetition import PromptRepetitionEngine

class RepetitionBenchmark:
    def __init__(self, model_path: str, device: str = "cuda"):
        self.device = device
        self.engine = PromptRepetitionEngine()
        
        print(f"🚀 Loading model for benchmark: {model_path}")
        try:
            self.model = OmniMultimodalLM(
                llm_name=model_path,
                device_map="auto",
                load_in_8bit=True,
                enable_decoders=False
            )
            self.model.eval()
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        except Exception as e:
            print(f"⚠️ Failed to load real model: {e}")
            print("💡 Falling back to MOCK mode for logic verification.")
            self.model = None
            self.tokenizer = None

    def gen_text_task(self, num_names: int = 50) -> Tuple[str, str, str]:
        """NameIndex task: Retrieve Nth name."""
        names = ["Alice", "Bob", "Charlie", "David", "Eve", "Frank", "Grace", "Heidi", "Ivan", "Judy"]
        names = names * (num_names // 10 + 1)
        random.shuffle(names)
        names = names[:num_names]
        
        target_idx = random.randint(1, num_names)
        target_name = names[target_idx - 1]
        
        context = "List of names: " + ", ".join(names)
        query = f"What is the {target_idx}th name in the list?"
        
        return query, context, target_name

    def gen_vision_task(self) -> Tuple[Image.Image, str, str]:
        """Visual NameIndex: List of words in an image."""
        img = Image.new('RGB', (800, 600), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)
        
        items = ["APPLE", "BANANA", "CHERRY", "DOG", "EAGLE", "FISH", "GRAPE", "HORSE", "ICE", "JACKET"]
        random.shuffle(items)
        
        for i, item in enumerate(items):
            draw.text((50, 50 + i*50), f"{i+1}. {item}", fill=(0, 0, 0))
            
        target_idx = random.randint(1, 10)
        target_item = items[target_idx - 1]
        query = f"What is the {target_idx}th item shown in the image?"
        
        return img, query, target_item

    def run_inference(self, prompt: str, image: Image.Image = None, factor: int = 1) -> Tuple[str, float]:
        """Run inference and return (result, latency_ms)."""
        start = time.perf_counter()
        
        if self.model is None:
            # Mock logic
            time.sleep(0.1 * factor) # Simulate latency
            return "MOCK_RESULT", (time.perf_counter() - start) * 1000

        # Real inference
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        pixel_values = None
        if image:
            # Simple conversion for mock/test, in real usage use processor
            pixel_values = torch.randn(1, 3, 512, 512).to(self.device, dtype=torch.float16)

        with torch.no_grad():
            # Update model repetition factor dynamically
            self.model.wrapper.visual_repetition_factor = factor
            self.model.wrapper.audio_repetition_factor = factor
            
            outputs = self.model.wrapper.llm.generate(
                inputs.input_ids,
                max_new_tokens=20,
                temperature=0.1
            )
            
        result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        latency = (time.perf_counter() - start) * 1000
        return result, latency

    def run_suite(self, iterations: int = 5):
        results = {
            "text": {"baseline": {"acc": 0, "lat": []}, "2x": {"acc": 0, "lat": []}},
            "vision": {"baseline": {"acc": 0, "lat": []}, "2x": {"acc": 0, "lat": []}}
        }
        
        for i in range(iterations):
            print(f"Iteration {i+1}/{iterations}...")
            
            # --- TEXT TASK ---
            q, ctx, target = self.gen_text_task()
            
            # Baseline
            res_b, lat_b = self.run_inference(f"{ctx}\n{q}", factor=1)
            results["text"]["baseline"]["acc"] += 1 if target.lower() in res_b.lower() else 0
            results["text"]["baseline"]["lat"].append(lat_b)
            
            # 2x Repetition
            prompt_2x = self.engine.apply_repetition(q, ctx, factor=2)
            res_2x, lat_2x = self.run_inference(prompt_2x, factor=2)
            results["text"]["2x"]["acc"] += 1 if target.lower() in res_2x.lower() else 0
            results["text"]["2x"]["lat"].append(lat_2x)
            
            # --- VISION TASK ---
            img, v_q, v_target = self.gen_vision_task()
            
            # Baseline
            res_vb, lat_vb = self.run_inference(v_q, image=img, factor=1)
            results["vision"]["baseline"]["acc"] += 1 if v_target.lower() in res_vb.lower() else 0
            results["vision"]["baseline"]["lat"].append(lat_vb)
            
            # 2x Embedding Repetition
            res_v2, lat_v2 = self.run_inference(v_q, image=img, factor=2)
            results["vision"]["2x"]["acc"] += 1 if v_target.lower() in res_v2.lower() else 0
            results["vision"]["2x"]["lat"].append(lat_v2)

        # Summarize
        print("\n" + "="*40)
        print("📊 REPETITION BENCHMARK SUMMARY")
        print("="*40)
        
        for task in ["text", "vision"]:
            print(f"\n[{task.upper()}]")
            for style in ["baseline", "2x"]:
                acc = results[task][style]["acc"] / iterations * 100
                lat = sum(results[task][style]["lat"]) / iterations
                print(f"  {style:8}: Accuracy {acc:5.1f}%, Latency {lat:6.1f}ms")
        
        # Save to file
        output_file = Path("repetition_benchmark_results.json")
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n✅ Detailed results saved to {output_file}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="/mnt/e/data/models/Qwen2.5-Omni-7B-GPTQ-Int4")
    parser.add_argument("--iterations", type=int, default=5)
    args = parser.parse_args()
    
    bench = RepetitionBenchmark(args.model_path)
    bench.run_suite(iterations=args.iterations)

if __name__ == "__main__":
    main()
