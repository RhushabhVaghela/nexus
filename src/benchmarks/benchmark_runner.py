#!/usr/bin/env python3
"""
Comprehensive Benchmark Runner
Measures performance, accuracy, and generates detailed metrics.

Metrics tracked:
- Timing: tokens/sec, latency, first token time
- Accuracy: Token accuracy, perplexity estimate
- Memory: GPU peak, GPU reserved, RAM usage
- System: Device info, batch sizes

Usage:
    python src/benchmarks/benchmark_runner.py --model /path/to/model --output-dir results
"""

import os
import sys
import time
import csv
import json
import gc
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional
from datetime import datetime

import torch

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import unified tracker
from src.metrics_tracker import MetricsTracker, BenchmarkMetrics, ALL_DATASETS
from src.omni.loader import OmniModelLoader


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark run."""
    model_path: str
    output_dir: str = "results"
    warmup_runs: int = 2
    benchmark_runs: int = 5
    max_new_tokens: int = 100
    batch_sizes: List[int] = field(default_factory=lambda: [1, 2, 4])


class BenchmarkRunner:
    """
    Runs comprehensive benchmarks on a model.
    """
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        # self.results is now managed by MetricsTracker, but we keep a local list for summary
        self.local_results: List[BenchmarkMetrics] = []
        self.model = None
        self.tokenizer = None
        self.tracker = MetricsTracker(output_dir=config.output_dir)
        self.loader = OmniModelLoader(config.model_path)
        
    def setup(self):
        """Load model and tokenizer using Universal Loader."""
        print(f"Loading model from {self.config.model_path}")
        self.model, self.tokenizer = self.loader.load(mode="full")
        self.model.eval()
        print(f"Model loaded on {self.model.device}")
    
    def _get_memory_stats(self) -> Dict[str, float]:
        """Get current memory statistics."""
        stats = {"gpu_peak_mb": 0, "gpu_reserved_mb": 0, "ram_mb": 0}
        
        if torch.cuda.is_available():
            stats["gpu_peak_mb"] = torch.cuda.max_memory_allocated() / (1024**2)
            stats["gpu_reserved_mb"] = torch.cuda.memory_reserved() / (1024**2)
        
        try:
            import psutil
            stats["ram_mb"] = psutil.Process().memory_info().rss / (1024**2)
        except ImportError:
            pass
        
        return stats
    
    def _clear_memory(self):
        """Clear GPU memory."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        gc.collect()
    
    def benchmark_generation(self, prompt: str, name: str = "generation") -> BenchmarkMetrics:
        """Benchmark text generation speed."""
        if self.model is None or self.tokenizer is None:
            self.setup()
            
        result = BenchmarkMetrics(
            name=name, 
            category="generation", 
            model_name=Path(self.config.model_path).name
        )
        
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            result.input_tokens = inputs["input_ids"].shape[1]
            
            self._clear_memory()
            
            # Warmup
            for _ in range(self.config.warmup_runs):
                with torch.no_grad():
                    self.model.generate(**inputs, max_new_tokens=10, do_sample=False)
            
            # Benchmark runs
            times = []
            first_token_times = []
            output_tokens_list = []
            
            for _ in range(self.config.benchmark_runs):
                self._clear_memory()
                
                start = time.perf_counter()
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=self.config.max_new_tokens,
                        do_sample=False,
                        return_dict_in_generate=True,
                        output_scores=True,
                    )
                end = time.perf_counter()
                
                times.append(end - start)
                new_tokens = outputs.sequences.shape[1] - inputs["input_ids"].shape[1]
                output_tokens_list.append(new_tokens)
                
                # Estimate first token time
                if hasattr(outputs, 'scores') and len(outputs.scores) > 0:
                    first_token_times.append(times[-1] / max(new_tokens, 1))
            
            result.total_time_s = sum(times) / len(times)
            result.output_tokens = int(sum(output_tokens_list) / len(output_tokens_list))
            result.total_tokens = result.input_tokens + result.output_tokens
            result.tokens_per_second = result.output_tokens / result.total_time_s
            result.latency_ms = result.total_time_s * 1000
            result.first_token_time_s = sum(first_token_times) / len(first_token_times) if first_token_times else 0
            
            memory = self._get_memory_stats()
            result.gpu_peak_mb = memory["gpu_peak_mb"]
            result.gpu_reserved_mb = memory["gpu_reserved_mb"]
            result.ram_mb = memory["ram_mb"]
            
        except Exception as e:
            result.success = False
            result.error = str(e)
        
        # Log to unified tracker
        self.tracker.log_benchmark(result)
        self.local_results.append(result)
        return result
    
    def benchmark_perplexity(self, text: str, name: str = "perplexity") -> BenchmarkMetrics:
        """Benchmark model perplexity (accuracy)."""
        if self.model is None or self.tokenizer is None:
            self.setup()

        result = BenchmarkMetrics(
            name=name, 
            category="accuracy", 
            model_name=Path(self.config.model_path).name
        )
        
        try:
            inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
            result.input_tokens = inputs["input_ids"].shape[1]
            
            with torch.no_grad():
                outputs = self.model(**inputs, labels=inputs["input_ids"])
            
            result.loss = outputs.loss.item()
            result.perplexity = torch.exp(outputs.loss).item()
            result.success = not (torch.isnan(outputs.loss) or torch.isinf(outputs.loss))
            
        except Exception as e:
            result.success = False
            result.error = str(e)
            
        # Log to unified tracker
        self.tracker.log_benchmark(result)
        self.local_results.append(result)
        return result

    def get_sample_prompts(self) -> List[str]:
        """Get sample prompts from ALL_DATASETS if available, else defaults."""
        prompts = []
        
        # Try to load a few samples from configured datasets
        try:
            from src.data.universal_loader import load_dataset_universal
            
            # Iterate through ALL categories and ALL paths
            # We shuffle categories to get a good mix
            import random
            categories = list(ALL_DATASETS.keys())
            random.shuffle(categories)
            
            for category in categories:
                paths = ALL_DATASETS[category]
                for path in paths:
                    if not Path(path).exists():
                         continue
                         
                    try:
                        # Take 1 sample from each to avoid overwhelming
                        res = load_dataset_universal(path, sample_size=1)
                        if res.dataset:
                             sample = res.dataset[0]
                             
                             # Handle standard 'messages' format vs raw 'text'
                             if isinstance(sample, dict):
                                 if "messages" in sample:
                                     # Get first user message content
                                     for msg in sample["messages"]:
                                         if msg.get("role") == "user":
                                             prompts.append(msg.get("content", ""))
                                             break
                                 elif "text" in sample:
                                     prompts.append(sample["text"])
                                 elif "instruction" in sample:
                                     prompts.append(sample["instruction"])
                                 elif "query" in sample:
                                     prompts.append(sample["query"])
                                 else:
                                     # Fallback: first value that is a string
                                     for v in sample.values():
                                         if isinstance(v, str) and len(v) > 5:
                                             prompts.append(v)
                                             break
                             elif isinstance(sample, str):
                                 prompts.append(sample)
                                 
                        if len(prompts) >= 10: # limit to 10 real samples
                            break
                    except Exception as e:
                        print(f"Failed to sample {path}: {e}")
                
                if len(prompts) >= 10:
                    break
                        
        except Exception as e:
            print(f"Warning: Could not load real datasets: {e}")
        
        # Filter empty and duplicates
        prompts = list(set([p for p in prompts if p and len(p) > 2]))
        
        # Fallback prompts if absolutely nothing loaded or too few
        if len(prompts) < 3:
            fallbacks = [
                "Explain quantum computing in simple terms.",
                "Write a poem about a robot learning to love.",
                "What are the main differences between Python and C++?",
                "Solve this math problem: If x + 2 = 10, what is x?",
                "Translate 'Hello world' to French, Spanish, and German.",
            ]
            prompts.extend(fallbacks)
            
        return prompts[:15] # Return a manageable set

    def run_all(self):
        """Run all benchmarks."""
        self.setup()
        
        print("\n" + "="*60)
        print("RUNNING COMPREHENSIVE BENCHMARKS")
        print("="*60)
        
        prompts = self.get_sample_prompts()
        
        # 1. Generation Benchmark
        print(f"\n[1/2] Generation Benchmarks ({len(prompts)} prompts)")
        for i, prompt in enumerate(prompts):
            print(f"  Running prompt {i+1}/{len(prompts)} (len={len(prompt)} chars)...")
            result = self.benchmark_generation(prompt, f"gen_sample_{i+1}")
            print(f"    -> {result.tokens_per_second:.1f} tok/s, {result.latency_ms:.0f}ms latency")
            
        # 2. Perplexity Benchmark
        print(f"\n[2/2] Perplexity Benchmarks")
        for i, prompt in enumerate(prompts):
             print(f"  Running prompt {i+1}/{len(prompts)}...")
             result = self.benchmark_perplexity(prompt, f"ppl_sample_{i+1}")
             print(f"    -> PPL={result.perplexity:.2f}")

    def print_summary(self):
        """Print benchmark summary."""
        print("\n" + "="*60)
        print("BENCHMARK SUMMARY")
        print("="*60)
        
        gen_results = [r for r in self.local_results if r.category == "generation" and r.success]
        
        if gen_results:
            avg_tps = sum(r.tokens_per_second for r in gen_results) / len(gen_results)
            avg_latency = sum(r.latency_ms for r in gen_results) / len(gen_results)
            print(f"\nGeneration Performance:")
            print(f"  Average: {avg_tps:.1f} tokens/sec")
            print(f"  Latency: {avg_latency:.0f}ms")
        
        print(f"\nFull results saved to: {self.config.output_dir}/benchmark_metrics.csv")
        print("="*60)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Run comprehensive benchmarks")
    parser.add_argument("--model", default="/mnt/e/data/models/Qwen2.5-0.5B", help="Model path")
    parser.add_argument("--output-dir", default="results", help="Output directory")
    parser.add_argument("--runs", type=int, default=3, help="Number of benchmark runs")
    parser.add_argument("--max-tokens", type=int, default=100, help="Max tokens to generate")
    args = parser.parse_args()
    
    config = BenchmarkConfig(
        model_path=args.model,
        output_dir=args.output_dir,
        benchmark_runs=args.runs,
        max_new_tokens=args.max_tokens,
    )
    
    runner = BenchmarkRunner(config)
    runner.run_all()
    runner.print_summary()
    
    return 0


if __name__ == "__main__":
    exit(main())
