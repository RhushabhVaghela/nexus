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
    python src/benchmarks/benchmark_runner.py --model /path/to/model --output results/benchmark.csv
"""

import os
import sys
import time
import csv
import json
import gc
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from datetime import datetime

import torch

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class BenchmarkResult:
    """Result from a single benchmark run."""
    name: str
    category: str  # "generation", "training", "accuracy"
    
    # Timing metrics
    total_time_s: float = 0.0
    first_token_time_s: float = 0.0
    tokens_per_second: float = 0.0
    latency_ms: float = 0.0
    
    # Token metrics
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    
    # Accuracy metrics
    token_accuracy: float = 0.0
    perplexity: float = 0.0
    loss: float = 0.0
    
    # Memory metrics
    gpu_peak_mb: float = 0.0
    gpu_reserved_mb: float = 0.0
    ram_mb: float = 0.0
    
    # Status
    success: bool = True
    error: str = ""


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark run."""
    model_path: str
    output_path: str = "results/benchmark.csv"
    warmup_runs: int = 2
    benchmark_runs: int = 5
    max_new_tokens: int = 100
    batch_sizes: List[int] = field(default_factory=lambda: [1, 2, 4])


class BenchmarkRunner:
    """
    Runs comprehensive benchmarks on a model.
    
    Measures:
    - Generation speed (tokens/sec)
    - Latency (first token time)
    - Accuracy (token match, perplexity)
    - Memory usage
    """
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.results: List[BenchmarkResult] = []
        self.model = None
        self.tokenizer = None
        
    def setup(self):
        """Load model and tokenizer."""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        print(f"Loading model from {self.config.model_path}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_path,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
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
    
    def benchmark_generation(self, prompt: str, name: str = "generation") -> BenchmarkResult:
        """
        Benchmark text generation speed.
        
        Measures:
        - Tokens per second
        - First token latency
        - Total generation time
        """
        result = BenchmarkResult(name=name, category="generation")
        
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
                
                # Estimate first token time (rough - from first score)
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
        
        return result
    
    def benchmark_batch_generation(self, prompts: List[str], name: str = "batch_gen") -> BenchmarkResult:
        """Benchmark batch generation speed."""
        result = BenchmarkResult(name=f"{name}_batch{len(prompts)}", category="generation")
        
        try:
            inputs = self.tokenizer(prompts, return_tensors="pt", padding=True).to(self.model.device)
            result.input_tokens = inputs["input_ids"].shape[1] * len(prompts)
            
            self._clear_memory()
            
            start = time.perf_counter()
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_new_tokens,
                    do_sample=False,
                )
            end = time.perf_counter()
            
            result.total_time_s = end - start
            result.output_tokens = (outputs.shape[1] - inputs["input_ids"].shape[1]) * len(prompts)
            result.tokens_per_second = result.output_tokens / result.total_time_s
            result.latency_ms = result.total_time_s * 1000
            
            memory = self._get_memory_stats()
            result.gpu_peak_mb = memory["gpu_peak_mb"]
            
        except Exception as e:
            result.success = False
            result.error = str(e)
        
        return result
    
    def benchmark_perplexity(self, text: str, name: str = "perplexity") -> BenchmarkResult:
        """
        Benchmark perplexity (lower is better).
        """
        result = BenchmarkResult(name=name, category="accuracy")
        
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
        
        return result
    
    def benchmark_token_accuracy(
        self, 
        prompts: List[str], 
        expected_outputs: List[str],
        name: str = "accuracy"
    ) -> BenchmarkResult:
        """
        Benchmark token-level accuracy.
        """
        result = BenchmarkResult(name=name, category="accuracy")
        
        try:
            total_correct = 0
            total_tokens = 0
            
            for prompt, expected in zip(prompts, expected_outputs):
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=len(self.tokenizer.encode(expected)),
                        do_sample=False,
                    )
                
                generated = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
                
                # Token-level comparison
                gen_tokens = self.tokenizer.encode(generated)
                exp_tokens = self.tokenizer.encode(expected)
                
                min_len = min(len(gen_tokens), len(exp_tokens))
                correct = sum(1 for g, e in zip(gen_tokens[:min_len], exp_tokens[:min_len]) if g == e)
                
                total_correct += correct
                total_tokens += len(exp_tokens)
            
            result.token_accuracy = total_correct / total_tokens if total_tokens > 0 else 0
            result.output_tokens = total_tokens
            
        except Exception as e:
            result.success = False
            result.error = str(e)
        
        return result
    
    def run_all(self, test_prompts: List[str] = None) -> List[BenchmarkResult]:
        """Run all benchmarks."""
        if test_prompts is None:
            test_prompts = [
                "Explain the concept of machine learning in one sentence.",
                "What is the capital of France?",
                "Write a Python function to calculate factorial.",
            ]
        
        self.setup()
        
        print("\n" + "="*60)
        print("RUNNING BENCHMARKS")
        print("="*60)
        
        # 1. Single generation
        print("\n[1/4] Single Generation Benchmark")
        for i, prompt in enumerate(test_prompts[:3]):
            result = self.benchmark_generation(prompt, f"gen_prompt{i+1}")
            self.results.append(result)
            print(f"  {result.name}: {result.tokens_per_second:.1f} tok/s, {result.latency_ms:.0f}ms")
        
        # 2. Batch generation
        print("\n[2/4] Batch Generation Benchmark")
        for batch_size in self.config.batch_sizes:
            prompts = test_prompts[:batch_size]
            result = self.benchmark_batch_generation(prompts, "batch")
            self.results.append(result)
            print(f"  {result.name}: {result.tokens_per_second:.1f} tok/s, {result.latency_ms:.0f}ms")
        
        # 3. Perplexity
        print("\n[3/4] Perplexity Benchmark")
        for i, prompt in enumerate(test_prompts[:3]):
            result = self.benchmark_perplexity(prompt, f"ppl_prompt{i+1}")
            self.results.append(result)
            print(f"  {result.name}: PPL={result.perplexity:.2f}, Loss={result.loss:.4f}")
        
        # 4. Summary
        print("\n[4/4] Computing Summary Statistics")
        
        return self.results
    
    def export_csv(self, path: str = None):
        """Export results to CSV."""
        if path is None:
            path = self.config.output_path
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        fieldnames = [
            "timestamp", "name", "category",
            "total_time_s", "first_token_time_s", "tokens_per_second", "latency_ms",
            "input_tokens", "output_tokens", "total_tokens",
            "token_accuracy", "perplexity", "loss",
            "gpu_peak_mb", "gpu_reserved_mb", "ram_mb",
            "success", "error",
        ]
        
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for result in self.results:
                row = {
                    "timestamp": datetime.now().isoformat(),
                    "name": result.name,
                    "category": result.category,
                    "total_time_s": f"{result.total_time_s:.4f}",
                    "first_token_time_s": f"{result.first_token_time_s:.4f}",
                    "tokens_per_second": f"{result.tokens_per_second:.2f}",
                    "latency_ms": f"{result.latency_ms:.1f}",
                    "input_tokens": result.input_tokens,
                    "output_tokens": result.output_tokens,
                    "total_tokens": result.total_tokens,
                    "token_accuracy": f"{result.token_accuracy:.4f}",
                    "perplexity": f"{result.perplexity:.4f}",
                    "loss": f"{result.loss:.4f}",
                    "gpu_peak_mb": f"{result.gpu_peak_mb:.1f}",
                    "gpu_reserved_mb": f"{result.gpu_reserved_mb:.1f}",
                    "ram_mb": f"{result.ram_mb:.1f}",
                    "success": result.success,
                    "error": result.error[:200] if result.error else "",
                }
                writer.writerow(row)
        
        print(f"\nResults exported to: {path}")
        return path
    
    def print_summary(self):
        """Print benchmark summary."""
        print("\n" + "="*60)
        print("BENCHMARK SUMMARY")
        print("="*60)
        
        gen_results = [r for r in self.results if r.category == "generation" and r.success]
        acc_results = [r for r in self.results if r.category == "accuracy" and r.success]
        
        if gen_results:
            avg_tps = sum(r.tokens_per_second for r in gen_results) / len(gen_results)
            avg_latency = sum(r.latency_ms for r in gen_results) / len(gen_results)
            peak_gpu = max(r.gpu_peak_mb for r in gen_results)
            print(f"\nGeneration Performance:")
            print(f"  Average: {avg_tps:.1f} tokens/sec")
            print(f"  Latency: {avg_latency:.0f}ms")
            print(f"  Peak GPU: {peak_gpu:.0f}MB")
        
        if acc_results:
            avg_ppl = sum(r.perplexity for r in acc_results if r.perplexity > 0) / max(len([r for r in acc_results if r.perplexity > 0]), 1)
            print(f"\nAccuracy Metrics:")
            print(f"  Average Perplexity: {avg_ppl:.2f}")
        
        print("="*60)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Run comprehensive benchmarks")
    parser.add_argument("--model", default="/mnt/e/data/models/Qwen2.5-0.5B", help="Model path")
    parser.add_argument("--output", default="results/benchmark.csv", help="Output CSV path")
    parser.add_argument("--runs", type=int, default=5, help="Number of benchmark runs")
    parser.add_argument("--max-tokens", type=int, default=100, help="Max tokens to generate")
    args = parser.parse_args()
    
    config = BenchmarkConfig(
        model_path=args.model,
        output_path=args.output,
        benchmark_runs=args.runs,
        max_new_tokens=args.max_tokens,
    )
    
    runner = BenchmarkRunner(config)
    results = runner.run_all()
    runner.export_csv()
    runner.print_summary()
    
    return 0


if __name__ == "__main__":
    exit(main())
