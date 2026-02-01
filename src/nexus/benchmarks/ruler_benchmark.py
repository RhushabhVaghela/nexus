#!/usr/bin/env python3
"""
RULER Benchmark Runner

Evaluates true context length of LLMs using RULER tasks.

Usage:
    python -m src.benchmarks.ruler_benchmark --model /path/to/model --lengths 4096,8192,16384,32768

Based on NVIDIA RULER (COLM 2024).
"""

import argparse
import json
import time
import logging
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple

import torch

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.benchmarks.ruler_tasks import (
    TaskConfig, TaskSample, RULERTask, TaskCategory,
    get_task, get_all_tasks, RULER_TASKS
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RULERConfig:
    """Configuration for RULER benchmark."""
    model_path: str = ""
    context_lengths: List[int] = field(default_factory=lambda: [4096, 8192, 16384, 32768])
    samples_per_task: int = 20
    tasks: List[str] = field(default_factory=lambda: list(RULER_TASKS.keys()))
    accuracy_threshold: float = 0.7  # Threshold for "effective" context
    device: str = "auto"
    output_path: Optional[str] = None
    max_new_tokens: int = 100
    temperature: float = 0.0  # Greedy for consistency
    seed: int = 42


@dataclass
class TaskResult:
    """Results for a single task at a specific context length."""
    task_name: str
    context_length: int
    accuracy: float
    avg_latency_ms: float
    samples_evaluated: int
    correct_count: int
    category: str


@dataclass
class RULERResult:
    """Complete RULER benchmark results."""
    model_path: str
    task_results: List[TaskResult]
    effective_context: int  # Longest context with >threshold accuracy
    overall_scores: Dict[int, float]  # context_length -> overall accuracy
    category_scores: Dict[str, Dict[int, float]]  # category -> {length -> accuracy}
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_path": self.model_path,
            "effective_context": self.effective_context,
            "overall_scores": self.overall_scores,
            "category_scores": self.category_scores,
            "task_results": [
                {
                    "task": r.task_name,
                    "context_length": r.context_length,
                    "accuracy": r.accuracy,
                    "avg_latency_ms": r.avg_latency_ms,
                }
                for r in self.task_results
            ]
        }


class RULERBenchmark:
    """
    RULER Benchmark Runner.
    
    Evaluates true context length by running synthetic tasks that require
    genuine long-context understanding.
    """
    
    def __init__(self, config: RULERConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.device = None
    
    def setup(self):
        """Load model and tokenizer."""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        logger.info(f"Loading model from {self.config.model_path}")
        
        # Determine device
        if self.config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.config.device)
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_path,
            trust_remote_code=True,
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_path,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
        self.model.eval()
        
        logger.info(f"Model loaded on {self.device}")
    
    def generate_response(self, prompt: str) -> Tuple[str, float]:
        """
        Generate model response for a prompt.
        
        Returns: (response, latency_ms)
        """
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.model.config.max_position_embeddings,
        ).to(self.model.device)
        
        start_time = time.perf_counter()
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                do_sample=self.config.temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        
        latency_ms = (time.perf_counter() - start_time) * 1000
        
        # Decode only new tokens
        response = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True,
        )
        
        return response.strip(), latency_ms
    
    def format_prompt(self, sample: TaskSample) -> str:
        """Format a sample into a model prompt."""
        prompt = f"""Context:
{sample.context}

Question: {sample.question}

Answer (be brief and precise):"""
        return prompt
    
    def evaluate_task(
        self,
        task: RULERTask,
        context_length: int,
        num_samples: int,
    ) -> TaskResult:
        """Evaluate a single task at a specific context length."""
        config = TaskConfig(
            context_length=context_length,
            num_samples=num_samples,
            seed=self.config.seed,
        )
        
        task_instance = type(task)(config)
        samples = task_instance.generate_samples(num_samples)
        
        correct = 0
        total_latency = 0.0
        
        for sample in samples:
            prompt = self.format_prompt(sample)
            
            try:
                response, latency = self.generate_response(prompt)
                total_latency += latency
                
                is_correct, _ = task_instance.evaluate_response(response, sample.expected_answer)
                if is_correct:
                    correct += 1
                    
            except Exception as e:
                logger.warning(f"Error on sample: {e}")
                continue
        
        accuracy = correct / num_samples if num_samples > 0 else 0.0
        avg_latency = total_latency / num_samples if num_samples > 0 else 0.0
        
        return TaskResult(
            task_name=task.name,
            context_length=context_length,
            accuracy=accuracy,
            avg_latency_ms=avg_latency,
            samples_evaluated=num_samples,
            correct_count=correct,
            category=task.category.value,
        )
    
    def run(self) -> RULERResult:
        """Run the complete RULER benchmark."""
        all_results: List[TaskResult] = []
        
        # Get tasks to evaluate
        tasks = [get_task(name) for name in self.config.tasks]
        
        logger.info(f"Running RULER with {len(tasks)} tasks at {len(self.config.context_lengths)} context lengths")
        
        for ctx_len in self.config.context_lengths:
            logger.info(f"\n=== Context Length: {ctx_len:,} ===")
            
            for task in tasks:
                logger.info(f"  Evaluating: {task.name}")
                
                result = self.evaluate_task(
                    task,
                    ctx_len,
                    self.config.samples_per_task,
                )
                
                all_results.append(result)
                logger.info(f"    Accuracy: {result.accuracy:.1%} | Latency: {result.avg_latency_ms:.0f}ms")
        
        # Compute aggregated scores
        overall_scores: Dict[int, float] = {}
        category_scores: Dict[str, Dict[int, float]] = {}
        
        for ctx_len in self.config.context_lengths:
            ctx_results = [r for r in all_results if r.context_length == ctx_len]
            overall_scores[ctx_len] = sum(r.accuracy for r in ctx_results) / len(ctx_results)
            
            for category in TaskCategory:
                cat_results = [r for r in ctx_results if r.category == category.value]
                if cat_results:
                    if category.value not in category_scores:
                        category_scores[category.value] = {}
                    category_scores[category.value][ctx_len] = (
                        sum(r.accuracy for r in cat_results) / len(cat_results)
                    )
        
        # Find effective context (longest with accuracy > threshold)
        effective_context = 0
        for ctx_len in sorted(self.config.context_lengths):
            if overall_scores[ctx_len] >= self.config.accuracy_threshold:
                effective_context = ctx_len
            else:
                break
        
        return RULERResult(
            model_path=self.config.model_path,
            task_results=all_results,
            effective_context=effective_context,
            overall_scores=overall_scores,
            category_scores=category_scores,
        )
    
    def print_results(self, result: RULERResult):
        """Print formatted results."""
        print("\n" + "="*70)
        print("      RULER BENCHMARK RESULTS")
        print("="*70)
        print(f"Model: {result.model_path}")
        print(f"Accuracy Threshold: {self.config.accuracy_threshold:.0%}")
        print()
        
        # Overall scores table
        print("Overall Accuracy by Context Length:")
        print("-" * 50)
        print(f"{'Context':>10} | {'Accuracy':>10} | {'Status':>15}")
        print("-" * 50)
        
        for ctx_len in sorted(result.overall_scores.keys()):
            acc = result.overall_scores[ctx_len]
            status = "✅ PASS" if acc >= self.config.accuracy_threshold else "❌ DEGRADED"
            print(f"{ctx_len:>10,} | {acc:>10.1%} | {status:>15}")
        
        print("-" * 50)
        
        # Category breakdown
        print("\nAccuracy by Category:")
        print("-" * 70)
        
        # Header
        header = f"{'Category':>15}"
        for ctx_len in sorted(self.config.context_lengths):
            header += f" | {ctx_len//1000}K"
        print(header)
        print("-" * 70)
        
        for category in result.category_scores:
            row = f"{category:>15}"
            for ctx_len in sorted(self.config.context_lengths):
                acc = result.category_scores[category].get(ctx_len, 0)
                row += f" | {acc:.0%}"
            print(row)
        
        print("-" * 70)
        
        # Summary
        print(f"\n{'='*70}")
        print(f"EFFECTIVE CONTEXT LENGTH: {result.effective_context:,} tokens")
        print(f"(Longest context with >{self.config.accuracy_threshold:.0%} accuracy)")
        print("="*70)


def run_ruler_benchmark(
    model_path: str,
    context_lengths: List[int] = None,
    samples_per_task: int = 20,
    output_path: Optional[str] = None,
) -> RULERResult:
    """Convenience function to run RULER benchmark."""
    config = RULERConfig(
        model_path=model_path,
        context_lengths=context_lengths or [4096, 8192, 16384, 32768],
        samples_per_task=samples_per_task,
        output_path=output_path,
    )
    
    benchmark = RULERBenchmark(config)
    benchmark.setup()
    result = benchmark.run()
    benchmark.print_results(result)
    
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)
        logger.info(f"Results saved to {output_path}")
    
    return result


def main():
    parser = argparse.ArgumentParser(description="RULER Benchmark for LLM Context Length")
    parser.add_argument("--model", type=str, required=True, help="Path to model")
    parser.add_argument("--lengths", type=str, default="4096,8192,16384,32768",
                        help="Comma-separated context lengths to test")
    parser.add_argument("--samples", type=int, default=20,
                        help="Samples per task (default: 20)")
    parser.add_argument("--tasks", type=str, default=None,
                        help="Comma-separated task names (default: all)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON file path")
    parser.add_argument("--threshold", type=float, default=0.7,
                        help="Accuracy threshold for effective context (default: 0.7)")
    
    args = parser.parse_args()
    
    lengths = [int(x) for x in args.lengths.split(",")]
    tasks = args.tasks.split(",") if args.tasks else list(RULER_TASKS.keys())
    
    config = RULERConfig(
        model_path=args.model,
        context_lengths=lengths,
        samples_per_task=args.samples,
        tasks=tasks,
        accuracy_threshold=args.threshold,
        output_path=args.output,
    )
    
    benchmark = RULERBenchmark(config)
    
    try:
        benchmark.setup()
    except Exception as e:
        print(f"Failed to load model: {e}")
        print("\nUsage:")
        print("  python -m src.benchmarks.ruler_benchmark --model /mnt/e/data/models/Qwen2.5-0.5B")
        return
    
    result = benchmark.run()
    benchmark.print_results(result)
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)
        logger.info(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
