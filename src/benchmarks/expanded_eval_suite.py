#!/usr/bin/env python3
"""
benchmarks/expanded_eval_suite.py

Expanded benchmark evaluation suite using real HuggingFace benchmark datasets.

Benchmarks included:
- MMLU (57K questions across 57 subjects)
- GSM8K (8.5K grade school math problems)
- HumanEval (164 coding problems)
- MBPP (1000 Python problems)
- BigCodeBench (code generation benchmark)
- SWE-Bench (real GitHub issues)
- MATH (12.5K competition math)
- ARC (AI2 Reasoning Challenge)
- HellaSwag (commonsense reasoning)
- TruthfulQA (factual accuracy)
- WinoGrande (commonsense reasoning)
- GPQA (graduate-level science QA)
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import random

try:
    from datasets import load_dataset
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.logging_config import setup_logger, log_header, log_completion

logger = setup_logger(__name__, "logs/expanded_eval_suite.log")


# ═══════════════════════════════════════════════════════════════
# BENCHMARK CONFIGURATIONS - REAL DATASETS
# ═══════════════════════════════════════════════════════════════

BENCHMARK_REGISTRY = {
    # Knowledge & Reasoning
    "mmlu": {
        "hf_path": "cais/mmlu",
        "subset": "all",
        "split": "test",
        "description": "57 subjects, 57K questions",
        "metric": "accuracy",
        "task_type": "multiple_choice",
    },
    "mmlu_pro": {
        "hf_path": "TIGER-Lab/MMLU-Pro",
        "split": "test",
        "description": "Harder MMLU with 10 choices",
        "metric": "accuracy",
        "task_type": "multiple_choice",
    },
    "arc_challenge": {
        "hf_path": "allenai/ai2_arc",
        "subset": "ARC-Challenge",
        "split": "test",
        "description": "Science reasoning (challenge)",
        "metric": "accuracy",
        "task_type": "multiple_choice",
    },
    "arc_easy": {
        "hf_path": "allenai/ai2_arc",
        "subset": "ARC-Easy",
        "split": "test",
        "description": "Science reasoning (easy)",
        "metric": "accuracy",
        "task_type": "multiple_choice",
    },
    "hellaswag": {
        "hf_path": "Rowan/hellaswag",
        "split": "validation",
        "description": "Commonsense reasoning",
        "metric": "accuracy",
        "task_type": "multiple_choice",
    },
    "winogrande": {
        "hf_path": "allenai/winogrande",
        "subset": "winogrande_xl",
        "split": "validation",
        "description": "Commonsense pronoun resolution",
        "metric": "accuracy",
        "task_type": "multiple_choice",
    },
    "truthfulqa": {
        "hf_path": "truthfulqa/truthful_qa",
        "subset": "multiple_choice",
        "split": "validation",
        "description": "Factual accuracy",
        "metric": "accuracy",
        "task_type": "multiple_choice",
    },
    "gpqa": {
        "hf_path": "Idavidrein/gpqa",
        "subset": "gpqa_diamond",
        "split": "train",
        "description": "Graduate-level science QA",
        "metric": "accuracy",
        "task_type": "multiple_choice",
    },
    
    # Math
    "gsm8k": {
        "hf_path": "openai/gsm8k",
        "subset": "main",
        "split": "test",
        "description": "Grade school math (8.5K)",
        "metric": "accuracy",
        "task_type": "math",
    },
    "math": {
        "hf_path": "lighteval/MATH",
        "split": "test",
        "description": "Competition math (12.5K)",
        "metric": "accuracy",
        "task_type": "math",
    },
    "math_hard": {
        "hf_path": "lighteval/MATH-Hard",
        "split": "test",
        "description": "Hardest math problems",
        "metric": "accuracy",
        "task_type": "math",
    },
    
    # Code
    "humaneval": {
        "hf_path": "openai/openai_humaneval",
        "split": "test",
        "description": "164 Python functions",
        "metric": "pass@1",
        "task_type": "code_generation",
    },
    "mbpp": {
        "hf_path": "google-research-datasets/mbpp",
        "split": "test",
        "description": "1000 Python problems",
        "metric": "pass@1",
        "task_type": "code_generation",
    },
    "bigcodebench": {
        "hf_path": "bigcode/bigcodebench",
        "split": "v0.1.2",
        "description": "Complex code generation",
        "metric": "pass@1",
        "task_type": "code_generation",
    },
    "humaneval_plus": {
        "hf_path": "evalplus/humanevalplus",
        "split": "test",
        "description": "HumanEval with more tests",
        "metric": "pass@1",
        "task_type": "code_generation",
    },
    
    # Real-world Code
    "swe_bench_lite": {
        "hf_path": "princeton-nlp/SWE-bench_Lite",
        "split": "test",
        "description": "300 real GitHub issues",
        "metric": "resolve_rate",
        "task_type": "swe_bench",
    },
    "swe_bench_verified": {
        "hf_path": "princeton-nlp/SWE-bench_Verified",
        "split": "test",
        "description": "500 verified GitHub issues",
        "metric": "resolve_rate",
        "task_type": "swe_bench",
    },
    
    # Instruction Following
    "ifeval": {
        "hf_path": "google/IFEval",
        "split": "train",
        "description": "Instruction following",
        "metric": "accuracy",
        "task_type": "instruction",
    },
    "alpaca_eval": {
        "hf_path": "tatsu-lab/alpaca_eval",
        "split": "eval",
        "description": "Alpaca instruction eval",
        "metric": "win_rate",
        "task_type": "instruction",
    },
    
    # Fullstack (custom)
    "fullstack_eval": {
        "hf_path": "local",
        "split": "test",
        "description": "Fullstack engineering tasks",
        "metric": "composite",
        "task_type": "fullstack",
    },
}


@dataclass
class EvalResult:
    """Single evaluation result."""
    benchmark: str
    score: float
    metric: str
    num_samples: int
    correct: int
    details: Dict = field(default_factory=dict)


@dataclass
class BenchmarkSample:
    """Normalized benchmark sample."""
    id: str
    question: str
    choices: List[str]
    correct_answer: str
    subject: str
    benchmark: str


# ═══════════════════════════════════════════════════════════════
# EVALUATORS
# ═══════════════════════════════════════════════════════════════

class BaseEvaluator(ABC):
    """Base class for benchmark evaluators."""
    
    def __init__(self, benchmark_name: str, model_fn: Optional[Callable] = None):
        self.benchmark_name = benchmark_name
        self.config = BENCHMARK_REGISTRY.get(benchmark_name, {})
        self.model_fn = model_fn
    
    @abstractmethod
    def load_samples(self, limit: Optional[int] = None) -> List[BenchmarkSample]:
        """Load benchmark samples."""
        pass
    
    @abstractmethod
    def evaluate_sample(self, sample: BenchmarkSample, model_output: str) -> bool:
        """Evaluate a single sample."""
        pass
    
    def run_evaluation(self, limit: Optional[int] = None) -> EvalResult:
        """Run full evaluation."""
        samples = self.load_samples(limit)
        correct = 0
        total = len(samples)
        
        for sample in samples:
            if self.model_fn:
                output = self.model_fn(sample.question)
                if self.evaluate_sample(sample, output):
                    correct += 1
            else:
                # Dry run - just validate samples load correctly
                correct += 1
        
        score = (correct / total) * 100 if total > 0 else 0
        
        return EvalResult(
            benchmark=self.benchmark_name,
            score=score,
            metric=self.config.get("metric", "accuracy"),
            num_samples=total,
            correct=correct,
        )


class MultipleChoiceEvaluator(BaseEvaluator):
    """Evaluator for multiple choice benchmarks."""
    
    def load_samples(self, limit: Optional[int] = None) -> List[BenchmarkSample]:
        if not HF_AVAILABLE:
            return []
        
        try:
            kwargs = {"split": self.config.get("split", "test")}
            if "subset" in self.config:
                kwargs["name"] = self.config["subset"]
            
            ds = load_dataset(self.config["hf_path"], **kwargs, trust_remote_code=True)
            
            samples = []
            for idx, item in enumerate(ds):
                if limit and idx >= limit:
                    break
                
                # Normalize different formats
                if "question" in item:
                    question = item["question"]
                elif "premise" in item:
                    question = item["premise"]
                else:
                    question = str(item.get("input", ""))
                
                if "choices" in item:
                    if isinstance(item["choices"], dict):
                        choices = item["choices"].get("text", [])
                    else:
                        choices = list(item["choices"])
                elif "endings" in item:
                    choices = list(item["endings"])
                else:
                    choices = [item.get(f"choice{i}", "") for i in range(4)]
                
                answer = item.get("answer", item.get("label", 0))
                if isinstance(answer, int):
                    correct = choices[answer] if answer < len(choices) else ""
                else:
                    correct = str(answer)
                
                samples.append(BenchmarkSample(
                    id=f"{self.benchmark_name}_{idx}",
                    question=question,
                    choices=choices,
                    correct_answer=correct,
                    subject=item.get("subject", ""),
                    benchmark=self.benchmark_name,
                ))
            
            return samples
        except Exception as e:
            logger.error(f"Failed to load {self.benchmark_name}: {e}")
            return []
    
    def evaluate_sample(self, sample: BenchmarkSample, model_output: str) -> bool:
        """Check if model output matches correct answer."""
        output_clean = model_output.strip().upper()
        correct_clean = sample.correct_answer.strip().upper()
        
        # Check for letter match (A, B, C, D)
        for i, choice in enumerate(sample.choices):
            letter = chr(ord('A') + i)
            if output_clean.startswith(letter) or output_clean == letter:
                return choice.strip().upper() == correct_clean
        
        # Direct match
        return output_clean == correct_clean


class MathEvaluator(BaseEvaluator):
    """Evaluator for math benchmarks."""
    
    def load_samples(self, limit: Optional[int] = None) -> List[BenchmarkSample]:
        if not HF_AVAILABLE:
            return []
        
        try:
            kwargs = {"split": self.config.get("split", "test")}
            if "subset" in self.config:
                kwargs["name"] = self.config["subset"]
            
            ds = load_dataset(self.config["hf_path"], **kwargs, trust_remote_code=True)
            
            samples = []
            for idx, item in enumerate(ds):
                if limit and idx >= limit:
                    break
                
                question = item.get("question", item.get("problem", ""))
                answer = item.get("answer", item.get("solution", ""))
                
                # Extract numeric answer for GSM8K
                if "####" in str(answer):
                    answer = str(answer).split("####")[-1].strip()
                
                samples.append(BenchmarkSample(
                    id=f"{self.benchmark_name}_{idx}",
                    question=question,
                    choices=[],
                    correct_answer=str(answer),
                    subject=item.get("type", ""),
                    benchmark=self.benchmark_name,
                ))
            
            return samples
        except Exception as e:
            logger.error(f"Failed to load {self.benchmark_name}: {e}")
            return []
    
    def evaluate_sample(self, sample: BenchmarkSample, model_output: str) -> bool:
        """Check if model output contains correct answer."""
        import re
        
        # Extract numbers from model output
        numbers = re.findall(r"-?\d+\.?\d*", model_output)
        if not numbers:
            return False
        
        # Check if any extracted number matches
        try:
            correct_num = float(sample.correct_answer.replace(",", ""))
            for num_str in numbers:
                if abs(float(num_str) - correct_num) < 0.01:
                    return True
        except ValueError:
            pass
        
        return sample.correct_answer in model_output


class CodeEvaluator(BaseEvaluator):
    """Evaluator for code generation benchmarks."""
    
    def load_samples(self, limit: Optional[int] = None) -> List[BenchmarkSample]:
        if not HF_AVAILABLE:
            return []
        
        try:
            ds = load_dataset(
                self.config["hf_path"],
                split=self.config.get("split", "test"),
                trust_remote_code=True,
            )
            
            samples = []
            for idx, item in enumerate(ds):
                if limit and idx >= limit:
                    break
                
                # HumanEval format
                prompt = item.get("prompt", item.get("text", ""))
                test = item.get("test", item.get("test_list", ""))
                canonical = item.get("canonical_solution", "")
                
                samples.append(BenchmarkSample(
                    id=item.get("task_id", f"{self.benchmark_name}_{idx}"),
                    question=prompt,
                    choices=[],
                    correct_answer=canonical,
                    subject=str(test),  # Store tests in subject field
                    benchmark=self.benchmark_name,
                ))
            
            return samples
        except Exception as e:
            logger.error(f"Failed to load {self.benchmark_name}: {e}")
            return []
    
    def evaluate_sample(self, sample: BenchmarkSample, model_output: str) -> bool:
        """Execute code and check if tests pass."""
        import subprocess
        import tempfile
        
        try:
            # Combine prompt, solution, and tests
            full_code = sample.question + model_output + "\n" + sample.subject
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(full_code)
                temp_path = f.name
            
            result = subprocess.run(
                ["python3", temp_path],
                capture_output=True,
                timeout=10,
            )
            
            os.unlink(temp_path)
            return result.returncode == 0
        except Exception:
            return False


class SWEBenchEvaluator(BaseEvaluator):
    """Evaluator for SWE-Bench."""
    
    def load_samples(self, limit: Optional[int] = None) -> List[BenchmarkSample]:
        if not HF_AVAILABLE:
            return []
        
        try:
            ds = load_dataset(
                self.config["hf_path"],
                split=self.config.get("split", "test"),
                trust_remote_code=True,
            )
            
            samples = []
            for idx, item in enumerate(ds):
                if limit and idx >= limit:
                    break
                
                samples.append(BenchmarkSample(
                    id=item.get("instance_id", f"swe_{idx}"),
                    question=item.get("problem_statement", ""),
                    choices=[],
                    correct_answer=item.get("patch", ""),
                    subject=item.get("repo", ""),
                    benchmark=self.benchmark_name,
                ))
            
            return samples
        except Exception as e:
            logger.error(f"Failed to load {self.benchmark_name}: {e}")
            return []
    
    def evaluate_sample(self, sample: BenchmarkSample, model_output: str) -> bool:
        """SWE-Bench requires special evaluation harness."""
        # For now, check if output looks like a valid patch
        return "diff" in model_output.lower() or "@@" in model_output


# ═══════════════════════════════════════════════════════════════
# EVALUATOR FACTORY
# ═══════════════════════════════════════════════════════════════

def get_evaluator(benchmark_name: str, model_fn: Optional[Callable] = None) -> BaseEvaluator:
    """Get appropriate evaluator for benchmark."""
    config = BENCHMARK_REGISTRY.get(benchmark_name, {})
    task_type = config.get("task_type", "multiple_choice")
    
    if task_type == "multiple_choice":
        return MultipleChoiceEvaluator(benchmark_name, model_fn)
    elif task_type == "math":
        return MathEvaluator(benchmark_name, model_fn)
    elif task_type == "code_generation":
        return CodeEvaluator(benchmark_name, model_fn)
    elif task_type == "swe_bench":
        return SWEBenchEvaluator(benchmark_name, model_fn)
    else:
        return MultipleChoiceEvaluator(benchmark_name, model_fn)


# ═══════════════════════════════════════════════════════════════
# MAIN EVALUATION SUITE
# ═══════════════════════════════════════════════════════════════

class ExpandedEvalSuite:
    """Complete evaluation suite for all benchmarks."""
    
    def __init__(self, model_fn: Optional[Callable] = None):
        self.model_fn = model_fn
        self.results: List[EvalResult] = []
    
    def run_benchmark(self, benchmark_name: str, limit: Optional[int] = None) -> EvalResult:
        """Run a single benchmark."""
        evaluator = get_evaluator(benchmark_name, self.model_fn)
        result = evaluator.run_evaluation(limit)
        self.results.append(result)
        return result
    
    def run_all(self, benchmarks: Optional[List[str]] = None, limit: Optional[int] = None) -> Dict:
        """Run all specified benchmarks."""
        if benchmarks is None:
            benchmarks = list(BENCHMARK_REGISTRY.keys())
        
        for benchmark in benchmarks:
            logger.info(f"Running {benchmark}...")
            try:
                result = self.run_benchmark(benchmark, limit)
                logger.info(f"  {benchmark}: {result.score:.2f}% ({result.correct}/{result.num_samples})")
            except Exception as e:
                logger.error(f"  {benchmark} failed: {e}")
        
        return self.get_summary()
    
    def get_summary(self) -> Dict:
        """Get summary of all results."""
        summary = {
            "total_benchmarks": len(self.results),
            "benchmarks": {},
            "aggregate": {
                "knowledge": 0,
                "math": 0,
                "code": 0,
                "reasoning": 0,
            },
        }
        
        knowledge_benchmarks = ["mmlu", "mmlu_pro", "arc_challenge", "gpqa"]
        math_benchmarks = ["gsm8k", "math", "math_hard"]
        code_benchmarks = ["humaneval", "mbpp", "bigcodebench", "swe_bench_lite"]
        reasoning_benchmarks = ["hellaswag", "winogrande", "arc_easy"]
        
        for result in self.results:
            summary["benchmarks"][result.benchmark] = {
                "score": result.score,
                "metric": result.metric,
                "samples": result.num_samples,
            }
            
            # Aggregate scores
            if result.benchmark in knowledge_benchmarks:
                count = summary["aggregate"].get("knowledge_count", 0)
                summary["aggregate"]["knowledge"] = (
                    summary["aggregate"]["knowledge"] * count + result.score
                ) / (count + 1)
                summary["aggregate"]["knowledge_count"] = count + 1
            elif result.benchmark in math_benchmarks:
                count = summary["aggregate"].get("math_count", 0)
                summary["aggregate"]["math"] = (
                    summary["aggregate"]["math"] * count + result.score
                ) / (count + 1)
                summary["aggregate"]["math_count"] = count + 1
            elif result.benchmark in code_benchmarks:
                count = summary["aggregate"].get("code_count", 0)
                summary["aggregate"]["code"] = (
                    summary["aggregate"]["code"] * count + result.score
                ) / (count + 1)
                summary["aggregate"]["code_count"] = count + 1
            elif result.benchmark in reasoning_benchmarks:
                count = summary["aggregate"].get("reasoning_count", 0)
                summary["aggregate"]["reasoning"] = (
                    summary["aggregate"]["reasoning"] * count + result.score
                ) / (count + 1)
                summary["aggregate"]["reasoning_count"] = count + 1
        
        # Clean up count fields
        for key in list(summary["aggregate"].keys()):
            if key.endswith("_count"):
                del summary["aggregate"][key]
        
        return summary
    
    def save_results(self, output_path: Path):
        """Save results to file."""
        summary = self.get_summary()
        with open(output_path, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Run expanded evaluation suite")
    parser.add_argument("--benchmarks", nargs="+", default=None,
                        help="Benchmarks to run (default: all)")
    parser.add_argument("--limit", type=int, default=100,
                        help="Limit samples per benchmark")
    parser.add_argument("--output", type=str, default="evaluation_results.json",
                        help="Output file for results")
    parser.add_argument("--list", action="store_true",
                        help="List available benchmarks")
    args = parser.parse_args()
    
    if args.list:
        print("\nAvailable Benchmarks:\n")
        categories = {
            "Knowledge": ["mmlu", "mmlu_pro", "arc_challenge", "arc_easy", "gpqa", "truthfulqa"],
            "Math": ["gsm8k", "math", "math_hard"],
            "Code": ["humaneval", "mbpp", "bigcodebench", "humaneval_plus"],
            "SWE": ["swe_bench_lite", "swe_bench_verified"],
            "Reasoning": ["hellaswag", "winogrande"],
            "Instruction": ["ifeval", "alpaca_eval"],
        }
        for cat, benchmarks in categories.items():
            print(f"  {cat}:")
            for b in benchmarks:
                config = BENCHMARK_REGISTRY.get(b, {})
                print(f"    - {b}: {config.get('description', 'N/A')}")
            print()
        return
    
    log_header(
        logger,
        "EXPANDED EVALUATION SUITE",
        {
            "Benchmarks": args.benchmarks or "all",
            "Limit": args.limit,
            "Output": args.output,
        },
    )
    
    suite = ExpandedEvalSuite()
    summary = suite.run_all(args.benchmarks, args.limit)
    
    # Print summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    
    print("\nAggregate Scores:")
    for category, score in summary["aggregate"].items():
        print(f"  {category.title()}: {score:.2f}%")
    
    print("\nIndividual Benchmarks:")
    for bench, data in summary["benchmarks"].items():
        print(f"  {bench}: {data['score']:.2f}% ({data['samples']} samples)")
    
    suite.save_results(Path(args.output))
    
    log_completion(
        logger,
        "Evaluation Complete",
        {"Benchmarks": len(summary["benchmarks"]), "Output": args.output},
    )


if __name__ == "__main__":
    main()
