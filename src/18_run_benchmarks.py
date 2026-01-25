#!/usr/bin/env python3
"""
13_run_benchmarks.py
Run comprehensive benchmarks on the trained model.
Evaluates against downloaded benchmark datasets.
"""

import os
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, List
import time

# Globals to be initialized in main()
logger = None
FastLanguageModel = None # Will be set by check_env if successful
torch = None # Will be set by check_env if successful
tqdm = None # Will be set by check_env if successful

def check_env():
    """Verify environment dependencies and import conditional libraries."""
    global FastLanguageModel, torch, tqdm # Declare globals to be modified
    
    try:
        import torch as _torch
        torch = _torch
    except ImportError:
        print("[ERROR] Missing dependency: torch")
        return False

    try:
        from tqdm import tqdm as _tqdm
        tqdm = _tqdm
    except ImportError:
        print("[ERROR] Missing dependency: tqdm")
        return False

    try:
        from unsloth import FastLanguageModel as _FastLanguageModel
        FastLanguageModel = _FastLanguageModel
    except ImportError:
        print("[ERROR] Missing dependency: unsloth")
        return False
        
    if os.environ.get("CONDA_DEFAULT_ENV") != "nexus":
        print("[ERROR] Must be run in 'nexus' conda environment.")
        return False
    return True


class BenchmarkRunner:
    """Run benchmarks against a trained model."""
    
    def __init__(self, model_path: str, benchmark_dir: str = "data/benchmarks"):
        self.model_path = model_path
        self.benchmark_dir = Path(benchmark_dir)
        self.results = {}
        self.model = None
        self.tokenizer = None
        
    def load_model(self):
        if not FastLanguageModel:
            logger.error("❌ Unsloth not installed")
            return False
            
        try:
            logger.info(f"📦 Loading model: {self.model_path}")
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=self.model_path,
                max_seq_length=4096,
                load_in_4bit=True,
                dtype=None
            )
            FastLanguageModel.for_inference(self.model)
            return True
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            return False

    def run_all(self) -> Dict[str, Any]:
        """Run all available benchmarks."""
        logger.info("="*60)
        logger.info("🎯 RUNNING COMPREHENSIVE BENCHMARKS")
        logger.info(f"   Model: {self.model_path}")
        logger.info("="*60)
        
        if not self.load_model():
            logger.warning("Thinking Mode: Running in dry-run/mock mode")
        
        # Check for benchmark datasets
        benchmarks = list(self.benchmark_dir.glob("*.jsonl"))
        if not benchmarks:
            logger.warning(f"No benchmark files found in {self.benchmark_dir}")
            logger.info("Run 05_download_benchmarks.py first to download datasets.")
            return {}
        
        for bench_file in benchmarks:
            bench_name = bench_file.stem
            logger.info(f"\n📊 Running benchmark: {bench_name}")
            
            try:
                result = self._run_benchmark(bench_file)
                self.results[bench_name] = result
                logger.info(f"   ✓ Score: {result.get('score', 0):.2%}")
            except Exception as e:
                logger.error(f"   ❌ Failed: {e}")
                self.results[bench_name] = {"error": str(e)}
        
        self._save_results()
        return self.results
    
    def _run_benchmark(self, bench_file: Path) -> Dict[str, Any]:
        """Run a single benchmark."""
        samples = []
        with open(bench_file) as f:
            for line in f:
                try:
                    samples.append(json.loads(line))
                except: pass
        
        # Limit samples for speed
        samples = samples[:100] 
        correct = 0
        total = 0
        
        for sample in tqdm(samples, desc=bench_file.stem):
            total += 1
            
            # Extract inputs (handle different formats)
            prompt = sample.get("question") or sample.get("prompt") or sample.get("input") or str(sample)
            answer = sample.get("answer") or sample.get("output") or sample.get("ground_truth")
            
            if not prompt: continue
            
            # Inference
            prediction = ""
            if self.model:
                inputs = self.tokenizer([prompt], return_tensors="pt").to("cuda")
                outputs = self.model.generate(**inputs, max_new_tokens=128)
                prediction = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            else:
                prediction = "mock_prediction" # Mock
            
            # Simple Exact Match or substring match grading
            if str(answer).lower() in prediction.lower():
                correct += 1
        
        return {
            "total": total,
            "correct": correct,
            "score": correct / total if total > 0 else 0,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
    
    def _save_results(self):
        """Save benchmark results."""
        output_file = Path("logs/benchmark_results.json")
        with open(output_file, "w") as f:
            json.dump(self.results, f, indent=2)
        logger.info(f"\n💾 Results saved to: {output_file}")


def main():
    if not check_env():
        return
        
    global logger
    os.makedirs('logs', exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/benchmarks.log'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)

    # Import locally
    global FastLanguageModel
    from unsloth import FastLanguageModel
    import torch
    from tqdm import tqdm
    
    parser = argparse.ArgumentParser(description="Run benchmarks on trained model")
    parser.add_argument("--model", type=str, default="checkpoints/stage1_sft/final",
                        help="Path to trained model")
    parser.add_argument("--benchmarks", type=str, default="data/benchmarks",
                        help="Directory containing benchmark datasets")
    args = parser.parse_args()
    
    runner = BenchmarkRunner(args.model, args.benchmarks)
    results = runner.run_all()
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("📊 BENCHMARK SUMMARY")
    logger.info("="*60)
    for name, result in results.items():
        if "error" in result:
            logger.info(f"   {name}: ERROR - {result['error']}")
        else:
            logger.info(f"   {name}: {result['score']:.2%}")
    logger.info("="*60)


if __name__ == "__main__":
    main()
