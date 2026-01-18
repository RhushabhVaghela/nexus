#!/usr/bin/env python3
"""
Stage 5: Comprehensive Evaluation (1-2 days)
Run FULL benchmark suite on trained model
Output: evaluation_results/ with detailed analysis
"""

import json
import torch
import logging
from pathlib import Path
from typing import Dict, List
from collections import defaultdict
import os

from unsloth import FastLanguageModel
from datasets import load_dataset
import tqdm

# Create logs directory if it doesn't exist
try:
    os.makedirs('logs', exist_ok=True)
except Exception:
    pass

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/evaluation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

BENCHMARKS = {
    "mmlu": {
        "dataset": "cais/mmlu",
        "config": "all",
        "split": "auxiliary",
        "samples": 100,  # Reduced for testing
    },
    "gsm8k": {
        "dataset": "openai/gsm8k",
        "config": "main",
        "split": "test",
        "samples": 100,
    },
    "humaneval": {
        "dataset": "openai/human_eval",
        "config": None,
        "split": "test",
        "samples": 50,
    },
}

def evaluate_benchmark(model, tokenizer, benchmark_name: str, dataset, max_samples: int = 100):
    """Evaluate on single benchmark"""
    correct = 0
    results = []
    
    logger.info(f"  Evaluating {benchmark_name}...")
    
    for idx, sample in enumerate(tqdm.tqdm(dataset.take(max_samples), desc=benchmark_name)):
        if benchmark_name == "mmlu":
            question = sample["question"]
            answer = sample["answerKey"]
            prompt = f"Question: {question}\nChoices: A) B) C) D)\nAnswer: "
        elif benchmark_name == "gsm8k":
            question = sample["question"]
            answer = sample["answer"].split("####")[-1].strip()
            prompt = f"Question: {question}\nAnswer: "
        elif benchmark_name == "humaneval":
            prompt = sample["prompt"]
            answer = sample["canonical_solution"]
        else:
            continue
        
        # Generate
        try:
            inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=256)
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Check correctness
            is_correct = str(answer).lower() in response.lower()
            if is_correct:
                correct += 1
            
            results.append({
                "id": idx,
                "prompt": prompt[:100],
                "answer": str(answer)[:50],
                "response": response[:100],
                "correct": is_correct
            })
        except Exception as e:
            logger.warning(f"    Error on sample {idx}: {e}")
    
    accuracy = correct / min(len(dataset), max_samples)
    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": min(len(dataset), max_samples),
        "results": results
    }

def main():
    logger.info("="*70)
    logger.info("📊 STAGE 5: COMPREHENSIVE EVALUATION")
    logger.info("="*70)
    
    # Choose model
    model_path = "checkpoints/stage3_grpo/final"
    if not Path(model_path).exists():
        logger.error(f"❌ Model not found: {model_path}")
        return
    
    # Load model
    logger.info("\n📦 Loading model for evaluation...")
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            max_seq_length=4096,
            load_in_4bit=True,
            dtype=torch.bfloat16,
        )
        FastLanguageModel.for_inference(model)
        logger.info("✓ Model loaded")
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        return
    
    # Evaluate
    output_dir = Path("evaluation_results")
    output_dir.mkdir(exist_ok=True)
    
    results = {}
    logger.info("\n🔍 Running benchmarks...")
    
    for bench_name, bench_config in BENCHMARKS.items():
        logger.info(f"\n📚 {bench_name.upper()}")
        
        try:
            if bench_config["config"]:
                dataset = load_dataset(
                    bench_config["dataset"],
                    bench_config["config"],
                    split=bench_config["split"],
                    trust_remote_code=True
                )
            else:
                dataset = load_dataset(
                    bench_config["dataset"],
                    split=bench_config["split"],
                    trust_remote_code=True
                )
            
            eval_result = evaluate_benchmark(
                model, tokenizer,
                bench_name, dataset,
                max_samples=bench_config["samples"]
            )
            
            results[bench_name] = eval_result
            logger.info(f"  Accuracy: {eval_result['accuracy']*100:.1f}%")
        except Exception as e:
            logger.error(f"  Failed: {e}")
    
    # Save results
    logger.info("\n💾 Saving results...")
    
    # Summary
    summary = {bench: results[bench]["accuracy"] for bench in results}
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    # Detailed
    with open(output_dir / "detailed_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"✓ Results saved to: {output_dir}")
    
    logger.info("\n" + "="*70)
    logger.info("✅ EVALUATION COMPLETE!")
    logger.info("="*70)
    logger.info("Results:")
    for bench, result in results.items():
        logger.info(f"  {bench}: {result['accuracy']*100:.1f}%")

if __name__ == "__main__":
    main()
