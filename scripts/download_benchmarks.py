#!/usr/bin/env python3
"""
FILE 1: 01_download_benchmarks.py
Download FULL official benchmark datasets (~50GB)
Output: data/benchmarks/*.jsonl
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict
from datasets import load_dataset
import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/download_benchmarks.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

BENCHMARKS_CONFIG = {
    "mmlu": {
        "dataset": "cais/mmlu",
        "config": "all",
        "split": "auxiliary",
        "output": "data/benchmarks/mmlu.jsonl",
        "description": "57k knowledge questions across 57 subjects"
    },
    "mmlu_pro": {
        "dataset": "TIGER-Lab/MMLU-Pro",
        "config": None,
        "split": "test",
        "output": "data/benchmarks/mmlu_pro.jsonl",
        "description": "12k harder curated MMLU"
    },
    "gsm8k": {
        "dataset": "openai/gsm8k",
        "config": "main",
        "split": "train",
        "output": "data/benchmarks/gsm8k.jsonl",
        "description": "1,319 grade school math problems"
    },
    "math": {
        "dataset": "hendrycks/competition_math",
        "config": None,
        "split": "train",
        "output": "data/benchmarks/math.jsonl",
        "description": "12,500 competition math problems"
    },
    "gpqa": {
        "dataset": "Idavidc/gpqa",
        "config": "gpqa_diamond",
        "split": "train",
        "output": "data/benchmarks/gpqa.jsonl",
        "description": "450 graduate physics Q&A"
    },
    "humaneval": {
        "dataset": "openai/human_eval",
        "config": None,
        "split": "test",
        "output": "data/benchmarks/humaneval.jsonl",
        "description": "164 programming functions"
    },
    "swe_bench_verified": {
        "dataset": "princeton-nlp/SWE-bench_verified",
        "config": None,
        "split": "test",
        "output": "data/benchmarks/swe_bench_verified.jsonl",
        "description": "500 verified GitHub issues"
    },
    "bigcodebench": {
        "dataset": "bigcode/bigcodebench",
        "config": "full",
        "split": "test",
        "output": "data/benchmarks/bigcodebench.jsonl",
        "description": "140+ diverse coding tasks"
    },
}

def download_benchmark(benchmark_name: str, config: Dict) -> int:
    """Download single benchmark and save as JSONL"""
    output_path = Path(config["output"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"\n📥 Downloading {benchmark_name}")
    logger.info(f"   Description: {config['description']}")
    logger.info(f"   Output: {output_path}")
    
    try:
        # Load dataset
        if config["config"]:
            dataset = load_dataset(
                config["dataset"],
                config["config"],
                split=config["split"],
                trust_remote_code=True
            )
        else:
            dataset = load_dataset(
                config["dataset"],
                split=config["split"],
                trust_remote_code=True
            )
        
        count = 0
        with open(output_path, "w") as f:
            for sample in tqdm.tqdm(dataset, desc=f"Saving {benchmark_name}"):
                f.write(json.dumps(sample) + "\n")
                count += 1
        
        logger.info(f"   ✓ Saved {count} samples")
        return count
    
    except Exception as e:
        logger.error(f"   ❌ Failed: {e}")
        return 0

def main():
    logger.info("="*70)
    logger.info("📥 DOWNLOADING BENCHMARK DATASETS")
    logger.info("="*70)
    logger.info(f"Total benchmarks: {len(BENCHMARKS_CONFIG)}")
    logger.info(f"Expected size: ~50GB")
    logger.info("="*70)
    
    total_samples = 0
    successful = 0
    
    for bench_name, bench_config in BENCHMARKS_CONFIG.items():
        count = download_benchmark(bench_name, bench_config)
        if count > 0:
            total_samples += count
            successful += 1
    
    logger.info("\n" + "="*70)
    logger.info("✅ DOWNLOAD COMPLETE")
    logger.info("="*70)
    logger.info(f"Successful: {successful}/{len(BENCHMARKS_CONFIG)}")
    logger.info(f"Total samples: {total_samples:,}")
    logger.info(f"Location: data/benchmarks/")
    logger.info("="*70)
    logger.info(f"\nNext: Run Stage 2 (Generate Trajectories)")
    logger.info(f"  python 02_generate_trajectories.py")

if __name__ == "__main__":
    main()
