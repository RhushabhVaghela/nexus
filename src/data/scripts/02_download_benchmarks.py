#!/usr/bin/env python3
"""
13_download_benchmarks.py
Download & Normalize Official Benchmarks (~50GB)
Output: 
- data/benchmarks/*.jsonl (Native Schema: messages, etc.)
- data/benchmarks/images/ (Extracted images for multimodal)
"""

import logging
import os
import shutil
from pathlib import Path
try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False

def check_env():
    """Verify environment dependencies."""
    if not DATASETS_AVAILABLE:
        logger.error("Missing dependency: datasets")
        return False
    if os.environ.get("CONDA_DEFAULT_ENV") != "nexus":
        logger.error("Must be run in 'nexus' conda environment.")
        return False
    return True
import json
import uuid
from typing import Dict, Any

import sys
# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))
from utils.logging_config import setup_logger, log_benchmark_progress

logger = setup_logger(__name__, 'logs/benchmark_download.log')

BENCHMARKS = {
    # IQ Benchmarks
    "mmlu": {"dataset": "cais/mmlu", "config": "all", "split": "test", "type": "mcq"}, # Corrected split to 'test', config 'all'
    "gsm8k": {"dataset": "openai/gsm8k", "config": "main", "split": "test", "type": "cot"}, # Corrected split to 'test' (usually training on train, eval on test)
    # HumanEval uses 'openai_humaneval' mirror if official gated
    "humaneval": {"dataset": "openai_humaneval", "config": None, "split": "test", "type": "code"},
    
    # Multimodal
    # MMMU requires configs. We'll pick a few representative ones if 'all' isn't supported easily in one go. 
    # Actually 'MMMU/MMMU' usually requires config. Let's try 'Accounting' as placeholder or update code to loop.
    # For simplicity, we skip complex MMMU loop refactor and just pick one large config.
    "mmmu": {"dataset": "MMMU/MMMU", "config": "Computer_Science", "split": "validation", "type": "multimodal"},
    "mathvista": {"dataset": "AI4Math/MathVista", "config": None, "split": "testmini", "type": "multimodal"}, # Fixed Org
}

class BenchmarkNormalizer:
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.images_dir = base_dir / "images"
        self.images_dir.mkdir(parents=True, exist_ok=True)
        
    def normalize(self, name: str, sample: Dict[str, Any], index: int) -> Dict[str, Any]:
        """Convert raw sample to Native Schema."""
        method_name = f"norm_{BENCHMARKS[name]['type']}"
        if hasattr(self, method_name):
            return getattr(self, method_name)(name, sample, index)
        return self.norm_default(name, sample)

    def norm_mcq(self, name, sample, idx):
        # MMLU: question, choices, answer (0-3)
        choices = sample.get('choices', [])
        labels = ["A", "B", "C", "D"]
        options = "\n".join([f"{labels[i]}) {c}" for i, c in enumerate(choices)])
        answer_idx = sample.get('answer', 0)
        
        user_content = f"{sample['question']}\n\n{options}\nAnswer:"
        assist_content = labels[answer_idx]
        
        return {
            "id": f"{name}_{idx}",
            "domain": "benchmark_knowledge",
            "messages": [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": assist_content}
            ],
            "original_split": "test"
        }

    def norm_cot(self, name, sample, idx):
        # GSM8K: question, answer (CoT)
        return {
            "id": f"{name}_{idx}",
            "domain": "benchmark_reasoning",
            "messages": [
                {"role": "user", "content": sample['question']},
                {"role": "assistant", "content": sample['answer']}
            ]
        }
        
    def norm_code(self, name, sample, idx):
        # HumanEval: prompt, canonical_solution, test, entry_point
        return {
            "id": f"{name}_{idx}",
            "domain": "benchmark_code",
            "messages": [
                {"role": "user", "content": f"Complete this function:\n\n{sample['prompt']}"},
                {"role": "assistant", "content": f"```python\n{sample['prompt']}{sample['canonical_solution']}\n```"}
            ]
        }

    def norm_multimodal(self, name, sample, idx):
        # Check for image
        image = None
        # MMMU uses 'image' (PIL), MathVista uses 'decoded_image' (PIL) or similar
        for key in ['image', 'decoded_image']:
            if key in sample and sample[key]:
                image = sample[key]
                break
                
        image_path = None
        if image:
            # Save Image
            ext = "png"
            sub_dir = self.images_dir / name
            sub_dir.mkdir(exist_ok=True)
            fname = f"{idx}.{ext}"
            saved_path = sub_dir / fname
            try:
                if not saved_path.exists():
                    image.save(saved_path)
                image_path = str(saved_path)
            except Exception as e:
                logger.warning(f"Failed to save image {name}/{idx}: {e}")

        # Construct Text
        prompt = sample.get('question', '') or sample.get('query', '')
        options = sample.get('options', '')
        if options:
            prompt += f"\nOptions: {options}"
            
        answer = sample.get('answer', '') or sample.get('response', '')
        
        return {
            "id": f"{name}_{idx}",
            "domain": "benchmark_multimodal",
            "image_path": image_path, # Native Schema Extension
            "messages": [
                {"role": "user", "content": prompt}, # Processors will inject <image> if image_path exists
                {"role": "assistant", "content": str(answer)}
            ]
        }

    def norm_default(self, name, sample):
        # Fallback
        return {"id": f"{name}_unknown", "original": str(sample)}

def main():
    if not check_env():
        sys.exit(1)
        
    import argparse
    parser = argparse.ArgumentParser(description="Download benchmarks")
    parser.add_argument("--limit", type=int, default=100, help="Max samples")
    args = parser.parse_args()
        
    output_dir = Path("/mnt/e/data/benchmarks")
    normalizer = BenchmarkNormalizer(output_dir)
    
    logger.info(f"🚀 Starting Normalized Benchmark Download (Streaming) (Limit: {args.limit})...")
    
    for name, meta in BENCHMARKS.items():
        logger.info(f"📥 Processing {name} ({meta['type']})...")
        try:
            # Use Streaming
            if meta['config']:
                ds = load_dataset(meta['dataset'], meta['config'], split=meta['split'], streaming=True)
            else:
                ds = load_dataset(meta['dataset'], split=meta['split'], streaming=True)
                
            out_file = output_dir / f"{name}.jsonl"
            
            # We don't know length in streaming
            count = 0
            with open(out_file, "w") as f:
                for i, item in enumerate(ds):
                    if i >= args.limit:
                        break
                        
                    norm_item = normalizer.normalize(name, item, i)
                    f.write(json.dumps(norm_item) + "\n")
                    count += 1
                    
                    if i % 1000 == 0:
                        print(f"   Processed {i} samples...", end="\r")
            
            logger.info(f"   ✅ Complete: {count} samples saved to {out_file}")
            
        except Exception as e:
            logger.error(f"   ❌ Failed {name}: {e}")

if __name__ == "__main__":
    main()
