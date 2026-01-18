#!/usr/bin/env python3
"""
utils/data_mixer.py
Mixes real and synthetic data at specified ratios to prevent model collapse.

Strategy: 30% real data + 70% synthetic data = optimal training mix
Reference: https://arxiv.org/abs/2510.01631
"""

import os
import sys
import json
import random
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import argparse

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.logging_config import setup_logger, log_header

# ═══════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════

CONFIG = {
    "real_ratio": 0.30,      # 30% real data (prevents collapse)
    "synthetic_ratio": 0.70,  # 70% synthetic data (provides diversity)
    
    "real_data_dirs": [
        "/mnt/e/data/real-datasets/code",
        "/mnt/e/data/real-datasets/reasoning",
        "/mnt/e/data/real-datasets/domain",
    ],
    
    "synthetic_data_dirs": [
        "/mnt/e/data/finetuned-fullstack-dataset",
        "/mnt/e/data/repetitive-query-dataset",
        "/mnt/e/data/repetitive-prompt-dataset",         # current file's output
        "/mnt/e/data/architecture-reasoning-dataset",
        "/mnt/e/data/qa-engineering-dataset",
        "/mnt/e/data/uiux-design-dataset",
        "/mnt/e/data/devops-engineering-dataset",
        "/mnt/e/data/repetitive-fullstack-dataset",      # NEW if you split fullstack out
    ],

    
    "output_dir": "/mnt/e/data/mixed-training",
    "samples_per_file": 100_000,
    "seed": 42,
}

logger = setup_logger(__name__, "logs/data_mixer.log")

# ═══════════════════════════════════════════════════════════════
# NORMALIZERS (Convert various formats to OpenAI messages)
# ═══════════════════════════════════════════════════════════════

def normalize_to_messages(sample: Dict, source: str = "unknown") -> Optional[Dict]:
    """
    Normalize any dataset format to OpenAI messages format.
    Works with: Alpaca, ShareGPT, OpenAI, custom formats.
    """
    
    messages = []
    
    # Already in messages format
    if "messages" in sample:
        messages = sample["messages"]
    
    # Alpaca format: instruction, input, output
    elif "instruction" in sample and "output" in sample:
        user_content = sample["instruction"]
        if sample.get("input"):
            user_content += f"\n\n{sample['input']}"
        messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": sample["output"]}
        ]
    
    # ShareGPT format: conversations
    elif "conversations" in sample:
        for turn in sample["conversations"]:
            role = "user" if turn.get("from") in ["human", "user"] else "assistant"
            messages.append({"role": role, "content": turn.get("value", "")})
    
    # Code format: prompt, completion
    elif "prompt" in sample and "completion" in sample:
        messages = [
            {"role": "user", "content": sample["prompt"]},
            {"role": "assistant", "content": sample["completion"]}
        ]
    
    # Question-Answer format
    elif "question" in sample and "answer" in sample:
        messages = [
            {"role": "user", "content": sample["question"]},
            {"role": "assistant", "content": sample["answer"]}
        ]
    
    # Text only (for continued pretraining)
    elif "text" in sample or "content" in sample:
        text = sample.get("text") or sample.get("content")
        messages = [
            {"role": "user", "content": "Continue this text:"},
            {"role": "assistant", "content": text}
        ]
    
    else:
        return None
    
    # Validate messages
    if len(messages) < 2:
        return None
    
    # Create normalized sample
    normalized = {
        "id": sample.get("id") or hashlib.md5(str(messages).encode()).hexdigest()[:12],
        "messages": messages,
        "source": source,
        "domain": sample.get("domain", "general"),
    }
    
    return normalized

# ═══════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════

def load_jsonl_files(directories: List[str], max_samples: Optional[int] = None) -> List[Dict]:
    """Load all JSONL files from directories."""
    samples = []
    
    for dir_path in directories:
        dir_path = Path(dir_path)
        if not dir_path.exists():
            logger.warning(f"Directory not found: {dir_path}")
            continue
        
        # Find all JSONL files recursively
        jsonl_files = list(dir_path.rglob("*.jsonl"))
        logger.info(f"Found {len(jsonl_files)} JSONL files in {dir_path}")
        
        for file_path in jsonl_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            sample = json.loads(line)
                            normalized = normalize_to_messages(sample, source=str(file_path.parent.name))
                            if normalized:
                                samples.append(normalized)
                                
                                if max_samples and len(samples) >= max_samples:
                                    return samples
            except Exception as e:
                logger.warning(f"Error reading {file_path}: {e}")
    
    return samples

def load_json_files(directories: List[str], max_samples: Optional[int] = None) -> List[Dict]:
    """Load JSON files (single array format)."""
    samples = []
    
    for dir_path in directories:
        dir_path = Path(dir_path)
        if not dir_path.exists():
            continue
        
        json_files = list(dir_path.rglob("*.json"))
        
        for file_path in json_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        for item in data:
                            normalized = normalize_to_messages(item, source=str(file_path.parent.name))
                            if normalized:
                                samples.append(normalized)
                                if max_samples and len(samples) >= max_samples:
                                    return samples
            except Exception as e:
                logger.warning(f"Error reading {file_path}: {e}")
    
    return samples

# ═══════════════════════════════════════════════════════════════
# DATA MIXING
# ═══════════════════════════════════════════════════════════════

def mix_datasets(
    real_samples: List[Dict],
    synthetic_samples: List[Dict],
    real_ratio: float = 0.30,
    seed: int = 42
) -> Tuple[List[Dict], Dict]:
    """
    Mix real and synthetic data at specified ratio.
    
    Returns:
        mixed_samples: Combined dataset
        stats: Statistics about the mix
    """
    random.seed(seed)
    
    # Calculate target counts based on synthetic size
    total_synthetic = len(synthetic_samples)
    total_real = len(real_samples)
    
    # Target: Use all synthetic, sample proportional real
    target_real = int(total_synthetic * (real_ratio / (1 - real_ratio)))
    target_real = min(target_real, total_real)  # Can't sample more than we have
    
    # Sample real data
    if target_real < total_real:
        real_sampled = random.sample(real_samples, target_real)
    else:
        real_sampled = real_samples
    
    # Combine
    mixed = real_sampled + synthetic_samples
    random.shuffle(mixed)
    
    # Compute stats
    actual_real_ratio = len(real_sampled) / len(mixed) if mixed else 0
    
    stats = {
        "total_samples": len(mixed),
        "real_samples": len(real_sampled),
        "synthetic_samples": len(synthetic_samples),
        "real_ratio": actual_real_ratio,
        "synthetic_ratio": 1 - actual_real_ratio,
        "target_ratio": real_ratio,
    }
    
    return mixed, stats

# ═══════════════════════════════════════════════════════════════
# SPLIT AND SAVE
# ═══════════════════════════════════════════════════════════════

def split_and_save(
    samples: List[Dict],
    output_dir: str,
    train_ratio: float = 0.95,
    val_ratio: float = 0.025,
    samples_per_file: int = 100_000
):
    """Split data and save to train/val/test directories."""
    
    random.shuffle(samples)
    total = len(samples)
    
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    
    splits = {
        "train": samples[:train_end],
        "val": samples[train_end:val_end],
        "test": samples[val_end:]
    }
    
    output_path = Path(output_dir)
    
    for split_name, split_samples in splits.items():
        split_dir = output_path / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        
        # Write in chunks
        for i in range(0, len(split_samples), samples_per_file):
            chunk = split_samples[i:i + samples_per_file]
            chunk_idx = i // samples_per_file
            file_path = split_dir / f"part_{chunk_idx:04d}.jsonl"
            
            with open(file_path, 'w', encoding='utf-8') as f:
                for sample in chunk:
                    f.write(json.dumps(sample, ensure_ascii=False) + "\n")
        
        logger.info(f"Saved {len(split_samples)} samples to {split_dir}")
    
    return splits

# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Mix real and synthetic datasets")
    parser.add_argument("--real-ratio", type=float, default=0.30, help="Ratio of real data (default: 0.30)")
    parser.add_argument("--output", type=str, default=CONFIG["output_dir"], help="Output directory")
    parser.add_argument("--max-synthetic", type=int, default=None, help="Max synthetic samples to load")
    parser.add_argument("--max-real", type=int, default=None, help="Max real samples to load")
    args = parser.parse_args()
    
    log_header(logger, "DATA MIXER: Real + Synthetic", {
        "Real Ratio": args.real_ratio,
        "Synthetic Ratio": 1 - args.real_ratio,
        "Output": args.output
    })
    
    # Load real data
    logger.info("Loading real datasets...")
    real_samples = load_jsonl_files(CONFIG["real_data_dirs"], max_samples=args.max_real)
    real_samples.extend(load_json_files(CONFIG["real_data_dirs"], max_samples=args.max_real))
    logger.info(f"Loaded {len(real_samples)} real samples")
    
    # Load synthetic data
    logger.info("Loading synthetic datasets...")
    synthetic_samples = load_jsonl_files(CONFIG["synthetic_data_dirs"], max_samples=args.max_synthetic)
    logger.info(f"Loaded {len(synthetic_samples)} synthetic samples")
    
    # Mix
    logger.info("Mixing datasets...")
    mixed, stats = mix_datasets(real_samples, synthetic_samples, real_ratio=args.real_ratio)
    
    logger.info(f"Mixed dataset statistics:")
    for key, value in stats.items():
        if isinstance(value, float):
            logger.info(f"  {key}: {value:.2%}")
        else:
            logger.info(f"  {key}: {value:,}")
    
    # Split and save
    logger.info("Splitting and saving...")
    split_and_save(mixed, args.output, samples_per_file=CONFIG["samples_per_file"])
    
    logger.info("✅ Data mixing complete!")
    logger.info(f"Output: {args.output}")

if __name__ == "__main__":
    main()
