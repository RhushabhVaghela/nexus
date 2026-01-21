#!/usr/bin/env python3
"""
20_load_premium_datasets.py
Load highest-quality preference, safety, and anti-refusal datasets with ratio-based sampling

Based on: new documentt.md
- UltraFeedback, HelpSteer, HH-RLHF for RLHF preferences
- PKU-SafeRLHF, BeaverTails for safety alignment
- Pure-Dove, Dolphin, Toxic-DPO for anti-refusal

Usage:
  python 20_load_premium_datasets.py --mode censored --target-samples 100000
  python 20_load_premium_datasets.py --mode uncensored --target-samples 500000
  python 20_load_premium_datasets.py --mode censored --show-breakdown
"""

import os
import sys
import json
import argparse
import random
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).parent))
from utils.logging_config import setup_logger, log_header, log_completion

try:
    from datasets import load_dataset, concatenate_datasets, Dataset
except ImportError:
    print("Install: pip install datasets")
    sys.exit(1)

# ═══════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════

CONFIG = {
    "output_base_dir": "/mnt/e/data/premium",
}

logger = setup_logger(__name__, "logs/load_premium_datasets.log")

# ═══════════════════════════════════════════════════════════════
# DATASET RATIOS (Based on new documentt.md)
# ═══════════════════════════════════════════════════════════════

CENSORED_RATIOS = {
    # RLHF Preference (70%)
    "openbmb/UltraFeedback": 0.20,           # GPT-4 quality ⭐⭐⭐⭐⭐
    "nvidia/HelpSteer": 0.15,                 # NVIDIA quality ⭐⭐⭐⭐⭐
    "Anthropic/hh-rlhf": 0.25,                # Industry standard ⭐⭐⭐⭐⭐
    "berkeley-nest/Nectar": 0.10,             # Multi-model diversity ⭐⭐⭐⭐
    
    # Safety (30%)
    "PKU-Alignment/PKU-SafeRLHF": 0.10,       # Research quality ⭐⭐⭐⭐⭐
    "PKU-Alignment/BeaverTails": 0.20,        # Comprehensive safety ⭐⭐⭐⭐⭐
}

UNCENSORED_RATIOS = {
    # RLHF Preference (60%)
    "openbmb/UltraFeedback": 0.20,           # GPT-4 quality ⭐⭐⭐⭐⭐
    "nvidia/HelpSteer": 0.15,                 # NVIDIA quality ⭐⭐⭐⭐⭐
    "berkeley-nest/Nectar": 0.15,             # Diversity ⭐⭐⭐⭐
    "OpenAssistant/oasst2": 0.10,             # Community quality ⭐⭐⭐⭐
    
    # Anti-Refusal (40%)
    "LDJnr/Pure-Dove": 0.05,                  # De-censoring ⭐⭐⭐⭐⭐ (small but powerful)
    "HuggingFaceH4/no_robots": 0.10,          # Human-written ⭐⭐⭐⭐⭐ (Official)
    "cognitivecomputations/dolphin": 0.15,    # Reasoning-focused ⭐⭐⭐⭐⭐
    "unalignment/toxic-dpo-v0.1": 0.05,       # Anti-safety DPO ⭐⭐⭐⭐
    "teknium/OpenHermes-2.5": 0.05,           # General capability ⭐⭐⭐⭐
}

# Dataset split configurations
DATASET_CONFIGS = {
    "Anthropic/hh-rlhf": {"split": "train"},
    "OpenAssistant/oasst2": {"split": "train"},
    "PKU-Alignment/PKU-SafeRLHF": {"split": "train"},
    "PKU-Alignment/BeaverTails": {"split": "train"},
    "openbmb/UltraFeedback": {"split": "train"},
    "nvidia/HelpSteer": {"split": "train"},
    "berkeley-nest/Nectar": {"split": "train"},
    "LDJnr/Pure-Dove": {"split": "train"},
    "HuggingFaceH4/no_robots": {"split": "train"},
    "cognitivecomputations/dolphin": {"split": "train"},
    "unalignment/toxic-dpo-v0.1": {"split": "train"},
    "teknium/OpenHermes-2.5": {"split": "train"},
}


# ═══════════════════════════════════════════════════════════════
# NORMALIZER
# ═══════════════════════════════════════════════════════════════

class PremiumNormalizer:
    @staticmethod
    def normalize(sample: Dict, dataset_name: str) -> Dict:
        """Normalize premium samples to Unified Messages Schema."""
        messages = []
        
        # 1. Already in messages/conversations format (Dolphin, No-Robots, etc.)
        if "messages" in sample:
            messages = sample["messages"]
        elif "conversations" in sample:
            # ShareGPT format
            for turn in sample["conversations"]:
                role = "user" if turn.get("from") in ["human", "user"] else "assistant"
                messages.append({"role": role, "content": turn.get("value", "")})

        # 2. Preference / Reward Modeling Format (UltraFeedback, HelpSteer)
        # Usually: columns like "instruction", "response" (or "chosen"), "model"
        # UltraFeedback processed: instruction, response
        # HelpSteer: prompt, response
        elif "prompt" in sample and "response" in sample:
            messages = [
                {"role": "user", "content": sample["prompt"]},
                {"role": "assistant", "content": sample["response"]}
            ]
        elif "instruction" in sample and "response" in sample:
            messages = [
                {"role": "user", "content": sample["instruction"]},
                {"role": "assistant", "content": sample["response"]}
            ]

        # 3. RLHF Pairs (HH-RLHF, SafeRLHF) -> Take 'chosen' or 'safe'
        # HH-RLHF: 'chosen' (str conversation), 'rejected' (str conversation)
        elif "chosen" in sample:
            # Determine if chosen is a list (messages) or string
            chosen = sample["chosen"]
            if isinstance(chosen, str):
                # Parser for Anthropic HH-RLHF string format
                # Format: "\n\nHuman: ... \n\nAssistant: ..."
                parts = chosen.split("\n\n")
                for part in parts:
                    if part.startswith("Human:"):
                        messages.append({"role": "user", "content": part.replace("Human:", "").strip()})
                    elif part.startswith("Assistant:"):
                        messages.append({"role": "assistant", "content": part.replace("Assistant:", "").strip()})
            elif isinstance(chosen, list):
                # Already a list of dicts
                messages = chosen

        # 4. PKU-SafeRLHF: prompt, response_0, response_1, safer_response_id
        elif "prompt" in sample and "response_0" in sample:
            safe_idx = int(sample.get("safer_response_id", 0))
            if safe_idx == -1: safe_idx = 0 # Fallback
            safe_response = sample.get(f"response_{safe_idx}", "")
            messages = [
                {"role": "user", "content": sample["prompt"]},
                {"role": "assistant", "content": safe_response}
            ]
            
        # Fallback validation
        if not messages:
            return None
            
        return {"messages": messages, "source": dataset_name}

# ═══════════════════════════════════════════════════════════════
# DATASET LOADER
# ═══════════════════════════════════════════════════════════════

class PremiumDatasetLoader:
    """Load premium datasets with ratio-based sampling."""
    
    def __init__(self, mode: str, target_samples: int, seed: int = 42):
        self.mode = mode
        self.target_samples = target_samples
        self.seed = seed
        
        self.ratios = CENSORED_RATIOS if mode == "censored" else UNCENSORED_RATIOS
        
        # Validate ratios
        ratio_sum = sum(self.ratios.values())
        if abs(ratio_sum - 1.0) > 0.001:
            raise ValueError(f"Ratios must sum to 1.0, got {ratio_sum}")
        
        logger.info(f"Mode: {mode.upper()}")
        logger.info(f"Target samples: {target_samples:,}")
        logger.info(f"Random seed: {seed}")
    
    def calculate_samples_per_dataset(self) -> Dict[str, int]:
        """Calculate exact number of samples to take from each dataset."""
        samples = {}
        for dataset_name, ratio in self.ratios.items():
            samples[dataset_name] = int(self.target_samples * ratio)
        
        # Adjust for rounding errors
        total_allocated = sum(samples.values())
        diff = self.target_samples - total_allocated
        if diff != 0:
            largest = max(samples, key=samples.get)
            samples[largest] += diff
        
        return samples
    
    def load_and_sample_dataset(self, dataset_name: str, num_samples: int) -> Dataset:
        """Load and sample a single dataset (Streaming)."""
        logger.info(f"Loading {dataset_name} (Streaming)...")
        
        try:
            config = DATASET_CONFIGS.get(dataset_name, {"split": "train"})
            # Force streaming
            dataset = load_dataset(dataset_name, **config, streaming=True)
            
            if isinstance(dataset, dict):
                # Should not happen with split specified, but safety check
                dataset = dataset.get('train', list(dataset.values())[0])
            
            # Shuffle buffer for randomness without full download
            dataset = dataset.shuffle(seed=self.seed, buffer_size=10000)
            
            # Take N items
            raw_samples = list(dataset.take(num_samples))
            
            # Normalize immediately
            norm_samples = []
            for s in raw_samples:
                norm = PremiumNormalizer.normalize(s, dataset_name)
                if norm:
                    norm_samples.append(norm)
            
            logger.info(f"  Streamed: {len(norm_samples):,} samples (Normalized)")
            
            # Convert to memory dataset for concatenation later
            return Dataset.from_list(norm_samples)
        
        except Exception as e:
            logger.error(f"  Failed: {e}")
            return None
    
    def load_all_datasets(self) -> Dataset:
        """Load all datasets according to ratios."""
        log_header(logger, f"LOADING PREMIUM DATASETS ({self.mode.upper()})", {
            "Target": self.target_samples,
            "Datasets": len(self.ratios)
        })
        
        samples_per_dataset = self.calculate_samples_per_dataset()
        
        # Log plan
        logger.info("Loading plan:")
        for name, num in samples_per_dataset.items():
            pct = self.ratios[name] * 100
            logger.info(f"  {name}: {num:,} ({pct:.1f}%)")
        
        # Load datasets
        loaded = []
        
        for dataset_name, num_samples in samples_per_dataset.items():
            dataset = self.load_and_sample_dataset(dataset_name, num_samples)
            if dataset:
                loaded.append(dataset)
        
        # Combine
        logger.info("Combining datasets...")
        
        if not loaded:
            raise ValueError("No datasets loaded!")
        
        combined = concatenate_datasets(loaded)
        combined = combined.shuffle(seed=self.seed)
        
        logger.info(f"Final dataset: {len(combined):,} samples")
        return combined
    
    def save_dataset(self, dataset: Dataset, output_dir: str):
        """Save dataset to disk."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving to {output_path}...")
        
        # Save as HuggingFace dataset
        dataset.save_to_disk(str(output_path))
        
        # Save metadata
        metadata = {
            "mode": self.mode,
            "target_samples": self.target_samples,
            "actual_samples": len(dataset),
            "seed": self.seed,
            "ratios": self.ratios,
            "samples_per_dataset": self.calculate_samples_per_dataset(),
        }
        
        with open(output_path / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Saved {len(dataset):,} samples + metadata.json")


# ═══════════════════════════════════════════════════════════════
# BREAKDOWN DISPLAY
# ═══════════════════════════════════════════════════════════════

def show_breakdown(mode: str, target_samples: int):
    """Show detailed ratio breakdown."""
    ratios = CENSORED_RATIOS if mode == "censored" else UNCENSORED_RATIOS
    
    print("\n" + "="*70)
    print(f"📊 RATIO BREAKDOWN ({mode.upper()}) - {target_samples:,} samples")
    print("="*70)
    
    if mode == "censored":
        categories = {
            "RLHF Preference (70%)": ["openbmb/UltraFeedback", "nvidia/HelpSteer", 
                                      "Anthropic/hh-rlhf", "berkeley-nest/Nectar"],
            "Safety Alignment (30%)": ["PKU-Alignment/PKU-SafeRLHF", "PKU-Alignment/BeaverTails"],
        }
    else:
        categories = {
            "RLHF Preference (60%)": ["openbmb/UltraFeedback", "nvidia/HelpSteer", 
                                      "berkeley-nest/Nectar", "OpenAssistant/oasst2"],
            "Anti-Refusal (40%)": ["LDJnr/Pure-Dove", "HuggingFaceH4/no_robots",
                                   "cognitivecomputations/dolphin", "unalignment/toxic-dpo-v0.1",
                                   "teknium/OpenHermes-2.5"],
        }
    
    for category, datasets in categories.items():
        print(f"\n{category}")
        print("-" * 70)
        
        subtotal = 0
        for ds in datasets:
            if ds in ratios:
                num = int(target_samples * ratios[ds])
                pct = ratios[ds] * 100
                subtotal += num
                print(f"  {ds:48} {num:>10,} ({pct:>5.1f}%)")
        
        print(f"  {'Subtotal':48} {subtotal:>10,}")
    
    print("\n" + "="*70)
    print(f"{'TOTAL':50} {target_samples:>10,} (100.0%)")
    print("="*70)


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Load premium datasets with ratio-based sampling")
    parser.add_argument("--mode", choices=["censored", "uncensored"], required=True)
    parser.add_argument("--target-samples", type=int, default=100000)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--show-breakdown", action="store_true")
    
    parser.add_argument("--limit", type=int, help="Override target-samples for testing (e.g. 50)")
    
    args = parser.parse_args()
    
    if args.limit:
        args.target_samples = args.limit
    
    # Enforce 'nexus' conda environment
    if os.environ.get("CONDA_DEFAULT_ENV") != "nexus":
        sys.exit("\033[0;31m[ERROR] Must be run in 'nexus' conda environment.\033[0m")

    # Show breakdown only
    if args.show_breakdown:
        show_breakdown(args.mode, args.target_samples)
        return
    
    # Set output dir
    output_dir = args.output_dir or f"{CONFIG['output_base_dir']}/{args.mode}_{args.target_samples}"
    
    log_header(logger, "PREMIUM DATASET LOADER", {
        "Mode": args.mode,
        "Target": args.target_samples,
        "Output": output_dir
    })
    
    # Load datasets
    loader = PremiumDatasetLoader(
        mode=args.mode,
        target_samples=args.target_samples,
        seed=args.seed
    )
    
    dataset = loader.load_all_datasets()
    
    # Save
    loader.save_dataset(dataset, output_dir)
    
    log_completion(logger, "Premium Dataset Loading", {
        "Mode": args.mode,
        "Total samples": len(dataset),
        "Output": output_dir
    })


if __name__ == "__main__":
    main()
