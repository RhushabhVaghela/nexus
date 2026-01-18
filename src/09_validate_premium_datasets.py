#!/usr/bin/env python3
"""
09_validate_premium_datasets.py
Validate premium RLHF/safety/anti-refusal datasets

Validates the datasets loaded by 03_load_premium_datasets.py:
- Schema compliance (preference pair format)
- Content quality
- Distribution verification

Usage:
  python 09_validate_premium_datasets.py --mode censored
"""

import os
import sys
import json
import hashlib
from pathlib import Path
from typing import Dict, Set
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))
from utils.logging_config import setup_logger, log_header, log_completion

# ═══════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════

CONFIG = {
    "premium_base_dir": "/mnt/e/data/premium",
}

logger = setup_logger(__name__, "logs/validate_premium.log")

# Expected datasets per mode
EXPECTED_CENSORED = [
    "openbmb/UltraFeedback", "nvidia/HelpSteer", "Anthropic/hh-rlhf",
    "berkeley-nest/Nectar", "PKU-Alignment/PKU-SafeRLHF", "PKU-Alignment/BeaverTails"
]

EXPECTED_UNCENSORED = [
    "openbmb/UltraFeedback", "nvidia/HelpSteer", "berkeley-nest/Nectar",
    "OpenAssistant/oasst2", "LDJnr/Pure-Dove", "teknium/no-robots-uncensored",
    "cognitivecomputations/dolphin", "unalignment/toxic-dpo-v0.1", "teknium/OpenHermes-2.5"
]


# ═══════════════════════════════════════════════════════════════
# VALIDATOR
# ═══════════════════════════════════════════════════════════════

class PremiumDatasetValidator:
    """Validate premium RLHF datasets."""
    
    def __init__(self, mode: str):
        self.mode = mode
        self.seen_hashes: Set[str] = set()
        self.stats = defaultdict(int)
        self.source_counts = defaultdict(int)
    
    def validate_preference_format(self, sample: Dict) -> bool:
        """Check if sample has valid preference pair format."""
        # Must have prompt/chosen/rejected OR messages
        if "chosen" in sample and "rejected" in sample:
            if not sample.get("prompt") and not sample.get("instruction"):
                self.stats["missing_prompt"] += 1
                return False
            return True
        
        if "messages" in sample:
            return True
        
        # Check for other common formats
        if "question" in sample or "instruction" in sample:
            return True
        
        self.stats["invalid_format"] += 1
        return False
    
    def validate_content(self, sample: Dict) -> bool:
        """Check content quality."""
        # Get relevant content fields
        content_fields = []
        
        for field in ["prompt", "chosen", "rejected", "instruction", "response", "question", "answer"]:
            if field in sample:
                content_fields.append(str(sample[field]))
        
        if "messages" in sample:
            for msg in sample["messages"]:
                content_fields.append(str(msg.get("content", "")))
        
        # Check if content exists
        total_length = sum(len(c) for c in content_fields)
        
        if total_length < 20:
            self.stats["too_short"] += 1
            return False
        
        return True
    
    def check_duplicate(self, sample: Dict) -> bool:
        """Check for duplicates."""
        content = json.dumps(sample, sort_keys=True)
        sample_hash = hashlib.md5(content.encode()).hexdigest()
        
        if sample_hash in self.seen_hashes:
            self.stats["duplicates"] += 1
            return True
        
        self.seen_hashes.add(sample_hash)
        return False
    
    def validate_sample(self, sample: Dict) -> bool:
        """Validate a single sample."""
        if not self.validate_preference_format(sample):
            return False
        
        if not self.validate_content(sample):
            return False
        
        if self.check_duplicate(sample):
            return False
        
        # Track source
        source = sample.get("source", "unknown")
        self.source_counts[source] += 1
        
        self.stats["valid"] += 1
        return True
    
    def validate_metadata(self, metadata_path: Path) -> Dict:
        """Validate metadata.json from premium loader."""
        if not metadata_path.exists():
            return {"error": "metadata.json not found"}
        
        with open(metadata_path) as f:
            metadata = json.load(f)
        
        # Verify mode
        if metadata.get("mode") != self.mode:
            logger.warning(f"Mode mismatch: expected {self.mode}, got {metadata.get('mode')}")
        
        # Verify ratios sum to 1
        ratios = metadata.get("ratios", {})
        ratio_sum = sum(ratios.values())
        if abs(ratio_sum - 1.0) > 0.01:
            logger.warning(f"Ratios don't sum to 1.0: {ratio_sum}")
        
        return metadata
    
    def get_summary(self) -> Dict:
        """Get validation summary."""
        total = sum(self.stats.values())
        valid = self.stats["valid"]
        
        return {
            "total_checked": total,
            "valid": valid,
            "valid_rate": f"{valid/total*100:.2f}%" if total > 0 else "0%",
            "issues": {k: v for k, v in self.stats.items() if k != "valid"},
            "source_distribution": dict(self.source_counts),
        }


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["censored", "uncensored"], default="censored")
    parser.add_argument("--input-dir", default=None)
    args = parser.parse_args()
    
    log_header(logger, f"VALIDATE PREMIUM DATASETS ({args.mode.upper()})", {
        "Mode": args.mode
    })
    
    validator = PremiumDatasetValidator(args.mode)
    
    # Find dataset directories
    base_dir = Path(args.input_dir) if args.input_dir else Path(CONFIG["premium_base_dir"])
    
    # Look for mode-specific directories
    pattern = f"{args.mode}_*"
    dataset_dirs = list(base_dir.glob(pattern))
    
    if not dataset_dirs:
        logger.warning(f"No premium datasets found matching {pattern} in {base_dir}")
        logger.info("Run 03_load_premium_datasets.py first")
        return
    
    for dataset_dir in dataset_dirs:
        logger.info(f"\nValidating: {dataset_dir}")
        
        # Check metadata
        metadata_path = dataset_dir / "metadata.json"
        metadata = validator.validate_metadata(metadata_path)
        
        if "error" not in metadata:
            logger.info(f"  Mode: {metadata.get('mode')}")
            logger.info(f"  Target samples: {metadata.get('target_samples'):,}")
            logger.info(f"  Actual samples: {metadata.get('actual_samples'):,}")
        
        # Validate samples (if in arrow format, need to load)
        try:
            from datasets import load_from_disk
            dataset = load_from_disk(str(dataset_dir))
            
            logger.info(f"  Loaded {len(dataset):,} samples")
            
            for sample in dataset:
                validator.validate_sample(dict(sample))
            
        except Exception as e:
            logger.error(f"  Could not load dataset: {e}")
    
    # Summary
    summary = validator.get_summary()
    
    log_completion(logger, "Premium Dataset Validation", {
        "Total checked": summary["total_checked"],
        "Valid": summary["valid"],
        "Valid rate": summary["valid_rate"],
    })
    
    # Log source distribution
    if summary["source_distribution"]:
        logger.info("\nSource distribution:")
        for source, count in sorted(summary["source_distribution"].items()):
            logger.info(f"  {source}: {count:,}")


if __name__ == "__main__":
    main()
