#!/usr/bin/env python3
"""
FILE 3: 03_validate_trajectories.py
Quality filtering and validation
Output: cold_start_filtered.jsonl
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any
from collections import defaultdict
import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/validate_trajectories.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def validate_trajectory(sample: Dict[str, Any]) -> tuple[bool, str]:
    """Validate single trajectory. Returns (is_valid, reason)"""
    
    # Check required fields
    if "user_query" not in sample:
        return False, "Missing user_query"
    if "trajectory" not in sample:
        return False, "Missing trajectory"
    if "domain" not in sample:
        return False, "Missing domain"
    
    trajectory = sample["trajectory"]
    
    # Check trajectory is list
    if not isinstance(trajectory, list):
        return False, "Trajectory not a list"
    
    # Check length
    if len(trajectory) < 5:
        return False, "Trajectory too short (<5 steps)"
    if len(trajectory) > 20:
        return False, "Trajectory too long (>20 steps)"
    
    # Check for required step types
    step_types = {step.get("type") for step in trajectory}
    
    if "think" not in step_types:
        return False, "Missing 'think' step"
    if "action" not in step_types:
        return False, "Missing 'action' step"
    if "final_answer" not in step_types:
        return False, "Missing 'final_answer' step"
    
    # Check for error+recovery pattern (preferred)
    has_error = "error" in step_types
    has_recovery = "recovery" in step_types
    
    if has_error and not has_recovery:
        return False, "Error without recovery"
    
    # Check domain
    valid_domains = {"math", "code", "fullstack", "analysis"}
    if sample.get("domain") not in valid_domains:
        return False, f"Invalid domain: {sample.get('domain')}"
    
    # Check query clarity
    query = sample.get("user_query", "")
    if len(query) < 30:
        return False, "Query too short"
    
    return True, "Valid"

def main():
    logger.info("="*70)
    logger.info("✅ VALIDATING TRAJECTORIES")
    logger.info("="*70)
    
    # Check input
    if not Path("cold_start_trajectories.jsonl").exists():
        logger.error("❌ cold_start_trajectories.jsonl not found")
        logger.error("   Run: python 02_generate_trajectories.py")
        return
    
    # Load and validate
    logger.info("\n📂 Loading trajectories...")
    trajectories = []
    with open("cold_start_trajectories.jsonl", "r") as f:
        for line in f:
            if line.strip():
                try:
                    traj = json.loads(line)
                    trajectories.append(traj)
                except json.JSONDecodeError:
                    pass
    
    logger.info(f"✓ Loaded {len(trajectories)} trajectories")
    
    # Validate
    logger.info("\n🔍 Validating...")
    valid = []
    invalid_reasons = defaultdict(int)
    
    for traj in tqdm.tqdm(trajectories, desc="Validating"):
        is_valid, reason = validate_trajectory(traj)
        if is_valid:
            valid.append(traj)
        else:
            invalid_reasons[reason] += 1
    
    # Save valid
    output_path = Path("cold_start_filtered.jsonl")
    with open(output_path, "w") as f:
        for traj in valid:
            f.write(json.dumps(traj) + "\n")
    
    # Report
    logger.info("\n" + "="*70)
    logger.info("✅ VALIDATION COMPLETE")
    logger.info("="*70)
    logger.info(f"Valid: {len(valid)}")
    logger.info(f"Invalid: {len(trajectories) - len(valid)}")
    logger.info(f"Pass rate: {len(valid)/len(trajectories)*100:.1f}%")
    logger.info(f"Output: {output_path}")
    
    if invalid_reasons:
        logger.info(f"\nInvalid reasons:")
        for reason, count in sorted(invalid_reasons.items(), key=lambda x: -x[1])[:5]:
            logger.info(f"  {reason}: {count}")
    
    # Domain distribution
    domains = defaultdict(int)
    for traj in valid:
        domains[traj.get("domain")] += 1
    
    logger.info(f"\nDomain distribution:")
    for domain, count in sorted(domains.items()):
        logger.info(f"  {domain}: {count}")
    
    logger.info("="*70)
    logger.info(f"\nNext: Run SFT Training")
    logger.info(f"  python 04_sft_training.py")

if __name__ == "__main__":
    main()
