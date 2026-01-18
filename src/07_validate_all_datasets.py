#!/usr/bin/env python3
"""
18_validate_all_datasets.py
Universal validation script for all generator outputs

Validates:
- Schema compliance (messages format)
- Content quality (length, structure)
- Deduplication
- Domain distribution

Usage:
  python 18_validate_all_datasets.py
  python 18_validate_all_datasets.py --input /mnt/e/data/specific-dataset
"""

import os
import sys
import json
import hashlib
import logging
from pathlib import Path
from typing import Dict, List, Set, Optional
from collections import defaultdict
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, str(Path(__file__).parent))
from utils.logging_config import setup_logger, log_header, log_completion

# ═══════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════

CONFIG = {
    "data_base_dir": "/mnt/e/data",
    "output_suffix": "_validated",
    "min_content_length": 50,
    "max_content_length": 100000,
    "min_messages": 2,
    "max_messages": 50,
    "num_workers": multiprocessing.cpu_count(),
}

logger = setup_logger(__name__, "logs/validation.log")

# ═══════════════════════════════════════════════════════════════
# VALIDATORS
# ═══════════════════════════════════════════════════════════════

class DatasetValidator:
    """Universal dataset validator."""
    
    def __init__(self):
        self.seen_hashes: Set[str] = set()
        self.stats = defaultdict(int)
        self.domain_counts = defaultdict(int)
    
    def validate_schema(self, sample: Dict) -> bool:
        """Check if sample has valid schema."""
        # Must have messages array
        if "messages" not in sample:
            self.stats["missing_messages"] += 1
            return False
        
        messages = sample["messages"]
        
        # Must be a list
        if not isinstance(messages, list):
            self.stats["invalid_messages_type"] += 1
            return False
        
        # Must have at least 2 messages
        if len(messages) < CONFIG["min_messages"]:
            self.stats["too_few_messages"] += 1
            return False
        
        if len(messages) > CONFIG["max_messages"]:
            self.stats["too_many_messages"] += 1
            return False
        
        # Each message must have role and content
        for msg in messages:
            if not isinstance(msg, dict):
                self.stats["invalid_message_format"] += 1
                return False
            if "role" not in msg or "content" not in msg:
                self.stats["missing_role_or_content"] += 1
                return False
            if msg["role"] not in ["system", "user", "assistant"]:
                self.stats["invalid_role"] += 1
                return False
        
        return True
    
    def validate_content(self, sample: Dict) -> bool:
        """Check content quality."""
        messages = sample.get("messages", [])
        
        total_length = sum(len(str(msg.get("content", ""))) for msg in messages)
        
        if total_length < CONFIG["min_content_length"]:
            self.stats["too_short"] += 1
            return False
        
        if total_length > CONFIG["max_content_length"]:
            self.stats["too_long"] += 1
            return False
        
        # Check for empty messages
        for msg in messages:
            content = str(msg.get("content", ""))
            if not content.strip():
                self.stats["empty_content"] += 1
                return False
        
        return True
    
    def check_duplicate(self, sample: Dict) -> bool:
        """Check if sample is duplicate."""
        content = json.dumps(sample.get("messages", []), sort_keys=True)
        sample_hash = hashlib.md5(content.encode()).hexdigest()
        
        if sample_hash in self.seen_hashes:
            self.stats["duplicates"] += 1
            return True
        
        self.seen_hashes.add(sample_hash)
        return False
    
    def validate_sample(self, sample: Dict) -> Optional[Dict]:
        """Validate a single sample. Returns sample if valid, None otherwise."""
        # Schema validation
        if not self.validate_schema(sample):
            return None
        
        # Content validation
        if not self.validate_content(sample):
            return None
        
        # Deduplication
        if self.check_duplicate(sample):
            return None
        
        # Track domain
        domain = sample.get("domain", sample.get("source", "unknown"))
        self.domain_counts[domain] += 1
        
        self.stats["valid"] += 1
        return sample
    
    def get_summary(self) -> Dict:
        """Get validation summary."""
        total_checked = sum(self.stats.values())
        valid = self.stats["valid"]
        
        return {
            "total_checked": total_checked,
            "valid": valid,
            "valid_rate": f"{valid/total_checked*100:.2f}%" if total_checked > 0 else "0%",
            "rejection_reasons": {k: v for k, v in self.stats.items() if k != "valid"},
            "domains": dict(self.domain_counts),
        }


# ═══════════════════════════════════════════════════════════════
# FILE PROCESSORS
# ═══════════════════════════════════════════════════════════════

def validate_jsonl_file(input_file: Path, output_file: Path) -> Dict:
    """Validate a single JSONL file."""
    validator = DatasetValidator()
    valid_samples = []
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    sample = json.loads(line)
                    validated = validator.validate_sample(sample)
                    if validated:
                        valid_samples.append(validated)
                except json.JSONDecodeError:
                    validator.stats["json_error"] += 1
    except Exception as e:
        logger.error(f"Error reading {input_file}: {e}")
        return {"error": str(e)}
    
    # Write validated samples
    if valid_samples:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            for sample in valid_samples:
                f.write(json.dumps(sample) + "\n")
    
    return {
        "input": str(input_file),
        "output": str(output_file),
        "summary": validator.get_summary()
    }


def validate_directory(data_dir: Path) -> Dict:
    """Validate all JSONL files in a directory."""
    results = []
    total_valid = 0
    total_checked = 0
    
    jsonl_files = list(data_dir.rglob("*.jsonl"))
    
    # Skip already validated files
    jsonl_files = [f for f in jsonl_files if "_validated" not in f.name]
    
    if not jsonl_files:
        return {"message": f"No JSONL files found in {data_dir}"}
    
    logger.info(f"Found {len(jsonl_files)} files to validate in {data_dir}")
    
    for input_file in jsonl_files:
        # Create output filename
        output_file = input_file.parent / f"{input_file.stem}_validated.jsonl"
        
        result = validate_jsonl_file(input_file, output_file)
        results.append(result)
        
        if "summary" in result:
            total_valid += result["summary"]["valid"]
            total_checked += result["summary"]["total_checked"]
    
    return {
        "directory": str(data_dir),
        "files_processed": len(jsonl_files),
        "total_checked": total_checked,
        "total_valid": total_valid,
        "valid_rate": f"{total_valid/total_checked*100:.2f}%" if total_checked > 0 else "0%",
    }


# ═══════════════════════════════════════════════════════════════
# BENCHMARK VALIDATOR
# ═══════════════════════════════════════════════════════════════

def validate_benchmarks() -> Dict:
    """Validate benchmark datasets."""
    benchmark_dir = Path(CONFIG["data_base_dir"]) / "benchmarks"
    
    if not benchmark_dir.exists():
        return {"error": "Benchmarks directory not found"}
    
    return validate_directory(benchmark_dir)


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    log_header(logger, "UNIVERSAL DATASET VALIDATOR", {
        "Base Dir": CONFIG["data_base_dir"],
        "Workers": CONFIG["num_workers"]
    })
    
    # Check for specific input
    specific_input = None
    for arg in sys.argv:
        if arg.startswith("--input="):
            specific_input = Path(arg.split("=")[1])
    
    if specific_input:
        # Validate specific directory
        if specific_input.is_file():
            output = specific_input.parent / f"{specific_input.stem}_validated.jsonl"
            result = validate_jsonl_file(specific_input, output)
        else:
            result = validate_directory(specific_input)
        logger.info(f"Result: {json.dumps(result, indent=2)}")
    else:
        # Validate all data directories
        base_dir = Path(CONFIG["data_base_dir"])
        
        all_results = {}
        
        # Validate each category
        for category in ["predistilled", "code", "domain", "benchmarks"]:
            cat_dir = base_dir / category
            if cat_dir.exists():
                logger.info(f"\n{'='*60}")
                logger.info(f"Validating: {category}")
                logger.info(f"{'='*60}")
                
                result = validate_directory(cat_dir)
                all_results[category] = result
                
                logger.info(f"  Valid: {result.get('total_valid', 0)}")
                logger.info(f"  Rate: {result.get('valid_rate', '0%')}")
        
        # Summary
        logger.info("\n" + "="*60)
        logger.info("VALIDATION SUMMARY")
        logger.info("="*60)
        
        total_valid = sum(r.get("total_valid", 0) for r in all_results.values())
        total_checked = sum(r.get("total_checked", 0) for r in all_results.values())
        
        logger.info(f"Total validated: {total_valid}")
        logger.info(f"Total checked: {total_checked}")
        logger.info(f"Overall rate: {total_valid/total_checked*100:.2f}%" if total_checked > 0 else "0%")
    
    log_completion(logger, "Dataset Validation", {"status": "complete"})


if __name__ == "__main__":
    main()
