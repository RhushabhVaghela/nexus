#!/usr/bin/env python3
"""
21_validate_benchmarks.py
Benchmark-specific validation for evaluation datasets

Validates:
- HumanEval (code correctness)
- MBPP (code generation)
- GSM8K (math reasoning)
- MMLU (knowledge)

Usage:
  python 21_validate_benchmarks.py
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).parent))
from utils.logging_config import setup_logger, log_header, log_completion

def check_env():
    """Verify environment dependencies."""
    if os.environ.get("CONDA_DEFAULT_ENV") != "nexus":
        print("[ERROR] Must be run in 'nexus' conda environment.")
        return False
    return True

# Globals to be initialized in main()
CONFIG = None
logger = None

class BenchmarkValidator:
    """Validates specific benchmark formats."""
    
    def __init__(self):
        self.stats = {
            "humaneval": {"valid": 0, "invalid": 0},
            "mbpp": {"valid": 0, "invalid": 0},
            "gsm8k": {"valid": 0, "invalid": 0},
            "mmlu": {"valid": 0, "invalid": 0},
        }
    
    def validate_humaneval(self, sample: Dict) -> bool:
        """Validate HumanEval format."""
        required = ["task_id", "prompt", "entry_point", "test"]
        
        for field in required:
            if field not in sample:
                return False
        
        # Check prompt contains function signature
        if "def " not in sample.get("prompt", ""):
            return False
        
        # Check test is not empty
        if not sample.get("test", "").strip():
            return False
        
        return True
    
    def validate_mbpp(self, sample: Dict) -> bool:
        """Validate MBPP format."""
        required = ["task_id", "text", "code", "test_list"]
        
        for field in required:
            if field not in sample:
                return False
        
        # Must have at least one test
        if not sample.get("test_list") or len(sample["test_list"]) == 0:
            return False
        
        return True
    
    def validate_gsm8k(self, sample: Dict) -> bool:
        """Validate GSM8K format."""
        # Must have question and answer
        if "question" not in sample and "problem" not in sample:
            return False
        
        if "answer" not in sample and "solution" not in sample:
            return False
        
        # Answer should contain a number
        answer = str(sample.get("answer", sample.get("solution", "")))
        if not any(c.isdigit() for c in answer):
            return False
        
        return True
    
    def validate_mmlu(self, sample: Dict) -> bool:
        """Validate MMLU format."""
        # Must have question and choices
        if "question" not in sample:
            return False
        
        # Must have choices/options
        choices = sample.get("choices", sample.get("options", []))
        if not choices or len(choices) < 2:
            return False
        
        # Must have correct answer
        if "answer" not in sample and "correct" not in sample:
            return False
        
        return True
    
    def validate_sample(self, sample: Dict, benchmark_type: str) -> bool:
        """Validate a sample based on benchmark type."""
        validators = {
            "humaneval": self.validate_humaneval,
            "mbpp": self.validate_mbpp,
            "gsm8k": self.validate_gsm8k,
            "mmlu": self.validate_mmlu,
        }
        
        validator = validators.get(benchmark_type.lower())
        if validator is None:
            return True  # Unknown type, pass through
        
        is_valid = validator(sample)
        
        if is_valid:
            self.stats[benchmark_type.lower()]["valid"] += 1
        else:
            self.stats[benchmark_type.lower()]["invalid"] += 1
        
        return is_valid


def detect_benchmark_type(filename: str, sample: Dict) -> str:
    """Detect benchmark type from filename or content."""
    filename_lower = filename.lower()
    
    if "humaneval" in filename_lower:
        return "humaneval"
    elif "mbpp" in filename_lower:
        return "mbpp"
    elif "gsm8k" in filename_lower or "gsm" in filename_lower:
        return "gsm8k"
    elif "mmlu" in filename_lower:
        return "mmlu"
    
    # Detect from content
    if "task_id" in sample and "entry_point" in sample:
        return "humaneval"
    elif "task_id" in sample and "test_list" in sample:
        return "mbpp"
    elif "choices" in sample or "options" in sample:
        return "mmlu"
    elif any(k in sample for k in ["question", "problem"]):
        return "gsm8k"
    
    return "unknown"


def validate_benchmark_file(input_file: Path, output_dir: Path) -> Dict:
    """Validate a benchmark file."""
    validator = BenchmarkValidator()
    valid_samples = []
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    sample = json.loads(line)
                    benchmark_type = detect_benchmark_type(input_file.name, sample)
                    
                    if validator.validate_sample(sample, benchmark_type):
                        sample["_benchmark_type"] = benchmark_type
                        valid_samples.append(sample)
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        logger.error(f"Error reading {input_file}: {e}")
        return {"error": str(e)}
    
    # Write validated
    if valid_samples:
        output_file = output_dir / f"{input_file.stem}_validated.jsonl"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for sample in valid_samples:
                f.write(json.dumps(sample) + "\n")
        
        logger.info(f"Validated {len(valid_samples)} samples -> {output_file}")
    
    return {
        "input": str(input_file),
        "valid_samples": len(valid_samples),
        "stats": validator.stats
    }


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    if not check_env():
        sys.exit(1)
        
    global CONFIG, logger
    CONFIG = {
        "benchmarks_dir": "/mnt/e/data/benchmarks",
        "output_dir": "/mnt/e/data/benchmarks/validated",
    }
    logger = setup_logger(__name__, "logs/validate_benchmarks.log")
    
    log_header(logger, "BENCHMARK VALIDATION", {
        "Input": CONFIG["benchmarks_dir"],
        "Output": CONFIG["output_dir"]
    })
    
    benchmarks_dir = Path(CONFIG["benchmarks_dir"])
    output_dir = Path(CONFIG["output_dir"])
    
    if not benchmarks_dir.exists():
        logger.warning(f"Benchmarks directory not found: {benchmarks_dir}")
        logger.info("Run download_and_normalize.py first to download benchmarks")
        return
    
    # Find all JSONL files
    jsonl_files = list(benchmarks_dir.rglob("*.jsonl"))
    jsonl_files = [f for f in jsonl_files if "_validated" not in f.name]
    
    if not jsonl_files:
        logger.warning("No benchmark files found")
        return
    
    logger.info(f"Found {len(jsonl_files)} benchmark files")
    
    total_valid = 0
    for jsonl_file in jsonl_files:
        result = validate_benchmark_file(jsonl_file, output_dir)
        total_valid += result.get("valid_samples", 0)
    
    log_completion(logger, "Benchmark Validation", {
        "files_processed": len(jsonl_files),
        "total_valid_samples": total_valid
    })


if __name__ == "__main__":
    main()
