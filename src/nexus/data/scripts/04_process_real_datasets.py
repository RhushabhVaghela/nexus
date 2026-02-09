#!/usr/bin/env python3
"""
03_process_real_datasets.py
Unified Real Data Processor

Processes all downloaded real datasets from /mnt/e/data/ and normalizes
them to the OpenAI messages format for training.

This REPLACES all synthetic generators (old 03-18) with pure real data processing.

Usage:
    python 03_process_real_datasets.py
    python 03_process_real_datasets.py --category=code
    python 03_process_real_datasets.py --category=domain
"""

import os
import sys
import json
import hashlib
import logging
from pathlib import Path
from typing import Dict, List, Optional, Generator
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import random

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

# ═══════════════════════════════════════════════════════════════
# DATA CATEGORIES AND MAPPINGS
# ═══════════════════════════════════════════════════════════════

CATEGORY_MAPPINGS = {
    # Pre-distilled knowledge datasets
    "predistilled": {
        "source_dirs": ["predistilled/magicoder", "predistilled/openmath", 
                       "predistilled/slimorca", "predistilled/dolphin"],
        "extensions": [".jsonl", ".json"],
        "domain": "instruction"
    },
    # Code datasets
    "code": {
        "source_dirs": ["code/code-feedback", "code/code-alpaca", 
                       "code/glaive-code", "code/commitpack"],
        "extensions": [".jsonl", ".json", ".py", ".js", ".ts", ".java", ".go", ".rs"],
        "domain": "code"
    },
    # Domain-specific from GitHub
    "devops": {
        "source_dirs": ["domain/terraform", "domain/ansible", "domain/kubernetes"],
        "extensions": [".tf", ".yaml", ".yml", ".json"],
        "domain": "devops"
    },
    "platform": {
        "source_dirs": ["domain/backstage"],
        "extensions": [".ts", ".tsx", ".yaml", ".json"],
        "domain": "platform"
    },
    "mobile": {
        "source_dirs": ["domain/flutter-samples", "domain/swift-composable", 
                       "domain/android-sunflower"],
        "extensions": [".dart", ".swift", ".kt", ".java"],
        "domain": "mobile"
    },
    "api": {
        "source_dirs": ["domain/openapi", "domain/graphql", "domain/grpc"],
        "extensions": [".yaml", ".yml", ".json", ".graphql", ".proto"],
        "domain": "api"
    },
    "observability": {
        "source_dirs": ["domain/otel", "domain/grafana"],
        "extensions": [".yaml", ".json", ".ts", ".py"],
        "domain": "observability"
    },
    "data_engineering": {
        "source_dirs": ["domain/airflow", "domain/dbt"],
        "extensions": [".py", ".sql", ".yaml", ".yml"],
        "domain": "data_engineering"
    },
    "security": {
        "source_dirs": ["domain/owasp"],
        "extensions": [".md", ".py", ".yaml"],
        "domain": "security"
    },
}

# ═══════════════════════════════════════════════════════════════
# MESSAGE FORMAT CONVERTERS
# ═══════════════════════════════════════════════════════════════

class RealDataProcessor:
    """Processes real datasets into messages format."""
    
    def __init__(self):
        self.seen_hashes = set()
        self.stats = {"processed": 0, "skipped": 0, "duplicates": 0}
    
    def normalize_jsonl_sample(self, sample: Dict, source: str) -> Optional[Dict]:
        """Convert various JSONL formats to messages format."""
        
        # Already in messages format
        if "messages" in sample and isinstance(sample["messages"], list):
            return self.validate_and_dedupe(sample, source)
        
        # Instruction/Response format (common in HuggingFace)
        if "instruction" in sample:
            messages = [
                {"role": "user", "content": sample["instruction"]}
            ]
            if "input" in sample and sample["input"]:
                messages[0]["content"] += f"\n\nInput: {sample['input']}"
            if "output" in sample:
                messages.append({"role": "assistant", "content": sample["output"]})
            elif "response" in sample:
                messages.append({"role": "assistant", "content": sample["response"]})
            
            return self.validate_and_dedupe({"messages": messages, "source": source}, source)
        
        # Question/Answer format
        if "question" in sample and "answer" in sample:
            return self.validate_and_dedupe({
                "messages": [
                    {"role": "user", "content": sample["question"]},
                    {"role": "assistant", "content": sample["answer"]}
                ],
                "source": source
            }, source)
        
        # Prompt/Response format
        if "prompt" in sample and ("response" in sample or "completion" in sample):
            response = sample.get("response") or sample.get("completion")
            return self.validate_and_dedupe({
                "messages": [
                    {"role": "user", "content": sample["prompt"]},
                    {"role": "assistant", "content": response}
                ],
                "source": source
            }, source)
        
        # Problem/Solution format (math/code)
        if "problem" in sample and "solution" in sample:
            return self.validate_and_dedupe({
                "messages": [
                    {"role": "user", "content": sample["problem"]},
                    {"role": "assistant", "content": sample["solution"]}
                ],
                "source": source
            }, source)
        
        # Text only - skip
        self.stats["skipped"] += 1
        return None
    
    def normalize_code_file(self, file_path: Path, content: str) -> Optional[Dict]:
        """Convert a code file to messages format."""
        
        ext = file_path.suffix.lstrip(".")
        relative_path = file_path.name
        
        # Determine language
        lang_map = {
            "py": "python", "js": "javascript", "ts": "typescript",
            "tsx": "tsx", "java": "java", "go": "go", "rs": "rust",
            "dart": "dart", "swift": "swift", "kt": "kotlin",
            "tf": "terraform", "yaml": "yaml", "yml": "yaml",
            "json": "json", "sql": "sql", "graphql": "graphql",
            "proto": "protobuf", "md": "markdown"
        }
        language = lang_map.get(ext, ext)
        
        # Create instruction prompt based on file type
        prompts = {
            "python": f"Write a Python file for: {relative_path}",
            "typescript": f"Create a TypeScript file: {relative_path}",
            "terraform": f"Write Terraform configuration: {relative_path}",
            "yaml": f"Create this configuration file: {relative_path}",
            "dart": f"Write Flutter/Dart code: {relative_path}",
            "swift": f"Create Swift code: {relative_path}",
            "kotlin": f"Write Kotlin code: {relative_path}",
            "sql": f"Write SQL: {relative_path}",
        }
        prompt = prompts.get(language, f"Create this file: {relative_path}")
        
        sample = {
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": f"```{language}\n{content}\n```"}
            ],
            "source": f"github/{file_path.parts[-3] if len(file_path.parts) > 3 else 'unknown'}",
            "domain": language,
        }
        
        return self.validate_and_dedupe(sample, sample["source"])
    
    def validate_and_dedupe(self, sample: Dict, source: str) -> Optional[Dict]:
        """Validate and deduplicate a sample."""
        
        messages = sample.get("messages", [])
        
        # Validate structure
        if len(messages) < 2:
            self.stats["skipped"] += 1
            return None
        
        # Validate content length
        total_length = sum(len(str(m.get("content", ""))) for m in messages)
        if total_length < CONFIG["min_content_length"]:
            self.stats["skipped"] += 1
            return None
        if total_length > CONFIG["max_content_length"]:
            self.stats["skipped"] += 1
            return None
        
        # Check for duplicates
        content_hash = hashlib.md5(json.dumps(messages, sort_keys=True).encode()).hexdigest()
        if content_hash in self.seen_hashes:
            self.stats["duplicates"] += 1
            return None
        self.seen_hashes.add(content_hash)
        
        # Add metadata
        sample["id"] = content_hash[:12]
        sample["source"] = source
        
        self.stats["processed"] += 1
        return sample
    
    def process_jsonl_file(self, file_path: Path) -> Generator[Dict, None, None]:
        """Process a JSONL file and yield normalized samples."""
        
        source = file_path.parent.name
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        sample = json.loads(line)
                        normalized = self.normalize_jsonl_sample(sample, source)
                        if normalized:
                            yield normalized
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
    
    def process_code_file(self, file_path: Path) -> Optional[Dict]:
        """Process a single code file."""
        
        try:
            content = file_path.read_text(encoding='utf-8', errors='ignore')
            
            # Skip files that are too short or too long
            if len(content) < CONFIG["min_content_length"]:
                return None
            if len(content) > CONFIG["max_content_length"]:
                return None
            
            return self.normalize_code_file(file_path, content)
        except Exception as e:
            return None
    
    def process_category(self, category: str, category_config: Dict) -> int:
        """Process all data for a category."""
        
        output_dir = Path(CONFIG["output_base_dir"]) / category
        output_dir.mkdir(parents=True, exist_ok=True)
        
        samples = []
        batch_num = 0
        total_count = 0
        
        for source_dir_name in category_config["source_dirs"]:
            source_dir = Path(CONFIG["data_base_dir"]) / source_dir_name
            
            if not source_dir.exists():
                logger.warning(f"Source directory not found: {source_dir}")
                continue
            
            logger.info(f"Processing {source_dir}...")
            
            for ext in category_config["extensions"]:
                for file_path in source_dir.rglob(f"*{ext}"):
                    if ext in [".jsonl", ".json"]:
                        for sample in self.process_jsonl_file(file_path):
                            samples.append(sample)
                            # Detect existing split in file path
                            existing_split = self._detect_split(file_path)
                            
                            if len(samples) >= CONFIG["samples_per_file"]:
                                self._process_and_write(output_dir, samples, batch_num, existing_split)
                                total_count += len(samples)
                                samples = []
                                batch_num += 1
                    else:
                        sample = self.process_code_file(file_path)
                        if sample:
                            samples.append(sample)
                            existing_split = self._detect_split(file_path)
                            
                            if len(samples) >= CONFIG["samples_per_file"]:
                                self._process_and_write(output_dir, samples, batch_num, existing_split)
                                total_count += len(samples)
                                samples = []
                                batch_num += 1
        
        # Write remaining samples
        if samples:
            # For remaining, we can't easily track mixed splits if they were mixed in the loop? 
            # Ideally samples list should only contain one type if we process file by file.
            # But here we accumulate across files.
            # Simplified assumption: last file's split applies, or default to split logic.
            # To be safe for mixed batch, we force random split if ambiguous.
            self._process_and_write(output_dir, samples, batch_num, None)
            total_count += len(samples)
        
        return total_count

    def _detect_split(self, file_path: Path) -> Optional[str]:
        """Detect if file belongs to a specific split based on path."""
        parts = {p.lower() for p in file_path.parts}
        if "test" in parts or "testing" in parts:
            return "test"
        if "val" in parts or "validation" in parts or "dev" in parts:
            return "val"
        if "train" in parts or "training" in parts:
            return "train"
        return None

    def _process_and_write(self, output_dir: Path, samples: List[Dict], batch_num: int, fixed_split: Optional[str] = None):
        """Write batch, either using fixed split or random splitting."""
        if fixed_split:
             self._write_batch(output_dir, samples, batch_num, fixed_split)
        else:
             self._process_and_write_split(output_dir, samples, batch_num)
    
    def _process_and_write_split(self, output_dir: Path, samples: List[Dict], batch_num: int):
        """Shuffle and split batch into train/val/test."""
        random.shuffle(samples)
        
        total = len(samples)
        train_end = int(total * 0.95)
        val_end = int(total * 0.975)
        
        self._write_batch(output_dir, samples[:train_end], batch_num, "train")
        self._write_batch(output_dir, samples[train_end:val_end], batch_num, "val")
        self._write_batch(output_dir, samples[val_end:], batch_num, "test")

    def _write_batch(self, output_dir: Path, samples: List[Dict], batch_num: int, split: str = "train"):
        """Write a batch of samples to a JSONL file."""
        
        output_file = output_dir / split / f"part_{batch_num:04d}.jsonl"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
        
        logger.info(f"Wrote {len(samples)} samples to {output_file}")


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    if not check_env():
        sys.exit(1)
        
    global CONFIG, logger
    CONFIG = {
        "data_base_dir": "/mnt/e/data",
        "output_base_dir": "/mnt/e/data/processed",
        "samples_per_file": 100_000,
        "min_content_length": 50,
        "max_content_length": 100_000,
        "num_workers": multiprocessing.cpu_count(),
    }
    logger = setup_logger(__name__, "logs/process_real_datasets.log")

    log_header(logger, "REAL DATA PROCESSOR", {
        "Data Dir": CONFIG["data_base_dir"],
        "Output Dir": CONFIG["output_base_dir"],
        "Workers": CONFIG["num_workers"]
    })
    
    # Parse category filter
    category_filter = None
    for arg in sys.argv:
        if arg.startswith("--category="):
            category_filter = arg.split("=")[1]
    
    processor = RealDataProcessor()
    
    total_samples = 0
    
    for category, config in CATEGORY_MAPPINGS.items():
        if category_filter and category != category_filter:
            continue
        
        logger.info(f"\n{'='*50}")
        logger.info(f"Processing category: {category}")
        logger.info(f"{'='*50}")
        
        count = processor.process_category(category, config)
        total_samples += count
        
        logger.info(f"Category {category}: {count} samples")
    
    log_completion(logger, "Real Data Processing", {
        "Total samples": total_samples,
        "Processed": processor.stats["processed"],
        "Skipped": processor.stats["skipped"],
        "Duplicates": processor.stats["duplicates"],
    })


if __name__ == "__main__":
    main()
