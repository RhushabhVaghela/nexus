#!/usr/bin/env python3
"""
download_and_normalize.py
Downloads real datasets and normalizes to OpenAI messages format.

Strategy: 100% Real Data, Sequential processing, Multicore per dataset
"""

import os
import sys
import yaml
import json
import hashlib
import multiprocessing
from pathlib import Path
from typing import Dict, List, Any, Optional
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import subprocess
import shutil

# ═══════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════

BASE_DIR = Path("/mnt/e/data")
CONFIG_PATH = Path(__file__).parent / "config" / "datasets.yaml"
NUM_WORKERS = multiprocessing.cpu_count()

# ═══════════════════════════════════════════════════════════════
# LOGGING
# ═══════════════════════════════════════════════════════════════

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/download_normalize.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

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
    
    # Code format: prompt, completion / solution
    elif "prompt" in sample and ("completion" in sample or "solution" in sample):
        messages = [
            {"role": "user", "content": sample["prompt"]},
            {"role": "assistant", "content": sample.get("completion") or sample.get("solution", "")}
        ]
    
    # Question-Answer format
    elif "question" in sample and "answer" in sample:
        messages = [
            {"role": "user", "content": sample["question"]},
            {"role": "assistant", "content": sample["answer"]}
        ]
        
    # Problem-Solution format (math datasets)
    elif "problem" in sample and "solution" in sample:
        messages = [
            {"role": "user", "content": sample["problem"]},
            {"role": "assistant", "content": sample["solution"]}
        ]

    # OpenMathInstruct (question, generated_solution, expected_answer)
    elif "question" in sample and "generated_solution" in sample:
        messages = [
            {"role": "user", "content": sample["question"]},
            {"role": "assistant", "content": sample["generated_solution"]}
        ]
        
    # CodeAlpaca (instruction, input, output)
    elif "instruction" in sample and "input" in sample and "output" in sample:
        user_msg = sample["instruction"]
        if sample["input"]:
            user_msg += f"\nInput:\n{sample['input']}"
        messages = [
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": sample["output"]}
        ]

    # CommitPackFT (old_contents, new_contents, message)
    elif "new_contents" in sample and "message" in sample:
        lang = sample.get("lang", "code")
        prompt = f"Commit Message: {sample['message']}\n"
        if sample.get("old_contents"):
             prompt += f"Update the following {lang} code:\n```{lang}\n{sample['old_contents']}\n```"
        else:
             prompt += f"Write a new {lang} file:"
             
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": f"```{lang}\n{sample['new_contents']}\n```"}
        ]

    # Text only (code files from The Stack)
    elif "content" in sample:
        # Create instruction from code
        lang = sample.get("lang", sample.get("language", "code"))
        messages = [
            {"role": "user", "content": f"Write {lang} code that implements the following:"},
            {"role": "assistant", "content": sample["content"]}
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
    }
    
    return normalized

# ═══════════════════════════════════════════════════════════════
# DATASET DOWNLOADERS
# ═══════════════════════════════════════════════════════════════

def download_huggingface(source: str, output_dir: Path, limit: int = 200000, **kwargs) -> bool:
    """Download dataset from HuggingFace."""
    try:
        from datasets import load_dataset
        
        logger.info(f"📥 Downloading {source} from HuggingFace (Streaming) (Limit: {limit})...")
        
        # Load dataset in streaming mode
        ds = load_dataset(source, streaming=True)
        
        # Get the main split
        if "train" in ds:
            split_data = ds["train"]
        else:
            split_data = ds[list(ds.keys())[0]]
            
        MAX_RAW_SAMPLES = limit
        
        logger.info(f"   Streaming up to {MAX_RAW_SAMPLES} samples...")
        
        # Normalize and save
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / "data.jsonl"
        
        count = 0
        with open(output_file, 'w', encoding='utf-8') as f:
            for i, sample in enumerate(split_data):
                if i >= MAX_RAW_SAMPLES:
                    break
                    
                normalized = normalize_to_messages(dict(sample), source=source)
                if normalized:
                    f.write(json.dumps(normalized, ensure_ascii=False) + "\n")
                    count += 1
        
        logger.info(f"   ✅ Saved {count} samples to {output_file}")
        return True
        
    except Exception as e:
        logger.error(f"   ❌ Failed to download {source}: {e}")
        return False

def download_github(source: str, output_dir: Path, **kwargs) -> bool:
    """Clone GitHub repository."""
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        
        repo_name = source.split("/")[-1]
        clone_dir = output_dir / repo_name
        
        if clone_dir.exists():
            logger.info(f"   ⏭️ {repo_name} already exists, skipping...")
            return True
        
        logger.info(f"📥 Cloning {source}...")
        
        result = subprocess.run(
            ["git", "clone", "--depth", "1", source, str(clone_dir)],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            logger.info(f"   ✅ Cloned to {clone_dir}")
            return True
        else:
            logger.error(f"   ❌ Clone failed: {result.stderr}")
            return False
            
    except Exception as e:
        logger.error(f"   ❌ Failed to clone {source}: {e}")
        return False

def process_github_to_jsonl(repo_dir: Path, output_file: Path, extensions: List[str] = None) -> int:
    """Convert GitHub repo files to JSONL format."""
    if extensions is None:
        extensions = [".py", ".js", ".ts", ".jsx", ".tsx", ".dart", ".swift", ".kt", 
                     ".yaml", ".yml", ".json", ".md", ".dockerfile", ".tf"]
    
    count = 0
    with open(output_file, 'w', encoding='utf-8') as f:
        for ext in extensions:
            for file_path in repo_dir.rglob(f"*{ext}"):
                try:
                    content = file_path.read_text(encoding='utf-8', errors='ignore')
                    if len(content) < 50 or len(content) > 100000:
                        continue
                    
                    sample = {
                        "id": hashlib.md5(content.encode()).hexdigest()[:12],
                        "messages": [
                            {"role": "user", "content": f"Write a {ext.lstrip('.')} file:"},
                            {"role": "assistant", "content": content}
                        ],
                        "source": str(repo_dir.name),
                        "file": str(file_path.relative_to(repo_dir))
                    }
                    f.write(json.dumps(sample, ensure_ascii=False) + "\n")
                    count += 1
                except Exception:
                    continue
    
    return count

# ═══════════════════════════════════════════════════════════════
# MAIN PROCESSING
# ═══════════════════════════════════════════════════════════════

def process_dataset(dataset_config: Dict, limit: int = 200000) -> bool:
    """Process a single dataset configuration."""
    name = dataset_config.get("name", "unknown")
    source = dataset_config.get("source")
    dtype = dataset_config.get("type", "huggingface")
    output_dir = BASE_DIR / dataset_config.get("output_dir", name)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Processing: {name}")
    logger.info(f"{'='*60}")
    
    if dtype == "huggingface":
        return download_huggingface(source, output_dir, limit=limit)
    elif dtype == "github":
        success = download_github(source, output_dir)
        if success:
            # Convert to JSONL
            repo_name = source.split("/")[-1]
            clone_dir = output_dir / repo_name
            jsonl_file = output_dir / "data.jsonl"
            count = process_github_to_jsonl(clone_dir, jsonl_file)
            logger.info(f"   ✅ Converted {count} files to {jsonl_file}")
        return success
    else:
        logger.error(f"Unknown dataset type: {dtype}")
        return False

def load_config() -> Dict:
    """Load dataset configuration."""
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, 'r') as f:
            return yaml.safe_load(f)
    else:
        logger.error(f"Config not found: {CONFIG_PATH}")
        sys.exit(1)

def main():
    """Main entry point - sequential processing with progress."""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=200000, help="Max samples per dataset")
    args = parser.parse_args()
    
    os.makedirs('logs', exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("🚀 MANUS PRIME: Real Data Download & Normalize")
    logger.info("=" * 60)
    logger.info(f"Output directory: {BASE_DIR}")
    logger.info(f"Workers per dataset: {NUM_WORKERS}")
    logger.info(f"Limit per dataset: {args.limit}")
    
    config = load_config()
    
    # Collect all datasets
    all_datasets = []
    
    # Priority order: predistilled → code → domains → benchmarks
    for category in ["predistilled", "code", "domains", "benchmarks"]:
        datasets = config.get(category, [])
        if datasets:
            logger.info(f"\n📦 Category: {category.upper()} ({len(datasets)} datasets)")
            all_datasets.extend(datasets)
    
    # Process sequentially
    total = len(all_datasets)
    success = 0
    failed = 0
    
    for i, dataset in enumerate(all_datasets, 1):
        logger.info(f"\n[{i}/{total}] Processing...")
        
        try:
            if process_dataset(dataset, limit=args.limit):
                success += 1
            else:
                failed += 1
        except Exception as e:
            logger.error(f"Error processing {dataset.get('name')}: {e}")
            failed += 1
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("📊 SUMMARY")
    logger.info("=" * 60)
    logger.info(f"   Total: {total}")
    logger.info(f"   Success: {success}")
    logger.info(f"   Failed: {failed}")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()
