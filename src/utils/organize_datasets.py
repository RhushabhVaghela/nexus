#!/usr/bin/env python3
"""
organize_datasets.py

Utility to organize unstructured datasets into categorized subdirectories.
Supports separating Benchmarks from Training Data.

Usage:
    python src/utils/organize_datasets.py --base-path /mnt/e/data --move
    python src/utils/organize_datasets.py --base-path /mnt/e/data --dry-run
"""

import os
import shutil
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Categories Configuration
TRAINING_CATEGORIES = {
    "cot": ["cot", "chain_of_thought", "chain-of-thought", "openthoughts"],
    "reasoning": ["reasoning", "math", "gsm8k", "theorem", "proof", "mmlu", "arc"],
    "thinking": ["think", "reflection", "monologue"],
    "tools": ["tool", "function", "api", "xlam", "hermes"],
    "code": ["code", "stack", "programming", "git", "codegen"],
    "general": ["general", "alpaca", "wizard", "instruct", "dialog", "chat", "ultrafeedback", "helpsteer", "big-bench"],
    "long_context": ["long", "context", "book", "story", "leval"],
    "multimodal": ["multimodal", "image-text", "emm1", "valor", "journey", "voice", "speech", "music", "audio"],
    "vision-qa": ["vision", "visual", "caption", "llava", "mathvista"],
    "video-understanding": ["video", "captioning", "msr-vtt", "vatex"],
    "image-generation": ["image-gen", "text-to-image", "laion", "diffusion"],
    "video-generation": ["video-gen", "text-to-video", "svd"],
    "podcast": ["podcast", "dialogue", "conversation", "interview"],
    "streaming": ["streaming"],
    "uncensored": ["uncensored", "lovable", "unrefined", "liberated", "raw_capability"]
}

# Benchmarks often share keywords with training (e.g. "math"), but have specific identifiers.
# We prioritize Benchmark detection.
BENCHMARK_KEYWORDS = [
    "benchmark", "eval", "evaluation", "test_set", "validation_set",
    "mmlu", "humaneval", "mbpp", "math_eval", "gsm8k_test"
]

# Model Categories
MODEL_CATEGORIES = {
    "vision": ["vision", "visual", "siglip", "clip", "vit", "llava", "image", "resampler", "qwen-vl", "obs"],
    "audio": ["audio", "whisper", "speech", "sound", "voice", "tts", "stt", "music"]
}

def inspect_content(file_path: Path) -> Optional[str]:
    """
    Inspect file content to guess category based on schema keys.
    Returns suggested category name or None.
    """
    try:
        keys = set()
        # Read first few lines/records
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            if file_path.suffix.lower() == '.jsonl':
                for _ in range(5):
                    line = f.readline()
                    if not line: break
                    try:
                        data = json.loads(line)
                        keys.update(data.keys())
                    except: pass
            elif file_path.suffix.lower() == '.json':
                try:
                    data = json.load(f)
                    if isinstance(data, list) and len(data) > 0:
                        keys.update(data[0].keys())
                    elif isinstance(data, dict):
                        keys.update(data.keys())
                except: pass
                
        # Heuristic Matching
        if "tool_calls" in keys or "function" in keys: return "tools"
        if "prompt" in keys and "response" in keys: return "cot" # Mid-dataset typical
        if "instruction" in keys and "output" in keys: return "cot" # Alpaca style
        if "messages" in keys: return "cot" # Chat format often implies instruct/cot
        if "image" in keys or "visual" in keys: return "vision-qa"
        if "problem" in keys and "solution" in keys: return "reasoning"
        if "question" in keys and "answer" in keys: return "reasoning" # or vision-qa, keyword fallback helps
        
    except Exception as e:
        logger.debug(f"Content inspection failed for {file_path.name}: {e}")
        
    return None

def get_destination(file_path: Path, base_path: Path) -> Tuple[str, str, str]:
    """
    Determine the destination type (datasets/benchmarks/encoders/decoders) and category.
    """
    filename = file_path.name.lower()
    
    # --- 1. Model Encoders/Decoders Handling ---
    if "encoders" in str(file_path.parent).lower():
         if any(kw in filename for kw in MODEL_CATEGORIES["vision"]):
             return "encoders", "vision-encoders", "vision-encoders"
         elif any(kw in filename for kw in MODEL_CATEGORIES["audio"]):
             return "encoders", "audio-encoders", "audio-encoders"
         pass 

    if "decoders" in str(file_path.parent).lower():
         if any(kw in filename for kw in MODEL_CATEGORIES["vision"]):
             return "decoders", "vision-decoders", "vision-decoders"
         elif any(kw in filename for kw in MODEL_CATEGORIES["audio"]):
             return "decoders", "audio-decoders", "audio-decoders"
         pass

    # --- 2. Datasets/Benchmarks Handling ---
    
    # Check Benchmark (keyword priority)
    is_benchmark = any(kw in filename for kw in BENCHMARK_KEYWORDS)
    
    # Determine Category
    best_category = "general"
    
    # A. Content Inspection (Strong Signal)
    content_guess = inspect_content(file_path)
    if content_guess:
        best_category = content_guess
        
    # B. Keyword Matching (Refinement / Fallback)
    # If content detection was generic (like 'cot' from messages), specific filename keywords might be better (e.g. 'math' -> reasoning)
    # Or if content detection failed.
    
    keyword_category = None
    # Prioritize 'uncensored' keyword above all others for future-proofing
    if any(kw in filename for kw in TRAINING_CATEGORIES["uncensored"]):
        keyword_category = "uncensored"
    else:
        for cat, keywords in TRAINING_CATEGORIES.items():
            if any(kw in filename for kw in keywords):
                keyword_category = cat
                break
    
    # Decision Matrix
    if keyword_category:
        # Keywords are usually more specific about *intent* (e.g. 'math' vs generic 'chat')
        best_category = keyword_category
    elif not content_guess:
        best_category = "general"
            
    if is_benchmark:
        return "benchmarks", best_category, best_category
    
    return "datasets", best_category, best_category

def organize_folder(source_dir: Path, target_base: Path, move: bool, is_model_dir: bool = False):
    """
    Generic organizer for a root folder (datasets, encoders, etc.)
    """
    if not source_dir.exists():
        logger.warning(f"Source directory not found, skipping: {source_dir}")
        return 0

    logger.info(f"Scanning {source_dir}...")
    
    # For models, we might be organizing directories (like HuggingFace model folders) not just files.
    # But for simplicity and safety, let's scan direct children first.
    # If the child is a directory and matches keywords, we move the whole directory.
    
    items_to_process = list(source_dir.iterdir())
    moves_count = 0
    
    for item in items_to_process:
        # SKIP existing target directories to avoid recursive mess
        if item.name in ["vision-encoders", "audio-encoders", "vision-decoders", "audio-decoders", "cot", "tools", "benchmarks", "uncensored"]:
            continue
            
        # Strategy:
        # If item is a file -> check name -> move
        # If item is a dir -> check name -> move whole dir (typical for models)
        
        type_folder, category_desc, sub_folder = get_destination(item, target_base)
        
        # Override for Models: ensure we are in the right root context
        if is_model_dir:
            # If we are scanning 'encoders', we only care if destination is 'encoders'
            if type_folder != source_dir.name: 
                # This item matched a dataset keyword? unlikely/risky, skip.
                # Or it didn't match any model keyword.
                continue
                
            # If it's already in the correct subfolder, get_destination ignores parent context logic above?
            # Actually get_destination uses 'encoders' string check on full path.
            # We just need to formulate the target path.
            dest_parent = target_base / type_folder / sub_folder
        else:
            # Datasets case
            if type_folder == "encoders" or type_folder == "decoders": 
                 # Safety: don't move datasets into model folders automatically unless explicitly distinct
                 continue
            dest_parent = target_base / type_folder / sub_folder

        dest_path = dest_parent / item.name
        
        # Check if already in correct place
        if item.parent.resolve() == dest_parent.resolve():
            continue
            
        if item.resolve() == dest_path.resolve():
             continue

        if move:
            try:
                dest_parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(item), str(dest_path))
                logger.info(f"Moved: {item.name} -> {type_folder}/{sub_folder}/")
                moves_count += 1
            except Exception as e:
                logger.error(f"Failed to move {item.name}: {e}")
        else:
            logger.info(f"[DRY-RUN] Would move: {item.name} -> {type_folder}/{sub_folder}/")
            moves_count += 1
            
    return moves_count

def organize(base_path: str, move: bool = False):
    """
    Deep scan and organize datasets.
    Specifically looks for 'uncensored' data anywhere and moves to root datasets/uncensored.
    """
    root = Path(base_path)
    datasets_root = root / "datasets"
    
    if not datasets_root.exists():
        logger.error(f"Datasets root not found: {datasets_root}")
        return

    logger.info(f"🚀 Starting deep organization scan at {datasets_root}...")
    
    # 1. First, find all 'uncensored' items recursively
    # We ignore the actual target folder: datasets/uncensored
    target_uncensored = datasets_root / "uncensored"
    target_uncensored.mkdir(parents=True, exist_ok=True)
    
    uncensored_items = []
    for item in datasets_root.rglob("*"):
        # Criteria for uncensored:
        # - Folder or file name contains "uncensored" or other keywords
        # - Not already the target folder itself or inside it
        if item.resolve() == target_uncensored.resolve():
            continue

        if any(kw in item.name.lower() for kw in TRAINING_CATEGORIES["uncensored"]):
            # Check if it's already in the target hierarchy
            if target_uncensored.resolve() not in item.resolve().parents:
                uncensored_items.append(item)
    
    moves_count = 0
    
    # Process Uncensored Moves
    for item in uncensored_items:
        # If it's a file inside a folder that is also marked for move, skip the file
        if any(p in uncensored_items for p in item.parents):
            continue
            
        dest_path = target_uncensored / item.name
        if move:
            try:
                if not dest_path.exists():
                    shutil.move(str(item), str(dest_path))
                    logger.info(f"✅ RELOCATED UNCENSORED: {item.relative_to(root)} -> datasets/uncensored/")
                    moves_count += 1
                else:
                    logger.warning(f"⚠️ Skipping move: {dest_path} already exists.")
            except Exception as e:
                logger.error(f"❌ Failed move {item.name}: {e}")
        else:
            logger.info(f"🔍 [DRY-RUN] Would relocate uncensored: {item.relative_to(root)} -> datasets/uncensored/")
            moves_count += 1

    # 2. Organize direct children into categories (Baseline logic)
    # We skip 'uncensored' since we just handled it
    for item in datasets_root.iterdir():
        if not item.is_dir() or item.name in TRAINING_CATEGORIES or item.name == "uncensored":
            continue
            
        # Try to guess category
        type_folder, category_desc, sub_folder = get_destination(item, root)
        if type_folder == "datasets" and sub_folder != "general":
            dest_parent = datasets_root / sub_folder
            if dest_parent != item.parent: # Avoid self-nesting
                dest_path = dest_parent / item.name
                if move:
                    dest_parent.mkdir(parents=True, exist_ok=True)
                    if not dest_path.exists():
                        shutil.move(str(item), str(dest_path))
                        logger.info(f"✅ Categorized {item.name} -> {sub_folder}/")
                        moves_count += 1
                else:
                    logger.info(f"🔍 [DRY-RUN] Would categorize: {item.name} -> {sub_folder}/")
                    moves_count += 1

    if not move:
        logger.info("\n✨ [DRY-RUN COMPLETE] Use --move to apply changes.")
    else:
        logger.info(f"\n✨ [COMPLETE] Organized {moves_count} items.")


def main():
    parser = argparse.ArgumentParser(description="Organize datasets and benchmarks.")
    parser.add_argument("--base-path", required=True, help="Base data path (containing 'datasets' folder)")
    parser.add_argument("--move", action="store_true", help="Execute physical moves")
    parser.add_argument("--dry-run", action="store_true", help="Print actions without executing")
    
    args = parser.parse_args()
    
    if args.move and args.dry_run:
        logger.error("Cannot specify both --move and --dry-run")
        return

    # Default to dry run if move not specified
    is_move = args.move
    if not is_move and not args.dry_run:
        logger.info("No action specified. Defaulting to --dry-run")
    
    organize(args.base_path, move=is_move)

if __name__ == "__main__":
    main()
