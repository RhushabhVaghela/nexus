#!/usr/bin/env python3
"""
Dataset & Model Component Organization Script

Restructures:
1. Datasets -> E:/data/datasets/{category}
2. Benchmarks -> E:/data/benchmarks/{category}
3. Encoders -> E:/data/encoders/{modality}
4. Decoders -> E:/data/decoders/{modality}

Usage:
    python scripts/organize_datasets.py [--dry-run]
"""

import os
import shutil
import argparse
import logging
from pathlib import Path
from typing import Dict, List

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Base Paths
DATA_ROOT = Path("/mnt/e/data") 
DATASETS_DIR = DATA_ROOT / "datasets"
BENCHMARKS_DIR = DATA_ROOT / "benchmarks"
ENCODERS_DIR = DATA_ROOT / "encoders"
DECODERS_DIR = DATA_ROOT / "decoders"

# Domain Mappings
DOMAIN_MAP = {
    "reasoning": [
        "O1", "CoT", "TeichAI", "Alibaba", "OpenO1", "deepseek", "kimi", "reasoning", "thinking"
    ],
    "math": [
        "gsm8k", "math", "Math", "Ineq", "Arith"
    ],
    "code": [
        "stack", "commit", "codex", "code", "codegen", "stackoverflow", "swe-bench"
    ],
    "tools": [
        "function", "api", "tool", "agent", "hermes", "gorilla", "glaive", "mind2web", "webarena"
    ],
    "long_context": [
        "long", "scrolls", "infinite", "leval", "ruler", "pageshift"
    ],
    "multimodal": [
        "vision", "audio", "video", "speech", "image", "caption", "mm1", "llava", "journey", 
        "music", "voice", "valor", "vatex", "stargate"
    ],
    "general": [
        "dialogue", "chat", "instruct", "alpaca", "wizard", "mmlu", "general", "smol", "sporc", "xtreme"
    ]
}

# Explicit Benchmark List
BENCHMARK_DATASETS = [
    "allenai_arc-agi", "google_bigbench-hard", "TAUR-Lab_MuSR", "cais_mmlu",
    "openai_gsm8k", 
    "princeton-nlp_SWE-bench", "gorilla-llm_Berkeley-Function-Calling",
    "THUDM_LongBench", "tau_scrolls", "xinrongzhang2022_InfiniteBench", "LEval_LEval",
    "THUDM_AgentBench", "ServiceNow_webarena-verified", "osunlp_Mind2Web",
    "google_xtreme",
]

# Encoder/Decoder Modalities
MODALITY_MAP = {
    "vision": ["vision", "image", "vit", "clip", "siglip"],
    "audio": ["audio", "speech", "whisper", "codec"],
    "text": ["bert", "roberta", "t5"], # rarely standalone for LLMs but possible
    "multimodal": ["chamaleon", "omni"]
}

def get_category(name: str) -> str:
    """Determine category based on name."""
    name_lower = name.lower()
    for category, keywords in DOMAIN_MAP.items():
        for keyword in keywords:
            if keyword.lower() in name_lower:
                return category
    return "general"

def get_modality(name: str) -> str:
    """Determine modality for encoders/decoders."""
    name_lower = name.lower()
    for modality, keywords in MODALITY_MAP.items():
        for keyword in keywords:
            if keyword.lower() in name_lower:
                return modality
    return "general"

def organize_folder(source_root: Path, target_root_base: Path, is_benchmark: bool = False, is_component: bool = False, dry_run: bool = False):
    """Generic organization function."""
    if not source_root.exists():
        logger.warning(f"Source dir not found: {source_root}")
        return

    all_items = [d for d in source_root.iterdir() if d.is_dir()]
    logger.info(f"Scanning {source_root} ({len(all_items)} items)...")

    for item in all_items:
        name = item.name
        
        # Don't move if it's already a category folder (check if name is a valid category/modality)
        if is_component:
             if name in MODALITY_MAP.keys() or name.endswith("-encoders") or name.endswith("-decoders"):
                 continue
        elif name in DOMAIN_MAP.keys():
            continue

        # Determine target
        if is_component:
            modality = get_modality(name)
            # Enforce structure: encoders/vision/ or encoders/vision-encoders/ ? 
            # User has audio-encoders, vision-encoders. Let's separate by clean modality name: encoders/vision
            target_dir = source_root / modality # Keep inside same root, just subfolder
        else:
            # Datasets logic
            # Check if explicit benchmark (only if processing datasets folder)
            is_explicit_benchmark = False
            if not is_benchmark: # If we are in datasets/, check if it should be in benchmarks/
                for bench in BENCHMARK_DATASETS:
                    if bench in name:
                        is_explicit_benchmark = True
                        break
            
            if is_explicit_benchmark:
                # Move to benchmarks/category
                cat = get_category(name)
                target_dir = BENCHMARKS_DIR / cat
            else:
                # Move to valid subset in current root
                cat = get_category(name)
                target_dir = source_root / cat

        # Execute
        if dry_run:
            logger.info(f"[DRY] Move {name} -> {target_dir}/{name}")
        else:
            if not target_dir.exists():
                target_dir.mkdir(parents=True, exist_ok=True)
            
            new_path = target_dir / name
            if new_path.exists():
                if new_path == item: continue # Same place
                logger.warning(f"Target exists: {new_path}, skipping.")
                continue
            
            try:
                shutil.move(str(item), str(new_path))
                logger.info(f"Moved {name} -> {target_dir.name}/{name}")
            except Exception as e:
                logger.error(f"Failed to move {name}: {e}")

def organize(dry_run: bool = False):
    if not dry_run:
        DATASETS_DIR.mkdir(parents=True, exist_ok=True)
        BENCHMARKS_DIR.mkdir(parents=True, exist_ok=True)
        ENCODERS_DIR.mkdir(parents=True, exist_ok=True)
        DECODERS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Organize Datasets (and move benchmarks out)
    logging.info("--- Organizing Datasets ---")
    organize_folder(DATASETS_DIR, DATASETS_DIR, is_benchmark=False, dry_run=dry_run)
    
    # 2. Organize Benchmarks (internal structure)
    logging.info("--- Organizing Benchmarks ---")
    organize_folder(BENCHMARKS_DIR, BENCHMARKS_DIR, is_benchmark=True, dry_run=dry_run)
    
    # 3. Organize Encoders
    logging.info("--- Organizing Encoders ---")
    organize_folder(ENCODERS_DIR, ENCODERS_DIR, is_component=True, dry_run=dry_run)
    
    # 4. Organize Decoders
    logging.info("--- Organizing Decoders ---")
    organize_folder(DECODERS_DIR, DECODERS_DIR, is_component=True, dry_run=dry_run)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Simulate moves")
    args = parser.parse_args()
    
    organize(dry_run=args.dry_run)
