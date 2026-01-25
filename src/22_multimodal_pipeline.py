#!/usr/bin/env python3
"""
22_multimodal_pipeline.py
Orchestrate multimodal data acquisition and testing.

Phases:
1. Download data (WebSight, Common Voice, FineVideo)
2. Generate test prompts (Vision, Audio, Video)

Usage:
  python 22_multimodal_pipeline.py --phase download
  python 22_multimodal_pipeline.py --phase test
"""

import os
import sys
import argparse
import json
from pathlib import Path

# Add project root to sys.path to allow absolute imports from 'src'
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logging_config import setup_logger, log_header
try:
    from src.capability_registry import DATASET_REGISTRY
    from src.data.universal_loader import DatasetManager
    from src.benchmarks.ruler_tasks import get_test_prompts
except ImportError:
    DATASET_REGISTRY = {}
    DatasetManager = None
    get_test_prompts = lambda: {}

def check_env():
    """Verify environment dependencies."""
    if os.environ.get("CONDA_DEFAULT_ENV") != "nexus":
        print("[ERROR] Must be run in 'nexus' conda environment.")
        return False
    return True

# Globals to be initialized in main()
logger = None

CONFIG = {
    "output_dir": "/mnt/e/data/datasets",
    "test_dir": "tests/multimodal_assets"
}

# logger will be initialized in main()

def run_download(limit: int):
    """Download all multimodal datasets using Unified Strategy (Kaggle -> HF)"""
    log_header(logger, "MULTIMODAL DATA DOWNLOAD (UNIFIED)", {**CONFIG, "Limit": limit})
    
    Path(CONFIG["output_dir"]).mkdir(parents=True, exist_ok=True)
    
    manager = DatasetManager(Path(CONFIG["output_dir"]))
    total_samples = 0
    
    # Process all modalities defined in registry
    for modality, datasets in DATASET_REGISTRY.items():
        logger.info(f"\n📲 Processing Modality: {modality.upper()}")
        for name, config in datasets.items():
            count = manager.download_and_process(modality, name, config, limit)
            total_samples += count
            
    logger.info("="*60)
    logger.info("✅ UNIFIED DOWNLOAD COMPLETE")
    logger.info(f"   Output: {CONFIG['output_dir']}")
    logger.info(f"   Total Samples: {total_samples}")
    logger.info("="*60)

def run_test_setup():
    """Setup test prompts and assets"""
    log_header(logger, "MULTIMODAL TEST SETUP", CONFIG)
    
    prompts = get_test_prompts()
    
    # Save prompts to JSON for evaluation scripts
    output_path = Path(CONFIG["test_dir"]) / "test_prompts.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(prompts, f, indent=2)
    
    logger.info(f"Saved {sum(len(v) for v in prompts.values())} test prompts to {output_path}")
    
    # Log required assets
    logger.info("\nRequired Test Assets:")
    for modality, items in prompts.items():
        logger.info(f"\n{modality.upper()}:")
        for item in items:
            logger.info(f"  - {item['input']}")

def main():
    if not check_env():
         sys.exit(1)
         
    global logger
    logger = setup_logger(__name__, "logs/multimodal.log")

    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["download", "test"], required=True)
    parser.add_argument("--limit", type=int, default=10, help="Max samples per dataset")
    args = parser.parse_args()
    
    if args.phase == "download":
        run_download(args.limit)
    elif args.phase == "test":
        run_test_setup()

if __name__ == "__main__":
    main()
