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

import argparse
import sys
from pathlib import Path
import json

sys.path.insert(0, str(Path(__file__).parent))
from multimodal import download_vision_data, download_audio_data, download_video_data, get_test_prompts
from utils.logging_config import setup_logger, log_header, log_completion

CONFIG = {
    "output_dir": "/mnt/e/data/multimodal",
    "test_dir": "tests/multimodal_assets"
}

logger = setup_logger(__name__, "logs/multimodal.log")

def run_download(limit: int):
    """Download all multimodal datasets"""
    log_header(logger, "MULTIMODAL DATA DOWNLOAD", {**CONFIG, "Limit": limit})
    
    Path(CONFIG["output_dir"]).mkdir(parents=True, exist_ok=True)
    
    download_vision_data(CONFIG["output_dir"], limit=limit)
    download_audio_data(CONFIG["output_dir"], limit=limit)
    download_video_data(CONFIG["output_dir"], limit=limit)
    
    log_completion(logger, "Multimodal Download", {"Output": CONFIG["output_dir"]})

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["download", "test"], default="download")
    parser.add_argument("--limit", type=int, default=1000, help="Max samples to download per modality")
    args = parser.parse_args()
    
    if args.phase == "download":
        run_download(args.limit)
    elif args.phase == "test":
        run_test_setup()
