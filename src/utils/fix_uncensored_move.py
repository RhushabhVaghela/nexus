#!/usr/bin/env python3
"""
Fix dataset structure:
1. Move 'uncensored' from inside 'general' (or any other cat) to root 'datasets/'.
2. Ensure it is a sibling to 'code', 'math', etc.
"""

import os
import shutil
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def fix_uncensored_location(data_root: str = "/mnt/e/data"):
    root = Path(data_root)
    datasets_dir = root / "datasets"
    benchmarks_dir = root / "benchmarks"
    
    for d_dir in [datasets_dir, benchmarks_dir]:
        if not d_dir.exists(): continue
        
        # Look for 'uncensored' inside any category
        for cat_dir in d_dir.iterdir():
            if cat_dir.is_dir() and cat_dir.name != "uncensored":
                nested_uncensored = cat_dir / "uncensored"
                if nested_uncensored.exists():
                    target = d_dir / "uncensored"
                    target.mkdir(parents=True, exist_ok=True)
                    
                    logger.info(f"Moving content from {nested_uncensored} to {target}...")
                    for item in nested_uncensored.iterdir():
                        dest = target / item.name
                        if not dest.exists():
                            shutil.move(str(item), str(dest))
                        else:
                            logger.warning(f"Conflict: {dest} already exists.")
                    
                    # Remove the now empty nested folder
                    try:
                        nested_uncensored.rmdir()
                        logger.info(f"Removed empty nested folder: {nested_uncensored}")
                    except OSError:
                        logger.warning(f"Could not remove {nested_uncensored} - might not be empty.")

    logger.info("Structure fix complete.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/mnt/e/data")
    args = parser.parse_args()
    fix_uncensored_location(args.root)
