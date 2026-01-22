#!/usr/bin/env python3
"""
Simplify dataset structure:
1. Move everything from censored/ back to root.
2. Delete censored/ folders.
3. Keep uncensored/ for special data.
"""

import os
import shutil
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def simplify(data_root: str = "/mnt/e/data"):
    root = Path(data_root)
    
    for d_type in ["datasets", "benchmarks"]:
        d_dir = root / d_type
        if not d_dir.exists(): continue
        
        censored_dir = d_dir / "censored"
        if censored_dir.exists():
            logger.info(f"Merging {censored_dir} back to {d_dir}...")
            for item in list(censored_dir.iterdir()):
                target = d_dir / item.name
                if not target.exists():
                    shutil.move(str(item), str(target))
                else:
                    # Generic merge logic
                    if item.is_dir():
                        for sub in item.iterdir():
                            sub_target = target / sub.name
                            if not sub_target.exists():
                                shutil.move(str(sub), str(sub_target))
                        if not any(item.iterdir()):
                            item.rmdir()
            
            if censored_dir.exists() and not any(censored_dir.iterdir()):
                censored_dir.rmdir()
                logger.info(f"Removed redundant {censored_dir}")

    logger.info("Simplification complete.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/mnt/e/data")
    args = parser.parse_args()
    simplify(args.root)
