import os
import shutil
import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

class NexusDataOrganizer:
    """
    Implements the Automatic Sorting and Content Detection features for Nexus.
    Moves raw datasets into categorized folders based on their content.
    """
    
    CATEGORIES = {
        "tools": ["tool_calls", "functions", "tool_trace"],
        "thinking": ["thinking_process", "reflection", "chain_of_thought", "scratchpad"],
        "vision": ["image", "image_path", "visual_input"],
        "audio": ["audio_path", "speech", "waveform"],
        "video": ["video_path", "frames"],
        "code": ["code", "snippet", "programming"],
        "math": ["gsm8k", "math_problem", "equation"],
        "remotion": ["remotion", "react_code", "video_script"]
    }

    def __init__(self, source_dir: str, target_base_dir: str):
        self.source_dir = Path(source_dir)
        self.target_base_dir = Path(target_base_dir)
        self.target_base_dir.mkdir(parents=True, exist_ok=True)

    def detect_category(self, file_path: Path) -> str:
        """
        Inspects the file content to determine its capability category.
        """
        try:
            sample = None
            if file_path.suffix == '.jsonl':
                with open(file_path, 'r', encoding='utf-8') as f:
                    line = f.readline()
                    if line:
                        sample = json.loads(line)
            elif file_path.suffix == '.json':
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list) and data:
                        sample = data[0]
                    elif isinstance(data, dict):
                        sample = data
            elif file_path.suffix == '.parquet':
                import pandas as pd
                df = pd.read_parquet(file_path)
                if not df.empty:
                    sample = df.iloc[0].to_dict()
            
            if not sample:
                return "general"

            # Check keys against categories
            keys = set(sample.keys())
            # Recursive check for nested keys if needed, but flat check first
            str_dump = json.dumps(sample)

            for category, keywords in self.CATEGORIES.items():
                # specific key check
                if any(k in keys for k in keywords):
                    return category
                
                # content check (slower but more robust)
                # if any(kw in str_dump for kw in keywords):
                #     return category
            
            # Heuristic for dataset names
            filename = file_path.name.lower()
            if "gsm8k" in filename or "math" in filename: return "math"
            if "code" in filename or "swe" in filename: return "code"
            
            return "general"

        except Exception as e:
            logger.error(f"Failed to detect category for {file_path}: {e}")
            return "error"

    def organize(self, dry_run: bool = False):
        """
        Scans source directory and moves files to categorized folders.
        """
        logger.info(f"Starting organization of {self.source_dir}...")
        
        # Scan for supported files
        files = []
        for ext in ['*.json', '*.jsonl', '*.parquet']:
            files.extend(list(self.source_dir.glob(f"**/{ext}")))
            
        count = 0
        for file_path in files:
            # Skip files already in categorized subfolders if source == target base
            if self.target_base_dir in file_path.parents and file_path.parent != self.target_base_dir:
                continue

            category = self.detect_category(file_path)
            
            if category == "error":
                continue

            target_dir = self.target_base_dir / category
            target_path = target_dir / file_path.name
            
            if not dry_run:
                target_dir.mkdir(parents=True, exist_ok=True)
                
                # Handle duplicate names
                if target_path.exists():
                    stem = target_path.stem
                    suffix = target_path.suffix
                    target_path = target_dir / f"{stem}_{file_path.stat().st_mtime_ns}{suffix}"

                print(f"Moving {file_path.name} -> {category}/")
                shutil.move(str(file_path), str(target_path))
            else:
                print(f"[Dry Run] Would move {file_path.name} -> {category}/")
            
            count += 1
            
        logger.info(f"Organized {count} datasets.")

if __name__ == "__main__":
    # Example usage
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    
    organizer = NexusDataOrganizer(args.source, args.target)
    organizer.organize(dry_run=args.dry_run)
