
import sys
from pathlib import Path
sys.path.append("src")
from mm_download_unified import DatasetManager, DATASET_REGISTRY, logger

# Force logger to stdout
import logging
logging.basicConfig(level=logging.INFO)

# Config
modality = "audio"
name = "common_voice"
config = DATASET_REGISTRY[modality][name]
print(f"Checking configuration for {name}:")
print(f"  Local Path: {config.get('local_path')}")
print(f"  Path Exists: {Path(config.get('local_path')).exists()}")

# Test processing
manager = DatasetManager(Path("/mnt/e/data/datasets"))
print("\n--- Starting Test Processing (Limit 1) ---")
count = manager.download_and_process(modality, name, config, sample_limit=1)
print(f"\n--- Result: Processed {count} samples ---")
