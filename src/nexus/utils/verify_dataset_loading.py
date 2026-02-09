import sys
import os
from pathlib import Path
import json

# Ensure src is in path
sys.path.append(str(Path(__file__).parent))

# Dynamic import logic handles the class loading
OmniDataset = None
logger = None

# Since we can't easily import from the script due to potential missing deps in the script header (mock vs real),
# I will copy the OmniDataset class logic entirely into this test script for isolation
# OR I can try to import if I am confident.
# Given the previous interactions, let's try to import the REFACTORED script if possible.
# Actually, 24_multimodal_training.py has imports that might fail (transformers).
# Let's rely on the mock imports inside 24_multimodal_training.py handling missing deps,
# BUT `verify_dataset_loading_import` doesn't exist.
# I will define a lightweight version of the test that attempts to import from the file.


def test_dataset_loading():
    print("🧪 TESTING DATASET LOADING & SAMPLING")
    print("-----------------------------------")

    # We need to import OmniDataset from src/24_multimodal_training.py
    # Since it's a script, we can use run_path or import if we add to path
    import importlib.util
from nexus.config.paths import DATASETS_DIR

    spec = importlib.util.spec_from_file_location(
        "multimodal_training", "src/training/scripts/24_multimodal_training.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["multimodal_training"] = mod
    spec.loader.exec_module(mod)
    OmniDataset = mod.OmniDataset

    data_path = DATASETS_DIR
    limit = 5

    print(f"📂 Data Path: {data_path}")
    print(f"📉 Limit per Dataset: {limit}")

    ds = OmniDataset(data_path, split="train", samples_per_dataset=limit)

    counts = {}

    print("\n🔄 Iterating...")
    total_yielded = 0
    for i, sample in enumerate(ds):
        total_yielded += 1
        # Verify sample has expected structure (dict with content)
        if not isinstance(sample, dict):
            print(f"  ⚠️ Sample {i} is not a dict: {type(sample)}")
        elif not sample:
            print(f"  ⚠️ Sample {i} is empty")
    print(f"  Total samples yielded: {total_yielded}")

    # Access internal state (white-box test)
    print("\n📊 Final Counts per Dataset (Internal State):")
    for name, count in ds.dataset_counts.items():
        pass_fail = "✅" if count == limit else "⚠️"
        if count < limit:
            pass_fail = "ℹ️ (Under limit)"  # Dataset might be small
        print(f"   - {name}: {count} {pass_fail}")

    print("\n-----------------------------------")
    print("Done.")


if __name__ == "__main__":
    test_dataset_loading()
