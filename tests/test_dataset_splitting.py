
import sys
import shutil
import json
import itertools
from pathlib import Path
import tempfile
import torch

# Mock the logger
class MockLogger:
    def info(self, msg): print(f"[INFO] {msg}")
    def warning(self, msg): print(f"[WARN] {msg}")
    def error(self, msg): print(f"[ERROR] {msg}")

logger = MockLogger()

# ---------------------------------------------------------
# COPY PASTE THE RELEVANT CLASS FROM 24_multimodal_training.py
# (We can't easily import because of missing dependencies in the env like Unsloth if run in isolation)
# But here we just need the logic. To be exact, I will import it if possible, 
# but it's safer to reproduce the logic or try to mock the class context.
# Let's try to mock the environment imports and then import the actual file.
# ---------------------------------------------------------

# Mock modules to allow importing 24_multimodal_training
sys.modules['transformers'] = type('Mock', (object,), {'Trainer': object, 'TrainingArguments': object})()
sys.modules['unsloth'] = type('Mock', (object,), {'FastLanguageModel': object})()
sys.modules['trl'] = type('Mock', (object,), {'SFTTrainer': object})()
sys.modules['peft'] = type('Mock', (object,), {'PeftModel': object})()

# Now we can hopefully import the class
import importlib.util
try:
    spec = importlib.util.spec_from_file_location("mm_training", "src/24_multimodal_training.py")
    mm_module = importlib.util.module_from_spec(spec)
    sys.modules["mm_training"] = mm_module
    spec.loader.exec_module(mm_module)
    OmniDataset = mm_module.OmniDataset
except Exception as e:
    print(f"❌ Failed to import OmniDataset: {e}")
    sys.exit(1)

def create_mock_dataset(root, name, structure):
    """
    Structure: {"folder": num_files} or "flat"
    """
    ds_path = root / name
    ds_path.mkdir()
    
    if structure == "flat":
        for i in range(10): # 10 files
            (ds_path / f"data_{i}.jsonl").write_text('{"messages": []}')
    else:
        for folder, count in structure.items():
            (ds_path / folder).mkdir()
            for i in range(count):
                (ds_path / folder / f"part_{i}.jsonl").write_text('{"messages": []}')
    return ds_path

def run_tests():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        print(f"📂 Created temp root: {root}")
        
        # Scenario 1: Only Train (Strict)
        create_mock_dataset(root, "Scenario1_TrainOnly", {"train": 5})
        
        # Scenario 2: Train + Test (Missing Val) (Strict)
        create_mock_dataset(root, "Scenario2_MissingVal", {"train": 5, "test": 2})
        
        # Scenario 3: Aliases (training, testing)
        create_mock_dataset(root, "Scenario3_Aliases", {"training": 5, "testing": 2})
        
        # Scenario 4: Flat (Auto Split)
        create_mock_dataset(root, "Scenario4_Flat", "flat")
        
        print("\n-------------------------------------------")
        
        # TEST 1: Train Split
        print("🔍 Testing Split: TRAIN")
        ds_train = OmniDataset(str(root), split="train")
        train_files = list(ds_train._get_files_for_split())
        
        # Expectations:
        # S1: 5 files
        # S2: 5 files
        # S3: 5 files
        # S4: ~9 files (90% of 10)
        
        counts = count_by_dataset(train_files)
        print(f"   Counts: {counts}")
        assert counts.get("Scenario1_TrainOnly") == 5
        assert counts.get("Scenario2_MissingVal") == 5
        assert counts.get("Scenario3_Aliases") == 5
        assert 8 <= counts.get("Scenario4_Flat", 0) <= 10 # Allow hash variance
        print("✅ Train Split Passed")

        # TEST 2: Val Split
        print("\n🔍 Testing Split: VAL")
        ds_val = OmniDataset(str(root), split="val")
        val_files = list(ds_val._get_files_for_split())
        
        # Expectations:
        # S1: 0 files (Strict mode, no val folder -> Empty)  <-- USER CONCERN
        # S2: 0 files (Strict mode, no val folder -> Empty)
        # S3: 0 files (No 'validation' folder)
        # S4: ~0-1 files (5% of 10)
        
        counts_val = count_by_dataset(val_files)
        print(f"   Counts: {counts_val}")
        assert counts_val.get("Scenario1_TrainOnly", 0) == 0
        assert counts_val.get("Scenario2_MissingVal", 0) == 0
        assert counts_val.get("Scenario3_Aliases", 0) == 0
        # Scenario 4 might be 0 or 1 depending on hash, but it SHOULD NOT be Empty if hash hits 90-95
        # With only 10 files, variance is high. But we check it doesn't crash.
        print("✅ Val Split Passed (Leakage Prevented)")
        
        # TEST 3: Aliased Val Split
        # Let's add an alias case
        create_mock_dataset(root, "Scenario3b_ExplicitVal", {"validation": 3})
        ds_val_alias = OmniDataset(str(root), split="val")
        val_files_alias = list(ds_val_alias._get_files_for_split())
        counts_alias = count_by_dataset(val_files_alias)
        print(f"\n🔍 Testing Split: VAL (Aliases)")
        print(f"   Counts: {counts_alias}")
        assert counts_alias.get("Scenario3b_ExplicitVal") == 3
        print("✅ Aliases Passed")

def count_by_dataset(files):
    counts = {}
    for p in files:
        # Parent name is dataset name
        # If inside subdir, it's parent.parent
        # BUT our mock structure is simple. 
        # S1: root/S1/train/file -> parent.parent.name = S1
        # S4: root/S4/file -> parent.name = S4
        
        parts = p.parts
        # Find index of tempdir root
        # Simplified: valid datasets are the direct children of root
        # p is absolute.
        # Let's just look at the name in the path
        for key in ["Scenario1_TrainOnly", "Scenario2_MissingVal", "Scenario3_Aliases", "Scenario4_Flat", "Scenario3b_ExplicitVal"]:
            if key in parts:
                counts[key] = counts.get(key, 0) + 1
    return counts

if __name__ == "__main__":
    run_tests()
