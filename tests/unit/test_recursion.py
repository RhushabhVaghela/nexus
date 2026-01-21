
import json
import os
import shutil
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.metrics_tracker import discover_datasets
from src.data.universal_loader import UniversalDataLoader

def test_recursive_discovery_and_loading(tmp_path):
    """
    Test that discover_datasets finds folders recursively
    and UniversalDataLoader loads files from subfolders.
    """
    print(f"Testing in {tmp_path}")
    
    # 1. Setup nested structure
    ds_root = tmp_path / "datasets"
    ds_root.mkdir()
    
    reasoning_dir = ds_root / "reasoning"
    reasoning_dir.mkdir()
    
    my_ds = reasoning_dir / "my_reasoning_ds"
    my_ds.mkdir()
    
    shard_dir = my_ds / "shards"
    shard_dir.mkdir()
    
    # Create data files
    with open(my_ds / "data1.jsonl", "w") as f:
        f.write(json.dumps({"text": "root sample"}) + "\n")
        
    with open(shard_dir / "data2.jsonl", "w") as f:
        f.write(json.dumps({"text": "shard sample"}) + "\n")
        
    # 2. Test Discovery
    discovered = discover_datasets(str(reasoning_dir))
    
    reasoning_paths = discovered.get("reasoning", [])
    print(f"Discovered reasoning paths: {reasoning_paths}")
    
    found_my_ds = any("my_reasoning_ds" in p for p in reasoning_paths)
    found_shards = any("shards" in p and p.endswith("shards") for p in reasoning_paths)
    
    assert found_my_ds, "my_reasoning_ds not discovered"
    assert not found_shards, "Duplicate shard directory discovered"
    
    # 3. Test Loading
    loader = UniversalDataLoader(my_ds)
    result = loader.load()
    
    print(f"Loaded {result.num_samples} samples")
    assert result.num_samples == 2, f"Expected 2 samples, got {result.num_samples}"
        
    samples = [s["text"] for s in result.dataset]
    assert "root sample" in samples, f"Missing 'root sample'. Got: {samples}"
    assert "shard sample" in samples, f"Missing 'shard sample'. Got: {samples}"
    
    print("✅ Recursive Discovery and Loading Verified!")

if __name__ == "__main__":
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        test_recursive_discovery_and_loading(Path(tmp))
        print("Test passed!")

