#!/usr/bin/env python3
"""
Unified Multi-Dataset Loader
Combines all available datasets for maximum training performance
"""

import sys
from pathlib import Path
from typing import List, Dict
import json

sys.path.insert(0, str(Path(__file__).parent.parent))
from multimodal.datasets.emm1_loader import EMM1Dataset


class UnifiedMultiDatasetLoader:
    """
    Combines multiple datasets for comprehensive any-to-any training.
    Automatically discovers and loads all processed datasets.
    """
    
    def __init__(
        self,
        data_sources: Dict[str, Dict] = None,
        sample_limit_per_dataset: int = 0
    ):
        """
        Args:
            data_sources: Dict of dataset configs
            sample_limit_per_dataset: Max samples per dataset (0=unlimited)
        """
        if data_sources is None:
            data_sources = self.get_default_sources()
        
        self.datasets = []
        self.dataset_names = []
        self.total_samples = 0
        
        print("🔍 Loading Unified Multi-Dataset...")
        
        for name, config in data_sources.items():
            print(f"\n  Loading {name}...")
            try:
                dataset = self._load_dataset(name, config, sample_limit_per_dataset)
                if dataset and len(dataset) > 0:
                    self.datasets.append(dataset)
                    self.dataset_names.append(name)
                    self.total_samples += len(dataset)
                    print(f"    ✓ Loaded {len(dataset):,} samples")
            except Exception as e:
                print(f"    ✗ Failed: {e}")
        
        print(f"\n✅ Total: {self.total_samples:,} samples from {len(self.datasets)} datasets")
    
    def _load_dataset(self, name: str, config: Dict, sample_limit: int):
        """Load individual dataset based on type."""
        dataset_type = config.get("type", "unknown")
        path = Path(config.get("path", ""))
        
        if dataset_type == "emm1":
            # E-MM1-100M multimodal dataset
            return EMM1Dataset(
                data_dir=str(path / "data"),
                shard_indices=config.get("shards", [1, 2, 3]),
                sample_limit=sample_limit
            )
        
        elif dataset_type == "jsonl":
            # Manual processed datasets (from process_manual_datasets.py)
            from torch.utils.data import Dataset
            
            class JSONLDataset(Dataset):
                def __init__(self, jsonl_path):
                    self.data = []
                    with open(jsonl_path, 'r') as f:
                        for line in f:
                            if line.strip():
                                self.data.append(json.loads(line))
                                if sample_limit > 0 and len(self.data) >= sample_limit:
                                    break
                
                def __len__(self):
                    return len(self.data)
                
                def __getitem__(self, idx):
                    return self.data[idx]
            
            jsonl_file = path / "data.jsonl"
            if jsonl_file.exists():
                return JSONLDataset(jsonl_file)
        
        return None
    
    @staticmethod
    def get_default_sources() -> Dict[str, Dict]:
        """Get all available datasets in /mnt/e/data."""
        return {
            # Primary: E-MM1-100M (100M+ samples, all modalities)
            "emm1": {
                "type": "emm1",
                "path": "/mnt/e/data/downloaded/E-MM1-100M",
                "shards": list(range(1, 17)),  # All 16 shards
                "priority": 1
            },
            
            # Manually processed datasets (from process_manual_datasets.py)
            "mathvista": {
                "type": "jsonl",
                "path": "/mnt/e/data/unified_multimodal/benchmarks/mathvista",
                "priority": 2
            },
            "mathverse": {
                "type": "jsonl",
                "path": "/mnt/e/data/unified_multimodal/benchmarks/mathverse",
                "priority": 2
            },
            "ineqmath": {
                "type": "jsonl",
                "path": "/mnt/e/data/unified_multimodal/benchmarks/ineqmath",
                "priority": 2
            },
            "mmlu": {
                "type": "jsonl",
                "path": "/mnt/e/data/unified_multimodal/benchmarks/mmlu",
                "priority": 2
            },
            "common_voice": {
                "type": "jsonl",
                "path": "/mnt/e/data/unified_multimodal/audio/common_voice",
                "priority": 2
            },
            "msr_vtt": {
                "type": "jsonl",
                "path": "/mnt/e/data/unified_multimodal/video/msr_vtt",
                "priority": 2
            },
            "vatex": {
                "type": "jsonl",
                "path": "/mnt/e/data/unified_multimodal/video/vatex",
                "priority": 2
            },
            "stackoverflow_quality": {
                "type": "jsonl",
                "path": "/mnt/e/data/unified_multimodal/premium_text/stackoverflow_quality",
                "priority": 2
            },
            "stackoverflow_questions": {
                "type": "jsonl",
                "path": "/mnt/e/data/unified_multimodal/premium_text/stackoverflow_questions",
                "priority": 2
            },
            "llava_onevision": {
                "type": "jsonl",
                "path": "/mnt/e/data/unified_multimodal/vision/llava_onevision",
                "priority": 2
            }
        }
    
    def __len__(self):
        return self.total_samples
    
    def __getitem__(self, idx):
        """Get item from appropriate dataset."""
        # Find which dataset this index belongs to
        cumulative = 0
        for dataset in self.datasets:
            if idx < cumulative + len(dataset):
                return dataset[idx - cumulative]
            cumulative += len(dataset)
        
        raise IndexError(f"Index {idx} out of range (total: {self.total_samples})")


if __name__ == "__main__":
    # Test loading
    print("Testing Unified Multi-Dataset Loader...")
    loader = UnifiedMultiDatasetLoader(sample_limit_per_dataset=10)
    
    print(f"\n📊 Summary:")
    print(f"  Total datasets: {len(loader.datasets)}")
    print(f"  Total samples: {len(loader):,}")
    print(f"  Datasets: {', '.join(loader.dataset_names)}")
