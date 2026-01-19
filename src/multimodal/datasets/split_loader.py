#!/usr/bin/env python3
"""
Split Dataset Loader with Train/Val/Test splits
Supports all sample sizes with proper splitting
"""

import sys
from pathlib import Path
from typing import List, Dict, Optional
import json
import random

sys.path.insert(0, str(Path(__file__).parent.parent))
from multimodal.datasets.emm1_loader import EMM1Dataset


class SplitDatasetLoader:
    """
    Dataset loader with train/val/test splits.
    Automatically splits datasets 80/10/10.
    """
    
    def __init__(
        self,
        split: str = "train",  # train, val, or test
        total_samples: int = 0,  # Total samples to load (0 = all)
        emm1_shards: Optional[List[int]] = None
    ):
        """
        Args:
            split: Which split to load (train/val/test)
            total_samples: Total samples across all datasets (0 = unlimited)
            emm1_shards: Which E-MM1 shards to use (None = auto-select based on split)
        """
        self.split = split
        self.total_samples = total_samples
        
        # Auto-select shards based on split if not specified
        if emm1_shards is None:
            if split == "train":
                emm1_shards = list(range(1, 14))  # Shards 1-13 (81%)
            elif split == "val":
                emm1_shards = [14, 15]  # Shards 14-15 (12%)
            elif split == "test":
                emm1_shards = [16]  # Shard 16 (6%)
            else:
                emm1_shards = [1]  # Default
        
        self.emm1_shards = emm1_shards
        self.datasets = []
        self.dataset_names = []
        self.samples_count = 0
        
        print(f"🔍 Loading {split.upper()} split...")
        if total_samples > 0:
            print(f"   Target samples: {total_samples:,}")
        
        self._load_datasets()
    
    def _load_datasets(self):
        """Load all datasets with splitting."""
        # Calculate samples per dataset
        if self.total_samples > 0:
            emm1_target = int(self.total_samples * 0.9)  # 90% from E-MM1
            manual_target = int(self.total_samples * 0.1)  # 10% from manual
        else:
            emm1_target = 0
            manual_target = 0
        
        # 1. Load E-MM1
        print(f"\n  Loading E-MM1 (shards {self.emm1_shards})...")
        try:
            emm1_dataset = EMM1Dataset(
                data_dir="/mnt/e/data/downloaded/E-MM1-100M/data",
                shard_indices=self.emm1_shards,
                sample_limit=emm1_target
            )
            if len(emm1_dataset) > 0:
                self.datasets.append(emm1_dataset)
                self.dataset_names.append("emm1")
                self.samples_count += len(emm1_dataset)
                print(f"    ✓ Loaded {len(emm1_dataset):,} samples")
        except Exception as e:
            print(f"    ✗ Failed: {e}")
        
        # 2. Load manual datasets with splitting
        manual_datasets = self._get_manual_datasets()
        samples_per_manual = manual_target // len(manual_datasets) if manual_target > 0 else 0
        
        for name, path in manual_datasets.items():
            if self.samples_count >= self.total_samples and self.total_samples > 0:
                break
            
            try:
                dataset = self._load_manual_with_split(name, path, samples_per_manual)
                if dataset and len(dataset) > 0:
                    self.datasets.append(dataset)
                    self.dataset_names.append(name)
                    self.samples_count += len(dataset)
            except Exception as e:
                print(f"    ✗ {name} failed: {e}")
        
        print(f"\n✅ {self.split.upper()} split: {self.samples_count:,} samples from {len(self.datasets)} datasets")
    
    def _load_manual_with_split(self, name: str, path: Path, target_samples: int):
        """Load manual dataset with train/val/test split."""
        from torch.utils.data import Dataset
        
        class SplitJSONLDataset(Dataset):
            def __init__(self, jsonl_path, split, target):
                self.data = []
                all_data = []
                
                # Load all data
                with open(jsonl_path, 'r') as f:
                    for line in f:
                        if line.strip():
                            all_data.append(json.loads(line))
                
                # Split 80/10/10
                random.seed(42)  # Reproducible
                random.shuffle(all_data)
                
                total = len(all_data)
                train_end = int(total * 0.8)
                val_end = int(total * 0.9)
                
                if split == "train":
                    self.data = all_data[:train_end]
                elif split == "val":
                    self.data = all_data[train_end:val_end]
                elif split == "test":
                    self.data = all_data[val_end:]
                
                # Limit if target specified
                if target > 0 and len(self.data) > target:
                    self.data = self.data[:target]
            
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, idx):
                return self.data[idx]
        
        jsonl_file = path / "data.jsonl"
        if jsonl_file.exists():
            return SplitJSONLDataset(jsonl_file, self.split, target_samples)
        return None
    
    @staticmethod
    def _get_manual_datasets() -> Dict[str, Path]:
        """Get paths to all manual datasets."""
        base = Path("/mnt/e/data/unified_multimodal")
        return {
            "mathvista": base / "benchmarks/mathvista",
            "mathverse": base / "benchmarks/mathverse",
            "ineqmath": base / "benchmarks/ineqmath",
            "mmlu": base / "benchmarks/mmlu",
            "common_voice": base / "audio/common_voice",
            "msr_vtt": base / "video/msr_vtt",
            "vatex": base / "video/vatex",
            "stackoverflow_quality": base / "premium_text/stackoverflow_quality",
            "stackoverflow_questions": base / "premium_text/stackoverflow_questions",
            "llava_onevision": base / "vision/llava_onevision",
        }
    
    def __len__(self):
        return self.samples_count
    
    def __getitem__(self, idx):
        """Get item from appropriate dataset."""
        cumulative = 0
        for dataset in self.datasets:
            if idx < cumulative + len(dataset):
                return dataset[idx - cumulative]
            cumulative += len(dataset)
        raise IndexError(f"Index {idx} out of range")


if __name__ == "__main__":
    # Test
    print("Testing train split with 1000 samples...")
    train = SplitDatasetLoader(split="train", total_samples=1000)
    print(f"Train: {len(train)} samples")
    
    val = SplitDatasetLoader(split="val", total_samples=100)
    print(f"Val: {len(val)} samples")
