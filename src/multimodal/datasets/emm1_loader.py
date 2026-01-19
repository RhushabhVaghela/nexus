#!/usr/bin/env python3
"""
E-MM1 Dataset Loader for Any-to-Any Training

Loads E-MM1-100M multimodal dataset with support for:
- Image, Audio, Video, Point Cloud, Text modalities
- Nearest-neighbor sharding structure
- Lazy loading of large parquet files
"""

import pandas as pd
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict, List, Optional
import json


class EMM1Dataset(Dataset):
    """E-MM1-100M multimodal dataset loader."""
    
    def __init__(
        self,
        data_dir: str = "/mnt/e/data/downloaded/E-MM1-100M/data",
        shard_indices: Optional[List[int]] = None,
        modalities: List[str] = ["image", "audio", "video", "text"],
        sample_limit: int = 0
    ):
        """
        Args:
            data_dir: Path to E-MM1 data directory
            shard_indices: List of shard indices to load (1-16), None = all
            modalities: Which modalities to include
            sample_limit: Max samples to load (0 = no limit)
        """
        self.data_dir = Path(data_dir)
        self.modalities = modalities
        
        # Load specified shards
        if shard_indices is None:
            shard_indices = list(range(1, 17))  # All 16 shards
        
        print(f"Loading E-MM1 shards: {shard_indices}")
        self.data = []
        total_loaded = 0
        
        for idx in shard_indices:
            shard_file = self.data_dir / f"nn_{idx:02d}.parquet"
            if not shard_file.exists():
                print(f"⚠️  Shard {idx} not found: {shard_file}")
                continue
            
            df = pd.read_parquet(shard_file)
            shard_samples = len(df)
            
            # Convert to list of dicts
            for _, row in df.iterrows():
                if sample_limit > 0 and total_loaded >= sample_limit:
                    break
                self.data.append(row.to_dict())
                total_loaded += 1
            
            print(f"  Loaded shard {idx}: {shard_samples:,} samples")
            
            if sample_limit > 0 and total_loaded >= sample_limit:
                break
        
        print(f"✓ Loaded {len(self.data):,} total samples from E-MM1-100M")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Returns a multimodal sample with:
        - caption: Text description
        - image_path: Path to image file (if available)
        - audio_path: Path to audio file (if available)  
        - video_path: Path to video file (if available)
        - text: Associated text content (if available)
        """
        row = self.data[idx]
        
        sample = {
            "id": f"emm1_{idx}",
            "caption": row.get("caption", ""),
            "nn_index": row.get("nn_index", 1),
        }
        
        # Add modality paths if available
        if "image" in self.modalities and row.get("file_name_image"):
            sample["image_path"] = str(
                Path(row.get("save_folder_image", "")) / row.get("file_name_image", "")
            )
        
        if "audio" in self.modalities and row.get("file_name_audio"):
            sample["audio_path"] = str(
                Path(row.get("save_folder_audio", "")) / row.get("file_name_audio", "")
            )
        
        if "video" in self.modalities and row.get("file_name_video"):
            sample["video_path"] = str(
                Path(row.get("save_folder_video", "")) / row.get("file_name_video", "")
            )
        
        # Add metadata
        sample["metadata"] = {
            "source_image": row.get("source_dataset_image"),
            "source_audio": row.get("source_dataset_audio"),
            "source_video": row.get("source_dataset_video"),
            "source_text": row.get("source_dataset_text"),
        }
        
        return sample


def emm1_collate_fn(batch):
    """
    Custom collator for E-MM1 dataset.
    Handles variable-length multimodal inputs.
    """
    # Filter out None samples
    batch = [b for b in batch if b is not None]
    if not batch:
        return {}
    
    # Stack captions
    captions = [b["caption"] for b in batch]
    
    # Collect paths (will be loaded by dataloader workers)
    image_paths = [b.get("image_path") for b in batch]
    audio_paths = [b.get("audio_path") for b in batch]
    video_paths = [b.get("video_path") for b in batch]
    
    return {
        "captions": captions,
        "image_paths": image_paths,
        "audio_paths": audio_paths,
        "video_paths": video_paths,
        "ids": [b["id"] for b in batch]
    }


if __name__ == "__main__":
    # Test loading
    print("Testing E-MM1 Dataset Loader...")
    dataset = EMM1Dataset(
        shard_indices=[1],  # Just first shard for testing
        sample_limit=10
    )
    
    print(f"\nDataset size: {len(dataset)}")
    print(f"\nFirst sample:")
    print(json.dumps(dataset[0], indent=2))
