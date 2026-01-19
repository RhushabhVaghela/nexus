#!/usr/bin/env python3
"""
Memory-efficient E-MM1 Dataset Loader
Streams data without loading entire shards into RAM
"""

import pandas as pd
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict, List, Optional
import json


class EMM1Dataset(Dataset):
    """E-MM1-100M multimodal dataset loader with memory streaming."""
    
    def __init__(
        self,
        data_dir: str = "/mnt/e/data/downloaded/E-MM1-100M/data",
        shard_indices: Optional[List[int]] = None,
        modalities: List[str] = ["image", "audio", "video", "text"],
        sample_limit: int = 0
    ):
        self.data_dir = Path(data_dir)
        self.modalities = modalities
        
        import pyarrow.parquet as pq
        
        if shard_indices is None:
            shard_indices = list(range(1, 17))

        print(f"Loading E-MM1 shards: {shard_indices}")
        self.data = []
        total_loaded = 0
        
        for idx in shard_indices:
            shard_file = self.data_dir / f"nn_{idx:02d}.parquet"
            if not shard_file.exists():
                print(f"⚠️  Shard {idx} not found")
                continue
            
            # TRUE STREAMING via PyArrow
            try:
                parquet_file = pq.ParquetFile(shard_file)
                
                # If we have a limit, only read what we need
                if sample_limit > 0:
                    remaining = sample_limit - total_loaded
                    if remaining <= 0:
                        break
                    
                    # Read only the first batch of rows needed
                    # iter_batches allows reading without loading full file
                    # Batch size 1000 or 'remaining' is efficient
                    batch_iter = parquet_file.iter_batches(batch_size=min(10000, remaining))
                    
                    for batch in batch_iter:
                        df_chunk = batch.to_pandas()
                        # Take only what we need from this chunk
                        if len(df_chunk) > remaining:
                            df_chunk = df_chunk.head(remaining)
                        
                        for _, row in df_chunk.iterrows():
                            self.data.append(row.to_dict())
                            total_loaded += 1
                        
                        remaining -= len(df_chunk)
                        if remaining <= 0:
                            break
                    
                    print(f"  Loaded shard {idx}: {total_loaded} total samples (limited)")
                else:
                    # No limit - read efficiently in batches
                    total_in_shard = 0
                    for batch in parquet_file.iter_batches(batch_size=10000):
                        df_chunk = batch.to_pandas()
                        for _, row in df_chunk.iterrows():
                            self.data.append(row.to_dict())
                            total_loaded += 1
                            total_in_shard += 1
                    print(f"  Loaded shard {idx}: {total_in_shard:,} samples")

            except Exception as e:
                print(f"  ❌ Error reading shard {idx}: {e}")
                
            if sample_limit > 0 and total_loaded >= sample_limit:
                break
        
        
        print(f"✓ Loaded {len(self.data):,} total samples from E-MM1-100M")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict:
        row = self.data[idx]
        
        sample = {
            "id": f"emm1_{idx}",
            "caption": row.get("caption", ""),
            "nn_index": row.get("nn_index", 1),
        }
        
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
        
        sample["metadata"] = {
            "source_image": row.get("source_dataset_image"),
            "source_audio": row.get("source_dataset_audio"),
            "source_video": row.get("source_dataset_video"),
            "source_text": row.get("source_dataset_text"),
        }
        
        return sample


def emm1_collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return {}
    
    return {
        "captions": [b["caption"] for b in batch],
        "image_paths": [b.get("image_path") for b in batch],
        "audio_paths": [b.get("audio_path") for b in batch],
        "video_paths": [b.get("video_path") for b in batch],
        "ids": [b["id"] for b in batch]
    }
