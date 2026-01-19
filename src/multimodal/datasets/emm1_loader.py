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



class EMM1Dataset(torch.utils.data.IterableDataset):
    """E-MM1-100M multimodal dataset loader with memory streaming."""
    
    def __init__(
        self,
        data_dir: str = "/mnt/e/data/datasets/E-MM1-100M/data",
        shard_indices: Optional[List[int]] = None,
        modalities: List[str] = ["image", "audio", "video", "text"],
        sample_limit: int = 0
    ):
        self.data_dir = Path(data_dir)
        self.modalities = modalities
        self.sample_limit = sample_limit
        
        if shard_indices is None:
            self.shard_indices = list(range(1, 17))
        else:
            self.shard_indices = shard_indices

        # Verify shards exist but DO NOT load them
        self.available_shards = []
        for idx in self.shard_indices:
            shard_file = self.data_dir / f"nn_{idx:02d}.parquet"
            if shard_file.exists():
                self.available_shards.append(shard_file)
            else:
                print(f"⚠️  Shard {idx} not found at {shard_file}")
                
        print(f"🌊 EMM1Dataset initialized. Found {len(self.available_shards)} shards. Streaming mode active.")
    
    def __iter__(self):
        import pyarrow.parquet as pq
        
        worker_info = torch.utils.data.get_worker_info()
        
        # Simple sharding: if multiple workers, split files among them
        if worker_info is not None:
            # Split shards among workers
            per_worker = int(len(self.available_shards) / float(worker_info.num_workers))
            worker_id = worker_info.id
            iter_start = worker_id * per_worker
            iter_end = min(iter_start + per_worker, len(self.available_shards))
            my_shards = self.available_shards[iter_start:iter_end]
        else:
            my_shards = self.available_shards

        total_yielded = 0
        
        for shard_file in my_shards:
            if self.sample_limit > 0 and total_yielded >= self.sample_limit:
                break
                
            try:
                parquet_file = pq.ParquetFile(shard_file)
                # Stream batches (e.g. 1000 rows at a time)
                for batch in parquet_file.iter_batches(batch_size=1000):
                    df_chunk = batch.to_pandas()
                    
                    for _, row in df_chunk.iterrows():
                        if self.sample_limit > 0 and total_yielded >= self.sample_limit:
                            return
                        
                        yield self._process_row(row.to_dict(), total_yielded)
                        total_yielded += 1
                        
            except Exception as e:
                print(f"❌ Error reading shard {shard_file}: {e}")

    def _process_row(self, row: Dict, idx: int) -> Dict:
        """Process a single raw row into a sample"""
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
            
        return sample

    def __len__(self):
        # IterableDataset doesn't strictly require len, but if we know it we can return it.
        # Since we are strictly streaming, we might not know strict length without scanning.
        # Return an estimate or raise NotImplementedError.
        # Returning limit if set, else large number or 0.
        return self.sample_limit if self.sample_limit > 0 else 100000000

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

