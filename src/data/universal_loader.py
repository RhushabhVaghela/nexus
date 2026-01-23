#!/usr/bin/env python3
"""
universal_loader.py - Nexus High-Performance Dataset Loader
"""

import os
import json
import csv
import gzip
import zipfile
import tarfile
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass
import logging
import multiprocessing

# Nexus Utilities
try:
    from src.utils.corruption_tracker import tracker as corruption_tracker
    from src.utils.schema_normalizer import SchemaNormalizer
except ImportError:
    corruption_tracker = None
    SchemaNormalizer = None

logger = logging.getLogger(__name__)

@dataclass
class LoadResult:
    dataset: Any
    format: str
    num_samples: int
    columns: List[str]
    source_path: str
    error: Optional[str] = None

def load_dataset_universal(path: Union[str, Path], sample_size: Optional[int] = None, fast_mode: bool = True) -> LoadResult:
    """Convenience function to load a dataset."""
    loader = UniversalDataLoader(path, fast_mode=fast_mode)
    return loader.load(sample_size=sample_size)

class GlobalIndexMap:
    def __init__(self, file_paths: List[Path], format: str):
        self.file_paths = sorted(file_paths)
        self.format = format
        self.index_map = [] 
        self.total_count = 0

    def _count_rows(self, path: Path) -> int:
        try:
            if self.format == "parquet":
                import pyarrow.parquet as pq
                return pq.ParquetFile(path).metadata.num_rows
            elif self.format == "jsonl":
                count = 0
                with open(path, 'rb') as f:
                    for _ in f: count += 1
                return count
            return 0
        except:
            return 0

    def _build_map(self, fast_mode: bool = False):
        print(f"🔍 Building Index Map for {len(self.file_paths)} shards ({self.format})...")
        target_files = self.file_paths[:1] if fast_mode else self.file_paths
        
        counts = []
        for path in target_files:
            print(f"   - Counting {path.name}...")
            count = self._count_rows(path)
            counts.append(count)
            
        for path, count in zip(target_files, counts):
            if count > 0:
                self.index_map.append((self.total_count, self.total_count + count, path))
                self.total_count += count
        
        print(f"✅ Map built: {self.total_count:,} samples indexed{' (Fast Mode)' if fast_mode else ''}.")

    def resolve(self, global_idx: int) -> Tuple[Path, int]:
        for start, end, path in self.index_map:
            if start <= global_idx < end:
                return path, global_idx - start
        raise IndexError(f"Index {global_idx} out of range (total: {self.total_count})")


class ArchiveStreamer:
    """Handles on-the-fly extraction for ZIP, 7z, and RAR."""
    def __init__(self, archive_path: Path):
        self.archive_path = archive_path
        self.suffix = archive_path.suffix.lower()

    def list_files(self) -> List[str]:
        if self.suffix == ".zip":
            with zipfile.ZipFile(self.archive_path, 'r') as f:
                return f.namelist()
        elif self.suffix == ".7z":
            import py7zr
            with py7zr.SevenZipFile(self.archive_path, mode='r') as f:
                return f.getnames()
        elif self.suffix == ".rar":
            import rarfile
            with rarfile.RarFile(self.archive_path) as f:
                return f.namelist()
        return []

    def get_content(self, file_name: str) -> bytes:
        if self.suffix == ".zip":
            with zipfile.ZipFile(self.archive_path, 'r') as f:
                return f.read(file_name)
        elif self.suffix == ".7z":
            import py7zr
            with py7zr.SevenZipFile(self.archive_path, mode='r') as f:
                extracted = f.read([file_name])
                return extracted[file_name].read()
        elif self.suffix == ".rar":
            import rarfile
            with rarfile.RarFile(self.archive_path) as f:
                return f.read(file_name)
        return b""

class UniversalDataLoader:
    def __init__(self, path: Union[str, Path], dataset_name: Optional[str] = None, fast_mode: bool = True, split: Optional[str] = None):
        self.path = Path(path)
        self.dataset_name = dataset_name or self.path.name
        self.index_map = None
        self.fast_mode = fast_mode
        self.split = split

    def detect_format(self) -> str:
        if not self.path.exists():
            raise FileNotFoundError(f"Path not found: {self.path}")
            
        if self.path.is_file():
            suffix = self.path.suffix.lower()
            if suffix in ('.zip', '.7z', '.rar'): return "archive"
            if suffix == '.parquet': return "parquet"
            if suffix == '.jsonl': return "jsonl"
            if suffix == '.json':
                # Peek to see if it's array or dict
                try:
                    with open(self.path, 'r', encoding='utf-8') as f:
                        content = f.read(1024).strip()
                        if content.startswith('['): return "json_array"
                        if content.startswith('{'): return "json_dict"
                except: pass
                return "json"
            if suffix == '.csv': return "csv"
            if suffix == '.txt': return "text"
        elif self.path.is_dir():
            # Check for sharded parquet
            if list(self.path.glob("data/*.parquet")):
                return "parquet_sharded"
            
            # Check for sharded jsonl
            if list(self.path.glob("data/*.jsonl")):
                return "jsonl_sharded"
            
            # Check for paired webp+txt
            if list(self.path.glob("*.webp")) and list(self.path.glob("*.txt")):
                return "paired_webp_txt"
            
            # Check for archives
            if list(self.path.glob("**/*.zip")) or list(self.path.glob("**/*.7z")) or list(self.path.glob("**/*.rar")):
                return "archive_dir"

            # Check for single files
            if list(self.path.glob("**/*.json")): return "json"
            if list(self.path.glob("**/*.csv")): return "csv"
            
        return "unknown"

    def load(self, sample_size: Optional[int] = None, recursion_depth: int = 0) -> LoadResult:
        """Load the entire dataset or a sample of it."""
        fmt = self.detect_format()
        dataset = []
        columns = []
        error = None
        
        try:
            # Handle directory recursion for generic discovery
            # Optimization: Skip recursion if we already have a format AND a sample size is requested
            if self.path.is_dir() and fmt not in ("parquet_sharded", "jsonl_sharded", "paired_webp_txt") and \
               not (fmt != "unknown" and sample_size):
                
                # Limit recursion depth to avoid infinite loops/extreme depth
                if recursion_depth < 3:
                    # Find all data files in subdirectories
                    all_files = []
                    for ext in [".json", ".jsonl", ".csv", ".parquet"]:
                        all_files.extend(list(self.path.rglob(f"*{ext}")))
                    
                    # Sort for consistency
                    all_files.sort()
                    
                    for file_path in all_files:
                        if file_path.resolve() == self.path.resolve():
                            continue
                            
                        sub_loader = UniversalDataLoader(file_path, fast_mode=self.fast_mode)
                        # Pass sample size to sub-loaders too
                        res = sub_loader.load(sample_size=sample_size, recursion_depth=recursion_depth + 1)
                        
                        if isinstance(res.dataset, list):
                            dataset.extend(res.dataset)
                        elif res.dataset:
                            dataset.append(res.dataset)
                        
                        if sample_size and len(dataset) >= sample_size:
                            break
            
            elif fmt == "json_array" or fmt == "json":
                if sample_size and sample_size > 0:
                    try:
                        import ijson
                        with open(self.path, 'rb') as f:
                            # Use ijson to stream items from array
                            parser = ijson.items(f, 'item')
                            for i, item in enumerate(parser):
                                dataset.append(item)
                                if len(dataset) >= sample_size:
                                    break
                        if not dataset:
                             # Try parsing as a list of dicts if 'item' failed
                             with open(self.path, 'rb') as f:
                                 parser = ijson.items(f, '')
                                 for i, item in enumerate(parser):
                                     if isinstance(item, list):
                                         dataset = item[:sample_size]
                                         break
                                     dataset.append(item)
                                     if len(dataset) >= sample_size:
                                         break
                    except Exception as e:
                        logger.warning(f"ijson failed, falling back to standard json: {e}")
                        with open(self.path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            dataset = data if isinstance(data, list) else [data]
                else:
                    with open(self.path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            dataset = data
                        else:
                            dataset = [data]
            elif fmt == "json_dict":
                if sample_size and sample_size > 0:
                    try:
                        import ijson
                        with open(self.path, 'rb') as f:
                            # Stream values from dict
                            parser = ijson.kvitems(f, '')
                            for i, (k, v) in enumerate(parser):
                                dataset.append(v)
                                if len(dataset) >= sample_size:
                                    break
                    except Exception as e:
                        logger.warning(f"ijson failed for dict, falling back: {e}")
                        with open(self.path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            dataset = list(data.values()) if isinstance(data, dict) else [data]
                else:
                    with open(self.path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        dataset = list(data.values()) if isinstance(data, dict) else [data]
            elif fmt == "jsonl":
                with open(self.path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            dataset.append(json.loads(line))
            elif fmt == "csv":
                import pandas as pd
                df = pd.read_csv(self.path)
                dataset = df.to_dict('records')
            elif fmt == "parquet":
                import pandas as pd
                df = pd.read_parquet(self.path)
                dataset = df.to_dict('records')
            elif fmt == "text":
                with open(self.path, 'r', encoding='utf-8') as f:
                    dataset = [{"text": line.strip()} for line in f if line.strip()]
            elif fmt in ("parquet_sharded", "jsonl_sharded"):
                # For sharded, we only load a sample if sample_size is provided, 
                # otherwise we might OOM if we try to load everything.
                count = sample_size or 100 # Default to 100 if no size
                dataset = [self.load_sample(i) for i in range(count)]
            else:
                error = f"Unsupported format: {fmt}"
                
            if sample_size and sample_size > 0 and len(dataset) > sample_size:
                dataset = dataset[:sample_size]
                
            if dataset and isinstance(dataset[0], dict):
                columns = list(dataset[0].keys())
                
        except Exception as e:
            error = str(e)
            logger.error(f"Failed to load dataset {self.path}: {e}")
            
        return LoadResult(
            dataset=dataset,
            format=fmt,
            num_samples=len(dataset),
            columns=columns,
            source_path=str(self.path),
            error=error
        )

    def _get_shards(self, extension: str) -> List[Path]:
        all_shards = list(self.path.glob(f"**/*{extension}"))
        if self.split:
            filtered = [f for f in all_shards if self.split.lower() in f.name.lower()]
            if filtered: return sorted(filtered)
        return sorted(all_shards)

    def load_sample(self, index: int) -> Dict[str, Any]:
        fmt = self.detect_format()
        try:
            if fmt == "parquet_sharded":
                if not self.index_map:
                    shards = self._get_shards(".parquet")
                    self.index_map = GlobalIndexMap(shards, "parquet")
                    self.index_map._build_map(fast_mode=self.fast_mode)
                file_path, local_idx = self.index_map.resolve(index)
                
                import pyarrow.parquet as pq
                table = pq.read_table(file_path)
                sample_table = table.slice(local_idx, 1)
                sample = {k: v[0] for k, v in sample_table.to_pydict().items()}
                return SchemaNormalizer.normalize(sample, self.dataset_name) if SchemaNormalizer else sample

            elif fmt == "jsonl_sharded":
                if not self.index_map:
                    shards = self._get_shards(".jsonl")
                    self.index_map = GlobalIndexMap(shards, "jsonl")
                    self.index_map._build_map(fast_mode=self.fast_mode)
                file_path, local_idx = self.index_map.resolve(index)
                with open(file_path, 'r', encoding='utf-8') as f:
                    for i, line in enumerate(f):
                        if i == local_idx:
                            sample = json.loads(line)
                            return SchemaNormalizer.normalize(sample, self.dataset_name) if SchemaNormalizer else sample
                return {}


            elif fmt == "paired_webp_txt":
                webp_files = sorted(list(self.path.glob("*.webp")))
                if not webp_files: return {}
                target_webp = webp_files[index % len(webp_files)]
                target_txt = target_webp.with_suffix(".txt")
                
                caption = ""
                if target_txt.exists():
                    with open(target_txt, 'r', encoding='utf-8') as f:
                        caption = f.read().strip()
                
                return {
                    "id": target_webp.stem,
                    "messages": [{"role": "user", "content": "Describe this image."}, {"role": "assistant", "content": caption}],
                    "modalities": {"image": [{"path": str(target_webp)}]}
                }
            
            elif fmt == "parquet":
                import pandas as pd
                df = pd.read_parquet(self.path, engine='pyarrow')
                sample = df.iloc[index % len(df)].to_dict()
                return SchemaNormalizer.normalize(sample, self.dataset_name) if SchemaNormalizer else sample

            elif fmt == "jsonl":
                with open(self.path, 'r', encoding='utf-8') as f:
                    for i, line in enumerate(f):
                        if i == index:
                            sample = json.loads(line)
                            return SchemaNormalizer.normalize(sample, self.dataset_name) if SchemaNormalizer else sample
                return {}

            elif fmt == "json":
                # Handle directory of JSONs or single JSON
                json_files = [self.path] if self.path.is_file() else list(self.path.glob("*.json"))
                if not json_files: return {}
                target_json = json_files[index % len(json_files)]
                with open(target_json, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        sample = data[index % len(data)]
                    else:
                        sample = data # Could be a dict sample
                    return SchemaNormalizer.normalize(sample, self.dataset_name) if SchemaNormalizer else sample

            elif fmt == "csv":
                csv_files = [self.path] if self.path.is_file() else list(self.path.glob("*.csv"))
                if not csv_files: return {}
                target_csv = csv_files[index % len(csv_files)]
                import pandas as pd
                df = pd.read_csv(target_csv)
                sample = df.iloc[index % len(df)].to_dict()
                return SchemaNormalizer.normalize(sample, self.dataset_name) if SchemaNormalizer else sample

            elif fmt == "archive_dir":
                archives = list(self.path.glob("*.zip")) + list(self.path.glob("*.7z")) + list(self.path.glob("*.rar"))
                if not archives: return {}
                target_archive = archives[index % len(archives)]
                streamer = ArchiveStreamer(target_archive)
                files = streamer.list_files()
                # Simple logic: pick a file from archive
                file_name = files[index % len(files)]
                content = streamer.get_content(file_name)
                # Try to parse content
                try:
                    sample = json.loads(content.decode('utf-8'))
                    return SchemaNormalizer.normalize(sample, self.dataset_name) if SchemaNormalizer else sample
                except:
                    return {"id": file_name, "content": "Binary or non-JSON data"}
                
        except Exception as e:
            if corruption_tracker:
                corruption_tracker.log_corrupted(str(self.path), str(e))
            # raise e # Don't raise, just return empty
        return {}
