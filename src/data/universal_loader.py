#!/usr/bin/env python3
"""
universal_loader.py
Universal dataset loader that handles any format automatically.

Supports:
- JSON (array or dict)
- JSONL (newline-delimited JSON)
- Parquet
- CSV
- HuggingFace Dataset format (arrow)
- Directory with multiple files

Usage:
    from src.data.universal_loader import UniversalDataLoader
    
    loader = UniversalDataLoader("/path/to/dataset")
    dataset = loader.load(sample_size=100)
"""

import os
import json
import csv
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class LoadResult:
    """Result of dataset loading."""
    dataset: Any
    format: str
    num_samples: int
    columns: List[str]
    source_path: str
    error: Optional[str] = None


class UniversalDataLoader:
    """Universal dataset loader that auto-detects and loads any format."""
    
    SUPPORTED_FORMATS = {
        ".json": "json",
        ".jsonl": "jsonl",
        ".parquet": "parquet",
        ".csv": "csv",
        ".arrow": "arrow",
        ".txt": "text",
    }
    
    def __init__(self, path: Union[str, Path]):
        """
        Args:
            path: Path to dataset file or directory
        """
        self.path = Path(path)
        self._detected_format = None
    
    def detect_format(self) -> str:
        """Auto-detect dataset format."""
        if not self.path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.path}")
        
        if self.path.is_file():
            suffix = self.path.suffix.lower()
            if suffix in self.SUPPORTED_FORMATS:
                if suffix == ".json":
                    return self._detect_json_format(self.path)
                return self.SUPPORTED_FORMATS[suffix]
            return "unknown"
        
        # Directory - check contents
        if self.path.is_dir():
            # Check for HF dataset format
            if (self.path / "dataset_info.json").exists():
                return "huggingface"
            if list(self.path.glob("**/*.arrow")):
                return "arrow"
            # Check for data files
            # Priority: parquet > jsonl > json > csv > text
            for ext in [".parquet", ".jsonl", ".json", ".csv", ".txt"]:
                files = list(self.path.glob(f"**/*{ext}"))
                if files:
                    if ext == ".json":
                        # Check first JSON file to see if it's array or line-delimited
                        return self._detect_json_format(files[0])
                    return self.SUPPORTED_FORMATS.get(ext, "unknown")
        
        return "unknown"
    
    def _detect_json_format(self, file_path: Path) -> str:
        """Detect if JSON file is array or line-delimited."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                first_char = f.read(1).strip()
                if first_char == '[':
                    return "json_array"
                elif first_char == '{':
                    # Could be dict or JSONL
                    f.seek(0)
                    first_line = f.readline()
                    try:
                        json.loads(first_line)
                        second_line = f.readline()
                        if second_line.strip():
                            try:
                                json.loads(second_line)
                                return "jsonl"
                            except:
                                return "json_dict"
                        return "json_dict"
                    except:
                        return "json_dict"
        except Exception:
            pass
        return "json"
    
    def load(self, sample_size: Optional[int] = None, split: str = "train") -> LoadResult:
        """
        Load dataset with auto-format detection.
        
        Args:
            sample_size: Max samples to load (None for all)
            split: Dataset split to load (for HF format)
        
        Returns:
            LoadResult with dataset and metadata
        """
        fmt = self.detect_format()
        self._detected_format = fmt
        
        logger.info(f"Loading {self.path} (format: {fmt})")
        
        try:
            if fmt == "huggingface":
                return self._load_huggingface(sample_size, split)
            elif fmt == "arrow":
                return self._load_arrow(sample_size)
            elif fmt in ("json", "json_array"):
                return self._load_json_array(sample_size)
            elif fmt in ("json_dict",):
                return self._load_json_dict(sample_size)
            elif fmt == "jsonl":
                return self._load_jsonl(sample_size)
            elif fmt == "parquet":
                return self._load_parquet(sample_size)
            elif fmt == "csv":
                return self._load_csv(sample_size)
            elif fmt == "text":
                return self._load_text(sample_size)
            else:
                # Try all formats
                return self._try_all_formats(sample_size)
                
        except Exception as e:
            logger.error(f"Load failed: {e}")
            return LoadResult(
                dataset=None,
                format=fmt,
                num_samples=0,
                columns=[],
                source_path=str(self.path),
                error=str(e),
            )
    
    def _load_huggingface(self, sample_size: Optional[int], split: str) -> LoadResult:
        """Load HuggingFace dataset format."""
        from datasets import load_from_disk
        
        ds = load_from_disk(str(self.path))
        
        # Handle DatasetDict
        if hasattr(ds, 'keys'):
            if split in ds:
                ds = ds[split]
            else:
                ds = ds[list(ds.keys())[0]]
        
        if sample_size and len(ds) > sample_size:
            ds = ds.select(range(sample_size))
        
        return LoadResult(
            dataset=ds,
            format="huggingface",
            num_samples=len(ds),
            columns=ds.column_names if hasattr(ds, 'column_names') else [],
            source_path=str(self.path),
        )
    
    def _load_arrow(self, sample_size: Optional[int]) -> LoadResult:
        """Load Arrow format files."""
        from datasets import Dataset
        import pyarrow as pa
        
        arrow_files = self._get_all_files([".arrow"])
        
        tables = []
        for f in arrow_files:
            try:
                with open(f, 'rb') as fp:
                    reader = pa.ipc.open_file(fp)
                    tables.append(reader.read_all())
                
                # Check if we have enough samples to stop loading more files
                if sample_size:
                    current_total = sum(len(t) for t in tables)
                    if current_total >= sample_size:
                        break
            except Exception as e:
                logger.warning(f"Failed to load arrow file {f}: {e}")
        
        if tables:
            table = pa.concat_tables(tables) if len(tables) > 1 else tables[0]
            ds = Dataset(table)
            
            if sample_size and len(ds) > sample_size:
                ds = ds.select(range(sample_size))
            
            return LoadResult(
                dataset=ds,
                format="arrow",
                num_samples=len(ds),
                columns=ds.column_names,
                source_path=str(self.path),
            )
        
        raise ValueError("No arrow files found")
    
    def _load_json_array(self, sample_size: Optional[int]) -> LoadResult:
        """Load JSON array format."""
        from datasets import Dataset
        
        json_files = self._get_all_files([".json"])
        data = []
        
        for file_path in json_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    file_data = json.load(f)
                
                if isinstance(file_data, dict):
                    file_data = list(file_data.values())
                
                if isinstance(file_data, list):
                    data.extend(file_data)
                
                if sample_size and len(data) >= sample_size:
                    data = data[:sample_size]
                    break
            except Exception as e:
                logger.warning(f"Error loading {file_path}: {e}")
        
        if not data:
            raise ValueError("No JSON data found")
            
        ds = Dataset.from_list(data)
        
        return LoadResult(
            dataset=ds,
            format="json_array",
            num_samples=len(ds),
            columns=ds.column_names,
            source_path=str(self.path),
        )
    
    def _load_json_dict(self, sample_size: Optional[int]) -> LoadResult:
        """Load JSON dict format (key: sample)."""
        # Same as json_array for loading logic
        return self._load_json_array(sample_size)
    
    def _load_jsonl(self, sample_size: Optional[int]) -> LoadResult:
        """Load JSONL (newline-delimited JSON) format."""
        from datasets import Dataset
        
        data = []
        jsonl_files = self._get_all_files([".jsonl", ".json"])
        
        for file_path in jsonl_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                data.append(json.loads(line))
                                if sample_size and len(data) >= sample_size:
                                    break
                            except json.JSONDecodeError:
                                continue
                if sample_size and len(data) >= sample_size:
                    break
            except Exception as e:
                logger.warning(f"Error loading {file_path}: {e}")
        
        if not data:
            raise ValueError("No valid JSONL data found")
        
        ds = Dataset.from_list(data)
        
        return LoadResult(
            dataset=ds,
            format="jsonl",
            num_samples=len(ds),
            columns=ds.column_names,
            source_path=str(self.path),
        )
    
    def _load_parquet(self, sample_size: Optional[int]) -> LoadResult:
        """Load Parquet format."""
        from datasets import load_dataset
        
        parquet_files = [str(f) for f in self._get_all_files([".parquet"])]
        if not parquet_files:
            raise ValueError("No parquet files found")
        
        # load_dataset can take a list of files
        ds = load_dataset("parquet", data_files={"train": parquet_files}, split="train")
        
        if sample_size and len(ds) > sample_size:
            ds = ds.select(range(sample_size))
        
        return LoadResult(
            dataset=ds,
            format="parquet",
            num_samples=len(ds),
            columns=ds.column_names,
            source_path=str(self.path),
        )
    
    def _load_csv(self, sample_size: Optional[int]) -> LoadResult:
        """Load CSV format."""
        from datasets import load_dataset
        
        csv_files = [str(f) for f in self._get_all_files([".csv"])]
        if not csv_files:
            raise ValueError("No CSV files found")
            
        ds = load_dataset("csv", data_files={"train": csv_files}, split="train")
        
        if sample_size and len(ds) > sample_size:
            ds = ds.select(range(sample_size))
        
        return LoadResult(
            dataset=ds,
            format="csv",
            num_samples=len(ds),
            columns=ds.column_names,
            source_path=str(self.path),
        )
    
    def _load_text(self, sample_size: Optional[int]) -> LoadResult:
        """Load plain text format (one sample per line)."""
        from datasets import Dataset
        
        text_files = self._get_all_files([".txt"])
        data = []
        
        for file_path in text_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            data.append({"text": line.strip()})
                            if sample_size and len(data) >= sample_size:
                                break
                if sample_size and len(data) >= sample_size:
                    break
            except Exception as e:
                logger.warning(f"Error loading {file_path}: {e}")
        
        if not data:
            raise ValueError("No text data found")
            
        ds = Dataset.from_list(data)
        
        return LoadResult(
            dataset=ds,
            format="text",
            num_samples=len(ds),
            columns=["text"],
            source_path=str(self.path),
        )

    def _get_all_files(self, extensions: List[str]) -> List[Path]:
        """Get all files matching extensions in a directory (recursive) or self.path if it's a file."""
        if self.path.is_file():
            suffix = self.path.suffix.lower()
            if suffix in extensions:
                return [self.path]
            return []
        
        all_files = []
        for ext in extensions:
            all_files.extend(list(self.path.glob(f"**/*{ext}")))
            
        # Optional: Sort files to ensure deterministic loading order
        all_files.sort()
        
        return all_files

    def _find_first_file(self, extensions: List[str]) -> Path:
        """Find first file matching extensions."""
        files = self._get_all_files(extensions)
        if files:
            return files[0]
        
        raise FileNotFoundError(f"No files with extensions {extensions} found in {self.path}")
    
    def _try_all_formats(self, sample_size: Optional[int]) -> LoadResult:
        """Try all formats until one works."""
        errors = []
        
        for method_name in ['_load_jsonl', '_load_json_array', '_load_parquet', 
                           '_load_csv', '_load_huggingface']:
            try:
                method = getattr(self, method_name)
                if method_name == '_load_huggingface':
                    return method(sample_size, "train")
                return method(sample_size)
            except Exception as e:
                errors.append(f"{method_name}: {e}")
        
        raise ValueError(f"All formats failed: {errors}")


def load_dataset_universal(path: Union[str, Path], 
                          sample_size: Optional[int] = None,
                          split: str = "train") -> LoadResult:
    """
    Convenience function to load any dataset format.
    
    Args:
        path: Path to dataset
        sample_size: Max samples to load
        split: Dataset split
    
    Returns:
        LoadResult with dataset
    """
    loader = UniversalDataLoader(path)
    return loader.load(sample_size=sample_size, split=split)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Universal Dataset Loader")
    parser.add_argument("path", help="Path to dataset")
    parser.add_argument("--sample-size", type=int, default=5, help="Samples to load")
    args = parser.parse_args()
    
    result = load_dataset_universal(args.path, sample_size=args.sample_size)
    
    print(f"\n{'='*60}")
    print(f"Dataset: {result.source_path}")
    print(f"Format: {result.format}")
    print(f"Samples: {result.num_samples}")
    print(f"Columns: {result.columns}")
    if result.error:
        print(f"Error: {result.error}")
    else:
        print(f"\nFirst sample:")
        print(result.dataset[0] if result.dataset else "None")
    print(f"{'='*60}")
