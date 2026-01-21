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
                return self.SUPPORTED_FORMATS[suffix]
            # Check content for JSON vs JSONL
            if suffix == ".json":
                return self._detect_json_format()
            return "unknown"
        
        # Directory - check contents
        if self.path.is_dir():
            # Check for HF dataset format
            if (self.path / "dataset_info.json").exists():
                return "huggingface"
            if list(self.path.glob("*.arrow")):
                return "arrow"
            # Check for data files
            for ext in [".parquet", ".json", ".jsonl", ".csv"]:
                files = list(self.path.glob(f"**/*{ext}"))
                if files:
                    return self.SUPPORTED_FORMATS.get(ext, "unknown")
        
        return "unknown"
    
    def _detect_json_format(self) -> str:
        """Detect if JSON file is array or line-delimited."""
        try:
            with open(self.path, 'r', encoding='utf-8') as f:
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
        
        arrow_files = list(self.path.glob("*.arrow")) if self.path.is_dir() else [self.path]
        
        tables = []
        for f in arrow_files[:1]:  # Load first file
            with open(f, 'rb') as fp:
                reader = pa.ipc.open_file(fp)
                tables.append(reader.read_all())
        
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
        
        with open(self.path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, dict):
            # Convert dict to list
            data = list(data.values())
        
        if sample_size:
            data = data[:sample_size]
        
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
        from datasets import Dataset
        
        with open(self.path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, dict):
            data = list(data.values())
        
        if sample_size:
            data = data[:sample_size]
        
        ds = Dataset.from_list(data)
        
        return LoadResult(
            dataset=ds,
            format="json_dict",
            num_samples=len(ds),
            columns=ds.column_names,
            source_path=str(self.path),
        )
    
    def _load_jsonl(self, sample_size: Optional[int]) -> LoadResult:
        """Load JSONL (newline-delimited JSON) format."""
        from datasets import Dataset
        
        data = []
        file_path = self.path if self.path.is_file() else self._find_first_file([".jsonl", ".json"])
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if sample_size and i >= sample_size:
                    break
                line = line.strip()
                if line:
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        
        if not data:
            raise ValueError("No valid JSONL data found")
        
        ds = Dataset.from_list(data)
        
        return LoadResult(
            dataset=ds,
            format="jsonl",
            num_samples=len(ds),
            columns=ds.column_names,
            source_path=str(file_path),
        )
    
    def _load_parquet(self, sample_size: Optional[int]) -> LoadResult:
        """Load Parquet format."""
        from datasets import load_dataset
        
        if self.path.is_dir():
            parquet_files = list(self.path.glob("**/*.parquet"))
            if not parquet_files:
                raise ValueError("No parquet files found")
            file_path = str(parquet_files[0])
        else:
            file_path = str(self.path)
        
        ds = load_dataset("parquet", data_files=file_path, split="train")
        
        if sample_size and len(ds) > sample_size:
            ds = ds.select(range(sample_size))
        
        return LoadResult(
            dataset=ds,
            format="parquet",
            num_samples=len(ds),
            columns=ds.column_names,
            source_path=file_path,
        )
    
    def _load_csv(self, sample_size: Optional[int]) -> LoadResult:
        """Load CSV format."""
        from datasets import load_dataset
        
        file_path = self.path if self.path.is_file() else self._find_first_file([".csv"])
        
        ds = load_dataset("csv", data_files=str(file_path), split="train")
        
        if sample_size and len(ds) > sample_size:
            ds = ds.select(range(sample_size))
        
        return LoadResult(
            dataset=ds,
            format="csv",
            num_samples=len(ds),
            columns=ds.column_names,
            source_path=str(file_path),
        )
    
    def _load_text(self, sample_size: Optional[int]) -> LoadResult:
        """Load plain text format (one sample per line)."""
        from datasets import Dataset
        
        file_path = self.path if self.path.is_file() else self._find_first_file([".txt"])
        
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if sample_size and i >= sample_size:
                    break
                if line.strip():
                    data.append({"text": line.strip()})
        
        ds = Dataset.from_list(data)
        
        return LoadResult(
            dataset=ds,
            format="text",
            num_samples=len(ds),
            columns=["text"],
            source_path=str(file_path),
        )
    
    def _find_first_file(self, extensions: List[str]) -> Path:
        """Find first file matching extensions."""
        if self.path.is_file():
            return self.path
        
        for ext in extensions:
            files = list(self.path.glob(f"**/*{ext}"))
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
