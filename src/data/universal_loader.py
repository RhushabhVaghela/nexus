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
import gzip
import zipfile
import tarfile
import shutil
import tempfile
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
    """Universal dataset loader that auto-detects and loads any format.
    
    Supports STREAMING extraction from archives (.gz, .zip, .tar.gz)
    without fully decompressing - efficient for sampling.
    """
    
    SUPPORTED_FORMATS = {
        ".json": "json",
        ".jsonl": "jsonl",
        ".parquet": "parquet",
        ".csv": "csv",
        ".arrow": "arrow",
        ".txt": "text",
    }
    
    ARCHIVE_FORMATS = {
        ".gz": "gzip",      # Gzip (including .jsonl.gz, .json.gz)
        ".zip": "zip",
        ".tar": "tar",
        ".tgz": "tar_gzip",
        ".rar": "rar",      # Requires unrar
    }
    
    def __init__(self, path: Union[str, Path]):
        """
        Args:
            path: Path to dataset file or directory
        """
        self.path = Path(path)
        self._detected_format = None
    
    def detect_format(self) -> str:
        """Auto-detect dataset format, including archives for streaming load."""
        if not self.path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.path}")
        
        if self.path.is_file():
            suffix = self.path.suffix.lower()
            
            # Check for archive formats FIRST (for streaming extraction)
            if suffix in self.ARCHIVE_FORMATS:
                # Detect inner format: e.g., .jsonl.gz -> "gzip_jsonl"
                inner_suffix = Path(self.path.stem).suffix.lower()
                archive_type = self.ARCHIVE_FORMATS[suffix]
                if inner_suffix:
                    return f"{archive_type}_{inner_suffix.lstrip('.')}"
                return archive_type
            
            # Check for .tar.gz (compound suffix)
            if self.path.name.endswith(".tar.gz"):
                return "tar_gzip"
            
            # Standard formats
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
            
    def _try_streaming_loader(self, sample_size: Optional[int]) -> Optional[LoadResult]:
        """Try using StreamingDatasetLoader for large datasets."""
        try:
            from src.data.streaming_trainer import StreamingDatasetLoader, StreamingConfig
            
            # Convert sample_size to max_samples
            config = StreamingConfig(buffer_size=10000, max_samples=sample_size)
            loader = StreamingDatasetLoader([str(self.path)], config)
            
            # Check if it's worth streaming (size check is internal to loader usually, but we can try)
            # If it works, it returns an IterableDataset
            dataset = loader.get_streaming_dataset()
            
            return LoadResult(
                dataset=dataset,
                format="streaming",
                num_samples=sample_size or 0, # Unknown total
                columns=[], # Unknown until iterated
                source_path=str(self.path)
            )
        except Exception as e:
            return None

    def load(self, sample_size: Optional[int] = None, split: str = "train") -> LoadResult:
        """
        Load dataset with auto-format detection.
        """
        fmt = self.detect_format()
        self._detected_format = fmt
        
        # AUTO-STREAMING CHECK
        # If file is > 1GB or directory, try streaming first for memory safety
        should_stream = False
        if self.path.is_file() and self.path.stat().st_size > 1024**3:
            should_stream = True
        elif self.path.is_dir():
            # Rough check
            total_size = sum(f.stat().st_size for f in self.path.glob("**/*") if f.is_file())
            if total_size > 1024**3:
                should_stream = True
                
        if should_stream:
            logger.info(f"🌊 Large dataset detected ({fmt}), attempting streaming load...")
            res = self._try_streaming_loader(sample_size)
            if res: 
                return res
        
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
            # Streaming archive formats (efficient - no full extraction)
            elif fmt.startswith("gzip_"):
                return self._load_gzip_stream(sample_size, fmt)
            elif fmt.startswith("zip"):
                return self._load_zip_stream(sample_size)
            elif fmt.startswith("tar"):
                return self._load_tar_stream(sample_size)
            elif fmt == "rar":
                return self._load_rar_stream(sample_size)
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

    def _load_gzip_stream(self, sample_size: Optional[int], fmt: str) -> LoadResult:
        """
        STREAMING gzip load - reads samples directly from compressed stream.
        Only decompresses the bytes needed for sampling - NOT the full file.
        
        Supports: .jsonl.gz, .json.gz, .csv.gz
        """
        from datasets import Dataset
        
        inner_format = fmt.replace("gzip_", "")  # e.g., "jsonl"
        data = []
        
        logger.info(f"📦 Streaming from gzip ({inner_format}) - sample_size={sample_size}")
        
        with gzip.open(self.path, 'rt', encoding='utf-8') as f:
            if inner_format in ("jsonl", "json"):
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            data.append(json.loads(line))
                            if sample_size and len(data) >= sample_size:
                                break  # STOP READING - efficient!
                        except json.JSONDecodeError:
                            continue
            elif inner_format == "csv":
                import csv
                reader = csv.DictReader(f)
                for row in reader:
                    data.append(dict(row))
                    if sample_size and len(data) >= sample_size:
                        break
            else:
                # Text fallback
                for line in f:
                    if line.strip():
                        data.append({"text": line.strip()})
                        if sample_size and len(data) >= sample_size:
                            break
        
        if not data:
            raise ValueError(f"No data found in gzip stream: {self.path}")
        
        ds = Dataset.from_list(data)
        return LoadResult(
            dataset=ds,
            format=fmt,
            num_samples=len(ds),
            columns=ds.column_names,
            source_path=str(self.path),
        )
    
    def _load_zip_stream(self, sample_size: Optional[int]) -> LoadResult:
        """
        STREAMING zip load - reads from first matching file in zip.
        Only extracts/reads the bytes needed.
        """
        from datasets import Dataset
        
        data = []
        logger.info(f"📦 Streaming from zip - sample_size={sample_size}")
        
        with zipfile.ZipFile(self.path, 'r') as zf:
            # Find first data file
            for name in zf.namelist():
                suffix = Path(name).suffix.lower()
                if suffix in (".jsonl", ".json", ".csv", ".txt"):
                    with zf.open(name) as f:
                        text_stream = f.read().decode('utf-8')
                        
                        if suffix in (".jsonl", ".json"):
                            for line in text_stream.split('\n'):
                                line = line.strip()
                                if line:
                                    try:
                                        data.append(json.loads(line))
                                        if sample_size and len(data) >= sample_size:
                                            break
                                    except json.JSONDecodeError:
                                        continue
                        elif suffix == ".csv":
                            import csv
                            import io
                            reader = csv.DictReader(io.StringIO(text_stream))
                            for row in reader:
                                data.append(dict(row))
                                if sample_size and len(data) >= sample_size:
                                    break
                        else:
                            for line in text_stream.split('\n'):
                                if line.strip():
                                    data.append({"text": line.strip()})
                                    if sample_size and len(data) >= sample_size:
                                        break
                    
                    if sample_size and len(data) >= sample_size:
                        break
        
        if not data:
            raise ValueError(f"No data found in zip: {self.path}")
        
        ds = Dataset.from_list(data)
        return LoadResult(
            dataset=ds,
            format="zip",
            num_samples=len(ds),
            columns=ds.column_names,
            source_path=str(self.path),
        )
    
    def _load_tar_stream(self, sample_size: Optional[int]) -> LoadResult:
        """
        STREAMING tar/tar.gz load - reads from compressed tar stream.
        """
        from datasets import Dataset
        
        data = []
        logger.info(f"📦 Streaming from tar - sample_size={sample_size}")
        
        mode = "r:gz" if ".gz" in str(self.path) or ".tgz" in str(self.path) else "r"
        
        with tarfile.open(self.path, mode) as tf:
            for member in tf:
                if member.isfile():
                    suffix = Path(member.name).suffix.lower()
                    if suffix in (".jsonl", ".json", ".csv", ".txt"):
                        f = tf.extractfile(member)
                        if f:
                            content = f.read().decode('utf-8')
                            
                            if suffix in (".jsonl", ".json"):
                                for line in content.split('\n'):
                                    line = line.strip()
                                    if line:
                                        try:
                                            data.append(json.loads(line))
                                            if sample_size and len(data) >= sample_size:
                                                break
                                        except json.JSONDecodeError:
                                            continue
                            else:
                                for line in content.split('\n'):
                                    if line.strip():
                                        data.append({"text": line.strip()})
                                        if sample_size and len(data) >= sample_size:
                                            break
                
                if sample_size and len(data) >= sample_size:
                    break
        
        if not data:
            raise ValueError(f"No data found in tar: {self.path}")
        
        ds = Dataset.from_list(data)
        return LoadResult(
            dataset=ds,
            format="tar",
            num_samples=len(ds),
            columns=ds.column_names,
            source_path=str(self.path),
        )
    
    def _load_rar_stream(self, sample_size: Optional[int]) -> LoadResult:
        """
        STREAMING RAR load - reads from RAR archive.
        Requires: rarfile library OR unrar command-line tool.
        """
        from datasets import Dataset
        
        data = []
        logger.info(f"📦 Streaming from RAR - sample_size={sample_size}")
        
        try:
            # Method 1: Try rarfile library (pip install rarfile)
            import rarfile
            
            with rarfile.RarFile(self.path, 'r') as rf:
                for name in rf.namelist():
                    suffix = Path(name).suffix.lower()
                    if suffix in (".jsonl", ".json", ".csv", ".txt"):
                        with rf.open(name) as f:
                            content = f.read().decode('utf-8')
                            
                            if suffix in (".jsonl", ".json"):
                                for line in content.split('\n'):
                                    line = line.strip()
                                    if line:
                                        try:
                                            data.append(json.loads(line))
                                            if sample_size and len(data) >= sample_size:
                                                break
                                        except json.JSONDecodeError:
                                            continue
                            elif suffix == ".csv":
                                import csv
                                import io
                                reader = csv.DictReader(io.StringIO(content))
                                for row in reader:
                                    data.append(dict(row))
                                    if sample_size and len(data) >= sample_size:
                                        break
                            else:
                                for line in content.split('\n'):
                                    if line.strip():
                                        data.append({"text": line.strip()})
                                        if sample_size and len(data) >= sample_size:
                                            break
                        
                        if sample_size and len(data) >= sample_size:
                            break
                            
        except ImportError:
            # Method 2: Fallback to unrar command
            import subprocess
            import tempfile
            
            logger.info("rarfile not installed, using unrar command...")
            
            # List files in RAR
            try:
                result = subprocess.run(
                    ["unrar", "l", str(self.path)],
                    capture_output=True,
                    encoding="utf-8"
                )
                
                # Extract to temp and read only needed files
                with tempfile.TemporaryDirectory() as tmpdir:
                    # Extract only first matching file
                    for line in result.stdout.split('\n'):
                        for ext in (".jsonl", ".json", ".csv", ".txt"):
                            if ext in line.lower():
                                # Extract this file
                                subprocess.run(
                                    ["unrar", "e", "-o+", str(self.path), tmpdir],
                                    capture_output=True
                                )
                                
                                # Read from extracted
                                for f in Path(tmpdir).glob(f"*{ext}"):
                                    with open(f, 'r', encoding='utf-8') as fp:
                                        for line in fp:
                                            if line.strip():
                                                try:
                                                    data.append(json.loads(line))
                                                except:
                                                    data.append({"text": line.strip()})
                                                if sample_size and len(data) >= sample_size:
                                                    break
                                    break
                                break
                        if data:
                            break
            except FileNotFoundError:
                raise RuntimeError("RAR extraction requires 'rarfile' library or 'unrar' command")
        
        if not data:
            raise ValueError(f"No data found in RAR: {self.path}")
        
        ds = Dataset.from_list(data)
        return LoadResult(
            dataset=ds,
            format="rar",
            num_samples=len(ds),
            columns=ds.column_names,
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
