"""
Universal Dataset Manager

Handles:
1. Loading datasets from the new domain-structured layout.
2. Auto-detecting formats (json, jsonl, parquet, arrow).
3. Unified splitting logic (Traing/Val/Test).
4. Providing a simple interface for training scripts.
5. Dynamic category discovery.
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
from dataclasses import dataclass

from datasets import load_dataset, Dataset, DatasetDict, concatenate_datasets

logger = logging.getLogger(__name__)

class UniversalDatasetManager:
    def __init__(self, mode: str = "default", data_root: str = "/mnt/e/data"):
        self.mode = mode.lower()
        self.data_root = Path(data_root)
        self.datasets_dir = self.data_root / "datasets"
        self.benchmarks_dir = self.data_root / "benchmarks"
        
        # Ensure directories exist
        if not self.datasets_dir.exists():
            logger.warning(f"Datasets dir not found: {self.datasets_dir}")

    def get_available_categories(self) -> List[str]:
        """Dynamically list all dataset categories found on disk, including the current mode."""
        if not self.datasets_dir.exists():
            return []
        cats = [d.name for d in self.datasets_dir.iterdir() if d.is_dir()]
        return cats

    def get_available_benchmarks(self) -> List[str]:
        """Dynamically list all benchmark categories, including the mode."""
        if not self.benchmarks_dir.exists():
            return []
        return [d.name for d in self.benchmarks_dir.iterdir() if d.is_dir()]

    def _detect_format(self, path: Path) -> str:
        """Auto-detect format of a dataset folder."""
        if not path.is_dir():
            s = path.suffix.lower()
            if s in ['.json', '.jsonl']: return 'json'
            if s in ['.parquet']: return 'parquet'
            if s in ['.arrow']: return 'arrow'
            if s in ['.csv']: return 'csv'
            return 'unknown'
            
        files = list(path.iterdir())
        for f in files:
            s = f.suffix.lower()
            if s == '.json': return 'json'
            if s == '.jsonl': return 'json'
            if s == '.parquet': return 'parquet'
            if s == '.arrow': return 'arrow'
            if s == '.csv': return 'csv'
        return 'unknown'

    def load_dataset_by_name(self, name: str, split: str = "train", limit: int = None) -> Optional[Dataset]:
        """Load a specific dataset by name (recursive search)."""
        found_path = None
        
        # Search datasets
        if (self.datasets_dir / name).exists():
            found_path = self.datasets_dir / name
        else:
            for cat in self.datasets_dir.iterdir():
                if cat.is_dir() and (cat / name).exists():
                    found_path = cat / name
                    break
        
        # Search benchmarks
        if not found_path and self.benchmarks_dir.exists():
            if (self.benchmarks_dir / name).exists():
                found_path = self.benchmarks_dir / name
            else:
                for cat in self.benchmarks_dir.iterdir():
                    if cat.is_dir() and (cat / name).exists():
                        found_path = cat / name
                        break
        
        if not found_path:
            logger.warning(f"Dataset {name} not found.")
            return None

        # Load
        fmt = self._detect_format(found_path)
        logger.info(f"Loading {name} [{fmt}] from {found_path}")
        
        try:
            if fmt == 'json':
                # Check for huggingface structure (dataset_dict.json)
                if (found_path / 'dataset_dict.json').exists():
                    ds_dict = load_dataset(str(found_path))
                    if split in ds_dict:
                        ds = ds_dict[split]
                    else:
                        ds = concatenate_datasets(list(ds_dict.values()))
                else:
                    # Raw json/jsonl files
                    ds = load_dataset("json", data_dir=str(found_path), split=split)
            elif fmt in ['parquet', 'arrow', 'csv']:
                ds = load_dataset(fmt, data_dir=str(found_path), split=split)
            else:
                # Fallback
                ds = load_dataset(str(found_path), split=split)

            if limit:
                ds = ds.select(range(min(limit, len(ds))))
            
            return ds

        except Exception as e:
            logger.error(f"Error loading {name}: {e}")
            return None

    def load_category(self, category: str, split: str = "train", limit: int = None) -> List[Dataset]:
        """Load all datasets in a given category."""
        target_dir = self.datasets_dir / category
        if not target_dir.exists():
            logger.warning(f"Category {category} not found.")
            return []
        
        datasets = []
        for item in target_dir.iterdir():
            if item.is_dir():
                ds = self.load_dataset_by_name(item.name, split=split, limit=limit)
                if ds:
                    datasets.append(ds)
        return datasets

    def get_unified_train_dataset(self, 
                                enabled_categories: List[str] = None, 
                                included_datasets: List[str] = None) -> Dataset:
        """
        Creates a single unified training dataset from selected categories/datasets.
        Normalizes interactions to 'messages' format.
        Automatically includes datasets from the current 'mode' folder (censored/uncensored).
        """
        all_ds = []
        
        # Determine categories to load
        cats = enabled_categories.copy() if enabled_categories else []
        
        # Only inject 'uncensored' as a special extension folder if requested
        if self.mode == "uncensored" and (self.datasets_dir / "uncensored").exists():
            if "uncensored" not in cats:
                logger.info(f"Extension mode: Adding 'uncensored' datasets")
                cats.append("uncensored")
        
        # Load categories
        if cats:
            for cat in cats:
                logger.info(f"Adding category: {cat}")
                cat_ds = self.load_category(cat)
                all_ds.extend(cat_ds)
        
        # Load specific datasets
        if included_datasets:
            for name in included_datasets:
                logger.info(f"Adding dataset: {name}")
                ds = self.load_dataset_by_name(name)
                if ds: 
                    all_ds.append(ds)
        
        if not all_ds:
            raise ValueError("No datasets loaded. Check categories/names.")
            
        # Normalize
        normalized = []
        for ds in all_ds:
            # Map common columns to 'messages'
            if 'messages' in ds.features:
                normalized.append(ds.select_columns(['messages']))
            elif 'conversations' in ds.features:
                ds = ds.rename_column('conversations', 'messages')
                normalized.append(ds.select_columns(['messages']))
            elif 'instruction' in ds.features and 'output' in ds.features:
                # Simple conversion map
                def to_messages(batch):
                    msgs = []
                    for inst, out in zip(batch['instruction'], batch['output']):
                         content = f"{inst}\n{batch.get('input',[''])[0]}"
                         msgs.append([
                             {"role": "user", "content": content.strip()},
                             {"role": "assistant", "content": out}
                         ])
                    return {"messages": msgs}
                
                ds = ds.map(to_messages, batched=True, remove_columns=ds.column_names)
                normalized.append(ds)
            else:
                logger.warning(f"Skipping dataset with features {ds.features} - cannot normalize to messages.")
        
        if not normalized:
            raise ValueError("No datasets could be normalized to 'messages' format.")

        logger.info(f"Concatenating {len(normalized)} datasets...")
        return concatenate_datasets(normalized)

    def split_dataset(self, dataset: Dataset, test_size=0.05, val_size=0.05) -> DatasetDict:
        """Universal splitter (90/5/5 default)."""
        train_test = dataset.train_test_split(test_size=(test_size + val_size))
        test_val = train_test['test'].train_test_split(test_size=val_size / (test_size + val_size))
        
        return DatasetDict({
            'train': train_test['train'],
            'validation': test_val['train'],
            'test': test_val['test']
        })
