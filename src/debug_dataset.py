
import os
import sys
import torch
import json
import itertools
from pathlib import Path
import logging

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DebugDataset")

class OmniDataset(torch.utils.data.IterableDataset):
    def __init__(self, data_path: str, split: str = "train", samples_per_dataset: int = 0):
        self.split = split
        self.limit = samples_per_dataset
        self.base_path = Path(data_path)
        self.dataset_counts = {}
        
        if not self.base_path.exists():
            logger.error(f"❌ Data path not found: {self.base_path}")
            
        self.dataset_dirs = [d for d in self.base_path.iterdir() if d.is_dir()] if self.base_path.is_dir() else [self.base_path]
        if not self.dataset_dirs: self.dataset_dirs = [self.base_path]
        
        logger.info(f"🌊 Initialized Streamable Dataset ({split}). Ready to stream from {len(self.dataset_dirs)} sources.")

    def _get_files_for_split(self):
        ALIASES = {
            "train": ["train", "training", "train_data"],
            "val": ["val", "validation", "eval", "evaluation", "dev"],
            "test": ["test", "testing"]
        }
        ALL_KNOWN_FOLDERS = set([name for sublist in ALIASES.values() for name in sublist])
        
        for ds_dir in self.dataset_dirs:
            try:
                subdirs = {d.name.lower() for d in ds_dir.iterdir() if d.is_dir()}
            except Exception:
                subdirs = set()
                
            has_explicit_structure = not subdirs.isdisjoint(ALL_KNOWN_FOLDERS)
            target_folders = []
            
            if has_explicit_structure:
                possible_names = ALIASES.get(self.split, [])
                for name in possible_names:
                    candidate = ds_dir / name
                    if candidate.exists():
                        target_folders.append(candidate)
                        
                if target_folders:
                    for folder in target_folders:
                         files_gen = itertools.chain(folder.rglob("*.jsonl"), folder.rglob("*.json"))
                         for p in files_gen: 
                             yield p
            else:
                all_files_gen = itertools.chain(ds_dir.rglob("*.jsonl"), ds_dir.rglob("*.json"))
                for p in all_files_gen:
                    h = hash(p.name) % 100
                    is_train = h < 90
                    is_val = 90 <= h < 95
                    is_test = h >= 95
                    
                    if self.split == "train" and is_train:
                        yield p
                    elif self.split == "val" and is_val:
                        yield p
                    elif self.split == "test" and is_test:
                        yield p

    def __iter__(self):
        file_iterator = self._get_files_for_split()
        
        for file_path in file_iterator:
            try:
                dataset_name = next((d.name for d in self.dataset_dirs if d in file_path.parents), file_path.parent.name)
            except:
                dataset_name = file_path.parent.name
            
            current_count = self.dataset_counts.get(dataset_name, 0)
            if self.limit > 0 and current_count >= self.limit:
                continue
                
            try:
                if file_path.suffix == ".jsonl":
                    with open(file_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            if not line.strip(): continue
                            try:
                                sample = json.loads(line)
                                if self._yield_sample(sample, dataset_name):
                                    processed = self._process_sample(sample, dataset_name)
                                    if processed:
                                        yield processed
                                    else:
                                        pass # Rejected
                                else:
                                    break
                            except json.JSONDecodeError: continue
                            
                elif file_path.suffix == ".json":
                    is_jsonl = False
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            if isinstance(data, list):
                                for sample in data:
                                    if self._yield_sample(sample, dataset_name):
                                        processed = self._process_sample(sample, dataset_name)
                                        if processed:
                                            yield processed
                                    else:
                                        break
                    except json.JSONDecodeError as e:
                        if "Extra data" in str(e):
                            is_jsonl = True
                    if is_jsonl:
                         with open(file_path, 'r', encoding='utf-8') as f:
                            for line in f:
                                if not line.strip(): continue
                                try:
                                    sample = json.loads(line)
                                    if self._yield_sample(sample, dataset_name):
                                        processed = self._process_sample(sample, dataset_name)
                                        if processed:
                                            yield processed
                                    else:
                                        break
                                except json.JSONDecodeError: continue
            except Exception as e:
                logger.warning(f"Error reading {file_path}: {e}")

    def _yield_sample(self, sample, dataset_name):
        if self.limit > 0 and self.dataset_counts.get(dataset_name, 0) >= self.limit:
            return False
        self.dataset_counts[dataset_name] = self.dataset_counts.get(dataset_name, 0) + 1
        return True

    def _process_sample(self, sample, ds_name):
        # 1. Normalize Schema
        messages = []
        
        # A. Native Messages
        if "messages" in sample:
            messages = sample["messages"]
            
        # B. CoT (prompt/response) or Alpaca (instruction/output)
        elif "prompt" in sample and "response" in sample:
            messages = [
                {"role": "user", "content": sample["prompt"]},
                {"role": "assistant", "content": sample["response"]}
            ]
        elif "instruction" in sample and "output" in sample:
            content = sample["instruction"]
            if sample.get("input"): content += f"\nInput: {sample['input']}"
            messages = [
                {"role": "user", "content": content},
                {"role": "assistant", "content": sample["output"]}
            ]
        elif "query" in sample and "answers" in sample:
            messages = [
                {"role": "user", "content": sample["query"]},
                {"role": "assistant", "content": sample["answers"]}
            ]
            
        # D. Math (problem/answer or question/answer)
        elif ("problem" in sample or "question" in sample) and ("answer" in sample or "solution" in sample):
            q = sample.get("problem") or sample.get("question")
            a = sample.get("answer") or sample.get("solution")
            messages = [
                {"role": "user", "content": q},
                {"role": "assistant", "content": a}
            ]
            
        if not messages: 
            logger.warning(f"[{ds_name}] REJECTED: No matching schema (keys: {list(sample.keys())})")
            return None
        
        modalities = sample.get("modalities", {})
        
        user_msg = next((m for m in messages if m["role"] == "user"), None)
        assistant_msg = next((m for m in messages if m["role"] == "assistant"), None)
        
        if not user_msg or not assistant_msg: 
            logger.warning(f"[{ds_name}] REJECTED: Missing user/assistant roles")
            return None
            
        return {"text": "ok"} # Dummy return for verification

if __name__ == "__main__":
    print("🚀 Starting Debug Stream...")
    ds = OmniDataset(data_path="/mnt/e/data/datasets", split="train", samples_per_dataset=5)
    
    count = 0
    rejected = 0
    
    print("Files found strategies:")
    files = list(itertools.islice(ds._get_files_for_split(), 20))
    print(f"File discovery sample: {files}")
    
    print("\nStreaming Samples:")
    # We essentially reconstruct the iter logic to count rejections
    # Or just iterate ds 
    
    # Let's iterate manually to catch internal logic
    iterator = iter(ds)
    try:
        for i in range(20):
            item = next(iterator)
            print(f"✅ Sample {i}: {item}")
            count += 1
    except StopIteration:
        print("End of stream.")
    except Exception as e:
        print(f"Error: {e}")
        
    print(f"\nYielded {count} samples.")
