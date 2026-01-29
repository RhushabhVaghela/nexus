import os
import json
import glob
from typing import List, Dict, Any, Generator, Optional
import csv
import pandas as pd # Optional, but good for parquet
from tqdm import tqdm

class UniversalDataLoader:
    """
    Unified Data Loader for Nexus Pipeline.
    Auto-detects dataset schema and normalizes to standard conversation format.
    Standard Format: {"messages": [{"role": "user", "content": ...}, {"role": "assistant", "content": ...}]}
    """
    
    def __init__(self, data_root: str = "/mnt/e/data/datasets"):
        # Ensure path is absolute if possible
        self.data_root = os.path.abspath(data_root)
        print(f"[DataLoader] Initialized at {self.data_root}")
        self.supported_extensions = ['*.jsonl', '*.json', '*.parquet', '*.csv', '*.tsv']

    def list_available_datasets(self) -> List[str]:
        """Scans the first two levels of data_root for datasets."""
        datasets = []
        if not os.path.exists(self.data_root):
            return []
            
        # Level 1: e.g. /mnt/e/data/datasets/reasoning
        for category in os.listdir(self.data_root):
            cat_path = os.path.join(self.data_root, category)
            if not os.path.isdir(cat_path): continue
            
            # Level 2: e.g. /mnt/e/data/datasets/reasoning/AI4Math_IneqMath
            for d in os.listdir(cat_path):
                full_d = os.path.join(cat_path, d)
                if not os.path.isdir(full_d): continue
                
                rel_d = os.path.join(category, d)
                # Check for files
                has_data = any(glob.glob(os.path.join(full_d, ext)) for ext in self.supported_extensions)
                if has_data:
                    datasets.append(rel_d.replace("\\", "/"))
        return sorted(datasets)

    def load_dataset(self, dataset_name: str, split: str = "train", limit: int = None) -> Generator[Dict, None, None]:
        """
        Stream normalized data from a dataset.
        Args:
            dataset_name: Name of the dataset folder (e.g., 'general/google_smol')
            split: 'train', 'test', 'validation' (matched loosely in filenames)
            limit: Max samples to yield
        """
        full_path = os.path.join(self.data_root, dataset_name)
        
        # Heuristic 1: If it's a directory, look for common data files
        if os.path.isdir(full_path):
            candidates = []
            for ext in [".jsonl", ".json", ".csv", ".tsv", ".parquet"]:
                matches = glob.glob(os.path.join(full_path, f"*{ext}"))
                candidates.extend(matches)
            
            if not candidates:
                # Look one level up (common in structure files shown)
                parent_dir = os.path.dirname(full_path)
                matches = glob.glob(os.path.join(parent_dir, f"*.csv")) + glob.glob(os.path.join(parent_dir, f"*.tsv"))
                candidates.extend(matches)

            if candidates:
                full_path = candidates[0] # Pick first match
                print(f"[DataLoader] Directory detected. Using candidate: {full_path}")
            else:
                print(f"[DataLoader] Warning: No data files found in {full_path}")
                return

        if full_path.endswith(".jsonl") or full_path.endswith(".json"):
            return self._yield_json(full_path, limit)
        elif full_path.endswith(".csv"):
            return self._yield_csv(full_path, limit)
        elif full_path.endswith(".tsv"):
            return self._yield_csv(full_path, delimiter="\t", limit=limit)
        elif full_path.endswith(".parquet"):
            return self._yield_parquet(full_path, limit)
        else:
            print(f"[DataLoader] Unsupported format: {full_path}")
            return []

    def _yield_json(self, file_path, limit):
        count = 0
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if limit and count >= limit: break
                line = line.strip()
                if not line: continue
                try:
                    raw_sample = json.loads(line)
                    normalized = self._normalize(raw_sample, file_path)
                    if normalized:
                        yield normalized
                        count += 1
                except json.JSONDecodeError:
                    continue

    def _yield_csv(self, file_path, limit, delimiter=","):
        count = 0
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter=delimiter)
            for row in reader:
                if limit and count >= limit: break
                normalized = self._normalize(row, file_path)
                if normalized:
                    yield normalized
                    count += 1

    def _yield_parquet(self, file_path, limit):
        # Requires pandas and pyarrow/fastparquet
        try:
            import pandas as pd
            df = pd.read_parquet(file_path)
            count = 0
            for _, row in df.iterrows():
                if limit and count >= limit: break
                normalized = self._normalize(row.to_dict(), file_path)
                if normalized:
                    yield normalized
                    count += 1
        except ImportError:
            print("[DataLoader] Error: pandas or pyarrow not installed.")
            return

    def _normalize(self, sample: Dict, source_path: str) -> Optional[Dict]:
        """
        Detect schema and convert to Standard Format.
        """
        # 1. Standard Messages Format (Already correct)
        if "messages" in sample:
            return sample

        # 2. Translation (Google Smol: src, trgs, sl, tl)
        if "src" in sample and "trgs" in sample:
            src_lang = sample.get("sl", "Input")
            tgt_lang = sample.get("tl", "Target")
            user_content = f"Translate from {src_lang} to {tgt_lang}:\n{sample['src']}"
            assistant_content = sample['trgs'][0] if isinstance(sample['trgs'], list) else sample['trgs']
            return {
                "messages": [
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": assistant_content}
                ]
            }

        # 3. Reasoning / Math (Problem/Solution or Question/Answer)
        if "problem" in sample and "solution" in sample:
            return {
                "messages": [
                    {"role": "user", "content": sample["problem"]},
                    {"role": "assistant", "content": sample["solution"]}
                ]
            }
        
        if "question" in sample and "answer" in sample:
            return {
                "messages": [
                    {"role": "user", "content": sample["question"]},
                    {"role": "assistant", "content": sample["answer"]}
                ]
            }

        # 4. Code (contents) / Instruction (instruction)
        if "contents" in sample:
            return {
                "messages": [
                    {"role": "user", "content": "Complete the following code:"},
                    {"role": "assistant", "content": sample["contents"]}
                ]
            }
            
        if "instruction" in sample and "output" in sample:
            input_text = sample.get("input", "")
            content = f"{sample['instruction']}\n{input_text}".strip()
            return {
                "messages": [
                    {"role": "user", "content": content},
                    {"role": "assistant", "content": sample["output"]}
                ]
            }

        # 5. Audio / Multimodal (e.g. Mozilla Common Voice / Speech Commands)
        if "sentence" in sample or "audio" in sample:
            content = sample.get("sentence", "Analyze this audio.")
            audio_path = sample.get("audio", {}).get("path") if isinstance(sample.get("audio"), dict) else sample.get("audio")
            res = {
                "messages": [
                    {"role": "user", "content": content},
                    {"role": "assistant", "content": sample.get("label", "Detected")}
                ]
            }
            if audio_path: res["audio"] = [audio_path]
            return res

        # 6. Vision (LLaVA / VQAs)
        if "image" in sample or "image_path" in sample:
            img = sample.get("image_path") or sample.get("image")
            # If image is a dict (HF format), extract path
            img_path = img.get("path") if isinstance(img, dict) else img
            
            user_msg = sample.get("question") or sample.get("instruction") or "What is in this image?"
            assistant_msg = sample.get("answer") or sample.get("output") or ""
            
            res = {
                "messages": [
                    {"role": "user", "content": user_msg},
                    {"role": "assistant", "content": assistant_msg}
                ]
            }
            if img_path: res["images"] = [img_path]
            return res

        # 7. Ultimate Fallback: Universal Heuristic Sanitizer
        try:
            from nexus_core.data.sanitizer import UniversalSanitizer
            sanitizer = UniversalSanitizer()
            clean_text = sanitizer.sanitize(sample)
            if clean_text and len(clean_text) > 5:
                return {
                    "messages": [
                        {"role": "user", "content": clean_text}
                    ]
                }
        except ImportError:
            pass

        return None

if __name__ == "__main__":
    # Test Driver
    loader = UniversalDataLoader()
    print("Testing Google Smol Loader...")
    gen = loader.load_dataset("general/google_smol", limit=2)
    for vid, sample in enumerate(gen):
        print(f"Sample {vid}: {json.dumps(sample, indent=2)}")
