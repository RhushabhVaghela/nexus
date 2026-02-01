import os
import json
import glob
import argparse
from typing import List, Dict, Any, Generator, Optional, Callable
import csv
import zlib
import pandas as pd # Optional, but good for parquet
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)

# Progress bar integration
try:
    from ..utils.progress import DataLoadingProgress, progress_iter
except ImportError:
    try:
        from src.utils.progress import DataLoadingProgress, progress_iter
    except ImportError:
        DataLoadingProgress = None
        progress_iter = None


# Default progress bar settings
USE_PROGRESS_BARS = os.environ.get('NEXUS_USE_PROGRESS', '1') == '1'


class MemorizationFilter:
    """
    Data filtering pipeline for memorization risk reduction.
    Based on paper 2601.15394 recommendations.
    """
    
    def __init__(self, 
                 entropy_threshold: float = 0.4,
                 risk_threshold: float = 0.5,
                 classifier_path: Optional[str] = None):
        """
        Initialize the memorization filter.
        
        Args:
            entropy_threshold: Minimum zlib entropy for a sample (lower = more compressible = higher memorization risk)
            risk_threshold: Maximum acceptable memorization risk score from classifier
            classifier_path: Path to pre-trained memorization classifier
        """
        self.entropy_threshold = entropy_threshold
        self.risk_threshold = risk_threshold
        self.classifier_path = classifier_path
        self.classifier = None
        
        if classifier_path and os.path.exists(classifier_path):
            self._load_classifier()
    
    def _load_classifier(self):
        """Load the pre-trained memorization classifier."""
        try:
            from .auditor import MemorizationClassifier
            self.classifier = MemorizationClassifier(self.classifier_path)
            logger.info(f"Loaded memorization classifier from {self.classifier_path}")
        except ImportError:
            logger.warning("Could not load MemorizationClassifier. Using entropy-only filtering.")
    
    @staticmethod
    def calculate_entropy(text: str) -> float:
        """Calculate normalized zlib entropy."""
        if not text:
            return 0.0
        try:
            compressed = zlib.compress(text.encode('utf-8'))
            return len(compressed) / len(text.encode('utf-8'))
        except Exception:
            return 0.0
    
    def should_filter(self, sample: Dict[str, Any], 
                     teacher_model=None, 
                     baseline_model=None,
                     tokenizer=None) -> bool:
        """
        Determine if a sample should be filtered due to high memorization risk.
        
        Args:
            sample: The data sample in standard format
            teacher_model: Optional teacher model for classifier-based filtering
            baseline_model: Optional baseline model for classifier-based filtering
            tokenizer: Optional tokenizer for classifier-based filtering
            
        Returns:
            True if sample should be filtered (high risk), False otherwise
        """
        # Extract text from sample
        if "messages" in sample:
            text = " ".join([m.get("content", "") for m in sample["messages"]])
        elif "text" in sample:
            text = sample["text"]
        else:
            # Try to extract any text content
            text = str(sample)
        
        # 1. Entropy-based filtering (always applied)
        entropy = self.calculate_entropy(text)
        if entropy < self.entropy_threshold:
            logger.debug(f"Filtering sample due to low entropy: {entropy:.4f} < {self.entropy_threshold}")
            return True
        
        # 2. Classifier-based filtering (if available)
        if self.classifier is not None and self.classifier.is_trained:
            if teacher_model is not None and tokenizer is not None:
                try:
                    from .auditor import MemorizationAuditor
                    auditor = MemorizationAuditor(tokenizer)
                    auditor.classifier = self.classifier
                    result = auditor.predict_memorization_risk(
                        text, teacher_model, baseline_model
                    )
                    
                    if result["risk_score"] > self.risk_threshold:
                        logger.debug(f"Filtering sample due to high risk: {result['risk_score']:.4f} > {self.risk_threshold}")
                        return True
                except Exception as e:
                    logger.warning(f"Classifier prediction failed: {e}. Using entropy-only filtering.")
        
        return False
    
    def filter_dataset(self, samples: Generator[Dict, None, None],
                      teacher_model=None,
                      baseline_model=None,
                      tokenizer=None) -> Generator[Dict, None, None]:
        """
        Filter a dataset stream, yielding only low-risk samples.
        
        Args:
            samples: Generator of data samples
            teacher_model: Optional teacher model
            baseline_model: Optional baseline model
            tokenizer: Optional tokenizer
            
        Yields:
            Samples that pass the filter criteria
        """
        filtered_count = 0
        total_count = 0
        
        for sample in samples:
            total_count += 1
            
            if self.should_filter(sample, teacher_model, baseline_model, tokenizer):
                filtered_count += 1
                continue
            
            yield sample
        
        if total_count > 0:
            reduction_rate = filtered_count / total_count
            logger.info(f"Filtered {filtered_count}/{total_count} samples ({reduction_rate:.2%} reduction)")


class UniversalDataLoader:
    """
    Unified Data Loader for Nexus Pipeline.
    Auto-detects dataset schema and normalizes to standard conversation format.
    Standard Format: {"messages": [{"role": "user", "content": ...}, {"role": "assistant", "content": ...}]}
    """
    
    def __init__(self, data_root: str = "/mnt/e/data/datasets", 
                 filter_memorization_risk: bool = False,
                 entropy_threshold: float = 0.4,
                 risk_threshold: float = 0.5,
                 classifier_path: Optional[str] = None):
        """
        Initialize the UniversalDataLoader.
        
        Args:
            data_root: Root directory for datasets
            filter_memorization_risk: Whether to enable memorization risk filtering
            entropy_threshold: Minimum entropy threshold for filtering
            risk_threshold: Maximum risk score threshold for classifier filtering
            classifier_path: Path to pre-trained memorization classifier
        """
        # Ensure path is absolute if possible
        self.data_root = os.path.abspath(data_root)
        print(f"[DataLoader] Initialized at {self.data_root}")
        self.supported_extensions = ['*.jsonl', '*.json', '*.parquet', '*.csv', '*.tsv']
        
        # Initialize memorization filter if enabled
        self.filter_memorization_risk = filter_memorization_risk
        self.memorization_filter = None
        
        if filter_memorization_risk:
            self.memorization_filter = MemorizationFilter(
                entropy_threshold=entropy_threshold,
                risk_threshold=risk_threshold,
                classifier_path=classifier_path
            )
            logger.info(f"Memorization filtering enabled: entropy_threshold={entropy_threshold}, risk_threshold={risk_threshold}")

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

    def load_dataset(self, dataset_name: str, split: str = "train", limit: int = None,
                     min_entropy: float = None,
                     teacher_model=None,
                     baseline_model=None,
                     tokenizer=None,
                     show_progress: bool = True) -> Generator[Dict, None, None]:
        """
        Stream normalized data from a dataset.
        Args:
            dataset_name: Name of the dataset folder (e.g., 'general/google_smol')
            split: 'train', 'test', 'validation' (matched loosely in filenames)
            limit: int: Max samples to yield
            min_entropy: float: Minimum zlib entropy required to yield a sample (e.g. 0.4)
            teacher_model: Teacher model for memorization filtering (if enabled)
            baseline_model: Baseline model for memorization filtering
            tokenizer: Tokenizer for memorization filtering
            show_progress: bool: Whether to show a progress bar
        """
        self.min_entropy = min_entropy
        full_path = os.path.join(self.data_root, dataset_name)
        
        # Initialize progress tracking
        progress = None
        if show_progress and USE_PROGRESS_BARS and DataLoadingProgress:
            progress = DataLoadingProgress(
                total_samples=limit,
                desc=f"Loading {dataset_name}",
                enabled=True
            )
            progress.start()
        
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

        # Get base generator based on file type
        if full_path.endswith(".jsonl") or full_path.endswith(".json"):
            base_generator = self._yield_json(full_path, limit)
        elif full_path.endswith(".csv"):
            base_generator = self._yield_csv(full_path, limit)
        elif full_path.endswith(".tsv"):
            base_generator = self._yield_csv(full_path, delimiter="\t", limit=limit)
        elif full_path.endswith(".parquet"):
            base_generator = self._yield_parquet(full_path, limit)
        else:
            print(f"[DataLoader] Unsupported format: {full_path}")
            return
        
        # Apply memorization filtering if enabled
        if self.filter_memorization_risk and self.memorization_filter is not None:
            return self.memorization_filter.filter_dataset(
                base_generator, teacher_model, baseline_model, tokenizer
            )
        
        return base_generator

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
                        if self.min_entropy:
                            combined_text = " ".join([m["content"] for m in normalized["messages"]])
                            entropy = self.calculate_entropy(combined_text)
                            if entropy < self.min_entropy:
                                continue
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
                    if self.min_entropy:
                        combined_text = " ".join([m["content"] for m in normalized["messages"]])
                        entropy = self.calculate_entropy(combined_text)
                        if entropy < self.min_entropy:
                            continue
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
                    if self.min_entropy:
                        combined_text = " ".join([m["content"] for m in normalized["messages"]])
                        entropy = self.calculate_entropy(combined_text)
                        if entropy < self.min_entropy:
                            continue
                    yield normalized
                    count += 1
        except ImportError:
            print("[DataLoader] Error: pandas or pyarrow not installed.")
            return
        except Exception as e:
            print(f"[DataLoader] Error reading parquet: {e}")
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

    @staticmethod
    def calculate_entropy(text: str) -> float:
        """Calculates normalized zlib entropy: (compressed length / original length)"""
        if not text: return 0.0
        try:
            compressed = zlib.compress(text.encode('utf-8'))
            return len(compressed) / len(text)
        except Exception:
            return 0.0


def main():
    """CLI entry point for the data loader with filtering options."""
    parser = argparse.ArgumentParser(
        description="Universal Data Loader with Memorization Filtering"
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="/mnt/e/data/datasets",
        help="Root directory for datasets"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name to load"
    )
    parser.add_argument(
        "--filter-memorization-risk",
        action="store_true",
        help="Enable memorization risk filtering (99.8%% reduction in memorized examples)"
    )
    parser.add_argument(
        "--entropy-threshold",
        type=float,
        default=0.4,
        help="Minimum entropy threshold for filtering (default: 0.4)"
    )
    parser.add_argument(
        "--risk-threshold",
        type=float,
        default=0.5,
        help="Maximum risk score threshold for classifier filtering (default: 0.5)"
    )
    parser.add_argument(
        "--classifier-path",
        type=str,
        default=None,
        help="Path to pre-trained memorization classifier"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of samples to load"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available datasets"
    )
    
    args = parser.parse_args()
    
    # Initialize loader
    loader = UniversalDataLoader(
        data_root=args.data_root,
        filter_memorization_risk=args.filter_memorization_risk,
        entropy_threshold=args.entropy_threshold,
        risk_threshold=args.risk_threshold,
        classifier_path=args.classifier_path
    )
    
    if args.list:
        datasets = loader.list_available_datasets()
        print("Available datasets:")
        for ds in datasets:
            print(f"  - {ds}")
        return
    
    # Load dataset
    print(f"Loading dataset: {args.dataset}")
    samples = loader.load_dataset(args.dataset, limit=args.limit)
    
    count = 0
    for sample in samples:
        count += 1
        if count <= 3:  # Show first 3 samples
            print(f"\nSample {count}:")
            print(json.dumps(sample, indent=2))
    
    print(f"\nTotal samples loaded: {count}")


if __name__ == "__main__":
    main()
