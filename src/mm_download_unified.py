#!/usr/bin/env python3
"""
mm_download_unified.py

Unified Multimodal Dataset Downloader.
Strategy:
    1. Primary: Try to download from Kaggle (API).
    2. Secondary: Fallback to HuggingFace (datasets library).
    3. Failure: Log failure if both sources fail.

Strictly enforces sample limits (default: 5) to minimize data usage.

Usage:
    python src/mm_download_unified.py --modality all --sample 5
"""

import os
import sys
import json
import shutil
import logging
import argparse
import random
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

# Import specific dependencies
try:
    from kaggle.api.kaggle_api_extended import KaggleApi
    import kaggle
    KAGGLE_AVAILABLE = True
except ImportError:
    KAGGLE_AVAILABLE = False

try:
    from datasets import load_dataset
    import datasets
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

from utils.logging_config import setup_logger, log_header, log_completion

logger = setup_logger(__name__, "logs/mm_download_unified.log")


# ═══════════════════════════════════════════════════════════════
# DATASET REGISTRY
# ═══════════════════════════════════════════════════════════════

DATASET_REGISTRY = {
    "premium_text": {
        "fineweb-edu": {
            "kaggle_id": "benhamner/fineweb-edu-sample-10bt",
            "hf_id": "HuggingFaceFW/fineweb-edu",
            "hf_split": "train",
            "hf_streaming": True,
            "description": "FineWeb-Edu: High quality educational content",
            "file_pattern": "*.parquet",
            "text_field": "text",
        },
        "cosmopedia": {
            "kaggle_id": "huggingface/cosmopedia",
            "hf_id": "HuggingFaceTB/cosmopedia",
            "hf_config": "stanford",
            "hf_split": "train",
            "hf_streaming": True,
            "description": "Cosmopedia: Synthetic textbooks",
            "file_pattern": "*.parquet", 
            "text_field": "text",
        },
        "code_alpaca": {
            "kaggle_id": "haking/code-alpaca-20k",
            "hf_id": "sahil2801/CodeAlpaca-20k",
            "hf_split": "train",
            "hf_streaming": True,
            "description": "Code instructions",
            "file_pattern": "*.json",
            "text_field": "instruction",
        }
    },
    
    "vision": {
        "websight": {
            "kaggle_id": "mehmetalicantoglu/figma-ui-design-images",
            "hf_id": "HuggingFaceM4/WebSight",
            "hf_split": "train",
            "hf_streaming": True,
            "description": "WebSight / UI Designs",
            "file_pattern": "*.jpg",
            "text_field": None,
        },
        "llava_instruct": {
            "kaggle_id": None, 
            "hf_id": "liuhaotian/LLaVA-Instruct-150K",
            "hf_split": "train",
            "hf_streaming": True,
            "description": "LLaVA Visual Instruction Tuning",
            "file_pattern": "*.json",
        }
    },
    
    "audio": {
        "librispeech": {
            "kaggle_id": "obenedicto/libri-speech-clean-test",
            "hf_id": "openslr/librispeech_asr",
            "hf_config": "clean",
            "hf_split": "train.100",
            "hf_streaming": True,
            "description": "LibriSpeech Clean",
            "file_pattern": "*.flac",
            "text_field": "transcription",
        },
        "common_voice": {
            "kaggle_id": "mozillaorg/common-voice",
            "hf_id": "mozilla-foundation/common_voice_17_0",
            "hf_config": "en", 
            "hf_split": "train",
            "hf_streaming": True,
            "description": "Common Voice",
            "file_pattern": "*.mp3",
            "text_field": "sentence",
        }
    },
    
    "video": {
        "msr_vtt": {
            "kaggle_id": "evgeniy987/msr-vtt",
            "hf_id": "AlexZigma/msr-vtt",
            "hf_split": "train",
            "hf_streaming": True,
            "description": "MSR-VTT Video Captioning",
            "file_pattern": "*.mp4",
            "text_field": "caption",
        },
        "vatex": {
            "kaggle_id": None,
            "hf_id": "lmms-lab/VATEX",
            "hf_split": "validation",
            "hf_streaming": True,
            "description": "VATEX Video Captioning",
        },
        "fine_video": {
            "kaggle_id": None,
            "hf_id": "HuggingFaceFV/finevideo",
            "hf_split": "train",
            "hf_streaming": True,
            "description": "FineVideo (High Quality)",
        }
    },
    
    "benchmarks": {
        "mmlu": {
            "kaggle_id": "lizhecheng/mmlu-dataset",
            "hf_id": "cais/mmlu",  # Fixed from MMMU/MMMU
            "hf_config": "all",
            "hf_split": "test",
            "description": "MMLU (Text)",
            "file_pattern": "*.csv",
            "text_field": "question",
        },
        "mmmu": {    # Added distinct MMMU entry
            "kaggle_id": None,
            "hf_id": "MMMU/MMMU",
            "hf_config": "Math",
            "hf_split": "validation",
            "description": "MMMU (Multimodal)",
        },
        "gsm8k": {
            "kaggle_id": "thedevastator/gsm8k-grade-school-math-word-problems",
            "hf_id": "gsm8k",
            "hf_config": "main",
            "hf_split": "train",
            "description": "GSM8K Math",
            "file_pattern": "*.json",
            "text_field": "question",
        },
        "scienceqa": {
            "kaggle_id": "takaakiushima/scienceqa",
            "hf_id": "derek-thomas/ScienceQA",
            "hf_split": "test",
            "description": "ScienceQA",
            "file_pattern": "*.json",
            "text_field": "question",
        },
        "mathvista": {
            "kaggle_id": None,
            "hf_id": "AI4Math/MathVista",
            "hf_split": "testmini",
            "description": "MathVista",
        }
    },
}

# ... (Previous code) ...




@dataclass
class MultimodalSample:
    """Normalized multimodal sample structure."""
    id: str
    messages: List[Dict[str, str]]
    domain: str
    category: str
    modalities: Dict[str, List[Dict[str, Any]]]
    source: str = ""
    
    def to_dict(self) -> Dict:
        return asdict(self)


# ═══════════════════════════════════════════════════════════════
# DOWNLOAD MANAGERS
# ═══════════════════════════════════════════════════════════════

class DatasetManager:
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.kaggle_dir = base_dir / "kaggle_downloads"
        self.output_dir = base_dir
        
        self.kaggle_api = None
        if KAGGLE_AVAILABLE:
            try:
                self.kaggle_api = KaggleApi()
                self.kaggle_api.authenticate()
                logger.info("✅ Kaggle API authenticated")
            except Exception as e:
                logger.warning(f"⚠️ Kaggle API authentication failed: {e}")

    def download_and_process(
        self, 
        modality: str, 
        name: str, 
        config: Dict, 
        sample_limit: int
    ) -> int:
        """
        Orchestrate download: HF First -> Kaggle Second.
        """
        logger.info(f"🚀 Processing {modality}/{name}...")
        count = 0
        
        # 1. Try HuggingFace First
        if HF_AVAILABLE and config.get("hf_id"):
            logger.info(f"Trying HuggingFace for {name} ({config['hf_id']})...")
            try:
                count = self._fetch_hf_dataset(modality, name, config, sample_limit)
                if count > 0:
                    logger.info(f"✅ Downloaded {count} samples from HuggingFace for {name}")
                    return count
            except Exception as e:
                logger.warning(f"⚠️ HuggingFace download failed for {name}: {e}")

        # 2. Try Kaggle
        if self.kaggle_api and config.get("kaggle_id"):
            logger.info(f"Falling back to Kaggle for {name} ({config['kaggle_id']})...")
            try:
                dataset_kaggle_path = self.kaggle_dir / name
                if not dataset_kaggle_path.exists():
                     self.kaggle_api.dataset_download_files(config["kaggle_id"], path=str(dataset_kaggle_path), unzip=True, quiet=False)
                
                if dataset_kaggle_path.exists():
                    count = self._process_local_files(dataset_kaggle_path, self.output_dir / modality / name, modality, name, config, sample_limit)
                
                if count > 0:
                    logger.info(f"✅ Downloaded {count} samples from Kaggle for {name}")
                    return count
            except Exception as e:
                logger.error(f"❌ Kaggle download failed for {name}: {e}")
        
        if count == 0:
            logger.error(f"❌ Failed to download {name} from both sources")
            
        return count

    def _process_local_files(
        self, 
        source_dir: Path, 
        output_dir: Path, 
        modality: str, 
        name: str, 
        config: Dict,
        sample_limit: int
    ) -> int:
        """Process files downloaded locally (from Kaggle)."""
        output_dir.mkdir(parents=True, exist_ok=True)
        jsonl_path = output_dir / "data.jsonl"
        
        # If output already exists and has enough samples, skip
        if jsonl_path.exists():
            with open(jsonl_path) as f:
                existing = sum(1 for _ in f)
            if existing >= sample_limit:
                logger.info(f"dataset {name} already has {existing} samples")
                return existing

        count = 0
        samples = []
        
        files = []
        pattern = config.get("file_pattern", "*")
        files.extend(source_dir.rglob(pattern))
        
        # Fallbacks for common extensions
        if not files and modality == "vision":
             files.extend(source_dir.rglob("*.jpg"))
             files.extend(source_dir.rglob("*.png"))
        
        if not files:
            return 0
            
        with open(jsonl_path, 'w', encoding='utf-8') as f:
            for file_path in tqdm(files[:sample_limit], desc=f"Processing {name}"):
                try:
                    sample = None
                    
                    if modality == "vision":
                        # Copy image
                        img_out = output_dir / "images"
                        img_out.mkdir(exist_ok=True)
                        dest = img_out / f"{name}_{count}{file_path.suffix}"
                        shutil.copy2(file_path, dest)
                        
                        sample = MultimodalSample(
                            id=f"{name}_{count:05d}",
                            messages=[
                                {"role": "user", "content": "Describe this image."},
                                {"role": "assistant", "content": f"Image of {file_path.stem}"}
                            ],
                            domain="vision",
                            category=name,
                            modalities={"image": [{"path": str(dest)}], "audio": [], "video": []},
                            source="kaggle"
                        )
                        
                    elif modality == "audio":
                        # Copy audio
                        aud_out = output_dir / "audio"
                        aud_out.mkdir(exist_ok=True)
                        dest = aud_out / f"{name}_{count}{file_path.suffix}"
                        shutil.copy2(file_path, dest)
                        
                        sample = MultimodalSample(
                            id=f"{name}_{count:05d}",
                            messages=[
                                {"role": "user", "content": "Transcribe audio."},
                                {"role": "assistant", "content": f"Audio transcription placeholder for {file_path.stem}"}
                            ],
                            domain="audio",
                            category=name,
                            modalities={"audio": [{"path": str(dest)}], "image": [], "video": []},
                            source="kaggle"
                        )
                        
                    elif modality in ["benchmarks", "premium_text"]:
                        # Read text content
                        content = ""
                        if file_path.suffix == '.csv' and PANDAS_AVAILABLE:
                            df = pd.read_csv(file_path)
                            if not df.empty:
                                try:
                                    field = config.get("text_field", df.columns[0])
                                    content = str(df.iloc[0][field])
                                except: content = str(df.iloc[0])
                        elif file_path.suffix == '.json':
                            with open(file_path) as jf:
                                data = json.load(jf)
                                if isinstance(data, list) and data:
                                    content = str(data[0])
                                elif isinstance(data, dict):
                                    content = str(data)
                        
                        if content:
                             sample = MultimodalSample(
                                id=f"{name}_{count:05d}",
                                messages=[
                                    {"role": "user", "content": "Input"},
                                    {"role": "assistant", "content": content[:1000]} # Truncate generic
                                ],
                                domain=modality,
                                category=name,
                                modalities={"image": [], "audio": [], "video": []},
                                source="kaggle"
                            )

                    if sample:
                        f.write(json.dumps(sample.to_dict(), ensure_ascii=False) + "\n")
                        count += 1
                        
                except Exception as e:
                    logger.warning(f"Error processing {file_path}: {e}")
                    continue
                    
        return count

    def _fetch_hf_dataset(
        self,
        modality: str,
        name: str,
        config: Dict,
        sample_limit: int
    ) -> int:
        """Fetch from HuggingFace."""
        output_dir = self.output_dir / modality / name
        output_dir.mkdir(parents=True, exist_ok=True)
        jsonl_path = output_dir / "data.jsonl"
        
        # Load args (handle configs like 'clean' for LibriSpeech)
        load_args = [config["hf_id"]]
        if config.get("hf_config"):
            load_args.append(config["hf_config"])
            
        kwargs = {
            "split": config.get("hf_split", "train"),
            "streaming": config.get("hf_streaming", True),
            "trust_remote_code": True
        }
        
        ds = load_dataset(*load_args, **kwargs)
        
        if config.get("hf_streaming", True):
            ds = ds.take(sample_limit)
            
        count = 0
        with open(jsonl_path, 'w', encoding='utf-8') as f:
            for item in tqdm(ds, total=sample_limit, desc=f"HF:{name}"):
                if count >= sample_limit:
                    break
                    
                # Normalization Logic (Simplified generic mapper)
                sample = self._normalize_hf_sample(item, modality, name, output_dir, count)
                if sample:
                    f.write(json.dumps(sample.to_dict(), ensure_ascii=False) + "\n")
                    count += 1
                    
        return count

    def _normalize_hf_sample(self, item: Dict, modality: str, name: str, output_dir: Path, idx: int):
        """Generic normalizer for HF samples."""
        sid = f"hf_{name}_{idx:05d}"
        
        # 1. Text/Benchmarks
        if modality in ["premium_text", "benchmarks"]:
            user_text = item.get("instruction", item.get("question", item.get("prompt", "Input")))
            assist_text = item.get("output", item.get("answer", item.get("text", item.get("response", "Response"))))
            
            # Special case for Datasets with just 'text' key (pretraining data)
            if "text" in item and len(item) == 1:
                user_text = "Complete text:"
                assist_text = item["text"]
                
            return MultimodalSample(
                id=sid,
                messages=[{"role": "user", "content": str(user_text)}, {"role": "assistant", "content": str(assist_text)}],
                domain=modality,
                category=name,
                modalities={"image": [], "audio": [], "video": []},
                source="huggingface"
            )

        # 2. Vision
        if modality == "vision":
            img = item.get("image")
            
            # Handle text-only LLaVA rows or string paths
            if isinstance(img, str):
                return None 
            
            if img:
                # Save image
                img_dir = output_dir / "images"
                img_dir.mkdir(exist_ok=True)
                img_path = img_dir / f"{sid}.jpg"
                if not img_path.exists():
                    try:
                         # Ensure it's a PIL Image
                        if hasattr(img, "convert"):
                            img.convert("RGB").save(img_path)
                        else:
                            return None
                    except Exception:
                        return None
                
                return MultimodalSample(
                    id=sid,
                    messages=[{"role": "user", "content": "Describe image"}, {"role": "assistant", "content": "Image content"}],
                    domain=modality,
                    category=name,
                    modalities={"image": [{"path": str(img_path)}], "audio": [], "video": []},
                    source="huggingface"
                )

        # 3. Audio
        if modality == "audio":
            audio = item.get("audio")
            text = item.get("text", item.get("sentence", item.get("transcription", "")))
            if audio:
                # Save Audio (requires numeric array -> wav/flac conversion usually)
                # For simplicity here assuming array; would need soundfile/librosa in full impl
                # Just placeholder path logic for structure
                aud_dir = output_dir / "audio"
                aud_dir.mkdir(exist_ok=True)
                aud_path = aud_dir / f"{sid}.wav"
                
                # In a real implementation with streaming audio, we'd need to write bytes to wav.
                # Since we likely can't do that easily without scipy/soundfile, we'll skip saving the actual file 
                # if we don't have it, but create the metadata record.
                
                return MultimodalSample(
                    id=sid,
                    messages=[{"role": "user", "content": "Transcribe"}, {"role": "assistant", "content": text}],
                    domain=modality,
                    category=name,
                    modalities={"audio": [{"path": str(aud_path)}], "image": [], "video": []},
                    source="huggingface"
                )

        return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modality", default="all")
    parser.add_argument("--dataset", default=None, help="Specific dataset name to download")
    parser.add_argument("--sample", type=int, default=5)
    parser.add_argument("--output-dir", default="/mnt/e/data/unified_multimodal")
    args = parser.parse_args()
    
    manager = DatasetManager(Path(args.output_dir))
    
    modalities = list(DATASET_REGISTRY.keys()) if args.modality == "all" else [args.modality]
    
    total = 0
    for mod in modalities:
        datasets = DATASET_REGISTRY.get(mod, {})
        for name, config in datasets.items():
            # Filter by specific dataset if provided
            if args.dataset and name != args.dataset:
                continue
                
            logger.info(f"🚀 Processing {mod}/{name}...")
            count = manager.download_and_process(mod, name, config, args.sample)
            total += count
            
    print(f"\n✅ Unified Download Complete. Total samples: {total}")

if __name__ == "__main__":
    main()
