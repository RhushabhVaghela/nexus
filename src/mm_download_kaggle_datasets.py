#!/usr/bin/env python3
"""
mm_download_kaggle_datasets.py

Download multimodal datasets from Kaggle as an alternative to HuggingFace.
Supports vision, audio, video, and benchmark datasets.

Requirements:
    pip install kaggle
    Configure: ~/.kaggle/kaggle.json with API credentials

Usage:
    python mm_download_kaggle_datasets.py --modality vision --sample 5
    python mm_download_kaggle_datasets.py --modality all --sample 1000
"""

import os
import sys
import json
import shutil
import logging
import argparse
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

try:
    from kaggle.api.kaggle_api_extended import KaggleApi
    KAGGLE_AVAILABLE = True
except ImportError:
    KAGGLE_AVAILABLE = False
    print("Warning: 'kaggle' library not available. Install with: pip install kaggle")

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

from utils.logging_config import setup_logger, log_header, log_completion

logger = setup_logger(__name__, "logs/mm_download_kaggle.log")


# ═══════════════════════════════════════════════════════════════
# KAGGLE DATASET REGISTRY
# ═══════════════════════════════════════════════════════════════

KAGGLE_DATASETS = {
    # 1. CORE TRAINING DATA (Text/Code)
    "premium_text": {
        "fineweb-edu": {
            "kaggle_id": "benhamner/fineweb-edu-sample-10bt", # Using sample for now, or use official if available
            "description": "FineWeb-Edu: High quality educational content",
            "file_pattern": "*.parquet",
            "text_field": "text",
        },
        "cosmopedia": {
            "kaggle_id": "huggingface/cosmopedia",
            "description": "Cosmopedia: Synthetic textbooks",
            "file_pattern": "*.parquet", 
            "text_field": "text",
        },
        "code_alpaca": {
            "kaggle_id": "haking/code-alpaca-20k",
            "description": "Code instructions",
            "file_pattern": "*.json",
            "text_field": "instruction",
        }
    },
    
    # 2. MULTIMODAL - VISION (Screenshots, Diagrams)
    "vision": {
        "websight_ui": {
            "kaggle_id": "mehmetalicantoglu/figma-ui-design-images",
            "description": "Figma UI Design Images",
            "file_pattern": "*.jpg",
            "text_field": None,  # Generate description from filename
        },
        "app_screenshots": {
            "kaggle_id": "priyamchoksi/50k-android-app-ui-screenshots",
            "description": "50K Android App UI Screenshots",
            "file_pattern": "*.png",
            "text_field": None,
        },
        "diagrams": {
            "kaggle_id": "simranjain17/1000-flowchart-diagrams",
            "description": "Flowchart and Infrastructure Diagrams",
            "file_pattern": "*.png",
            "text_field": None,
        },
    },
    
    # 3. MULTIMODAL - AUDIO (Speech)
    "audio": {
        "librispeech": {
            "kaggle_id": "obenedicto/libri-speech-clean-test",
            "description": "LibriSpeech Clean (Speech-to-Text)",
            "file_pattern": "*.flac",
            "text_field": "transcription",
        },
        "common_voice": {
            "kaggle_id": "mozillaorg/common-voice",
            "description": "Common Voice (Sample)",
            "file_pattern": "*.mp3",
            "text_field": "sentence",
        },
        "speech_commands": {
            "kaggle_id": "neehakurelli/google-speech-commands",
            "description": "Google Speech Commands",
            "file_pattern": "*.wav",
            "text_field": "label",
        }
    },
    
    # 4. MULTIMODAL - VIDEO
    "video": {
        "msr_vtt": {
            "kaggle_id": "evgeniy987/msr-vtt",
            "description": "MSR-VTT Video Captioning",
            "file_pattern": "*.mp4",
            "text_field": "caption",
        },
        "ucf101": {
            "kaggle_id": "pevogam/ucf101",
            "description": "UCF101 Action Recognition",
            "file_pattern": "*.avi",
            "text_field": "label",
        }
    },
    
    # 5. BENCHMARKS (Evaluation)
    "benchmarks": {
        "mmlu": {
            "kaggle_id": "lizhecheng/mmlu-dataset",
            "description": "MMLU (Massive Multitask Language Understanding)",
            "file_pattern": "*.csv",
            "text_field": "question",
        },
        "gsm8k": {
            "kaggle_id": "thedevastator/gsm8k-grade-school-math-word-problems",
            "description": "GSM8K (Grade School Math)",
            "file_pattern": "*.json",
            "text_field": "question",
        },
        "humaneval": {
            "kaggle_id": "alexanderdiaz/humaneval",
            "description": "HumanEval (Code Generation)",
            "file_pattern": "*.jsonl",
            "text_field": "prompt",
        },
        "mbpp": {
            "kaggle_id": "rohanrao/mbpp-dataset",
            "description": "MBPP (Mostly Basic Python Problems)",
            "file_pattern": "*.jsonl", 
            "text_field": "text",
        },
        "truthfulqa": {
            "kaggle_id": "jondurbin/truthful-qa",
            "description": "TruthfulQA",
            "file_pattern": "*.csv",
            "text_field": "Question",
        },
        "arc": {
            "kaggle_id": "abhishek/arc-challenge-and-easy-sets",
            "description": "ARC (AI2 Reasoning Challenge)",
            "file_pattern": "*.csv",
            "text_field": "question",
        }
    },
}


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
# KAGGLE API WRAPPER
# ═══════════════════════════════════════════════════════════════

class KaggleDownloader:
    """Wrapper for Kaggle API operations."""
    
    def __init__(self, download_dir: Path):
        self.download_dir = download_dir
        self.api = None
        
        if KAGGLE_AVAILABLE:
            try:
                self.api = KaggleApi()
                self.api.authenticate()
                logger.info("Kaggle API authenticated successfully")
            except Exception as e:
                logger.error(f"Kaggle API authentication failed: {e}")
                logger.info("Configure ~/.kaggle/kaggle.json with your API credentials")
    
    def download_dataset(self, kaggle_id: str, target_dir: Path) -> bool:
        """Download a Kaggle dataset."""
        if not self.api:
            logger.error("Kaggle API not available")
            return False
        
        target_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            logger.info(f"Downloading {kaggle_id}...")
            self.api.dataset_download_files(
                kaggle_id,
                path=str(target_dir),
                unzip=True,
                quiet=False,
            )
            logger.info(f"✅ Downloaded {kaggle_id} to {target_dir}")
            return True
        except Exception as e:
            logger.error(f"Failed to download {kaggle_id}: {e}")
            return False
    
    def list_files(self, kaggle_id: str) -> List[str]:
        """List files in a Kaggle dataset."""
        if not self.api:
            return []
        
        try:
            files = self.api.dataset_list_files(kaggle_id)
            return [f.name for f in files.files]
        except Exception as e:
            logger.warning(f"Failed to list files for {kaggle_id}: {e}")
            return []


# ═══════════════════════════════════════════════════════════════
# DATASET PROCESSORS
# ═══════════════════════════════════════════════════════════════

def process_text_dataset(
    data_dir: Path,
    output_dir: Path,
    config: Dict,
    sample_limit: int,
    dataset_name: str,
) -> int:
    """Process text/code dataset from Kaggle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find data files
    data_files = []
    for ext in ["*.parquet", "*.json", "*.jsonl", "*.csv", "*.txt"]:
        data_files.extend(data_dir.rglob(ext))
    
    if not data_files:
        logger.warning(f"No data files found in {data_dir}")
        return 0
    
    count = 0
    jsonl_file = output_dir / "data.jsonl"
    text_field = config.get("text_field", "text")
    
    with open(jsonl_file, 'w', encoding='utf-8') as f:
        for data_file in data_files:
            if count >= sample_limit:
                break
            
            try:
                # Handle Parquet (FineWeb-Edu, Cosmopedia)
                if data_file.suffix == ".parquet" and PANDAS_AVAILABLE:
                    df = pd.read_parquet(data_file)
                    for idx, row in df.iterrows():
                        if count >= sample_limit:
                            break
                        
                        text = str(row.get(text_field, ""))
                        if not text or len(text) < 10:
                            continue
                            
                        # Use first 100 chars as 'instruction' if none exists
                        instruction = "Complete the following text:"
                        if dataset_name == "code_alpaca":
                            instruction = row.get("instruction", "Write code for:")
                        
                        sample_id = f"kaggle_text_{dataset_name}_{count:07d}"
                        
                        record = MultimodalSample(
                            id=sample_id,
                            messages=[
                                {"role": "user", "content": instruction},
                                {"role": "assistant", "content": text},
                            ],
                            domain="text",
                            category=f"kaggle_{dataset_name}",
                            modalities={"image": [], "audio": [], "video": []},
                            source=f"Kaggle:{dataset_name}",
                        )
                        
                        f.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
                        count += 1
                
                # Handle JSON/JSONL (Code Alpaca)
                elif data_file.suffix in [".json", ".jsonl"]:
                    with open(data_file) as jf:
                        if data_file.suffix == ".jsonl":
                            iterator = jf
                        else:
                            data = json.load(jf)
                            if isinstance(data, list):
                                iterator = data
                            else:
                                continue
                                
                        for item in iterator:
                            if count >= sample_limit:
                                break
                            
                            if isinstance(item, str):
                                item = json.loads(item)
                                
                            text = item.get(text_field, item.get("output", ""))
                            instruction = item.get("instruction", "Write code for:")
                            
                            if not text:
                                continue
                                
                            sample_id = f"kaggle_text_{dataset_name}_{count:07d}"
                            
                            record = MultimodalSample(
                                id=sample_id,
                                messages=[
                                    {"role": "user", "content": instruction},
                                    {"role": "assistant", "content": text},
                                ],
                                domain="text",
                                category=f"kaggle_{dataset_name}",
                                modalities={"image": [], "audio": [], "video": []},
                                source=f"Kaggle:{dataset_name}",
                            )
                            
                            f.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
                            count += 1
                    
            except Exception as e:
                logger.warning(f"Error processing {data_file}: {e}")
                continue
                
    logger.info(f"✅ Text-{dataset_name}: {count} samples saved to {output_dir}")
    return count

def process_vision_dataset(
    data_dir: Path,
    output_dir: Path,
    config: Dict,
    sample_limit: int,
    dataset_name: str,
) -> int:
    """Process vision dataset from Kaggle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"
    images_dir.mkdir(exist_ok=True)
    
    # Find image files
    pattern = config.get("file_pattern", "*.jpg")
    image_files = list(data_dir.rglob(pattern))
    
    if not image_files:
        # Try other common extensions
        for ext in ["*.png", "*.jpg", "*.jpeg", "*.webp"]:
            image_files = list(data_dir.rglob(ext))
            if image_files:
                break
    
    if not image_files:
        logger.warning(f"No image files found in {data_dir}")
        return 0
    
    logger.info(f"Found {len(image_files)} images in {data_dir}")
    
    count = 0
    jsonl_file = output_dir / "data.jsonl"
    
    with open(jsonl_file, 'w', encoding='utf-8') as f:
        for img_path in tqdm(image_files[:sample_limit], desc=f"Vision-{dataset_name}"):
            try:
                sample_id = f"kaggle_vision_{dataset_name}_{count:07d}"
                
                # Copy image to output
                dest_path = images_dir / f"{sample_id}{img_path.suffix}"
                shutil.copy2(img_path, dest_path)
                
                # Generate description from filename/path
                desc = img_path.stem.replace("_", " ").replace("-", " ")
                
                record = MultimodalSample(
                    id=sample_id,
                    messages=[
                        {"role": "user", "content": "Describe what you see in this image."},
                        {"role": "assistant", "content": f"This image appears to be: {desc}"},
                    ],
                    domain="multimodal_vision",
                    category="kaggle_vision",
                    modalities={
                        "image": [{"path": str(dest_path), "type": "image"}],
                        "audio": [],
                        "video": [],
                    },
                    source=f"Kaggle:{dataset_name}",
                )
                
                f.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
                count += 1
                
            except Exception as e:
                logger.warning(f"Error processing {img_path}: {e}")
                continue
    
    logger.info(f"✅ Vision-{dataset_name}: {count} samples saved to {output_dir}")
    return count


def process_audio_dataset(
    data_dir: Path,
    output_dir: Path,
    config: Dict,
    sample_limit: int,
    dataset_name: str,
) -> int:
    """Process audio dataset from Kaggle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    audio_output_dir = output_dir / "audio"
    audio_output_dir.mkdir(exist_ok=True)
    
    # Find audio files
    audio_files = []
    for ext in ["*.wav", "*.mp3", "*.flac", "*.ogg"]:
        audio_files.extend(data_dir.rglob(ext))
    
    if not audio_files:
        logger.warning(f"No audio files found in {data_dir}")
        return 0
    
    logger.info(f"Found {len(audio_files)} audio files in {data_dir}")
    
    # Look for transcription file
    transcript_file = None
    for tf in ["transcripts.txt", "transcription.txt", "labels.csv", "metadata.csv"]:
        tf_path = data_dir / tf
        if tf_path.exists():
            transcript_file = tf_path
            break
    
    # Load transcripts if available
    transcripts = {}
    if transcript_file and PANDAS_AVAILABLE:
        try:
            if transcript_file.suffix == ".csv":
                df = pd.read_csv(transcript_file)
                if "transcription" in df.columns:
                    transcripts = dict(zip(df.iloc[:, 0], df["transcription"]))
                elif "text" in df.columns:
                    transcripts = dict(zip(df.iloc[:, 0], df["text"]))
            else:
                with open(transcript_file) as f:
                    for line in f:
                        parts = line.strip().split("|")
                        if len(parts) >= 2:
                            transcripts[parts[0]] = parts[1]
        except Exception as e:
            logger.warning(f"Failed to load transcripts: {e}")
    
    count = 0
    jsonl_file = output_dir / "data.jsonl"
    
    with open(jsonl_file, 'w', encoding='utf-8') as f:
        for audio_path in tqdm(audio_files[:sample_limit], desc=f"Audio-{dataset_name}"):
            try:
                sample_id = f"kaggle_audio_{dataset_name}_{count:07d}"
                
                # Copy audio to output
                dest_path = audio_output_dir / f"{sample_id}{audio_path.suffix}"
                shutil.copy2(audio_path, dest_path)
                
                # Get transcript or use folder name as label
                transcript = transcripts.get(audio_path.stem, "")
                if not transcript:
                    # Use parent folder name as label (common for speech commands)
                    transcript = audio_path.parent.name.replace("_", " ")
                
                record = MultimodalSample(
                    id=sample_id,
                    messages=[
                        {"role": "user", "content": "Transcribe this audio: [AUDIO]"},
                        {"role": "assistant", "content": transcript},
                    ],
                    domain="multimodal_audio",
                    category="kaggle_audio",
                    modalities={
                        "image": [],
                        "audio": [{"path": str(dest_path), "type": "speech"}],
                        "video": [],
                    },
                    source=f"Kaggle:{dataset_name}",
                )
                
                f.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
                count += 1
                
            except Exception as e:
                logger.warning(f"Error processing {audio_path}: {e}")
                continue
    
    logger.info(f"✅ Audio-{dataset_name}: {count} samples saved to {output_dir}")
    return count


def process_benchmark_dataset(
    data_dir: Path,
    output_dir: Path,
    config: Dict,
    sample_limit: int,
    dataset_name: str,
) -> int:
    """Process benchmark dataset from Kaggle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find data files
    data_files = []
    for ext in ["*.csv", "*.json", "*.jsonl", "*.parquet"]:
        data_files.extend(data_dir.rglob(ext))
    
    if not data_files:
        logger.warning(f"No data files found in {data_dir}")
        return 0
    
    logger.info(f"Found {len(data_files)} data files in {data_dir}")
    
    count = 0
    jsonl_file = output_dir / "data.jsonl"
    
    with open(jsonl_file, 'w', encoding='utf-8') as f:
        for data_file in data_files:
            if count >= sample_limit:
                break
            
            try:
                # Load the data file
                if data_file.suffix == ".csv" and PANDAS_AVAILABLE:
                    df = pd.read_csv(data_file)
                    for idx, row in df.iterrows():
                        if count >= sample_limit:
                            break
                        
                        # Find question and answer columns
                        question = ""
                        answer = ""
                        options = []
                        
                        for col in row.index:
                            col_lower = col.lower()
                            if "question" in col_lower or "prompt" in col_lower:
                                question = str(row[col])
                            elif "answer" in col_lower or "solution" in col_lower:
                                answer = str(row[col])
                            elif col_lower in ["a", "b", "c", "d"] or "option" in col_lower:
                                if pd.notna(row[col]):
                                    options.append(str(row[col]))
                        
                        if not question:
                            continue
                        
                        sample_id = f"kaggle_bench_{dataset_name}_{count:07d}"
                        
                        user_content = question
                        if options:
                            user_content += "\n\nOptions:\n" + "\n".join([f"{chr(65+i)}) {o}" for i, o in enumerate(options)])
                        
                        record = MultimodalSample(
                            id=sample_id,
                            messages=[
                                {"role": "user", "content": user_content},
                                {"role": "assistant", "content": answer or "A"},
                            ],
                            domain="benchmark",
                            category=f"kaggle_{dataset_name}",
                            modalities={"image": [], "audio": [], "video": []},
                            source=f"Kaggle:{dataset_name}",
                        )
                        
                        f.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
                        count += 1
                
                elif data_file.suffix == ".json":
                    with open(data_file) as jf:
                        data = json.load(jf)
                    
                    if isinstance(data, list):
                        items = data
                    elif isinstance(data, dict):
                        items = list(data.values())[0] if data else []
                    else:
                        continue
                    
                    for item in items:
                        if count >= sample_limit:
                            break
                        
                        if not isinstance(item, dict):
                            continue
                        
                        question = item.get("question", item.get("prompt", ""))
                        answer = item.get("answer", item.get("solution", ""))
                        
                        if not question:
                            continue
                        
                        sample_id = f"kaggle_bench_{dataset_name}_{count:07d}"
                        
                        record = MultimodalSample(
                            id=sample_id,
                            messages=[
                                {"role": "user", "content": str(question)},
                                {"role": "assistant", "content": str(answer)},
                            ],
                            domain="benchmark",
                            category=f"kaggle_{dataset_name}",
                            modalities={"image": [], "audio": [], "video": []},
                            source=f"Kaggle:{dataset_name}",
                        )
                        
                        f.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
                        count += 1
                
                elif data_file.suffix == ".jsonl":
                    with open(data_file) as jf:
                        for line in jf:
                            if count >= sample_limit:
                                break
                            
                            try:
                                item = json.loads(line)
                                question = item.get("prompt", item.get("question", ""))
                                answer = item.get("canonical_solution", item.get("answer", ""))
                                
                                if not question:
                                    continue
                                
                                sample_id = f"kaggle_bench_{dataset_name}_{count:07d}"
                                
                                record = MultimodalSample(
                                    id=sample_id,
                                    messages=[
                                        {"role": "user", "content": str(question)},
                                        {"role": "assistant", "content": str(answer)},
                                    ],
                                    domain="benchmark",
                                    category=f"kaggle_{dataset_name}",
                                    modalities={"image": [], "audio": [], "video": []},
                                    source=f"Kaggle:{dataset_name}",
                                )
                                
                                f.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
                                count += 1
                            except json.JSONDecodeError:
                                continue
                                
            except Exception as e:
                logger.warning(f"Error processing {data_file}: {e}")
                continue
    
    logger.info(f"✅ Benchmark-{dataset_name}: {count} samples saved to {output_dir}")
    return count


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Download multimodal datasets from Kaggle"
    )
    parser.add_argument(
        "--modality",
        type=str,
        choices=["vision", "audio", "video", "benchmarks", "all"],
        default="all",
        help="Which modality to download",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=100,
        help="Number of samples per dataset",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/mnt/e/data/kaggle_multimodal",
        help="Output directory",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Specific dataset to download",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available datasets",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip download, only process existing files",
    )
    args = parser.parse_args()
    
    if args.list:
        print("\n📦 Available Kaggle Datasets:\n")
        for modality, datasets in KAGGLE_DATASETS.items():
            print(f"  {modality.upper()}:")
            for name, config in datasets.items():
                print(f"    - {name}")
                print(f"      Kaggle: {config['kaggle_id']}")
                print(f"      Description: {config['description']}")
            print()
        return
    
    log_header(
        logger,
        "KAGGLE MULTIMODAL DATASET DOWNLOADER",
        {
            "Modality": args.modality,
            "Sample limit": args.sample,
            "Output": args.output_dir,
        },
    )
    
    output_dir = Path(args.output_dir)
    download_dir = output_dir / "downloads"
    download_dir.mkdir(parents=True, exist_ok=True)
    
    downloader = KaggleDownloader(download_dir)
    total_samples = 0
    
    modalities = ["premium_text", "vision", "audio", "video", "benchmarks"] if args.modality == "all" else [args.modality]
    
    for modality in modalities:
        datasets = KAGGLE_DATASETS.get(modality, {})
        
        for name, config in datasets.items():
            if args.dataset and args.dataset.lower() != name.lower():
                continue
            
            kaggle_id = config["kaggle_id"]
            ds_download_dir = download_dir / name
            ds_output_dir = output_dir / modality / name
            
            # Download if not skipped and not already downloaded
            if not args.skip_download:
                if not ds_download_dir.exists() or not list(ds_download_dir.iterdir()):
                    success = downloader.download_dataset(kaggle_id, ds_download_dir)
                    if not success:
                        logger.warning(f"Skipping {name} - download failed")
                        continue
            
            if not ds_download_dir.exists():
                logger.warning(f"Skipping {name} - data directory not found")
                continue
            
            # Process the dataset
            if modality == "premium_text":
                count = process_text_dataset(
                    ds_download_dir, ds_output_dir, config, args.sample, name
                )
            elif modality == "vision":
                count = process_vision_dataset(
                    ds_download_dir, ds_output_dir, config, args.sample, name
                )
            elif modality == "audio":
                count = process_audio_dataset(
                    ds_download_dir, ds_output_dir, config, args.sample, name
                )
            elif modality == "video":
                count = process_audio_dataset(  # Video uses same processor for now
                    ds_download_dir, ds_output_dir, config, args.sample, name
                )
            elif modality == "benchmarks":
                count = process_benchmark_dataset(
                    ds_download_dir, ds_output_dir, config, args.sample, name
                )
            else:
                count = 0
            
            total_samples += count
    
    print(f"\n{'='*60}")
    print(f"🎉 Kaggle Dataset Download Complete")
    print(f"   Total samples: {total_samples}")
    print(f"   Modalities: {len(modalities)}")
    print(f"   Output: {args.output_dir}")
    print(f"{'='*60}")
    
    print(f"\n🎉 Total samples processed: {total_samples}")
    print(f"📁 Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()
