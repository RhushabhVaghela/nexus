#!/usr/bin/env python3
"""
process_manual_datasets.py

Process manually downloaded datasets from /mnt/e/data/downloaded
and normalize them to the MultimodalSample format.

Usage:
    python src/process_manual_datasets.py --dataset mathvista --sample 5
    python src/process_manual_datasets.py --dataset all --sample 10
"""

import os
import sys
import json
import csv
import zipfile
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from tqdm import tqdm
import argparse
import logging

sys.path.insert(0, str(Path(__file__).parent))
from utils.logging_config import setup_logger

logger = setup_logger(__name__, "logs/process_manual_datasets.log")


@dataclass
class MultimodalSample:
    """Unified schema for all multimodal samples."""
    id: str
    messages: List[Dict[str, str]]
    domain: str
    category: str
    modalities: Dict[str, List[Dict[str, str]]]
    source: str

    def to_dict(self):
        return asdict(self)


# Dataset Registry
MANUAL_DATASETS = {
    "mathvista": {
        "path": "/mnt/e/data/downloaded/AI4Math_MathVista",
        "modality": "benchmarks",
        "processor": "mathvista"
    },
    "mathverse": {
        "path": "/mnt/e/data/downloaded/AI4Math_MathVerse",
        "modality": "benchmarks",
        "processor": "mathverse"
    },
    "ineqmath": {
        "path": "/mnt/e/data/downloaded/AI4Math_IneqMath",
        "modality": "benchmarks",
        "processor": "ineqmath"
    },
    "mmlu": {
        "path": "/mnt/e/data/downloaded/cais_mmlu",
        "modality": "benchmarks",
        "processor": "mmlu"
    },
    "common_voice": {
        "path": "/mnt/e/data/downloaded/Mozilla_Common-Voice",
        "modality": "audio",
        "processor": "common_voice"
    },
    "msr_vtt": {
        "path": "/mnt/e/data/downloaded/VLM2Vec_MSR-VTT",
        "modality": "video",
        "processor": "msr_vtt"
    },
    "vatex": {
        "path": "/mnt/e/data/downloaded/qingy2024_VaTeX",
        "modality": "video",
        "processor": "vatex"
    },
    "llava_onevision": {
        "path": "/mnt/e/data/downloaded/mvp-lab_LLaVA-OneVision-1.5-RL-Data",
        "modality": "vision",
        "processor": "llava_onevision"
    },
    "stackoverflow_quality": {
        "path": "/mnt/e/data/downloaded/imoore_60k-stack-overflow-questions-with-quality-rateing",
        "modality": "premium_text",
        "processor": "stackoverflow_quality"
    },
    "stackoverflow_questions": {
        "path": "/mnt/e/data/downloaded/pacovaldez_stackoverflow-questions",
        "modality": "premium_text",
        "processor": "stackoverflow_questions"
    },
    "codegenerate3": {
        "path": "/mnt/e/data/downloaded/samiyasamiya_codegenrate3",
        "modality": "premium_text",
        "processor": "codegenerate3"
    }
}


class ManualDatasetProcessor:
    def __init__(self, output_base: Path = Path("/mnt/e/data/unified_multimodal")):
        self.output_base = output_base
        self.output_base.mkdir(parents=True, exist_ok=True)

    def process_mathvista(self, dataset_path: Path, sample_limit: int = 0) -> int:
        """Process MathVista dataset."""
        logger.info("Processing MathVista dataset...")
        
        # Check if images are extracted
        images_dir = dataset_path / "data"
        images_zip = dataset_path / "images.zip"
        
        if not images_dir.exists() and images_zip.exists():
            logger.info("Extracting images.zip...")
            with zipfile.ZipFile(images_zip, 'r') as zip_ref:
                zip_ref.extractall(dataset_path)
        
        # Load annotations
        annot_file = dataset_path / "annot_testmini.json"
        if not annot_file.exists():
            logger.error(f"Annotation file not found: {annot_file}")
            return 0
        
        with open(annot_file, 'r', encoding='utf-8') as f:
            annotations = json.load(f)
        
        # Prepare output
        output_dir = self.output_base / "benchmarks" / "mathvista"
        output_dir.mkdir(parents=True, exist_ok=True)
        jsonl_path = output_dir / "data.jsonl"
        
        count = 0
        with open(jsonl_path, 'w', encoding='utf-8') as out_f:
            for idx, (key, value) in enumerate(tqdm(annotations.items(), desc="MathVista")):
                if sample_limit > 0 and count >= sample_limit:
                    break
                
                question = value.get("question", "Solve the problem")
                answer = value.get("answer_latex", value.get("answer", "Unknown"))
                
                image_path = None
                if "image" in value:
                    image_path = str(dataset_path / "data" / value["image"])
                
                sample = MultimodalSample(
                    id=f"mathvista_{key}",
                    messages=[
                        {"role": "user", "content": question},
                        {"role": "assistant", "content": str(answer)}
                    ],
                    domain="benchmarks",
                    category="mathvista",
                    modalities={
                        "image": [{"path": image_path}] if image_path else [],
                        "audio": [],
                        "video": []
                    },
                    source="manual"
                )
                
                out_f.write(json.dumps(sample.to_dict(), ensure_ascii=False) + "\n")
                count += 1
        
        logger.info(f"✅ Processed {count} MathVista samples")
        return count

    def process_mathverse(self, dataset_path: Path, sample_limit: int = 0) -> int:
        """Process MathVerse dataset."""
        logger.info("Processing MathVerse dataset...")
        
        json_file = dataset_path / "testmini.json"
        if not json_file.exists():
            logger.error(f"JSON file not found: {json_file}")
            return 0
        
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        output_dir = self.output_base / "benchmarks" / "mathverse"
        output_dir.mkdir(parents=True, exist_ok=True)
        jsonl_path = output_dir / "data.jsonl"
        
        count = 0
        with open(jsonl_path, 'w', encoding='utf-8') as out_f:
            for item in tqdm(data, desc="MathVerse"):
                if sample_limit > 0 and count >= sample_limit:
                    break
                
                question = item.get("question", "")
                answer = item.get("answer", "")
                image = item.get("image", "")
                
                if not question:
                    continue
                
                image_path = None
                if image:
                    image_path = str(dataset_path / image)
                
                sample = MultimodalSample(
                    id=f"mathverse_{count:05d}",
                    messages=[
                        {"role": "user", "content": question},
                        {"role": "assistant", "content": str(answer)}
                    ],
                    domain="benchmarks",
                    category="mathverse",
                    modalities={
                        "image": [{"path": image_path}] if image_path else [],
                        "audio": [],
                        "video": []
                    },
                    source="manual"
                )
                
                out_f.write(json.dumps(sample.to_dict(), ensure_ascii=False) + "\n")
                count += 1
        
        logger.info(f"✅ Processed {count} MathVerse samples")
        return count

    def process_ineqmath(self, dataset_path: Path, sample_limit: int = 0) -> int:
        """Process IneqMath dataset."""
        logger.info("Processing IneqMath dataset...")
        
        json_dir = dataset_path / "json"
        train_file = json_dir / "train.json"
        
        if not train_file.exists():
            logger.error(f"Train file not found: {train_file}")
            return 0
        
        with open(train_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        output_dir = self.output_base / "benchmarks" / "ineqmath"
        output_dir.mkdir(parents=True, exist_ok=True)
        jsonl_path = output_dir / "data.jsonl"
        
        count = 0
        with open(jsonl_path, 'w', encoding='utf-8') as out_f:
            for item in tqdm(data, desc="IneqMath"):
                if sample_limit > 0 and count >= sample_limit:
                    break
                
                problem = item.get("problem", "")
                solution = item.get("solution", "")
                
                if not problem:
                    continue
                
                sample = MultimodalSample(
                    id=f"ineqmath_{count:05d}",
                    messages=[
                        {"role": "user", "content": problem},
                        {"role": "assistant", "content": str(solution)}
                    ],
                    domain="benchmarks",
                    category="ineqmath",
                    modalities={
                        "image": [],
                        "audio": [],
                        "video": []
                    },
                    source="manual"
                )
                
                out_f.write(json.dumps(sample.to_dict(), ensure_ascii=False) + "\n")
                count += 1
        
        logger.info(f"✅ Processed {count} IneqMath samples")
        return count

    def process_mmlu(self, dataset_path: Path, sample_limit: int = 0) -> int:
        """Process MMLU dataset."""
        logger.info("Processing MMLU dataset...")
        
        try:
            import pandas as pd
        except ImportError:
            logger.error("pandas is required for MMLU processing")
            return 0
        
        subject_dirs = [d for d in dataset_path.iterdir() 
                       if d.is_dir() and d.name not in ['.cache', 'auxiliary_train', 'all']]
        
        if not subject_dirs:
            logger.error("No subject directories found")
            return 0
        
        output_dir = self.output_base / "benchmarks" / "mmlu"
        output_dir.mkdir(parents=True, exist_ok=True)
        jsonl_path = output_dir / "data.jsonl"
        
        count = 0
        with open(jsonl_path, 'w', encoding='utf-8') as out_f:
            for subject_dir in tqdm(subject_dirs[:5], desc="MMLU"):
                dev_file = subject_dir / "dev-00000-of-00001.parquet"
                if not dev_file.exists():
                    continue
                
                df = pd.read_parquet(dev_file)
                
                for idx, row in df.iterrows():
                    if sample_limit > 0 and count >= sample_limit:
                        break
                    
                    question = row.get("question", "")
                    choices = row.get("choices", [])
                    answer = row.get("answer", "")
                    
                    if not question:
                        continue
                    
                    question_text = f"{question}\nChoices:\n"
                    for i, choice in enumerate(choices):
                        question_text += f"{chr(65+i)}: {choice}\n"
                    
                    sample = MultimodalSample(
                        id=f"mmlu_{subject_dir.name}_{count:05d}",
                        messages=[
                            {"role": "user", "content": question_text},
                            {"role": "assistant", "content": str(answer)}
                        ],
                        domain="benchmarks",
                        category="mmlu",
                        modalities={
                            "image": [],
                            "audio": [],
                            "video": []
                        },
                        source="manual"
                    )
                    
                    out_f.write(json.dumps(sample.to_dict(), ensure_ascii=False) + "\n")
                    count += 1
                    
                if sample_limit > 0 and count >= sample_limit:
                    break
        
        logger.info(f"✅ Processed {count} MMLU samples")
        return count

    def process_common_voice(self, dataset_path: Path, sample_limit: int = 0) -> int:
        """Process Common Voice dataset."""
        logger.info("Processing Common Voice dataset...")
        
        csv_file = dataset_path / "cv-valid-train.csv"
        audio_dir = dataset_path / "cv-valid-train"
        
        if not csv_file.exists():
            logger.error(f"CSV file not found: {csv_file}")
            return 0
        
        output_dir = self.output_base / "audio" / "common_voice"
        output_dir.mkdir(parents=True, exist_ok=True)
        jsonl_path = output_dir / "data.jsonl"
        
        count = 0
        with open(jsonl_path, 'w', encoding='utf-8') as out_f:
            with open(csv_file, 'r', encoding='utf-8') as csv_f:
                reader = csv.DictReader(csv_f)
                for row in tqdm(reader, desc="CommonVoice"):
                    if sample_limit > 0 and count >= sample_limit:
                        break
                    
                    filename = row.get("filename", "")
                    text = row.get("text", "")
                    
                    if not filename or not text:
                        continue
                    
                    audio_path = audio_dir / filename
                    
                    sample = MultimodalSample(
                        id=f"common_voice_{count:05d}",
                        messages=[
                            {"role": "user", "content": "Transcribe the audio"},
                            {"role": "assistant", "content": text}
                        ],
                        domain="audio",
                        category="common_voice",
                        modalities={
                            "image": [],
                            "audio": [{"path": str(audio_path)}],
                            "video": []
                        },
                        source="manual"
                    )
                    
                    out_f.write(json.dumps(sample.to_dict(), ensure_ascii=False) + "\n")
                    count += 1
        
        logger.info(f"✅ Processed {count} Common Voice samples")
        return count

    def process_msr_vtt(self, dataset_path: Path, sample_limit: int = 0) -> int:
        """Process MSR-VTT dataset."""
        logger.info("Processing MSR-VTT dataset...")
        
        json_file = dataset_path / "msrvtt_train_7k.json"
        if not json_file.exists():
            logger.error(f"JSON file not found: {json_file}")
            return 0
        
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        output_dir = self.output_base / "video" / "msr_vtt"
        output_dir.mkdir(parents=True, exist_ok=True)
        jsonl_path = output_dir / "data.jsonl"
        
        count = 0
        with open(jsonl_path, 'w', encoding='utf-8') as out_f:
            annotations = data if isinstance(data, list) else data.get("annotations", [])
            for idx, item in enumerate(tqdm(annotations, desc="MSR-VTT")):
                if sample_limit > 0 and count >= sample_limit:
                    break
                
                video_id = item.get("video_id", "")
                captions = item.get("caption", [])
                
                if not video_id or not captions:
                    continue
                
                caption = captions[0] if isinstance(captions, list) and captions else str(captions)
                video_path = dataset_path / "raw_videos" / f"{video_id}.mp4"
                
                sample = MultimodalSample(
                    id=f"msr_vtt_{video_id}",
                    messages=[
                        {"role": "user", "content": "Describe the video"},
                        {"role": "assistant", "content": caption}
                    ],
                    domain="video",
                    category="msr_vtt",
                    modalities={
                        "image": [],
                        "audio": [],
                        "video": [{"path": str(video_path)}]
                    },
                    source="manual"
                )
                
                out_f.write(json.dumps(sample.to_dict(), ensure_ascii=False) + "\n")
                count += 1
        
        logger.info(f"✅ Processed {count} MSR-VTT samples")
        return count

    def process_vatex(self, dataset_path: Path, sample_limit: int = 0) -> int:
        """Process VaTeX dataset."""
        logger.info("Processing VaTeX dataset...")
        
        json_file = dataset_path / "vatex_training_v1.0.json"
        if not json_file.exists():
            logger.error(f"JSON file not found: {json_file}")
            return 0
        
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        output_dir = self.output_base / "video" / "vatex"
        output_dir.mkdir(parents=True, exist_ok=True)
        jsonl_path = output_dir / "data.jsonl"
        
        count = 0
        with open(jsonl_path, 'w', encoding='utf-8') as out_f:
            for idx, item in enumerate(tqdm(data, desc="VaTeX")):
                if sample_limit > 0 and count >= sample_limit:
                    break
                
                video_id = item.get("videoID", "")
                captions = item.get("enCap", [])
                
                if not video_id or not captions:
                    continue
                
                caption = captions[0] if isinstance(captions, list) else str(captions)
                video_path = dataset_path / "vatex-dataset" / f"{video_id}.mp4"
                
                sample = MultimodalSample(
                    id=f"vatex_{video_id}",
                    messages=[
                        {"role": "user", "content": "Describe the video"},
                        {"role": "assistant", "content": caption}
                    ],
                    domain="video",
                    category="vatex",
                    modalities={
                        "image": [],
                        "audio": [],
                        "video": [{"path": str(video_path)}]
                    },
                    source="manual"
                )
                
                out_f.write(json.dumps(sample.to_dict(), ensure_ascii=False) + "\n")
                count += 1
        
        logger.info(f"✅ Processed {count} VaTeX samples")
        return count

    def process_stackoverflow_quality(self, dataset_path: Path, sample_limit: int = 0) -> int:
        """Process StackOverflow Quality dataset."""
        logger.info("Processing StackOverflow Quality dataset...")
        
        csv_file = dataset_path / "train.csv"
        if not csv_file.exists():
            logger.error(f"CSV file not found: {csv_file}")
            return 0
        
        output_dir = self.output_base / "premium_text" / "stackoverflow_quality"
        output_dir.mkdir(parents=True, exist_ok=True)
        jsonl_path = output_dir / "data.jsonl"
        
        count = 0
        with open(jsonl_path, 'w', encoding='utf-8') as out_f:
            with open(csv_file, 'r', encoding='utf-8') as csv_f:
                reader = csv.DictReader(csv_f)
                for row in tqdm(reader, desc="StackOverflow"):
                    if sample_limit > 0 and count >= sample_limit:
                        break
                    
                    title = row.get("Title", "")
                    body = row.get("Body", "")
                    tags = row.get("Tags", "")
                    
                    if not title or not body:
                        continue
                    
                    question = f"Title: {title}\nTags: {tags}\n\nQuestion: {body}"
                    
                    sample = MultimodalSample(
                        id=f"stackoverflow_{count:05d}",
                        messages=[
                            {"role": "user", "content": "Help me with this coding question"},
                            {"role": "assistant", "content": question}
                        ],
                        domain="premium_text",
                        category="stackoverflow_quality",
                        modalities={
                            "image": [],
                            "audio": [],
                            "video": []
                        },
                        source="manual"
                    )
                    
                    out_f.write(json.dumps(sample.to_dict(), ensure_ascii=False) + "\n")
                    count += 1
        
        logger.info(f"✅ Processed {count} StackOverflow samples")
        return count

    def process_llava_onevision(self, dataset_path: Path, sample_limit: int = 0) -> int:
        """Process LLaVA-OneVision dataset."""
        logger.info("Processing LLaVA-OneVision dataset...")
        
        try:
            import pandas as pd
        except ImportError:
            logger.error("pandas is required for LLaVA-OneVision processing")
            return 0
        
        stage_dir = dataset_path / "stage1-normal"
        if not stage_dir.exists():
            logger.error(f"Stage directory not found: {stage_dir}")
            return 0
        
        subdirs = [d for d in stage_dir.iterdir() if d.is_dir()]
        
        output_dir = self.output_base / "vision" / "llava_onevision"
        output_dir.mkdir(parents=True, exist_ok=True)
        jsonl_path = output_dir / "data.jsonl"
        
        count = 0
        with open(jsonl_path, 'w', encoding='utf-8') as out_f:
            for subdir in tqdm(subdirs, desc="LLaVA-OneVision"):
                parquet_files = list(subdir.glob("*.parquet"))
                for parquet_file in parquet_files:
                    if sample_limit > 0 and count >= sample_limit:
                        break
                    
                    df = pd.read_parquet(parquet_file)
                    
                    for idx, row in df.iterrows():
                        if sample_limit > 0 and count >= sample_limit:
                            break
                        
                        problem = row.get("problem", "")
                        answer = row.get("answer", "")
                        
                        if not problem:
                            continue
                        
                        # Convert answer array to string if needed
                        if hasattr(answer, '__iter__') and not isinstance(answer, str):
                            answer = str(answer[0]) if len(answer) > 0 else str(answer)
                        
                        sample = MultimodalSample(
                            id=f"llava_onevision_{count:05d}",
                            messages=[
                                {"role": "user", "content": problem},
                                {"role": "assistant", "content": str(answer)}
                            ],
                            domain="vision",
                            category="llava_onevision",
                            modalities={
                                "image": [],  # Image paths would be extracted from <image> tags
                                "audio": [],
                                "video": []
                            },
                            source="manual"
                        )
                        
                        out_f.write(json.dumps(sample.to_dict(), ensure_ascii=False) + "\n")
                        count += 1
                        
                    if sample_limit > 0 and count >= sample_limit:
                        break
                        
                if sample_limit > 0 and count >= sample_limit:
                    break
        
        logger.info(f"✅ Processed {count} LLaVA-OneVision samples")
        return count

    def process_stackoverflow_questions(self, dataset_path: Path, sample_limit: int = 0) -> int:
        """Process StackOverflow Questions dataset."""
        logger.info("Processing StackOverflow Questions dataset...")
        
        try:
            import pandas as pd
        except ImportError:
            logger.error("pandas is required for StackOverflow Questions processing")
            return 0
        
        data_dir = dataset_path / "data" / "post_questions_train"
        if not data_dir.exists():
            logger.error(f"Data directory not found: {data_dir}")
            return 0
        
        parquet_files = list(data_dir.glob("*.parquet"))
        
        output_dir = self.output_base / "premium_text" / "stackoverflow_questions"
        output_dir.mkdir(parents=True, exist_ok=True)
        jsonl_path = output_dir / "data.jsonl"
        
        count = 0
        with open(jsonl_path, 'w', encoding='utf-8') as out_f:
            for parquet_file in tqdm(parquet_files, desc="StackOverflow-Q"):
                if sample_limit > 0 and count >= sample_limit:
                    break
                
                df = pd.read_parquet(parquet_file)
                
                for idx, row in df.iterrows():
                    if sample_limit > 0 and count >= sample_limit:
                        break
                    
                    title = row.get("Title", "")
                    body = row.get("Body", "")
                    
                    if not title or not body:
                        continue
                    
                    question = f"Title: {title}\n\n{body}"
                    
                    sample = MultimodalSample(
                        id=f"stackoverflow_q_{count:05d}",
                        messages=[
                            {"role": "user", "content": "Help with this coding question"},
                            {"role": "assistant", "content": question}
                        ],
                        domain="premium_text",
                        category="stackoverflow_questions",
                        modalities={
                            "image": [],
                            "audio": [],
                            "video": []
                        },
                        source="manual"
                    )
                    
                    out_f.write(json.dumps(sample.to_dict(), ensure_ascii=False) + "\n")
                    count += 1
                    
                if sample_limit > 0 and count >= sample_limit:
                    break
        
        logger.info(f"✅ Processed {count} StackOverflow Questions samples")
        return count

    def process_dataset(self, dataset_name: str, sample_limit: int = 0) -> int:
        """Route to appropriate processor."""
        if dataset_name not in MANUAL_DATASETS:
            logger.error(f"Unknown dataset: {dataset_name}")
            return 0
        
        config = MANUAL_DATASETS[dataset_name]
        dataset_path = Path(config["path"])
        processor_name = config["processor"]
        
        if not dataset_path.exists():
            logger.error(f"Dataset path not found: {dataset_path}")
            return 0
        
        # Route to processor
        processor_map = {
            "mathvista": self.process_mathvista,
            "mathverse": self.process_mathverse,
            "ineqmath": self.process_ineqmath,
            "mmlu": self.process_mmlu,
            "common_voice": self.process_common_voice,
            "msr_vtt": self.process_msr_vtt,
            "vatex": self.process_vatex,
            "stackoverflow_quality": self.process_stackoverflow_quality,
            "llava_onevision": self.process_llava_onevision,
            "stackoverflow_questions": self.process_stackoverflow_questions,
        }
        
        processor_func = processor_map.get(processor_name)
        if processor_func:
            return processor_func(dataset_path, sample_limit)
        else:
            logger.warning(f"Processor not implemented: {processor_name}")
            return 0


def main():
    parser = argparse.ArgumentParser(description="Process manually downloaded datasets")
    parser.add_argument("--dataset", default="all", help="Dataset to process or 'all'")
    parser.add_argument("--sample", type=int, default=5, help="Sample limit (0 = no limit)")
    parser.add_argument("--output-dir", default="/mnt/e/data/unified_multimodal", help="Output directory")
    args = parser.parse_args()
    
    processor = ManualDatasetProcessor(Path(args.output_dir))
    
    if args.dataset == "all":
        total = 0
        for dataset_name in MANUAL_DATASETS.keys():
            logger.info(f"\n{'='*60}\nProcessing {dataset_name}...\n{'='*60}")
            count = processor.process_dataset(dataset_name, args.sample)
            total += count
        logger.info(f"\n✅ Total samples processed: {total}")
    else:
        count = processor.process_dataset(args.dataset, args.sample)
        logger.info(f"✅ Processed {count} samples from {args.dataset}")


if __name__ == "__main__":
    main()
