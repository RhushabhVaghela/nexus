#!/usr/bin/env python3
"""
mm_download_video_datasets.py

Download and process real video understanding datasets from HuggingFace.

Datasets supported:
- FineVideo (HuggingFaceM4/FineVideo): 43K videos, 3425 hours
- Video-MME (lmms-lab/Video-MME): Video understanding benchmark
- ActivityNet (Efficient-Large-Model/ActivityNet-QA): Activity recognition
- Panda-70M (LanguageBind/Panda-70M): Large-scale video-caption pairs
- WebVid (TempoFunk/webvid-10M): 10M video-text pairs
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Iterator
from dataclasses import dataclass

try:
    from datasets import load_dataset, IterableDataset
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

sys.path.insert(0, str(Path(__file__).parent))
from utils.logging_config import setup_logger, log_header, log_completion

logger = setup_logger(__name__, "logs/mm_download_video.log")


# ═══════════════════════════════════════════════════════════════
# DATASET CONFIGURATIONS - REAL HUGGINGFACE DATASETS
# ═══════════════════════════════════════════════════════════════

VIDEO_DATASETS = {
    "finevideo": {
        "hf_path": "HuggingFaceM4/FineVideo",
        "split": "train",
        "description": "43K videos with rich metadata and scene annotations",
        "video_key": "video",
        "text_keys": ["description", "caption"],
        "size_gb": 150,
        "priority": "high",
    },
    "video_mme": {
        "hf_path": "lmms-lab/Video-MME",
        "split": "test",
        "description": "Video understanding benchmark with multiple choice QA",
        "video_key": "video",
        "text_keys": ["question", "answer"],
        "size_gb": 20,
        "priority": "high",
    },
    "activitynet_qa": {
        "hf_path": "Efficient-Large-Model/ActivityNet-QA",
        "split": "train",
        "description": "Activity recognition QA dataset",
        "video_key": "video_id",
        "text_keys": ["question", "answer"],
        "size_gb": 30,
        "priority": "medium",
    },
    "panda_70m": {
        "hf_path": "LanguageBind/Panda-70M",
        "split": "train",
        "description": "70M video-caption pairs (streaming recommended)",
        "video_key": "video",
        "text_keys": ["caption"],
        "size_gb": 500,
        "priority": "low",
        "streaming": True,
    },
    "webvid_10m": {
        "hf_path": "TempoFunk/webvid-10M",
        "split": "train",
        "description": "10M video-text pairs from web",
        "video_key": "video",
        "text_keys": ["caption"],
        "size_gb": 200,
        "priority": "low",
        "streaming": True,
    },
    "msrvtt": {
        "hf_path": "AlexZigma/msr-vtt",
        "split": "train",
        "description": "MSR-VTT video captioning dataset",
        "video_key": "video",
        "text_keys": ["caption"],
        "size_gb": 40,
        "priority": "medium",
    },
}

# Video understanding task templates
VIDEO_TASKS = {
    "captioning": {
        "user_templates": [
            "Describe what happens in this video.",
            "Provide a detailed caption for this video.",
            "What is shown in this video?",
        ],
    },
    "qa": {
        "user_templates": [
            "Answer the following question about the video: {question}",
            "Based on the video, {question}",
            "Watch this video and answer: {question}",
        ],
    },
    "summarization": {
        "user_templates": [
            "Summarize the key events in this video.",
            "What are the main points shown in this video?",
            "Create a brief summary of this video content.",
        ],
    },
    "code_understanding": {
        "user_templates": [
            "Explain the code being demonstrated in this video.",
            "What programming concepts are shown in this screencast?",
            "Describe the development workflow in this video.",
        ],
    },
}


@dataclass
class VideoSample:
    """Normalized video sample."""
    id: str
    video_path: str
    text: str
    task_type: str
    source_dataset: str
    metadata: Dict


def normalize_finevideo(sample: Dict, idx: int) -> Optional[VideoSample]:
    """Normalize FineVideo sample."""
    try:
        video_data = sample.get("video", {})
        description = sample.get("description", "") or sample.get("caption", "")
        
        return VideoSample(
            id=f"finevideo_{idx:08d}",
            video_path=video_data.get("path", "") if isinstance(video_data, dict) else str(video_data),
            text=description,
            task_type="captioning",
            source_dataset="FineVideo",
            metadata={
                "scene_annotations": sample.get("scene_annotations", []),
                "duration": sample.get("duration", 0),
            },
        )
    except Exception:
        return None


def normalize_video_mme(sample: Dict, idx: int) -> Optional[VideoSample]:
    """Normalize Video-MME sample."""
    try:
        question = sample.get("question", "")
        answer = sample.get("answer", "")
        
        return VideoSample(
            id=f"video_mme_{idx:08d}",
            video_path=str(sample.get("video", "")),
            text=f"Q: {question}\nA: {answer}",
            task_type="qa",
            source_dataset="Video-MME",
            metadata={
                "options": sample.get("options", []),
                "category": sample.get("category", ""),
            },
        )
    except Exception:
        return None


def normalize_activitynet(sample: Dict, idx: int) -> Optional[VideoSample]:
    """Normalize ActivityNet-QA sample."""
    try:
        return VideoSample(
            id=f"activitynet_{idx:08d}",
            video_path=sample.get("video_id", ""),
            text=f"Q: {sample.get('question', '')}\nA: {sample.get('answer', '')}",
            task_type="qa",
            source_dataset="ActivityNet-QA",
            metadata={
                "activity_type": sample.get("activity", ""),
            },
        )
    except Exception:
        return None


def normalize_webvid(sample: Dict, idx: int) -> Optional[VideoSample]:
    """Normalize WebVid sample."""
    try:
        return VideoSample(
            id=f"webvid_{idx:08d}",
            video_path=str(sample.get("video", "") or sample.get("url", "")),
            text=sample.get("caption", ""),
            task_type="captioning",
            source_dataset="WebVid-10M",
            metadata={},
        )
    except Exception:
        return None


NORMALIZERS = {
    "finevideo": normalize_finevideo,
    "video_mme": normalize_video_mme,
    "activitynet_qa": normalize_activitynet,
    "panda_70m": normalize_webvid,
    "webvid_10m": normalize_webvid,
    "msrvtt": normalize_webvid,
}


def sample_to_messages(vs: VideoSample) -> Dict:
    """Convert VideoSample to OpenAI messages format."""
    import random
    
    task_templates = VIDEO_TASKS.get(vs.task_type, VIDEO_TASKS["captioning"])
    user_template = random.choice(task_templates["user_templates"])
    
    # Format user prompt
    if "{question}" in user_template and "Q:" in vs.text:
        question = vs.text.split("Q:")[1].split("\n")[0].strip() if "Q:" in vs.text else ""
        user_content = user_template.format(question=question)
        assistant_content = vs.text.split("A:")[-1].strip() if "A:" in vs.text else vs.text
    else:
        user_content = user_template
        assistant_content = vs.text
    
    return {
        "id": vs.id,
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ],
        "domain": "multimodal_video",
        "category": f"video_{vs.task_type}",
        "source_dataset": vs.source_dataset,
        "modalities": {
            "image": [],
            "audio": [],
            "video": [
                {
                    "path": vs.video_path,
                    "type": "video_clip",
                    "source": vs.source_dataset,
                }
            ],
        },
        "metadata": vs.metadata,
    }


def download_and_process_dataset(
    dataset_name: str,
    output_dir: Path,
    limit: Optional[int] = None,
    streaming: bool = False,
) -> int:
    """Download and process a single video dataset."""
    if not HF_AVAILABLE:
        logger.error("datasets library not available")
        return 0
    
    config = VIDEO_DATASETS.get(dataset_name)
    if not config:
        logger.error(f"Unknown dataset: {dataset_name}")
        return 0
    
    normalizer = NORMALIZERS.get(dataset_name)
    if not normalizer:
        logger.error(f"No normalizer for: {dataset_name}")
        return 0
    
    logger.info(f"Loading {dataset_name} from {config['hf_path']}...")
    
    use_streaming = streaming or config.get("streaming", False)
    
    try:
        ds = load_dataset(
            config["hf_path"],
            split=config["split"],
            streaming=use_streaming,
            trust_remote_code=True,
        )
    except Exception as e:
        logger.error(f"Failed to load {dataset_name}: {e}")
        return 0
    
    output_path = output_dir / f"{dataset_name}.jsonl"
    total = 0
    
    with open(output_path, "w", encoding="utf-8") as f:
        for idx, sample in enumerate(ds):
            if limit and idx >= limit:
                break
            
            normalized = normalizer(sample, idx)
            if normalized:
                messages_sample = sample_to_messages(normalized)
                f.write(json.dumps(messages_sample, ensure_ascii=False) + "\n")
                total += 1
            
            if (idx + 1) % 1000 == 0:
                logger.info(f"Processed {idx + 1} samples from {dataset_name}")
    
    logger.info(f"Wrote {total} samples to {output_path}")
    return total


def main():
    parser = argparse.ArgumentParser(description="Download video understanding datasets")
    parser.add_argument("--datasets", nargs="+", default=["finevideo", "video_mme"],
                        choices=list(VIDEO_DATASETS.keys()) + ["all"],
                        help="Datasets to download")
    parser.add_argument("--output-dir", type=str,
                        default="/mnt/e/data/multimodal-fullstack-dataset/video",
                        help="Output directory")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit samples per dataset")
    parser.add_argument("--streaming", action="store_true",
                        help="Use streaming mode for large datasets")
    parser.add_argument("--list", action="store_true",
                        help="List available datasets")
    args = parser.parse_args()
    
    if args.list:
        print("\nAvailable Video Datasets:\n")
        for name, config in VIDEO_DATASETS.items():
            print(f"  {name}")
            print(f"    HuggingFace: {config['hf_path']}")
            print(f"    Size: ~{config['size_gb']} GB")
            print(f"    Priority: {config['priority']}")
            print(f"    Description: {config['description']}")
            print()
        return
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    datasets_to_download = list(VIDEO_DATASETS.keys()) if "all" in args.datasets else args.datasets
    
    log_header(
        logger,
        "VIDEO UNDERSTANDING DATASET DOWNLOADER",
        {
            "Datasets": ", ".join(datasets_to_download),
            "Output": str(output_dir),
            "Limit": args.limit or "None",
            "Streaming": args.streaming,
        },
    )
    
    total_samples = 0
    for dataset_name in datasets_to_download:
        count = download_and_process_dataset(
            dataset_name,
            output_dir,
            args.limit,
            args.streaming,
        )
        total_samples += count
    
    log_completion(
        logger,
        "Video Dataset Download",
        {
            "Total samples": total_samples,
            "Datasets": len(datasets_to_download),
            "Output": str(output_dir),
        },
    )


if __name__ == "__main__":
    main()
