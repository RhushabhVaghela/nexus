#!/usr/bin/env python3
"""
mm_download_multimodal_datasets.py

Unified multimodal dataset fetcher with --sample parameter support.
Downloads vision, audio, video, and benchmark datasets for omni-modal training.

Features:
- Streaming mode to avoid massive downloads
- Sample parameter for testing with smaller datasets
- Normalization to OpenAI messages format with modalities field
- Proper handling of image, audio, and video modalities

Usage:
    python mm_download_multimodal_datasets.py --modality vision --sample 1000
    python mm_download_multimodal_datasets.py --modality audio --sample 500
    python mm_download_multimodal_datasets.py --modality all --sample 100
"""

import os
import sys
import json
import logging
import argparse
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False
    print("Warning: 'datasets' library not available. Using mock mode.")

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

from utils.logging_config import setup_logger, log_header

# ═══════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════

DEFAULT_CONFIG = {
    "base_dir": "/mnt/e/data/multimodal",
    "vision": {
        "WebSight": {
            "source": "HuggingFaceM4/WebSight",
            "split": "train",
            "sample": 250000,
            "streaming": True,
        }
    },
    "audio": {
        "Common_Voice_EN": {
            "source": "mozilla-foundation/common_voice_17_0",
            "language": "en",
            "split": "train",
            "sample": 250000,
            "streaming": True,
        },
        "Common_Voice_ES": {
            "source": "mozilla-foundation/common_voice_17_0",
            "language": "es",
            "split": "train",
            "sample": 100000,
            "streaming": True,
        },
    },
    "video": {
        "FineVideo": {
            "source": "HuggingFaceFV/finevideo",
            "split": "train",
            "sample": 10000,
            "streaming": True,
            "extract_frames": 8,
        }
    },
    "benchmarks": {
        "MMMU": {
            "source": "MMMU/MMMU",
            "split": "validation",
            "sample": 9500,
        },
        "MathVista": {
            "source": "AI4Math/MathVista",
            "split": "testmini",
            "sample": 6141,
        },
    }
}

logger = setup_logger(__name__, "logs/mm_download.log")


# ═══════════════════════════════════════════════════════════════
# DATA CLASSES
# ═══════════════════════════════════════════════════════════════

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
# VISION FETCHER
# ═══════════════════════════════════════════════════════════════

def fetch_vision_dataset(
    config: Dict,
    output_dir: Path,
    sample_limit: int,
    dataset_name: str = "WebSight"
) -> int:
    """
    Fetch vision dataset with sample parameter.
    
    Args:
        config: Dataset configuration
        output_dir: Output directory
        sample_limit: Maximum samples to fetch
        dataset_name: Name of the dataset
        
    Returns:
        Number of samples fetched
    """
    logger.info(f"📸 Fetching {dataset_name} with sample={sample_limit}...")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"
    images_dir.mkdir(exist_ok=True)
    
    if not HAS_DATASETS:
        logger.warning("Using mock mode - no real data fetched")
        return _generate_mock_vision_samples(output_dir, sample_limit)
    
    try:
        ds = load_dataset(
            config["source"],
            split=config.get("split", "train"),
            streaming=config.get("streaming", True),
            trust_remote_code=True,
        )
    except Exception as e:
        logger.error(f"Failed to load dataset {config['source']}: {e}")
        logger.info("Falling back to mock mode")
        return _generate_mock_vision_samples(output_dir, sample_limit)
    
    count = 0
    jsonl_file = output_dir / "data.jsonl"
    
    with open(jsonl_file, 'w', encoding='utf-8') as f:
        for sample in tqdm(ds, total=sample_limit, desc=f"Vision-{dataset_name}"):
            if count >= sample_limit:
                break
            
            try:
                # Generate unique ID
                sample_id = f"vision_{dataset_name.lower()}_{count:07d}"
                
                # Save image if present
                image_path = None
                if "image" in sample and sample["image"] is not None:
                    image_path = images_dir / f"{sample_id}.jpg"
                    try:
                        sample["image"].save(str(image_path), quality=85, optimize=True)
                    except Exception as img_err:
                        logger.warning(f"Failed to save image for {sample_id}: {img_err}")
                        image_path = None
                
                # Get text content
                text_content = sample.get("text", "") or sample.get("caption", "") or ""
                if not text_content:
                    continue
                
                # Create normalized sample
                record = MultimodalSample(
                    id=sample_id,
                    messages=[
                        {"role": "user", "content": "Analyze this UI screenshot and describe what you see."},
                        {"role": "assistant", "content": text_content[:2000]},  # Limit length
                    ],
                    domain="multimodal_vision",
                    category="ui_screenshot",
                    modalities={
                        "image": [{"path": str(image_path), "type": "screenshot"}] if image_path else [],
                        "audio": [],
                        "video": [],
                    },
                    source=dataset_name,
                )
                
                f.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
                count += 1
                
            except Exception as e:
                logger.warning(f"Error processing sample {count}: {e}")
                continue
    
    logger.info(f"✅ Vision-{dataset_name}: {count} samples saved to {output_dir}")
    return count


def _generate_mock_vision_samples(output_dir: Path, sample_limit: int) -> int:
    """Generate mock vision samples for testing."""
    jsonl_file = output_dir / "data.jsonl"
    count = min(sample_limit, 100)  # Limit mock samples
    
    with open(jsonl_file, 'w', encoding='utf-8') as f:
        for i in range(count):
            record = MultimodalSample(
                id=f"vision_mock_{i:07d}",
                messages=[
                    {"role": "user", "content": "Analyze this UI screenshot."},
                    {"role": "assistant", "content": f"This is a mock UI description #{i}."},
                ],
                domain="multimodal_vision",
                category="ui_screenshot_mock",
                modalities={"image": [], "audio": [], "video": []},
                source="mock",
            )
            f.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
    
    logger.info(f"Generated {count} mock vision samples")
    return count


# ═══════════════════════════════════════════════════════════════
# AUDIO FETCHER
# ═══════════════════════════════════════════════════════════════

def fetch_audio_dataset(
    config: Dict,
    output_dir: Path,
    sample_limit: int,
    dataset_name: str = "Common_Voice"
) -> int:
    """
    Fetch audio dataset with sample parameter.
    
    Args:
        config: Dataset configuration
        output_dir: Output directory
        sample_limit: Maximum samples to fetch
        dataset_name: Name of the dataset
        
    Returns:
        Number of samples fetched
    """
    language = config.get("language", "en")
    logger.info(f"🎤 Fetching {dataset_name} ({language}) with sample={sample_limit}...")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    audio_dir = output_dir / "audio"
    audio_dir.mkdir(exist_ok=True)
    
    if not HAS_DATASETS:
        logger.warning("Using mock mode - no real data fetched")
        return _generate_mock_audio_samples(output_dir, sample_limit, language)
    
    try:
        ds = load_dataset(
            config["source"],
            language,
            split=config.get("split", "train"),
            streaming=config.get("streaming", True),
            trust_remote_code=True,
        )
    except Exception as e:
        logger.error(f"Failed to load dataset {config['source']}: {e}")
        logger.info("Falling back to mock mode")
        return _generate_mock_audio_samples(output_dir, sample_limit, language)
    
    count = 0
    jsonl_file = output_dir / "data.jsonl"
    
    with open(jsonl_file, 'w', encoding='utf-8') as f:
        for sample in tqdm(ds, total=sample_limit, desc=f"Audio-{language}"):
            if count >= sample_limit:
                break
            
            try:
                sample_id = f"audio_{language}_{count:07d}"
                
                # Get transcript
                transcript = sample.get("sentence", "") or sample.get("text", "")
                if not transcript:
                    continue
                
                # Save audio if present and soundfile is available
                audio_path = None
                if "audio" in sample and sample["audio"] is not None:
                    try:
                        import soundfile as sf
                        import numpy as np
                        
                        audio_data = sample["audio"].get("array")
                        sr = sample["audio"].get("sampling_rate", 16000)
                        
                        if audio_data is not None:
                            audio_path = audio_dir / f"{sample_id}.wav"
                            
                            # Resample to 16kHz if needed
                            if sr != 16000:
                                try:
                                    import librosa
                                    audio_data = librosa.resample(
                                        np.array(audio_data), 
                                        orig_sr=sr, 
                                        target_sr=16000
                                    )
                                    sr = 16000
                                except ImportError:
                                    pass  # Keep original sample rate
                            
                            sf.write(str(audio_path), audio_data, sr)
                            
                    except ImportError:
                        logger.warning("soundfile not available, skipping audio save")
                    except Exception as audio_err:
                        logger.warning(f"Failed to save audio for {sample_id}: {audio_err}")
                        audio_path = None
                
                # Create normalized sample
                record = MultimodalSample(
                    id=sample_id,
                    messages=[
                        {"role": "user", "content": "Transcribe this audio: [AUDIO]"},
                        {"role": "assistant", "content": transcript},
                    ],
                    domain="multimodal_audio",
                    category="speech_transcription",
                    modalities={
                        "image": [],
                        "audio": [{"path": str(audio_path), "type": "speech", "language": language}] if audio_path else [],
                        "video": [],
                    },
                    source=dataset_name,
                )
                
                f.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
                count += 1
                
            except Exception as e:
                logger.warning(f"Error processing audio sample {count}: {e}")
                continue
    
    logger.info(f"✅ Audio-{language}: {count} samples saved to {output_dir}")
    return count


def _generate_mock_audio_samples(output_dir: Path, sample_limit: int, language: str) -> int:
    """Generate mock audio samples for testing."""
    jsonl_file = output_dir / "data.jsonl"
    count = min(sample_limit, 100)
    
    with open(jsonl_file, 'w', encoding='utf-8') as f:
        for i in range(count):
            record = MultimodalSample(
                id=f"audio_{language}_mock_{i:07d}",
                messages=[
                    {"role": "user", "content": "Transcribe this audio: [AUDIO]"},
                    {"role": "assistant", "content": f"This is a mock transcript #{i} in {language}."},
                ],
                domain="multimodal_audio",
                category="speech_mock",
                modalities={"image": [], "audio": [], "video": []},
                source="mock",
            )
            f.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
    
    logger.info(f"Generated {count} mock audio samples for {language}")
    return count


# ═══════════════════════════════════════════════════════════════
# VIDEO FETCHER
# ═══════════════════════════════════════════════════════════════

def fetch_video_dataset(
    config: Dict,
    output_dir: Path,
    sample_limit: int,
    dataset_name: str = "FineVideo"
) -> int:
    """
    Fetch video dataset with keyframe extraction.
    
    Args:
        config: Dataset configuration
        output_dir: Output directory
        sample_limit: Maximum samples to fetch
        dataset_name: Name of the dataset
        
    Returns:
        Number of samples fetched
    """
    logger.info(f"🎬 Fetching {dataset_name} with sample={sample_limit}...")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(exist_ok=True)
    
    if not HAS_DATASETS:
        logger.warning("Using mock mode - no real data fetched")
        return _generate_mock_video_samples(output_dir, sample_limit)
    
    try:
        ds = load_dataset(
            config["source"],
            split=config.get("split", "train"),
            streaming=config.get("streaming", True),
            trust_remote_code=True,
        )
    except Exception as e:
        logger.error(f"Failed to load dataset {config['source']}: {e}")
        logger.info("Falling back to mock mode")
        return _generate_mock_video_samples(output_dir, sample_limit)
    
    count = 0
    jsonl_file = output_dir / "data.jsonl"
    extract_frames = config.get("extract_frames", 8)
    
    with open(jsonl_file, 'w', encoding='utf-8') as f:
        for sample in tqdm(ds, total=sample_limit, desc=f"Video-{dataset_name}"):
            if count >= sample_limit:
                break
            
            try:
                sample_id = f"video_{count:07d}"
                
                # Get description
                description = sample.get("text", "") or sample.get("caption", "") or ""
                if not description:
                    continue
                
                # Extract keyframes if video is present
                frame_paths = []
                if "mp4" in sample or "video" in sample:
                    try:
                        frame_paths = _extract_keyframes(
                            sample, 
                            frames_dir, 
                            sample_id, 
                            extract_frames
                        )
                    except Exception as vid_err:
                        logger.warning(f"Failed to extract frames for {sample_id}: {vid_err}")
                
                # Create normalized sample
                record = MultimodalSample(
                    id=sample_id,
                    messages=[
                        {"role": "user", "content": "Describe what happens in this video."},
                        {"role": "assistant", "content": description[:2000]},
                    ],
                    domain="multimodal_video",
                    category="video_description",
                    modalities={
                        "image": [{"path": p, "type": "video_frame"} for p in frame_paths],
                        "audio": [],
                        "video": [],  # We store extracted frames instead
                    },
                    source=dataset_name,
                )
                
                f.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
                count += 1
                
            except Exception as e:
                logger.warning(f"Error processing video sample {count}: {e}")
                continue
    
    logger.info(f"✅ Video-{dataset_name}: {count} samples saved to {output_dir}")
    return count


def _extract_keyframes(sample: Dict, frames_dir: Path, sample_id: str, num_frames: int) -> List[str]:
    """Extract keyframes from video sample."""
    frame_paths = []
    
    try:
        import cv2
        import tempfile
        
        video_data = sample.get("mp4") or sample.get("video")
        if video_data is None:
            return []
        
        # Write video to temp file
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            if isinstance(video_data, bytes):
                tmp.write(video_data)
            else:
                return []
            tmp_path = tmp.name
        
        try:
            cap = cv2.VideoCapture(tmp_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if total_frames == 0:
                return []
            
            frame_indices = [int(i * total_frames / num_frames) for i in range(num_frames)]
            
            for idx, frame_idx in enumerate(frame_indices):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if ret:
                    frame_path = frames_dir / f"{sample_id}_frame_{idx:02d}.jpg"
                    cv2.imwrite(str(frame_path), frame)
                    frame_paths.append(str(frame_path))
            
            cap.release()
        finally:
            os.unlink(tmp_path)
            
    except ImportError:
        logger.warning("cv2 not available, skipping frame extraction")
    
    return frame_paths


def _generate_mock_video_samples(output_dir: Path, sample_limit: int) -> int:
    """Generate mock video samples for testing."""
    jsonl_file = output_dir / "data.jsonl"
    count = min(sample_limit, 50)
    
    with open(jsonl_file, 'w', encoding='utf-8') as f:
        for i in range(count):
            record = MultimodalSample(
                id=f"video_mock_{i:07d}",
                messages=[
                    {"role": "user", "content": "Describe what happens in this video."},
                    {"role": "assistant", "content": f"This is a mock video description #{i}."},
                ],
                domain="multimodal_video",
                category="video_mock",
                modalities={"image": [], "audio": [], "video": []},
                source="mock",
            )
            f.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
    
    logger.info(f"Generated {count} mock video samples")
    return count


# ═══════════════════════════════════════════════════════════════
# BENCHMARK FETCHER
# ═══════════════════════════════════════════════════════════════

def fetch_benchmark_dataset(
    config: Dict,
    output_dir: Path,
    sample_limit: int,
    dataset_name: str = "MMMU"
) -> int:
    """
    Fetch benchmark dataset for evaluation.
    
    Args:
        config: Dataset configuration
        output_dir: Output directory
        sample_limit: Maximum samples to fetch
        dataset_name: Name of the dataset
        
    Returns:
        Number of samples fetched
    """
    logger.info(f"📊 Fetching benchmark {dataset_name} with sample={sample_limit}...")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"
    images_dir.mkdir(exist_ok=True)
    
    if not HAS_DATASETS:
        logger.warning("Using mock mode - no real data fetched")
        return _generate_mock_benchmark_samples(output_dir, sample_limit, dataset_name)
    
    try:
        # Handle different dataset configurations
        if "config" in config:
            ds = load_dataset(
                config["source"],
                config["config"],
                split=config.get("split", "validation"),
                trust_remote_code=True,
            )
        else:
            ds = load_dataset(
                config["source"],
                split=config.get("split", "validation"),
                trust_remote_code=True,
            )
    except Exception as e:
        logger.error(f"Failed to load benchmark {config['source']}: {e}")
        return _generate_mock_benchmark_samples(output_dir, sample_limit, dataset_name)
    
    count = 0
    jsonl_file = output_dir / "data.jsonl"
    
    with open(jsonl_file, 'w', encoding='utf-8') as f:
        for sample in tqdm(ds, total=sample_limit, desc=f"Benchmark-{dataset_name}"):
            if count >= sample_limit:
                break
            
            try:
                sample_id = f"benchmark_{dataset_name.lower()}_{count:07d}"
                
                # Get question and answer
                question = sample.get("question", "") or sample.get("problem", "")
                answer = sample.get("answer", "") or sample.get("solution", "")
                options = sample.get("options", []) or sample.get("choices", [])
                
                if not question:
                    continue
                
                # Build user content with options if present
                user_content = question
                if options:
                    options_text = "\n".join([f"{chr(65+i)}) {opt}" for i, opt in enumerate(options)])
                    user_content += f"\n\nOptions:\n{options_text}"
                
                # Save image if present
                image_path = None
                if "image" in sample and sample["image"] is not None:
                    image_path = images_dir / f"{sample_id}.png"
                    try:
                        sample["image"].save(str(image_path))
                    except Exception:
                        image_path = None
                
                # Create normalized sample
                record = MultimodalSample(
                    id=sample_id,
                    messages=[
                        {"role": "user", "content": user_content},
                        {"role": "assistant", "content": str(answer)},
                    ],
                    domain="benchmark",
                    category=dataset_name.lower(),
                    modalities={
                        "image": [{"path": str(image_path), "type": "benchmark_image"}] if image_path else [],
                        "audio": [],
                        "video": [],
                    },
                    source=dataset_name,
                )
                
                f.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
                count += 1
                
            except Exception as e:
                logger.warning(f"Error processing benchmark sample {count}: {e}")
                continue
    
    logger.info(f"✅ Benchmark-{dataset_name}: {count} samples saved to {output_dir}")
    return count


def _generate_mock_benchmark_samples(output_dir: Path, sample_limit: int, dataset_name: str) -> int:
    """Generate mock benchmark samples for testing."""
    jsonl_file = output_dir / "data.jsonl"
    count = min(sample_limit, 50)
    
    with open(jsonl_file, 'w', encoding='utf-8') as f:
        for i in range(count):
            record = MultimodalSample(
                id=f"benchmark_{dataset_name.lower()}_mock_{i:07d}",
                messages=[
                    {"role": "user", "content": f"Mock benchmark question #{i}"},
                    {"role": "assistant", "content": f"Answer: A"},
                ],
                domain="benchmark",
                category=f"{dataset_name.lower()}_mock",
                modalities={"image": [], "audio": [], "video": []},
                source="mock",
            )
            f.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
    
    logger.info(f"Generated {count} mock benchmark samples for {dataset_name}")
    return count


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def load_config(config_path: Optional[str] = None) -> Dict:
    """Load configuration from YAML file or use defaults."""
    if config_path and HAS_YAML:
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                return config
        except Exception as e:
            logger.warning(f"Failed to load config from {config_path}: {e}")
    
    return DEFAULT_CONFIG


def main():
    parser = argparse.ArgumentParser(
        description="Download multimodal datasets for omni-modal training."
    )
    parser.add_argument(
        "--modality",
        type=str,
        choices=["vision", "audio", "video", "benchmarks", "all"],
        default="all",
        help="Which modality to download (default: all)",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Override sample limit for testing (e.g., 100 for quick test)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/mnt/e/data/multimodal",
        help="Base output directory",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/multimodal_datasets.yaml",
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Specific dataset to download (e.g., 'WebSight', 'MMMU')",
    )
    args = parser.parse_args()
    
    log_header(
        logger,
        "MULTIMODAL DATASET DOWNLOADER",
        {
            "Modality": args.modality,
            "Sample limit": args.sample or "default",
            "Output": args.output_dir,
        },
    )
    
    # Load configuration
    config = load_config(args.config)
    base_dir = Path(args.output_dir)
    
    total_samples = 0
    
    # Vision datasets
    if args.modality in ["vision", "all"]:
        vision_config = config.get("vision", DEFAULT_CONFIG["vision"])
        for name, ds_config in vision_config.items():
            if args.dataset and args.dataset.lower() != name.lower():
                continue
            sample_limit = args.sample or ds_config.get("sample", 10000)
            output_dir = base_dir / "vision" / name.lower()
            count = fetch_vision_dataset(ds_config, output_dir, sample_limit, name)
            total_samples += count
    
    # Audio datasets
    if args.modality in ["audio", "all"]:
        audio_config = config.get("audio", DEFAULT_CONFIG["audio"])
        for name, ds_config in audio_config.items():
            if args.dataset and args.dataset.lower() != name.lower():
                continue
            sample_limit = args.sample or ds_config.get("sample", 10000)
            lang = ds_config.get("language", "en")
            output_dir = base_dir / "audio" / lang
            count = fetch_audio_dataset(ds_config, output_dir, sample_limit, name)
            total_samples += count
    
    # Video datasets
    if args.modality in ["video", "all"]:
        video_config = config.get("video", DEFAULT_CONFIG["video"])
        for name, ds_config in video_config.items():
            if args.dataset and args.dataset.lower() != name.lower():
                continue
            sample_limit = args.sample or ds_config.get("sample", 1000)
            output_dir = base_dir / "video" / name.lower()
            count = fetch_video_dataset(ds_config, output_dir, sample_limit, name)
            total_samples += count
    
    # Benchmark datasets
    if args.modality in ["benchmarks", "all"]:
        bench_config = config.get("benchmarks", DEFAULT_CONFIG["benchmarks"])
        for name, ds_config in bench_config.items():
            if args.dataset and args.dataset.lower() != name.lower():
                continue
            sample_limit = args.sample or ds_config.get("sample", 1000)
            output_dir = base_dir / "benchmarks" / name.lower()
            count = fetch_benchmark_dataset(ds_config, output_dir, sample_limit, name)
            total_samples += count
    
    logger.info(f"\n{'='*60}")
    logger.info(f"🎉 Total samples downloaded: {total_samples}")
    logger.info(f"📁 Output directory: {base_dir}")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
