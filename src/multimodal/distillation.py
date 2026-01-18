"""
Multimodal Data Processor (Format Normalizer)
Converts raw real datasets (WebSight, CommonVoice, FineVideo) into the Unified Messages Schema.
Ensures consistency with the Native Model Schema (src/04_process_real_datasets.py).
NOW SUPPORTS EXISTING SPLITS & TRAIN/VAL/TEST LOGIC.
"""
import json
import logging
import uuid
import random
from pathlib import Path
from typing import Dict, List, Any
try:
    from datasets import load_from_disk
except ImportError:
    def load_from_disk(path): return []

def safe_load_from_disk(path):
    """Load data from disk, handling both Real (Arrow) and Sim (JSON) formats."""
    import json
    p = Path(path)
    # Check for Simulation Artifact first (data.json)
    if (p / "data.json").exists():
        with open(p / "data.json", 'r') as f:
            return json.load(f) # Returns list (flat dataset)
            
    # Fallback to real lib if available
    try:
        return load_from_disk(str(path))
    except:
        return []

logger = logging.getLogger(__name__)

class MultimodalDataProcessor:
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.processed_dir = self.data_dir / "processed"
        self.processed_dir.mkdir(exist_ok=True)

    def _write_splits(self, samples: List[Dict], output_base_name: str, force_split: str = None):
        """
        Write samples to files. 
        If force_split is provided (e.g. 'test'), write all to that split.
        Otherwise, shuffle and split 95/2.5/2.5.
        """
        if force_split:
            out_file = self.processed_dir / force_split / f"{output_base_name}.jsonl"
            out_file.parent.mkdir(parents=True, exist_ok=True)
            with open(out_file, 'w') as f:
                for s in samples:
                    f.write(json.dumps(s) + "\n")
            logger.info(f"Saved {len(samples)} samples to {out_file} ({force_split})")
            return

        # Default Splitting Logic
        random.shuffle(samples)
        total = len(samples)
        train_end = int(total * 0.95)
        val_end = int(total * 0.975)
        
        splits = {
            "train": samples[:train_end],
            "val": samples[train_end:val_end],
            "test": samples[val_end:]
        }
        
        for split, data in splits.items():
            if not data: continue
            out_file = self.processed_dir / split / f"{output_base_name}.jsonl"
            out_file.parent.mkdir(parents=True, exist_ok=True)
            with open(out_file, 'w') as f:
                for s in data:
                    f.write(json.dumps(s) + "\n")
            logger.info(f"Saved {len(data)} {split} samples to {out_file}")

    def process_vision(self, input_path: Path):
        """Process WebSight -> Splits"""
        logger.info(f"Processing Vision Data from {input_path}")
        try:
            ds = safe_load_from_disk(str(input_path))
            images_dir = self.processed_dir / "images"
            images_dir.mkdir(exist_ok=True)
            
            # Check if dataset has splits in itself
            # datasets object might have keys like 'train', 'test'
            if hasattr(ds, "keys") and "train" in ds.keys():
                # Process each split separately
                for split_name in ds.keys():
                    self._process_dataset_split(ds[split_name], split_name, "vision_instruct", images_dir)
            else:
                # Flat dataset
                self._process_dataset_split(ds, None, "vision_instruct", images_dir)
            
        except Exception as e:
            logger.error(f"Failed to process vision data: {e}")

    def _process_dataset_split(self, ds, split_name, base_name, images_dir):
        """Helper to process a specific split or flat ds"""
        samples = []
        for i, item in enumerate(ds):
            img_name = f"websight_{uuid.uuid4().hex[:8]}.jpg"
            img_path = images_dir / img_name
            
            prompt = "Convert this design to HTML/Tailwind Code."
            response = item.get("text", "")
            
            message = {
                "messages": [
                    {
                        "role": "user", 
                        "content": [
                            {"type": "image", "image": str(img_path)},
                            {"type": "text", "text": prompt}
                            ]
                    },
                    {"role": "assistant", "content": response}
                ],
                "source": "websight",
                "modality": "vision"
            }
            samples.append(message)
            if i >= 500: break
        
        # Map HF split names to ours
        target_split = None
        if split_name:
            if "test" in split_name: target_split = "test"
            elif "val" in split_name or "dev" in split_name: target_split = "val"
            elif "train" in split_name: target_split = "train"
        
        self._write_splits(samples, base_name, force_split=target_split)

    def process_audio(self, input_path: Path):
        """Process CommonVoice -> Splits"""
        logger.info(f"Processing Audio Data from {input_path}")
        try:
            ds = safe_load_from_disk(str(input_path))
            if hasattr(ds, "keys") and "train" in ds.keys():
                for split_name in ds.keys():
                    self._process_audio_split(ds[split_name], split_name, "audio_instruct")
            else:
                self._process_audio_split(ds, None, "audio_instruct")
            
        except Exception as e:
            logger.error(f"Failed to process audio data: {e}")

    def _process_audio_split(self, ds, split_name, base_name):
        samples = []
        for i, item in enumerate(ds):
            audio_path = item.get("path", f"audio_{i}.mp3")
            prompt = "Transcribe the following audio."
            response = item.get("sentence", "")
            
            message = {
                "messages": [
                    {
                        "role": "user", 
                        "content": [
                            {"type": "audio", "audio": str(audio_path)},
                            {"type": "text", "text": prompt}
                            ]
                    },
                    {"role": "assistant", "content": response}
                ],
                "source": "common_voice",
                "modality": "audio"
            }
            samples.append(message)
            if i >= 500: break
            
        target_split = None
        if split_name:
            if "test" in split_name: target_split = "test"
            elif "val" in split_name: target_split = "val"
            elif "train" in split_name: target_split = "train"
        
    def process_video(self, input_path: Path):
        """Process FineVideo -> Splits"""
        logger.info(f"Processing Video Data from {input_path}")
        try:
            ds = safe_load_from_disk(str(input_path))
            video_dir = self.processed_dir / "videos"
            video_dir.mkdir(exist_ok=True)
            
            if hasattr(ds, "keys") and "train" in ds.keys():
                for split_name in ds.keys():
                    self._process_video_split(ds[split_name], split_name, "video_instruct", video_dir)
            else:
                self._process_video_split(ds, None, "video_instruct", video_dir)
        except Exception as e:
            logger.error(f"Failed to process video data: {e}")

    def _process_video_split(self, ds, split_name, base_name, video_dir):
        samples = []
        for i, item in enumerate(ds):
            # Simulate video path or extraction
            vid_name = f"finevideo_{uuid.uuid4().hex[:8]}.mp4"
            vid_path = video_dir / vid_name
            
            # FineVideo Structure Strategy
            # Use 'title' or 'transcripts' if available
            # item is a dict
            content_text = item.get("title")
            if not content_text:
                # Try transcript
                # transcripts is usually a dict or list
                trans = item.get("transcripts")
                if trans and isinstance(trans, dict):
                     # e.g. {'en': '...'}
                     content_text = next(iter(trans.values()), "Video content")
                else:
                     content_text = "Video content"
            
            prompt = "Describe the events in this video."
            response = content_text # Use title/transcript as the 'ground truth' for captioning task
            
            message = {
                "messages": [
                    {
                        "role": "user", 
                        "content": [
                            {"type": "video", "video": str(vid_path)},
                            {"type": "text", "text": prompt}
                            ]
                    },
                    {"role": "assistant", "content": response}
                ],
                "source": "finevideo",
                "modality": "video"
            }
            samples.append(message)
            if i >= 500: break
            
        target_split = None
        if split_name:
            if "test" in split_name: target_split = "test"
            elif "val" in split_name: target_split = "val"
            elif "train" in split_name: target_split = "train"
        
        self._write_splits(samples, base_name, force_split=target_split)

    def run(self):
        vis_in = self.data_dir / "vision"
        if vis_in.exists(): self.process_vision(vis_in)
            
        aud_in = self.data_dir / "audio"
        if aud_in.exists(): self.process_audio(aud_in)
            
        vid_in = self.data_dir / "video"
        if vid_in.exists(): self.process_video(vid_in)

if __name__ == "__main__":
    processor = MultimodalDataProcessor("/mnt/e/data/multimodal")
    processor.run()
