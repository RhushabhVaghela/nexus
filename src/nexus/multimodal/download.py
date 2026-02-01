"""
Download multimodal datasets (Vision, Audio, Video)
Based on: multimodal_datasets.md artifact
"""
import os
from pathlib import Path
from datasets import load_dataset, Dataset

def download_vision_data(output_dir: str, limit: int = 10000):
    """Download WebSight (Vision) - Streaming Mode"""
    print(f"🖼️  Downloading WebSight (Vision) (Limit: {limit})...")
    try:
        # Stream the dataset to avoid downloading 2TB+
        ds = load_dataset("HuggingFaceM4/WebSight", split="train", streaming=True)
        
        save_path = Path(output_dir) / "vision"
        save_path.mkdir(parents=True, exist_ok=True)
        
        print(f"   Streaming {limit} samples...")
        samples = list(ds.take(limit))
        
        static_ds = Dataset.from_list(samples)
        static_ds.save_to_disk(str(save_path))
        
        print(f"✅ Saved Vision data to {save_path}")
    except Exception as e:
        print(f"❌ Vision download failed: {e}")
        raise e

def download_audio_data(output_dir: str, limit: int = 1000):
    """Download Common Voice (Audio)"""
    print(f"🎤 Downloading Common Voice (Audio) (Limit: {limit})...")
    try:
        # Stream
        ds = load_dataset("mozilla-foundation/common_voice_17_0", "en", split="train", streaming=True, trust_remote_code=True)
        
        save_path = Path(output_dir) / "audio"
        save_path.mkdir(parents=True, exist_ok=True)
        
        print(f"   Streaming {limit} samples...")
        samples = list(ds.take(limit))
        
        static_ds = Dataset.from_list(samples)
        static_ds.save_to_disk(str(save_path))
        
        print(f"✅ Saved Audio data to {save_path}")
    except Exception as e:
        print(f"❌ Audio download failed: {e}")
        raise e

def download_video_data(output_dir: str, limit: int = 100):
    """Download FineVideo (Video)"""
    print(f"🎬 Downloading FineVideo (Video) (Limit: {limit})...")
    try:
        ds = load_dataset("HuggingFaceM4/FineVideo", split="train", streaming=True)
        
        save_path = Path(output_dir) / "video"
        save_path.mkdir(parents=True, exist_ok=True)
        
        print(f"   Streaming {limit} samples...")
        samples = list(ds.take(limit))
        
        static_ds = Dataset.from_list(samples)
        static_ds.save_to_disk(str(save_path))
        
        print(f"✅ Saved Video data to {save_path}")
    except Exception as e:
        print(f"❌ Video download failed: {e}")
        raise e

def get_test_prompts():
    """Return standard test prompts for verification."""
    return {
        "vision": [
            {"id": "vision_01", "input": "tests/assets/test_image_01.jpg", "prompt": "Describe this UI design."},
            {"id": "vision_02", "input": "tests/assets/test_screenshot_01.png", "prompt": "Convert this screenshot to HTML."},
            {"id": "vision_03", "input": "assets/test_images/dashboard_ui.png", "prompt": "Analyze this dashboard."}
        ],
        "audio": [
            {"id": "audio_01", "input": "tests/assets/test_audio_01.mp3", "prompt": "Transcribe this audio file."}
        ],
        "video": [
            {"id": "video_01", "input": "tests/assets/test_video_01.mp4", "prompt": "Summarize the events in this video."}
        ]
    }
