"""
Download multimodal datasets (Vision, Audio, Video)
Based on: multimodal_datasets.md artifact
"""
import os
from pathlib import Path
try:
    from datasets import load_dataset, Dataset
except ImportError:
    print("⚠️ 'datasets' library not found. Using SIMULATION MODE.")
    
    # Simulation classes
    class MockDataset:
        def __init__(self, name, size=100, split="train"):
            self.name = name
            self.size = size
            self.split = split
            
        def take(self, n):
            # Yeild mock data based on name
            limit = min(n, self.size)
            for i in range(limit):
                if "WebSight" in self.name:
                    yield {"image": None, "text": f"Simulated WebSight sample {i}"}
                elif "Common" in self.name:
                    yield {"path": f"/tmp/sim_audio_{i}.mp3", "sentence": f"Simulated Audio {i}"}
                elif "FineVideo" in self.name:
                    yield {"video": f"/tmp/sim_video_{i}.mp4", "text": f"Simulated Video {i}"}
                else:
                    yield {"text": f"Generic sample {i}"}
                    
    def load_dataset(name, **kwargs):
        print(f"   [SIMULATION] Loading mock dataset: {name}")
        return MockDataset(name)
        
    # Mock Dataset.from_list
    class Dataset:
        @staticmethod
        def from_list(data):
            return MockArrowDataset(data)
            
    class MockArrowDataset:
        def __init__(self, data):
            self.data = data
        def save_to_disk(self, path):
            import json
            Path(path).mkdir(parents=True, exist_ok=True)
            # Save as JSONL for simulation compatibility with distillation
            with open(Path(path) / "data.json", 'w') as f:
                json.dump(self.data, f)
            print(f"   [SIMULATION] Saved {len(self.data)} mock samples to {path}")

def download_vision_data(output_dir: str, limit: int = 10000):
    """Download WebSight (Vision) - Streaming Mode"""
    print(f"🖼️  Downloading WebSight (Vision) (Limit: {limit})...")
    try:
        # Stream the dataset to avoid downloading 2TB+
        ds = load_dataset("HuggingFaceM4/WebSight", split="train", streaming=True)
        
        save_path = Path(output_dir) / "vision"
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Take 10k samples
        # Note: We can't use save_to_disk simply with IterableDataset, we must iterate
        # But to keep similar API for the processor, we can write to a generator or save as Arrow chunks
        # Use a custom generator to save to disk in a format load_from_disk can read?
        # OR: Save as JSONL which is more portable for our pipeline. 
        # The previous code used save_to_disk. 
        # To maintain compatibility with 'load_from_disk' in distillation.py, we should better use 'Dataset.from_generator' or write to JSONL and change distillation.
        
        # ACTUALLY: The user wants efficient download. 
        # The best compatibility is: Stream -> Take N -> Dataset.from_list -> save_to_disk
        
        print(f"   Streaming {limit} samples...")
        samples = list(ds.take(limit))
        
        # Convert to standard Dataset to save to disk (so load_from_disk works)
        from datasets import Dataset
        # Note: WebSight features are image/text. 
        # We need to handle Image features carefully when converting to list and back.
        # But 'samples' list contains PIL images. Dataset.from_list handles this? Yes usually.
        
        static_ds = Dataset.from_list(samples)
        static_ds.save_to_disk(str(save_path))
        
        print(f"✅ Saved Vision data to {save_path}")
    except Exception as e:
        print(f"❌ Vision download failed: {e}")

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
        
        from datasets import Dataset
        # Audio feature is a dict array/struct. Dataset.from_list handles it.
        static_ds = Dataset.from_list(samples)
        static_ds.save_to_disk(str(save_path))
        
        print(f"✅ Saved Audio data to {save_path}")
    except Exception as e:
        print(f"❌ Audio download failed: {e}")

def download_video_data(output_dir: str, limit: int = 100):
    """Download FineVideo (Video)"""
    print(f"🎬 Downloading FineVideo (Video) (Limit: {limit})...")
    try:
        ds = load_dataset("HuggingFaceM4/FineVideo", split="train", streaming=True)
        
        save_path = Path(output_dir) / "video"
        save_path.mkdir(parents=True, exist_ok=True)
        
        print(f"   Streaming {limit} samples...")
        samples = list(ds.take(limit))
        
        from datasets import Dataset
        static_ds = Dataset.from_list(samples)
        static_ds.save_to_disk(str(save_path))
        
        print(f"✅ Saved Video data to {save_path}")
    except Exception as e:
        print(f"❌ Video download failed: {e}")
def get_test_prompts():
    """Return standard test prompts for verification."""
    return {
        "vision": [
            {"input": "tests/assets/test_image_01.jpg", "prompt": "Describe this UI design."},
            {"input": "tests/assets/test_screenshot_01.png", "prompt": "Convert this screenshot to HTML."}
        ],
        "audio": [
            {"input": "tests/assets/test_audio_01.mp3", "prompt": "Transcribe this audio file."}
        ],
        "video": [
            {"input": "tests/assets/test_video_01.mp4", "prompt": "Summarize the events in this video."}
        ]
    }
