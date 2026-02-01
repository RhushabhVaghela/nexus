#!/usr/bin/env python3
"""
mm_download_voice_assistant_lite.py

Standalone script to download a specific "Lite" subset (50k samples) of
the gpt-omni/VoiceAssistant-400K dataset.

This uses streaming to avoid downloading the massive 200GB+ dataset.
It saves audio files to disk and creates a data.jsonl for training.
"""

import os
import sys
import json
import logging
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Any

# Ensure we can import utils if running from src/
sys.path.insert(0, str(Path(__file__).parent))

try:
    from datasets import load_dataset, Audio
    import soundfile as sf
except ImportError:
    print("❌ Error: Missing required libraries.")
    print("Please install them with: pip install datasets soundfile numpy")
    sys.exit(1)

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration
DATASET_NAME = "gpt-omni/VoiceAssistant-400K"
DEFAULT_LIMIT = 50000
OUTPUT_DIR = Path("/mnt/e/data/VoiceAssistant_Lite_v2")

def process_sample(sample: Dict[str, Any], count: int, audio_output_dir: Path) -> Dict[str, Any]:
    """
    Extracts audio and text from a sample and saves audio to disk.
    Returns a normalized dictionary for training.
    """
    sample_id = f"va_400k_{count:06d}"
    
    # Extract Texts
    question_text = sample.get("question", "")
    answer_text = sample.get("answer", "")
    
    # Extract Audio (Question)
    # Note: In streaming mode, Audio features come as dicts with 'array' and 'sampling_rate'
    audio_path = None
    if "question_audio" in sample and sample["question_audio"] is not None:
        audio_data = sample["question_audio"]
        # Debug structure
        # logger.info(f"Audio data type: {type(audio_data)}")
        
        # Determine filename
        audio_filename = f"{sample_id}_q.wav"
        output_path = audio_output_dir / audio_filename
        
        # Save Audio
        try:
            # Check for raw bytes (streaming mode common format)
            if isinstance(audio_data, dict) and "bytes" in audio_data and audio_data["bytes"]:
                with open(output_path, "wb") as f_msg:
                    f_msg.write(audio_data["bytes"])
                audio_path = str(output_path)
            
            # Check for array/sampling_rate (decoded format)
            elif isinstance(audio_data, dict) and "array" in audio_data:
                array = audio_data.get("array")
                sr = audio_data.get("sampling_rate")
                
                if array is not None and sr is not None:
                    sf.write(str(output_path), array, sr)
                    audio_path = str(output_path)
            
            # Handle AudioDecoder from streaming with torch
            elif hasattr(audio_data, "_hf_encoded") and isinstance(audio_data._hf_encoded, dict):
                 enc = audio_data._hf_encoded
                 if "bytes" in enc and enc["bytes"]:
                     with open(output_path, "wb") as f_msg:
                        f_msg.write(enc["bytes"])
                     audio_path = str(output_path)
                 elif "path" in enc and enc["path"]:
                     # If it's a local path (unlikely in streaming unless cached)
                     pass
                 else:
                     logger.warning(f"AudioDecoder _hf_encoded has no bytes: {enc.keys()}")

            else:
                 logger.warning(f"Unknown audio format for {sample_id}: {type(audio_data)}")
                 # logger.warning(f"Attributes: {dir(audio_data)}")

        except Exception as e:
            logger.warning(f"Failed to save audio for {sample_id}: {e}")

    # Build Record
    # We follow the 'MultimodalSample' schema roughly
    record = {
        "id": sample_id,
        "messages": [
            {"role": "user", "content": question_text},
            {"role": "assistant", "content": answer_text}
        ],
        "domain": "voice_assistant",
        "category": "conversation",
        "modalities": {
            "audio": [
                {"path": audio_path, "type": "question_speech"}
            ] if audio_path else []
        },
        "source": DATASET_NAME
    }
    
    return record

def main():
    parser = argparse.ArgumentParser(description="Download VoiceAssistant-400K Lite")
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT, help="Number of samples to download")
    parser.add_argument("--output", type=str, default=str(OUTPUT_DIR), help="Output directory")
    args = parser.parse_args()
    
    limit = args.limit
    out_dir = Path(args.output)
    audio_dir = out_dir / "wavs"
    
    # Create directories
    out_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(exist_ok=True)
    
    logger.info(f"🚀 Starting download of {DATASET_NAME}")
    logger.info(f"📦 Limit: {limit} samples")
    logger.info(f"📂 Output: {out_dir}")
    
    # Load Dataset (Streaming Mode)
    logger.info("📡 Connecting to Hugging Face (Streaming)...")
    try:
        ds = load_dataset(DATASET_NAME, split="train", streaming=True, trust_remote_code=True)
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return

    # Process Samples
    count = 0
    skipped = 0
    jsonl_path = out_dir / "data.jsonl"
    
    with open(jsonl_path, "w", encoding="utf-8") as f:
        # pbar = tqdm(total=limit, desc="Downloading", unit="sample")
        logger.info("Iterating dataset...")
        
        for i, sample in enumerate(ds):
            if count >= limit:
                break
                
            try:
                # logger.info(f"Processing sample {i}")
                record = process_sample(sample, count, audio_dir)
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                
                count += 1
                if count % 1000 == 0:
                    logger.info(f"Processsed {count} samples")
                # pbar.update(1)
                
            except Exception as e:
                logger.warning(f"Error processing sample {count}: {e}")
                skipped += 1
                
        # pbar.close()
        
    logger.info("✅ Download Complete!")
    logger.info(f"📊 Total Saved: {count}")
    logger.info(f"⏭️  Skipped/Failed: {skipped}")
    logger.info(f"💾 JSONL Location: {jsonl_path}")
    logger.info(f"🎵 Audio Location: {audio_dir}")

if __name__ == "__main__":
    main()
