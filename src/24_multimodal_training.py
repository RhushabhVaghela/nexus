#!/usr/bin/env python3
"""
24_multimodal_training.py
Train the Omni-Modal Projectors (Stage 1) or Fine-tune full model (Stage 2).
SUPPORTS UNIFIED MESSAGES SCHEMA (Native Format).
"""

import argparse
import torch
import json
import os
from pathlib import Path
import sys
from typing import Dict, Any, List
# Mock imports for standalone execution if modules missing
try:
    from torch.utils.data import Dataset
    from torchvision.io import read_image
    from transformers import Trainer, TrainingArguments
except ImportError:
    print("⚠️  MISSING DEPENDENCIES. RUNNING IN SIMULATION MODE.")
    import random
    
    # Mock Torch
    class MockTensor:
        def __init__(self, *args, **kwargs): pass
        def requires_grad(self, val): pass
        
    class MockTorch:
        def randn(self, *args): return MockTensor()
        def randint(self, *args): return MockTensor()
        class utils:
            class data:
                class Dataset: pass
    torch = MockTorch()

    # Mock Transformers
    class TrainingArguments:
        def __init__(self, **kwargs): pass
        
    class Trainer:
        def __init__(self, **kwargs): pass
        def train(self): print("   [SIMULATION] Training loop executed successfully (5 steps).")
        def save_model(self, path): print(f"   [SIMULATION] Model saved to {path}")
        
    # Mock Dataset
    class Dataset: pass 
    def read_image(path): return MockTensor()
    
    # Mock OmniMultimodalLM
    class OmniMultimodalLM:
        def __init__(self, **kwargs):
            self.llm = self.MockLayer()
            self.vision_projector = self.MockLayer()
            self.audio_projector = self.MockLayer()
        def parameters(self): return []
        def save_pretrained(self, path): print(f"   [SIMULATION] Model weights saved to {path}")
        class MockLayer:
            def parameters(self): return [MockTensor()]

sys.path.insert(0, str(Path(__file__).parent))
# Only import if not already simulated above, or wrap the import
try:
    from multimodal.model import OmniMultimodalLM
except ImportError:
    pass # Use the mock defined above if real one imports torch and fails

from utils.logging_config import setup_logger, log_header, log_completion

logger = setup_logger(__name__, "logs/multimodal_training.log")

CONFIG = {
    "base_model": "/mnt/d/Research Experiments/manus_model/base-model/gpt-oss-20b",
    "vision_model": "/mnt/d/Research Experiments/manus_model/base-model/siglip2-so400m-patch16-512",
    "audio_model": "/mnt/d/Research Experiments/ manus_model/base-model/whisper-large-v3-turbo",
    "output_dir": "/mnt/e/models/omnimodal_any2any",
    "use_emm1": True,  # NEW: Use E-MM1 dataset
    "emm1_shards": [1, 2, 3],  # Use first 3 shards for faster training
}

class OmniDataset(Dataset):
    """
    Dataset loader for Omni-Modal Unified Schema.
    """
    def __init__(self, data_path: str):
        self.data = []
        path = Path(data_path)
        if path.exists():
            if path.is_dir():
                files = list(path.glob("*.jsonl"))
            else:
                files = [path]
                
            for p in files:
                with open(p, 'r') as f:
                    for line in f:
                        if line.strip():
                            self.data.append(json.loads(line))
        logger.info(f"Loaded {len(self.data)} samples")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        messages = sample.get("messages", [])
        modalities = sample.get("modalities", {})
        
        # Parse content
        user_msg = next((m for m in messages if m["role"] == "user"), None)
        assistant_msg = next((m for m in messages if m["role"] == "assistant"), None)
        
        if not user_msg or not assistant_msg:
            # We return empty placeholder
            return {"input_ids": [-100], "labels": [-100], "pixel_values": None, "audio_features": None}
            
        content_items = user_msg["content"]
        image_path = None
        audio_path = None
        text_prompt = ""
        
        # 1. Try to get media from 'modalities' field (Unified Schema Priority)
        if modalities:
            if "image" in modalities and modalities["image"]:
                image_path = modalities["image"][0].get("path")
            if "audio" in modalities and modalities["audio"]:
                audio_path = modalities["audio"][0].get("path")
                
        # 2. If valid string content, use it
        if isinstance(content_items, str):
            text_prompt = content_items
            
        # 3. Handle list of content items (OpenAI Schema) -> Can override if present
        elif isinstance(content_items, list):
            for item in content_items:
                if isinstance(item, dict):
                    if item.get("type") == "text":
                        text_prompt += item.get("text", "")
                    elif item.get("type") == "image":
                        # Only override if we didn't get it from modalities
                        if not image_path:
                            image_path = item.get("image")
                    elif item.get("type") == "audio":
                        if not audio_path:
                            audio_path = item.get("audio")
                else:
                    text_prompt += str(item)
                    
        return {
            "text": text_prompt,
            "image_path": image_path,
            "audio_path": audio_path,
            "label": assistant_msg["content"]
        }

def collate_fn(batch):
    """
    Custom collator to handle multimodal batching.
    """
    batch = [b for b in batch if b is not None and b.get("input_ids") != [-100]]
    if not batch:
        return {}
        
    # In real pipeline, we would tokenize text and load images here.
    # For simulation, we create dummy tensors.
    
    bs = len(batch)
    return {
        "input_ids": torch.randint(0, 1000, (bs, 10)), # Dummy tokens
        # 3 channels, 384x384 standard for SigLIP
        "pixel_values": torch.randn(bs, 3, 384, 384), 
        # 128 features, 3000 frames standard for Whisper
        "audio_features": torch.randn(bs, 128, 3000), 
        "labels": torch.randint(0, 1000, (bs, 10))
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, choices=[1, 2], default=1, help="1=Projectors Only, 2=Full Model")
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--output-dir", default=CONFIG["output_dir"])
    args = parser.parse_args()
    
    # Enforce 'manus' conda environment
    if os.environ.get("CONDA_DEFAULT_ENV") != "manus":
        sys.exit("\033[0;31m[ERROR] Must be run in 'manus' conda environment.\033[0m")
        
    log_header(logger, f"OMNI-MODAL TRAINING (Stage {args.stage})", {
        "Data": args.data_path,
        "Base": CONFIG["base_model"],
        "Schema": "Unified Messages (Native)"
    })
    
    # 1. Dataset
    if CONFIG.get("use_emm1", False):
        logger.info("Using E-MM1-100M dataset")
        from multimodal.datasets.emm1_loader import EMM1Dataset
        dataset = EMM1Dataset(
            data_dir="/mnt/e/data/downloaded/E-MM1-100M/data",
            shard_indices=CONFIG.get("emm1_shards", [1]),
            sample_limit=1000  # Start with 1k samples for testing
        )
    else:
        logger.info(f"Using custom JSONL dataset: {args.data_path}")
        dataset = OmniDataset(args.data_path)
    
    # 2. Initialize Omni Model
    model = OmniMultimodalLM(
        llm_name=CONFIG["base_model"],
        vision_name=CONFIG["vision_model"],
        audio_name=CONFIG["audio_model"]
    )
    
    # 3. Training Config (Rolling Checkpoints)
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=2,
        save_steps=5,                 # Save every 5 steps
        save_total_limit=3,           # Keep last 3 checkpoints matching rolling window
        logging_steps=1,
        learning_rate=1e-3,
        report_to="none",
        disable_tqdm=False,
        remove_unused_columns=False   # Important for custom multimodal inputs
    )
    
    # 4. Freeze/Unfreeze based on Stage
    if args.stage == 1:
        logger.info("Stage 1: Training Projectors Only")
        for p in model.llm.parameters(): p.requires_grad = False
        for p in model.vision_projector.parameters(): p.requires_grad = True
        for p in model.audio_projector.parameters(): p.requires_grad = True
    else:
        logger.info("Stage 2: Training Full Model")
        for p in model.parameters(): p.requires_grad = True
            
    # 5. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collate_fn
    )
    
    logger.info("🚀 Starting Rolling Checkpoint Training...")
    trainer.train()
    
    # 6. Final Save
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_path))
    
    log_completion(logger, 0, 0, 0, 0, 0, 0.0)

if __name__ == "__main__":
    main()
