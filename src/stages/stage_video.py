#!/usr/bin/env python3
"""
stage_video.py
Video understanding training stage.

Trains on video-text datasets for temporal reasoning and video comprehension.
"""

import torch
from typing import Dict, Any
from pathlib import Path
from datasets import load_dataset

from .base import BaseStage, StageConfig


class VideoUnderstandingStage(BaseStage):
    """Video understanding training with temporal reasoning."""
    
    CAPABILITY_NAME = "video-understanding"
    
    DATASET_PATTERNS = [
        "HuggingFaceM4/webvid",
        "*video*",
        "*temporal*",
        "*MSR-VTT*",
    ]
    
    def prepare(self) -> bool:
        """Load model and video datasets."""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        self.logger.info(f"Loading model from {self.config.base_model_path}")
        
        if self.config.dry_run:
            self.logger.info("[DRY-RUN] Would load model and video datasets")
            return True
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.base_model_path,
                trust_remote_code=True,
            )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.base_model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load video dataset
            self.logger.info("Loading video datasets dynamic...")
            self.train_dataset = self.load_dynamic_datasets()
            
            if self.train_dataset:
                self.logger.info(f"Loaded: {len(self.train_dataset)} samples")
            else:
                self.logger.warning("No datasets loaded")
            
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to prepare: {e}")
            return False
    
    def train(self) -> Dict[str, Any]:
        """Train on video caption data with temporal context."""
        if self.config.dry_run:
            self.logger.info("[DRY-RUN] Simulating video understanding training...")
            for epoch in range(self.config.epochs):
                self.logger.info(f"[DRY-RUN] Epoch {epoch+1}/{self.config.epochs}")
                for step in range(10):
                    self.current_step += 1
            return {"success": True, "dry_run": True, "steps": self.current_step}
        
        if not self.train_dataset:
            self.logger.warning("No training data, skipping")
            return {"success": True, "steps": 0, "skipped": True}
        
        self.logger.info("Starting video understanding training...")
        
        from src.training_controller import training_step_hook
        
        total_loss = 0.0
        
        for epoch in range(self.config.epochs):
            self.logger.info(f"Epoch {epoch + 1}/{self.config.epochs}")
            
            for sample in self.train_dataset:
                self.current_step += 1
                
                training_step_hook(
                    self.model, self.optimizer, self.current_step,
                    str(self.checkpoint_dir)
                )
                
                # Get caption with temporal context
                caption = sample.get("caption", sample.get("text", ""))
                if not caption:
                    continue
                
                # Add temporal framing
                text = f"[VIDEO] Describe this video:\n{caption}"
                
                inputs = self.tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                    padding=True,
                )
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
                inputs["labels"] = inputs["input_ids"].clone()
                
                outputs = self.model(**inputs)
                loss = outputs.loss
                total_loss += loss.item()
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                if self.current_step % 100 == 0:
                    avg = total_loss / self.current_step
                    self.logger.info(f"Step {self.current_step}, Avg Loss: {avg:.4f}")
        
        return {
            "success": True,
            "steps": self.current_step,
            "final_loss": total_loss / max(self.current_step, 1),
        }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Video Understanding Training")
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--sample-size", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()
    
    config = StageConfig(
        capability_name="video-understanding",
        base_model_path=args.base_model,
        output_dir=args.output_dir,
        sample_size=args.sample_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        dry_run=args.dry_run,
    )
    stage = VideoUnderstandingStage(config)
    return 0 if stage.run().get("success") else 1

if __name__ == "__main__":
    exit(main())
