#!/usr/bin/env python3
"""
stage_podcast.py
NotebookLM-style podcast generation training stage.

Trains on conversational podcast datasets to generate natural dialogue.
"""

import torch
from typing import Dict, Any
from pathlib import Path
from datasets import load_dataset, concatenate_datasets

from .base import BaseStage, StageConfig


class PodcastStage(BaseStage):
    """Podcast generation training - NotebookLM style dialogue."""
    
    CAPABILITY_NAME = "podcast"
    
    DATASET_PATTERNS = [
        "mozilla-foundation/common_voice_*",
        "*podcast*",
        "*conversation*",
        "*dialog*",
    ]
    
    def prepare(self) -> bool:
        """Load model and podcast datasets."""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        self.logger.info(f"Loading model from {self.config.base_model_path}")
        
        if self.config.dry_run:
            self.logger.info("[DRY-RUN] Would load model and podcast datasets")
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
            
            # Load podcast-style datasets
            self.logger.info("Loading podcast/conversational datasets...")
            
            datasets = []
            
            # Try conversational dataset
            try:
                ds = load_dataset(
                    "daily_dialog",
                    split="train",
                    trust_remote_code=True,
                )
                if self.config.sample_size > 0:
                    ds = ds.select(range(min(self.config.sample_size, len(ds))))
                datasets.append(ds)
                self.logger.info(f"Loaded daily_dialog: {len(ds)} samples")
            except Exception as e:
                self.logger.warning(f"Could not load daily_dialog: {e}")
            
            if datasets:
                self.train_dataset = concatenate_datasets(datasets)
                self.logger.info(f"Total training samples: {len(self.train_dataset)}")
            else:
                self.train_dataset = None
            
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to prepare: {e}")
            return False
    
    def train(self) -> Dict[str, Any]:
        """Train on podcast dialogue data."""
        if self.config.dry_run:
            self.logger.info("[DRY-RUN] Simulating podcast training...")
            for epoch in range(self.config.epochs):
                self.logger.info(f"[DRY-RUN] Epoch {epoch+1}/{self.config.epochs}")
                for step in range(10):
                    self.current_step += 1
            return {"success": True, "dry_run": True, "steps": self.current_step}
        
        if self.train_dataset is None or len(self.train_dataset) == 0:
            self.logger.warning("No training data, skipping")
            return {"success": True, "steps": 0, "skipped": True}
        
        self.logger.info("Starting podcast training...")
        
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
                
                # Handle different dataset formats
                if "dialog" in sample:
                    text = " [TURN] ".join(sample["dialog"])
                elif "text" in sample:
                    text = sample["text"]
                else:
                    continue
                
                inputs = self.tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=2048,
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
    parser = argparse.ArgumentParser(description="Podcast Training Stage")
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--sample-size", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()
    
    config = StageConfig(
        capability_name="podcast",
        base_model_path=args.base_model,
        output_dir=args.output_dir,
        sample_size=args.sample_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        dry_run=args.dry_run,
    )
    stage = PodcastStage(config)
    return 0 if stage.run().get("success") else 1

if __name__ == "__main__":
    exit(main())
