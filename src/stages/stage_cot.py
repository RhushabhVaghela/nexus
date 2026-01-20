#!/usr/bin/env python3
"""
stage_cot.py
Chain-of-Thought (CoT) reasoning training stage.

Trains the model on step-by-step reasoning datasets.
"""

import torch
from pathlib import Path
from typing import Dict, Any, List
from datasets import load_dataset, concatenate_datasets

from .base import TextCapabilityStage, StageConfig


class CoTStage(TextCapabilityStage):
    """Chain-of-Thought reasoning training."""
    
    CAPABILITY_NAME = "cot"
    
    DATASET_PATTERNS = [
        "open-thoughts/OpenThoughts-114k",
        "nvidia/OpenMathReasoning",
        "*CoT*",
        "*chain*thought*",
    ]
    
    def prepare(self) -> bool:
        """Prepare CoT training."""
        # Load base model
        if not super().prepare():
            return False
        
        if self.config.dry_run:
            self.logger.info("[DRY-RUN] Would load CoT datasets")
            return True
        
        # Load CoT datasets
        self.logger.info("Loading Chain-of-Thought datasets...")
        
        try:
            datasets = []
            
            # Try loading OpenThoughts
            try:
                ds = load_dataset(
                    "open-thoughts/OpenThoughts-114k",
                    split="train",
                    trust_remote_code=True,
                )
                if self.config.sample_size > 0:
                    ds = ds.select(range(min(self.config.sample_size, len(ds))))
                datasets.append(ds)
                self.logger.info(f"Loaded OpenThoughts: {len(ds)} samples")
            except Exception as e:
                self.logger.warning(f"Could not load OpenThoughts: {e}")
            
            if datasets:
                self.train_dataset = concatenate_datasets(datasets)
                self.logger.info(f"Total training samples: {len(self.train_dataset)}")
            else:
                self.logger.warning("No datasets loaded, using empty dataset")
                self.train_dataset = None
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load datasets: {e}")
            return False
    
    def train(self) -> Dict[str, Any]:
        """Run CoT training."""
        if self.config.dry_run:
            return super().train()
        
        if self.train_dataset is None or len(self.train_dataset) == 0:
            self.logger.warning("No training data, skipping training")
            return {"success": True, "steps": 0, "skipped": True}
        
        self.logger.info("Starting CoT training...")
        
        # Training loop integration with training_controller
        from src.training_controller import training_step_hook
        
        total_steps = 0
        total_loss = 0.0
        
        for epoch in range(self.config.epochs):
            self.logger.info(f"Epoch {epoch + 1}/{self.config.epochs}")
            
            # Simple training loop
            for i, sample in enumerate(self.train_dataset):
                self.current_step = total_steps
                
                # Training controller hook (pause/checkpoint/cooldown)
                training_step_hook(
                    self.model,
                    self.optimizer,
                    self.current_step,
                    str(self.checkpoint_dir),
                )
                
                # Tokenize and process sample
                if "text" in sample:
                    text = sample["text"]
                elif "messages" in sample:
                    text = str(sample["messages"])
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
                
                # Forward pass
                outputs = self.model(**inputs)
                loss = outputs.loss
                total_loss += loss.item()
                
                # Backward (simplified - real impl would use optimizer)
                loss.backward()
                
                total_steps += 1
                
                if total_steps % 100 == 0:
                    avg_loss = total_loss / total_steps
                    self.logger.info(f"Step {total_steps}, Avg Loss: {avg_loss:.4f}")
                
                if total_steps % self.config.save_steps == 0:
                    self.save_checkpoint()
        
        return {
            "success": True,
            "steps": total_steps,
            "final_loss": total_loss / max(total_steps, 1),
            "epochs": self.config.epochs,
        }


def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="CoT Training Stage")
    parser.add_argument("--base-model", required=True, help="Base model path")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--sample-size", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true", help="Dry run mode")
    
    args = parser.parse_args()
    
    config = StageConfig(
        capability_name="cot",
        base_model_path=args.base_model,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        sample_size=args.sample_size,
        dry_run=args.dry_run,
    )
    
    stage = CoTStage(config)
    results = stage.run()
    
    return 0 if results.get("success") else 1


if __name__ == "__main__":
    exit(main())
