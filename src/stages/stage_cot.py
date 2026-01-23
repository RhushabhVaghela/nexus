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
            
        # Initialize optimizer
        if self.model:
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.learning_rate)
            
        # Load CoT datasets
        self.logger.info("Loading Chain-of-Thought datasets dynamic...")
        self.train_dataset = self.load_dynamic_datasets()
        
        if self.train_dataset:
            self.logger.info(f"Total training samples: {len(self.train_dataset)}")
        else:
            self.logger.warning("No datasets loaded")
            
        return True
    
    def train(self) -> Dict[str, Any]:
        """Run CoT training."""
        if self.config.dry_run:
            return super().train()
        
        if self.train_dataset is None or len(self.train_dataset) == 0:
            self.logger.warning("No training data, skipping training")
            return {"success": True, "steps": 0, "skipped": True}
            
        self.logger.info("Starting CoT training...")
        
        # Tokenize dataset once (if small)
        self.logger.info("Tokenizing dataset...")
        def tokenize_function(sample):
            if "text" in sample:
                text = sample["text"]
            elif "messages" in sample:
                text = str(sample["messages"])
            else:
                text = ""
            return self.tokenizer(text, truncation=True, max_length=2048, padding="max_length")
            
        tokenized_dataset = self.train_dataset.map(tokenize_function, batched=False)
        tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
        
        # Training loop integration with training_controller
        from src.training_controller import training_step_hook
        from torch.utils.data import DataLoader
        
        train_dataloader = DataLoader(tokenized_dataset, batch_size=self.config.batch_size, shuffle=True)
        
        total_steps = 0
        total_loss = 0.0
        
        for epoch in range(self.config.epochs):
            self.logger.info(f"Epoch {epoch + 1}/{self.config.epochs}")
            
            for batch in train_dataloader:
                self.current_step = total_steps
                
                # Training controller hook (pause/checkpoint/cooldown)
                training_step_hook(
                    self.model,
                    self.optimizer,
                    self.current_step,
                    str(self.checkpoint_dir),
                )
                
                # Move to device
                inputs = {k: v.to(self.model.device) for k, v in batch.items()}
                inputs["labels"] = inputs["input_ids"].clone()
                
                # Forward pass
                outputs = self.model(**inputs)
                loss = outputs.loss
                total_loss += loss.item()
                
                # Backward and optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_steps += 1
                
                if total_steps % 10 == 0:
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
