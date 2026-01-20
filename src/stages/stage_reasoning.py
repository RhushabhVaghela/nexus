#!/usr/bin/env python3
"""
stage_reasoning.py
Multi-level reasoning training stage.

Trains on advanced reasoning and math datasets.
"""

from typing import Dict, Any
from datasets import load_dataset, concatenate_datasets

from .base import TextCapabilityStage, StageConfig


class ReasoningStage(TextCapabilityStage):
    """Multi-level reasoning training."""
    
    CAPABILITY_NAME = "reasoning"
    
    DATASET_PATTERNS = [
        "nvidia/OpenMathReasoning",
        "AI-MO/NuminaMath-CoT",
        "*reasoning*",
        "*math*",
    ]
    
    def prepare(self) -> bool:
        """Prepare reasoning training."""
        if not super().prepare():
            return False
        
        if self.config.dry_run:
            self.logger.info("[DRY-RUN] Would load reasoning datasets")
            return True
        
        self.logger.info("Loading reasoning datasets...")
        
        try:
            datasets = []
            
            # NuminaMath-CoT
            try:
                ds = load_dataset(
                    "AI-MO/NuminaMath-CoT",
                    split="train",
                    trust_remote_code=True,
                )
                if self.config.sample_size > 0:
                    ds = ds.select(range(min(self.config.sample_size, len(ds))))
                datasets.append(ds)
                self.logger.info(f"Loaded NuminaMath-CoT: {len(ds)} samples")
            except Exception as e:
                self.logger.warning(f"Could not load NuminaMath: {e}")
            
            if datasets:
                self.train_dataset = concatenate_datasets(datasets)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load datasets: {e}")
            return False
    
    def train(self) -> Dict[str, Any]:
        """Run reasoning training."""
        if self.config.dry_run:
            return super().train()
        
        # Similar to CoT but with reasoning-specific processing
        return super().train()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Reasoning Training Stage")
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--sample-size", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    
    args = parser.parse_args()
    
    config = StageConfig(
        capability_name="reasoning",
        base_model_path=args.base_model,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        sample_size=args.sample_size,
        dry_run=args.dry_run,
    )
    
    stage = ReasoningStage(config)
    results = stage.run()
    
    return 0 if results.get("success") else 1


if __name__ == "__main__":
    exit(main())
