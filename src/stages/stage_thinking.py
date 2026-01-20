#!/usr/bin/env python3
"""
stage_thinking.py
Extended thinking/reflection training stage.
"""

from typing import Dict, Any
from datasets import load_dataset

from .base import TextCapabilityStage, StageConfig


class ThinkingStage(TextCapabilityStage):
    """Extended thinking/reflection training."""
    
    CAPABILITY_NAME = "thinking"
    
    DATASET_PATTERNS = [
        "simplescaling/s1K-1.1",
        "open-thoughts/OpenThoughts-114k",
    ]
    
    def prepare(self) -> bool:
        if not super().prepare():
            return False
        
        if self.config.dry_run:
            self.logger.info("[DRY-RUN] Would load thinking datasets")
            return True
        
        self.logger.info("Loading extended thinking datasets...")
        
        try:
            ds = load_dataset(
                "simplescaling/s1K-1.1",
                split="train",
                trust_remote_code=True,
            )
            if self.config.sample_size > 0:
                ds = ds.select(range(min(self.config.sample_size, len(ds))))
            self.train_dataset = ds
            self.logger.info(f"Loaded: {len(ds)} samples")
            return True
        except Exception as e:
            self.logger.warning(f"Could not load dataset: {e}")
            return True
    
    def train(self) -> Dict[str, Any]:
        if self.config.dry_run:
            return super().train()
        return super().train()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Extended Thinking Training")
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--sample-size", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()
    
    config = StageConfig(
        capability_name="thinking",
        base_model_path=args.base_model,
        output_dir=args.output_dir,
        sample_size=args.sample_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        dry_run=args.dry_run,
    )
    stage = ThinkingStage(config)
    return 0 if stage.run().get("success") else 1

if __name__ == "__main__":
    exit(main())
