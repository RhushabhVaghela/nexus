#!/usr/bin/env python3
"""
stage_tools.py
Tool/function calling training stage.
"""

from typing import Dict, Any
from datasets import load_dataset

from .base import TextCapabilityStage, StageConfig


class ToolsStage(TextCapabilityStage):
    """Tool calling capability training."""
    
    CAPABILITY_NAME = "tool-calling"
    
    DATASET_PATTERNS = [
        "argilla/Synth-APIGen-v0.1",
        "Salesforce/xlam-function-calling-60k",
        "*tool*",
        "*function*call*",
    ]
    
    def prepare(self) -> bool:
        if not super().prepare():
            return False
        
        if self.config.dry_run:
            self.logger.info("[DRY-RUN] Would load tool calling datasets")
            return True
        
        self.logger.info("Loading tool calling datasets...")
        
        try:
            ds = load_dataset(
                "Salesforce/xlam-function-calling-60k",
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
    parser = argparse.ArgumentParser(description="Tool Calling Training")
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--sample-size", type=int, default=0)
    args = parser.parse_args()
    
    config = StageConfig(
        capability_name="tool-calling",
        base_model_path=args.base_model,
        output_dir=args.output_dir,
        sample_size=args.sample_size,
        dry_run=args.dry_run,
    )
    stage = ToolsStage(config)
    return 0 if stage.run().get("success") else 1

if __name__ == "__main__":
    exit(main())
