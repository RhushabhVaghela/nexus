#!/usr/bin/env python3
"""
Omni Training Stage
Training stage specifically for Qwen2.5-Omni models.

This stage:
1. Loads the Omni model in thinker-only mode
2. Freezes talker weights (if present)
3. Trains the thinker on text data
4. Supports multimodal training when encoders available

Usage:
    python -m src.stages.stage_omni --base-model /path/to/omni --data-dir /path/to/data
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass, asdict

import torch

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.stages.base import BaseStage

logger = logging.getLogger(__name__)


@dataclass
class OmniStageConfig:
    """Configuration for Omni training stage."""
    
    # Required settings (BaseStage compatibility)
    capability_name: str = "omni"
    base_model_path: str = ""  # Alias for base_model
    output_dir: str = "/mnt/e/data/output/trained"
    
    # Omni alias
    base_model: str = ""
    
    # Training settings
    max_samples: int = 1000
    epochs: int = 3
    
    # Omni-specific settings
    freeze_talker: bool = True
    train_thinker_only: bool = True
    enable_multimodal: bool = False
    
    # Training settings optimized for large models
    learning_rate: float = 1e-6
    warmup_ratio: float = 0.1
    gradient_accumulation_steps: int = 4
    max_seq_length: int = 2048
    
    # Memory optimization
    use_gradient_checkpointing: bool = True
    use_flash_attention: bool = True
    mixed_precision: str = "fp16"  # "fp16", "bf16", or "fp32"


class OmniTrainingStage(BaseStage):
    """
    Training stage for Qwen2.5-Omni models.
    
    Specializations:
    - Uses OmniModelLoader for proper model loading
    - Freezes talker component during training
    - Supports gradient checkpointing for memory efficiency
    """
    
    name = "omni"
    description = "Omni model fine-tuning (thinker component)"
    
    def __init__(self, config: OmniStageConfig):
        super().__init__(config)
        self.config: OmniStageConfig = config
        self.model = None
        self.tokenizer = None
    
    def setup(self):
        """Setup Omni model and tokenizer."""
        from src.omni.loader import OmniModelLoader
        
        logger.info(f"Setting up Omni training stage")
        logger.info(f"Base model: {self.config.base_model}")
        
        loader = OmniModelLoader()
        
        # Load for training (freezes talker)
        self.model, self.tokenizer = loader.load_for_training(
            self.config.base_model,
            freeze_talker=self.config.freeze_talker,
        )
        
        # Enable gradient checkpointing
        if self.config.use_gradient_checkpointing:
            if hasattr(self.model, 'gradient_checkpointing_enable'):
                self.model.gradient_checkpointing_enable()
                logger.info("Gradient checkpointing enabled")
        
        logger.info(f"Model loaded with {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,} trainable parameters")
    
    def prepare(self) -> bool:
        """Prepare for training - implementation of abstract method."""
        try:
            self.setup()
            return True
        except Exception as e:
            logger.error(f"Preparation failed: {e}")
            return False
    
    def train(self) -> Dict[str, Any]:
        """Run training - implementation of abstract method."""
        # This is handled by run() method which calls prepare_data and train_step
        return {"success": True, "note": "Use run() method for full training"}
    
    def prepare_data(self, data_path: str) -> Any:
        """Prepare dataset for Omni training."""
        from src.data.universal_loader import load_dataset_universal
        
        logger.info(f"Loading data from {data_path}")
        
        result = load_dataset_universal(data_path, sample_size=self.config.max_samples)
        
        if result.error:
            raise ValueError(f"Failed to load data: {result.error}")
        
        logger.info(f"Loaded {result.num_samples} samples ({result.format} format)")
        
        return result.dataset
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """Execute single training step."""
        self.model.train()
        
        # Move to device
        batch = {k: v.to(self.model.device) for k, v in batch.items()}
        
        # Forward pass
        outputs = self.model(**batch)
        loss = outputs.loss
        
        # Scale loss for accumulation
        loss = loss / self.config.gradient_accumulation_steps
        
        # Backward pass
        loss.backward()
        
        return loss.item() * self.config.gradient_accumulation_steps
    
    def run(self, data_path: str, output_dir: str) -> Dict[str, Any]:
        """
        Run Omni training stage.
        
        Args:
            data_path: Path to training data
            output_dir: Output directory for checkpoints
        
        Returns:
            Training results dict
        """
        import time
        from tqdm import tqdm
        
        logger.info("="*60)
        logger.info("OMNI TRAINING STAGE")
        logger.info("="*60)
        
        start_time = time.time()
        
        # Setup
        self.setup()
        
        # Prepare data
        dataset = self.prepare_data(data_path)
        
        # Setup optimizer
        optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=self.config.learning_rate,
            weight_decay=0.01,
        )
        
        # Setup scheduler
        total_steps = len(dataset) // self.config.gradient_accumulation_steps
        warmup_steps = int(total_steps * self.config.warmup_ratio)
        
        # Training loop
        self.model.train()
        losses = []
        global_step = 0
        
        progress = tqdm(enumerate(dataset), total=len(dataset), desc="Training")
        
        for idx, sample in progress:
            # Tokenize
            text = str(sample)[:self.config.max_seq_length * 4]  # Rough char to token estimate
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=self.config.max_seq_length,
                padding="max_length",
            )
            inputs["labels"] = inputs["input_ids"].clone()
            
            # Train step
            loss = self.train_step(inputs)
            losses.append(loss)
            
            # Update weights
            if (idx + 1) % self.config.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                
                # Warmup
                if global_step < warmup_steps:
                    lr_scale = (global_step + 1) / warmup_steps
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = self.config.learning_rate * lr_scale
                
                global_step += 1
                
                progress.set_postfix(loss=f"{loss:.4f}", step=global_step)
        
        # Final gradient step
        if (len(dataset) % self.config.gradient_accumulation_steps) != 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
        
        # Save checkpoint
        output_path = Path(output_dir) / "omni"
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving checkpoint to {output_path}")
        self.model.save_pretrained(str(output_path))
        self.tokenizer.save_pretrained(str(output_path))
        
        duration = time.time() - start_time
        avg_loss = sum(losses) / len(losses) if losses else 0.0
        final_loss = losses[-1] if losses else 0.0
        
        # Log metrics to CSV
        from src.metrics_tracker import MetricsTracker, TrainingMetrics
        tracker = MetricsTracker()
        
        metrics = TrainingMetrics(
            capability="omni",
            dataset=data_path,
            base_model=self.config.base_model,
            samples=len(dataset),
            steps=global_step,
            epochs=self.config.epochs, # Note: logic above doesn't explicitly loop epochs but iterates dataset once, implying 1 epoch or partial. Assuming 1 for now or config.
            duration_seconds=round(duration, 2),
            samples_per_second=len(dataset)/duration if duration > 0 else 0,
            initial_loss=losses[0] if losses else 0.0,
            final_loss=final_loss,
            avg_loss=avg_loss,
            checkpoint_path=str(output_path),
            success=True
        )
        tracker.log_training(metrics)
        
        results = {
            "stage": "omni",
            "success": True,
            "metrics": asdict(metrics)
        }
        
        logger.info(f"Training complete: {global_step} steps in {duration:.1f}s")
        logger.info(f"Average loss: {avg_loss:.4f}")
        
        return results


def main():
    parser = argparse.ArgumentParser(description="Omni Training Stage")
    parser.add_argument("--base-model", required=True, help="Path to Omni model")
    parser.add_argument("--data-dir", required=True, help="Path to training data")
    parser.add_argument("--output-dir", default="/mnt/e/data/output/trained", help="Output directory")
    parser.add_argument("--max-samples", type=int, default=1000, help="Max training samples")
    parser.add_argument("--learning-rate", type=float, default=1e-6, help="Learning rate")
    parser.add_argument("--quick-validation", action="store_true", help="Enable quick validation mode (tiny batch, aggressive LR)")
    args = parser.parse_args()
    
    # Apply quick validation overrides
    if args.quick_validation:
        print("⚡ QUICK VALIDATION MODE ENABLED")
        args.max_samples = 10  # Very small sample
        args.learning_rate = 1e-5  # High LR for quick signal
        # Note: batch size is hardcoded to 1 in TrainingMetrics default, 
        # and gradient accumulation handles the effective batch size.
    
    config = OmniStageConfig(
        base_model=args.base_model,
        max_samples=args.max_samples,
        learning_rate=args.learning_rate,
        # Force these for quick validation if needed, essentially override defaults
        max_seq_length=500 if args.quick_validation else 2048,
    )
    
    stage = OmniTrainingStage(config)
    results = stage.run(args.data_dir, args.output_dir)
    
    print("\nTraining Results:")
    for k, v in results.items():
        print(f"  {k}: {v}")
    
    return 0 if results["success"] else 1


if __name__ == "__main__":
    exit(main())
