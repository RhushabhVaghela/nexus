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

from src.stages.base import BaseStage, StageConfig
from src.utils.repetition import PromptRepetitionEngine

logger = logging.getLogger(__name__)


@dataclass
class OmniStageConfig(StageConfig):
    """Configuration for Omni training stage."""
    
    # Required settings (BaseStage compatibility)
    # capability_name is inherited from StageConfig
    
    # Omni-specific settings
    freeze_talker: bool = True
    train_thinker_only: bool = True
    enable_multimodal: bool = False
    
    # Training settings optimized for large models
    warmup_ratio: float = 0.1
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
        logger.info(f"Base model: {self.config.base_model_path}")
        
        loader = OmniModelLoader()
        
        # Load for training (freezes talker)
        # Pass repetition factors if supported by loader/model
        self.model, self.tokenizer = loader.load_for_training(
            self.config.base_model_path,
            freeze_talker=self.config.freeze_talker,
            visual_repetition_factor=self.config.repetition_factor,
            audio_repetition_factor=self.config.repetition_factor
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
        logger.info(f"Preparing data for Omni stage...")
        
        if data_path == "dynamic" or not Path(data_path).exists():
            self.logger.info("Using dynamic dataset discovery for Omni stage")
            return self.load_dynamic_datasets()
            
        from src.data.universal_loader import load_dataset_universal
        logger.info(f"Loading data from {data_path}")
        result = load_dataset_universal(data_path, sample_size=self.config.sample_size)
        
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
    
    def run(self, data_path: str = "dynamic") -> Dict[str, Any]:
        """
        Run Omni training stage.
        """
        import time
        from tqdm import tqdm
        
        logger.info("="*60)
        logger.info("OMNI TRAINING STAGE")
        logger.info("="*60)
        
        if self.config.repetition_factor > 1:
            logger.info(f"Using Prompt Repetition: {self.config.repetition_factor}x ({self.config.repetition_style})")
        
        start_time = time.time()
        
        # Phase 1: Prepare
        if not self.prepare():
            return {"success": False, "error": "Preparation failed"}
            
        # Prepare data
        dataset = self.prepare_data(data_path)
        if not dataset:
             return {"success": False, "error": "No data found"}
        
        # Setup optimizer
        optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=self.config.learning_rate,
            weight_decay=0.01,
        )
        self.optimizer = optimizer
        
        # Setup scheduler
        total_steps = len(dataset) // self.config.gradient_accumulation_steps
        warmup_steps = int(total_steps * self.config.warmup_ratio)
        
        # Training loop
        self.model.train()
        losses = []
        global_step = 0
        
        progress = tqdm(enumerate(dataset), total=len(dataset), desc="Training")
        
        for idx, sample in progress:
            # Tokenize and process sample
            text = str(sample)
            
            # Apply Repetition if enabled
            if self.config.repetition_factor > 1:
                text = PromptRepetitionEngine.apply_repetition(
                    text,
                    factor=self.config.repetition_factor,
                    style=self.config.repetition_style
                )
            
            text = text[:self.config.max_seq_length * 4]  # Rough char to token estimate
            
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
                self.current_step = global_step
                
                progress.set_postfix(loss=f"{loss:.4f}", step=global_step)
        
        # Save final checkpoint
        output_path = self.save_checkpoint(is_final=True)
        
        duration = time.time() - start_time
        avg_loss = sum(losses) / len(losses) if losses else 0.0
        final_loss = losses[-1] if losses else 0.0
        
        # Log metrics to CSV
        from src.metrics_tracker import MetricsTracker, TrainingMetrics
        tracker = MetricsTracker()
        
        metrics = TrainingMetrics(
            capability="omni",
            dataset=data_path,
            base_model=self.config.base_model_path,
            samples=len(dataset),
            steps=global_step,
            epochs=self.config.epochs,
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
    parser.add_argument("--sample-size", type=int, default=1000, help="Max training samples")
    parser.add_argument("--learning-rate", type=float, default=1e-6, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--quick-validation", action="store_true", help="Enable quick validation mode")
    
    # Repetition args
    parser.add_argument("--repetition-factor", type=int, default=1, help="Prompt repetition factor")
    parser.add_argument("--repetition-style", type=str, default="baseline", help="Repetition style")
    
    args = parser.parse_args()
    
    # Apply quick validation overrides
    if args.quick_validation:
        print("⚡ QUICK VALIDATION MODE ENABLED")
        args.sample_size = 10
        args.learning_rate = 1e-5
    
    config = OmniStageConfig(
        capability_name="omni",
        base_model_path=args.base_model,
        output_dir=args.output_dir,
        sample_size=args.sample_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        max_seq_length=500 if args.quick_validation else 2048,
        repetition_factor=args.repetition_factor,
        repetition_style=args.repetition_style,
    )
    
    stage = OmniTrainingStage(config)
    results = stage.run(args.data_dir)
    
    print("\nTraining Results:")
    for k, v in results.items():
        print(f"  {k}: {v}")
    
    return 0 if results["success"] else 1


if __name__ == "__main__":
    exit(main())
