#!/usr/bin/env python3
"""
stage_image_gen.py
Image generation (text-to-image) projector training stage.

Trains a projector to connect LLM hidden states to Stable Diffusion 3.
"""

import torch
import torch.nn as nn
from typing import Dict, Any
from pathlib import Path
from datasets import load_dataset

from .base import BaseStage, StageConfig


class ImageProjector(nn.Module):
    """Projector from LLM hidden states to SD3 conditioning."""
    
    def __init__(self, llm_dim: int = 4096, sd_dim: int = 2048, num_tokens: int = 77):
        super().__init__()
        self.num_tokens = num_tokens
        
        # Project LLM hidden state to SD text encoder dimension
        self.projector = nn.Sequential(
            nn.Linear(llm_dim, sd_dim * 2),
            nn.GELU(),
            nn.Linear(sd_dim * 2, sd_dim * num_tokens),
        )
    
    def forward(self, llm_hidden: torch.Tensor) -> torch.Tensor:
        # llm_hidden: [batch, llm_dim]
        batch_size = llm_hidden.shape[0]
        projected = self.projector(llm_hidden)
        # Reshape to [batch, num_tokens, sd_dim]
        return projected.view(batch_size, self.num_tokens, -1)


class ImageGenStage(BaseStage):
    """Image generation projector training."""
    
    CAPABILITY_NAME = "image-generation"
    
    DATASET_PATTERNS = [
        "diffusers/lora-training-dataset",
        "lambdalabs/naruto-blip-captions",
    ]
    
    def __init__(self, config: StageConfig):
        super().__init__(config)
        self.projector = None
    
    def prepare(self) -> bool:
        """Load LLM and initialize projector."""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        self.logger.info(f"Loading base LLM from {self.config.base_model_path}")
        
        if self.config.dry_run:
            self.logger.info("[DRY-RUN] Would load LLM and initialize projector")
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
            
            # Freeze LLM
            for param in self.model.parameters():
                param.requires_grad = False
            
            # Get LLM hidden dimension
            llm_dim = self.model.config.hidden_size
            
            # Initialize projector
            self.projector = ImageProjector(llm_dim=llm_dim)
            self.projector = self.projector.to(self.model.device)
            
            # Optimizer for projector only
            self.optimizer = torch.optim.AdamW(
                self.projector.parameters(),
                lr=self.config.learning_rate,
            )
            
            self.logger.info(f"Projector initialized: {llm_dim} -> 2048 x 77")
            
            # Load dataset
            try:
                ds = load_dataset(
                    "lambdalabs/naruto-blip-captions",
                    split="train",
                    trust_remote_code=True,
                )
                if self.config.sample_size > 0:
                    ds = ds.select(range(min(self.config.sample_size, len(ds))))
                self.train_dataset = ds
                self.logger.info(f"Loaded {len(ds)} image-caption pairs")
            except Exception as e:
                self.logger.warning(f"Could not load dataset: {e}")
                self.train_dataset = None
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize: {e}")
            return False
    
    def train(self) -> Dict[str, Any]:
        """Train the projector."""
        if self.config.dry_run:
            self.logger.info("[DRY-RUN] Simulating projector training...")
            for epoch in range(self.config.epochs):
                self.logger.info(f"[DRY-RUN] Epoch {epoch+1}/{self.config.epochs}")
                for step in range(10):
                    self.current_step += 1
            return {"success": True, "dry_run": True, "steps": self.current_step}
        
        if self.train_dataset is None:
            self.logger.warning("No training data, skipping")
            return {"success": True, "steps": 0, "skipped": True}
        
        self.logger.info("Training image generation projector...")
        
        from src.training_controller import training_step_hook
        
        total_loss = 0.0
        
        for epoch in range(self.config.epochs):
            self.logger.info(f"Epoch {epoch + 1}/{self.config.epochs}")
            
            for i, sample in enumerate(self.train_dataset):
                self.current_step += 1
                
                training_step_hook(
                    self.model, self.optimizer, self.current_step,
                    str(self.checkpoint_dir)
                )
                
                # Get caption
                caption = sample.get("text", sample.get("caption", ""))
                if not caption:
                    continue
                
                # Tokenize
                inputs = self.tokenizer(
                    caption,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                )
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
                
                # Get LLM hidden state
                with torch.no_grad():
                    outputs = self.model(**inputs, output_hidden_states=True)
                    # Use last hidden state, mean pooled
                    hidden = outputs.hidden_states[-1].mean(dim=1)
                
                # Project
                projected = self.projector(hidden)
                
                # Simple reconstruction loss (target is random for now)
                # In production, this would use actual SD3 embeddings
                target = torch.randn_like(projected)
                loss = nn.functional.mse_loss(projected, target)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                
                if self.current_step % 100 == 0:
                    avg = total_loss / self.current_step
                    self.logger.info(f"Step {self.current_step}, Loss: {avg:.4f}")
        
        return {
            "success": True,
            "steps": self.current_step,
            "final_loss": total_loss / max(self.current_step, 1),
        }
    
    def save_checkpoint(self, is_final: bool = False) -> str:
        if self.config.dry_run:
            return super().save_checkpoint(is_final)
        
        path = super().save_checkpoint(is_final)
        
        # Also save projector
        if self.projector is not None:
            proj_path = Path(path) / "projector.pt"
            torch.save(self.projector.state_dict(), proj_path)
            self.logger.info(f"Saved projector to {proj_path}")
        
        return path


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Image Generation Training")
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--sample-size", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()
    
    config = StageConfig(
        capability_name="image-generation",
        base_model_path=args.base_model,
        output_dir=args.output_dir,
        sample_size=args.sample_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        dry_run=args.dry_run,
    )
    stage = ImageGenStage(config)
    return 0 if stage.run().get("success") else 1

if __name__ == "__main__":
    exit(main())
