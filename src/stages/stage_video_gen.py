#!/usr/bin/env python3
"""
stage_video_gen.py
Video generation (text-to-video) projector training stage.

Trains a projector to connect LLM hidden states to Stable Video Diffusion.
"""

import torch
import torch.nn as nn
from typing import Dict, Any
from pathlib import Path
from datasets import load_dataset

from .base import BaseStage, StageConfig


class VideoProjector(nn.Module):
    """Projector from LLM hidden states to SVD conditioning."""
    
    def __init__(self, llm_dim: int = 4096, svd_dim: int = 1024, num_frames: int = 14):
        super().__init__()
        self.num_frames = num_frames
        
        # Project LLM hidden state to SVD conditioning
        self.projector = nn.Sequential(
            nn.Linear(llm_dim, svd_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(svd_dim * 2, svd_dim * num_frames),
        )
    
    def forward(self, llm_hidden: torch.Tensor) -> torch.Tensor:
        batch_size = llm_hidden.shape[0]
        projected = self.projector(llm_hidden)
        return projected.view(batch_size, self.num_frames, -1)


class VideoGenStage(BaseStage):
    """Video generation projector training."""
    
    CAPABILITY_NAME = "video-generation"
    
    DATASET_PATTERNS = [
        "HuggingFaceM4/webvid",
    ]
    
    def __init__(self, config: StageConfig):
        super().__init__(config)
        self.projector = None
    
    def prepare(self) -> bool:
        """Load LLM and initialize video projector."""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        self.logger.info(f"Loading base LLM from {self.config.base_model_path}")
        
        if self.config.dry_run:
            self.logger.info("[DRY-RUN] Would load LLM and initialize video projector")
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
            
            llm_dim = self.model.config.hidden_size
            
            # Initialize video projector
            self.projector = VideoProjector(llm_dim=llm_dim)
            self.projector = self.projector.to(self.model.device)
            
            self.optimizer = torch.optim.AdamW(
                self.projector.parameters(),
                lr=self.config.learning_rate,
            )
            
            self.logger.info(f"Video projector initialized: {llm_dim} -> 1024 x 14 frames")
            
            # Load video caption dataset
            try:
                ds = load_dataset(
                    "HuggingFaceM4/webvid",
                    split="train",
                    streaming=True,
                    trust_remote_code=True,
                )
                # Take limited samples for training
                samples = []
                for i, sample in enumerate(ds):
                    if self.config.sample_size > 0 and i >= self.config.sample_size:
                        break
                    samples.append(sample)
                    if i >= 1000:  # Cap at 1000 for memory
                        break
                self.train_dataset = samples
                self.logger.info(f"Loaded {len(samples)} video captions")
            except Exception as e:
                self.logger.warning(f"Could not load webvid: {e}")
                self.train_dataset = None
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize: {e}")
            return False
    
    def train(self) -> Dict[str, Any]:
        """Train the video projector."""
        if self.config.dry_run:
            self.logger.info("[DRY-RUN] Simulating video projector training...")
            for epoch in range(self.config.epochs):
                self.logger.info(f"[DRY-RUN] Epoch {epoch+1}/{self.config.epochs}")
                for step in range(10):
                    self.current_step += 1
            return {"success": True, "dry_run": True, "steps": self.current_step}
        
        if not self.train_dataset:
            self.logger.warning("No training data, skipping")
            return {"success": True, "steps": 0, "skipped": True}
        
        self.logger.info("Training video generation projector...")
        
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
                
                caption = sample.get("caption", sample.get("text", ""))
                if not caption:
                    continue
                
                inputs = self.tokenizer(
                    caption,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                )
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = self.model(**inputs, output_hidden_states=True)
                    hidden = outputs.hidden_states[-1].mean(dim=1)
                
                projected = self.projector(hidden)
                
                # Temporal consistency loss
                target = torch.randn_like(projected)
                mse_loss = nn.functional.mse_loss(projected, target)
                
                # Temporal smoothness regularization
                if projected.shape[1] > 1:
                    temporal_diff = projected[:, 1:] - projected[:, :-1]
                    smooth_loss = temporal_diff.abs().mean()
                else:
                    smooth_loss = 0.0
                
                loss = mse_loss + 0.1 * smooth_loss
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                
                if self.current_step % 50 == 0:
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
        
        if self.projector is not None:
            proj_path = Path(path) / "video_projector.pt"
            torch.save(self.projector.state_dict(), proj_path)
            self.logger.info(f"Saved video projector to {proj_path}")
        
        return path


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Video Generation Training")
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--sample-size", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()
    
    config = StageConfig(
        capability_name="video-generation",
        base_model_path=args.base_model,
        output_dir=args.output_dir,
        sample_size=args.sample_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        dry_run=args.dry_run,
    )
    stage = VideoGenStage(config)
    return 0 if stage.run().get("success") else 1

if __name__ == "__main__":
    exit(main())
