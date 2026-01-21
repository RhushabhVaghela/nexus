#!/usr/bin/env python3
"""
base.py
Base class for all capability training stages.

All stage scripts inherit from BaseStage and implement:
- prepare(): Validate and load datasets
- train(): Run the training loop
- save(): Save checkpoints and model
"""

import os
import sys
import json
import torch
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.training_controller import (
    setup_signal_handlers,
    training_step_hook,
    check_pause_state,
    scan_and_extract_datasets,
)
from src.metrics_tracker import ALL_DATASETS
from src.data.universal_loader import load_dataset_universal


@dataclass
class StageConfig:
    """Configuration for a training stage."""
    
    capability_name: str
    base_model_path: str
    output_dir: str
    datasets: List[str] = field(default_factory=list)
    batch_size: int = 1
    epochs: int = 3
    learning_rate: float = 2e-5
    sample_size: int = 0  # 0 = all samples
    gradient_accumulation_steps: int = 4
    save_steps: int = 500
    eval_steps: int = 100
    dry_run: bool = False
    # Distillation options (optional)
    enable_distillation: bool = False
    teacher_model_path: Optional[str] = None
    distillation_alpha: float = 0.5  # Weight for soft targets vs hard labels
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "capability_name": self.capability_name,
            "base_model_path": self.base_model_path,
            "output_dir": self.output_dir,
            "datasets": self.datasets,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "sample_size": self.sample_size,
        }


class BaseStage(ABC):
    """Base class for all training stages."""
    
    def __init__(self, config: StageConfig):
        self.config = config
        self.logger = self._setup_logger()
        self.model = None
        self.tokenizer = None
        self.train_dataset = None
        self.optimizer = None
        self.current_step = 0
        
        # Create output directories
        self.checkpoint_dir = Path(config.output_dir) / "checkpoints" / config.capability_name
        self.log_dir = Path(config.output_dir) / "logs" / config.capability_name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logging for the stage."""
        logger = logging.getLogger(f"stage_{self.config.capability_name}")
        logger.setLevel(logging.INFO)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter(
            f"[%(asctime)s] [%(name)s] %(levelname)s: %(message)s"
        )
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        
        return logger
    
    def load_teacher_model(self):
        """
        Load teacher model for distillation (if enabled).
        Teacher model is used to generate soft targets.
        """
        if not self.config.enable_distillation:
            return None
            
        if not self.config.teacher_model_path:
            self.logger.warning("Distillation enabled but no teacher_model_path specified")
            return None
            
        from transformers import AutoModelForCausalLM
        
        self.logger.info(f"Loading teacher model from {self.config.teacher_model_path}")
        teacher = AutoModelForCausalLM.from_pretrained(
            self.config.teacher_model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        teacher.eval()  # Teacher is always in eval mode
        return teacher
    
    def compute_distillation_loss(
        self, 
        student_logits: torch.Tensor, 
        teacher_logits: torch.Tensor, 
        temperature: float = 2.0
    ) -> torch.Tensor:
        """
        Compute KL-divergence loss for distillation (soft targets).
        
        Args:
            student_logits: Output logits from student model
            teacher_logits: Output logits from teacher model
            temperature: Softmax temperature (higher = softer targets)
            
        Returns:
            KL divergence loss
        """
        import torch.nn.functional as F
        
        student_probs = F.log_softmax(student_logits / temperature, dim=-1)
        teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
        
        loss = F.kl_div(student_probs, teacher_probs, reduction="batchmean")
        # Scale by T^2 as per Hinton et al.
        return loss * (temperature ** 2)
    
    def load_dynamic_datasets(self) -> Any:
        """
        Dynamically load datasets based on the capability name.
        Looks up paths in ALL_DATASETS and combines them.
        """
        from datasets import concatenate_datasets
        
        paths = ALL_DATASETS.get(self.config.capability_name, [])
        if not paths:
            self.logger.warning(f"No dynamic datasets found for capability: {self.config.capability_name}")
            return None
            
        self.logger.info(f"Loading {len(paths)} datasets for {self.config.capability_name}...")
        
        datasets = []
        for path in paths:
            if not Path(path).exists():
                continue
            try:
                res = load_dataset_universal(path, sample_size=self.config.sample_size)
                if res.dataset:
                    datasets.append(res.dataset)
                    self.logger.info(f"  ✓ Loaded {path} ({len(res.dataset)} samples)")
            except Exception as e:
                self.logger.warning(f"  ✗ Failed to load {path}: {e}")
                
        if not datasets:
            return None
            
        if len(datasets) == 1:
            return datasets[0]
            
        return concatenate_datasets(datasets)

    @abstractmethod
    def prepare(self) -> bool:
        """
        Prepare for training.
        
        - Load base model and tokenizer
        - Load and process datasets
        - Validate requirements
        
        Returns:
            True if preparation successful, False otherwise
        """
        pass
    
    @abstractmethod
    def train(self) -> Dict[str, Any]:
        """
        Run the training loop.
        
        Returns:
            Dictionary with training results (loss, metrics, etc.)
        """
        pass
    
    def save_checkpoint(self, is_final: bool = False) -> str:
        """Save a checkpoint."""
        if self.config.dry_run:
            self.logger.info("[DRY-RUN] Would save checkpoint")
            return str(self.checkpoint_dir / "dry_run_checkpoint")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_name = f"checkpoint_step{self.current_step}_{timestamp}"
        if is_final:
            checkpoint_name = f"final_model_{timestamp}"
        
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        
        self.logger.info(f"Saving checkpoint to {checkpoint_path}")
        
        # Save model
        if self.model is not None:
            self.model.save_pretrained(str(checkpoint_path))
        
        # Save tokenizer
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(str(checkpoint_path))
        
        # Save optimizer state
        if self.optimizer is not None:
            torch.save(
                self.optimizer.state_dict(),
                checkpoint_path / "optimizer.pt"
            )
        
        # Save training state
        training_state = {
            "step": self.current_step,
            "config": self.config.to_dict(),
            "timestamp": timestamp,
        }
        with open(checkpoint_path / "training_state.json", "w") as f:
            json.dump(training_state, f, indent=2)
        
        return str(checkpoint_path)
    
    def run(self) -> Dict[str, Any]:
        """
        Main entry point to run the stage.
        
        Returns:
            Dictionary with stage results
        """
        self.logger.info(f"=" * 60)
        self.logger.info(f"Starting stage: {self.config.capability_name}")
        self.logger.info(f"=" * 60)
        
        if self.config.dry_run:
            self.logger.info("[DRY-RUN MODE] No actual training will occur")
        
        # Setup signal handlers for pause/resume
        setup_signal_handlers()
        
        # Phase 1: Prepare
        self.logger.info("Phase 1: Preparing...")
        if not self.prepare():
            self.logger.error("Preparation failed!")
            return {"success": False, "error": "Preparation failed"}
        
        # Phase 2: Train
        self.logger.info("Phase 2: Training...")
        results = self.train()
        
        # Phase 3: Save final checkpoint
        self.logger.info("Phase 3: Saving final model...")
        final_path = self.save_checkpoint(is_final=True)
        results["final_checkpoint"] = final_path
        
        self.logger.info(f"Stage complete: {self.config.capability_name}")
        self.logger.info(f"Results: {json.dumps(results, indent=2)}")
        
        return results


class TextCapabilityStage(BaseStage):
    """Base class for text-only capabilities (CoT, reasoning, tools, etc.)."""
    
    def prepare(self) -> bool:
        """Load model and datasets for text-only training."""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        self.logger.info(f"Loading model from {self.config.base_model_path}")
        
        if self.config.dry_run:
            self.logger.info("[DRY-RUN] Would load model and tokenizer")
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
            
            # Ensure pad token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.logger.info("Model loaded successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            return False
    
    def train(self) -> Dict[str, Any]:
        """Run training loop with training controller integration."""
        if self.config.dry_run:
            self.logger.info("[DRY-RUN] Simulating training...")
            for epoch in range(self.config.epochs):
                for step in range(10):  # Simulate 10 steps per epoch
                    self.current_step += 1
                    self.logger.info(f"[DRY-RUN] Epoch {epoch+1}, Step {step+1}")
            return {
                "success": True,
                "dry_run": True,
                "simulated_steps": self.current_step,
            }
        
        # Real training would go here
        # For now, return a template that specific stages will override
        return {
            "success": True,
            "steps": self.current_step,
            "final_loss": 0.0,
        }
