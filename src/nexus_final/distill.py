import torch
import torch.nn as nn
import os
import sys
import shutil
import json
import signal
from torch.utils.data import DataLoader
from .loss_functions import ActivationAnchoringLoss
from ..student.core import NexusStudentCore
from typing import Dict, Any
from tqdm import tqdm

class NexusTrainer:
    """
    Main training loop for Nexus Distillation.
    Implements:
    - Activation Anchoring (Protected Subspaces)
    - Curriculum Learning for Alpha
    - Dynamic Recovery Step (Loss Spike Rollback)
    - Signal Handling (Pause/Resume)
    - Gradient Norm Monitoring
    - Router Entropy Loss (Diversity)
    """
    def __init__(
        self,
        student: NexusStudentCore,
        adapters: Dict[str, Any],
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        device: str = "cuda",
        config: Dict = {}
    ):
        self.student = student
        self.adapters = adapters
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.device = device
        self.config = config
        
        # --- Loss Configuration ---
        self.base_alpha = config.get("alpha", 1.0)
        self.loss_fn = ActivationAnchoringLoss(
            alpha_ce=config.get("alpha_ce", 1.0),
            alpha_hidden=config.get("alpha_hidden", 0.5),
            alpha_critical=config.get("critical_weight", 10.0)
        )
        
        # --- Recovery & Safety Config ---
        self.max_grad_norm = config.get("max_grad_norm", 1.0)
        self.loss_spike_threshold = config.get("loss_spike_threshold", 2.0)
        self.checkpoint_dir = config.get("checkpoint_dir", "checkpoints")
        self.warmup_epochs = config.get("warmup_epochs", 5)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        self.global_step = 0
        self.prev_loss = float('inf')
        self.best_val_loss = float('inf')
        
        # Register Signal Handler for Graceful Shutdown
        signal.signal(signal.SIGINT, self.handle_signal)
        signal.signal(signal.SIGTERM, self.handle_signal)

    def handle_signal(self, signum, frame):
        print(f"\n[Trainer] Signal {signum} received. Saving checkpoint and exiting...")
        self.save_checkpoint("interrupted")
        sys.exit(0)

    def save_checkpoint(self, tag="latest"):
        path = os.path.join(self.checkpoint_dir, f"checkpoint_{tag}.pt")
        torch.save({
            'step': self.global_step,
            'student_state': self.student.state_dict(),
            'adapter_states': {k: v.state_dict() for k, v in self.adapters.items()},
            'optimizer': self.optimizer.state_dict(),
            'prev_loss': self.prev_loss,
            'best_val_loss': self.best_val_loss
        }, path)
        print(f"[Checkpoint] Saved to {path}")
        return path

    def load_checkpoint(self, tag="latest"):
        path = os.path.join(self.checkpoint_dir, f"checkpoint_{tag}.pt")
        if os.path.exists(path):
            print(f"[Recovery] Rolling back to {tag}...")
            ckpt = torch.load(path, map_location=self.device)
            self.student.load_state_dict(ckpt['student_state'])
            for k, v in self.adapters.items():
                v.load_state_dict(ckpt['adapter_states'][k])
            self.optimizer.load_state_dict(ckpt['optimizer'])
            self.global_step = ckpt['step']
            self.prev_loss = ckpt['prev_loss']
            self.best_val_loss = ckpt.get('best_val_loss', float('inf'))
            return True
        return False

    def get_alpha(self, epoch):
        """Curriculum Learning for Alpha: Start low, ramp up to full strength."""
        if epoch < self.warmup_epochs:
            return self.base_alpha * (epoch / self.warmup_epochs)
        return self.base_alpha

    def train(self, epochs: int):
        self.student.train()
        for adapter in self.adapters.values():
            adapter.train()
            
        # Initial Safe State
        self.save_checkpoint("recovery_safe")
        
        for epoch in range(epochs):
            # Dynamic Alpha
            current_alpha = self.get_alpha(epoch)
            self.loss_fn.alpha_hidden = current_alpha # Update loss weight dynamically
            
            print(f"\n=== Epoch {epoch+1}/{epochs} | Alpha: {current_alpha:.4f} ===")
            
            with tqdm(self.train_loader, desc=f"Epoch {epoch+1}", dynamic_ncols=True) as pbar:
                for batch in pbar:
                    try:
                        loss_metrics = self.training_step(batch)
                        
                        # Update Progress Bar
                        pbar.set_postfix(loss_metrics)
                        
                        # Periodic Validation
                        if self.global_step % 100 == 0:
                            val_loss = self.evaluate()
                            if val_loss < self.best_val_loss:
                                self.best_val_loss = val_loss
                                self.save_checkpoint("best")
                            elif val_loss > self.best_val_loss * 1.05:
                                print(f"[Alert] Validation Loss Divergence: {val_loss:.4f} vs Best {self.best_val_loss:.4f}")
                                print("[Recovery] Rolling back to best checkpoint and increasing anchors...")
                                self.load_checkpoint("best")
                                # Increase anchor weight to stabilize
                                self.loss_fn.alpha_critical *= 1.2
                                print(f"[Recovery] Anchor Weight increased to {self.loss_fn.alpha_critical:.2f}")
                                
                    except RuntimeError as e:
                        if "out of memory" in str(e):
                            print("[Hardware] OOM Detected! Attempting recovery...")
                            torch.cuda.empty_cache()
                            # In a real loop we might skip batch or reduce BS here, 
                            # but for now we just clear cache and hope next batch fits
                            continue
                        else:
                            raise e

            self.save_checkpoint(f"epoch_{epoch+1}")

    def training_step(self, batch):
        input_ids = batch['input_ids'].to(self.device)
        teacher_feats = {k: v.to(self.device) for k, v in batch['teacher_features'].items()}
        labels = batch['labels'].to(self.device)
        
        self.optimizer.zero_grad()
        
        # 1. Adapter Projection
        student_compatible_feats = {}
        for name, adapter in self.adapters.items():
            if name in teacher_feats:
                student_compatible_feats[name] = adapter(teacher_feats[name])
        
        # 2. Student Forward
        outputs = self.student(
            input_ids=input_ids,
            adapter_hidden_states=student_compatible_feats,
            labels=labels,
            output_router_logits=True # Needed for entropy loss
        )
        
        # 3. Loss Calculation
        # Distillation Loss (includes CE + Hidden + Anchoring)
        # Assuming we match the first active adapter feature for simplicity in this template
        # Real impl would iterate/avg
        primary_adapter = list(student_compatible_feats.keys())[0] if student_compatible_feats else None
        
        loss_distill = 0
        if primary_adapter:
             loss_distill = self.loss_fn(
                student_logits=outputs['logits'],
                teacher_logits=batch['teacher_logits'].to(self.device), # Assuming teacher logits in batch
                student_states=outputs['hidden_states'],
                teacher_states=student_compatible_feats[primary_adapter]
            )
        
        # Router Entropy Loss (Prevent Collapse)
        loss_entropy = 0
        if hasattr(outputs, 'router_logits') and outputs['router_logits'] is not None:
            probs = torch.softmax(outputs['router_logits'], dim=-1)
            # Minimize Negative Entropy (Maximize Diversity)
            loss_entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1).mean()
            
        # Entropy Weight (Beta) - could be dynamic, hardcoded 0.01 for now
        beta_entropy = self.config.get("beta_entropy", 0.01)
        
        total_loss = loss_distill + (beta_entropy * loss_entropy)
        
        # Spike Detection
        if total_loss.item() > self.prev_loss * self.loss_spike_threshold and self.global_step > 10:
            print(f"[Alert] Loss Spike: {total_loss.item():.4f}. Rolling back...")
            self.load_checkpoint("recovery_safe")
            return {"loss": "ROLLBACK"}

        total_loss.backward()
        
        # Gradient Norm Monitoring
        grad_norm = torch.nn.utils.clip_grad_norm_(self.student.parameters(), self.max_grad_norm)
        # Monitoring hook for tensorboard/logging could go here
            
        self.optimizer.step()
        
        self.prev_loss = total_loss.item()
        self.global_step += 1
        
        if self.global_step % 10 == 0:
            self.save_checkpoint("recovery_safe")
            
        return {
            "loss": f"{total_loss.item():.4f}", 
            "gnorm": f"{grad_norm:.2f}"
        }

    def evaluate(self):
        self.student.eval()
        total_val_loss = 0
        steps = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                # Simplified eval forward
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                outputs = self.student(input_ids=input_ids, labels=labels)
                total_val_loss += outputs['loss'].item()
                steps += 1
        
        self.student.train()
        return total_val_loss / max(steps, 1)
