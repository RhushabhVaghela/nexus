import torch
import os
import shutil
from torch.utils.data import DataLoader
from .loss import ActivationAnchoringLoss
# from .loss_functions import ActivationAnchoringLoss # Removed duplicated logic
from nexus_core.student.core import NexusStudentCore
from nexus_core.adapters.reasoning_adapter import ReasoningAdapter
from typing import Dict, Any

class NexusTrainer:
    def __init__(
        self,
        student: NexusStudentCore,
        adapters: Dict[str, Any],
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        device: str = "cuda",
        config: Dict = {}
    ):
        self.student = student
        self.adapters = adapters
        self.train_loader = train_loader
        self.optimizer = optimizer
        self.device = device
        self.config = config
        
        self.loss_fn = ActivationAnchoringLoss(
            alpha_mse=config.get("alpha", 1.0),
            beta_var=config.get("beta", 0.5),
            critical_weight=config.get("critical_weight", 10.0)
        )
        
        # Recovery Step Config
        self.max_grad_norm = config.get("max_grad_norm", 1.0)
        self.loss_spike_threshold = config.get("loss_spike_threshold", 2.0) # 2x previous loss
        self.checkpoint_dir = config.get("checkpoint_dir", "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        self.global_step = 0
        self.prev_loss = float('inf')

    def save_checkpoint(self, tag="latest"):
        path = os.path.join(self.checkpoint_dir, f"checkpoint_{tag}.pt")
        torch.save({
            'step': self.global_step,
            'student_state': self.student.state_dict(),
            'adapter_states': {k: v.state_dict() for k, v in self.adapters.items()},
            'optimizer': self.optimizer.state_dict(),
            'prev_loss': self.prev_loss
        }, path)
        return path

    def load_checkpoint(self, tag="latest"):
        path = os.path.join(self.checkpoint_dir, f"checkpoint_{tag}.pt")
        if os.path.exists(path):
            print(f"[Recovery] Rolling back to {tag}...")
            ckpt = torch.load(path)
            self.student.load_state_dict(ckpt['student_state'])
            for k, v in self.adapters.items():
                v.load_state_dict(ckpt['adapter_states'][k])
            self.optimizer.load_state_dict(ckpt['optimizer'])
            self.global_step = ckpt['step']
            self.prev_loss = ckpt['prev_loss']
            return True
        return False

    def train_epoch(self):
        self.student.train()
        for adapter in self.adapters.values():
            adapter.train()
            
        # Save initial state for recovery
        self.save_checkpoint("recovery_safe")
        
        for batch_idx, batch in enumerate(self.train_loader):
            # batch: input_ids, teacher_features (dict), labels
            input_ids = batch['input_ids'].to(self.device)
            teacher_feats = {k: v.to(self.device) for k, v in batch['teacher_features'].items()}
            labels = batch['labels'].to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward Pass (Student)
            # 1. Project Teacher Features via Adapters
            student_compatible_feats = {}
            for name, adapter in self.adapters.items():
                if name in teacher_feats:
                    # Assuming Reasoning Adapter for now
                    student_compatible_feats[name] = adapter(teacher_feats[name])
            
            # 2. Student Core Forward
            outputs = self.student(
                input_ids=input_ids,
                adapter_hidden_states=student_compatible_feats,
                labels=labels
            )
            
            # Loss Calculation
            # CE Loss
            ce_loss = outputs['loss']
            
            # Distillation Loss (Activation Anchoring (protected subspaces))
            distill_loss = 0
            for name, s_feat in student_compatible_feats.items():
                # We need to match student projected features to... what?
                # Ah, Distillation usually matches Student Hidden States to Teacher Hidden States.
                # But here, the Adapters Project Teacher -> Student Space.
                # So we want the Adapter Output to match... the Student's Internal State?
                # Or do we want the Student's Internal State to match the Adapter Output?
                # "Activation-Guided Consolidation": The Student learns to mimic the thought process.
                # So Student Hidden States ~ Adapter Output (which is Teacher Projected).
                
                # Use the last hidden state of student? Or layer-wise?
                # For simplicity: Match Last Hidden State
                distill_loss += self.loss_fn(
                    outputs['hidden_states'], # Student
                    s_feat, # Teacher (Projected)
                    anchoring_mask=None # In full impl, pass mask
                )
            
            total_loss = ce_loss + distill_loss
            
            # Check for Loss Spike (Recovery Step)
            if total_loss.item() > self.prev_loss * self.loss_spike_threshold and self.global_step > 10:
                print(f"[Alert] Loss Spike detected: {total_loss.item():.4f} > {self.prev_loss:.4f}")
                print("[Recovery] Initiating Rollback...")
                self.load_checkpoint("recovery_safe")
                # Skip this batch or reduce LR?
                # Simple: Skip batch and continue
                continue
                
            # Backward
            total_loss.backward()
            
            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(self.student.parameters(), self.max_grad_norm)
            
            self.optimizer.step()
            
            # Update state
            self.prev_loss = total_loss.item()
            self.global_step += 1
            
            if batch_idx % 10 == 0:
                print(f"Step {self.global_step} | Loss: {total_loss.item():.4f} (CE: {ce_loss.item():.4f})")
                self.save_checkpoint("recovery_safe") # Update safe point if successful
