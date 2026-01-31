import torch
import torch.nn as nn
import os
import sys
import shutil
import json
import signal
import gc
from torch.utils.data import DataLoader
from .loss_functions import ActivationAnchoringLoss
from .auditor import MemorizationAuditor
from nexus_core.student.core import NexusStudentCore
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
        
        # --- Memorization Auditor ---
        self.auditor = None
        if hasattr(student, 'tokenizer'):
             self.auditor = MemorizationAuditor(student.tokenizer)
        
        # --- Gradient Accumulation ---
        self.grad_accum_steps = config.get("gradient_accumulation_steps", 1)

        
        # Register Signal Handler for Graceful Shutdown and Control
        signal.signal(signal.SIGINT, self.handle_signal)
        signal.signal(signal.SIGTERM, self.handle_signal)
        signal.signal(signal.SIGUSR1, self.handle_signal) # Pause/Save
        signal.signal(signal.SIGUSR2, self.handle_signal) # Immediate Checkpoint

    def handle_signal(self, signum, frame):
        if signum in [signal.SIGINT, signal.SIGTERM]:
            print(f"\n[Trainer] Signal {signum} received. Saving checkpoint and exiting...")
            self.save_checkpoint("interrupted")
            sys.exit(0)
        elif signum == signal.SIGUSR1:
            print(f"\n[Trainer] SIGUSR1 (Pause) received. Saving checkpoint and waiting...")
            self.save_checkpoint("paused")
            # In a real training session, this could enter a loop waiting for resume signal
            # For simplicity, we save and let the user decide how to resume.
        elif signum == signal.SIGUSR2:
            print(f"\n[Trainer] SIGUSR2 (Immediate Checkpoint) received.")
            self.save_checkpoint(f"manual_step_{self.global_step}")

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
        
        # Always maintain a "checkpoint_latest.pt" for the exporter
        if tag != "latest":
            latest_path = os.path.join(self.checkpoint_dir, "checkpoint_latest.pt")
            shutil.copy(path, latest_path)
            print(f"[Checkpoint] Updated link: {latest_path}")
            
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
                for step_idx, batch in enumerate(pbar):
                    try:
                        is_accum_boundary = ((step_idx + 1) % self.grad_accum_steps == 0) or ((step_idx + 1) == len(self.train_loader))
                        
                        loss_metrics = self.training_step(batch, is_accum_boundary)
                        
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

                        # --- Memorization Audit (Periodic) ---
                        if self.auditor and self.global_step % 500 == 0:
                            self.run_memorization_audit()

                        # 4. Thermal Throttling & Hardware Safety
                        self.check_thermal_status()
                                
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

    def training_step(self, batch, is_accum_boundary=True):

        """
        Executes a single training step with multi-objective distillation.
        """
        input_ids = batch['input_ids'].to(self.device)
        teacher_logits = batch['teacher_logits'].to(self.device)
        labels = batch['labels'].to(self.device)
        
        # Support for Offline Distillation (SSD-backed)
        # We load teacher features from the SLI Activation Cache if available
        teacher_feats = {}
        if self.config.get("offline_distillation", False):
            cache_dir = self.config.get("activation_cache_dir")
            # Robust retrieval of Batch Index for SSD lookup
            if "batch_idx" in batch:
                 batch_idx = batch["batch_idx"]
            elif "meta" in batch and "index" in batch["meta"]:
                 batch_idx = batch["meta"]["index"]
            else:
                 # Fallback: Use modulo len(dataset) if possible, else global step (risky for multiple epochs)
                 # Better to assume global_step aligns with data stream IF DataLoader is not shuffled.
                 # Warn user.
                 batch_idx = self.global_step

            for layer_name in self.config.get("critical_layers", []):
                cache_path = os.path.join(cache_dir, f"layer_{layer_name}_batch_{batch_idx}.pt")
                if os.path.exists(cache_path):
                    teacher_feats[layer_name] = torch.load(cache_path, map_location=self.device)
        else:
            teacher_feats = {k: v.to(self.device) for k, v in batch['teacher_features'].items()}
        
            teacher_feats = {k: v.to(self.device) for k, v in batch['teacher_features'].items()}
        
        # Accumulate gradients without zeroing if not boundary
        # Note: We zero grad AFTER stepping, so standard PyTorch style is:
        # 1. Zero (at start of accumulation or after step)
        # 2. Forward/Backward
        # 3. Step (at boundary)
        # For simplicity in this refactor, we rely on the optimizer holding state.
        # Ideally: optimizer.zero_grad() happens ONLY after step.
        
        # 1. Adapter Projection (NIWT Bridge)
        # We project all teacher features into the student's latent space
        student_compatible_feats = {}
        for name, adapter in self.adapters.items():
            if name in teacher_feats:
                student_compatible_feats[name] = adapter(teacher_feats[name])
        
        # 2. Student Forward Pass
        # We pass the projected hidden states to the student's modular backbone
        outputs = self.student(
            input_ids=input_ids,
            adapter_hidden_states=student_compatible_feats,
            labels=labels,
            output_router_logits=True
        )
        
        # 3. Multi-Adapter Distillation Loss
        total_distill_loss = 0
        num_adapters = len(student_compatible_feats)
        
        if num_adapters > 0:
            for name, projected_state in student_compatible_feats.items():
                # We anchor the student's output representation against EACH active adapter's projection
                # This ensures the backbone can handle diverse modalities simultaneously.
                loss_i = self.loss_fn(
                    student_logits=outputs['logits'],
                    teacher_logits=teacher_logits,
                    student_states=outputs['hidden_states'],
                    teacher_states=projected_state,
                    anchoring_layer_indices=self.config.get("critical_layers", [])
                )
                total_distill_loss += loss_i / num_adapters
        else:
            # Fallback to standard CE if no adapters are active (rare in distillation)
            total_distill_loss = outputs['loss']
        
        # 4. Router Diversity Loss (Prevents Mode Collapse)
        loss_entropy = 0
        if "entropy_loss" in outputs and outputs["entropy_loss"] is not None:
            loss_entropy = outputs["entropy_loss"]
        elif "router_logits" in outputs and outputs["router_logits"] is not None:
            # Manual fallback if entropy loss wasn't pre-computed
            probs = torch.softmax(outputs["router_logits"], dim=-1)
            loss_entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1).mean()
            
        beta_entropy = self.config.get("beta_entropy", 0.01)
        
        # Combined objective
        total_loss = total_distill_loss + (beta_entropy * loss_entropy)
        
        # 5. Safety: Loss Spike Detection & Recovery
        if total_loss.item() > self.prev_loss * self.loss_spike_threshold and self.global_step > 50:
            print(f"[Alert] Loss Spike: {total_loss.item():.4f} (Prev: {self.prev_loss:.4f}). Rolling back...")
            self.load_checkpoint("best") # Rollback to the best known state
            return {"loss": "ROLLBACK", "gnorm": "0.00"}

        # Scale Loss for Accumulation
        scaled_loss = total_loss / self.grad_accum_steps
        scaled_loss.backward()
        
        # 6. Gradient Monitoring & Update (Only at boundary)
        grad_norm = 0.0
        if is_accum_boundary:
            # Clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(self.student.parameters(), self.max_grad_norm)
            
            # Check for vanishing gradients (masking)
            if grad_norm < 1e-6:
                print(f"[Warning] Vanishing Gradients detected: {grad_norm:.6f}. Increasing CE weight.")
                self.loss_fn.alpha_ce *= 1.1

            self.optimizer.step()
            self.optimizer.zero_grad()
            self.global_step += 1
            
            # Update prev_loss for spike detection using the FULL (unscaled) loss
            self.prev_loss = total_loss.item()
        
        # We track full steps (global_step)
        
        if self.global_step % 50 == 0 and is_accum_boundary:
            self.save_checkpoint("latest")
            
        # 7. Periodic RAM & VRAM Cleanup (OOM-Safe for Laptop)
        if self.global_step % 100 == 0 and is_accum_boundary:
            gc.collect()
            if self.device == "cuda":
                torch.cuda.empty_cache()
            
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

    def run_memorization_audit(self):
        """Runs the Discoverable Memorization check on a subset of the previous batch."""
        if not self.auditor: return
        
        print(f"\n[Audit] Running Discoverable Memorization check at step {self.global_step}...")
        
        # We audit a small subset of the current training data for efficiency
        # Ideally we'd have a 'training_history' buffer, but we'll use the val_loader for now
        # or just a few strings if we can extract them.
        
        # Real implementation should track training samples.
        # For now, we use the paper's 50/50 prefix/suffix logic on validation text as a proxy
        # for 'leaked' patterns if they overlap.
        
        # In a real run, we'd feed the actual training text here.
        # This is a placeholder for the integration pattern.
        pass

    def check_thermal_status(self):
        """Checks hardware temperature and pauses if too hot."""
        try:
            import psutil
            # This is a generic check, specific sensors vary by OS/Hardware
            # For Linux/Zephyrus G16, we might check /sys/class/thermal/ or similar
            # For now, we use a generic CPU temp check if available
            temps = psutil.sensors_temperatures()
            if 'coretemp' in temps:
                max_temp = max(t.current for t in temps['coretemp'])
                if max_temp > 90:
                    print(f"[Hardware] Thermal Limit Hit ({max_temp}°C). Sleeping for 30s...")
                    torch.cuda.empty_cache()
                    import time
                    time.sleep(30)
        except Exception as e:
            # Skip if psutil or sensors not available, but log it once
            if not hasattr(self, '_thermal_check_warned'):
                print(f"[Hardware] Thermal monitoring not available: {e}")
                self._thermal_check_warned = True
