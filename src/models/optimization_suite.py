import argparse
import time
import os
import signal
import sys
import torch
import numpy as np
from tqdm import tqdm

# Mock pynvml for safety if not installed (Production should require it)
try:
    import pynvml
    pynvml.nvmlInit()
    HAS_NVML = True
except ImportError:
    HAS_NVML = False

class ThermalWatchdog:
    """
    Monitors GPU temperature and throttles execution if it exceeds limits.
    """
    def __init__(self, threshold=88.0, cooldown_period=10.0):
        self.threshold = threshold
        self.cooldown_period = cooldown_period
        self.handle = None
        if HAS_NVML:
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(0) # Assuming single GPU

    def check(self):
        if not self.handle:
            return
        
        try:
            temp = pynvml.nvmlDeviceGetTemperature(self.handle, pynvml.NVML_TEMPERATURE_GPU)
            if temp >= self.threshold:
                print(f"\n[ThermalWatchdog] GPU Temp {temp}°C exceeds threshold {self.threshold}°C. Cooling down for {self.cooldown_period}s...")
                time.sleep(self.cooldown_period)
        except Exception as e:
            print(f"[ThermalWatchdog] Error polling temperature: {e}")

class GradNormMonitor:
    """
    Tracks gradient norms to detect 'Gradient Masking' (Anchor dominance).
    """
    def __init__(self):
        self.task_norms = []
        self.anchor_norms = []
    
    def check_health(self, task_grad_norm, anchor_grad_norm):
        if anchor_grad_norm == 0: 
            return True # Edge case
            
        ratio = task_grad_norm / (anchor_grad_norm + 1e-8)
        self.task_norms.append(task_grad_norm)
        self.anchor_norms.append(anchor_grad_norm)
        
        # Log masking if Task Gradient is tiny compared to Anchor
        if ratio < 0.01:
            print(f"\n[GradMonitor] WARNING: Gradient Masking detected! Ratio: {ratio:.4f}")
            return False
        return True

class SynergyMonitor:
    """Tracks cases where the Student succeeds while the Teacher fails."""
    def __init__(self):
        self.synergy_count = 0
        self.total_samples = 0
        self.synergy_log = []

    def record(self, target, student_res, teacher_hit):
        self.total_samples += 1
        student_hit = target.lower() in student_res.lower()
        
        # Synergy occurs if Student is Right AND Teacher was Wrong
        if student_hit and not teacher_hit:
            self.synergy_count += 1
            self.synergy_log.append({
                "target": target,
                "student_output": student_res[:100]
            })

    def get_report(self):
        rate = (self.synergy_count / self.total_samples) if self.total_samples > 0 else 0
        return {"count": self.synergy_count, "rate": f"{rate:.1%}"}

def compute_optimal_batch_size(vram_free_mb):
    """Dynamic batch sizing based on VRAM headroom."""
    if vram_free_mb > 12000:
        return 16
    elif vram_free_mb > 8000:
        return 8
    else:
        return 4

