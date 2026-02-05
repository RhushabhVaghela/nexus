import os
import torch
import time
import psutil
import pynvml

class HardwareMonitor:
    def __init__(self, device_id=0):
        self.device_id = device_id
        try:
            pynvml.nvmlInit()
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
            self.gpu_available = True
        except Exception as e:
            print(f"Warning: NVML initialization failed: {e}. GPU monitoring disabled.")
            self.gpu_available = False

    def check_vram(self):
        """Returns free VRAM in MB."""
        if not self.gpu_available:
            return 0
        info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
        return info.free / 1024**2

    def check_temp(self):
        """Returns GPU temperature in Celsius."""
        if not self.gpu_available:
            return 0
        return pynvml.nvmlDeviceGetTemperature(self.handle, pynvml.NVML_TEMPERATURE_GPU)

    def optimize_batch_size(self, current_bs, min_bs=1, target_vram_gb=12.0):
        """
        Dynamically adjusts batch size based on VRAM pressure.
        Returns the recommended new batch size.
        """
        free_vram_mb = self.check_vram()
        free_vram_gb = free_vram_mb / 1024
        
        # If we are dangerously low (<2GB), halve immediately
        if free_vram_gb < 2.0:
            new_bs = max(min_bs, current_bs // 2)
            print(f"[Hardware] VRAM Critical ({free_vram_gb:.1f}GB). Halving BS to {new_bs}.")
            return new_bs
            
        # If we have massive headroom (>10GB free while running), try growing conservatively
        # Note: This is risky during training, usually we just shrink.
        # For safety, we only shrink in this implementation.
        
        return current_bs

    def thermal_throttle(self, threshold_c=85, cooldown_sec=30):
        """
        Blocks execution if GPU is overheating.
        """
        if not self.gpu_available:
            return
            
        temp = self.check_temp()
        if temp >= threshold_c:
            print(f"[Hardware] Thermal Throttling: GPU at {temp}°C (Limit {threshold_c}°C). Sleeping {cooldown_sec}s...")
            time.sleep(cooldown_sec)
            
            # Re-check
            new_temp = self.check_temp()
            print(f"[Hardware] Resumed. Temp now {new_temp}°C.")

monitor = HardwareMonitor()
