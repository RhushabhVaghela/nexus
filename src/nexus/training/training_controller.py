#!/usr/bin/env python3
"""
training_controller.py
Advanced training controller with pause, resume, and cooldown features.

Features:
- Pause/Resume training with SIGUSR1 signal (kill -USR1 <pid>)
- Emergency checkpoint with SIGUSR2 signal
- Automatic cooldown intervals
- Compressed dataset handling (.gz, .tar.gz, .zip)
- Hardware-friendly batch size reduction
"""

import os
import sys
import signal
import time
import tarfile
import zipfile
import gzip
import shutil
import subprocess
import logging
from pathlib import Path
from typing import Optional
import torch

# Setup logging
logger = logging.getLogger(__name__)

# ============ CONSTANTS (DEFAULTS) ============
COOLDOWN_INTERVAL_STEPS = 500  # Cooldown every N steps
COOLDOWN_DURATION_SECONDS = 60  # 1 minute cooldown
DEFAULT_GPU_TEMP_THRESHOLD = 83  # Celsius - pause if exceeded
DEFAULT_CPU_TEMP_THRESHOLD = 88  # Celsius - pause if exceeded
CHECKPOINT_DIR = "/mnt/e/data/models/checkpoints"

# ============ CONFIGURABLE STATE ============
_gpu_temp_threshold = DEFAULT_GPU_TEMP_THRESHOLD
_cpu_temp_threshold = DEFAULT_CPU_TEMP_THRESHOLD

# ============ GLOBAL STATE ============
_paused = False
_checkpoint_requested = False


def signal_handler_pause(signum, frame):
    """Handle SIGUSR1 - Toggle pause state."""
    global _paused
    _paused = not _paused
    state = "PAUSED" if _paused else "RESUMED"
    print(f"\n🔔 Signal received: Training {state}")
    

def signal_handler_checkpoint(signum, frame):
    """Handle SIGUSR2 - Request emergency checkpoint."""
    global _checkpoint_requested
    _checkpoint_requested = True
    print("\n🔔 Signal received: Emergency checkpoint requested")


def setup_signal_handlers():
    """Setup Unix signal handlers for pause/checkpoint."""
    if sys.platform != "win32":
        signal.signal(signal.SIGUSR1, signal_handler_pause)
        signal.signal(signal.SIGUSR2, signal_handler_checkpoint)
        print(f"📡 Signal handlers registered (PID: {os.getpid()})")
        print(f"   Pause/Resume: kill -USR1 {os.getpid()}")
        print(f"   Checkpoint:   kill -USR2 {os.getpid()}")


def set_thresholds(gpu_temp: int = None, cpu_temp: int = None):
    """
    Configure custom temperature thresholds.
    
    Args:
        gpu_temp: GPU temperature threshold in Celsius (default: 83)
        cpu_temp: CPU temperature threshold in Celsius (default: 88)
    """
    global _gpu_temp_threshold, _cpu_temp_threshold
    if gpu_temp is not None:
        _gpu_temp_threshold = gpu_temp
        print(f"⚙️ GPU temp threshold set to {gpu_temp}°C")
    if cpu_temp is not None:
        _cpu_temp_threshold = cpu_temp
        print(f"⚙️ CPU temp threshold set to {cpu_temp}°C")


def get_gpu_temperature() -> float:
    """
    Get current GPU temperature in Celsius using nvidia-smi.
    
    Returns:
        GPU temperature in Celsius, or 0.0 if unable to determine
        
    Note:
        Requires nvidia-smi to be installed and accessible in PATH.
        Returns 0.0 on non-CUDA systems or if nvidia-smi is unavailable.
    """
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=temperature.gpu", 
             "--format=csv,nounits,noheader"],
            encoding="utf-8",
            stderr=subprocess.DEVNULL
        )
        temps = [float(t.strip()) for t in output.strip().split("\n") if t.strip()]
        return max(temps) if temps else 0.0
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        logger.debug(f"nvidia-smi not available: {e}")
        return 0.0
    except ValueError as e:
        logger.warning(f"Failed to parse GPU temperature: {e}")
        return 0.0
    except Exception as e:
        logger.warning(f"Unexpected error getting GPU temperature: {e}")
        return 0.0


def get_cpu_temperature() -> float:
    """
    Get current CPU temperature in Celsius.
    Works on Linux via /sys or sensors command.
    
    Returns:
        CPU temperature in Celsius, or 0.0 if unable to determine
        
    Note:
        Tries multiple methods in order of reliability:
        1. /sys/class/thermal/thermal_zone0/temp (most reliable on Linux)
        2. lm-sensors command (sensors -u)
    """
    # Method 1: Try /sys thermal zones (most reliable on Linux)
    thermal_path = Path("/sys/class/thermal/thermal_zone0/temp")
    if thermal_path.exists():
        try:
            with open(thermal_path) as f:
                temp_millidegrees = float(f.read().strip())
                return temp_millidegrees / 1000.0
        except (IOError, OSError, ValueError) as e:
            logger.debug(f"Failed to read thermal zone temperature: {e}")
    
    # Method 2: Try lm-sensors
    try:
        output = subprocess.check_output(
            ["sensors", "-u"],
            encoding="utf-8",
            stderr=subprocess.DEVNULL
        )
        for line in output.split("\n"):
            if "temp1_input" in line or "Core 0" in line:
                parts = line.split(":")
                if len(parts) > 1:
                    try:
                        return float(parts[1].strip())
                    except ValueError:
                        continue
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        logger.debug(f"lm-sensors not available or failed: {e}")
    except Exception as e:
        logger.warning(f"Unexpected error getting CPU temperature: {e}")
    
    return 0.0


def check_and_cooldown(current_step: int) -> bool:
    """
    Check if cooldown is needed and perform it.
    Monitors both GPU and CPU temperatures.
    
    Returns:
        True if cooldown was performed
    """
    # Check by interval
    if current_step > 0 and current_step % COOLDOWN_INTERVAL_STEPS == 0:
        print(f"\n❄️ Scheduled cooldown at step {current_step}...")
        _perform_cooldown()
        return True
    
    # Check GPU temperature
    gpu_temp = get_gpu_temperature()
    if gpu_temp > _gpu_temp_threshold:
        print(f"\n🔥 GPU at {gpu_temp}°C (threshold: {_gpu_temp_threshold}°C) - Cooling down...")
        _perform_cooldown()
        return True
    
    # Check CPU temperature
    cpu_temp = get_cpu_temperature()
    if cpu_temp > _cpu_temp_threshold:
        print(f"\n🔥 CPU at {cpu_temp}°C (threshold: {_cpu_temp_threshold}°C) - Cooling down...")
        _perform_cooldown()
        return True
    
    return False


def _perform_cooldown():
    """Perform cooldown by pausing and clearing cache."""
    # Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    # Wait for cooldown
    for remaining in range(COOLDOWN_DURATION_SECONDS, 0, -10):
        print(f"   Cooling: {remaining}s remaining...", end="\r")
        time.sleep(10)
    print(f"   ✓ Cooldown complete. Resuming training...")


def check_pause_state():
    """Check if training should be paused. Blocks until resumed."""
    global _paused
    if _paused:
        print("⏸️ Training PAUSED. Send SIGUSR1 to resume.")
        while _paused:
            time.sleep(1)
        print("▶️ Training RESUMED.")


def check_checkpoint_request(model, optimizer, step, output_dir) -> bool:
    """
    Check if emergency checkpoint was requested.
    
    Returns:
        True if checkpoint was saved
    """
    global _checkpoint_requested
    if _checkpoint_requested:
        _checkpoint_requested = False
        save_emergency_checkpoint(model, optimizer, step, output_dir)
        return True
    return False


def save_emergency_checkpoint(model, optimizer, step, output_dir):
    """Save an emergency checkpoint."""
    checkpoint_path = Path(output_dir) / f"emergency_checkpoint_step_{step}"
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    
    print(f"💾 Saving emergency checkpoint to {checkpoint_path}...")
    
    # Save model
    if hasattr(model, "save_pretrained"):
        model.save_pretrained(checkpoint_path)
    else:
        torch.save(model.state_dict(), checkpoint_path / "model.pt")
    
    # Save optimizer state
    torch.save({
        "step": step,
        "optimizer_state_dict": optimizer.state_dict(),
    }, checkpoint_path / "optimizer.pt")
    
    print(f"✅ Emergency checkpoint saved at step {step}")


# ============ COMPRESSED FILE HANDLING ============

def extract_if_compressed(file_path: str, target_dir: Optional[str] = None) -> str:
    """
    Extract compressed files (.gz, .tar.gz, .zip) if needed.
    
    Args:
        file_path: Path to file (compressed or not)
        target_dir: Where to extract (default: same directory)
        
    Returns:
        Path to extracted content
    """
    path = Path(file_path)
    target = Path(target_dir) if target_dir else path.parent
    
    # .tar.gz or .tgz
    if path.suffix in [".gz", ".tgz"] and ".tar" in path.name:
        extracted_path = target / path.name.replace(".tar.gz", "").replace(".tgz", "")
        if not extracted_path.exists():
            print(f"📦 Extracting {path.name}...")
            with tarfile.open(path, "r:gz") as tar:
                tar.extractall(target)
        return str(extracted_path)
    
    # .gz (gzip single file)
    elif path.suffix == ".gz":
        extracted_path = target / path.stem
        if not extracted_path.exists():
            print(f"📦 Extracting {path.name}...")
            with gzip.open(path, "rb") as f_in:
                with open(extracted_path, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)
        return str(extracted_path)
    
    # .zip
    elif path.suffix == ".zip":
        extracted_path = target / path.stem
        if not extracted_path.exists():
            print(f"📦 Extracting {path.name}...")
            with zipfile.ZipFile(path, "r") as zip_ref:
                zip_ref.extractall(extracted_path)
        return str(extracted_path)
    
    # .rar (requires external tool)
    elif path.suffix == ".rar":
        extracted_path = target / path.stem
        if not extracted_path.exists():
            print(f"📦 Extracting {path.name} (requires unrar)...")
            subprocess.run(["unrar", "x", str(path), str(target)], check=True)
        return str(extracted_path)
    
    # Not compressed
    return str(path)


def scan_and_extract_datasets(dataset_dir: str) -> list:
    """
    Scan dataset directory and extract any compressed files.
    
    Returns:
        List of dataset paths (extracted)
    """
    dataset_paths = []
    base_path = Path(dataset_dir)
    
    for item in base_path.iterdir():
        if item.is_dir():
            dataset_paths.append(str(item))
        elif item.suffix in [".gz", ".zip", ".rar", ".tgz"]:
            extracted = extract_if_compressed(str(item))
            dataset_paths.append(extracted)
    
    return dataset_paths


# ============ MAIN INTEGRATION ============

def training_step_hook(model, optimizer, step, output_dir):
    """
    Call this at each training step to handle:
    - Pause/resume
    - Emergency checkpoint
    - Cooldown
    
    Usage:
        for step, batch in enumerate(dataloader):
            training_step_hook(model, optimizer, step, output_dir)
            # ... your training code ...
    """
    # 1. Check pause state
    check_pause_state()
    
    # 2. Check for emergency checkpoint request
    check_checkpoint_request(model, optimizer, step, output_dir)
    
    # 3. Check for scheduled cooldown
    check_and_cooldown(step)


if __name__ == "__main__":
    # Demo
    print("Training Controller Demo")
    setup_signal_handlers()
    
    print("\nTesting compressed file detection...")
    test_files = [
        "dataset.tar.gz",
        "images.zip", 
        "data.gz",
        "folder/",
    ]
    for f in test_files:
        print(f"  {f} -> would extract to: {Path(f).stem}")
    
    print("\nController ready. Use signals to control training.")
    print("Press Ctrl+C to exit demo.")
    
    try:
        step = 0
        while True:
            check_pause_state()
            check_and_cooldown(step)
            step += 1
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nDemo exited.")
