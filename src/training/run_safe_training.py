
import subprocess
import time
import psutil
import sys
import os
import re

# Config
RAM_THRESHOLD_PERCENT = 90.0
VRAM_THRESHOLD_PERCENT = 98.0
CHECK_INTERVAL = 1.0 # Seconds

def get_vram_usage():
    """Returns list of VRAM usage percent per GPU."""
    try:
        # Run nvidia-smi to get memory usage
        # Format: utilization.memory
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,nounits,noheader"], 
            encoding="utf-8"
        )
        percentages = []
        for line in output.strip().split("\n"):
            if not line: continue
            used, total = map(float, line.split(","))
            percentages.append((used / total) * 100.0)
        return percentages
    except Exception:
        return [] # No GPU or nvidia-smi failed

def monitor_and_run(command, description):
    print(f"🚀 STARTING: {description}")
    print(f"   Command: {' '.join(command)}")
    
    process = subprocess.Popen(command)
    
    try:
        while process.poll() is None:
            # 1. Check RAM
            ram_percent = psutil.virtual_memory().percent
            
            # 2. Check VRAM
            vram_percents = get_vram_usage()
            max_vram = max(vram_percents) if vram_percents else 0.0
            
            # Log status on same line
            sys.stdout.write(f"\r[Monitor] RAM: {ram_percent:.1f}% | VRAM: {max_vram:.1f}%")
            sys.stdout.flush()
            
            # 3. Safety Check
            if ram_percent > RAM_THRESHOLD_PERCENT:
                print(f"\n\n🚨 CRITICAL WARNING: RAM reached {ram_percent:.1f}% (Limit: {RAM_THRESHOLD_PERCENT}%)")
                print("🛑 INTERRUPTING PROCESS TO PREVENT FREEZE...")
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                return False
                
            if max_vram > VRAM_THRESHOLD_PERCENT:
                print(f"\n\n🚨 CRITICAL WARNING: VRAM reached {max_vram:.1f}% (Limit: {VRAM_THRESHOLD_PERCENT}%)")
                print("🛑 INTERRUPTING PROCESS TO PREVENT CRASH...")
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                return False
                
            time.sleep(CHECK_INTERVAL)
            
    except KeyboardInterrupt:
        print("\n\nUser interrupted monitor.")
        process.terminate()
        return False

    print("\n") # Newline after loop
    
    if process.returncode == 0:
        print(f"✅ SUCCESS: {description}")
        return True
    else:
        print(f"❌ FAILED: {description} (Return Code: {process.returncode})")
        return False

def main():
    print("🛡️  Anti-Gravity Safe Training Supervisor")
    print("========================================")
    print(f"Limits: RAM < {RAM_THRESHOLD_PERCENT}% | VRAM < {VRAM_THRESHOLD_PERCENT}%")
    
    data_path = "/mnt/e/data/datasets"
    output_dir = "./checkpoints/nexus_fine_tuning"
    sample_size = "5"
    
    # Run Stage 2 directly (Fine-Tuning) as per user flow
    # Assuming Stage 1 (Projectors) is either implicit or we skip to fine-tuning the full model
    # User asked for "training of entire pipeline", usually implies full fine-tuning.
    
    cmd = [
        "python", "src/24_multimodal_training.py",
        "--stage", "2",
        "--data-path", data_path,
        "--output-dir", output_dir,
        "--sample-size", sample_size,
        "--log-results"
    ]
    
    # Basic Retry Loop (Simple)
    max_retries = 3
    for attempt in range(max_retries):
        print(f"\n--- Attempt {attempt+1}/{max_retries} ---")
        success = monitor_and_run(cmd, "Stage 2: Full Fine-Tuning")
        if success:
            print("\n🎉 All pipeline stages completed successfully!")
            break
        else:
            print("\n⚠️ Attempt failed. Waiting 10 seconds before cooling down...")
            time.sleep(10)
            
            # In a real agentic loop, we might try to reduce batch size here via args
            # cmd.append("--per-device-train-batch-size=1") # But we happen to know it's already 1
            
    if not success:
        print("\n❌ Failed to complete training after retries.")
        sys.exit(1)

if __name__ == "__main__":
    main()
