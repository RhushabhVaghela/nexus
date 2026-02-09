"""
Anti-Gravity Safe Training Supervisor

Monitors RAM and VRAM during training and kills the process before OOM.
Now WSL2-aware via MemoryGuard with:
    - Lower RAM thresholds for WSL (78% vs 90% native)
    - Faster polling (0.5s vs 1.0s, 0.1s under pressure)
    - Swap monitoring (WSL OOM killer triggers on swap pressure)
    - Graduated response: cleanup → throttle → terminate
"""

import subprocess
import time

try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
import sys
import os
import re
import gc

# Import centralized guard
try:
    from src.utils.memory_guard import guard, MemoryPressure, take_snapshot

    GUARD_AVAILABLE = True
except ImportError:
    guard = None
    GUARD_AVAILABLE = False

# ── Thresholds ──────────────────────────────────────────────────────────
# These are now WSL-aware via MemoryGuard.
# Fallback values if guard is unavailable.

if GUARD_AVAILABLE:
    _thresholds = guard.get_thresholds()
    RAM_THRESHOLD_PERCENT = _thresholds["ram_critical_percent"]  # 78% WSL, 90% native
    VRAM_THRESHOLD_PERCENT = _thresholds["vram_deadly_percent"]  # 94% WSL, 98% native
    CHECK_INTERVAL = _thresholds["check_interval_seconds"]  # 0.5s WSL, 1.0s native
    FAST_CHECK_INTERVAL = _thresholds[
        "fast_check_interval_seconds"
    ]  # 0.1s WSL, 0.25s native
    IS_WSL = _thresholds["is_wsl"]
else:
    RAM_THRESHOLD_PERCENT = 90.0
    VRAM_THRESHOLD_PERCENT = 98.0
    CHECK_INTERVAL = 1.0
    FAST_CHECK_INTERVAL = 0.25
    IS_WSL = False


def get_vram_usage():
    """Returns list of VRAM usage percent per GPU."""
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total",
                "--format=csv,nounits,noheader",
            ],
            encoding="utf-8",
        )
        percentages = []
        for line in output.strip().split("\n"):
            if not line:
                continue
            used, total = map(float, line.split(","))
            percentages.append((used / total) * 100.0)
        return percentages
    except Exception:
        return []  # No GPU or nvidia-smi failed


def _try_cleanup():
    """Attempt emergency memory cleanup before killing."""
    if GUARD_AVAILABLE:
        guard.emergency_cleanup(aggressive=True)
    else:
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


def monitor_and_run(command, description):
    print(f"🚀 STARTING: {description}")
    print(f"   Command: {' '.join(command)}")
    if IS_WSL:
        print(f"   ⚠️  WSL2 detected — using tighter memory thresholds")
    print(
        f"   Limits: RAM < {RAM_THRESHOLD_PERCENT}% | VRAM < {VRAM_THRESHOLD_PERCENT}%"
    )
    print(f"   Poll interval: {CHECK_INTERVAL}s (fast: {FAST_CHECK_INTERVAL}s)")

    process = subprocess.Popen(command)
    cleanup_attempted = False
    current_interval = CHECK_INTERVAL

    try:
        while process.poll() is None:
            # 1. Check RAM
            if PSUTIL_AVAILABLE:
                ram = psutil.virtual_memory()
                ram_percent = ram.percent
                ram_free_gb = ram.available / (1024**3)
            else:
                ram_percent = 0.0
                ram_free_gb = 0.0

            # 2. Check VRAM
            vram_percents = get_vram_usage()
            max_vram = max(vram_percents) if vram_percents else 0.0

            # 3. Check swap (WSL-specific)
            swap_warning = ""
            if IS_WSL and PSUTIL_AVAILABLE:
                swap = psutil.swap_memory()
                if swap.percent > 50:
                    swap_warning = f" | SWAP: {swap.percent:.0f}%⚠️"

            # Log status on same line
            sys.stdout.write(
                f"\r[Monitor] RAM: {ram_percent:.1f}% ({ram_free_gb:.1f}GB free) | "
                f"VRAM: {max_vram:.1f}%{swap_warning}    "
            )
            sys.stdout.flush()

            # 4. Graduated response
            if GUARD_AVAILABLE:
                pressure = guard.get_pressure()

                if pressure == MemoryPressure.DEADLY:
                    print(f"\n\n🚨 DEADLY: Memory pressure at DEADLY level!")
                    print("🛑 KILLING PROCESS TO PREVENT SYSTEM CRASH...")
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                    return False

                elif pressure == MemoryPressure.CRITICAL and not cleanup_attempted:
                    print(f"\n\n⚠️  CRITICAL: Attempting emergency cleanup...")
                    _try_cleanup()
                    cleanup_attempted = True
                    # Give it a moment to take effect
                    time.sleep(2.0)
                    # Re-check
                    new_pressure = guard.get_pressure()
                    if new_pressure in (MemoryPressure.CRITICAL, MemoryPressure.DEADLY):
                        print(
                            f"   Cleanup insufficient. Pressure still {new_pressure.value}."
                        )
                        print("🛑 TERMINATING PROCESS...")
                        process.terminate()
                        try:
                            process.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            process.kill()
                        return False
                    else:
                        print(
                            f"   ✅ Cleanup helped! Pressure now {new_pressure.value}. Continuing..."
                        )
                        cleanup_attempted = False  # Allow future cleanup attempts

                elif pressure == MemoryPressure.HIGH:
                    current_interval = FAST_CHECK_INTERVAL  # Poll faster
                else:
                    current_interval = CHECK_INTERVAL
                    cleanup_attempted = False

            else:
                # Legacy path: simple threshold check
                if ram_percent > RAM_THRESHOLD_PERCENT:
                    print(
                        f"\n\n🚨 CRITICAL WARNING: RAM reached {ram_percent:.1f}% (Limit: {RAM_THRESHOLD_PERCENT}%)"
                    )
                    print("🛑 INTERRUPTING PROCESS TO PREVENT FREEZE...")
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                    return False

                if max_vram > VRAM_THRESHOLD_PERCENT:
                    print(
                        f"\n\n🚨 CRITICAL WARNING: VRAM reached {max_vram:.1f}% (Limit: {VRAM_THRESHOLD_PERCENT}%)"
                    )
                    print("🛑 INTERRUPTING PROCESS TO PREVENT CRASH...")
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                    return False

            time.sleep(current_interval)

    except KeyboardInterrupt:
        print("\n\nUser interrupted monitor.")
        process.terminate()
        return False

    print("\n")  # Newline after loop

    if process.returncode == 0:
        print(f"✅ SUCCESS: {description}")
        return True
    else:
        print(f"❌ FAILED: {description} (Return Code: {process.returncode})")
        return False


def main():
    print("🛡️  Anti-Gravity Safe Training Supervisor")
    print("========================================")
    env_label = "WSL2" if IS_WSL else "Native"
    print(f"Environment: {env_label}")
    print(f"Limits: RAM < {RAM_THRESHOLD_PERCENT}% | VRAM < {VRAM_THRESHOLD_PERCENT}%")
    print(f"Poll: {CHECK_INTERVAL}s normal | {FAST_CHECK_INTERVAL}s under pressure")

    if GUARD_AVAILABLE:
        guard.print_status()

    data_path = "/mnt/e/data/datasets"
    output_dir = "./checkpoints/nexus_fine_tuning"
    sample_size = "5"

    cmd = [
        "python",
        "src/24_multimodal_training.py",
        "--stage",
        "2",
        "--data-path",
        data_path,
        "--output-dir",
        output_dir,
        "--sample-size",
        sample_size,
        "--log-results",
    ]

    # Pre-flight check
    if GUARD_AVAILABLE:
        safe, msg = guard.preflight_check(
            estimated_ram_gb=18.0,  # From memory_config.py plan
            estimated_vram_gb=14.0,
            operation_name="Stage 2 Fine-Tuning",
        )
        if not safe:
            print(f"\n⚠️  Pre-flight check failed: {msg}")
            print("   Attempting cleanup before starting...")
            guard.emergency_cleanup(aggressive=True)
            safe, msg = guard.preflight_check(
                estimated_ram_gb=18.0,
                estimated_vram_gb=14.0,
                operation_name="Stage 2 Fine-Tuning (post-cleanup)",
            )
            if not safe:
                print(f"\n❌ Still insufficient memory after cleanup: {msg}")
                print("   Consider closing other applications or reducing model size.")
                sys.exit(1)

    # Basic Retry Loop
    max_retries = 3
    success = False
    for attempt in range(max_retries):
        print(f"\n--- Attempt {attempt + 1}/{max_retries} ---")
        success = monitor_and_run(cmd, "Stage 2: Full Fine-Tuning")
        if success:
            print("\n🎉 All pipeline stages completed successfully!")
            break
        else:
            print("\n⚠️ Attempt failed. Waiting 10 seconds before cooling down...")
            if GUARD_AVAILABLE:
                guard.emergency_cleanup(aggressive=True)
            time.sleep(10)

    if not success:
        print("\n❌ Failed to complete training after retries.")
        sys.exit(1)


if __name__ == "__main__":
    main()
