"""
Memory utilities for Nexus model loading and inference.

Now delegates threshold decisions to the centralized MemoryGuard.
Uses torch.cuda.mem_get_info() instead of memory_allocated() for
accurate VRAM readings (mem_get_info accounts for all allocations,
not just PyTorch-tracked ones).
"""

import torch

try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
import logging

logger = logging.getLogger(__name__)

# Import the centralized guard
try:
    from src.utils.memory_guard import guard, MemoryPressure

    GUARD_AVAILABLE = True
except ImportError:
    guard = None
    GUARD_AVAILABLE = False
    logger.warning("memory_guard not available, using legacy thresholds")


def check_memory_headroom(vram_headroom_gb=None, ram_headroom_gb=None):
    """
    Checks if the system has enough RAM and VRAM headroom.

    Headroom defaults are now WSL-aware via MemoryGuard:
        - WSL2:   4.0GB RAM, 2.0GB VRAM
        - Native: 2.0GB RAM, 1.0GB VRAM

    Uses torch.cuda.mem_get_info() for accurate VRAM (not memory_allocated).

    Returns: Tuple (bool, str) - Success flag and status message.
    """
    # Get WSL-aware defaults from guard
    if GUARD_AVAILABLE:
        thresholds = guard.get_thresholds()
        default_ram = thresholds["min_ram_headroom_gb"]
        default_vram = thresholds["min_vram_headroom_gb"]
    else:
        default_ram = 2.0
        default_vram = 1.0

    ram_headroom_gb = ram_headroom_gb if ram_headroom_gb is not None else default_ram
    vram_headroom_gb = (
        vram_headroom_gb if vram_headroom_gb is not None else default_vram
    )

    status = []
    success = True

    # 1. VRAM Check — use mem_get_info for accurate readings
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            try:
                free_bytes, total_bytes = torch.cuda.mem_get_info(i)
                free_vram = free_bytes / (1024**3)
                total_vram = total_bytes / (1024**3)
            except Exception:
                # Fallback to old method if mem_get_info fails
                props = torch.cuda.get_device_properties(i)
                total_vram = props.total_memory / (1024**3)
                used_vram = torch.cuda.memory_allocated(i) / (1024**3)
                free_vram = total_vram - used_vram

            if free_vram < vram_headroom_gb:
                status.append(
                    f"VRAM[{i}] Low: {free_vram:.1f}GB free (Need {vram_headroom_gb}GB)"
                )
                success = False
            else:
                status.append(f"VRAM[{i}] OK: {free_vram:.1f}GB free")

    # 2. RAM Check
    if PSUTIL_AVAILABLE:
        ram = psutil.virtual_memory()
        free_ram = ram.available / (1024**3)
        if free_ram < ram_headroom_gb:
            status.append(f"RAM Low: {free_ram:.1f}GB free (Need {ram_headroom_gb}GB)")
            success = False
        else:
            status.append(f"RAM OK: {free_ram:.1f}GB free")
    else:
        status.append("RAM: psutil not available, skipping RAM check")

    # 3. WSL swap warning
    if GUARD_AVAILABLE and guard.is_wsl and PSUTIL_AVAILABLE:
        swap = psutil.swap_memory()
        if swap.percent > 50:
            status.append(f"WSL Swap Warning: {swap.percent:.0f}% used — OOM risk")
            success = False

    # 4. Log via guard if available
    if GUARD_AVAILABLE:
        pressure = guard.get_pressure()
        if pressure in (MemoryPressure.CRITICAL, MemoryPressure.DEADLY):
            status.append(f"MemoryGuard: {pressure.value.upper()} pressure!")
            success = False

    return success, " | ".join(status)


def get_recommended_batch_size(base_batch=1, max_batch=16):
    """
    Dynamically recommends a batch size based on free VRAM.
    Uses mem_get_info for accurate readings.
    Also considers memory pressure from MemoryGuard.
    """
    if not torch.cuda.is_available():
        return base_batch

    # Get actual free VRAM
    free_vram = torch.cuda.mem_get_info()[0] / (1024**3)

    # If guard is available and pressure is high, be conservative
    if GUARD_AVAILABLE:
        pressure = guard.get_pressure()
        if pressure == MemoryPressure.DEADLY:
            return base_batch
        elif pressure == MemoryPressure.CRITICAL:
            return base_batch
        elif pressure == MemoryPressure.HIGH:
            # Halve the heuristic
            recommended = int(free_vram // 5.0)
            return max(base_batch, min(recommended, max_batch))

    # Normal heuristic: 1 batch ~ 2.5GB for a 2B student in FP16/half
    recommended = int(free_vram // 2.5)
    return max(base_batch, min(recommended, max_batch))


def estimate_model_vram_gb(config, bits=4):
    """
    Estimates model VRAM footprint in GB.
    Heuristic: ~12 * L * H^2 for param count.
    """
    h = getattr(config, "hidden_size", getattr(config, "d_model", 2048))
    l = getattr(config, "num_hidden_layers", getattr(config, "num_layers", 12))

    # Estimate total parameters
    # Factor 12 covers 4.5 for attention (Q,K,V,O) + 7.5 for MLP (Gate, Up, Down / 4:8)
    est_params = h * h * l * 12

    bytes_per_param = bits / 8
    model_size_gb = (est_params * bytes_per_param) / (1024**3)

    # Add 20% overhead for KV cache, gradients, etc.
    return model_size_gb * 1.2


def should_use_sli(config, safety_factor=0.8):
    """
    Returns True if model should use SLI based on current VRAM.
    Uses mem_get_info for accurate VRAM measurement.
    """
    if not torch.cuda.is_available():
        return False

    est_vram = estimate_model_vram_gb(config, bits=4)  # Assume 4-bit load
    free_vram = torch.cuda.mem_get_info()[0] / (1024**3)

    return est_vram > (free_vram * safety_factor)
