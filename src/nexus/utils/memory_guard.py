"""
Memory Guard — Centralized WSL-Aware Memory Safety System for Nexus

This module is the SINGLE source of truth for all memory safety decisions.
It replaces the scattered, inconsistent thresholds across 20+ files with
one unified, WSL-aware memory management layer.

Key Features:
    1. WSL2 Detection — auto-detects WSL and lowers thresholds
    2. Dynamic System Profiling — detects actual RAM/VRAM instead of hardcoding
    3. Swap Monitoring — WSL OOM killer activates before psutil reports 100%
    4. Pre-flight Estimation — blocks operations that would exceed safe limits
    5. Graduated Response — warn → throttle → block → kill (not just kill)

Usage:
    from nexus.utils.memory_guard import guard

    # Quick safety check before an expensive operation
    if guard.is_safe():
        do_expensive_thing()
    else:
        guard.emergency_cleanup()

    # Pre-flight check with estimated allocation
    guard.preflight_check(estimated_ram_gb=4.0, estimated_vram_gb=8.0)

    # Get WSL-adjusted thresholds for other modules
    thresholds = guard.get_thresholds()

Author: Nexus Team
"""

import os
import gc
import time
import logging
import platform
import threading
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any, Callable, List
from pathlib import Path

try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────


class MemoryPressure(Enum):
    """Graduated memory pressure levels."""

    SAFE = "safe"  # Plenty of headroom
    ELEVATED = "elevated"  # Getting tight, start being careful
    HIGH = "high"  # Throttle new allocations
    CRITICAL = "critical"  # Emergency cleanup needed
    DEADLY = "deadly"  # Abort operations immediately


@dataclass(frozen=True)
class MemoryThresholds:
    """
    Memory thresholds as RAM/VRAM percentages.
    WSL gets tighter thresholds because the WSL2 OOM killer activates
    before Linux-native systems would.
    """

    # RAM thresholds (percent used)
    ram_elevated: float = 60.0
    ram_high: float = 75.0
    ram_critical: float = 85.0
    ram_deadly: float = 92.0

    # VRAM thresholds (percent used)
    vram_elevated: float = 70.0
    vram_high: float = 80.0
    vram_critical: float = 90.0
    vram_deadly: float = 96.0

    # Minimum headroom in GB (absolute floor)
    min_ram_headroom_gb: float = 3.0
    min_vram_headroom_gb: float = 1.5

    # Monitoring
    check_interval_seconds: float = 0.5
    fast_check_interval_seconds: float = 0.1  # When pressure is HIGH+


# Native Linux / bare-metal defaults
NATIVE_THRESHOLDS = MemoryThresholds(
    ram_elevated=70.0,
    ram_high=80.0,
    ram_critical=90.0,
    ram_deadly=95.0,
    vram_elevated=75.0,
    vram_high=85.0,
    vram_critical=92.0,
    vram_deadly=98.0,
    min_ram_headroom_gb=2.0,
    min_vram_headroom_gb=1.0,
    check_interval_seconds=1.0,
    fast_check_interval_seconds=0.25,
)

# WSL2 — much tighter because WSL OOM killer is aggressive.
# WSL2 shares physical RAM with Windows host. With 32GB host RAM,
# WSL2 typically sees ~16GB. Windows itself uses 4-6GB, so actual
# safe headroom is smaller than it appears.
WSL_THRESHOLDS = MemoryThresholds(
    ram_elevated=50.0,
    ram_high=62.0,
    ram_critical=72.0,
    ram_deadly=78.0,
    vram_elevated=60.0,
    vram_high=72.0,
    vram_critical=82.0,
    vram_deadly=88.0,
    min_ram_headroom_gb=5.0,
    min_vram_headroom_gb=3.0,
    check_interval_seconds=0.3,
    fast_check_interval_seconds=0.1,
)


# ─────────────────────────────────────────────────────────────────────────────
# System Detection
# ─────────────────────────────────────────────────────────────────────────────


def detect_wsl() -> bool:
    """
    Detect if running under WSL2.

    Checks:
    1. /proc/version contains 'microsoft' or 'WSL'
    2. WSL_DISTRO_NAME environment variable is set
    3. /proc/sys/fs/binfmt_misc/WSLInterop exists
    """
    # Method 1: /proc/version
    try:
        version_path = Path("/proc/version")
        if version_path.exists():
            version_text = version_path.read_text().lower()
            if "microsoft" in version_text or "wsl" in version_text:
                return True
    except (OSError, PermissionError):
        pass

    # Method 2: Environment variable
    if os.environ.get("WSL_DISTRO_NAME"):
        return True

    # Method 3: WSLInterop
    try:
        if Path("/proc/sys/fs/binfmt_misc/WSLInterop").exists():
            return True
    except (OSError, PermissionError):
        pass

    return False


def detect_wsl_memory_limit() -> Optional[float]:
    """
    Detect WSL2's configured memory limit from .wslconfig.

    WSL2 defaults to 50% of host RAM (or 8GB, whichever is less on older
    versions), but users can override this in %USERPROFILE%/.wslconfig.

    Returns the limit in GB, or None if not detectable.
    """
    # WSL2 exposes the effective limit through /proc/meminfo
    # The "MemTotal" in WSL2 already reflects the .wslconfig limit
    if not PSUTIL_AVAILABLE:
        return None
    try:
        mem = psutil.virtual_memory()
        total_gb = mem.total / (1024**3)
        return total_gb
    except Exception:
        return None


def detect_swap_usage() -> Tuple[float, float, float]:
    """
    Returns (swap_used_gb, swap_total_gb, swap_percent).
    In WSL2, high swap usage is an early warning of imminent OOM.
    """
    if not PSUTIL_AVAILABLE:
        return 0.0, 0.0, 0.0
    try:
        swap = psutil.swap_memory()
        used_gb = swap.used / (1024**3)
        total_gb = swap.total / (1024**3)
        percent = swap.percent
        return used_gb, total_gb, percent
    except Exception:
        return 0.0, 0.0, 0.0


def get_system_profile() -> Dict[str, Any]:
    """
    Build a comprehensive system profile for memory management decisions.
    """
    profile = {
        "platform": platform.system(),
        "is_wsl": detect_wsl(),
        "python_version": platform.python_version(),
    }

    # RAM
    if PSUTIL_AVAILABLE:
        try:
            mem = psutil.virtual_memory()
            profile["ram_total_gb"] = round(mem.total / (1024**3), 2)
            profile["ram_available_gb"] = round(mem.available / (1024**3), 2)
            profile["ram_used_percent"] = round(mem.percent, 1)
        except Exception:
            profile["ram_total_gb"] = 0
            profile["ram_available_gb"] = 0
            profile["ram_used_percent"] = 0
    else:
        profile["ram_total_gb"] = 0
        profile["ram_available_gb"] = 0
        profile["ram_used_percent"] = 0

    # Swap
    swap_used, swap_total, swap_pct = detect_swap_usage()
    profile["swap_used_gb"] = round(swap_used, 2)
    profile["swap_total_gb"] = round(swap_total, 2)
    profile["swap_percent"] = round(swap_pct, 1)

    # WSL memory limit
    if profile["is_wsl"]:
        wsl_limit = detect_wsl_memory_limit()
        profile["wsl_memory_limit_gb"] = round(wsl_limit, 2) if wsl_limit else None

    # GPU / VRAM
    profile["gpu_count"] = 0
    profile["gpus"] = []

    if TORCH_AVAILABLE and torch.cuda.is_available():
        profile["gpu_count"] = torch.cuda.device_count()
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            free, total = torch.cuda.mem_get_info(i)
            profile["gpus"].append(
                {
                    "id": i,
                    "name": props.name,
                    "total_vram_gb": round(total / (1024**3), 2),
                    "free_vram_gb": round(free / (1024**3), 2),
                    "used_vram_gb": round((total - free) / (1024**3), 2),
                    "used_vram_percent": round((1 - free / total) * 100, 1)
                    if total > 0
                    else 0,
                }
            )

    return profile


# ─────────────────────────────────────────────────────────────────────────────
# Snapshot — point-in-time memory readings
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class MemorySnapshot:
    """Point-in-time memory state."""

    timestamp: float

    # RAM
    ram_total_gb: float
    ram_available_gb: float
    ram_used_percent: float

    # Swap
    swap_used_gb: float
    swap_total_gb: float
    swap_percent: float

    # VRAM (per GPU)
    gpus: List[Dict[str, float]] = field(default_factory=list)

    # Derived
    pressure: MemoryPressure = MemoryPressure.SAFE

    @property
    def ram_free_gb(self) -> float:
        return self.ram_available_gb

    @property
    def worst_vram_percent(self) -> float:
        if not self.gpus:
            return 0.0
        return max(g.get("used_percent", 0) for g in self.gpus)

    @property
    def total_vram_free_gb(self) -> float:
        return sum(g.get("free_gb", 0) for g in self.gpus)


def take_snapshot() -> MemorySnapshot:
    """Capture current memory state across RAM + all GPUs."""
    if PSUTIL_AVAILABLE:
        mem = psutil.virtual_memory()
        ram_total = round(mem.total / (1024**3), 2)
        ram_avail = round(mem.available / (1024**3), 2)
        ram_pct = round(mem.percent, 1)
    else:
        ram_total = 0.0
        ram_avail = 0.0
        ram_pct = 0.0
    swap_used, swap_total, swap_pct = detect_swap_usage()

    gpus = []
    if TORCH_AVAILABLE and torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            try:
                free, total = torch.cuda.mem_get_info(i)
                gpus.append(
                    {
                        "id": i,
                        "total_gb": round(total / (1024**3), 2),
                        "free_gb": round(free / (1024**3), 2),
                        "used_gb": round((total - free) / (1024**3), 2),
                        "used_percent": round((1 - free / total) * 100, 1)
                        if total > 0
                        else 0,
                    }
                )
            except Exception:
                pass

    return MemorySnapshot(
        timestamp=time.time(),
        ram_total_gb=ram_total,
        ram_available_gb=ram_avail,
        ram_used_percent=ram_pct,
        swap_used_gb=round(swap_used, 2),
        swap_total_gb=round(swap_total, 2),
        swap_percent=round(swap_pct, 1),
        gpus=gpus,
    )


# ─────────────────────────────────────────────────────────────────────────────
# MemoryGuard — the main class
# ─────────────────────────────────────────────────────────────────────────────


class MemoryGuard:
    """
    Centralized memory safety manager with ENFORCEMENT.

    Responsibilities:
        - Detect WSL and choose appropriate thresholds
        - ENFORCE hard limits on CUDA VRAM and process RSS
        - Provide a unified API for all memory checks
        - Graduated pressure response (not just kill)
        - Pre-flight estimation before heavy operations
        - Emergency cleanup (gc + torch.cuda.empty_cache)
        - Background monitor thread with auto-cleanup
        - Safe allocation context manager
    """

    def __init__(
        self,
        thresholds: Optional[MemoryThresholds] = None,
        enforce: bool = True,
        auto_monitor: bool = True,
    ):
        self._is_wsl = detect_wsl()
        self._thresholds = thresholds or (
            WSL_THRESHOLDS if self._is_wsl else NATIVE_THRESHOLDS
        )
        self._profile = get_system_profile()
        self._callbacks: Dict[MemoryPressure, List[Callable]] = {
            p: [] for p in MemoryPressure
        }
        self._monitor_thread: Optional[threading.Thread] = None
        self._monitor_stop = threading.Event()
        self._last_snapshot: Optional[MemorySnapshot] = None
        self._lock = threading.Lock()
        self._limits_enforced = False
        self._cuda_fraction_set = False
        self._auto_cleanup_registered = False

        # Enforce hard limits at the OS/CUDA level
        if enforce:
            self._enforce_limits()

        env = "WSL2" if self._is_wsl else "Native Linux"
        logger.info(
            f"MemoryGuard initialized: {env} | "
            f"RAM: {self._profile.get('ram_total_gb', '?')}GB | "
            f"GPUs: {self._profile.get('gpu_count', 0)} | "
            f"RAM deadly threshold: {self._thresholds.ram_deadly}% | "
            f"Limits enforced: {self._limits_enforced}"
        )

        # Log .wslconfig advisory for WSL users
        if self._is_wsl:
            self._log_wslconfig_advisory()

        # Auto-register cleanup callbacks when enforcement is enabled
        if enforce:
            self._register_auto_cleanup()

        # Auto-start background monitor if requested
        if auto_monitor:
            self.start_monitor()

    # ── Properties ──────────────────────────────────────────────────────

    @property
    def is_wsl(self) -> bool:
        return self._is_wsl

    @property
    def thresholds(self) -> MemoryThresholds:
        return self._thresholds

    @property
    def profile(self) -> Dict[str, Any]:
        return self._profile

    # ── Enforcement ──────────────────────────────────────────────────────

    def _enforce_limits(self):
        """
        Set hard memory limits at the OS and CUDA level.

        This is the critical piece that was MISSING — the guard was advisory
        only. These limits cause MemoryError / torch.cuda.OutOfMemoryError
        BEFORE the OOM killer has a chance to crash WSL.

        1. CUDA: torch.cuda.set_per_process_memory_fraction()
           Caps PyTorch VRAM usage to (deadly_threshold - buffer)% of total.

        2. Process RSS: resource.setrlimit(RLIMIT_AS)
           Caps the process virtual address space so malloc fails with
           MemoryError instead of the kernel OOM-killing the process.
        """
        enforced_any = False

        # ── 1. CUDA VRAM hard cap ──
        if TORCH_AVAILABLE and torch.cuda.is_available():
            try:
                # Calculate fraction: leave buffer_gb free on each GPU
                for i in range(torch.cuda.device_count()):
                    props = torch.cuda.get_device_properties(i)
                    total_vram_gb = props.total_memory / (1024**3)

                    # The fraction we allow PyTorch to use
                    # deadly_threshold is the % at which we consider it fatal
                    # We set the hard cap BELOW that to prevent ever reaching it
                    buffer_gb = self._thresholds.min_vram_headroom_gb
                    usable_gb = total_vram_gb - buffer_gb
                    fraction = max(0.5, min(0.95, usable_gb / total_vram_gb))

                    torch.cuda.set_per_process_memory_fraction(fraction, i)
                    self._cuda_fraction_set = True
                    enforced_any = True
                    logger.info(
                        f"CUDA GPU[{i}] hard limit set: "
                        f"{fraction:.2%} of {total_vram_gb:.1f}GB = "
                        f"{usable_gb:.1f}GB usable, {buffer_gb:.1f}GB reserved"
                    )
            except Exception as e:
                logger.warning(f"Failed to set CUDA memory fraction: {e}")

        # ── 2. Process virtual memory cap (Linux/WSL) ──
        if not PSUTIL_AVAILABLE:
            logger.debug("psutil not available, skipping process memory limit")
        else:
            try:
                import resource

                mem = psutil.virtual_memory()
                total_ram_bytes = mem.total
                total_ram_gb = total_ram_bytes / (1024**3)

                # Leave min_ram_headroom_gb free for the OS/WSL
                buffer_bytes = int(self._thresholds.min_ram_headroom_gb * (1024**3))
                limit_bytes = total_ram_bytes - buffer_bytes

                # Don't set a limit lower than 4GB (would break Python itself)
                min_limit = 4 * (1024**3)
                if limit_bytes < min_limit:
                    logger.warning(
                        f"RAM too low to enforce limit: {total_ram_gb:.1f}GB total, "
                        f"{self._thresholds.min_ram_headroom_gb}GB headroom. Skipping."
                    )
                else:
                    # Set soft limit (hard limit stays at system max)
                    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
                    # Only tighten, never loosen an existing limit
                    if hard == resource.RLIM_INFINITY or limit_bytes < hard:
                        resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, hard))
                        enforced_any = True
                        limit_gb = limit_bytes / (1024**3)
                        logger.info(
                            f"Process RAM soft limit set: "
                            f"{limit_gb:.1f}GB (total {total_ram_gb:.1f}GB - "
                            f"{self._thresholds.min_ram_headroom_gb:.1f}GB buffer)"
                        )
                    else:
                        logger.info(
                            f"Existing RLIMIT_AS ({hard / (1024**3):.1f}GB) is "
                            f"tighter than our limit. Keeping existing."
                        )
            except ImportError:
                logger.debug("resource module not available (not Linux/WSL)")
            except (ValueError, OSError) as e:
                logger.warning(f"Failed to set process memory limit: {e}")

        self._limits_enforced = enforced_any

    def _log_wslconfig_advisory(self):
        """
        Log advisory for WSL users about optimal .wslconfig settings.
        """
        ram_gb = self._profile.get("ram_total_gb", 0)
        gpu_count = self._profile.get("gpu_count", 0)

        # WSL2 default is min(50% host RAM, 8GB) on older builds,
        # or 50% on newer builds. For 32GB host, that's 16GB.
        recommended_ram = max(8, int(ram_gb * 0.85))  # Use up to 85% for WSL VM
        recommended_swap = max(4, int(ram_gb * 0.25))  # 25% for swap

        logger.info(
            f"WSL2 detected with {ram_gb:.0f}GB visible RAM. "
            f"Recommended .wslconfig (at %%USERPROFILE%%\\.wslconfig):\n"
            f"  [wsl2]\n"
            f"  memory={recommended_ram}GB\n"
            f"  swap={recommended_swap}GB\n"
            f"  processors={os.cpu_count() or 4}\n"
            f"  {'gpuSupport=true' if gpu_count > 0 else ''}\n"
            f"  localhostForwarding=true\n"
            f"Run 'wsl --shutdown' after editing .wslconfig for changes to take effect."
        )

    def _register_auto_cleanup(self):
        """
        Register a pressure callback that auto-runs emergency_cleanup
        when pressure reaches CRITICAL or DEADLY.
        """
        if self._auto_cleanup_registered:
            return

        def _auto_cleanup_cb(snap: "MemorySnapshot"):
            logger.warning(
                f"Auto-cleanup triggered at {snap.pressure.value} pressure "
                f"(RAM {snap.ram_used_percent:.1f}%, VRAM {snap.worst_vram_percent:.1f}%)"
            )
            self.emergency_cleanup(aggressive=(snap.pressure == MemoryPressure.DEADLY))

        self.on_pressure(MemoryPressure.CRITICAL, _auto_cleanup_cb)
        self.on_pressure(MemoryPressure.DEADLY, _auto_cleanup_cb)
        self._auto_cleanup_registered = True
        logger.debug(
            "Auto-cleanup callbacks registered for CRITICAL and DEADLY pressure"
        )

    # ── Safe Allocation Context Manager ──────────────────────────────────

    class _SafeAllocationContext:
        """
        Context manager that checks memory before and after a block.
        If pre-check fails, raises MemoryError instead of proceeding.
        If post-check shows CRITICAL+, runs emergency cleanup.
        """

        def __init__(
            self,
            guard_ref,
            estimated_ram_gb: float,
            estimated_vram_gb: float,
            operation: str,
        ):
            self._guard = guard_ref
            self._est_ram = estimated_ram_gb
            self._est_vram = estimated_vram_gb
            self._operation = operation

        def __enter__(self):
            safe, msg = self._guard.preflight_check(
                estimated_ram_gb=self._est_ram,
                estimated_vram_gb=self._est_vram,
                operation_name=self._operation,
            )
            if not safe:
                # Try cleanup first
                self._guard.emergency_cleanup(aggressive=True)
                safe2, msg2 = self._guard.preflight_check(
                    estimated_ram_gb=self._est_ram,
                    estimated_vram_gb=self._est_vram,
                    operation_name=f"{self._operation} (post-cleanup)",
                )
                if not safe2:
                    raise MemoryError(
                        f"Insufficient memory for '{self._operation}': {msg2}"
                    )
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            # Post-operation cleanup if pressure is high
            pressure = self._guard.get_pressure()
            if pressure in (MemoryPressure.CRITICAL, MemoryPressure.DEADLY):
                logger.warning(
                    f"Post-operation pressure {pressure.value} after "
                    f"'{self._operation}'. Running cleanup."
                )
                self._guard.emergency_cleanup(aggressive=True)
            return False  # Don't suppress exceptions

    def safe_allocate(
        self,
        estimated_ram_gb: float = 0.0,
        estimated_vram_gb: float = 0.0,
        operation: str = "allocation",
    ):
        """
        Context manager for safe memory allocation.

        Usage:
            with guard.safe_allocate(ram_gb=4.0, vram_gb=8.0, operation="load model"):
                model = load_large_model()

        Raises MemoryError if allocation would exceed safe limits.
        Runs cleanup after if pressure is high.
        """
        return self._SafeAllocationContext(
            self, estimated_ram_gb, estimated_vram_gb, operation
        )

    # ── Core API ────────────────────────────────────────────────────────

    def snapshot(self) -> MemorySnapshot:
        """Take a fresh memory snapshot and classify pressure."""
        snap = take_snapshot()
        snap.pressure = self._classify_pressure(snap)
        with self._lock:
            self._last_snapshot = snap
        return snap

    def get_pressure(self) -> MemoryPressure:
        """Get current memory pressure level (takes a fresh snapshot)."""
        return self.snapshot().pressure

    def is_safe(
        self, min_ram_gb: Optional[float] = None, min_vram_gb: Optional[float] = None
    ) -> bool:
        """
        Quick safety check.

        Returns True if memory pressure is acceptable for new allocations.
        On WSL, blocks at HIGH pressure too (not just CRITICAL/DEADLY)
        because the WSL2 OOM killer triggers earlier than native Linux.
        Optionally checks absolute headroom requirements.
        """
        snap = self.snapshot()

        # On WSL, be stricter — block at HIGH because WSL OOM killer
        # can activate before we reach CRITICAL
        if self._is_wsl:
            if snap.pressure in (
                MemoryPressure.HIGH,
                MemoryPressure.CRITICAL,
                MemoryPressure.DEADLY,
            ):
                return False
        else:
            if snap.pressure in (MemoryPressure.CRITICAL, MemoryPressure.DEADLY):
                return False

        min_ram = min_ram_gb or self._thresholds.min_ram_headroom_gb
        if snap.ram_available_gb < min_ram:
            return False

        if min_vram_gb is not None and snap.gpus:
            min_vram = min_vram_gb or self._thresholds.min_vram_headroom_gb
            if snap.total_vram_free_gb < min_vram:
                return False

        return True

    def preflight_check(
        self,
        estimated_ram_gb: float = 0.0,
        estimated_vram_gb: float = 0.0,
        operation_name: str = "operation",
    ) -> Tuple[bool, str]:
        """
        Pre-flight memory estimation before an expensive operation.

        Returns (safe_to_proceed, message).
        Checks whether the estimated allocation would push us into
        CRITICAL or DEADLY pressure.
        """
        snap = self.snapshot()
        issues = []

        # RAM check
        projected_ram_free = snap.ram_available_gb - estimated_ram_gb
        if projected_ram_free < self._thresholds.min_ram_headroom_gb:
            issues.append(
                f"RAM: {operation_name} needs ~{estimated_ram_gb:.1f}GB but only "
                f"{snap.ram_available_gb:.1f}GB free (need {self._thresholds.min_ram_headroom_gb:.1f}GB headroom)"
            )

        # VRAM check
        if estimated_vram_gb > 0 and snap.gpus:
            max_free_vram = max(g.get("free_gb", 0) for g in snap.gpus)
            if (
                estimated_vram_gb
                > max_free_vram - self._thresholds.min_vram_headroom_gb
            ):
                issues.append(
                    f"VRAM: {operation_name} needs ~{estimated_vram_gb:.1f}GB but only "
                    f"{max_free_vram:.1f}GB free (need {self._thresholds.min_vram_headroom_gb:.1f}GB headroom)"
                )

        # Swap warning (WSL-specific)
        if self._is_wsl and snap.swap_percent > 50.0:
            issues.append(
                f"WSL Swap: {snap.swap_percent:.0f}% used ({snap.swap_used_gb:.1f}GB) — "
                f"high swap in WSL2 signals imminent OOM"
            )

        if issues:
            msg = f"PREFLIGHT FAILED for '{operation_name}': " + " | ".join(issues)
            logger.warning(msg)
            return False, msg

        msg = (
            f"PREFLIGHT OK for '{operation_name}': "
            f"RAM {snap.ram_available_gb:.1f}GB free, "
            f"VRAM {snap.total_vram_free_gb:.1f}GB free"
        )
        logger.info(msg)
        return True, msg

    def get_thresholds(self) -> Dict[str, float]:
        """
        Export thresholds as a plain dict for other modules.

        This is how sliding_window_buffer, prefetch_engine, and health.py
        should get their thresholds instead of hardcoding them.
        """
        t = self._thresholds
        return {
            "ram_elevated_percent": t.ram_elevated,
            "ram_high_percent": t.ram_high,
            "ram_critical_percent": t.ram_critical,
            "ram_deadly_percent": t.ram_deadly,
            "vram_elevated_percent": t.vram_elevated,
            "vram_high_percent": t.vram_high,
            "vram_critical_percent": t.vram_critical,
            "vram_deadly_percent": t.vram_deadly,
            "min_ram_headroom_gb": t.min_ram_headroom_gb,
            "min_vram_headroom_gb": t.min_vram_headroom_gb,
            "check_interval_seconds": t.check_interval_seconds,
            "fast_check_interval_seconds": t.fast_check_interval_seconds,
            "is_wsl": self._is_wsl,
        }

    # ── Emergency Cleanup ───────────────────────────────────────────────

    def emergency_cleanup(self, aggressive: bool = False) -> Dict[str, float]:
        """
        Run emergency memory cleanup.

        Args:
            aggressive: If True, also clears CUDA caches and forces GC on
                        all generations (slower but recovers more memory).

        Returns dict with before/after measurements.
        """
        before = take_snapshot()
        logger.warning(
            f"Emergency cleanup triggered (aggressive={aggressive}). "
            f"RAM: {before.ram_used_percent:.1f}% | "
            f"VRAM worst: {before.worst_vram_percent:.1f}%"
        )

        # Step 1: Python GC
        if aggressive:
            gc.collect(0)
            gc.collect(1)
            gc.collect(2)
        else:
            gc.collect()

        # Step 2: CUDA cleanup
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()
            if aggressive:
                torch.cuda.synchronize()
                # Second pass after sync
                gc.collect()
                torch.cuda.empty_cache()

        after = take_snapshot()
        ram_freed = before.ram_used_percent - after.ram_used_percent
        vram_freed = before.worst_vram_percent - after.worst_vram_percent

        logger.info(
            f"Cleanup complete. "
            f"RAM: {before.ram_used_percent:.1f}% → {after.ram_used_percent:.1f}% (freed {ram_freed:+.1f}%) | "
            f"VRAM: {before.worst_vram_percent:.1f}% → {after.worst_vram_percent:.1f}% (freed {vram_freed:+.1f}%)"
        )

        return {
            "ram_before_pct": before.ram_used_percent,
            "ram_after_pct": after.ram_used_percent,
            "ram_freed_pct": ram_freed,
            "vram_before_pct": before.worst_vram_percent,
            "vram_after_pct": after.worst_vram_percent,
            "vram_freed_pct": vram_freed,
        }

    # ── Callbacks ───────────────────────────────────────────────────────

    def on_pressure(
        self, level: MemoryPressure, callback: Callable[[MemorySnapshot], None]
    ):
        """Register a callback for when a pressure level is reached."""
        self._callbacks[level].append(callback)

    def _fire_callbacks(self, snap: MemorySnapshot):
        """Fire all registered callbacks for the current pressure level."""
        for cb in self._callbacks.get(snap.pressure, []):
            try:
                cb(snap)
            except Exception as e:
                logger.error(f"Memory pressure callback error: {e}")

    # ── Pressure Classification ─────────────────────────────────────────

    def _classify_pressure(self, snap: MemorySnapshot) -> MemoryPressure:
        """
        Classify memory pressure using both RAM and VRAM.
        Uses the WORST of the two to determine the level.
        Also factors in swap usage on WSL.
        """
        t = self._thresholds

        # RAM pressure
        ram_pct = snap.ram_used_percent
        if ram_pct >= t.ram_deadly:
            ram_level = MemoryPressure.DEADLY
        elif ram_pct >= t.ram_critical:
            ram_level = MemoryPressure.CRITICAL
        elif ram_pct >= t.ram_high:
            ram_level = MemoryPressure.HIGH
        elif ram_pct >= t.ram_elevated:
            ram_level = MemoryPressure.ELEVATED
        else:
            ram_level = MemoryPressure.SAFE

        # VRAM pressure (worst GPU)
        vram_pct = snap.worst_vram_percent
        if vram_pct >= t.vram_deadly:
            vram_level = MemoryPressure.DEADLY
        elif vram_pct >= t.vram_critical:
            vram_level = MemoryPressure.CRITICAL
        elif vram_pct >= t.vram_high:
            vram_level = MemoryPressure.HIGH
        elif vram_pct >= t.vram_elevated:
            vram_level = MemoryPressure.ELEVATED
        else:
            vram_level = MemoryPressure.SAFE

        # Absolute headroom check
        headroom_level = MemoryPressure.SAFE
        if snap.ram_available_gb < t.min_ram_headroom_gb:
            headroom_level = MemoryPressure.CRITICAL

        # WSL swap escalation: >50% swap used means we're in trouble
        swap_level = MemoryPressure.SAFE
        if self._is_wsl and snap.swap_percent > 70:
            swap_level = MemoryPressure.CRITICAL
        elif self._is_wsl and snap.swap_percent > 50:
            swap_level = MemoryPressure.HIGH

        # Return the worst
        levels = [ram_level, vram_level, headroom_level, swap_level]
        pressure_order = [
            MemoryPressure.SAFE,
            MemoryPressure.ELEVATED,
            MemoryPressure.HIGH,
            MemoryPressure.CRITICAL,
            MemoryPressure.DEADLY,
        ]
        return max(levels, key=lambda l: pressure_order.index(l))

    # ── Background Monitor ──────────────────────────────────────────────

    def start_monitor(self, on_pressure_change: Optional[Callable] = None):
        """
        Start background monitoring thread.

        The thread takes snapshots at the configured interval and fires
        callbacks when pressure changes. Interval speeds up when pressure
        is HIGH or above.
        """
        if self._monitor_thread and self._monitor_thread.is_alive():
            logger.warning("Monitor already running")
            return

        self._monitor_stop.clear()

        def _monitor_loop():
            last_pressure = MemoryPressure.SAFE
            while not self._monitor_stop.is_set():
                snap = self.snapshot()

                # Fire callbacks
                self._fire_callbacks(snap)

                # Pressure change notification
                if snap.pressure != last_pressure:
                    logger.info(
                        f"Memory pressure changed: {last_pressure.value} → {snap.pressure.value} | "
                        f"RAM: {snap.ram_used_percent:.1f}% | VRAM: {snap.worst_vram_percent:.1f}%"
                    )
                    if on_pressure_change:
                        try:
                            on_pressure_change(last_pressure, snap.pressure, snap)
                        except Exception as e:
                            logger.error(f"Pressure change callback error: {e}")
                    last_pressure = snap.pressure

                # Adaptive interval
                if snap.pressure in (
                    MemoryPressure.HIGH,
                    MemoryPressure.CRITICAL,
                    MemoryPressure.DEADLY,
                ):
                    interval = self._thresholds.fast_check_interval_seconds
                else:
                    interval = self._thresholds.check_interval_seconds

                self._monitor_stop.wait(interval)

        self._monitor_thread = threading.Thread(
            target=_monitor_loop, daemon=True, name="memory-guard"
        )
        try:
            self._monitor_thread.start()
            logger.info("Memory guard background monitor started")
        except RuntimeError:
            # Thread creation can fail in constrained environments
            self._monitor_thread = None
            logger.warning(
                "Memory guard: could not start monitor thread (thread limit reached)"
            )

    def stop_monitor(self):
        """Stop the background monitoring thread."""
        self._monitor_stop.set()
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)
            self._monitor_thread = None
        logger.info("Memory guard monitor stopped")

    # ── Utility ─────────────────────────────────────────────────────────

    def print_status(self):
        """Print a human-readable memory status report."""
        snap = self.snapshot()
        env = "WSL2" if self._is_wsl else "Native"

        print(f"\n{'=' * 60}")
        print(f"  Memory Guard Status — {env}")
        print(f"{'=' * 60}")
        print(f"  Pressure:  {snap.pressure.value.upper()}")
        print(
            f"  RAM:       {snap.ram_used_percent:.1f}% used "
            f"({snap.ram_available_gb:.1f}GB free / {snap.ram_total_gb:.1f}GB total)"
        )
        print(
            f"  Swap:      {snap.swap_percent:.1f}% used "
            f"({snap.swap_used_gb:.1f}GB / {snap.swap_total_gb:.1f}GB)"
        )

        for gpu in snap.gpus:
            print(
                f"  GPU[{gpu['id']}]:    {gpu['used_percent']:.1f}% used "
                f"({gpu['free_gb']:.1f}GB free / {gpu['total_gb']:.1f}GB total)"
            )

        t = self._thresholds
        print(f"\n  Thresholds ({env}):")
        print(
            f"    RAM:  elevated={t.ram_elevated}% | high={t.ram_high}% | "
            f"critical={t.ram_critical}% | deadly={t.ram_deadly}%"
        )
        print(
            f"    VRAM: elevated={t.vram_elevated}% | high={t.vram_high}% | "
            f"critical={t.vram_critical}% | deadly={t.vram_deadly}%"
        )
        print(
            f"    Min headroom: RAM {t.min_ram_headroom_gb}GB | VRAM {t.min_vram_headroom_gb}GB"
        )
        print(f"{'=' * 60}\n")

    def __repr__(self) -> str:
        env = "WSL2" if self._is_wsl else "Native"
        snap = self._last_snapshot
        if snap:
            return (
                f"<MemoryGuard {env} pressure={snap.pressure.value} "
                f"ram={snap.ram_used_percent:.0f}% vram={snap.worst_vram_percent:.0f}%>"
            )
        return f"<MemoryGuard {env} (no snapshot yet)>"


# ─────────────────────────────────────────────────────────────────────────────
# Module-level singleton
# ─────────────────────────────────────────────────────────────────────────────

# Detect if we're running under a test harness — skip enforcement to avoid
# setting real OS limits (RLIMIT_AS, CUDA fraction) on the test process.
_IN_TEST = (
    "PYTEST_CURRENT_TEST" in os.environ
    or "pytest" in os.environ.get("_", "")
    or "_pytest" in __import__("sys").modules
)

guard = MemoryGuard(
    enforce=not _IN_TEST,
    auto_monitor=True,
)
"""
Module-level singleton. Import and use:
    from nexus.utils.memory_guard import guard

The singleton enforces limits and auto-starts the background monitor.
The monitor thread is a daemon thread and will not block process exit.
Under test harnesses (pytest), enforcement is skipped automatically.
"""


# ─────────────────────────────────────────────────────────────────────────────
# Convenience functions
# ─────────────────────────────────────────────────────────────────────────────


def is_wsl() -> bool:
    """Quick check: are we running in WSL2?"""
    return guard.is_wsl


def is_safe(
    min_ram_gb: Optional[float] = None, min_vram_gb: Optional[float] = None
) -> bool:
    """Quick check: is memory pressure acceptable?"""
    return guard.is_safe(min_ram_gb=min_ram_gb, min_vram_gb=min_vram_gb)


def get_pressure() -> MemoryPressure:
    """Get current memory pressure level."""
    return guard.get_pressure()


def preflight(
    estimated_ram_gb: float = 0, estimated_vram_gb: float = 0, name: str = "operation"
) -> Tuple[bool, str]:
    """Pre-flight memory check for an operation."""
    return guard.preflight_check(estimated_ram_gb, estimated_vram_gb, name)


def cleanup(aggressive: bool = False) -> Dict[str, float]:
    """Emergency memory cleanup."""
    return guard.emergency_cleanup(aggressive=aggressive)


if __name__ == "__main__":
    guard.print_status()
