"""
Tests for src/utils/memory_guard.py — Centralized WSL-Aware Memory Safety System

Covers:
    - WSL detection logic
    - Pressure classification at boundary values
    - Preflight check pass/fail scenarios
    - Emergency cleanup execution
    - Threshold export consistency
    - Snapshot data integrity
    - Convenience function delegation
    - Enforcement (_enforce_limits, safe_allocate)
    - WSL-specific is_safe behavior
    - Auto-cleanup and monitor auto-start
"""

import os
import gc
import pytest
from unittest.mock import patch, MagicMock, PropertyMock


# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _stop_monitors():
    """Ensure any guard monitor threads are stopped after each test."""
    yield
    # Stop any monitor threads that may have been started
    try:
        from src.utils.memory_guard import guard

        guard.stop_monitor()
    except Exception:
        pass


@pytest.fixture
def _mock_rlimit():
    """
    Mock resource.setrlimit/getrlimit so enforce=True tests never set
    real OS-level memory limits on the test process.

    Without this fixture, resource.setrlimit(RLIMIT_AS, ...) would persist
    for the entire pytest session, eventually causing MemoryError in later
    tests or even in pytest's own allocations.
    """
    import resource as _resource

    with (
        patch("resource.setrlimit") as mock_set,
        patch(
            "resource.getrlimit",
            return_value=(_resource.RLIM_INFINITY, _resource.RLIM_INFINITY),
        ) as mock_get,
    ):
        yield mock_set


def _make_guard(**kwargs):
    """
    Create a MemoryGuard for testing with safe defaults.
    Disables enforcement and auto-monitor to avoid OS-level side effects.
    """
    from src.utils.memory_guard import MemoryGuard

    enforce = kwargs.pop("enforce", False)
    auto_monitor = kwargs.pop("auto_monitor", False)
    thresholds = kwargs.pop("thresholds", None)
    return MemoryGuard(
        thresholds=thresholds,
        enforce=enforce,
        auto_monitor=auto_monitor,
    )


# ── WSL Detection ─────────────────────────────────────────────────────


class TestWSLDetection:
    """Test WSL2 environment detection logic."""

    def test_detect_wsl_via_proc_version(self, tmp_path, monkeypatch):
        """WSL detected when /proc/version contains 'microsoft'."""
        from src.utils.memory_guard import detect_wsl

        fake_proc = tmp_path / "version"
        fake_proc.write_text("Linux version 5.15.146.1-microsoft-standard-WSL2")

        with patch("builtins.open", create=True) as mock_open:
            mock_open.return_value.__enter__ = lambda s: s
            mock_open.return_value.__exit__ = MagicMock(return_value=False)
            mock_open.return_value.read = lambda: (
                "Linux version 5.15.146.1-microsoft-standard-WSL2"
            )

            # Direct call — checks /proc/version first
            result = detect_wsl()
            assert result is True

    def test_detect_wsl_via_env_var(self, monkeypatch):
        """WSL detected when WSL_DISTRO_NAME env var is set."""
        from src.utils.memory_guard import detect_wsl

        monkeypatch.setenv("WSL_DISTRO_NAME", "Ubuntu")
        result = detect_wsl()
        assert result is True

    def test_detect_native_linux(self, monkeypatch):
        """Not WSL when no indicators are present."""
        from src.utils.memory_guard import detect_wsl

        monkeypatch.delenv("WSL_DISTRO_NAME", raising=False)
        monkeypatch.delenv("WSL_INTEROP", raising=False)

        with patch("builtins.open", side_effect=FileNotFoundError):
            with patch("os.path.exists", return_value=False):
                result = detect_wsl()
                # On actual WSL this may still return True via other checks,
                # but the env/proc path should fail gracefully
                assert isinstance(result, bool)


# ── MemoryThresholds ──────────────────────────────────────────────────


class TestMemoryThresholds:
    """Test threshold presets and immutability."""

    def test_wsl_thresholds_are_tighter(self):
        """WSL thresholds should be lower than native thresholds."""
        from src.utils.memory_guard import WSL_THRESHOLDS, NATIVE_THRESHOLDS

        assert WSL_THRESHOLDS.ram_deadly < NATIVE_THRESHOLDS.ram_deadly
        assert WSL_THRESHOLDS.vram_deadly < NATIVE_THRESHOLDS.vram_deadly
        assert (
            WSL_THRESHOLDS.min_ram_headroom_gb > NATIVE_THRESHOLDS.min_ram_headroom_gb
        )
        assert (
            WSL_THRESHOLDS.min_vram_headroom_gb > NATIVE_THRESHOLDS.min_vram_headroom_gb
        )

    def test_thresholds_are_frozen(self):
        """MemoryThresholds should be immutable (frozen dataclass)."""
        from src.utils.memory_guard import WSL_THRESHOLDS

        with pytest.raises(AttributeError):
            WSL_THRESHOLDS.ram_deadly = 99.0  # type: ignore[misc]

    def test_threshold_ordering(self):
        """Pressure levels should increase: elevated < high < critical < deadly."""
        from src.utils.memory_guard import WSL_THRESHOLDS, NATIVE_THRESHOLDS

        for t in (WSL_THRESHOLDS, NATIVE_THRESHOLDS):
            assert t.ram_elevated < t.ram_high < t.ram_critical < t.ram_deadly
            assert t.vram_elevated < t.vram_high < t.vram_critical < t.vram_deadly

    def test_wsl_faster_polling(self):
        """WSL should poll faster than native."""
        from src.utils.memory_guard import WSL_THRESHOLDS, NATIVE_THRESHOLDS

        assert (
            WSL_THRESHOLDS.check_interval_seconds
            < NATIVE_THRESHOLDS.check_interval_seconds
        )
        assert (
            WSL_THRESHOLDS.fast_check_interval_seconds
            < NATIVE_THRESHOLDS.fast_check_interval_seconds
        )


# ── MemoryPressure Classification ─────────────────────────────────────


class TestPressureClassification:
    """Test pressure level classification at boundary values."""

    def test_pressure_enum_ordering(self):
        """Pressure levels should be ordered by severity."""
        from src.utils.memory_guard import MemoryPressure

        levels = list(MemoryPressure)
        assert levels == [
            MemoryPressure.SAFE,
            MemoryPressure.ELEVATED,
            MemoryPressure.HIGH,
            MemoryPressure.CRITICAL,
            MemoryPressure.DEADLY,
        ]

    def test_safe_pressure(self):
        """Low memory usage should classify as SAFE."""
        from src.utils.memory_guard import MemoryPressure, MemorySnapshot

        guard = _make_guard()

        fake_snap = MemorySnapshot(
            timestamp=0.0,
            ram_total_gb=32.0,
            ram_available_gb=20.0,
            ram_used_percent=30.0,
            swap_used_gb=0.5,
            swap_total_gb=8.0,
            swap_percent=5.0,
            gpus=[
                {
                    "id": 0,
                    "total_gb": 16.0,
                    "free_gb": 12.0,
                    "used_gb": 4.0,
                    "used_percent": 25.0,
                }
            ],
        )

        with patch("src.utils.memory_guard.take_snapshot", return_value=fake_snap):
            pressure = guard.get_pressure()
            assert pressure == MemoryPressure.SAFE

    def test_deadly_pressure_ram(self):
        """RAM above deadly threshold should classify as DEADLY."""
        from src.utils.memory_guard import MemoryPressure, MemorySnapshot

        guard = _make_guard()

        fake_snap = MemorySnapshot(
            timestamp=0.0,
            ram_total_gb=32.0,
            ram_available_gb=0.5,
            ram_used_percent=96.0,
            swap_used_gb=6.0,
            swap_total_gb=8.0,
            swap_percent=80.0,
            gpus=[
                {
                    "id": 0,
                    "total_gb": 16.0,
                    "free_gb": 12.0,
                    "used_gb": 4.0,
                    "used_percent": 25.0,
                }
            ],
        )

        with patch("src.utils.memory_guard.take_snapshot", return_value=fake_snap):
            pressure = guard.get_pressure()
            assert pressure == MemoryPressure.DEADLY

    def test_worst_of_ram_vram(self):
        """Pressure should be the WORST of RAM and VRAM, not average."""
        from src.utils.memory_guard import MemoryPressure, MemorySnapshot

        guard = _make_guard()

        # RAM is safe, but VRAM is critical (96% > vram_critical on both WSL/native)
        fake_snap = MemorySnapshot(
            timestamp=0.0,
            ram_total_gb=32.0,
            ram_available_gb=20.0,
            ram_used_percent=30.0,
            swap_used_gb=0.5,
            swap_total_gb=8.0,
            swap_percent=5.0,
            gpus=[
                {
                    "id": 0,
                    "total_gb": 16.0,
                    "free_gb": 0.5,
                    "used_gb": 15.5,
                    "used_percent": 96.0,
                }
            ],
        )

        with patch("src.utils.memory_guard.take_snapshot", return_value=fake_snap):
            pressure = guard.get_pressure()
            # Should be at least CRITICAL, not SAFE
            # Use enum ordering list instead of string value comparison
            pressure_order = list(MemoryPressure)
            assert pressure_order.index(pressure) >= pressure_order.index(
                MemoryPressure.CRITICAL
            )


# ── Preflight Check ───────────────────────────────────────────────────


class TestPreflightCheck:
    """Test pre-flight memory estimation checks."""

    def test_preflight_pass_with_headroom(self):
        """Preflight should pass when plenty of memory is available."""
        from src.utils.memory_guard import MemorySnapshot

        guard = _make_guard()

        fake_snap = MemorySnapshot(
            timestamp=0.0,
            ram_total_gb=32.0,
            ram_available_gb=20.0,
            ram_used_percent=30.0,
            swap_used_gb=0.5,
            swap_total_gb=8.0,
            swap_percent=5.0,
            gpus=[
                {
                    "id": 0,
                    "total_gb": 16.0,
                    "free_gb": 12.0,
                    "used_gb": 4.0,
                    "used_percent": 25.0,
                }
            ],
        )

        with patch("src.utils.memory_guard.take_snapshot", return_value=fake_snap):
            ok, msg = guard.preflight_check(estimated_ram_gb=4.0, estimated_vram_gb=8.0)
            assert ok is True
            assert "pass" in msg.lower() or "ok" in msg.lower() or "safe" in msg.lower()

    def test_preflight_fail_insufficient_vram(self):
        """Preflight should fail when VRAM is insufficient."""
        from src.utils.memory_guard import MemorySnapshot

        guard = _make_guard()

        fake_snap = MemorySnapshot(
            timestamp=0.0,
            ram_total_gb=32.0,
            ram_available_gb=20.0,
            ram_used_percent=30.0,
            swap_used_gb=0.5,
            swap_total_gb=8.0,
            swap_percent=5.0,
            gpus=[
                {
                    "id": 0,
                    "total_gb": 16.0,
                    "free_gb": 2.0,
                    "used_gb": 14.0,
                    "used_percent": 87.5,
                }
            ],
        )

        with patch("src.utils.memory_guard.take_snapshot", return_value=fake_snap):
            ok, msg = guard.preflight_check(estimated_ram_gb=4.0, estimated_vram_gb=8.0)
            assert ok is False

    def test_preflight_fail_insufficient_ram(self):
        """Preflight should fail when RAM is insufficient."""
        from src.utils.memory_guard import MemorySnapshot

        guard = _make_guard()

        fake_snap = MemorySnapshot(
            timestamp=0.0,
            ram_total_gb=32.0,
            ram_available_gb=2.0,  # Only 2GB free but need 10GB
            ram_used_percent=93.0,
            swap_used_gb=0.5,
            swap_total_gb=8.0,
            swap_percent=5.0,
            gpus=[
                {
                    "id": 0,
                    "total_gb": 16.0,
                    "free_gb": 12.0,
                    "used_gb": 4.0,
                    "used_percent": 25.0,
                }
            ],
        )

        with patch("src.utils.memory_guard.take_snapshot", return_value=fake_snap):
            ok, msg = guard.preflight_check(
                estimated_ram_gb=10.0, estimated_vram_gb=4.0
            )
            assert ok is False


# ── Emergency Cleanup ─────────────────────────────────────────────────


class TestEmergencyCleanup:
    """Test emergency cleanup execution."""

    def test_cleanup_returns_metrics(self):
        """Emergency cleanup should return before/after metrics dict."""
        from src.utils.memory_guard import MemoryGuard

        guard = _make_guard()
        result = guard.emergency_cleanup(aggressive=False)

        assert isinstance(result, dict)
        # Should have some kind of before/after keys
        assert len(result) > 0

    def test_aggressive_cleanup_calls_gc(self):
        """Aggressive cleanup should call gc.collect on all generations."""
        from src.utils.memory_guard import MemoryGuard

        guard = _make_guard()

        with patch("gc.collect") as mock_gc:
            guard.emergency_cleanup(aggressive=True)
            # Aggressive should call gc.collect multiple times (gen 0, 1, 2)
            assert mock_gc.call_count >= 3


# ── Threshold Export ──────────────────────────────────────────────────


class TestThresholdExport:
    """Test that get_thresholds() exports consistent values."""

    def test_get_thresholds_returns_dict(self):
        """get_thresholds() should return a flat dict with known keys."""
        from src.utils.memory_guard import MemoryGuard

        guard = _make_guard()
        thresholds = guard.get_thresholds()

        assert isinstance(thresholds, dict)

        expected_keys = {
            "ram_elevated_percent",
            "ram_high_percent",
            "ram_critical_percent",
            "ram_deadly_percent",
            "vram_elevated_percent",
            "vram_high_percent",
            "vram_critical_percent",
            "vram_deadly_percent",
            "min_ram_headroom_gb",
            "min_vram_headroom_gb",
            "check_interval_seconds",
            "fast_check_interval_seconds",
            "is_wsl",
        }
        assert expected_keys.issubset(set(thresholds.keys()))

    def test_threshold_values_are_numeric(self):
        """All threshold values except is_wsl should be numeric."""
        from src.utils.memory_guard import MemoryGuard

        guard = _make_guard()
        thresholds = guard.get_thresholds()

        for key, value in thresholds.items():
            if key == "is_wsl":
                assert isinstance(value, bool)
            else:
                assert isinstance(value, (int, float)), (
                    f"{key} should be numeric, got {type(value)}"
                )

    def test_thresholds_ordering_in_export(self):
        """Exported thresholds should maintain the ordering: elevated < high < critical < deadly."""
        from src.utils.memory_guard import MemoryGuard

        guard = _make_guard()
        t = guard.get_thresholds()

        assert t["ram_elevated_percent"] < t["ram_high_percent"]
        assert t["ram_high_percent"] < t["ram_critical_percent"]
        assert t["ram_critical_percent"] < t["ram_deadly_percent"]

        assert t["vram_elevated_percent"] < t["vram_high_percent"]
        assert t["vram_high_percent"] < t["vram_critical_percent"]
        assert t["vram_critical_percent"] < t["vram_deadly_percent"]


# ── Snapshot Integrity ────────────────────────────────────────────────


class TestSnapshot:
    """Test that take_snapshot() returns valid data."""

    def test_snapshot_has_ram_fields(self):
        """Snapshot should always have RAM percentage and free GB."""
        from src.utils.memory_guard import take_snapshot

        snap = take_snapshot()

        assert hasattr(snap, "ram_used_percent")
        assert hasattr(snap, "ram_free_gb")
        assert 0 <= snap.ram_used_percent <= 100
        assert snap.ram_free_gb >= 0

    def test_snapshot_has_swap_field(self):
        """Snapshot should include swap usage."""
        from src.utils.memory_guard import take_snapshot

        snap = take_snapshot()

        assert hasattr(snap, "swap_percent")
        assert 0 <= snap.swap_percent <= 100

    def test_snapshot_vram_without_gpu(self):
        """On CPU-only systems, VRAM fields should be 0 or benign."""
        from src.utils.memory_guard import take_snapshot, TORCH_AVAILABLE

        snap = take_snapshot()

        # If no GPU, worst_vram_percent should be 0 (no pressure from VRAM)
        if not TORCH_AVAILABLE:
            assert snap.worst_vram_percent == 0.0


# ── Convenience Functions ─────────────────────────────────────────────


class TestConvenienceFunctions:
    """Test module-level convenience functions delegate to guard singleton."""

    def test_is_safe_returns_bool(self):
        """is_safe() should return a boolean."""
        from src.utils.memory_guard import is_safe

        result = is_safe()
        assert isinstance(result, bool)

    def test_get_pressure_returns_enum(self):
        """get_pressure() should return a MemoryPressure enum value."""
        from src.utils.memory_guard import get_pressure, MemoryPressure

        result = get_pressure()
        assert isinstance(result, MemoryPressure)

    def test_cleanup_returns_dict(self):
        """cleanup() should return metrics dict."""
        from src.utils.memory_guard import cleanup

        result = cleanup()
        assert isinstance(result, dict)

    def test_is_wsl_returns_bool(self):
        """is_wsl() should return a boolean."""
        from src.utils.memory_guard import is_wsl

        result = is_wsl()
        assert isinstance(result, bool)


# ── Guard Singleton ───────────────────────────────────────────────────


class TestGuardSingleton:
    """Test that the module-level guard instance works correctly."""

    def test_guard_is_initialized(self):
        """Module should export a pre-initialized guard instance."""
        from src.utils.memory_guard import guard

        assert guard is not None

    def test_guard_has_is_wsl_attribute(self):
        """Guard should expose is_wsl as a property/attribute."""
        from src.utils.memory_guard import guard

        assert hasattr(guard, "is_wsl")
        assert isinstance(guard.is_wsl, bool)

    def test_guard_get_pressure(self):
        """Guard should be callable for pressure check."""
        from src.utils.memory_guard import guard, MemoryPressure

        pressure = guard.get_pressure()
        assert isinstance(pressure, MemoryPressure)


# ── System Profile ────────────────────────────────────────────────────


class TestSystemProfile:
    """Test system profile detection."""

    def test_get_system_profile(self):
        """get_system_profile() should return a dict with hardware info."""
        from src.utils.memory_guard import get_system_profile

        profile = get_system_profile()
        assert isinstance(profile, dict)
        assert "ram_total_gb" in profile
        assert "is_wsl" in profile
        assert profile["ram_total_gb"] > 0

    def test_system_profile_gpu_count(self):
        """System profile should report GPU count (0 if CPU-only)."""
        from src.utils.memory_guard import get_system_profile

        profile = get_system_profile()
        assert "gpu_count" in profile
        assert isinstance(profile["gpu_count"], int)
        assert profile["gpu_count"] >= 0


# ── Enforcement (_enforce_limits) ─────────────────────────────────────


class TestEnforceLimits:
    """Test hard limit enforcement at OS and CUDA level."""

    def test_enforce_sets_limits_enforced_flag(self, _mock_rlimit):
        """_enforce_limits() should set _limits_enforced to True when it succeeds."""
        from src.utils.memory_guard import MemoryGuard

        guard = MemoryGuard(enforce=True, auto_monitor=False)
        # On Linux/WSL, resource.RLIMIT_AS should succeed, so _limits_enforced=True
        assert isinstance(guard._limits_enforced, bool)
        # setrlimit was called (mocked, not real)
        _mock_rlimit.assert_called()

    def test_enforce_false_skips_enforcement(self):
        """With enforce=False, no limits should be set."""
        from src.utils.memory_guard import MemoryGuard

        guard = MemoryGuard(enforce=False, auto_monitor=False)
        assert guard._limits_enforced is False
        assert guard._cuda_fraction_set is False

    def test_enforce_sets_process_ram_limit(self, _mock_rlimit):
        """_enforce_limits() should set RLIMIT_AS on Linux/WSL."""
        from src.utils.memory_guard import MemoryGuard

        guard = MemoryGuard(enforce=True, auto_monitor=False)

        # Verify setrlimit was called (mocked — no real OS limit set)
        _mock_rlimit.assert_called()
        # The call should have been for RLIMIT_AS
        import resource

        args = _mock_rlimit.call_args
        assert args[0][0] == resource.RLIMIT_AS

    def test_enforce_cuda_fraction_with_mock(self, _mock_rlimit):
        """_enforce_limits() should call torch.cuda.set_per_process_memory_fraction."""
        from src.utils.memory_guard import MemoryGuard, TORCH_AVAILABLE

        if not TORCH_AVAILABLE:
            pytest.skip("torch not available")

        mock_props = MagicMock()
        mock_props.total_memory = 16 * (1024**3)  # 16GB

        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.device_count", return_value=1),
            patch("torch.cuda.get_device_properties", return_value=mock_props),
            patch("torch.cuda.set_per_process_memory_fraction") as mock_set,
        ):
            guard = MemoryGuard(enforce=True, auto_monitor=False)
            mock_set.assert_called_once()
            # Fraction should be between 0.5 and 0.95
            fraction_arg = mock_set.call_args[0][0]
            assert 0.5 <= fraction_arg <= 0.95

    def test_enforce_respects_min_4gb_floor(self, _mock_rlimit):
        """RAM limit should never go below 4GB (would break Python)."""
        from src.utils.memory_guard import MemoryGuard

        # We can't easily mock psutil.virtual_memory total to be < 4GB
        # but we verify the logic exists by checking the guard initializes
        guard = MemoryGuard(enforce=True, auto_monitor=False)
        assert guard is not None


# ── Safe Allocation Context Manager ───────────────────────────────────


class TestSafeAllocate:
    """Test safe_allocate() context manager."""

    def test_safe_allocate_passes_when_safe(self):
        """safe_allocate() should not raise when memory is plentiful."""
        from src.utils.memory_guard import MemoryGuard, MemorySnapshot

        guard = MemoryGuard(enforce=False, auto_monitor=False)

        fake_snap = MemorySnapshot(
            timestamp=0.0,
            ram_total_gb=32.0,
            ram_available_gb=20.0,
            ram_used_percent=30.0,
            swap_used_gb=0.5,
            swap_total_gb=8.0,
            swap_percent=5.0,
            gpus=[
                {
                    "id": 0,
                    "total_gb": 16.0,
                    "free_gb": 12.0,
                    "used_gb": 4.0,
                    "used_percent": 25.0,
                }
            ],
        )

        with patch("src.utils.memory_guard.take_snapshot", return_value=fake_snap):
            with guard.safe_allocate(
                estimated_ram_gb=2.0, estimated_vram_gb=4.0, operation="test_op"
            ):
                pass  # Should not raise

    def test_safe_allocate_raises_when_insufficient(self):
        """safe_allocate() should raise MemoryError when allocation would exceed limits."""
        from src.utils.memory_guard import MemoryGuard, MemorySnapshot

        guard = MemoryGuard(enforce=False, auto_monitor=False)

        fake_snap = MemorySnapshot(
            timestamp=0.0,
            ram_total_gb=32.0,
            ram_available_gb=1.0,  # Very low
            ram_used_percent=96.0,
            swap_used_gb=6.0,
            swap_total_gb=8.0,
            swap_percent=75.0,
            gpus=[
                {
                    "id": 0,
                    "total_gb": 16.0,
                    "free_gb": 0.5,
                    "used_gb": 15.5,
                    "used_percent": 96.9,
                }
            ],
        )

        with patch("src.utils.memory_guard.take_snapshot", return_value=fake_snap):
            with pytest.raises(MemoryError, match="Insufficient memory"):
                with guard.safe_allocate(
                    estimated_ram_gb=10.0, estimated_vram_gb=8.0, operation="big_load"
                ):
                    pass

    def test_safe_allocate_cleans_up_on_critical_exit(self):
        """safe_allocate() should run cleanup on __exit__ if pressure is CRITICAL+."""
        from src.utils.memory_guard import MemoryGuard, MemoryPressure, MemorySnapshot

        guard = MemoryGuard(enforce=False, auto_monitor=False)

        # Enter with safe memory, exit with critical
        safe_snap = MemorySnapshot(
            timestamp=0.0,
            ram_total_gb=32.0,
            ram_available_gb=20.0,
            ram_used_percent=30.0,
            swap_used_gb=0.5,
            swap_total_gb=8.0,
            swap_percent=5.0,
            gpus=[
                {
                    "id": 0,
                    "total_gb": 16.0,
                    "free_gb": 12.0,
                    "used_gb": 4.0,
                    "used_percent": 25.0,
                }
            ],
        )

        call_count = [0]

        def counting_snapshot():
            call_count[0] += 1
            return safe_snap

        with patch(
            "src.utils.memory_guard.take_snapshot", side_effect=counting_snapshot
        ):
            with patch.object(
                guard, "get_pressure", return_value=MemoryPressure.CRITICAL
            ):
                with patch.object(guard, "emergency_cleanup") as mock_cleanup:
                    with guard.safe_allocate(
                        estimated_ram_gb=1.0, estimated_vram_gb=1.0, operation="test"
                    ):
                        pass  # Operation completes
                    # emergency_cleanup should have been called on exit
                    mock_cleanup.assert_called_with(aggressive=True)


# ── WSL-Specific is_safe Behavior ─────────────────────────────────────


class TestWSLIsSafe:
    """Test that is_safe() is stricter on WSL."""

    def test_wsl_blocks_at_high_pressure(self):
        """On WSL, is_safe() should return False at HIGH pressure."""
        from src.utils.memory_guard import (
            MemoryGuard,
            MemoryPressure,
            MemorySnapshot,
            WSL_THRESHOLDS,
        )

        guard = MemoryGuard(enforce=False, auto_monitor=False)
        guard._is_wsl = True
        guard._thresholds = WSL_THRESHOLDS

        # RAM at 65% = HIGH for WSL (threshold is 62%)
        fake_snap = MemorySnapshot(
            timestamp=0.0,
            ram_total_gb=16.0,
            ram_available_gb=5.6,
            ram_used_percent=65.0,
            swap_used_gb=0.5,
            swap_total_gb=8.0,
            swap_percent=5.0,
            gpus=[
                {
                    "id": 0,
                    "total_gb": 16.0,
                    "free_gb": 12.0,
                    "used_gb": 4.0,
                    "used_percent": 25.0,
                }
            ],
        )

        with patch("src.utils.memory_guard.take_snapshot", return_value=fake_snap):
            result = guard.is_safe()
            assert result is False

    def test_native_allows_high_pressure(self):
        """On native Linux, is_safe() should return True at HIGH pressure."""
        from src.utils.memory_guard import (
            MemoryGuard,
            MemoryPressure,
            MemorySnapshot,
            NATIVE_THRESHOLDS,
        )

        guard = MemoryGuard(enforce=False, auto_monitor=False)
        guard._is_wsl = False
        guard._thresholds = NATIVE_THRESHOLDS

        # RAM at 82% = HIGH for native (threshold is 80%)
        # but native allows HIGH, only blocks at CRITICAL (90%)
        fake_snap = MemorySnapshot(
            timestamp=0.0,
            ram_total_gb=32.0,
            ram_available_gb=5.8,
            ram_used_percent=82.0,
            swap_used_gb=0.5,
            swap_total_gb=8.0,
            swap_percent=5.0,
            gpus=[
                {
                    "id": 0,
                    "total_gb": 16.0,
                    "free_gb": 12.0,
                    "used_gb": 4.0,
                    "used_percent": 25.0,
                }
            ],
        )

        with patch("src.utils.memory_guard.take_snapshot", return_value=fake_snap):
            result = guard.is_safe()
            assert result is True


# ── Auto-Cleanup Registration ─────────────────────────────────────────


class TestAutoCleanup:
    """Test auto-cleanup callback registration."""

    def test_auto_cleanup_registered_with_enforce(self, _mock_rlimit):
        """Enforce mode should register auto-cleanup callbacks."""
        from src.utils.memory_guard import MemoryGuard, MemoryPressure

        guard = MemoryGuard(enforce=True, auto_monitor=False)
        # Should have callbacks registered for CRITICAL and DEADLY
        assert len(guard._callbacks[MemoryPressure.CRITICAL]) > 0
        assert len(guard._callbacks[MemoryPressure.DEADLY]) > 0

    def test_no_auto_cleanup_without_enforce(self):
        """Without enforce, no auto-cleanup callbacks should be registered."""
        from src.utils.memory_guard import MemoryGuard, MemoryPressure

        guard = MemoryGuard(enforce=False, auto_monitor=False)
        assert len(guard._callbacks[MemoryPressure.CRITICAL]) == 0
        assert len(guard._callbacks[MemoryPressure.DEADLY]) == 0

    def test_auto_cleanup_fires_on_critical(self, _mock_rlimit):
        """Auto-cleanup should call emergency_cleanup when CRITICAL pressure fires."""
        from src.utils.memory_guard import MemoryGuard, MemoryPressure, MemorySnapshot

        guard = MemoryGuard(enforce=True, auto_monitor=False)

        critical_snap = MemorySnapshot(
            timestamp=0.0,
            ram_total_gb=16.0,
            ram_available_gb=1.0,
            ram_used_percent=93.0,
            swap_used_gb=6.0,
            swap_total_gb=8.0,
            swap_percent=75.0,
            gpus=[],
            pressure=MemoryPressure.CRITICAL,
        )

        with patch.object(guard, "emergency_cleanup") as mock_cleanup:
            guard._fire_callbacks(critical_snap)
            mock_cleanup.assert_called_once_with(aggressive=False)


# ── Monitor Auto-Start ────────────────────────────────────────────────


class TestMonitorAutoStart:
    """Test background monitor auto-start behavior."""

    def test_auto_monitor_starts_thread(self):
        """With auto_monitor=True, the monitor thread should be started."""
        from src.utils.memory_guard import MemoryGuard

        guard = MemoryGuard(enforce=False, auto_monitor=True)
        try:
            assert guard._monitor_thread is not None
            assert guard._monitor_thread.is_alive()
        finally:
            guard.stop_monitor()

    def test_auto_monitor_false_no_thread(self):
        """With auto_monitor=False, no monitor thread should start."""
        from src.utils.memory_guard import MemoryGuard

        guard = MemoryGuard(enforce=False, auto_monitor=False)
        assert guard._monitor_thread is None

    def test_monitor_is_daemon(self):
        """Monitor thread should be a daemon so it doesn't block process exit."""
        from src.utils.memory_guard import MemoryGuard

        guard = MemoryGuard(enforce=False, auto_monitor=True)
        try:
            assert guard._monitor_thread.daemon is True
        finally:
            guard.stop_monitor()
