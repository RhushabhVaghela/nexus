"""
GPU-Accelerated Compression Utility
====================================

Unified interface for GPU (cuda_zstd) and CPU (zstandard) ZSTD compression
with automatic fallback. Wraps the Custom-NVComp-with-ZSTD library for
GPU-accelerated compression/decompression, falling back transparently to
CPU zstandard when CUDA is unavailable.

Integration points in Nexus:
  - nexus.optimizations.compression_optimized  (ZSTDQuantizedCompressor)
  - nexus.optimizations.async_decompression    (AsyncDecompressor)

Usage::

    from nexus.utils.gpu_compression import get_compressor, GPUCompressionConfig

    cfg = GPUCompressionConfig(compression_level=3, prefer_gpu=True)
    compressor = get_compressor(cfg)

    compressed = compressor.compress(data)
    original   = compressor.decompress(compressed)

    # Batch API (GPU only — falls back to sequential on CPU)
    results = compressor.compress_batch([buf1, buf2, buf3])
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Union

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Availability detection
# ---------------------------------------------------------------------------

_GPU_AVAILABLE: bool = False
_CPU_AVAILABLE: bool = False
_gpu_import_error: Optional[str] = None
_cpu_import_error: Optional[str] = None

try:
    import cuda_zstd

    _GPU_AVAILABLE = cuda_zstd.is_cuda_available()
    if not _GPU_AVAILABLE:
        _gpu_import_error = "cuda_zstd imported but no CUDA device detected"
except ImportError as exc:
    _gpu_import_error = str(exc)
except Exception as exc:
    _gpu_import_error = f"cuda_zstd import failed: {exc}"

try:
    import zstandard as _zstd_cpu  # noqa: F401

    _CPU_AVAILABLE = True
except ImportError as exc:
    _cpu_import_error = str(exc)


def is_gpu_compression_available() -> bool:
    """Return True if GPU compression via cuda_zstd is usable."""
    return _GPU_AVAILABLE


def is_cpu_compression_available() -> bool:
    """Return True if CPU compression via zstandard is usable."""
    return _CPU_AVAILABLE


def get_backend_info() -> Dict[str, Any]:
    """Return diagnostic information about available backends."""
    info: Dict[str, Any] = {
        "gpu_available": _GPU_AVAILABLE,
        "cpu_available": _CPU_AVAILABLE,
        "active_backend": "none",
    }
    if _GPU_AVAILABLE:
        info["active_backend"] = "gpu"
        try:
            info["gpu_device"] = cuda_zstd.get_cuda_device_info()
        except Exception:
            info["gpu_device"] = "unknown"
    elif _CPU_AVAILABLE:
        info["active_backend"] = "cpu"

    if _gpu_import_error:
        info["gpu_import_error"] = _gpu_import_error
    if _cpu_import_error:
        info["cpu_import_error"] = _cpu_import_error
    return info


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class GPUCompressionConfig:
    """Configuration for the unified compression interface.

    Parameters
    ----------
    compression_level : int
        ZSTD level (1-22). Higher = better ratio, slower.
    prefer_gpu : bool
        If True, use GPU when available; if False, always use CPU.
    use_hybrid_engine : bool
        If True, use cuda_zstd's HybridEngine which auto-routes between
        CPU (libzstd) and GPU based on data size. Recommended for mixed
        workloads where some buffers are small.
    batch_enabled : bool
        If True, batch compress/decompress uses a single GPU launch.
    """

    compression_level: int = 3
    prefer_gpu: bool = True
    use_hybrid_engine: bool = True
    batch_enabled: bool = True


# ---------------------------------------------------------------------------
# Compressor interface
# ---------------------------------------------------------------------------

BufferLike = Union[bytes, bytearray, memoryview]


class _GPUCompressor:
    """GPU-backed compressor using cuda_zstd."""

    def __init__(self, config: GPUCompressionConfig) -> None:
        self._config = config
        self._manager: Optional[Any] = None
        self._hybrid: Optional[Any] = None
        self._use_hybrid = config.use_hybrid_engine

    # -- lazy init so we don't allocate GPU memory until first use ----------

    def _get_manager(self) -> Any:
        if self._manager is None:
            self._manager = cuda_zstd.Manager(level=self._config.compression_level)
        return self._manager

    def _get_hybrid(self) -> Any:
        if self._hybrid is None:
            self._hybrid = cuda_zstd.HybridEngine(level=self._config.compression_level)
        return self._hybrid

    # -- public API ---------------------------------------------------------

    def compress(self, data: BufferLike) -> bytes:
        if self._use_hybrid:
            return self._get_hybrid().compress(data)
        return self._get_manager().compress(data)

    def decompress(self, data: BufferLike) -> bytes:
        if self._use_hybrid:
            return self._get_hybrid().decompress(data)
        return self._get_manager().decompress(data)

    def compress_batch(self, inputs: Sequence[BufferLike]) -> List[bytes]:
        return self._get_manager().compress_batch(list(inputs))

    def decompress_batch(self, inputs: Sequence[BufferLike]) -> List[bytes]:
        return self._get_manager().decompress_batch(list(inputs))

    @property
    def backend(self) -> str:
        return "gpu"

    def close(self) -> None:
        if self._manager is not None:
            self._manager.close()
            self._manager = None
        if self._hybrid is not None:
            self._hybrid.close()
            self._hybrid = None


class _CPUCompressor:
    """CPU-backed compressor using zstandard."""

    def __init__(self, config: GPUCompressionConfig) -> None:
        import zstandard

        self._level = config.compression_level
        self._cctx = zstandard.ZstdCompressor(level=self._level)
        self._dctx = zstandard.ZstdDecompressor()

    def compress(self, data: BufferLike) -> bytes:
        return self._cctx.compress(bytes(data))

    def decompress(self, data: BufferLike) -> bytes:
        return self._dctx.decompress(bytes(data))

    def compress_batch(self, inputs: Sequence[BufferLike]) -> List[bytes]:
        return [self.compress(buf) for buf in inputs]

    def decompress_batch(self, inputs: Sequence[BufferLike]) -> List[bytes]:
        return [self.decompress(buf) for buf in inputs]

    @property
    def backend(self) -> str:
        return "cpu"

    def close(self) -> None:
        pass  # nothing to release


class _NullCompressor:
    """Stub when neither backend is available — raises on use."""

    def compress(self, data: BufferLike) -> bytes:
        raise RuntimeError(
            "No ZSTD compression backend available. "
            "Install 'zstandard' (pip install zstandard) for CPU support, "
            "or build 'cuda_zstd' for GPU support."
        )

    def decompress(self, data: BufferLike) -> bytes:
        raise RuntimeError("No ZSTD compression backend available.")

    def compress_batch(self, inputs: Sequence[BufferLike]) -> List[bytes]:
        raise RuntimeError("No ZSTD compression backend available.")

    def decompress_batch(self, inputs: Sequence[BufferLike]) -> List[bytes]:
        raise RuntimeError("No ZSTD compression backend available.")

    @property
    def backend(self) -> str:
        return "none"

    def close(self) -> None:
        pass


# Union type for type hints
Compressor = Union[_GPUCompressor, _CPUCompressor, _NullCompressor]


def get_compressor(config: Optional[GPUCompressionConfig] = None) -> Compressor:
    """Create a compressor with automatic backend selection.

    Priority: GPU (cuda_zstd) > CPU (zstandard) > error stub.

    Parameters
    ----------
    config : GPUCompressionConfig, optional
        If None, uses defaults (level=3, prefer_gpu=True).

    Returns
    -------
    Compressor
        An object with compress/decompress/compress_batch/decompress_batch
        methods and a ``backend`` property indicating which engine is active.
    """
    if config is None:
        config = GPUCompressionConfig()

    if config.prefer_gpu and _GPU_AVAILABLE:
        logger.info(
            "GPU compression enabled (cuda_zstd, level=%d, hybrid=%s)",
            config.compression_level,
            config.use_hybrid_engine,
        )
        return _GPUCompressor(config)

    if _CPU_AVAILABLE:
        if config.prefer_gpu and not _GPU_AVAILABLE:
            logger.info(
                "GPU compression unavailable (%s), falling back to CPU zstandard",
                _gpu_import_error or "unknown reason",
            )
        else:
            logger.info(
                "CPU compression selected (zstandard, level=%d)",
                config.compression_level,
            )
        return _CPUCompressor(config)

    logger.warning(
        "No ZSTD backend available (GPU: %s, CPU: %s)",
        _gpu_import_error or "not installed",
        _cpu_import_error or "not installed",
    )
    return _NullCompressor()
