"""Lightweight wrapper interface for CUDA-ZSTD compression.

This module provides a small abstraction so Nexus code can interact
with a CUDA ZSTD implementation. During unit tests we may not have a
real GPU-backed library available; the wrapper supports a fallback
pure-python zstandard-based implementation if `cuda_zstd` is not
installed. The wrapper API mirrors the usage in the Perplexity doc.
"""

from __future__ import annotations

from typing import Optional

try:
    # Try to import a hypothetical cuda_zstd package (the user's custom lib)
    import cuda_zstd as _cuda_zstd  # type: ignore
except Exception:
    _cuda_zstd = None

try:
    import zstandard as zstd  # type: ignore
except Exception:
    zstd = None


class ZstdStreamingManager:
    """Adapter exposing compress/decompress methods.

    If a GPU-backed library is present it is used; otherwise fall back
    to python zstandard (if available) which is sufficient for tests.
    """

    def __init__(self, level: int = 5):
        self.level = int(level)

        if _cuda_zstd is not None:
            self.impl = _cuda_zstd.create_streaming_manager(level=self.level)  # type: ignore
        elif zstd is not None:
            self._cctx = (
                zstd.ZstdCompressor(level=self.level) if zstd is not None else None
            )
            self._dctx = zstd.ZstdDecompressor() if zstd is not None else None
            self.impl = None
        else:
            # Minimal no-op fallback that stores raw bytes; keeps tests working
            self.impl = None

    def compress(self, data: bytes) -> bytes:
        if hasattr(self.impl, "compress"):
            return self.impl.compress(data)
        if zstd is not None and self._cctx is not None:
            return self._cctx.compress(data)
        # fallback: return data unchanged
        return data

    def decompress(self, data: bytes) -> bytes:
        if hasattr(self.impl, "decompress"):
            return self.impl.decompress(data)
        if zstd is not None and self._dctx is not None:
            return self._dctx.decompress(data)
        # fallback: return data unchanged
        return data


def create_streaming_manager(level: int = 5) -> ZstdStreamingManager:
    return ZstdStreamingManager(level=level)
