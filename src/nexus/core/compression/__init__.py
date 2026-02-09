"""
Compression module.

Provides CUDA-accelerated ZSTD compression with smart routing,
COVER dictionary training, and CPU fallback chain.
"""

from nexus.core.compression.cuda_zstd_wrapper import ZstdStreamingManager

__all__ = ["ZstdStreamingManager"]
