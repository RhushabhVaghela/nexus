"""
Comprehensive unit tests for async_decompression.py
Tests CUDA stream management, parallel decompression, memory management, and error handling.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import torch
import threading
import queue
from typing import Tuple

from src.optimizations.async_decompression import (
    AsyncDecompressionConfig,
    CUDAStreamManager,
    LayerBufferPool,
    AsyncDecompressor,
    StreamingLayerLoader,
)


class TestAsyncDecompressionConfig:
    """Test AsyncDecompressionConfig dataclass."""

    def test_default_config_values(self):
        """Test default configuration values."""
        config = AsyncDecompressionConfig()
        assert config.num_worker_threads == 4
        assert config.prefetch_depth == 3
        assert config.use_cuda_streams is True
        assert config.compression_format == "zstd"
        assert config.buffer_pool_size == 10

    def test_custom_config_values(self):
        """Test custom configuration values."""
        config = AsyncDecompressionConfig(
            num_worker_threads=8,
            prefetch_depth=5,
            use_cuda_streams=False,
            compression_format="lz4",
            buffer_pool_size=20,
        )
        assert config.num_worker_threads == 8
        assert config.prefetch_depth == 5
        assert config.use_cuda_streams is False
        assert config.compression_format == "lz4"
        assert config.buffer_pool_size == 20

    def test_config_immutable_after_creation(self):
        """Test that config is a dataclass and can be modified."""
        config = AsyncDecompressionConfig()
        config.num_worker_threads = 16
        assert config.num_worker_threads == 16


class TestCUDAStreamManager:
    """Test CUDAStreamManager for stream management."""

    @patch("src.optimizations.async_decompression.torch.cuda")
    def test_init_with_cuda_available(self, mock_cuda):
        """Test initialization when CUDA is available."""
        mock_cuda.is_available.return_value = True
        mock_cuda.default_stream.return_value = Mock()

        manager = CUDAStreamManager(num_streams=3)
        assert len(manager.streams) == 3
        assert manager.num_streams == 3
        assert manager.default_stream is not None

    @patch("src.optimizations.async_decompression.torch.cuda")
    def test_init_without_cuda(self, mock_cuda):
        """Test initialization when CUDA is not available."""
        mock_cuda.is_available.return_value = False

        manager = CUDAStreamManager(num_streams=3)
        assert len(manager.streams) == 0
        assert manager.default_stream is None

    @patch("src.optimizations.async_decompression.torch.cuda")
    def test_get_next_stream_round_robin(self, mock_cuda):
        """Test round-robin stream selection."""
        mock_cuda.is_available.return_value = True
        mock_cuda.default_stream.return_value = Mock()

        manager = CUDAStreamManager(num_streams=3)
        stream1 = manager.get_next_stream()
        stream2 = manager.get_next_stream()
        stream3 = manager.get_next_stream()
        stream4 = manager.get_next_stream()  # Should wrap around

        assert stream1 is not None
        assert stream2 is not None
        assert stream3 is not None
        assert stream4 == stream1

    @patch("src.optimizations.async_decompression.torch.cuda")
    def test_get_next_stream_no_streams(self, mock_cuda):
        """Test get_next_stream when no streams available."""
        mock_cuda.is_available.return_value = False

        manager = CUDAStreamManager(num_streams=3)
        result = manager.get_next_stream()
        assert result is None

    @patch("src.optimizations.async_decompression.torch.cuda")
    def test_get_compute_stream(self, mock_cuda):
        """Test compute stream retrieval."""
        mock_cuda.is_available.return_value = True
        mock_cuda.default_stream.return_value = Mock()

        manager = CUDAStreamManager(num_streams=3)
        compute_stream = manager.get_compute_stream()
        assert compute_stream == manager.streams[0]

    @patch("src.optimizations.async_decompression.torch.cuda")
    def test_get_decompress_stream(self, mock_cuda):
        """Test decompression stream retrieval."""
        mock_cuda.is_available.return_value = True
        mock_cuda.default_stream.return_value = Mock()

        manager = CUDAStreamManager(num_streams=3)
        decompress_stream = manager.get_decompress_stream()
        assert decompress_stream == manager.streams[1]

    @patch("src.optimizations.async_decompression.torch.cuda")
    def test_get_decompress_stream_single_stream(self, mock_cuda):
        """Test decompression stream with only one stream."""
        mock_cuda.is_available.return_value = True
        mock_cuda.default_stream.return_value = Mock()

        manager = CUDAStreamManager(num_streams=1)
        decompress_stream = manager.get_decompress_stream()
        assert decompress_stream == manager.streams[0]

    @patch("src.optimizations.async_decompression.torch.cuda")
    def test_synchronize_all(self, mock_cuda):
        """Test synchronizing all streams."""
        mock_cuda.is_available.return_value = True
        mock_cuda.default_stream.return_value = Mock()

        manager = CUDAStreamManager(num_streams=2)
        manager.synchronize_all()

        for stream in manager.streams:
            stream.synchronize.assert_called_once()
        manager.default_stream.synchronize.assert_called_once()

    @patch("src.optimizations.async_decompression.torch.cuda")
    def test_synchronize_all_no_cuda(self, mock_cuda):
        """Test synchronize when no CUDA available."""
        mock_cuda.is_available.return_value = False

        manager = CUDAStreamManager(num_streams=3)
        # Should not raise any errors
        manager.synchronize_all()

    @patch("src.optimizations.async_decompression.torch.cuda")
    def test_wait_for_decompression(self, mock_cuda):
        """Test waiting for decompression stream."""
        mock_cuda.is_available.return_value = True
        mock_cuda.default_stream.return_value = Mock()

        manager = CUDAStreamManager(num_streams=3)
        compute_stream = Mock()

        manager.wait_for_decompression(compute_stream)
        compute_stream.wait_stream.assert_called_once()


class TestLayerBufferPool:
    """Test LayerBufferPool for memory management."""

    def test_init_default_max_size(self):
        """Test initialization with default max size."""
        pool = LayerBufferPool()
        assert pool.max_size == 10

    def test_init_custom_max_size(self):
        """Test initialization with custom max size."""
        pool = LayerBufferPool(max_size=5)
        assert pool.max_size == 5

    def test_get_buffer_new_layer(self):
        """Test getting buffer for new layer."""
        pool = LayerBufferPool(max_size=2)
        shape = (128, 768)
        dtype = torch.float16
        device = "cuda"

        buffer = pool.get_buffer("layer1", shape, dtype, device)

        assert buffer.shape == shape
        assert buffer.dtype == dtype

    def test_get_buffer_existing_layer_shape_match(self):
        """Test buffer reuse for same layer with matching shape."""
        pool = LayerBufferPool(max_size=2)
        shape = (128, 768)
        dtype = torch.float16
        device = "cuda"

        buffer1 = pool.get_buffer("layer1", shape, dtype, device)
        pool.return_buffer("layer1", buffer1)
        buffer2 = pool.get_buffer("layer1", shape, dtype, device)

        assert buffer2 is buffer1

    def test_get_buffer_shape_mismatch(self):
        """Test buffer replacement when shape changes."""
        pool = LayerBufferPool(max_size=2)
        dtype = torch.float16
        device = "cuda"

        buffer1 = pool.get_buffer("layer1", (128, 768), dtype, device)
        buffer2 = pool.get_buffer("layer1", (256, 768), dtype, device)

        assert buffer2.shape == (256, 768)

    def test_get_buffer_dtype_mismatch(self):
        """Test buffer replacement when dtype changes."""
        pool = LayerBufferPool(max_size=2)
        shape = (128, 768)
        device = "cuda"

        buffer1 = pool.get_buffer("layer1", shape, torch.float16, device)
        buffer2 = pool.get_buffer("layer1", shape, torch.float32, device)

        assert buffer2.dtype == torch.float32

    def test_return_buffer(self):
        """Test returning buffer to pool."""
        pool = LayerBufferPool(max_size=2)
        shape = (128, 768)
        dtype = torch.float16
        device = "cuda"

        buffer = pool.get_buffer("layer1", shape, dtype, device)
        pool.return_buffer("layer1", buffer)

        # Should be able to get it back
        buffer2 = pool.get_buffer("layer1", shape, dtype, device)
        assert buffer2 is buffer

    def test_return_buffer_full_pool(self):
        """Test dropping buffer when pool is full."""
        pool = LayerBufferPool(max_size=1)
        shape = (128, 768)
        dtype = torch.float16
        device = "cuda"

        buffer1 = pool.get_buffer("layer1", shape, dtype, device)
        pool.return_buffer("layer1", buffer1)

        buffer2 = pool.get_buffer("layer1", shape, dtype, device)
        # Buffer2 should be a new buffer, not the old one
        assert buffer2 is not buffer1

    def test_clear(self):
        """Test clearing all buffers."""
        pool = LayerBufferPool(max_size=10)
        shape = (128, 768)
        dtype = torch.float16
        device = "cuda"

        pool.get_buffer("layer1", shape, dtype, device)
        pool.get_buffer("layer2", shape, dtype, device)
        pool.clear()

        # Should be able to get new buffers
        buffer = pool.get_buffer("layer1", shape, dtype, device)
        assert buffer.shape == shape


class TestAsyncDecompressor:
    """Test AsyncDecompressor for async decompression."""

    @patch("src.optimizations.async_decompression.ThreadPoolExecutor")
    def test_init_default_config(self, mock_executor):
        """Test initialization with default config."""
        decompressor = AsyncDecompressor()

        assert decompressor.config.num_worker_threads == 4
        assert decompressor.config.prefetch_depth == 3

    @patch("src.optimizations.async_decompression.torch.cuda")
    def test_decompress_layer_async_cache_hit(self, mock_cuda):
        """Test cache hit during async decompression."""
        mock_cuda.is_available.return_value = False

        config = AsyncDecompressionConfig(use_cuda_streams=False)
        decompressor = AsyncDecompressor(config)

        # Pre-populate cache
        cached_tensor = torch.randn(128, 768, dtype=torch.float16)
        decompressor.decompressed_cache["layer1"] = cached_tensor

        result = decompressor.decompress_layer_async(
            "layer1", b"compressed", (128, 768), torch.float16, "cpu"
        )

        assert result is cached_tensor
        assert decompressor.stats["cache_hits"] == 1

    @patch("src.optimizations.async_decompression.torch.cuda")
    @patch("src.optimizations.async_decompression.importlib.import_module")
    def test_decompress_layer_async_zstd_format(self, mock_import, mock_cuda):
        """Test ZSTD decompression format."""
        mock_cuda.is_available.return_value = False

        # Mock zstandard module
        mock_zstd = Mock()
        mock_decompressor = Mock()
        mock_zstd.ZstdDecompressor.return_value = mock_decompressor
        mock_decompressor.decompress.return_value = b"decompressed_data"
        mock_import.return_value = mock_zstd

        config = AsyncDecompressionConfig(
            compression_format="zstd", use_cuda_streams=False
        )
        decompressor = AsyncDecompressor(config)

        compressed_data = b"test_compressed"
        result = decompressor.decompress_layer_async(
            "layer1", compressed_data, (10,), torch.float32, "cpu"
        )

        mock_decompressor.decompress.assert_called_once_with(compressed_data)

    @patch("src.optimizations.async_decompression.torch.cuda")
    @patch("src.optimizations.async_decompression.importlib.import_module")
    def test_decompress_layer_async_lz4_format(self, mock_import, mock_cuda):
        """Test LZ4 decompression format."""
        mock_cuda.is_available.return_value = False

        # Mock lz4.frame module
        mock_lz4 = Mock()
        mock_lz4.frame.decompress.return_value = b"decompressed_data"
        mock_import.return_value = mock_lz4

        config = AsyncDecompressionConfig(
            compression_format="lz4", use_cuda_streams=False
        )
        decompressor = AsyncDecompressor(config)

        compressed_data = b"test_compressed"
        result = decompressor.decompress_layer_async(
            "layer1", compressed_data, (10,), torch.float32, "cpu"
        )

        mock_lz4.frame.decompress.assert_called_once_with(compressed_data)

    @patch("src.optimizations.async_decompression.torch.cuda")
    def test_decompress_layer_async_unknown_format(self, mock_cuda):
        """Test fallback for unknown compression format."""
        mock_cuda.is_available.return_value = False

        config = AsyncDecompressionConfig(
            compression_format="unknown", use_cuda_streams=False
        )
        decompressor = AsyncDecompressor(config)

        result = decompressor.decompress_layer_async(
            "layer1", b"data", (10,), torch.float32, "cpu"
        )

        assert result is not None

    @patch("src.optimizations.async_decompression.ThreadPoolExecutor")
    def test_prefetch_layer(self, mock_executor):
        """Test layer prefetching."""
        mock_cuda = Mock()
        mock_cuda.is_available.return_value = False

        with patch("src.optimizations.async_decompression.torch.cuda", mock_cuda):
            config = AsyncDecompressionConfig(use_cuda_streams=False)
            decompressor = AsyncDecompressor(config)

            # Mock the executor
            mock_future = Mock()
            mock_executor.return_value.submit.return_value = mock_future

            decompressor.prefetch_layer(
                "layer1", b"compressed", (128, 768), torch.float16, "cpu"
            )

            mock_executor.return_value.submit.assert_called_once()
            assert decompressor.stats["async_decompressions"] == 1

    @patch("src.optimizations.async_decompression.ThreadPoolExecutor")
    @patch("src.optimizations.async_decompression.time.time")
    def test_get_prefetched_layer_success(self, mock_time, mock_executor):
        """Test successful retrieval of prefetched layer."""
        mock_cuda = Mock()
        mock_cuda.is_available.return_value = False

        with patch("src.optimizations.async_decompression.torch.cuda", mock_cuda):
            mock_time.side_effect = [0, 0.001]

            config = AsyncDecompressionConfig(use_cuda_streams=False)
            decompressor = AsyncDecompressor(config)

            # Mock prefetch queue
            mock_future = Mock()
            mock_tensor = torch.randn(128, 768)
            mock_future.result.return_value = mock_tensor

            decompressor.prefetch_queue.put(("layer1", mock_future))

            result = decompressor.get_prefetched_layer("layer1", timeout=1.0)

            assert result is mock_tensor
            assert decompressor.stats["total_wait_time_ms"] > 0

    @patch("src.optimizations.async_decompression.ThreadPoolExecutor")
    def test_get_prefetched_layer_timeout(self, mock_executor):
        """Test timeout when layer not prefetched."""
        mock_cuda = Mock()
        mock_cuda.is_available.return_value = False

        with patch("src.optimizations.async_decompression.torch.cuda", mock_cuda):
            config = AsyncDecompressionConfig(use_cuda_streams=False)
            decompressor = AsyncDecompressor(config)

            result = decompressor.get_prefetched_layer("nonexistent", timeout=0.1)

            assert result is None

    @patch("src.optimizations.async_decompression.torch.cuda")
    @patch("src.optimizations.async_decompression.importlib.import_module")
    def test_decompress_zstd_import_error(self, mock_import, mock_cuda):
        """Test ZSTD decompression when import fails."""
        mock_cuda.is_available.return_value = False
        mock_import.side_effect = ImportError("No module named 'zstandard'")

        config = AsyncDecompressionConfig(compression_format="zstd")
        decompressor = AsyncDecompressor(config)

        buffer = torch.empty((10,), dtype=torch.float32)
        result = decompressor._decompress_zstd(b"data", buffer)

        # Should return the buffer unchanged
        assert result is buffer

    @patch("src.optimizations.async_decompression.torch.cuda")
    def test_wait_for_layer_cached(self, mock_cuda):
        """Test wait_for_layer when layer is already cached."""
        mock_cuda.is_available.return_value = False

        config = AsyncDecompressionConfig(use_cuda_streams=False)
        decompressor = AsyncDecompressor(config)

        cached_tensor = torch.randn(128, 768)
        decompressor.decompressed_cache["layer1"] = cached_tensor

        result = decompressor.wait_for_layer("layer1")

        assert result is cached_tensor

    @patch("src.optimizations.async_decompression.torch.cuda")
    def test_wait_for_layer_not_found(self, mock_cuda):
        """Test wait_for_layer when layer not found."""
        mock_cuda.is_available.return_value = False

        config = AsyncDecompressionConfig(use_cuda_streams=False)
        decompressor = AsyncDecompressor(config)

        result = decompressor.wait_for_layer("nonexistent")

        assert result is None

    def test_clear_cache(self):
        """Test clearing decompression cache."""
        config = AsyncDecompressionConfig(use_cuda_streams=False)
        decompressor = AsyncDecompressor(config)

        # Add some cached data
        decompressor.decompressed_cache["layer1"] = torch.randn(128, 768)
        decompressor.decompressed_cache["layer2"] = torch.randn(128, 768)

        decompressor.clear_cache()

        assert len(decompressed_cache := decompressor.decompressed_cache) == 0

    def test_get_stats(self):
        """Test getting decompression statistics."""
        config = AsyncDecompressionConfig(use_cuda_streams=False)
        decompressor = AsyncDecompressor(config)

        # Set some stats
        decompressor.stats["total_decompressions"] = 10
        decompressor.stats["async_decompressions"] = 5
        decompressor.stats["cache_hits"] = 3
        decompressor.stats["total_wait_time_ms"] = 100

        stats = decompressor.get_stats()

        assert stats["async_ratio"] == 0.5
        assert stats["avg_wait_time_ms"] == 10.0
        assert stats["cache_hit_rate"] == 0.3

    def test_get_stats_empty(self):
        """Test getting stats when no decompressions."""
        config = AsyncDecompressionConfig(use_cuda_streams=False)
        decompressor = AsyncDecompressor(config)

        stats = decompressor.get_stats()

        assert stats["async_ratio"] == 0.0
        assert stats["avg_wait_time_ms"] == 0.0
        assert stats["cache_hit_rate"] == 0.0

    @patch("src.optimizations.async_decompression.torch.cuda")
    def test_shutdown(self, mock_cuda):
        """Test shutdown procedure."""
        mock_cuda.is_available.return_value = True
        mock_stream = Mock()
        mock_cuda.default_stream.return_value = mock_stream

        config = AsyncDecompressionConfig(use_cuda_streams=True)
        decompressor = AsyncDecompressor(config)

        # Add some cached data
        decompressor.decompressed_cache["layer1"] = torch.randn(128, 768)

        decompressor.shutdown()

        mock_stream.synchronize.assert_called_once()


class TestStreamingLayerLoader:
    """Test StreamingLayerLoader for layer loading."""

    @patch("src.optimizations.async_decompression.AsyncDecompressor")
    def test_init(self, mock_decompressor_class):
        """Test initialization."""
        compressed_layers = {"layer1": b"compressed"}
        layer_shapes = {"layer1": (128, 768)}

        loader = StreamingLayerLoader(compressed_layers, layer_shapes)

        assert loader.compressed_layers == compressed_layers
        assert loader.layer_shapes == layer_shapes
        assert loader.current_layer_idx == 0

    @patch("src.optimizations.async_decompression.torch.cuda")
    @patch("src.optimizations.async_decompression.AsyncDecompressor")
    def test_load_layer_from_cache(self, mock_decompressor_class, mock_cuda):
        """Test loading layer from prefetched cache."""
        mock_cuda.is_available.return_value = False

        mock_decompressor = Mock()
        mock_decompressor.get_prefetched_layer.return_value = torch.randn(128, 768)
        mock_decompressor_class.return_value = mock_decompressor

        compressed_layers = {"layer1": b"compressed"}
        layer_shapes = {"layer1": (128, 768)}

        loader = StreamingLayerLoader(compressed_layers, layer_shapes)

        result = loader.load_layer("layer1", "cpu")

        mock_decompressor.get_prefetched_layer.assert_called_once_with("layer1")
        assert result is not None

    @patch("src.optimizations.async_decompression.torch.cuda")
    @patch("src.optimizations.async_decompression.AsyncDecompressor")
    def test_load_layer_decompress(self, mock_decompressor_class, mock_cuda):
        """Test decompressing layer when not cached."""
        mock_cuda.is_available.return_value = False

        mock_decompressor = Mock()
        mock_decompressor.get_prefetched_layer.return_value = None  # Not cached
        mock_decompressor.decompress_layer_async.return_value = torch.randn(128, 768)
        mock_decompressor_class.return_value = mock_decompressor

        compressed_layers = {"layer1": b"compressed"}
        layer_shapes = {"layer1": (128, 768)}

        loader = StreamingLayerLoader(compressed_layers, layer_shapes)

        result = loader.load_layer("layer1", "cpu")

        mock_decompressor.decompress_layer_async.assert_called_once()

    @patch("src.optimizations.async_decompression.torch.cuda")
    @patch("src.optimizations.async_decompression.AsyncDecompressor")
    def test_prefetch_next_layers(self, mock_decompressor_class, mock_cuda):
        """Test prefetching next layers."""
        mock_cuda.is_available.return_value = False

        mock_decompressor = Mock()
        mock_decompressor.decompressed_cache = {}  # Empty cache
        mock_decompressor_class.return_value = mock_decompressor

        compressed_layers = {"layer1": b"data1", "layer2": b"data2", "layer3": b"data3"}
        layer_shapes = {
            "layer1": (128, 768),
            "layer2": (128, 768),
            "layer3": (128, 768),
        }

        loader = StreamingLayerLoader(compressed_layers, layer_shapes)
        loader.prefetch_next_layers(current_idx=0, num_ahead=2)

        # Should prefetch layer2 and layer3
        assert mock_decompressor.prefetch_layer.call_count == 2
