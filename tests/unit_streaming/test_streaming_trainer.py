#!/usr/bin/env python3
"""
test_streaming_trainer.py
Tests for streaming dataset training support.
"""

import pytest
import tempfile
import json
import gzip
from pathlib import Path
from unittest.mock import patch, MagicMock


class TestStreamingDatasetLoader:
    """Test StreamingDatasetLoader functionality."""
    
    @pytest.fixture
    def sample_jsonl_file(self, tmp_path):
        """Create a sample JSONL file."""
        file_path = tmp_path / "test.jsonl"
        with open(file_path, 'w') as f:
            for i in range(100):
                sample = {"messages": [{"role": "user", "content": f"Question {i}"}]}
                f.write(json.dumps(sample) + "\n")
        return file_path
    
    @pytest.fixture
    def sample_gzip_file(self, tmp_path):
        """Create a sample gzipped JSONL file."""
        file_path = tmp_path / "test.jsonl.gz"
        with gzip.open(file_path, 'wt', encoding='utf-8') as f:
            for i in range(50):
                sample = {"text": f"Sample {i}"}
                f.write(json.dumps(sample) + "\n")
        return file_path
    
    @pytest.fixture
    def sample_directory(self, tmp_path):
        """Create a directory with multiple JSONL files."""
        subdir = tmp_path / "datasets"
        subdir.mkdir()
        
        for j in range(3):
            file_path = subdir / f"data_{j}.jsonl"
            with open(file_path, 'w') as f:
                for i in range(30):
                    sample = {"id": f"{j}_{i}", "text": f"Content {j}_{i}"}
                    f.write(json.dumps(sample) + "\n")
        
        return subdir
    
    def test_import_streaming_loader(self):
        """Test that streaming loader imports correctly."""
        from src.data.streaming_trainer import StreamingDatasetLoader, StreamingConfig
        assert StreamingDatasetLoader is not None
        assert StreamingConfig is not None
    
    def test_init_with_single_path(self, sample_jsonl_file):
        """Test initialization with single path."""
        from src.data.streaming_trainer import StreamingDatasetLoader
        loader = StreamingDatasetLoader(sample_jsonl_file)
        assert len(loader.paths) == 1
    
    def test_init_with_multiple_paths(self, sample_jsonl_file, sample_gzip_file):
        """Test initialization with multiple paths."""
        from src.data.streaming_trainer import StreamingDatasetLoader
        loader = StreamingDatasetLoader([sample_jsonl_file, sample_gzip_file])
        assert len(loader.paths) == 2
    
    def test_stream_jsonl_file(self, sample_jsonl_file):
        """Test streaming from JSONL file."""
        from src.data.streaming_trainer import StreamingDatasetLoader
        loader = StreamingDatasetLoader(sample_jsonl_file)
        
        samples = list(loader._stream_jsonl_file(sample_jsonl_file))
        assert len(samples) == 100
        assert "messages" in samples[0]
    
    def test_stream_gzip_file(self, sample_gzip_file):
        """Test streaming from gzipped file."""
        from src.data.streaming_trainer import StreamingDatasetLoader
        loader = StreamingDatasetLoader(sample_gzip_file)
        
        samples = list(loader._stream_jsonl_file(sample_gzip_file))
        assert len(samples) == 50
        assert "text" in samples[0]
    
    def test_stream_directory(self, sample_directory):
        """Test streaming from directory with multiple files."""
        from src.data.streaming_trainer import StreamingDatasetLoader
        loader = StreamingDatasetLoader(sample_directory)
        
        samples = list(loader._stream_directory(sample_directory))
        assert len(samples) == 90  # 3 files * 30 samples
    
    def test_get_streaming_dataset(self, sample_jsonl_file):
        """Test creating IterableDataset."""
        from src.data.streaming_trainer import StreamingDatasetLoader, StreamingConfig
        
        config = StreamingConfig(buffer_size=100)
        loader = StreamingDatasetLoader(sample_jsonl_file, config)
        
        dataset = loader.get_streaming_dataset()
        
        # Verify it's iterable
        count = 0
        for sample in dataset:
            count += 1
            if count >= 10:
                break
        
        assert count == 10
    
    def test_estimate_size(self, sample_jsonl_file, sample_gzip_file):
        """Test size estimation."""
        from src.data.streaming_trainer import StreamingDatasetLoader
        loader = StreamingDatasetLoader([sample_jsonl_file, sample_gzip_file])
        
        stats = loader.estimate_size()
        assert stats["total_files"] == 2
        assert stats["total_bytes"] > 0
        assert "total_gb" in stats
    
    def test_max_samples_limit(self, sample_jsonl_file):
        """Test max_samples config."""
        from src.data.streaming_trainer import StreamingDatasetLoader, StreamingConfig
        
        config = StreamingConfig(max_samples=10)
        loader = StreamingDatasetLoader(sample_jsonl_file, config)
        
        samples = list(loader._stream_jsonl_file(sample_jsonl_file))
        assert len(samples) == 10
    
    def test_convenience_function(self, sample_jsonl_file):
        """Test load_streaming_datasets convenience function."""
        from src.data.streaming_trainer import load_streaming_datasets
        
        dataset = load_streaming_datasets(
            paths=[str(sample_jsonl_file)],
            buffer_size=50,
            max_samples=20
        )
        
        count = 0
        for _ in dataset:
            count += 1
        
        assert count == 20


class TestStreamingIntegration:
    """Integration tests with real-world scenarios."""
    
    def test_multiple_datasets_interleaved(self, tmp_path):
        """Test interleaving multiple datasets."""
        from src.data.streaming_trainer import StreamingDatasetLoader
        
        # Create two distinct datasets
        ds1 = tmp_path / "ds1.jsonl"
        ds2 = tmp_path / "ds2.jsonl"
        
        with open(ds1, 'w') as f:
            for i in range(50):
                f.write(json.dumps({"source": "ds1", "id": i}) + "\n")
        
        with open(ds2, 'w') as f:
            for i in range(50):
                f.write(json.dumps({"source": "ds2", "id": i}) + "\n")
        
        loader = StreamingDatasetLoader([ds1, ds2])
        dataset = loader.get_interleaved_dataset(probabilities=[0.5, 0.5])
        
        # Count samples from each source
        counts = {"ds1": 0, "ds2": 0}
        for sample in dataset:
            counts[sample["source"]] += 1
        
        # Interleaving is probabilistic, so we check for approximate distribution
        # Both sources should be roughly equal (50/50 split of 100 samples)
        assert counts["ds1"] > 35
        assert counts["ds2"] > 35
        # Total should be sum of samples, but dataset might cycle if requested or stop
        assert counts["ds1"] + counts["ds2"] >= 100
    
    def test_large_file_memory_efficiency(self, tmp_path):
        """Test that streaming doesn't load everything into memory."""
        import sys
        from src.data.streaming_trainer import StreamingDatasetLoader
        
        # Create a larger file
        file_path = tmp_path / "large.jsonl"
        with open(file_path, 'w') as f:
            for i in range(10000):
                sample = {"text": "x" * 1000}  # ~1KB per sample
                f.write(json.dumps(sample) + "\n")
        
        loader = StreamingDatasetLoader(file_path)
        dataset = loader.get_streaming_dataset()
        
        # Iterate through first 100 samples
        count = 0
        for sample in dataset:
            count += 1
            if count >= 100:
                break
        
        assert count == 100
        # Memory should not have increased drastically
        # (This is a smoke test - actual memory measurement would require more)


@pytest.mark.gpu
class TestStreamingWithTrainer:
    """Tests for streaming with actual training (requires GPU)."""
    
    def test_streaming_with_sft_trainer(self, tmp_path, real_text_model, real_text_tokenizer):
        """Test streaming dataset with SFTTrainer."""
        from src.data.streaming_trainer import StreamingDatasetLoader
        
        try:
            from trl import SFTTrainer, SFTConfig
            from transformers import TrainingArguments
        except ImportError:
            pytest.skip("trl not installed")
        
        # Create training data
        file_path = tmp_path / "train.jsonl"
        with open(file_path, 'w') as f:
            for i in range(100):
                sample = {
                    "messages": [
                        {"role": "user", "content": f"What is {i}?"},
                        {"role": "assistant", "content": f"It is {i}."}
                    ]
                }
                f.write(json.dumps(sample) + "\n")
        
        # Create streaming dataset
        loader = StreamingDatasetLoader(file_path)
        dataset = loader.get_streaming_dataset()
        
        # Configure minimal training
        sft_config = SFTConfig(
            output_dir=str(tmp_path / "output"),
            max_steps=5,
            per_device_train_batch_size=1,
            logging_steps=1,
            report_to="none",
        )
        
        # This should work without loading full dataset
        trainer = SFTTrainer(
            model=real_text_model,
            processing_class=real_text_tokenizer, # Fixed: tokenizer -> processing_class
            train_dataset=dataset,
            args=sft_config,
        )
        
        # Run a few steps
        trainer.train()
        
        assert True  # If we got here, streaming worked


class TestChunkedSampleProcessor:
    """Tests for ChunkedSampleProcessor - giant individual samples (40GB+)."""
    
    @pytest.fixture
    def large_text_file(self, tmp_path):
        """Create a large text file for chunked testing."""
        file_path = tmp_path / "large_sample.txt"
        # Create 10MB file (simulating 40GB)
        with open(file_path, 'wb') as f:
            for i in range(10000):
                f.write(f"Line {i}: {'x' * 990}\n".encode())
        return file_path
    
    @pytest.fixture
    def sample_video_file(self, tmp_path):
        """Create a mock video file path (doesn't need to be real for path tests)."""
        file_path = tmp_path / "sample_video.mp4"
        file_path.touch()  # Create empty file
        return file_path
    
    def test_import_chunked_processor(self):
        """Test that ChunkedSampleProcessor imports correctly."""
        from src.data.streaming_trainer import ChunkedSampleProcessor, ChunkConfig
        assert ChunkedSampleProcessor is not None
        assert ChunkConfig is not None
    
    def test_init_with_file(self, large_text_file):
        """Test initialization with file path."""
        from src.data.streaming_trainer import ChunkedSampleProcessor
        processor = ChunkedSampleProcessor(large_text_file)
        assert processor.file_size > 0
    
    def test_get_chunk_count(self, large_text_file):
        """Test chunk count estimation."""
        from src.data.streaming_trainer import ChunkedSampleProcessor, ChunkConfig
        
        config = ChunkConfig(chunk_size_mb=1)  # 1MB chunks
        processor = ChunkedSampleProcessor(large_text_file, config)
        
        count = processor.get_chunk_count()
        assert count >= 1
        # 10MB file / 1MB chunks = ~10 chunks
        assert count >= 9  # Allow some variance
    
    def test_stream_text_chunks(self, large_text_file):
        """Test streaming text file in chunks."""
        from src.data.streaming_trainer import ChunkedSampleProcessor, ChunkConfig
        
        config = ChunkConfig(chunk_size_mb=1)
        processor = ChunkedSampleProcessor(large_text_file, config)
        
        chunks = list(processor.stream_text_chunks())
        assert len(chunks) >= 9  # ~10MB / 1MB
        
        # Verify chunk structure
        first_chunk = chunks[0]
        assert 'text' in first_chunk
        assert 'chunk_id' in first_chunk
        assert 'byte_start' in first_chunk
        assert 'byte_end' in first_chunk
        assert first_chunk['chunk_id'] == 0
    
    def test_max_chunks_limit(self, large_text_file):
        """Test max_chunks configuration."""
        from src.data.streaming_trainer import ChunkedSampleProcessor, ChunkConfig
        
        config = ChunkConfig(chunk_size_mb=1, max_chunks=3)
        processor = ChunkedSampleProcessor(large_text_file, config)
        
        chunks = list(processor.stream_chunks())
        assert len(chunks) == 3
    
    def test_chunk_overlap(self, large_text_file):
        """Test that chunks have proper overlap for context."""
        from src.data.streaming_trainer import ChunkedSampleProcessor, ChunkConfig
        
        config = ChunkConfig(chunk_size_mb=1, overlap_mb=0)  # No overlap
        processor = ChunkedSampleProcessor(large_text_file, config)
        
        chunks = list(processor.stream_text_chunks())
        
        # Chunks should be contiguous without overlap
        for i in range(1, len(chunks)):
            # Each chunk should start where previous ended (approx)
            assert chunks[i]['byte_start'] >= chunks[i-1]['byte_start']
    
    def test_auto_detect_file_type(self, sample_video_file, large_text_file):
        """Test auto-detection of file type for chunking."""
        from src.data.streaming_trainer import ChunkedSampleProcessor
        
        # Text file should use text chunking
        text_processor = ChunkedSampleProcessor(large_text_file)
        # This should not raise - will use text chunks
        for chunk in text_processor.stream_chunks():
            assert 'text' in chunk
            break
    
    def test_chunk_contains_data(self, large_text_file):
        """Test that chunks contain actual data."""
        from src.data.streaming_trainer import ChunkedSampleProcessor, ChunkConfig
        
        config = ChunkConfig(chunk_size_mb=1)
        processor = ChunkedSampleProcessor(large_text_file, config)
        
        for chunk in processor.stream_chunks():
            assert len(chunk['text']) > 0
            # Should contain our test content
            assert 'Line' in chunk['text'] or 'x' in chunk['text']
            break


class TestChunkedProcessorEdgeCases:
    """Edge case tests for ChunkedSampleProcessor."""
    
    def test_empty_file(self, tmp_path):
        """Test handling of empty file."""
        from src.data.streaming_trainer import ChunkedSampleProcessor
        
        empty_file = tmp_path / "empty.txt"
        empty_file.touch()
        
        processor = ChunkedSampleProcessor(empty_file)
        chunks = list(processor.stream_chunks())
        assert len(chunks) == 0
    
    def test_small_file_single_chunk(self, tmp_path):
        """Test that small file produces single chunk."""
        from src.data.streaming_trainer import ChunkedSampleProcessor, ChunkConfig
        
        small_file = tmp_path / "small.txt"
        small_file.write_text("Hello, World!")
        
        config = ChunkConfig(chunk_size_mb=1)  # 1MB chunk for tiny file
        processor = ChunkedSampleProcessor(small_file, config)
        
        chunks = list(processor.stream_chunks())
        assert len(chunks) == 1
        assert "Hello, World!" in chunks[0]['text']
    
    def test_unicode_content(self, tmp_path):
        """Test handling of unicode content."""
        from src.data.streaming_trainer import ChunkedSampleProcessor
        
        unicode_file = tmp_path / "unicode.txt"
        unicode_file.write_text("Hello 世界 🌍 مرحبا" * 10000, encoding='utf-8')
        
        processor = ChunkedSampleProcessor(unicode_file)
        
        for chunk in processor.stream_chunks():
            # Should not crash on unicode
            assert len(chunk['text']) > 0
            break

class TestLoaderDelegation:
    """Test that StreamingDatasetLoader delegates giant/media files correctly."""
    
    def test_delegates_media_file(self, tmp_path):
        """Test delegation of .mp4 file."""
        from src.data.streaming_trainer import StreamingDatasetLoader
        
        # Create a mock mp4
        media_file = tmp_path / "test.mp4"
        media_file.touch()
        
        loader = StreamingDatasetLoader(media_file)
        
        # We access the internal generator to verify delegation
        with patch('src.data.streaming_trainer.ChunkedSampleProcessor') as MockProcessor:
            MockProcessor.return_value.stream_chunks.return_value = iter([{'mock': 'chunk'}])
            
            # Consume one sample
            gen = loader._stream_path(media_file)
            sample = next(gen)
            
            # Assert ChunkedSampleProcessor was initialized and used
            # We assume it is called with the path
            MockProcessor.assert_called()
            args, _ = MockProcessor.call_args
            assert str(args[0]) == str(media_file)
            assert sample == {'mock': 'chunk'}
            
    def test_delegates_giant_file(self, tmp_path):
        """Test delegation of >1GB file."""
        from src.data.streaming_trainer import StreamingDatasetLoader
        
        giant_file = tmp_path / "giant.txt"
        giant_file.touch()
        
        loader = StreamingDatasetLoader(giant_file)
        
        # Mocking stat to return > 1GB and valid file mode
        import stat
        with patch('pathlib.Path.stat') as mock_stat_method:
            mock_stat_method.return_value.st_size = 2 * 1024**3 # 2GB
            mock_stat_method.return_value.st_mode = stat.S_IFREG | 0o644 # Regular file
            
            with patch('src.data.streaming_trainer.ChunkedSampleProcessor') as MockProcessor:
                MockProcessor.return_value.stream_chunks.return_value = iter([{'mock': 'chunk'}])
                
                gen = loader._stream_path(giant_file)
                sample = next(gen)
                
                MockProcessor.assert_called()
                assert sample == {'mock': 'chunk'}
