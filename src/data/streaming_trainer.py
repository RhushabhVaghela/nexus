#!/usr/bin/env python3
"""
streaming_trainer.py
Streaming training support for 500GB+ datasets AND giant individual samples.

Two modes:
1. STREAMING: Many small samples loaded on-demand (never all in memory)
2. CHUNKED: Giant individual samples (40GB+) processed in chunks

Usage:
    # Streaming many samples
    from src.data.streaming_trainer import StreamingDatasetLoader
    loader = StreamingDatasetLoader(["/path/to/datasets"])
    dataset = loader.get_streaming_dataset()
    
    # Chunked giant sample
    from src.data.streaming_trainer import ChunkedSampleProcessor
    processor = ChunkedSampleProcessor("/path/to/40gb_video.mp4")
    for chunk in processor.stream_chunks():
        # Each chunk fits in memory
        train_on_chunk(chunk)
"""

import json
import gzip
import logging
import mmap
from pathlib import Path
from typing import List, Optional, Iterator, Dict, Any, Union, Generator
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Try to import HuggingFace datasets
try:
    from datasets import IterableDataset, interleave_datasets, load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False
    logger.warning("datasets library not installed")


# =============================================================================
# CHUNKED PROCESSING for giant individual samples (40GB+)
# =============================================================================

@dataclass
class ChunkConfig:
    """Configuration for chunked processing of giant samples."""
    chunk_size_mb: int = 100          # Chunk size in MB
    overlap_mb: int = 10              # Overlap between chunks (for context)
    max_chunks: Optional[int] = None  # Limit chunks (None = all)
    
    # For video/audio
    frames_per_chunk: int = 100       # Frames per chunk for video
    seconds_per_chunk: float = 30.0   # Seconds per chunk for audio


class ChunkedSampleProcessor:
    """
    Process giant individual samples (40GB+) in memory-efficient chunks.
    
    Supports:
    - Text files: Read in byte chunks with overlap
    - Video files: Extract frames in batches
    - Audio files: Process in time segments
    - Binary files: Memory-mapped reading
    """
    
    def __init__(self, file_path: Union[str, Path], config: Optional[ChunkConfig] = None):
        self.path = Path(file_path)
        self.config = config or ChunkConfig()
        self.file_size = self.path.stat().st_size if self.path.exists() else 0
        
        logger.info(f"ChunkedSampleProcessor: {self.path.name} ({self.file_size / 1e9:.2f} GB)")
    
    def stream_text_chunks(self) -> Generator[Dict[str, Any], None, None]:
        """
        Stream text file in chunks with overlap for context preservation.
        
        Yields:
            Dict with 'text', 'chunk_id', 'byte_start', 'byte_end'
        """
        chunk_bytes = int(self.config.chunk_size_mb * 1024 * 1024)
        overlap_bytes = int(self.config.overlap_mb * 1024 * 1024)
        
        with open(self.path, 'rb') as f:
            chunk_id = 0
            byte_pos = 0
            
            while True:
                # Seek with overlap (except first chunk)
                if chunk_id > 0 and byte_pos > overlap_bytes:
                    f.seek(byte_pos - overlap_bytes)
                    byte_start = byte_pos - overlap_bytes
                else:
                    byte_start = byte_pos
                
                # Read chunk
                data = f.read(chunk_bytes)
                if not data:
                    break
                
                try:
                    text = data.decode('utf-8', errors='ignore')
                except:
                    text = data.decode('latin-1', errors='ignore')
                
                yield {
                    'text': text,
                    'chunk_id': chunk_id,
                    'byte_start': byte_start,
                    'byte_end': byte_start + len(data),
                    'total_size': self.file_size,
                }
                
                byte_pos = f.tell()
                chunk_id += 1
                
                if self.config.max_chunks and chunk_id >= self.config.max_chunks:
                    break
    
    def stream_video_frames(self) -> Generator[Dict[str, Any], None, None]:
        """
        Stream video frames in batches without loading full video.
        
        Requires: opencv-python (cv2)
        """
        try:
            import cv2
        except ImportError:
            raise ImportError("opencv-python required: pip install opencv-python")
        
        cap = cv2.VideoCapture(str(self.path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        frames_per_chunk = self.config.frames_per_chunk
        chunk_id = 0
        
        while True:
            frames = []
            for _ in range(frames_per_chunk):
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
            
            if not frames:
                break
            
            yield {
                'frames': frames,
                'chunk_id': chunk_id,
                'frame_start': chunk_id * frames_per_chunk,
                'frame_end': chunk_id * frames_per_chunk + len(frames),
                'fps': fps,
                'total_frames': total_frames,
            }
            
            chunk_id += 1
            if self.config.max_chunks and chunk_id >= self.config.max_chunks:
                break
        
        cap.release()
    
    def stream_audio_segments(self) -> Generator[Dict[str, Any], None, None]:
        """
        Stream audio in time segments without loading full file.
        
        Requires: librosa or soundfile
        """
        try:
            import soundfile as sf
        except ImportError:
            raise ImportError("soundfile required: pip install soundfile")
        
        with sf.SoundFile(str(self.path)) as audio_file:
            sr = audio_file.samplerate
            total_samples = len(audio_file)
            samples_per_chunk = int(self.config.seconds_per_chunk * sr)
            
            chunk_id = 0
            while True:
                audio_data = audio_file.read(samples_per_chunk)
                if len(audio_data) == 0:
                    break
                
                yield {
                    'audio': audio_data,
                    'chunk_id': chunk_id,
                    'sample_rate': sr,
                    'time_start': chunk_id * self.config.seconds_per_chunk,
                    'duration': len(audio_data) / sr,
                    'total_duration': total_samples / sr,
                }
                
                chunk_id += 1
                if self.config.max_chunks and chunk_id >= self.config.max_chunks:
                    break
    
    def stream_chunks(self) -> Generator[Dict[str, Any], None, None]:
        """
        Auto-detect file type and stream appropriate chunks.
        """
        suffix = self.path.suffix.lower()
        
        if suffix in ['.mp4', '.avi', '.mov', '.mkv', '.webm']:
            yield from self.stream_video_frames()
        elif suffix in ['.mp3', '.wav', '.flac', '.ogg', '.m4a']:
            yield from self.stream_audio_segments()
        else:
            # Default to text/binary chunked reading
            yield from self.stream_text_chunks()
    
    def get_chunk_count(self) -> int:
        """Estimate number of chunks without reading file."""
        chunk_bytes = self.config.chunk_size_mb * 1024 * 1024
        return max(1, self.file_size // chunk_bytes)


# =============================================================================
# STREAMING for many small samples
# =============================================================================


@dataclass
class StreamingConfig:
    """Configuration for streaming training."""
    buffer_size: int = 10000          # Shuffle buffer size
    num_shards: int = 1               # Number of shards for distributed
    shard_id: int = 0                 # Current shard ID
    prefetch_factor: int = 2          # Prefetch multiplier
    max_samples: Optional[int] = None # Limit total samples (None = infinite)


class StreamingDatasetLoader:
    """
    Streaming dataset loader for memory-efficient training on 500GB+ datasets.
    
    Supports:
    - JSONL files (streaming line-by-line)
    - Compressed files (.gz, .zip)
    - HuggingFace datasets (streaming mode)
    - Multiple dataset sources (interleaved)
    """
    
    def __init__(
        self,
        paths: Union[str, Path, List[Union[str, Path]]],
        config: Optional[StreamingConfig] = None
    ):
        """
        Initialize streaming loader.
        
        Args:
            paths: Single path or list of paths to datasets
            config: Streaming configuration
        """
        if isinstance(paths, (str, Path)):
            paths = [paths]
        
        self.paths = [Path(p) for p in paths]
        self.config = config or StreamingConfig()
        
        logger.info(f"StreamingDatasetLoader initialized with {len(self.paths)} sources")
        for p in self.paths:
            logger.info(f"  Source: {p}")
    
    def _stream_jsonl_file(self, file_path: Path) -> Iterator[Dict[str, Any]]:
        """Stream samples from a JSONL file."""
        count = 0
        
        # Handle compressed files
        if file_path.suffix == '.gz':
            opener = lambda: gzip.open(file_path, 'rt', encoding='utf-8')
        else:
            opener = lambda: open(file_path, 'r', encoding='utf-8')
        
        with opener() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                try:
                    sample = json.loads(line)
                    yield sample
                    count += 1
                    
                    if self.config.max_samples and count >= self.config.max_samples:
                        return
                except json.JSONDecodeError:
                    continue
    
    def _stream_directory(self, dir_path: Path) -> Iterator[Dict[str, Any]]:
        """Stream samples from all JSONL files in a directory."""
        extensions = ['*.jsonl', '*.jsonl.gz', '*.json', '*.json.gz']
        
        for ext in extensions:
            for file_path in sorted(dir_path.rglob(ext)):
                logger.debug(f"Streaming from: {file_path}")
                yield from self._stream_jsonl_file(file_path)
    
    def _stream_path(self, path: Path) -> Iterator[Dict[str, Any]]:
        """Stream samples from a path (file or directory)."""
        if path.is_file():
            # GIANT FILE / MEDIA CHECK
            suffix = path.suffix.lower()
            is_media = suffix in ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.mp3', '.wav', '.flac', '.ogg']
            # Treat as giant if > 1GB or explicitly media, BUT prioritize JSONL handling for strict JSONL files
            is_huge = path.stat().st_size > 1 * 1024**3
            
            # If it's a huge text/jsonl file, we might still want line-by-line (handled by _stream_jsonl_file)
            # UNLESS it's a single massive line or binary.
            # For now, we trust _stream_jsonl_file for .jsonl/.json regardless of size (it is streaming!).
            # We use ChunkedSampleProcessor for media OR non-jsonl huge files (e.g. huge .txt corpus as one string)
            
            if is_media or (is_huge and suffix not in ['.jsonl', '.json', '.gz']):
                logger.info(f"📦 Detected giant/media file: {path.name}. Using ChunkedSampleProcessor.")
                try:
                    # Initialize processor with default config (can be passed via self.config if extended)
                    processor = ChunkedSampleProcessor(path)
                    yield from processor.stream_chunks()
                except Exception as e:
                    logger.error(f"Failed to chunk file {path}: {e}")
            else:
                yield from self._stream_jsonl_file(path)
        elif path.is_dir():
            yield from self._stream_directory(path)
        else:
            logger.warning(f"Path not found: {path}")
    
    def _create_generator(self):
        """Create a generator that yields from all paths."""
        def gen():
            for path in self.paths:
                logger.info(f"📂 Streaming from: {path}")
                yield from self._stream_path(path)
        return gen
    
    def get_streaming_dataset(self) -> "IterableDataset":
        """
        Get a HuggingFace IterableDataset for streaming training.
        
        Returns:
            IterableDataset that streams samples on-demand
        """
        if not HAS_DATASETS:
            raise ImportError("datasets library required: pip install datasets")
        
        # Create IterableDataset from generator
        dataset = IterableDataset.from_generator(self._create_generator())
        
        # Apply shuffle buffer for randomization
        if self.config.buffer_size > 0:
            dataset = dataset.shuffle(buffer_size=self.config.buffer_size)
        
        logger.info(f"Created streaming dataset with buffer_size={self.config.buffer_size}")
        return dataset
    
    def get_interleaved_dataset(
        self,
        probabilities: Optional[List[float]] = None,
        stopping_strategy: str = "all_exhausted"
    ) -> "IterableDataset":
        """
        Create interleaved dataset from multiple sources.
        
        Args:
            probabilities: Sampling probabilities for each source (uniform if None)
            stopping_strategy: "all_exhausted" or "first_exhausted"
        
        Returns:
            Interleaved IterableDataset
        """
        if not HAS_DATASETS:
            raise ImportError("datasets library required")
        
        # Create separate dataset for each path
        datasets = []
        for path in self.paths:
            def make_gen(p):
                return lambda: self._stream_path(p)
            
            ds = IterableDataset.from_generator(make_gen(path))
            datasets.append(ds)
        
        # Interleave with probabilities
        if len(datasets) == 1:
            return datasets[0].shuffle(buffer_size=self.config.buffer_size)
        
        interleaved = interleave_datasets(
            datasets,
            probabilities=probabilities,
            stopping_strategy=stopping_strategy
        )
        
        if self.config.buffer_size > 0:
            interleaved = interleaved.shuffle(buffer_size=self.config.buffer_size)
        
        logger.info(f"Created interleaved dataset from {len(datasets)} sources")
        return interleaved
    
    def estimate_size(self) -> Dict[str, Any]:
        """Estimate total dataset size without loading."""
        stats = {
            "total_files": 0,
            "total_bytes": 0,
            "estimated_samples": 0,
            "paths": []
        }
        
        for path in self.paths:
            path_stats = {"path": str(path), "files": 0, "bytes": 0}
            
            if path.is_file():
                path_stats["files"] = 1
                path_stats["bytes"] = path.stat().st_size
            elif path.is_dir():
                for ext in ['*.jsonl', '*.jsonl.gz', '*.json']:
                    for f in path.rglob(ext):
                        path_stats["files"] += 1
                        path_stats["bytes"] += f.stat().st_size
            
            stats["total_files"] += path_stats["files"]
            stats["total_bytes"] += path_stats["bytes"]
            stats["paths"].append(path_stats)
        
        # Rough estimate: ~500 bytes per sample average
        stats["estimated_samples"] = stats["total_bytes"] // 500
        stats["total_gb"] = stats["total_bytes"] / (1024**3)
        
        return stats


def load_streaming_datasets(
    paths: List[str],
    buffer_size: int = 10000,
    max_samples: Optional[int] = None
) -> "IterableDataset":
    """
    Convenience function to load streaming datasets.
    
    Args:
        paths: List of dataset paths
        buffer_size: Shuffle buffer size
        max_samples: Optional sample limit
    
    Returns:
        IterableDataset for training
    """
    config = StreamingConfig(
        buffer_size=buffer_size,
        max_samples=max_samples
    )
    loader = StreamingDatasetLoader(paths, config)
    return loader.get_streaming_dataset()


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", help="Dataset paths")
    parser.add_argument("--estimate", action="store_true", help="Estimate size only")
    args = parser.parse_args()
    
    loader = StreamingDatasetLoader(args.paths)
    
    if args.estimate:
        stats = loader.estimate_size()
        print(f"\n📊 Dataset Statistics:")
        print(f"   Total files: {stats['total_files']}")
        print(f"   Total size: {stats['total_gb']:.2f} GB")
        print(f"   Estimated samples: {stats['estimated_samples']:,}")
    else:
        # Stream first 5 samples
        dataset = loader.get_streaming_dataset()
        print("\n🔄 First 5 samples:")
        for i, sample in enumerate(dataset):
            if i >= 5:
                break
            print(f"  {i+1}: {str(sample)[:100]}...")
