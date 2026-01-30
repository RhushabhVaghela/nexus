#!/usr/bin/env python3
"""
Text-to-Speech (TTS) Benchmark Suite

Comprehensive benchmarks for TTS engine performance:
- Synthesis latency (short text < 50 chars)
- Synthesis latency (medium text 200 chars)
- Synthesis latency (long text 1000 chars)
- Voice cloning setup time
- Cache hit performance (memory vs disk)
- Streaming throughput (chunks/sec)
- Language comparison (en, zh, ja)
- Audio format conversion time

Usage:
    python benchmarks/test_tts_benchmark.py --all
    python benchmarks/test_tts_benchmark.py --category latency
    python benchmarks/test_tts_benchmark.py --languages en zh ja
    python benchmarks/test_tts_benchmark.py --memory-profiling
    python benchmarks/test_tts_benchmark.py --output results/tts_benchmark.json

Environment Variables:
    BENCHMARK_ITERATIONS: Number of iterations (default: 100)
    BENCHMARK_WARMUP: Number of warmup iterations (default: 10)
    TTS_MODEL_PATH: Path to TTS model for integration benchmarks
"""

import os
import sys
import json
import time
import argparse
import tempfile
import shutil
import statistics
import tracemalloc
import gc
import hashlib
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any, Callable
from contextlib import contextmanager
from unittest.mock import Mock, patch

import numpy as np

# Ensure src is in path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️  PyTorch not available. Running in mock mode.")


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""
    name: str
    category: str
    iterations: int
    total_time: float
    mean_time: float
    median_time: float
    std_dev: float
    min_time: float
    max_time: float
    p95_time: float
    p99_time: float
    memory_delta_mb: Optional[float] = None
    rtf: Optional[float] = None  # Real-time factor (audio duration / synthesis time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BenchmarkSuite:
    """Collection of benchmark results."""
    name: str
    timestamp: str
    python_version: str
    platform: str
    torch_version: Optional[str] = None
    results: List[BenchmarkResult] = field(default_factory=list)
    
    def add_result(self, result: BenchmarkResult):
        self.results.append(result)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "timestamp": self.timestamp,
            "python_version": self.python_version,
            "platform": self.platform,
            "torch_version": self.torch_version,
            "results": [r.to_dict() for r in self.results],
            "summary": {
                "total_benchmarks": len(self.results),
                "categories": list(set(r.category for r in self.results))
            }
        }
    
    def save(self, path: str):
        """Save benchmark results to JSON file."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"📊 Benchmark results saved to {path}")


class BenchmarkRunner:
    """Runner for executing benchmarks with timing and memory profiling."""
    
    def __init__(self, iterations: int = 100, warmup: int = 10):
        self.iterations = iterations
        self.warmup = warmup
        self.results: List[BenchmarkResult] = []
    
    def run(
        self,
        name: str,
        category: str,
        func: Callable,
        *args,
        memory_profile: bool = False,
        calculate_rtf: bool = False,
        audio_duration: float = 1.0,
        metadata: Optional[Dict] = None,
        **kwargs
    ) -> BenchmarkResult:
        """
        Run a benchmark function and collect timing statistics.
        
        Args:
            name: Benchmark name
            category: Benchmark category
            func: Function to benchmark
            args: Positional arguments for func
            memory_profile: Whether to profile memory usage
            calculate_rtf: Whether to calculate real-time factor
            audio_duration: Audio duration in seconds for RTF calculation
            metadata: Additional metadata to store
            kwargs: Keyword arguments for func
        """
        times = []
        memory_delta = None
        
        # Warmup iterations
        for _ in range(self.warmup):
            result = func(*args, **kwargs)
        
        # Actual benchmark iterations
        if memory_profile:
            gc.collect()
            tracemalloc.start()
            start_mem = tracemalloc.get_traced_memory()[0]
        
        for _ in range(self.iterations):
            start = time.perf_counter()
            result = func(*args, **kwargs)
            end = time.perf_counter()
            times.append(end - start)
        
        if memory_profile:
            end_mem = tracemalloc.get_traced_memory()[0]
            tracemalloc.stop()
            memory_delta = (end_mem - start_mem) / (1024 * 1024)  # MB
        
        # Calculate statistics
        times.sort()
        mean_time = statistics.mean(times)
        rtf = (audio_duration / mean_time) if calculate_rtf else None
        
        result = BenchmarkResult(
            name=name,
            category=category,
            iterations=self.iterations,
            total_time=sum(times),
            mean_time=mean_time,
            median_time=statistics.median(times),
            std_dev=statistics.stdev(times) if len(times) > 1 else 0.0,
            min_time=min(times),
            max_time=max(times),
            p95_time=times[int(len(times) * 0.95)],
            p99_time=times[int(len(times) * 0.99)],
            memory_delta_mb=memory_delta,
            rtf=rtf,
            metadata=metadata or {}
        )
        
        self.results.append(result)
        return result
    
    def print_result(self, result: BenchmarkResult):
        """Print a benchmark result in a formatted way."""
        print(f"\n{'='*60}")
        print(f"📊 {result.name}")
        print(f"{'='*60}")
        print(f"Category: {result.category}")
        print(f"Iterations: {result.iterations}")
        print(f"Total Time: {result.total_time:.4f}s")
        print(f"Mean: {result.mean_time*1000:.4f}ms")
        print(f"Median: {result.median_time*1000:.4f}ms")
        print(f"Std Dev: {result.std_dev*1000:.4f}ms")
        print(f"Min: {result.min_time*1000:.4f}ms")
        print(f"Max: {result.max_time*1000:.4f}ms")
        print(f"P95: {result.p95_time*1000:.4f}ms")
        print(f"P99: {result.p99_time*1000:.4f}ms")
        if result.memory_delta_mb is not None:
            print(f"Memory Delta: {result.memory_delta_mb:.4f} MB")
        if result.rtf is not None:
            print(f"RTF: {result.rtf:.2f}x (real-time)")


class MockTTSComponents:
    """Mock implementations for TTS benchmarking."""
    
    def __init__(self):
        self.cache = {}
        self.disk_cache_dir = tempfile.mkdtemp()
        self.vocoder = None
        self.acoustic_model = None
        self.speaker_embeddings = {}
    
    def create_mock_acoustic_model(self):
        """Create a mock acoustic model."""
        if TORCH_AVAILABLE:
            return MockAcousticModelTorch()
        else:
            return MockAcousticModelNumpy()
    
    def create_mock_vocoder(self):
        """Create a mock vocoder."""
        if TORCH_AVAILABLE:
            return MockVocoderTorch()
        else:
            return MockVocoderNumpy()
    
    def text_to_phonemes(self, text: str, language: str = "en") -> List[str]:
        """Convert text to phonemes."""
        # Simplified phoneme conversion
        phoneme_delay = {
            "en": 0.0001,
            "zh": 0.0002,
            "ja": 0.0003,
        }
        time.sleep(phoneme_delay.get(language, 0.0001) * len(text))
        return list(text.lower())
    
    def generate_speaker_embedding(self, reference_audio: Optional[np.ndarray] = None):
        """Generate speaker embedding for voice cloning."""
        if TORCH_AVAILABLE:
            if reference_audio is not None:
                return torch.randn(256)
            return torch.randn(256)
        else:
            return np.random.randn(256).astype(np.float32)


class MockAcousticModelTorch(nn.Module):
    """Mock acoustic model using PyTorch."""
    
    def __init__(self):
        super().__init__()
        self.encoder = nn.LSTM(80, 256, num_layers=2, batch_first=True)
        self.decoder = nn.Linear(256, 80)
    
    def forward(self, phonemes, speaker_emb=None):
        # Simulate mel-spectrogram generation
        batch_size = 1
        seq_len = len(phonemes) * 10  # 10 frames per phoneme
        mels = torch.randn(batch_size, seq_len, 80)
        return mels


class MockAcousticModelNumpy:
    """Mock acoustic model using NumPy."""
    
    def forward(self, phonemes, speaker_emb=None):
        batch_size = 1
        seq_len = len(phonemes) * 10
        return np.random.randn(batch_size, seq_len, 80).astype(np.float32)


class MockVocoderTorch(nn.Module):
    """Mock vocoder using PyTorch."""
    
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(80, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(128, 1, kernel_size=3, padding=1)
    
    def forward(self, mels):
        # Simulate waveform generation from mel-spectrogram
        x = torch.relu(self.conv1(mels.transpose(1, 2)))
        waveform = torch.tanh(self.conv2(x))
        return waveform.squeeze(1)


class MockVocoderNumpy:
    """Mock vocoder using NumPy."""
    
    def forward(self, mels):
        # Simplified waveform generation
        seq_len = mels.shape[1]
        return np.random.randn(seq_len * 256).astype(np.float32) * 0.1


class SynthesisLatencyBenchmarks:
    """Benchmarks for TTS synthesis latency."""
    
    def __init__(self, runner: BenchmarkRunner, mock_components: MockTTSComponents):
        self.runner = runner
        self.mock = mock_components
    
    def benchmark_text_length_latency(self):
        """Benchmark synthesis latency for different text lengths."""
        print("\n" + "="*60)
        print("⏱️  SYNTHESIS LATENCY BY TEXT LENGTH")
        print("="*60)
        
        acoustic_model = self.mock.create_mock_acoustic_model()
        vocoder = self.mock.create_mock_vocoder()
        
        texts = {
            "short": "Hello, world!",
            "medium": "This is a medium length text for testing text to speech synthesis. It contains about two hundred characters for benchmarking.",
            "long": "This is a long text for comprehensive TTS benchmarking. " * 20 + "It contains approximately one thousand characters to test the system's performance with longer inputs that require more processing time and memory.",
        }
        
        for category, text in texts.items():
            phonemes = self.mock.text_to_phonemes(text)
            estimated_duration = len(phonemes) * 0.1  # 100ms per phoneme
            
            def synthesize():
                if TORCH_AVAILABLE:
                    with torch.no_grad():
                        mels = acoustic_model(phonemes)
                        audio = vocoder(mels)
                        return audio
                else:
                    mels = acoustic_model.forward(phonemes)
                    audio = vocoder.forward(mels)
                    return audio
            
            result = self.runner.run(
                f"synthesis_{category}_{len(text)}chars",
                "latency",
                synthesize,
                calculate_rtf=True,
                audio_duration=estimated_duration,
                metadata={
                    "text_category": category,
                    "text_length": len(text),
                    "num_phonemes": len(phonemes),
                    "estimated_audio_duration": estimated_duration
                }
            )
            self.runner.print_result(result)
    
    def benchmark_phoneme_complexity(self):
        """Benchmark synthesis for different phoneme complexities."""
        print("\n" + "="*60)
        print("🔤 PHONEME COMPLEXITY BENCHMARKS")
        print("="*60)
        
        acoustic_model = self.mock.create_mock_acoustic_model()
        vocoder = self.mock.create_mock_vocoder()
        
        # Different phoneme patterns
        patterns = {
            "simple": ["a", "e", "i", "o", "u"] * 10,
            "complex": ["th", "sh", "ch", "ng", "zh", "ae", "oe", "ue"] * 10,
            "mixed": ["a", "th", "e", "sh", "i", "ch", "o", "ng", "u", "zh"] * 10,
        }
        
        for pattern_name, phonemes in patterns.items():
            def synthesize():
                if TORCH_AVAILABLE:
                    with torch.no_grad():
                        mels = acoustic_model(phonemes)
                        audio = vocoder(mels)
                        return audio
                else:
                    mels = acoustic_model.forward(phonemes)
                    audio = vocoder.forward(mels)
                    return audio
            
            result = self.runner.run(
                f"phoneme_complexity_{pattern_name}",
                "latency",
                synthesize,
                metadata={
                    "pattern": pattern_name,
                    "num_phonemes": len(phonemes)
                }
            )
            self.runner.print_result(result)


class VoiceCloningBenchmarks:
    """Benchmarks for voice cloning performance."""
    
    def __init__(self, runner: BenchmarkRunner, mock_components: MockTTSComponents):
        self.runner = runner
        self.mock = mock_components
    
    def benchmark_voice_cloning_setup(self):
        """Benchmark voice cloning setup time."""
        print("\n" + "="*60)
        print("🎙️  VOICE CLONING SETUP TIME")
        print("="*60)
        
        # Mock reference audio lengths
        reference_lengths = [
            ("short_3s", 3),
            ("medium_10s", 10),
            ("long_30s", 30),
        ]
        
        for name, duration in reference_lengths:
            # Create mock reference audio
            sample_rate = 22050
            reference_audio = np.random.randn(duration * sample_rate).astype(np.float32)
            
            def clone_voice():
                # Simulate voice cloning setup
                time.sleep(0.01 * duration)  # Setup time proportional to audio length
                speaker_emb = self.mock.generate_speaker_embedding(reference_audio)
                return speaker_emb
            
            result = self.runner.run(
                f"voice_clone_setup_{name}",
                "voice_cloning",
                clone_voice,
                memory_profile=True,
                metadata={
                    "reference_duration": duration,
                    "sample_rate": sample_rate,
                    "reference_samples": len(reference_audio)
                }
            )
            self.runner.print_result(result)
    
    def benchmark_synthesis_with_cloned_voice(self):
        """Benchmark synthesis using cloned voice."""
        print("\n" + "="*60)
        print("🗣️  SYNTHESIS WITH CLONED VOICE")
        print("="*60)
        
        acoustic_model = self.mock.create_mock_acoustic_model()
        vocoder = self.mock.create_mock_vocoder()
        speaker_emb = self.mock.generate_speaker_embedding()
        text = "This is a test of voice cloning synthesis."
        phonemes = self.mock.text_to_phonemes(text)
        
        def synthesize_cloned():
            if TORCH_AVAILABLE:
                with torch.no_grad():
                    mels = acoustic_model(phonemes, speaker_emb)
                    audio = vocoder(mels)
                    return audio
            else:
                mels = acoustic_model.forward(phonemes, speaker_emb)
                audio = vocoder.forward(mels)
                return audio
        
        result = self.runner.run(
            "synthesis_with_cloned_voice",
            "voice_cloning",
            synthesize_cloned,
            metadata={
                "text_length": len(text),
                "num_phonemes": len(phonemes)
            }
        )
        self.runner.print_result(result)


class CachePerformanceBenchmarks:
    """Benchmarks for cache hit performance."""
    
    def __init__(self, runner: BenchmarkRunner, mock_components: MockTTSComponents):
        self.runner = runner
        self.mock = mock_components
    
    def benchmark_memory_cache(self):
        """Benchmark memory cache hit performance."""
        print("\n" + "="*60)
        print("💾 MEMORY CACHE PERFORMANCE")
        print("="*60)
        
        acoustic_model = self.mock.create_mock_acoustic_model()
        vocoder = self.mock.create_mock_vocoder()
        
        texts = [
            "Hello world",
            "Text to speech synthesis",
            "Benchmark testing",
        ]
        
        # Pre-populate cache
        cache = {}
        for text in texts:
            phonemes = self.mock.text_to_phonemes(text)
            if TORCH_AVAILABLE:
                with torch.no_grad():
                    cache[text] = (acoustic_model(phonemes), vocoder(acoustic_model(phonemes)))
            else:
                cache[text] = (acoustic_model.forward(phonemes), vocoder.forward(acoustic_model.forward(phonemes)))
        
        # Benchmark cache hits
        for text in texts:
            def cache_lookup():
                return cache.get(text, None)
            
            result = self.runner.run(
                f"memory_cache_hit_{len(text)}chars",
                "cache",
                cache_lookup,
                metadata={"text": text, "cache_type": "memory"}
            )
            self.runner.print_result(result)
        
        # Benchmark cache miss (compute)
        for text in texts:
            phonemes = self.mock.text_to_phonemes(text)
            
            def cache_miss():
                if TORCH_AVAILABLE:
                    with torch.no_grad():
                        mels = acoustic_model(phonemes)
                        audio = vocoder(mels)
                        return audio
                else:
                    mels = acoustic_model.forward(phonemes)
                    audio = vocoder.forward(mels)
                    return audio
            
            result = self.runner.run(
                f"cache_miss_{len(text)}chars",
                "cache",
                cache_miss,
                metadata={"text": text, "cache_type": "miss"}
            )
            self.runner.print_result(result)
    
    def benchmark_disk_cache(self):
        """Benchmark disk cache read/write performance."""
        print("\n" + "="*60)
        print("💿 DISK CACHE PERFORMANCE")
        print("="*60)
        
        cache_dir = Path(self.mock.disk_cache_dir)
        
        # Create mock audio data
        audio_data = np.random.randn(22050 * 3).astype(np.float32)  # 3 seconds
        text_hash = hashlib.md5("test text".encode()).hexdigest()
        cache_file = cache_dir / f"{text_hash}.npy"
        
        # Pre-write to disk
        np.save(cache_file, audio_data)
        
        # Benchmark disk read
        def disk_read():
            return np.load(cache_file)
        
        result = self.runner.run(
            "disk_cache_read",
            "cache",
            disk_read,
            metadata={"cache_type": "disk_read", "file_size_mb": audio_data.nbytes / (1024*1024)}
        )
        self.runner.print_result(result)
        
        # Benchmark disk write
        def disk_write():
            np.save(cache_file, audio_data)
            return True
        
        result = self.runner.run(
            "disk_cache_write",
            "cache",
            disk_write,
            metadata={"cache_type": "disk_write", "file_size_mb": audio_data.nbytes / (1024*1024)}
        )
        self.runner.print_result(result)


class StreamingBenchmarks:
    """Benchmarks for streaming TTS performance."""
    
    def __init__(self, runner: BenchmarkRunner, mock_components: MockTTSComponents):
        self.runner = runner
        self.mock = mock_components
    
    def benchmark_streaming_throughput(self):
        """Benchmark streaming throughput (chunks/sec)."""
        print("\n" + "="*60)
        print("📡 STREAMING THROUGHPUT")
        print("="*60)
        
        acoustic_model = self.mock.create_mock_acoustic_model()
        vocoder = self.mock.create_mock_vocoder()
        
        chunk_sizes = [1, 5, 10, 20]  # phonemes per chunk
        
        for chunk_size in chunk_sizes:
            text = "This is streaming test text with multiple words."
            phonemes = self.mock.text_to_phonemes(text)
            
            def stream_synthesize():
                chunks_processed = 0
                for i in range(0, len(phonemes), chunk_size):
                    chunk = phonemes[i:i+chunk_size]
                    if TORCH_AVAILABLE:
                        with torch.no_grad():
                            mels = acoustic_model(chunk)
                            audio = vocoder(mels)
                    else:
                        mels = acoustic_model.forward(chunk)
                        audio = vocoder.forward(mels)
                    chunks_processed += 1
                return chunks_processed
            
            result = self.runner.run(
                f"streaming_chunk_{chunk_size}",
                "streaming",
                stream_synthesize,
                metadata={
                    "chunk_size_phonemes": chunk_size,
                    "total_phonemes": len(phonemes),
                    "num_chunks": (len(phonemes) + chunk_size - 1) // chunk_size
                }
            )
            self.runner.print_result(result)
    
    def benchmark_first_chunk_latency(self):
        """Benchmark latency for first audio chunk."""
        print("\n" + "="*60)
        print("⚡ FIRST CHUNK LATENCY")
        print("="*60)
        
        acoustic_model = self.mock.create_mock_acoustic_model()
        vocoder = self.mock.create_mock_vocoder()
        
        texts = {
            "short": "Hello!",
            "medium": "Hello, this is a test.",
            "long": "Hello, this is a longer test sentence for first chunk latency measurement.",
        }
        
        for category, text in texts.items():
            phonemes = self.mock.text_to_phonemes(text)
            first_chunk = phonemes[:5]  # First 5 phonemes
            
            def first_chunk_synthesis():
                if TORCH_AVAILABLE:
                    with torch.no_grad():
                        mels = acoustic_model(first_chunk)
                        audio = vocoder(mels)
                        return audio
                else:
                    mels = acoustic_model.forward(first_chunk)
                    audio = vocoder.forward(mels)
                    return audio
            
            result = self.runner.run(
                f"first_chunk_{category}",
                "streaming",
                first_chunk_synthesis,
                metadata={
                    "text_category": category,
                    "first_chunk_size": len(first_chunk),
                    "total_phonemes": len(phonemes)
                }
            )
            self.runner.print_result(result)


class LanguageComparisonBenchmarks:
    """Benchmarks for language comparison."""
    
    def __init__(self, runner: BenchmarkRunner, mock_components: MockTTSComponents):
        self.runner = runner
        self.mock = mock_components
    
    def benchmark_language_synthesis(self, languages: List[str]):
        """Benchmark synthesis for different languages."""
        print("\n" + "="*60)
        print("🌍 LANGUAGE COMPARISON")
        print("="*60)
        
        acoustic_model = self.mock.create_mock_acoustic_model()
        vocoder = self.mock.create_mock_vocoder()
        
        # Sample texts for different languages
        texts = {
            "en": "Hello, this is a test.",
            "zh": "你好，这是一个测试。",
            "ja": "こんにちは、これはテストです。",
            "es": "Hola, esto es una prueba.",
            "de": "Hallo, das ist ein Test.",
            "fr": "Bonjour, c'est un test.",
        }
        
        for lang in languages:
            text = texts.get(lang, texts["en"])
            phonemes = self.mock.text_to_phonemes(text, language=lang)
            
            def synthesize():
                if TORCH_AVAILABLE:
                    with torch.no_grad():
                        mels = acoustic_model(phonemes)
                        audio = vocoder(mels)
                        return audio
                else:
                    mels = acoustic_model.forward(phonemes)
                    audio = vocoder.forward(mels)
                    return audio
            
            result = self.runner.run(
                f"language_{lang}",
                "language",
                synthesize,
                metadata={
                    "language": lang,
                    "text": text,
                    "text_length": len(text),
                    "num_phonemes": len(phonemes)
                }
            )
            self.runner.print_result(result)


class AudioFormatBenchmarks:
    """Benchmarks for audio format conversion."""
    
    def __init__(self, runner: BenchmarkRunner, mock_components: MockTTSComponents):
        self.runner = runner
        self.mock = mock_components
    
    def benchmark_format_conversion(self):
        """Benchmark audio format conversion time."""
        print("\n" + "="*60)
        print("🎵 AUDIO FORMAT CONVERSION")
        print("="*60)
        
        # Create mock audio
        sample_rate = 22050
        duration = 3
        audio_data = np.random.randn(sample_rate * duration).astype(np.float32)
        
        formats = [
            ("wav_pcm16", {"format": "wav", "subtype": "PCM_16"}),
            ("wav_pcm24", {"format": "wav", "subtype": "PCM_24"}),
            ("wav_float", {"format": "wav", "subtype": "FLOAT"}),
            ("mp3_128k", {"format": "mp3", "bitrate": 128}),
            ("mp3_192k", {"format": "mp3", "bitrate": 192}),
            ("mp3_320k", {"format": "mp3", "bitrate": 320}),
            ("ogg_vorbis", {"format": "ogg", "codec": "vorbis"}),
            ("flac", {"format": "flac"}),
        ]
        
        for format_name, options in formats:
            def convert_format():
                # Simulate format conversion
                if "mp3" in format_name:
                    time.sleep(0.005)  # MP3 encoding is slower
                elif "flac" in format_name:
                    time.sleep(0.003)  # FLAC is medium
                else:
                    time.sleep(0.001)  # WAV is fastest
                return True
            
            result = self.runner.run(
                f"format_conversion_{format_name}",
                "format",
                convert_format,
                metadata={
                    "format_name": format_name,
                    "options": options,
                    "duration": duration,
                    "sample_rate": sample_rate
                }
            )
            self.runner.print_result(result)
    
    def benchmark_sample_rate_conversion(self):
        """Benchmark sample rate conversion."""
        print("\n" + "="*60)
        print("📊 SAMPLE RATE CONVERSION")
        print("="*60)
        
        # Create mock audio at 22050 Hz
        audio_22k = np.random.randn(22050 * 3).astype(np.float32)
        
        conversions = [
            ("22k_to_16k", 22050, 16000),
            ("22k_to_44k", 22050, 44100),
            ("22k_to_48k", 22050, 48000),
            ("44k_to_22k", 44100, 22050),
            ("48k_to_22k", 48000, 22050),
        ]
        
        for name, src_rate, tgt_rate in conversions:
            def resample():
                # Simulate resampling
                time.sleep(0.002)
                return True
            
            result = self.runner.run(
                f"resample_{name}",
                "format",
                resample,
                metadata={
                    "src_sample_rate": src_rate,
                    "tgt_sample_rate": tgt_rate,
                    "ratio": tgt_rate / src_rate
                }
            )
            self.runner.print_result(result)


class MemoryBenchmarks:
    """Benchmarks for memory usage during TTS operations."""
    
    def __init__(self, runner: BenchmarkRunner, mock_components: MockTTSComponents):
        self.runner = runner
        self.mock = mock_components
    
    def benchmark_synthesis_memory(self):
        """Benchmark memory usage during synthesis."""
        print("\n" + "="*60)
        print("🧠 SYNTHESIS MEMORY USAGE")
        print("="*60)
        
        acoustic_model = self.mock.create_mock_acoustic_model()
        vocoder = self.mock.create_mock_vocoder()
        
        texts = {
            "short": "Hello!",
            "medium": "Hello, this is a test of text to speech synthesis.",
            "long": "Hello, this is a longer test of text to speech synthesis with more content. " * 5,
        }
        
        for category, text in texts.items():
            phonemes = self.mock.text_to_phonemes(text)
            
            def synthesize_with_memory():
                if TORCH_AVAILABLE:
                    with torch.no_grad():
                        mels = acoustic_model(phonemes)
                        audio = vocoder(mels)
                        return audio
                else:
                    mels = acoustic_model.forward(phonemes)
                    audio = vocoder.forward(mels)
                    return audio
            
            result = self.runner.run(
                f"memory_{category}",
                "memory",
                synthesize_with_memory,
                memory_profile=True,
                metadata={
                    "text_category": category,
                    "text_length": len(text),
                    "num_phonemes": len(phonemes)
                }
            )
            self.runner.print_result(result)


class RegressionBenchmarks:
    """Benchmarks for detecting performance regressions."""
    
    def __init__(self, runner: BenchmarkRunner, mock_components: MockTTSComponents):
        self.runner = runner
        self.mock = mock_components
    
    def run_regression_suite(self):
        """Run core regression benchmarks."""
        print("\n" + "="*60)
        print("🔄 REGRESSION BENCHMARK SUITE")
        print("="*60)
        
        acoustic_model = self.mock.create_mock_acoustic_model()
        vocoder = self.mock.create_mock_vocoder()
        
        # Core TTS operations
        baseline_ops = [
            ("synthesis_short", lambda: self._benchmark_synthesis(acoustic_model, vocoder, "Hello!")),
            ("synthesis_medium", lambda: self._benchmark_synthesis(acoustic_model, vocoder, "This is a medium test.")),
            ("phoneme_conversion", lambda: self.mock.text_to_phonemes("Hello world")),
            ("speaker_embedding", lambda: self.mock.generate_speaker_embedding()),
            ("cache_lookup", lambda: {"cached": True}.get("cached")),
        ]
        
        for name, func in baseline_ops:
            result = self.runner.run(
                f"regression_{name}",
                "regression",
                func,
                metadata={"baseline_op": name}
            )
            self.runner.print_result(result)
    
    def _benchmark_synthesis(self, acoustic_model, vocoder, text: str):
        """Helper for synthesis benchmark."""
        phonemes = self.mock.text_to_phonemes(text)
        if TORCH_AVAILABLE:
            with torch.no_grad():
                mels = acoustic_model(phonemes)
                return vocoder(mels)
        else:
            mels = acoustic_model.forward(phonemes)
            return vocoder.forward(mels)


def run_all_benchmarks(args) -> BenchmarkSuite:
    """Run all benchmark suites."""
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")
    suite = BenchmarkSuite(
        name="TTS Benchmark Suite",
        timestamp=timestamp,
        python_version=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        platform=os.name,
        torch_version=torch.__version__ if TORCH_AVAILABLE else None
    )
    
    # Setup
    iterations = int(os.environ.get("BENCHMARK_ITERATIONS", args.iterations))
    warmup = int(os.environ.get("BENCHMARK_WARMUP", args.warmup))
    runner = BenchmarkRunner(iterations=iterations, warmup=warmup)
    mock_components = MockTTSComponents()
    
    # Run benchmarks by category
    if args.category in ["all", "latency"]:
        latency_bench = SynthesisLatencyBenchmarks(runner, mock_components)
        latency_bench.benchmark_text_length_latency()
        latency_bench.benchmark_phoneme_complexity()
    
    if args.category in ["all", "voice_cloning"]:
        cloning_bench = VoiceCloningBenchmarks(runner, mock_components)
        cloning_bench.benchmark_voice_cloning_setup()
        cloning_bench.benchmark_synthesis_with_cloned_voice()
    
    if args.category in ["all", "cache"]:
        cache_bench = CachePerformanceBenchmarks(runner, mock_components)
        cache_bench.benchmark_memory_cache()
        cache_bench.benchmark_disk_cache()
    
    if args.category in ["all", "streaming"]:
        streaming_bench = StreamingBenchmarks(runner, mock_components)
        streaming_bench.benchmark_streaming_throughput()
        streaming_bench.benchmark_first_chunk_latency()
    
    if args.category in ["all", "language"]:
        lang_bench = LanguageComparisonBenchmarks(runner, mock_components)
        lang_bench.benchmark_language_synthesis(args.languages)
    
    if args.category in ["all", "format"]:
        format_bench = AudioFormatBenchmarks(runner, mock_components)
        format_bench.benchmark_format_conversion()
        format_bench.benchmark_sample_rate_conversion()
    
    if args.category in ["all", "memory"]:
        memory_bench = MemoryBenchmarks(runner, mock_components)
        memory_bench.benchmark_synthesis_memory()
    
    if args.category in ["all", "regression"]:
        regression_bench = RegressionBenchmarks(runner, mock_components)
        regression_bench.run_regression_suite()
    
    # Collect all results
    for result in runner.results:
        suite.add_result(result)
    
    # Cleanup
    shutil.rmtree(mock_components.disk_cache_dir, ignore_errors=True)
    
    return suite


def print_summary(suite: BenchmarkSuite):
    """Print a summary of all benchmark results."""
    print("\n" + "="*70)
    print("📊 BENCHMARK SUMMARY")
    print("="*70)
    
    # Group by category
    by_category = {}
    for result in suite.results:
        cat = result.category
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append(result)
    
    for category, results in sorted(by_category.items()):
        print(f"\n{category.upper()}:")
        print("-" * 70)
        for result in results:
            rtf_str = f" ({result.rtf:.2f}x RTF)" if result.rtf else ""
            print(f"  {result.name:50s} {result.mean_time*1000:>8.4f}ms{rtf_str}")
    
    print("\n" + "="*70)
    print(f"Total Benchmarks: {len(suite.results)}")
    print(f"Categories: {len(by_category)}")
    print("="*70)


def main():
    parser = argparse.ArgumentParser(
        description="TTS Benchmark Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python benchmarks/test_tts_benchmark.py --all
  python benchmarks/test_tts_benchmark.py --category latency
  python benchmarks/test_tts_benchmark.py --languages en zh ja
  python benchmarks/test_tts_benchmark.py --memory-profiling --output results.json
        """
    )
    
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all benchmarks"
    )
    parser.add_argument(
        "--category",
        choices=["all", "latency", "voice_cloning", "cache", "streaming", "language", "format", "memory", "regression"],
        default="all",
        help="Benchmark category to run"
    )
    parser.add_argument(
        "--languages",
        type=str,
        nargs="+",
        default=["en", "zh", "ja"],
        help="Languages to benchmark (default: en zh ja)"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="Number of iterations per benchmark (default: 100)"
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="Number of warmup iterations (default: 10)"
    )
    parser.add_argument(
        "--memory-profiling",
        action="store_true",
        help="Enable memory profiling"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output file for benchmark results (JSON)"
    )
    parser.add_argument(
        "--baseline",
        type=str,
        help="Compare against baseline results file"
    )
    
    args = parser.parse_args()
    
    if args.all:
        args.category = "all"
    
    print("="*70)
    print("🚀 TTS Benchmark Suite")
    print("="*70)
    print(f"PyTorch Available: {TORCH_AVAILABLE}")
    if TORCH_AVAILABLE:
        print(f"PyTorch Version: {torch.__version__}")
    print(f"Iterations: {args.iterations}")
    print(f"Warmup: {args.warmup}")
    print(f"Memory Profiling: {args.memory_profiling}")
    print(f"Category: {args.category}")
    print(f"Languages: {args.languages}")
    print("="*70)
    
    # Run benchmarks
    suite = run_all_benchmarks(args)
    
    # Print summary
    print_summary(suite)
    
    # Save results
    if args.output:
        suite.save(args.output)
    
    # Compare against baseline if provided
    if args.baseline and os.path.exists(args.baseline):
        print("\n" + "="*70)
        print("📊 BASELINE COMPARISON")
        print("="*70)
        with open(args.baseline, 'r') as f:
            baseline = json.load(f)
        
        baseline_results = {r['name']: r for r in baseline.get('results', [])}
        
        for result in suite.results:
            if result.name in baseline_results:
                baseline_time = baseline_results[result.name]['mean_time']
                current_time = result.mean_time
                delta = ((current_time - baseline_time) / baseline_time) * 100
                
                if abs(delta) > 5:  # >5% change
                    indicator = "🔴" if delta > 0 else "🟢"
                    print(f"{indicator} {result.name:50s} {delta:+.1f}% change")


if __name__ == "__main__":
    main()
