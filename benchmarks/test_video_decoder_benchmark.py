#!/usr/bin/env python3
"""
Video Decoder Benchmark Suite

Comprehensive benchmarks for video generation performance:
- Video generation time (256x256, 512x512, 1024x1024)
- Frame generation rate (FPS achieved)
- Memory usage peak during generation
- Export format performance (MP4 vs GIF vs WebM)
- CPU vs GPU performance
- With/without VAE slicing/tiling

Usage:
    python benchmarks/test_video_decoder_benchmark.py --all
    python benchmarks/test_video_decoder_benchmark.py --category generation
    python benchmarks/test_video_decoder_benchmark.py --resolutions 256 512
    python benchmarks/test_video_decoder_benchmark.py --memory-profiling
    python benchmarks/test_video_decoder_benchmark.py --output results/video_benchmark.json

Environment Variables:
    BENCHMARK_ITERATIONS: Number of iterations (default: 50)
    BENCHMARK_WARMUP: Number of warmup iterations (default: 5)
    ENABLE_GPU_BENCHMARKS: Set to '1' to include GPU benchmarks
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
    fps: Optional[float] = None  # Frames per second
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
    
    def __init__(self, iterations: int = 50, warmup: int = 5):
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
        calculate_fps: bool = False,
        num_frames: int = 1,
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
            calculate_fps: Whether to calculate FPS
            num_frames: Number of frames for FPS calculation
            metadata: Additional metadata to store
            kwargs: Keyword arguments for func
        """
        times = []
        memory_delta = None
        
        # Warmup iterations
        for _ in range(self.warmup):
            result = func(*args, **kwargs)
            if TORCH_AVAILABLE and isinstance(result, (torch.Tensor, tuple)):
                torch.cuda.synchronize() if torch.cuda.is_available() else None
        
        # Actual benchmark iterations
        if memory_profile:
            gc.collect()
            tracemalloc.start()
            start_mem = tracemalloc.get_traced_memory()[0]
        
        for _ in range(self.iterations):
            start = time.perf_counter()
            result = func(*args, **kwargs)
            if TORCH_AVAILABLE and isinstance(result, (torch.Tensor, tuple)):
                torch.cuda.synchronize() if torch.cuda.is_available() else None
            end = time.perf_counter()
            times.append(end - start)
        
        if memory_profile:
            end_mem = tracemalloc.get_traced_memory()[0]
            tracemalloc.stop()
            memory_delta = (end_mem - start_mem) / (1024 * 1024)  # MB
        
        # Calculate statistics
        times.sort()
        mean_time = statistics.mean(times)
        fps = (num_frames / mean_time) if calculate_fps else None
        
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
            fps=fps,
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
        if result.fps is not None:
            print(f"FPS: {result.fps:.2f}")


class MockVideoComponents:
    """Mock implementations for video decoder benchmarking."""
    
    def __init__(self):
        self.vae_encoder = None
        self.vae_decoder = None
        self.unet = None
    
    def create_mock_vae(self, latent_dim: int = 4):
        """Create a mock VAE for video generation."""
        if TORCH_AVAILABLE:
            return MockVAETorch(latent_dim)
        else:
            return MockVAENumpy(latent_dim)
    
    def create_mock_unet(self, in_channels: int = 4, out_channels: int = 4):
        """Create a mock UNet for video generation."""
        if TORCH_AVAILABLE:
            return MockUNetTorch(in_channels, out_channels)
        else:
            return MockUNetNumpy(in_channels, out_channels)
    
    def generate_random_latents(self, batch_size: int, channels: int, frames: int, height: int, width: int):
        """Generate random latent tensors for video generation."""
        if TORCH_AVAILABLE:
            return torch.randn(batch_size, channels, frames, height, width)
        else:
            return np.random.randn(batch_size, channels, frames, height, width).astype(np.float32)


class MockVAETorch(nn.Module):
    """Mock VAE using PyTorch."""
    
    def __init__(self, latent_dim: int = 4):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv3d(64, latent_dim * 2, kernel_size=3, stride=2, padding=1),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(latent_dim, 64, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            nn.ConvTranspose3d(64, 3, kernel_size=4, stride=2, padding=1),
        )
        self.use_slicing = False
        self.use_tiling = False
    
    def encode(self, x):
        h = self.encoder(x)
        return h[:, :self.latent_dim]
    
    def decode(self, z):
        if self.use_slicing:
            return self._decode_with_slicing(z)
        elif self.use_tiling:
            return self._decode_with_tiling(z)
        else:
            return self.decoder(z)
    
    def _decode_with_slicing(self, z):
        """Decode with VAE slicing for memory efficiency."""
        # Slice along frame dimension
        slices = []
        frame_chunk = 4
        for i in range(0, z.shape[2], frame_chunk):
            z_slice = z[:, :, i:i+frame_chunk, :, :]
            slices.append(self.decoder(z_slice))
        return torch.cat(slices, dim=2)
    
    def _decode_with_tiling(self, z):
        """Decode with VAE tiling for memory efficiency."""
        # Tile along spatial dimensions
        tile_size = 64
        tiles = []
        for h in range(0, z.shape[3], tile_size):
            for w in range(0, z.shape[4], tile_size):
                z_tile = z[:, :, :, h:h+tile_size, w:w+tile_size]
                tiles.append(self.decoder(z_tile))
        # Simplified - in reality would need to stitch tiles
        return tiles[0] if tiles else self.decoder(z)


class MockVAENumpy:
    """Mock VAE using NumPy for environments without PyTorch."""
    
    def __init__(self, latent_dim: int = 4):
        self.latent_dim = latent_dim
        self.use_slicing = False
        self.use_tiling = False
    
    def encode(self, x):
        # Simplified encoding - just return smaller size
        return x[:, :, ::4, ::8, ::8] if len(x.shape) == 5 else x
    
    def decode(self, z):
        if self.use_slicing:
            return self._decode_with_slicing(z)
        elif self.use_tiling:
            return self._decode_with_tiling(z)
        else:
            # Simple upsampling
            return np.repeat(np.repeat(np.repeat(z, 4, axis=2), 8, axis=3), 8, axis=4)
    
    def _decode_with_slicing(self, z):
        slices = []
        frame_chunk = 4
        for i in range(0, z.shape[2], frame_chunk):
            z_slice = z[:, :, i:i+frame_chunk, :, :]
            slices.append(np.repeat(np.repeat(np.repeat(z_slice, 4, axis=2), 8, axis=3), 8, axis=4))
        return np.concatenate(slices, axis=2)
    
    def _decode_with_tiling(self, z):
        tile_size = 64
        tiles = []
        for h in range(0, z.shape[3], tile_size):
            for w in range(0, z.shape[4], tile_size):
                z_tile = z[:, :, :, h:h+tile_size, w:w+tile_size]
                tiles.append(np.repeat(np.repeat(np.repeat(z_tile, 4, axis=2), 8, axis=3), 8, axis=4))
        return tiles[0] if tiles else np.repeat(np.repeat(np.repeat(z, 4, axis=2), 8, axis=3), 8, axis=4)


class MockUNetTorch(nn.Module):
    """Mock UNet using PyTorch."""
    
    def __init__(self, in_channels: int = 4, out_channels: int = 4):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(128, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv3d(128, out_channels, kernel_size=3, padding=1)
    
    def forward(self, x, t=None):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.conv3(x)
        return x


class MockUNetNumpy:
    """Mock UNet using NumPy."""
    
    def __init__(self, in_channels: int = 4, out_channels: int = 4):
        self.in_channels = in_channels
        self.out_channels = out_channels
    
    def forward(self, x, t=None):
        # Simplified forward pass
        return x[:, :self.out_channels] if x.shape[1] > self.out_channels else x


class VideoGenerationBenchmarks:
    """Benchmarks for video generation performance."""
    
    def __init__(self, runner: BenchmarkRunner, mock_components: MockVideoComponents):
        self.runner = runner
        self.mock = mock_components
    
    def benchmark_generation_time(self, resolutions: List[int]):
        """Benchmark video generation time for different resolutions."""
        print("\n" + "="*60)
        print("🎬 VIDEO GENERATION TIME")
        print("="*60)
        
        num_frames = 16
        num_inference_steps = 50
        
        for resolution in resolutions:
            vae = self.mock.create_mock_vae()
            unet = self.mock.create_mock_unet()
            
            # Calculate latent dimensions
            latent_h = resolution // 8
            latent_w = resolution // 8
            latent_frames = num_frames // 4
            
            latents = self.mock.generate_random_latents(1, 4, latent_frames, latent_h, latent_w)
            
            if TORCH_AVAILABLE:
                vae = vae.to(torch.float32)
                unet = unet.to(torch.float32)
                latents = latents.to(torch.float32)
                if torch.cuda.is_available():
                    vae = vae.cuda()
                    unet = unet.cuda()
                    latents = latents.cuda()
            
            def generate_video():
                with torch.no_grad() if TORCH_AVAILABLE else contextmanager(lambda: (yield))():
                    # Simulate denoising steps
                    for _ in range(num_inference_steps):
                        if TORCH_AVAILABLE:
                            noise_pred = unet(latents)
                            latents = latents - 0.1 * noise_pred
                    # Decode to video
                    video = vae.decode(latents)
                    return video
            
            result = self.runner.run(
                f"generation_{resolution}x{resolution}_{num_frames}f",
                "generation",
                generate_video,
                calculate_fps=True,
                num_frames=num_frames,
                metadata={
                    "resolution": f"{resolution}x{resolution}",
                    "num_frames": num_frames,
                    "inference_steps": num_inference_steps,
                    "latent_h": latent_h,
                    "latent_w": latent_w
                }
            )
            self.runner.print_result(result)
    
    def benchmark_frame_generation_rate(self):
        """Benchmark frame generation rate (FPS) for different configurations."""
        print("\n" + "="*60)
        print("⚡ FRAME GENERATION RATE (FPS)")
        print("="*60)
        
        configs = [
            ("low_res_short", 256, 8),
            ("low_res_long", 256, 32),
            ("med_res_short", 512, 8),
            ("med_res_long", 512, 16),
            ("high_res_short", 1024, 4),
            ("high_res_long", 1024, 8),
        ]
        
        for name, resolution, num_frames in configs:
            vae = self.mock.create_mock_vae()
            unet = self.mock.create_mock_unet()
            
            latent_h = resolution // 8
            latent_w = resolution // 8
            latent_frames = num_frames // 4
            
            latents = self.mock.generate_random_latents(1, 4, latent_frames, latent_h, latent_w)
            
            if TORCH_AVAILABLE:
                latents = latents.to(torch.float32)
            
            def generate_frames():
                with torch.no_grad() if TORCH_AVAILABLE else contextmanager(lambda: (yield))():
                    for _ in range(20):  # 20 denoising steps
                        if TORCH_AVAILABLE:
                            noise_pred = unet(latents)
                            latents = latents - 0.1 * noise_pred
                    video = vae.decode(latents)
                    return video
            
            result = self.runner.run(
                f"fps_{name}",
                "fps",
                generate_frames,
                calculate_fps=True,
                num_frames=num_frames,
                metadata={
                    "config": name,
                    "resolution": resolution,
                    "num_frames": num_frames
                }
            )
            self.runner.print_result(result)


class MemoryBenchmarks:
    """Benchmarks for memory usage during video generation."""
    
    def __init__(self, runner: BenchmarkRunner, mock_components: MockVideoComponents):
        self.runner = runner
        self.mock = mock_components
    
    def benchmark_memory_peak(self):
        """Benchmark peak memory usage during generation."""
        print("\n" + "="*60)
        print("💾 MEMORY USAGE PEAK")
        print("="*60)
        
        configs = [
            ("256x256_16f", 256, 16),
            ("512x512_16f", 512, 16),
            ("1024x1024_16f", 1024, 16),
            ("512x512_32f", 512, 32),
            ("1024x1024_8f", 1024, 8),
        ]
        
        for name, resolution, num_frames in configs:
            vae = self.mock.create_mock_vae()
            unet = self.mock.create_mock_unet()
            
            latent_h = resolution // 8
            latent_w = resolution // 8
            latent_frames = num_frames // 4
            
            latents = self.mock.generate_random_latents(1, 4, latent_frames, latent_h, latent_w)
            
            if TORCH_AVAILABLE:
                latents = latents.to(torch.float32)
            
            def generate_with_memory():
                with torch.no_grad() if TORCH_AVAILABLE else contextmanager(lambda: (yield))():
                    for _ in range(20):
                        if TORCH_AVAILABLE:
                            noise_pred = unet(latents)
                            latents = latents - 0.1 * noise_pred
                    video = vae.decode(latents)
                    return video
            
            result = self.runner.run(
                f"memory_{name}",
                "memory",
                generate_with_memory,
                memory_profile=True,
                metadata={
                    "config": name,
                    "resolution": resolution,
                    "num_frames": num_frames
                }
            )
            self.runner.print_result(result)


class ExportFormatBenchmarks:
    """Benchmarks for export format performance."""
    
    def __init__(self, runner: BenchmarkRunner, mock_components: MockVideoComponents):
        self.runner = runner
        self.mock = mock_components
    
    def benchmark_export_formats(self):
        """Benchmark different export format performance."""
        print("\n" + "="*60)
        print("📤 EXPORT FORMAT PERFORMANCE")
        print("="*60)
        
        # Create mock video data
        resolution = 512
        num_frames = 16
        fps = 8
        
        formats = [
            ("mp4_h264", "mp4", {"codec": "h264", "quality": "high"}),
            ("mp4_hevc", "mp4", {"codec": "hevc", "quality": "high"}),
            ("webm_vp9", "webm", {"codec": "vp9", "quality": "high"}),
            ("gif", "gif", {"quality": "medium"}),
        ]
        
        for format_name, ext, options in formats:
            # Generate mock video
            if TORCH_AVAILABLE:
                video = torch.randn(1, 3, num_frames, resolution, resolution)
            else:
                video = np.random.randn(1, 3, num_frames, resolution, resolution).astype(np.float32)
            
            def export_video():
                # Simulate export process
                temp_path = f"/tmp/test_video.{ext}"
                # Mock encoding delay based on format complexity
                if "h264" in format_name:
                    time.sleep(0.001)  # Fast
                elif "hevc" in format_name:
                    time.sleep(0.002)  # Medium
                elif "vp9" in format_name:
                    time.sleep(0.003)  # Slower
                elif "gif" in format_name:
                    time.sleep(0.005)  # Slowest
                return temp_path
            
            result = self.runner.run(
                f"export_{format_name}",
                "export",
                export_video,
                metadata={
                    "format": format_name,
                    "extension": ext,
                    "options": options,
                    "resolution": resolution,
                    "num_frames": num_frames,
                    "fps": fps
                }
            )
            self.runner.print_result(result)
    
    def benchmark_export_quality_presets(self):
        """Benchmark export with different quality presets."""
        print("\n" + "="*60)
        print("🎨 EXPORT QUALITY PRESETS")
        print("="*60)
        
        resolution = 512
        num_frames = 16
        
        quality_presets = [
            ("low", {"crf": 28, "preset": "ultrafast"}),
            ("medium", {"crf": 23, "preset": "medium"}),
            ("high", {"crf": 18, "preset": "slow"}),
            ("lossless", {"crf": 0, "preset": "veryslow"}),
        ]
        
        for quality, settings in quality_presets:
            if TORCH_AVAILABLE:
                video = torch.randn(1, 3, num_frames, resolution, resolution)
            else:
                video = np.random.randn(1, 3, num_frames, resolution, resolution).astype(np.float32)
            
            def export_quality():
                # Simulate quality-based encoding
                if quality == "low":
                    time.sleep(0.0005)
                elif quality == "medium":
                    time.sleep(0.001)
                elif quality == "high":
                    time.sleep(0.003)
                else:  # lossless
                    time.sleep(0.01)
                return True
            
            result = self.runner.run(
                f"export_quality_{quality}",
                "export_quality",
                export_quality,
                metadata={
                    "quality": quality,
                    "settings": settings,
                    "resolution": resolution,
                    "num_frames": num_frames
                }
            )
            self.runner.print_result(result)


class HardwareComparisonBenchmarks:
    """Benchmarks comparing CPU vs GPU performance."""
    
    def __init__(self, runner: BenchmarkRunner, mock_components: MockVideoComponents):
        self.runner = runner
        self.mock = mock_components
    
    def benchmark_cpu_vs_gpu(self):
        """Compare CPU and GPU performance for video generation."""
        print("\n" + "="*60)
        print("🖥️  CPU vs GPU PERFORMANCE")
        print("="*60)
        
        if not TORCH_AVAILABLE:
            print("⚠️  PyTorch not available, skipping CPU/GPU benchmarks")
            return
        
        resolution = 256
        num_frames = 8
        latent_h = resolution // 8
        latent_w = resolution // 8
        latent_frames = num_frames // 4
        
        devices = [("cpu", torch.device("cpu"))]
        if torch.cuda.is_available():
            devices.append(("gpu", torch.device("cuda")))
        
        for device_name, device in devices:
            vae = self.mock.create_mock_vae().to(device)
            unet = self.mock.create_mock_unet().to(device)
            latents = self.mock.generate_random_latents(1, 4, latent_frames, latent_h, latent_w).to(device)
            
            def generate():
                with torch.no_grad():
                    for _ in range(10):
                        noise_pred = unet(latents)
                        latents = latents - 0.1 * noise_pred
                    video = vae.decode(latents)
                    return video
            
            result = self.runner.run(
                f"hardware_{device_name}_{resolution}x{resolution}",
                "hardware",
                generate,
                calculate_fps=True,
                num_frames=num_frames,
                metadata={
                    "device": device_name,
                    "resolution": resolution,
                    "num_frames": num_frames
                }
            )
            self.runner.print_result(result)


class VAEOptimizationBenchmarks:
    """Benchmarks for VAE slicing and tiling optimizations."""
    
    def __init__(self, runner: BenchmarkRunner, mock_components: MockVideoComponents):
        self.runner = runner
        self.mock = mock_components
    
    def benchmark_vae_slicing(self):
        """Benchmark VAE decoding with and without slicing."""
        print("\n" + "="*60)
        print("🔪 VAE SLICING OPTIMIZATION")
        print("="*60)
        
        resolutions = [256, 512, 1024]
        num_frames = 16
        
        for resolution in resolutions:
            latent_h = resolution // 8
            latent_w = resolution // 8
            latent_frames = num_frames // 4
            
            latents = self.mock.generate_random_latents(1, 4, latent_frames, latent_h, latent_w)
            
            # Without slicing
            vae_no_slice = self.mock.create_mock_vae()
            vae_no_slice.use_slicing = False
            vae_no_slice.use_tiling = False
            
            if TORCH_AVAILABLE:
                vae_no_slice = vae_no_slice.to(torch.float32)
                latents_copy = latents.clone().to(torch.float32)
            else:
                latents_copy = latents.copy()
            
            def decode_no_slice():
                with torch.no_grad() if TORCH_AVAILABLE else contextmanager(lambda: (yield))():
                    return vae_no_slice.decode(latents_copy)
            
            result = self.runner.run(
                f"vae_no_slicing_{resolution}x{resolution}",
                "vae_optimization",
                decode_no_slice,
                metadata={"resolution": resolution, "optimization": "none"}
            )
            self.runner.print_result(result)
            
            # With slicing
            vae_slice = self.mock.create_mock_vae()
            vae_slice.use_slicing = True
            vae_slice.use_tiling = False
            
            if TORCH_AVAILABLE:
                vae_slice = vae_slice.to(torch.float32)
                latents_copy2 = latents.clone().to(torch.float32)
            else:
                latents_copy2 = latents.copy()
            
            def decode_with_slice():
                with torch.no_grad() if TORCH_AVAILABLE else contextmanager(lambda: (yield))():
                    return vae_slice.decode(latents_copy2)
            
            result = self.runner.run(
                f"vae_slicing_{resolution}x{resolution}",
                "vae_optimization",
                decode_with_slice,
                metadata={"resolution": resolution, "optimization": "slicing"}
            )
            self.runner.print_result(result)
    
    def benchmark_vae_tiling(self):
        """Benchmark VAE decoding with and without tiling."""
        print("\n" + "="*60)
        print("🧱 VAE TILING OPTIMIZATION")
        print("="*60)
        
        resolutions = [512, 1024]
        num_frames = 8
        
        for resolution in resolutions:
            latent_h = resolution // 8
            latent_w = resolution // 8
            latent_frames = num_frames // 4
            
            latents = self.mock.generate_random_latents(1, 4, latent_frames, latent_h, latent_w)
            
            # Without tiling
            vae_no_tile = self.mock.create_mock_vae()
            vae_no_tile.use_slicing = False
            vae_no_tile.use_tiling = False
            
            if TORCH_AVAILABLE:
                vae_no_tile = vae_no_tile.to(torch.float32)
                latents_copy = latents.clone().to(torch.float32)
            else:
                latents_copy = latents.copy()
            
            def decode_no_tile():
                with torch.no_grad() if TORCH_AVAILABLE else contextmanager(lambda: (yield))():
                    return vae_no_tile.decode(latents_copy)
            
            result = self.runner.run(
                f"vae_no_tiling_{resolution}x{resolution}",
                "vae_optimization",
                decode_no_tile,
                metadata={"resolution": resolution, "optimization": "none"}
            )
            self.runner.print_result(result)
            
            # With tiling
            vae_tile = self.mock.create_mock_vae()
            vae_tile.use_slicing = False
            vae_tile.use_tiling = True
            
            if TORCH_AVAILABLE:
                vae_tile = vae_tile.to(torch.float32)
                latents_copy2 = latents.clone().to(torch.float32)
            else:
                latents_copy2 = latents.copy()
            
            def decode_with_tile():
                with torch.no_grad() if TORCH_AVAILABLE else contextmanager(lambda: (yield))():
                    return vae_tile.decode(latents_copy2)
            
            result = self.runner.run(
                f"vae_tiling_{resolution}x{resolution}",
                "vae_optimization",
                decode_with_tile,
                metadata={"resolution": resolution, "optimization": "tiling"}
            )
            self.runner.print_result(result)


class RegressionBenchmarks:
    """Benchmarks for detecting performance regressions."""
    
    def __init__(self, runner: BenchmarkRunner, mock_components: MockVideoComponents):
        self.runner = runner
        self.mock = mock_components
    
    def run_regression_suite(self):
        """Run core regression benchmarks."""
        print("\n" + "="*60)
        print("🔄 REGRESSION BENCHMARK SUITE")
        print("="*60)
        
        # Core video operations
        baseline_ops = [
            ("vae_decode_256", lambda: self._benchmark_vae_decode(256, 16)),
            ("vae_decode_512", lambda: self._benchmark_vae_decode(512, 16)),
            ("unet_forward_256", lambda: self._benchmark_unet(256)),
            ("export_mp4", lambda: self._benchmark_export("mp4")),
        ]
        
        for name, func in baseline_ops:
            result = self.runner.run(
                f"regression_{name}",
                "regression",
                func,
                metadata={"baseline_op": name}
            )
            self.runner.print_result(result)
    
    def _benchmark_vae_decode(self, resolution: int, num_frames: int):
        """Helper for VAE decode benchmark."""
        vae = self.mock.create_mock_vae()
        latent_h = resolution // 8
        latent_w = resolution // 8
        latent_frames = num_frames // 4
        latents = self.mock.generate_random_latents(1, 4, latent_frames, latent_h, latent_w)
        if TORCH_AVAILABLE:
            with torch.no_grad():
                return vae.decode(latents.to(torch.float32))
        return vae.decode(latents)
    
    def _benchmark_unet(self, resolution: int):
        """Helper for UNet forward benchmark."""
        unet = self.mock.create_mock_unet()
        latent_size = resolution // 8
        latents = self.mock.generate_random_latents(1, 4, 4, latent_size, latent_size)
        if TORCH_AVAILABLE:
            with torch.no_grad():
                return unet(latents.to(torch.float32))
        return unet.forward(latents)
    
    def _benchmark_export(self, format_type: str):
        """Helper for export benchmark."""
        time.sleep(0.001)  # Mock export delay
        return f"/tmp/test.{format_type}"


def run_all_benchmarks(args) -> BenchmarkSuite:
    """Run all benchmark suites."""
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")
    suite = BenchmarkSuite(
        name="Video Decoder Benchmark Suite",
        timestamp=timestamp,
        python_version=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        platform=os.name,
        torch_version=torch.__version__ if TORCH_AVAILABLE else None
    )
    
    # Setup
    iterations = int(os.environ.get("BENCHMARK_ITERATIONS", args.iterations))
    warmup = int(os.environ.get("BENCHMARK_WARMUP", args.warmup))
    runner = BenchmarkRunner(iterations=iterations, warmup=warmup)
    mock_components = MockVideoComponents()
    
    # Run benchmarks by category
    if args.category in ["all", "generation"]:
        gen_bench = VideoGenerationBenchmarks(runner, mock_components)
        gen_bench.benchmark_generation_time(args.resolutions)
        gen_bench.benchmark_frame_generation_rate()
    
    if args.category in ["all", "memory"]:
        mem_bench = MemoryBenchmarks(runner, mock_components)
        mem_bench.benchmark_memory_peak()
    
    if args.category in ["all", "export"]:
        export_bench = ExportFormatBenchmarks(runner, mock_components)
        export_bench.benchmark_export_formats()
        export_bench.benchmark_export_quality_presets()
    
    if args.category in ["all", "hardware"]:
        hw_bench = HardwareComparisonBenchmarks(runner, mock_components)
        hw_bench.benchmark_cpu_vs_gpu()
    
    if args.category in ["all", "vae_optimization"]:
        vae_bench = VAEOptimizationBenchmarks(runner, mock_components)
        vae_bench.benchmark_vae_slicing()
        vae_bench.benchmark_vae_tiling()
    
    if args.category in ["all", "regression"]:
        regression_bench = RegressionBenchmarks(runner, mock_components)
        regression_bench.run_regression_suite()
    
    # Collect all results
    for result in runner.results:
        suite.add_result(result)
    
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
            fps_str = f" ({result.fps:.2f} FPS)" if result.fps else ""
            print(f"  {result.name:50s} {result.mean_time*1000:>8.4f}ms{fps_str}")
    
    print("\n" + "="*70)
    print(f"Total Benchmarks: {len(suite.results)}")
    print(f"Categories: {len(by_category)}")
    print("="*70)


def main():
    parser = argparse.ArgumentParser(
        description="Video Decoder Benchmark Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python benchmarks/test_video_decoder_benchmark.py --all
  python benchmarks/test_video_decoder_benchmark.py --category generation
  python benchmarks/test_video_decoder_benchmark.py --resolutions 256 512
  python benchmarks/test_video_decoder_benchmark.py --memory-profiling --output results.json
        """
    )
    
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all benchmarks"
    )
    parser.add_argument(
        "--category",
        choices=["all", "generation", "memory", "export", "hardware", "vae_optimization", "regression"],
        default="all",
        help="Benchmark category to run"
    )
    parser.add_argument(
        "--resolutions",
        type=int,
        nargs="+",
        default=[256, 512, 1024],
        help="Video resolutions to benchmark (default: 256 512 1024)"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=50,
        help="Number of iterations per benchmark (default: 50)"
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="Number of warmup iterations (default: 5)"
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
    print("🚀 Video Decoder Benchmark Suite")
    print("="*70)
    print(f"PyTorch Available: {TORCH_AVAILABLE}")
    if TORCH_AVAILABLE:
        print(f"PyTorch Version: {torch.__version__}")
        print(f"CUDA Available: {torch.cuda.is_available()}")
    print(f"Iterations: {args.iterations}")
    print(f"Warmup: {args.warmup}")
    print(f"Memory Profiling: {args.memory_profiling}")
    print(f"Category: {args.category}")
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
