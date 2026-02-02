#!/usr/bin/env python3
"""
NVFP4 Benchmark Suite

Comprehensive benchmarks for NVFP4 quantization format comparing:
- Storage size vs BF16, INT8, FP32
- Quantization/dequantization speed
- I/O throughput improvement
- Memory usage
- Compression ratios

Usage:
    pytest benchmarks/test_nvfp4_benchmark.py -v
    pytest benchmarks/test_nvfp4_benchmark.py --benchmark-save=nvfp4_results
    pytest benchmarks/test_nvfp4_benchmark.py --benchmark-json=nvfp4_results.json
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import json
import time
import sys
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Any

# Ensure src is in path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from nexus.models.sli.nvfp4_loader import (
    NVFP4StreamingLoader,
    NVFP4Config,
    NVFP4Mode,
    NVFP4Quantizer,
    QuantizedTensor,
    quantize_to_nvfp4,
    dequantize_from_nvfp4,
)


@dataclass
class FormatComparisonResult:
    """Result of format comparison."""
    format_name: str
    size_bytes: int
    original_size_bytes: int
    compression_ratio: float
    mean_quantize_time_ms: float
    mean_dequantize_time_ms: float
    memory_usage_mb: float


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_tensors():
    """Create sample tensors of various sizes."""
    torch.manual_seed(42)
    return {
        "small": torch.randn(512, 512),
        "medium": torch.randn(2048, 4096),
        "large": torch.randn(8192, 8192),
        "attention": torch.randn(4096, 4096),
        "ffn": torch.randn(4096, 11008),
    }


@pytest.fixture
def sample_layers():
    """Create sample transformer layers."""
    return {
        "attention": nn.Linear(4096, 4096),
        "ffn": nn.Linear(4096, 11008),
        "projection": nn.Linear(4096, 4096),
        "gate": nn.Linear(4096, 4096),
    }


@pytest.fixture
def nvfp4_loader(tmp_path):
    """Create NVFP4 loader with temp cache."""
    config = NVFP4Config(mode=NVFP4Mode.SOFTWARE, block_size=16)
    return NVFP4StreamingLoader(config, cache_dir=str(tmp_path), device="cpu")


@pytest.fixture
def nvfp4_quantizer():
    """Create NVFP4 quantizer."""
    config = NVFP4Config(mode=NVFP4Mode.SOFTWARE, block_size=16)
    return NVFP4Quantizer(config)


# ============================================================================
# Storage Size Comparison Benchmarks
# ============================================================================

class TestStorageSizeComparison:
    """Benchmark storage size comparison across formats."""
    
    def test_fp32_storage_size(self, sample_tensors, benchmark):
        """Measure FP32 storage size."""
        def measure_fp32():
            total_bytes = 0
            for name, tensor in sample_tensors.items():
                fp32_tensor = tensor.to(torch.float32)
                total_bytes += fp32_tensor.numel() * 4  # 4 bytes per float32
            return total_bytes
        
        result = benchmark(measure_fp32)
        assert result > 0
    
    def test_bf16_storage_size(self, sample_tensors, benchmark):
        """Measure BF16 storage size."""
        def measure_bf16():
            total_bytes = 0
            for name, tensor in sample_tensors.items():
                bf16_tensor = tensor.to(torch.bfloat16)
                total_bytes += bf16_tensor.numel() * 2  # 2 bytes per bfloat16
            return total_bytes
        
        result = benchmark(measure_bf16)
        assert result > 0
    
    def test_int8_storage_size(self, sample_tensors, benchmark):
        """Measure INT8 storage size."""
        def measure_int8():
            total_bytes = 0
            for name, tensor in sample_tensors.items():
                # Simulate INT8 quantization
                int8_tensor = tensor.to(torch.int8)
                total_bytes += int8_tensor.numel() * 1  # 1 byte per int8
            return total_bytes
        
        result = benchmark(measure_int8)
        assert result > 0
    
    def test_nvfp4_storage_size(self, sample_tensors, nvfp4_quantizer, benchmark):
        """Measure NVFP4 storage size."""
        def measure_nvfp4():
            total_bytes = 0
            for name, tensor in sample_tensors.items():
                quantized = nvfp4_quantizer.quantize_tensor(tensor, name=name)
                # NVFP4 uses 0.5 bytes per element + scale overhead
                data_bytes = quantized.data.numel() * 1  # Stored as FP8 (1 byte)
                scale_bytes = quantized.scale.numel() * 4  # FP32 scales
                total_bytes += data_bytes + scale_bytes
            return total_bytes
        
        result = benchmark(measure_nvfp4)
        assert result > 0
    
    def test_storage_size_comparison_detailed(self, sample_tensors, nvfp4_quantizer):
        """Detailed storage size comparison with percentages."""
        results = {}
        
        for name, tensor in sample_tensors.items():
            original_size = tensor.numel() * 4  # FP32 baseline
            
            # FP32
            fp32_size = original_size
            
            # BF16
            bf16_size = tensor.numel() * 2
            
            # INT8
            int8_size = tensor.numel() * 1
            
            # NVFP4
            quantized = nvfp4_quantizer.quantize_tensor(tensor, name=name)
            nvfp4_size = quantized.data.numel() * 1 + quantized.scale.numel() * 4
            
            results[name] = {
                "shape": list(tensor.shape),
                "fp32_mb": fp32_size / (1024 * 1024),
                "bf16_mb": bf16_size / (1024 * 1024),
                "int8_mb": int8_size / (1024 * 1024),
                "nvfp4_mb": nvfp4_size / (1024 * 1024),
                "nvfp4_vs_fp32": (1 - nvfp4_size / fp32_size) * 100,
                "nvfp4_vs_bf16": (1 - nvfp4_size / bf16_size) * 100,
                "nvfp4_vs_int8": (nvfp4_size / int8_size - 1) * 100,
            }
        
        # Print comparison table
        print("\n" + "="*80)
        print("STORAGE SIZE COMPARISON")
        print("="*80)
        print(f"{'Tensor':<15} {'FP32 (MB)':<12} {'BF16 (MB)':<12} {'INT8 (MB)':<12} {'NVFP4 (MB)':<12} {'vs FP32':<10} {'vs BF16':<10}")
        print("-"*80)
        
        for name, data in results.items():
            print(f"{name:<15} {data['fp32_mb']:<12.2f} {data['bf16_mb']:<12.2f} "
                  f"{data['int8_mb']:<12.2f} {data['nvfp4_mb']:<12.2f} "
                  f"{data['nvfp4_vs_fp32']:<9.1f}% {data['nvfp4_vs_bf16']:<9.1f}%")
        
        print("="*80)
        
        # Save results
        output_path = Path("benchmarks/results")
        output_path.mkdir(exist_ok=True)
        with open(output_path / "nvfp4_storage_comparison.json", "w") as f:
            json.dump(results, f, indent=2)
        
        # Assert NVFP4 saves space vs FP32 and BF16
        for name, data in results.items():
            assert data['nvfp4_vs_fp32'] > 50, f"NVFP4 should save >50% vs FP32 for {name}"
            assert data['nvfp4_vs_bf16'] > 20, f"NVFP4 should save >20% vs BF16 for {name}"


# ============================================================================
# Quantization/Dequantization Speed Benchmarks
# ============================================================================

class TestQuantizationSpeed:
    """Benchmark quantization/dequantization speed."""
    
    def test_nvfp4_quantize_speed_small(self, sample_tensors, nvfp4_quantizer, benchmark):
        """Benchmark NVFP4 quantization speed for small tensors."""
        tensor = sample_tensors["small"]
        result = benchmark(nvfp4_quantizer.quantize_tensor, tensor, name="small")
        assert isinstance(result, QuantizedTensor)
    
    def test_nvfp4_quantize_speed_medium(self, sample_tensors, nvfp4_quantizer, benchmark):
        """Benchmark NVFP4 quantization speed for medium tensors."""
        tensor = sample_tensors["medium"]
        result = benchmark(nvfp4_quantizer.quantize_tensor, tensor, name="medium")
        assert isinstance(result, QuantizedTensor)
    
    def test_nvfp4_quantize_speed_large(self, sample_tensors, nvfp4_quantizer, benchmark):
        """Benchmark NVFP4 quantization speed for large tensors."""
        tensor = sample_tensors["large"]
        result = benchmark(nvfp4_quantizer.quantize_tensor, tensor, name="large")
        assert isinstance(result, QuantizedTensor)
    
    def test_nvfp4_dequantize_speed(self, sample_tensors, nvfp4_quantizer, benchmark):
        """Benchmark NVFP4 dequantization speed."""
        tensor = sample_tensors["medium"]
        quantized = nvfp4_quantizer.quantize_tensor(tensor, name="dequant_test")
        result = benchmark(nvfp4_quantizer.dequantize_tensor, quantized)
        assert isinstance(result, torch.Tensor)
    
    def test_fp32_to_bf16_speed(self, sample_tensors, benchmark):
        """Benchmark FP32 to BF16 conversion speed (baseline)."""
        tensor = sample_tensors["medium"]
        result = benchmark(tensor.to, torch.bfloat16)
        assert result.dtype == torch.bfloat16
    
    def test_quantization_speed_comparison(self, sample_tensors, nvfp4_quantizer):
        """Compare quantization speeds across formats."""
        tensor = sample_tensors["large"]
        iterations = 100
        
        # Time NVFP4 quantization
        nvfp4_times = []
        for _ in range(iterations):
            start = time.perf_counter()
            nvfp4_quantizer.quantize_tensor(tensor, name="speed_test")
            nvfp4_times.append(time.perf_counter() - start)
        nvfp4_mean = np.mean(nvfp4_times) * 1000  # ms
        
        # Time BF16 conversion (baseline)
        bf16_times = []
        for _ in range(iterations):
            start = time.perf_counter()
            tensor.to(torch.bfloat16)
            bf16_times.append(time.perf_counter() - start)
        bf16_mean = np.mean(bf16_times) * 1000  # ms
        
        # Time INT8 quantization
        int8_times = []
        for _ in range(iterations):
            start = time.perf_counter()
            # Simulate INT8 quantization
            _ = tensor.to(torch.int8)
            int8_times.append(time.perf_counter() - start)
        int8_mean = np.mean(int8_times) * 1000  # ms
        
        results = {
            "tensor_shape": list(tensor.shape),
            "tensor_elements": tensor.numel(),
            "nvfp4_quantize_ms": nvfp4_mean,
            "bf16_convert_ms": bf16_mean,
            "int8_convert_ms": int8_mean,
            "nvfp4_vs_bf16_ratio": nvfp4_mean / bf16_mean if bf16_mean > 0 else 0,
            "nvfp4_vs_int8_ratio": nvfp4_mean / int8_mean if int8_mean > 0 else 0,
        }
        
        print("\n" + "="*80)
        print("QUANTIZATION SPEED COMPARISON")
        print("="*80)
        print(f"Tensor shape: {tensor.shape}")
        print(f"Elements: {tensor.numel():,}")
        print(f"\n{'Format':<15} {'Mean Time (ms)':<20}")
        print("-"*40)
        print(f"{'NVFP4':<15} {nvfp4_mean:<20.4f}")
        print(f"{'BF16':<15} {bf16_mean:<20.4f}")
        print(f"{'INT8':<15} {int8_mean:<20.4f}")
        print("="*80)
        
        # Save results
        output_path = Path("benchmarks/results")
        output_path.mkdir(exist_ok=True)
        with open(output_path / "nvfp4_quantization_speed.json", "w") as f:
            json.dump(results, f, indent=2)


# ============================================================================
# I/O Throughput Benchmarks
# ============================================================================

class TestIOThroughput:
    """Benchmark I/O throughput improvements."""
    
    def test_nvfp4_io_throughput(self, sample_tensors, nvfp4_quantizer, benchmark):
        """Benchmark NVFP4 I/O throughput (quantized size / time)."""
        def io_simulation():
            total_bytes = 0
            start = time.perf_counter()
            
            for name, tensor in sample_tensors.items():
                quantized = nvfp4_quantizer.quantize_tensor(tensor, name=name)
                # Simulate I/O by copying data
                _ = quantized.data.clone()
                _ = quantized.scale.clone()
                total_bytes += quantized.data.numel() * 1 + quantized.scale.numel() * 4
            
            elapsed = time.perf_counter() - start
            throughput_mbps = (total_bytes / (1024 * 1024)) / elapsed if elapsed > 0 else 0
            return throughput_mbps
        
        result = benchmark(io_simulation)
        assert result > 0
    
    def test_fp32_io_throughput(self, sample_tensors, benchmark):
        """Benchmark FP32 I/O throughput (baseline)."""
        def io_simulation():
            total_bytes = 0
            start = time.perf_counter()
            
            for name, tensor in sample_tensors.items():
                fp32_tensor = tensor.to(torch.float32)
                _ = fp32_tensor.clone()
                total_bytes += fp32_tensor.numel() * 4
            
            elapsed = time.perf_counter() - start
            throughput_mbps = (total_bytes / (1024 * 1024)) / elapsed if elapsed > 0 else 0
            return throughput_mbps
        
        result = benchmark(io_simulation)
        assert result > 0
    
    def test_io_throughput_comparison(self, sample_tensors, nvfp4_quantizer):
        """Compare I/O throughput between formats."""
        iterations = 50
        
        # NVFP4 throughput
        nvfp4_throughputs = []
        for _ in range(iterations):
            total_bytes = 0
            start = time.perf_counter()
            
            for name, tensor in sample_tensors.items():
                quantized = nvfp4_quantizer.quantize_tensor(tensor, name=name)
                _ = quantized.data.clone()
                _ = quantized.scale.clone()
                total_bytes += quantized.data.numel() * 1 + quantized.scale.numel() * 4
            
            elapsed = time.perf_counter() - start
            throughput = (total_bytes / (1024 * 1024)) / elapsed if elapsed > 0 else 0
            nvfp4_throughputs.append(throughput)
        
        # FP32 throughput
        fp32_throughputs = []
        for _ in range(iterations):
            total_bytes = 0
            start = time.perf_counter()
            
            for name, tensor in sample_tensors.items():
                fp32_tensor = tensor.to(torch.float32)
                _ = fp32_tensor.clone()
                total_bytes += fp32_tensor.numel() * 4
            
            elapsed = time.perf_counter() - start
            throughput = (total_bytes / (1024 * 1024)) / elapsed if elapsed > 0 else 0
            fp32_throughputs.append(throughput)
        
        # BF16 throughput
        bf16_throughputs = []
        for _ in range(iterations):
            total_bytes = 0
            start = time.perf_counter()
            
            for name, tensor in sample_tensors.items():
                bf16_tensor = tensor.to(torch.bfloat16)
                _ = bf16_tensor.clone()
                total_bytes += bf16_tensor.numel() * 2
            
            elapsed = time.perf_counter() - start
            throughput = (total_bytes / (1024 * 1024)) / elapsed if elapsed > 0 else 0
            bf16_throughputs.append(throughput)
        
        results = {
            "nvfp4_mean_mbps": np.mean(nvfp4_throughputs),
            "nvfp4_std_mbps": np.std(nvfp4_throughputs),
            "fp32_mean_mbps": np.mean(fp32_throughputs),
            "fp32_std_mbps": np.std(fp32_throughputs),
            "bf16_mean_mbps": np.mean(bf16_throughputs),
            "bf16_std_mbps": np.std(bf16_throughputs),
            "improvement_vs_fp32": (np.mean(nvfp4_throughputs) / np.mean(fp32_throughputs) - 1) * 100,
            "improvement_vs_bf16": (np.mean(nvfp4_throughputs) / np.mean(bf16_throughputs) - 1) * 100,
        }
        
        print("\n" + "="*80)
        print("I/O THROUGHPUT COMPARISON")
        print("="*80)
        print(f"{'Format':<15} {'Mean (MB/s)':<15} {'Std (MB/s)':<15}")
        print("-"*50)
        print(f"{'NVFP4':<15} {results['nvfp4_mean_mbps']:<15.2f} {results['nvfp4_std_mbps']:<15.2f}")
        print(f"{'FP32':<15} {results['fp32_mean_mbps']:<15.2f} {results['fp32_std_mbps']:<15.2f}")
        print(f"{'BF16':<15} {results['bf16_mean_mbps']:<15.2f} {results['bf16_std_mbps']:<15.2f}")
        print("-"*50)
        print(f"NVFP4 vs FP32: +{results['improvement_vs_fp32']:.1f}%")
        print(f"NVFP4 vs BF16: +{results['improvement_vs_bf16']:.1f}%")
        print("="*80)
        
        # Save results
        output_path = Path("benchmarks/results")
        output_path.mkdir(exist_ok=True)
        with open(output_path / "nvfp4_io_throughput.json", "w") as f:
            json.dump(results, f, indent=2)
        
        # Assert NVFP4 has better throughput due to smaller size
        assert results['improvement_vs_fp32'] > 0, "NVFP4 should have higher throughput than FP32"


# ============================================================================
# Memory Usage Benchmarks
# ============================================================================

class TestMemoryUsage:
    """Benchmark memory usage across formats."""
    
    def test_nvfp4_memory_usage(self, sample_tensors, nvfp4_quantizer, benchmark):
        """Measure NVFP4 memory usage."""
        def measure_memory():
            total_bytes = 0
            for name, tensor in sample_tensors.items():
                quantized = nvfp4_quantizer.quantize_tensor(tensor, name=name)
                total_bytes += quantized.data.numel() * quantized.data.element_size()
                total_bytes += quantized.scale.numel() * quantized.scale.element_size()
            return total_bytes / (1024 * 1024)  # MB
        
        result = benchmark(measure_memory)
        assert result > 0
    
    def test_fp32_memory_usage(self, sample_tensors, benchmark):
        """Measure FP32 memory usage (baseline)."""
        def measure_memory():
            total_bytes = 0
            for name, tensor in sample_tensors.items():
                fp32_tensor = tensor.to(torch.float32)
                total_bytes += fp32_tensor.numel() * 4
            return total_bytes / (1024 * 1024)  # MB
        
        result = benchmark(measure_memory)
        assert result > 0
    
    def test_memory_usage_comparison(self, sample_tensors, nvfp4_quantizer):
        """Detailed memory usage comparison."""
        results = {}
        
        for name, tensor in sample_tensors.items():
            # FP32 memory
            fp32_mb = (tensor.numel() * 4) / (1024 * 1024)
            
            # BF16 memory
            bf16_mb = (tensor.numel() * 2) / (1024 * 1024)
            
            # INT8 memory
            int8_mb = (tensor.numel() * 1) / (1024 * 1024)
            
            # NVFP4 memory
            quantized = nvfp4_quantizer.quantize_tensor(tensor, name=name)
            nvfp4_mb = (quantized.data.numel() * 1 + quantized.scale.numel() * 4) / (1024 * 1024)
            
            results[name] = {
                "shape": list(tensor.shape),
                "fp32_mb": fp32_mb,
                "bf16_mb": bf16_mb,
                "int8_mb": int8_mb,
                "nvfp4_mb": nvfp4_mb,
                "savings_vs_fp32_mb": fp32_mb - nvfp4_mb,
                "savings_vs_bf16_mb": bf16_mb - nvfp4_mb,
                "savings_vs_fp32_pct": (1 - nvfp4_mb / fp32_mb) * 100,
                "savings_vs_bf16_pct": (1 - nvfp4_mb / bf16_mb) * 100,
            }
        
        # Calculate totals
        total_fp32 = sum(r["fp32_mb"] for r in results.values())
        total_bf16 = sum(r["bf16_mb"] for r in results.values())
        total_nvfp4 = sum(r["nvfp4_mb"] for r in results.values())
        
        print("\n" + "="*80)
        print("MEMORY USAGE COMPARISON")
        print("="*80)
        print(f"{'Tensor':<15} {'FP32 (MB)':<12} {'BF16 (MB)':<12} {'NVFP4 (MB)':<12} {'Saved vs FP32':<15} {'Saved vs BF16':<15}")
        print("-"*80)
        
        for name, data in results.items():
            print(f"{name:<15} {data['fp32_mb']:<12.2f} {data['bf16_mb']:<12.2f} "
                  f"{data['nvfp4_mb']:<12.2f} {data['savings_vs_fp32_mb']:<14.2f}MB {data['savings_vs_bf16_mb']:<14.2f}MB")
        
        print("-"*80)
        print(f"{'TOTAL':<15} {total_fp32:<12.2f} {total_bf16:<12.2f} {total_nvfp4:<12.2f} "
              f"{total_fp32 - total_nvfp4:<14.2f}MB {total_bf16 - total_nvfp4:<14.2f}MB")
        print(f"\nTotal Savings vs FP32: {(1 - total_nvfp4 / total_fp32) * 100:.1f}%")
        print(f"Total Savings vs BF16: {(1 - total_nvfp4 / total_bf16) * 100:.1f}%")
        print("="*80)
        
        # Save results
        output_path = Path("benchmarks/results")
        output_path.mkdir(exist_ok=True)
        with open(output_path / "nvfp4_memory_usage.json", "w") as f:
            json.dump({
                "per_tensor": results,
                "totals": {
                    "fp32_mb": total_fp32,
                    "bf16_mb": total_bf16,
                    "nvfp4_mb": total_nvfp4,
                    "savings_vs_fp32_pct": (1 - total_nvfp4 / total_fp32) * 100,
                    "savings_vs_bf16_pct": (1 - total_nvfp4 / total_bf16) * 100,
                }
            }, f, indent=2)
        
        # Assert significant memory savings
        assert (1 - total_nvfp4 / total_fp32) > 0.5, "NVFP4 should save >50% memory vs FP32"
        assert (1 - total_nvfp4 / total_bf16) > 0.2, "NVFP4 should save >20% memory vs BF16"


# ============================================================================
# Compression Ratio Benchmarks
# ============================================================================

class TestCompressionRatios:
    """Benchmark compression ratios."""
    
    def test_compression_ratios(self, sample_tensors, nvfp4_quantizer):
        """Calculate and verify compression ratios."""
        ratios = {}
        
        for name, tensor in sample_tensors.items():
            original_size = tensor.numel() * 4  # FP32 baseline
            quantized = nvfp4_quantizer.quantize_tensor(tensor, name=name)
            compressed_size = quantized.data.numel() * 1 + quantized.scale.numel() * 4
            
            ratios[name] = {
                "original_bytes": original_size,
                "compressed_bytes": compressed_size,
                "compression_ratio": original_size / compressed_size,
                "space_saved_pct": (1 - compressed_size / original_size) * 100,
            }
        
        print("\n" + "="*80)
        print("COMPRESSION RATIOS")
        print("="*80)
        print(f"{'Tensor':<15} {'Original (B)':<15} {'Compressed (B)':<17} {'Ratio':<10} {'Saved %':<10}")
        print("-"*70)
        
        for name, data in ratios.items():
            print(f"{name:<15} {data['original_bytes']:<15,} {data['compressed_bytes']:<17,} "
                  f"{data['compression_ratio']:<10.2f}x {data['space_saved_pct']:<9.1f}%")
        
        avg_ratio = np.mean([r["compression_ratio"] for r in ratios.values()])
        print("-"*70)
        print(f"Average compression ratio: {avg_ratio:.2f}x")
        print("="*80)
        
        # Save results
        output_path = Path("benchmarks/results")
        output_path.mkdir(exist_ok=True)
        with open(output_path / "nvfp4_compression_ratios.json", "w") as f:
            json.dump({
                "per_tensor": ratios,
                "average_ratio": avg_ratio,
            }, f, indent=2)
        
        # Assert good compression ratios
        assert avg_ratio > 2.0, f"Average compression ratio should be >2.0x, got {avg_ratio:.2f}x"
    
    def test_layer_compression(self, sample_layers, nvfp4_loader):
        """Test compression for actual transformer layers."""
        results = {}
        
        for name, layer in sample_layers.items():
            # Calculate original size
            original_size = sum(p.numel() * p.element_size() for p in layer.parameters())
            
            # Quantize layer
            is_attention = (name == "attention")
            quantized = nvfp4_loader.quantize_layer(layer, is_attention=is_attention, layer_name=name)
            
            # Calculate compressed size
            compressed_size = 0
            for buf_name, buf in quantized.named_buffers():
                compressed_size += buf.numel() * buf.element_size()
            
            results[name] = {
                "type": "attention" if is_attention else "ffn",
                "original_bytes": original_size,
                "compressed_bytes": compressed_size,
                "compression_ratio": original_size / compressed_size if compressed_size > 0 else 0,
            }
        
        print("\n" + "="*80)
        print("LAYER COMPRESSION RESULTS")
        print("="*80)
        print(f"{'Layer':<15} {'Type':<12} {'Original (B)':<15} {'Compressed (B)':<17} {'Ratio':<10}")
        print("-"*70)
        
        for name, data in results.items():
            print(f"{name:<15} {data['type']:<12} {data['original_bytes']:<15,} "
                  f"{data['compressed_bytes']:<17,} {data['compression_ratio']:<10.2f}x")
        
        print("="*80)
        
        # Save results
        output_path = Path("benchmarks/results")
        output_path.mkdir(exist_ok=True)
        with open(output_path / "nvfp4_layer_compression.json", "w") as f:
            json.dump(results, f, indent=2)


# ============================================================================
# End-to-End Benchmark
# ============================================================================

class TestEndToEndBenchmark:
    """End-to-end NVFP4 benchmark."""
    
    def test_full_pipeline_performance(self, sample_layers, nvfp4_loader):
        """Test full NVFP4 pipeline performance."""
        results = {
            "quantize_times": [],
            "dequantize_times": [],
            "cache_times": [],
            "load_times": [],
        }
        
        model_id = "test_model"
        
        for layer_idx, (name, layer) in enumerate(sample_layers.items()):
            # Time quantization
            start = time.perf_counter()
            quantized = nvfp4_loader.quantize_layer(layer, is_attention=(name == "attention"), layer_name=name)
            results["quantize_times"].append((time.perf_counter() - start) * 1000)
            
            # Time caching
            start = time.perf_counter()
            nvfp4_loader.cache_layer(model_id, layer_idx, quantized)
            results["cache_times"].append((time.perf_counter() - start) * 1000)
            
            # Time loading
            start = time.perf_counter()
            loaded = nvfp4_loader.load_layer(model_id, layer_idx)
            results["load_times"].append((time.perf_counter() - start) * 1000)
            
            # Time dequantization
            start = time.perf_counter()
            dequantized = nvfp4_loader.dequantize_layer(loaded)
            results["dequantize_times"].append((time.perf_counter() - start) * 1000)
        
        # Calculate statistics
        summary = {
            "quantize_mean_ms": np.mean(results["quantize_times"]),
            "dequantize_mean_ms": np.mean(results["dequantize_times"]),
            "cache_mean_ms": np.mean(results["cache_times"]),
            "load_mean_ms": np.mean(results["load_times"]),
            "total_layers": len(sample_layers),
        }
        
        print("\n" + "="*80)
        print("FULL PIPELINE PERFORMANCE")
        print("="*80)
        print(f"Operation          Mean Time (ms)")
        print("-"*40)
        print(f"Quantization       {summary['quantize_mean_ms']:<10.4f}")
        print(f"Dequantization     {summary['dequantize_mean_ms']:<10.4f}")
        print(f"Cache Write        {summary['cache_mean_ms']:<10.4f}")
        print(f"Cache Read         {summary['load_mean_ms']:<10.4f}")
        print("="*80)
        
        # Get loader stats
        stats = nvfp4_loader.get_stats()
        print(f"\nLoader Stats:")
        print(f"  Layers loaded: {stats['layers_loaded']}")
        print(f"  Layers quantized: {stats['layers_quantized']}")
        print(f"  Total load time: {stats['load_time_ms']:.2f} ms")
        print(f"  Total quantize time: {stats['quantize_time_ms']:.2f} ms")
        print("="*80)
        
        # Save results
        output_path = Path("benchmarks/results")
        output_path.mkdir(exist_ok=True)
        with open(output_path / "nvfp4_pipeline_performance.json", "w") as f:
            json.dump({
                "summary": summary,
                "raw_times": results,
                "loader_stats": stats,
            }, f, indent=2)
        
        # Assert reasonable performance
        assert summary['quantize_mean_ms'] < 100, "Quantization should be fast"
        assert summary['dequantize_mean_ms'] < 100, "Dequantization should be fast"


# ============================================================================
# JSON Report Generation
# ============================================================================

def pytest_sessionfinish(session, exitstatus):
    """Generate comprehensive JSON report after all tests."""
    try:
        output_path = Path("benchmarks/results")
        output_path.mkdir(exist_ok=True)
        
        # Collect all result files
        report = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "benchmark_type": "NVFP4 Performance Benchmarks",
        }
        
        # Load individual result files if they exist
        result_files = [
            "nvfp4_storage_comparison.json",
            "nvfp4_quantization_speed.json",
            "nvfp4_io_throughput.json",
            "nvfp4_memory_usage.json",
            "nvfp4_compression_ratios.json",
            "nvfp4_layer_compression.json",
            "nvfp4_pipeline_performance.json",
        ]
        
        for filename in result_files:
            filepath = output_path / filename
            if filepath.exists():
                with open(filepath) as f:
                    report[filename.replace(".json", "")] = json.load(f)
        
        # Save comprehensive report
        with open(output_path / "nvfp4_benchmark_report.json", "w") as f:
            json.dump(report, f, indent=2)
        
        print(f"\n📊 Comprehensive NVFP4 benchmark report saved to: {output_path / 'nvfp4_benchmark_report.json'}")
        
    except Exception as e:
        print(f"Warning: Could not generate final report: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
