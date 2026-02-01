#!/usr/bin/env python3
"""
Benchmark Data Processing
Measures throughput of SchemaNormalizer and dataset loading.
"""

import time
import json
import random
from typing import List, Dict, Any
from pathlib import Path
from src.utils.schema_normalizer import SchemaNormalizer
from src.metrics_tracker import MetricsTracker, BenchmarkMetrics

def generate_mock_samples(n: int) -> List[Dict[str, Any]]:
    samples = []
    for i in range(n):
        samples.append({
            "id": f"sample_{i}",
            "instruction": f"Instruction {i} " * 10,
            "output": f"Output {i} " * 20,
            "image_path": f"path/to/image_{i}.png",
            "audio_path": f"path/to/audio_{i}.wav",
            "video_path": f"path/to/video_{i}.mp4",
        })
    return samples

def benchmark_normalization(n: int = 10000):
    print(f"Benchmarking SchemaNormalizer with {n} samples...")
    samples = generate_mock_samples(n)
    datasets = ["google_MusicCaps", "LDJnr_Pure-Dove", "LucasFang_JourneyDB-GoT", "Unknown"]
    
    tracker = MetricsTracker(output_dir="results")
    
    start_time = time.perf_counter()
    for sample in samples:
        ds = random.choice(datasets)
        SchemaNormalizer.normalize(sample, ds)
    end_time = time.perf_counter()
    
    duration = end_time - start_time
    throughput = n / duration
    
    print(f"Throughput: {throughput:.2f} samples/sec")
    print(f"Total time: {duration:.4f}s")
    
    metrics = BenchmarkMetrics(
        name="schema_normalization_throughput",
        category="data_processing",
        total_time_s=duration,
        tokens_per_second=throughput, # Using this field for throughput
        success=True
    )
    tracker.log_benchmark(metrics)
    return metrics

def main():
    benchmark_normalization(20000)

if __name__ == "__main__":
    main()
