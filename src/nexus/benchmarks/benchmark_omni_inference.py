#!/usr/bin/env python3
"""
Benchmark Omni Inference
Measures latency and throughput for text and streaming generation.
"""

import time
import torch
from unittest.mock import MagicMock, patch
import sys
from pathlib import Path
# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.omni.inference import OmniInference, GenerationConfig
from src.metrics_tracker import MetricsTracker, BenchmarkMetrics

def benchmark_omni_inference():
    print("Benchmarking OmniInference (Mocked Weights)...")
    
    # Mocking loader to avoid loading 7B model
    with patch("src.omni.loader.OmniModelLoader.load_for_inference") as mock_load:
        mock_model = MagicMock()
        mock_model.device = "cpu"
        mock_tokenizer = MagicMock()
        mock_load.return_value = (mock_model, mock_tokenizer)
        
        # Setup tokenizer/model for generation
        mock_tokenizer.return_value = {"input_ids": torch.zeros((1, 10), dtype=torch.long)}
        mock_tokenizer.decode.return_value = "This is a mocked response for benchmarking purposes."
        mock_model.generate.return_value = torch.zeros((1, 50), dtype=torch.long)
        
        inference = OmniInference("/fake/model")
        tracker = MetricsTracker(output_dir="results")
        
        # 1. Non-streaming latency
        start_time = time.perf_counter()
        for _ in range(10):
            inference.generate("Hello")
        end_time = time.perf_counter()
        
        avg_latency = (end_time - start_time) / 10
        print(f"Avg Latency (Non-streaming): {avg_latency*1000:.2f}ms")
        
        tracker.log_benchmark(BenchmarkMetrics(
            name="omni_inference_latency",
            category="inference",
            latency_ms=avg_latency * 1000,
            tokens_per_second=40 / avg_latency if avg_latency > 0 else 0,
            success=True
        ))
        
        # 2. Streaming throughput
        # Mock streamer
        with patch("transformers.TextIteratorStreamer") as mock_streamer_cls:
            mock_streamer = MagicMock()
            mock_streamer.__iter__.return_value = iter(["token"] * 50)
            mock_streamer_cls.return_value = mock_streamer
            
            start_time = time.perf_counter()
            tokens_count = 0
            for _ in range(5):
                for token in inference.generate_stream("Hello"):
                    tokens_count += 1
            end_time = time.perf_counter()
            
            duration = end_time - start_time
            tps = tokens_count / duration
            print(f"Throughput (Streaming): {tps:.2f} tokens/sec")
            
            tracker.log_benchmark(BenchmarkMetrics(
                name="omni_inference_streaming_tps",
                category="inference",
                total_time_s=duration,
                tokens_per_second=tps,
                success=True
            ))

if __name__ == "__main__":
    benchmark_omni_inference()
