#!/usr/bin/env python3
"""
Integration test for Context Length Systems.

Tests the Bookmark Indexation system with realistic context lengths
to verify actual context handling capabilities.

Run this to test YOUR hardware:
    python tests/integration/test_context_length.py --tokens 100000
"""

import argparse
import time
import gc
import sys
import psutil
from pathlib import Path
from typing import Tuple, Dict, Any

import torch

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.reasoning.bookmark_indexation import (
    BookmarkIndexation, BookmarkConfig, create_bookmark_indexation
)


def get_gpu_memory() -> Tuple[float, float]:
    """Get GPU memory usage (used, total) in GB."""
    if torch.cuda.is_available():
        used = torch.cuda.memory_allocated() / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        return used, total
    return 0.0, 0.0


def get_ram_usage() -> Tuple[float, float]:
    """Get RAM usage (used, total) in GB."""
    mem = psutil.virtual_memory()
    return mem.used / 1e9, mem.total / 1e9


def estimate_max_tokens(vram_gb: float, ram_gb: float, model_size_b: float = 7.0) -> Dict[str, int]:
    """
    Estimate maximum context length for different configurations.
    
    Based on typical memory requirements:
    - Model weights: ~2 bytes per param (bf16)
    - KV cache per token: ~1MB for 7B model (32 layers, 32 heads, 128 dim)
    """
    # Model memory
    model_mem_gb = model_size_b * 2  # 2 bytes per param
    
    # Available for KV
    kv_vram_gb = max(0, vram_gb - model_mem_gb - 2)  # Reserve 2GB
    kv_ram_gb = max(0, ram_gb - 8)  # Reserve 8GB for system
    
    # KV per token (approximate)
    kv_per_token_mb = 0.5  # ~0.5MB per token for 7B model
    
    vram_tokens = int((kv_vram_gb * 1024) / kv_per_token_mb)
    ram_tokens = int((kv_ram_gb * 1024) / kv_per_token_mb)
    
    return {
        "vram_only": vram_tokens,
        "vram_plus_ram": vram_tokens + ram_tokens,
        "with_disk": 10_000_000,  # Effectively unlimited
    }


def run_context_length_test(
    target_tokens: int,
    chunk_size: int = 1024,
    hidden_dim: int = 4096,
    num_heads: int = 32,
    head_dim: int = 128,
    use_gpu: bool = True,
    tmp_path: str = "/tmp/context_test",
) -> Dict[str, Any]:
    """
    Test actual context length handling.
    
    Returns metrics about the test.
    """
    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
    
    print(f"\n{'='*60}")
    print(f"Context Length Test: {target_tokens:,} tokens")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Chunk size: {chunk_size}")
    
    # Configure for testing
    config = BookmarkConfig(
        vram_capacity=min(32768, target_tokens // 2),  # Half in VRAM
        ram_capacity=min(131072, target_tokens),        # Rest in RAM
        block_size=64,
        bookmark_dim=256,
        top_k_retrieve=min(1024, target_tokens // 10),
        disk_cache_path=tmp_path,
    )
    
    system = BookmarkIndexation(config, hidden_dim=hidden_dim)
    if use_gpu and torch.cuda.is_available():
        system = system.to(device)
    
    # Track metrics
    start_time = time.time()
    vram_start, vram_total = get_gpu_memory()
    ram_start, ram_total = get_ram_usage()
    
    tokens_added = 0
    chunk_times = []
    
    print(f"\nAdding {target_tokens:,} tokens in chunks of {chunk_size}...")
    
    try:
        while tokens_added < target_tokens:
            chunk_start = time.time()
            
            # Generate chunk
            actual_chunk = min(chunk_size, target_tokens - tokens_added)
            
            hidden = torch.randn(1, actual_chunk, hidden_dim, device=device)
            keys = torch.randn(1, actual_chunk, num_heads, head_dim, device=device)
            values = torch.randn(1, actual_chunk, num_heads, head_dim, device=device)
            
            # Add to system
            system.add_tokens(hidden, keys, values)
            
            tokens_added += actual_chunk
            chunk_time = time.time() - chunk_start
            chunk_times.append(chunk_time)
            
            # Progress update
            if tokens_added % (chunk_size * 10) == 0 or tokens_added >= target_tokens:
                vram_now, _ = get_gpu_memory()
                ram_now, _ = get_ram_usage()
                stats = system.get_stats()
                
                print(f"  {tokens_added:,}/{target_tokens:,} tokens | "
                      f"VRAM: {vram_now:.1f}GB | RAM: {ram_now:.1f}GB | "
                      f"Blocks: V={stats['vram_blocks']} R={stats['ram_blocks']}")
            
            # Clear intermediate tensors
            del hidden, keys, values
            if use_gpu:
                torch.cuda.empty_cache()
    
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"\n⚠️  OOM at {tokens_added:,} tokens!")
            return {
                "success": False,
                "tokens_achieved": tokens_added,
                "error": "OOM",
            }
        raise
    
    total_time = time.time() - start_time
    vram_end, _ = get_gpu_memory()
    ram_end, _ = get_ram_usage()
    
    # Test retrieval
    print(f"\nTesting retrieval...")
    query = torch.randn(1, 128, hidden_dim, device=device)
    retrieve_start = time.time()
    k_ret, v_ret, positions = system.retrieve(query, top_k=1024)
    retrieve_time = time.time() - retrieve_start
    
    stats = system.get_stats()
    
    results = {
        "success": True,
        "target_tokens": target_tokens,
        "tokens_achieved": tokens_added,
        "total_time_sec": total_time,
        "tokens_per_sec": tokens_added / total_time,
        "avg_chunk_time_ms": sum(chunk_times) / len(chunk_times) * 1000,
        "retrieve_time_ms": retrieve_time * 1000,
        "positions_retrieved": len(positions) if positions else 0,
        "vram_used_gb": vram_end - vram_start,
        "ram_used_gb": ram_end - ram_start,
        "vram_blocks": stats["vram_blocks"],
        "ram_blocks": stats["ram_blocks"],
        "total_bookmarks": stats["total_bookmarks"],
    }
    
    print(f"\n{'='*60}")
    print(f"✅ Test Complete!")
    print(f"{'='*60}")
    print(f"Tokens processed: {tokens_added:,}")
    print(f"Total time: {total_time:.1f}s ({tokens_added/total_time:,.0f} tokens/sec)")
    print(f"VRAM used: {results['vram_used_gb']:.2f} GB")
    print(f"RAM used: {results['ram_used_gb']:.2f} GB")
    print(f"Retrieve time: {retrieve_time*1000:.1f}ms")
    print(f"Positions retrieved: {len(positions) if positions else 0}")
    
    return results


def run_hardware_benchmark():
    """Benchmark your specific hardware configuration."""
    print("\n" + "="*60)
    print("      HARDWARE BENCHMARK")
    print("="*60)
    
    # Get hardware info
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    else:
        gpu_name = "CPU only"
        vram_gb = 0
    
    ram_gb = psutil.virtual_memory().total / 1e9
    
    print(f"\nHardware Detected:")
    print(f"  GPU: {gpu_name}")
    print(f"  VRAM: {vram_gb:.1f} GB")
    print(f"  RAM: {ram_gb:.1f} GB")
    
    # Estimate capacities
    estimates = estimate_max_tokens(vram_gb, ram_gb)
    
    print(f"\nEstimated Maximum Context (7B model):")
    print(f"  VRAM only:     {estimates['vram_only']:>10,} tokens")
    print(f"  VRAM + RAM:    {estimates['vram_plus_ram']:>10,} tokens")
    print(f"  With Disk:     {estimates['with_disk']:>10,} tokens (unlimited)")
    
    # Run progressive tests
    print(f"\nRunning progressive context length tests...")
    
    test_sizes = [10_000, 50_000, 100_000]
    results = []
    
    for size in test_sizes:
        print(f"\n--- Testing {size:,} tokens ---")
        try:
            result = run_context_length_test(
                target_tokens=size,
                chunk_size=1024,
                hidden_dim=1024,  # Smaller for testing
                num_heads=8,
                head_dim=64,
                use_gpu=torch.cuda.is_available(),
            )
            results.append(result)
        except Exception as e:
            print(f"Error: {e}")
            break
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Summary
    print("\n" + "="*60)
    print("      BENCHMARK SUMMARY")
    print("="*60)
    
    for r in results:
        status = "✅" if r.get("success", False) else "❌"
        tokens = r.get("tokens_achieved", 0)
        speed = r.get("tokens_per_sec", 0)
        print(f"{status} {tokens:>10,} tokens @ {speed:,.0f} tokens/sec")
    
    max_achieved = max(r.get("tokens_achieved", 0) for r in results) if results else 0
    print(f"\nMaximum tested: {max_achieved:,} tokens")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Test actual context length handling")
    parser.add_argument("--tokens", type=int, default=100_000,
                        help="Number of tokens to test (default: 100000)")
    parser.add_argument("--chunk-size", type=int, default=1024,
                        help="Chunk size for processing (default: 1024)")
    parser.add_argument("--benchmark", action="store_true",
                        help="Run hardware benchmark")
    parser.add_argument("--cpu", action="store_true",
                        help="Force CPU mode (no GPU)")
    
    args = parser.parse_args()
    
    if args.benchmark:
        run_hardware_benchmark()
    else:
        run_context_length_test(
            target_tokens=args.tokens,
            chunk_size=args.chunk_size,
            use_gpu=not args.cpu,
        )


if __name__ == "__main__":
    main()
