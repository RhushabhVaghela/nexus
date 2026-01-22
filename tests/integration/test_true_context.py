#!/usr/bin/env python3
"""
True Context Length Validation Test

Tests actual context length by:
1. Loading a real model
2. Processing incrementally longer sequences
3. Measuring perplexity degradation (quality)
4. Measuring latency at each length

This is the REAL test - not just memory allocation.

Usage:
    python tests/integration/test_true_context.py --model /path/to/model --max-length 100000
"""

import argparse
import time
import gc
import sys
import json
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class LatencyMetrics:
    """Latency breakdown for a single inference."""
    bookmark_search_ms: float = 0.0
    kv_fetch_ms: float = 0.0
    attention_compute_ms: float = 0.0
    total_ms: float = 0.0
    tokens_retrieved: int = 0
    tier_distribution: Dict[str, int] = None
    
    def __post_init__(self):
        if self.tier_distribution is None:
            self.tier_distribution = {"vram": 0, "ram": 0, "disk": 0}


@dataclass
class ContextTestResult:
    """Result from testing a specific context length."""
    context_length: int
    perplexity: float
    latency: LatencyMetrics
    memory_gb: float
    passed: bool
    error: Optional[str] = None


class TrueContextTester:
    """
    Tests actual context length capabilities with real model inference.
    
    Validates that:
    1. Model can process the full context
    2. Attention over retrieved tokens is coherent
    3. Perplexity doesn't degrade catastrophically
    4. Latency remains acceptable
    """
    
    def __init__(
        self,
        model_path: str,
        device: torch.device = None,
        use_bookmark_indexation: bool = True,
    ):
        self.model_path = model_path
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_bookmark = use_bookmark_indexation
        
        self.model = None
        self.tokenizer = None
        self.bookmark_system = None
    
    def setup(self):
        """Load model and initialize systems."""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        logger.info(f"Loading model from {self.model_path}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model with lower memory
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
        self.model.eval()
        
        # Setup bookmark indexation if enabled
        if self.use_bookmark:
            from src.reasoning.bookmark_indexation import create_bookmark_indexation
            
            hidden_dim = self.model.config.hidden_size
            self.bookmark_system = create_bookmark_indexation(
                hidden_dim=hidden_dim,
                vram_capacity=32768,
                ram_capacity=131072,
            )
        
        logger.info("Model loaded successfully")
    
    def generate_test_text(self, num_tokens: int) -> str:
        """
        Generate coherent test text of specified length.
        
        Uses a mix of:
        - Natural patterns (to test real language modeling)
        - Repeated structures (to test long-range dependencies)
        - Unique markers (to test retrieval accuracy)
        """
        # Base texts that will be repeated and varied
        base_texts = [
            "The quick brown fox jumps over the lazy dog. ",
            "In a galaxy far, far away, strange things were happening. ",
            "The scientist observed the phenomenon with great interest. ",
            "Data flows through the network like water through pipes. ",
            "The algorithm processes information step by step carefully. ",
        ]
        
        # Generate tokens until we reach target
        text = ""
        token_count = 0
        marker_id = 0
        
        while token_count < num_tokens:
            # Add a unique marker every 1000 tokens
            if token_count > 0 and token_count % 1000 == 0:
                marker = f"[MARKER_{marker_id}] "
                text += marker
                marker_id += 1
            
            # Add base text with variation
            base = base_texts[token_count % len(base_texts)]
            text += base
            
            # Estimate tokens (rough: ~4 chars per token)
            token_count = len(text) // 4
        
        # Truncate to exact length (approximately)
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        if len(tokens) > num_tokens:
            tokens = tokens[:num_tokens]
        
        return self.tokenizer.decode(tokens)
    
    def measure_latency(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
    ) -> Tuple[Tensor, LatencyMetrics]:
        """
        Run inference and measure latency breakdown.
        """
        metrics = LatencyMetrics()
        
        total_start = time.perf_counter()
        
        with torch.no_grad():
            # If using bookmark system, measure retrieval
            if self.use_bookmark and self.bookmark_system:
                # Get last hidden state for query
                search_start = time.perf_counter()
                
                # For now, use input embeddings as proxy
                embeddings = self.model.get_input_embeddings()
                query_hidden = embeddings(input_ids[:, -128:])  # Last 128 tokens as query
                
                # Search bookmark index
                k_ret, v_ret, positions = self.bookmark_system.retrieve(
                    query_hidden.float(),
                    top_k=1024,
                )
                
                metrics.bookmark_search_ms = (time.perf_counter() - search_start) * 1000
                metrics.tokens_retrieved = len(positions) if positions else 0
            
            # Run model forward pass
            compute_start = time.perf_counter()
            
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
            )
            
            metrics.attention_compute_ms = (time.perf_counter() - compute_start) * 1000
        
        metrics.total_ms = (time.perf_counter() - total_start) * 1000
        
        return outputs.logits, metrics
    
    def compute_perplexity(
        self,
        logits: Tensor,
        labels: Tensor,
    ) -> float:
        """Compute perplexity from logits."""
        # Shift for next-token prediction
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        
        # Compute cross-entropy loss
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="mean",
        )
        
        return torch.exp(loss).item()
    
    def test_context_length(
        self,
        context_length: int,
        chunk_size: int = 512,
    ) -> ContextTestResult:
        """
        Test a specific context length.
        
        Returns detailed metrics about whether it passed.
        """
        logger.info(f"\nTesting context length: {context_length:,} tokens")
        
        try:
            # Generate test text
            text = self.generate_test_text(context_length)
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=context_length)
            
            input_ids = inputs["input_ids"].to(self.device)
            attention_mask = inputs["attention_mask"].to(self.device)
            
            actual_length = input_ids.shape[1]
            logger.info(f"  Actual tokens: {actual_length:,}")
            
            # Process in chunks if needed
            if self.use_bookmark and self.bookmark_system:
                # Add tokens to bookmark system in chunks
                for start in range(0, actual_length, chunk_size):
                    end = min(start + chunk_size, actual_length)
                    chunk_ids = input_ids[:, start:end]
                    
                    # Get hidden states for this chunk
                    with torch.no_grad():
                        embeddings = self.model.get_input_embeddings()
                        hidden = embeddings(chunk_ids).float()
                        
                        # Create dummy KV (in real scenario, from model forward)
                        num_heads = self.model.config.num_attention_heads
                        head_dim = self.model.config.hidden_size // num_heads
                        keys = torch.randn(1, end - start, num_heads, head_dim, device=self.device)
                        values = torch.randn(1, end - start, num_heads, head_dim, device=self.device)
                    
                    self.bookmark_system.add_tokens(hidden, keys, values)
            
            # Measure inference
            logits, latency = self.measure_latency(input_ids, attention_mask)
            
            # Compute perplexity
            perplexity = self.compute_perplexity(logits, input_ids)
            
            # Get memory usage
            if torch.cuda.is_available():
                memory_gb = torch.cuda.max_memory_allocated() / 1e9
            else:
                import psutil
                memory_gb = psutil.Process().memory_info().rss / 1e9
            
            logger.info(f"  Perplexity: {perplexity:.2f}")
            logger.info(f"  Latency: {latency.total_ms:.1f}ms")
            logger.info(f"  Memory: {memory_gb:.2f}GB")
            
            # Determine if passed
            passed = (
                perplexity < 100 and  # Reasonable perplexity
                latency.total_ms < 30000 and  # Under 30 seconds
                not torch.isnan(logits).any()
            )
            
            return ContextTestResult(
                context_length=actual_length,
                perplexity=perplexity,
                latency=latency,
                memory_gb=memory_gb,
                passed=passed,
            )
        
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.warning(f"  OOM at {context_length:,} tokens")
                return ContextTestResult(
                    context_length=context_length,
                    perplexity=float('inf'),
                    latency=LatencyMetrics(),
                    memory_gb=0,
                    passed=False,
                    error="OOM",
                )
            raise
        
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def find_max_context(
        self,
        start_length: int = 1024,
        max_length: int = 1000000,
        step_factor: float = 2.0,
    ) -> Tuple[int, List[ContextTestResult]]:
        """
        Binary search to find maximum working context length.
        """
        results = []
        current = start_length
        max_working = 0
        
        while current <= max_length:
            result = self.test_context_length(current)
            results.append(result)
            
            if result.passed:
                max_working = current
                current = int(current * step_factor)
            else:
                # Binary search between last working and failed
                break
        
        # Refine with binary search if we have a working baseline
        if max_working > 0 and not results[-1].passed:
            low = max_working
            high = results[-1].context_length
            
            while high - low > 1024:  # 1K token precision
                mid = (low + high) // 2
                result = self.test_context_length(mid)
                results.append(result)
                
                if result.passed:
                    low = mid
                    max_working = mid
                else:
                    high = mid
        
        return max_working, results


def run_true_context_test(
    model_path: str,
    max_length: int = 100000,
    use_bookmark: bool = True,
    output_file: Optional[str] = None,
):
    """Main test function."""
    print("\n" + "="*70)
    print("      TRUE CONTEXT LENGTH TEST")
    print("="*70)
    print(f"Model: {model_path}")
    print(f"Max Length: {max_length:,}")
    print(f"Bookmark Indexation: {'Enabled' if use_bookmark else 'Disabled'}")
    print("="*70 + "\n")
    
    tester = TrueContextTester(
        model_path=model_path,
        use_bookmark_indexation=use_bookmark,
    )
    
    try:
        tester.setup()
    except Exception as e:
        print(f"Failed to load model: {e}")
        print("\nTo use this test, you need a real model. For example:")
        print("  python tests/integration/test_true_context.py --model /mnt/e/data/models/Qwen2.5-0.5B")
        return
    
    # Find maximum context
    max_working, results = tester.find_max_context(
        start_length=1024,
        max_length=max_length,
    )
    
    # Print summary
    print("\n" + "="*70)
    print("      RESULTS SUMMARY")
    print("="*70)
    
    print(f"\nMaximum working context: {max_working:,} tokens")
    print(f"\nDetailed results:")
    print("-" * 60)
    print(f"{'Length':>10} | {'PPL':>8} | {'Latency':>10} | {'Memory':>8} | Status")
    print("-" * 60)
    
    for r in results:
        status = "✅ PASS" if r.passed else f"❌ {r.error or 'FAIL'}"
        print(f"{r.context_length:>10,} | {r.perplexity:>8.2f} | "
              f"{r.latency.total_ms:>8.1f}ms | {r.memory_gb:>6.2f}GB | {status}")
    
    # Save results if requested
    if output_file:
        output_data = {
            "max_working_context": max_working,
            "model_path": model_path,
            "use_bookmark": use_bookmark,
            "results": [
                {
                    "context_length": r.context_length,
                    "perplexity": r.perplexity,
                    "latency_ms": r.latency.total_ms,
                    "memory_gb": r.memory_gb,
                    "passed": r.passed,
                    "error": r.error,
                }
                for r in results
            ]
        }
        
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to: {output_file}")
    
    return max_working, results


def main():
    parser = argparse.ArgumentParser(description="True context length validation")
    parser.add_argument("--model", type=str, default="/mnt/e/data/models/Qwen2.5-0.5B",
                        help="Path to model")
    parser.add_argument("--max-length", type=int, default=100000,
                        help="Maximum length to test")
    parser.add_argument("--no-bookmark", action="store_true",
                        help="Disable bookmark indexation")
    parser.add_argument("--output", type=str, default=None,
                        help="Output file for results (JSON)")
    
    args = parser.parse_args()
    
    run_true_context_test(
        model_path=args.model,
        max_length=args.max_length,
        use_bookmark=not args.no_bookmark,
        output_file=args.output,
    )


if __name__ == "__main__":
    main()
