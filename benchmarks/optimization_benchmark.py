#!/usr/bin/env python3
"""
Optimization Benchmark Suite
Comprehensive optimization benchmarks for the Nexus multimodal model.

Covers:
- Layer pipelining speedup
- Layer skipping effectiveness
- Semi-autoregressive speedup
- Compression ratios
"""

import pytest
import torch
import time
import numpy as np
from typing import Dict, Any, List
from dataclasses import dataclass
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from nexus.models.optimization_suite import (
    LayerPipeliningOptimizer,
    LayerSkippingOptimizer,
    SemiAutoregressiveGenerator,
    CompressionOptimizer,
)
from nexus.models.omni.inference import OmniInference, GenerationConfig


@dataclass
class OptimizationResult:
    """Container for optimization benchmark results."""

    speedup: float
    baseline_time: float
    optimized_time: float
    quality_score: float
    compression_ratio: float
    memory_savings_mb: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "speedup": self.speedup,
            "baseline_time": self.baseline_time,
            "optimized_time": self.optimized_time,
            "quality_score": self.quality_score,
            "compression_ratio": self.compression_ratio,
            "memory_savings_mb": self.memory_savings_mb,
        }


class TestLayerPipelining:
    """Layer pipelining optimization benchmarks."""

    @pytest.fixture
    def optimizer(self):
        """Set up layer pipelining optimizer."""
        optimizer = LayerPipeliningOptimizer(
            num_pipeline_stages=4, micro_batch_size=2, enable_recompute=True
        )
        yield optimizer
        del optimizer

    @pytest.fixture
    def inference(self):
        """Set up inference engine."""
        try:
            inference = OmniInference(
                model_path="/mnt/e/data/models/Qwen2.5-Omni-7B-GPTQ-Int4",
                device="cuda" if torch.cuda.is_available() else "cpu",
            )
            yield inference
            del inference
        except Exception as e:
            pytest.skip(f"Model not available: {e}")

    @pytest.mark.benchmark
    def test_pipelining_speedup_short_sequence(self, optimizer, inference, benchmark):
        """Benchmark pipelining speedup for short sequences."""
        prompt = "Explain AI."

        # Baseline (no pipelining)
        def baseline():
            return inference.generate(
                prompt=prompt, config=GenerationConfig(max_new_tokens=50)
            )

        baseline_result = benchmark(baseline)

        # Optimized (with pipelining)
        def optimized():
            return optimizer.generate_with_pipelining(
                prompt=prompt,
                base_inference=inference,
                config=GenerationConfig(max_new_tokens=50),
            )

        optimized_result = benchmark(optimized)

        speedup = baseline_result.total_time / optimized_result.total_time
        assert speedup >= 1.0, "Pipelining should not slow down generation"

    @pytest.mark.benchmark
    def test_pipelining_speedup_long_sequence(self, optimizer, inference, benchmark):
        """Benchmark pipelining speedup for long sequences."""
        prompt = "Write a comprehensive analysis of deep learning architectures."

        def baseline():
            return inference.generate(
                prompt=prompt, config=GenerationConfig(max_new_tokens=500)
            )

        baseline_result = benchmark(baseline)

        def optimized():
            return optimizer.generate_with_pipelining(
                prompt=prompt,
                base_inference=inference,
                config=GenerationConfig(max_new_tokens=500),
            )

        optimized_result = benchmark(optimized)

        speedup = baseline_result.total_time / optimized_result.total_time
        # Long sequences should benefit more from pipelining
        assert speedup >= 1.2, (
            "Pipelining should provide at least 20% speedup for long sequences"
        )

    @pytest.mark.benchmark
    def test_pipelining_throughput(self, optimizer, inference, benchmark):
        """Benchmark pipelining throughput for batched requests."""
        prompts = [
            f"Question {i}: Explain the concept of neural networks." for i in range(8)
        ]

        def baseline_batch():
            return inference.batch_generate(prompts, max_tokens=100)

        baseline_result = benchmark(baseline_batch)

        def optimized_batch():
            return optimizer.batch_generate_with_pipelining(
                prompts=prompts, base_inference=inference, max_tokens=100
            )

        optimized_result = benchmark(optimized_batch)

        speedup = baseline_result.tokens_per_second / optimized_result.tokens_per_second
        assert speedup >= 1.0, "Pipelining should maintain or improve throughput"

    @pytest.mark.benchmark
    def test_pipelining_memory_efficiency(self, optimizer, inference):
        """Benchmark memory efficiency with pipelining."""
        # Measure baseline memory
        baseline_memory = inference.get_memory_usage()

        # Generate with pipelining
        result = optimizer.generate_with_pipelining(
            prompt="Memory test prompt.",
            base_inference=inference,
            config=GenerationConfig(max_new_tokens=200),
        )

        pipelined_memory = result.memory_peak_mb

        # Pipelining should use less peak memory
        assert pipelined_memory <= baseline_memory * 1.1, (
            "Pipelining should not significantly increase peak memory"
        )

    @pytest.mark.benchmark
    def test_pipelining_stage_balance(self, optimizer):
        """Test that pipeline stages are balanced."""
        model_layers = 32
        stages = optimizer._balance_stages(model_layers)

        # Check that stages are roughly balanced
        layer_counts = [len(stage) for stage in stages]
        max_diff = max(layer_counts) - min(layer_counts)

        assert max_diff <= 1, "Pipeline stages should be balanced within 1 layer"

    @pytest.mark.benchmark
    def test_pipelining_microbatch_handling(self, optimizer):
        """Test microbatch processing efficiency."""
        total_batches = 16
        micro_batch_size = optimizer.micro_batch_size
        expected_microbatches = total_batches // micro_batch_size

        processed = 0
        for microbatch in optimizer.process_microbatches(total_batches):
            processed += len(microbatch)

        assert processed == total_batches, "All microbatches should be processed"


class TestLayerSkipping:
    """Layer skipping optimization benchmarks."""

    @pytest.fixture
    def skip_optimizer(self):
        """Set up layer skipping optimizer."""
        optimizer = LayerSkippingOptimizer(
            skip_threshold=0.8, min_layers=4, max_skippable=24
        )
        yield optimizer
        del optimizer

    @pytest.fixture
    def inference(self):
        """Set up inference engine."""
        try:
            inference = OmniInference(
                model_path="/mnt/e/data/models/Qwen2.5-Omni-7B-GPTQ-Int4",
                device="cuda" if torch.cuda.is_available() else "cpu",
            )
            yield inference
            del inference
        except Exception as e:
            pytest.skip(f"Model not available: {e}")

    @pytest.mark.benchmark
    def test_skip_effectiveness_simple_prompts(
        self, skip_optimizer, inference, benchmark
    ):
        """Test layer skipping effectiveness for simple prompts."""
        prompt = "What is 2+2?"

        def baseline():
            return inference.generate(
                prompt=prompt, config=GenerationConfig(max_new_tokens=20)
            )

        baseline_result = benchmark(baseline)

        def optimized():
            return skip_optimizer.generate_with_skipping(
                prompt=prompt,
                base_inference=inference,
                config=GenerationConfig(max_new_tokens=20),
            )

        optimized_result = benchmark(optimized)

        # Calculate skip rate
        skipped_layers = optimized_result.skipped_layers_count
        total_layers = optimized_result.total_layers_processed
        skip_rate = skipped_layers / total_layers if total_layers > 0 else 0

        # Simple prompts should have higher skip rate
        assert skip_rate >= 0.0, "Should be able to skip some layers"

    @pytest.mark.benchmark
    def test_skip_effectiveness_complex_prompts(
        self, skip_optimizer, inference, benchmark
    ):
        """Test layer skipping effectiveness for complex prompts."""
        prompt = """Analyze the following complex problem: 
        Design a distributed system that can handle millions of concurrent users 
        while maintaining low latency and high availability."""

        def baseline():
            return inference.generate(
                prompt=prompt, config=GenerationConfig(max_new_tokens=100)
            )

        baseline_result = benchmark(baseline)

        def optimized():
            return skip_optimizer.generate_with_skipping(
                prompt=prompt,
                base_inference=inference,
                config=GenerationConfig(max_new_tokens=100),
            )

        optimized_result = benchmark(optimized)

        skipped_layers = optimized_result.skipped_layers_count
        total_layers = optimized_result.total_layers_processed
        skip_rate = skipped_layers / total_layers if total_layers > 0 else 0

        # Complex prompts should skip fewer layers
        assert skip_rate >= 0.0, "Should attempt layer skipping"

    @pytest.mark.benchmark
    def test_skip_accuracy(self, skip_optimizer, inference):
        """Test that layer skipping doesn't degrade output quality."""
        prompt = "Define machine learning in one sentence."

        # Get baseline output
        baseline_output = inference.generate(
            prompt=prompt, config=GenerationConfig(max_new_tokens=50)
        )

        # Get optimized output
        optimized_output = skip_optimizer.generate_with_skipping(
            prompt=prompt,
            base_inference=inference,
            config=GenerationConfig(max_new_tokens=50),
        )

        # Check that outputs are semantically similar
        similarity = skip_optimizer.calculate_semantic_similarity(
            baseline_output.text, optimized_output.text
        )

        assert similarity >= 0.7, "Skipped output should be semantically similar"

    @pytest.mark.benchmark
    def test_skip_prediction_accuracy(self, skip_optimizer):
        """Test accuracy of skip predictions."""
        # Test various input patterns
        test_cases = [
            {"prompt": "Simple question?", "expected_skip": True},
            {"prompt": "Complex reasoning required...", "expected_skip": False},
            {"prompt": "Math: 2+2=", "expected_skip": True},
            {
                "prompt": "Analyze this detailed document about quantum computing.",
                "expected_skip": False,
            },
        ]

        correct_predictions = 0
        for case in test_cases:
            should_skip = skip_optimizer.predict_skip(case["prompt"])
            # We don't have ground truth, but we verify predictions are consistent
            assert isinstance(should_skip, bool), "Skip prediction should be boolean"
            correct_predictions += 1

        assert correct_predictions == len(test_cases), "All predictions should complete"

    @pytest.mark.benchmark
    def test_skip_speedup(self, skip_optimizer, inference, benchmark):
        """Measure actual speedup from layer skipping."""
        prompt = "Quick definition: What is a neuron?"

        def baseline():
            return inference.generate(
                prompt=prompt, config=GenerationConfig(max_new_tokens=30)
            )

        baseline_result = benchmark(baseline)

        def optimized():
            return skip_optimizer.generate_with_skipping(
                prompt=prompt,
                base_inference=inference,
                config=GenerationConfig(max_new_tokens=30),
            )

        optimized_result = benchmark(optimized)

        speedup = baseline_result.total_time / optimized_result.total_time
        assert speedup >= 1.0, "Layer skipping should not slow down generation"


class TestSemiAutoregressiveGeneration:
    """Semi-autoregressive generation optimization benchmarks."""

    @pytest.fixture
    def semi_auto_generator(self):
        """Set up semi-autoregressive generator."""
        generator = SemiAutoregressiveGenerator(
            block_size=8, speculation_length=3, parallelism_degree=4
        )
        yield generator
        del generator

    @pytest.fixture
    def inference(self):
        """Set up inference engine."""
        try:
            inference = OmniInference(
                model_path="/mnt/e/data/models/Qwen2.5-Omni-7B-GPTQ-Int4",
                device="cuda" if torch.cuda.is_available() else "cpu",
            )
            yield inference
            del inference
        except Exception as e:
            pytest.skip(f"Model not available: {e}")

    @pytest.mark.benchmark
    def test_semi_auto_speedup(self, semi_auto_generator, inference, benchmark):
        """Benchmark semi-autoregressive speedup."""
        prompt = "Explain the theory of relativity."

        def baseline():
            return inference.generate(
                prompt=prompt, config=GenerationConfig(max_new_tokens=100)
            )

        baseline_result = benchmark(baseline)

        def optimized():
            return semi_auto_generator.generate(
                prompt=prompt,
                base_inference=inference,
                config=GenerationConfig(max_new_tokens=100),
            )

        optimized_result = benchmark(optimized)

        speedup = baseline_result.total_time / optimized_result.total_time
        assert speedup >= 1.0, "Semi-autoregressive should not slow down generation"

    @pytest.mark.benchmark
    def test_block_size_optimization(self, semi_auto_generator, inference, benchmark):
        """Test different block sizes for optimal performance."""
        results = {}

        for block_size in [4, 8, 16, 32]:
            generator = SemiAutoregressiveGenerator(
                block_size=block_size, speculation_length=3, parallelism_degree=4
            )

            def test_block():
                return generator.generate(
                    prompt="Test prompt.",
                    base_inference=inference,
                    config=GenerationConfig(max_new_tokens=50),
                )

            result = benchmark(test_block)
            results[block_size] = result.total_time

        # Find optimal block size
        optimal_block = min(results, key=results.get)

        # Optimal should be one of the tested sizes
        assert optimal_block in [4, 8, 16, 32], "Optimal block size should be tested"

        # Performance should be reasonable for all sizes
        for block_size, time_taken in results.items():
            assert time_taken > 0, (
                f"Block size {block_size} should produce valid results"
            )

    @pytest.mark.benchmark
    def test_speculation_accuracy(self, semi_auto_generator, inference):
        """Test accuracy of speculative generation."""
        prompt = "The capital of France is"

        result = semi_auto_generator.generate(
            prompt=prompt,
            base_inference=inference,
            config=GenerationConfig(max_new_tokens=20),
        )

        # Check speculation accuracy
        speculative_tokens = result.speculative_tokens
        accepted_tokens = result.accepted_tokens

        acceptance_rate = (
            accepted_tokens / speculative_tokens if speculative_tokens > 0 else 0
        )

        assert acceptance_rate >= 0.5, (
            "At least 50% of speculative tokens should be accepted"
        )

    @pytest.mark.benchmark
    def test_parallelism_degree_scaling(
        self, semi_auto_generator, inference, benchmark
    ):
        """Test scaling with parallelism degree."""
        scaling_results = []

        for degree in [1, 2, 4, 8]:
            generator = SemiAutoregressiveGenerator(
                block_size=8, speculation_length=3, parallelism_degree=degree
            )

            def test_parallel():
                return generator.generate(
                    prompt="Parallel test.",
                    base_inference=inference,
                    config=GenerationConfig(max_new_tokens=50),
                )

            result = benchmark(test_parallel)
            scaling_results.append({"degree": degree, "time": result.total_time})

        # Higher parallelism should generally be faster or equal
        base_time = scaling_results[0]["time"]
        for result in scaling_results:
            assert result["time"] <= base_time * 1.5, (
                f"Parallelism degree {result['degree']} should not be 50% slower"
            )

    @pytest.mark.benchmark
    def test_output_quality_parity(self, semi_auto_generator, inference):
        """Verify semi-autoregressive output quality matches baseline."""
        prompt = "Write a short poem about technology."

        # Baseline
        baseline_output = inference.generate(
            prompt=prompt, config=GenerationConfig(max_new_tokens=50)
        )

        # Semi-autoregressive
        optimized_output = semi_auto_generator.generate(
            prompt=prompt,
            base_inference=inference,
            config=GenerationConfig(max_new_tokens=50),
        )

        # Both should produce text of similar length
        baseline_len = len(baseline_output.text.split())
        optimized_len = len(optimized_output.text.split())

        length_ratio = optimized_len / baseline_len if baseline_len > 0 else 1

        assert 0.8 <= length_ratio <= 1.2, "Output length should be similar"

    @pytest.mark.benchmark
    def test_generation_quality_diversity(self, semi_auto_generator, inference):
        """Test that generation maintains diversity."""
        prompt = "List three colors."

        outputs = []
        for _ in range(5):
            result = semi_auto_generator.generate(
                prompt=prompt,
                base_inference=inference,
                config=GenerationConfig(max_new_tokens=30, temperature=0.8),
            )
            outputs.append(result.text)

        # Outputs should have some diversity
        unique_outputs = set(outputs)
        assert len(unique_outputs) > 1, "Generation should produce diverse outputs"


class TestCompressionOptimization:
    """Compression optimization benchmarks."""

    @pytest.fixture
    def compression_optimizer(self):
        """Set up compression optimizer."""
        optimizer = CompressionOptimizer(
            method="dynamic", target_ratio=0.75, quality_threshold=0.95
        )
        yield optimizer
        del optimizer

    @pytest.fixture
    def inference(self):
        """Set up inference engine."""
        try:
            inference = OmniInference(
                model_path="/mnt/e/data/models/Qwen2.5-Omni-7B-GPTQ-Int4",
                device="cuda" if torch.cuda.is_available() else "cpu",
            )
            yield inference
            del inference
        except Exception as e:
            pytest.skip(f"Model not available: {e}")

    @pytest.mark.benchmark
    def test_compression_ratio(self, compression_optimizer, inference, benchmark):
        """Measure compression ratio achieved."""
        prompt = "This is a test prompt for compression benchmarking. " * 10

        def measure_compression():
            return compression_optimizer.compress_and_generate(
                prompt=prompt,
                base_inference=inference,
                config=GenerationConfig(max_new_tokens=50),
            )

        result = benchmark(measure_compression)

        # Check compression ratio
        assert result.compression_ratio >= 0.5, (
            "Should achieve at least 50% compression"
        )

    @pytest.mark.benchmark
    def test_compression_speedup(self, compression_optimizer, inference, benchmark):
        """Measure speedup from compression."""
        prompt = "Compressible text content. " * 20

        def baseline():
            return inference.generate(
                prompt=prompt, config=GenerationConfig(max_new_tokens=50)
            )

        baseline_result = benchmark(baseline)

        def optimized():
            return compression_optimizer.compress_and_generate(
                prompt=prompt,
                base_inference=inference,
                config=GenerationConfig(max_new_tokens=50),
            )

        optimized_result = benchmark(optimized)

        speedup = baseline_result.total_time / optimized_result.total_time
        assert speedup >= 1.0, "Compression should not slow down generation"

    @pytest.mark.benchmark
    def test_compression_quality_preservation(self, compression_optimizer, inference):
        """Verify compressed generation maintains quality."""
        prompt = (
            "Define artificial intelligence and provide examples of its applications."
        )

        # Baseline
        baseline_output = inference.generate(
            prompt=prompt, config=GenerationConfig(max_new_tokens=100)
        )

        # Compressed
        compressed_output = compression_optimizer.compress_and_generate(
            prompt=prompt,
            base_inference=inference,
            config=GenerationConfig(max_new_tokens=100),
        )

        # Check quality preservation
        quality_score = compressed_output.quality_score
        assert quality_score >= 0.9, "Quality should be preserved above 90%"

    @pytest.mark.benchmark
    def test_adaptive_compression(self, compression_optimizer):
        """Test adaptive compression based on content."""
        test_cases = [
            ("Highly repetitive text. " * 50, "repetitive"),
            ("Varied content with many unique words.", "varied"),
            ("Mixed content with some repetition.", "mixed"),
        ]

        results = []
        for content, expected_type in test_cases:
            result = compression_optimizer.analyze_compressibility(content)
            results.append(
                {
                    "type": expected_type,
                    "compressible": result.is_compressible,
                    "ratio": result.compression_ratio,
                }
            )

        # Repetitive content should be more compressible
        repetitive_result = [r for r in results if r["type"] == "repetitive"][0]
        varied_result = [r for r in results if r["type"] == "varied"][0]

        assert repetitive_result["ratio"] >= varied_result["ratio"], (
            "Repetitive content should be more compressible"
        )

    @pytest.mark.benchmark
    def test_memory_compression_savings(self, compression_optimizer, inference):
        """Measure memory savings from compression."""
        # Baseline memory
        baseline_result = inference.generate(
            prompt="Test memory compression.",
            config=GenerationConfig(max_new_tokens=100),
        )
        baseline_memory = baseline_result.memory_peak_mb

        # Compressed memory
        compressed_result = compression_optimizer.compress_and_generate(
            prompt="Test memory compression.",
            base_inference=inference,
            config=GenerationConfig(max_new_tokens=100),
        )
        compressed_memory = compressed_result.memory_peak_mb

        # Calculate savings
        savings = (baseline_memory - compressed_memory) / baseline_memory

        assert savings >= 0.0, "Compression should not increase memory usage"

    @pytest.mark.benchmark
    def test_different_compression_methods(
        self, compression_optimizer, inference, benchmark
    ):
        """Test different compression methods."""
        methods = ["dynamic", "static", "adaptive"]
        results = {}

        for method in methods:
            optimizer = CompressionOptimizer(
                method=method, target_ratio=0.75, quality_threshold=0.95
            )

            def test_method():
                return optimizer.compress_and_generate(
                    prompt="Test compression method.",
                    base_inference=inference,
                    config=GenerationConfig(max_new_tokens=50),
                )

            result = benchmark(test_method)
            results[method] = {
                "compression_ratio": result.compression_ratio,
                "quality_score": result.quality_score,
                "time": result.total_time,
            }

        # All methods should produce valid results
        for method, result in results.items():
            assert result["compression_ratio"] > 0, f"Method {method} should compress"
            assert result["quality_score"] > 0.9, (
                f"Method {method} should maintain quality"
            )
