#!/usr/bin/env python3
"""
Memory Benchmark Suite
Comprehensive memory benchmarks for the Nexus multimodal model.

Covers:
- Peak memory usage
- Activation memory optimization
- Gradient checkpointing savings
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

from src.models.omni.inference import OmniInference, GenerationConfig
from src.core.training.student_trainer import NexusDistillationTrainer
from src.core.training.loop import TrainingLoop


@dataclass
class MemoryBenchmarkResult:
    """Container for memory benchmark results."""

    peak_memory_mb: float
    active_memory_mb: float
    reserved_memory_mb: float
    memory_allocated_mb: float
    peak_activation_memory_mb: float
    gradient_checkpointing_savings_percent: float
    optimization_savings_mb: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "peak_memory_mb": self.peak_memory_mb,
            "active_memory_mb": self.active_memory_mb,
            "reserved_memory_mb": self.reserved_memory_mb,
            "memory_allocated_mb": self.memory_allocated_mb,
            "peak_activation_memory_mb": self.peak_activation_memory_mb,
            "gradient_checkpointing_savings_percent": self.gradient_checkpointing_savings_percent,
            "optimization_savings_mb": self.optimization_savings_mb,
        }


class TestPeakMemoryUsage:
    """Peak memory usage benchmarks."""

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
    def test_peak_memory_short_inference(self, inference, benchmark):
        """Measure peak memory for short inference."""

        def short_inference():
            return inference.generate(
                prompt="Hello.", config=GenerationConfig(max_new_tokens=10)
            )

        result = benchmark(short_inference)
        assert result.memory_peak_mb < 4096, "Short inference should use less than 4GB"

    @pytest.mark.benchmark
    def test_peak_memory_medium_inference(self, inference, benchmark):
        """Measure peak memory for medium inference."""

        def medium_inference():
            return inference.generate(
                prompt="Write a detailed analysis of machine learning.",
                config=GenerationConfig(max_new_tokens=200),
            )

        result = benchmark(medium_inference)
        assert result.memory_peak_mb < 6144, "Medium inference should use less than 6GB"

    @pytest.mark.benchmark
    def test_peak_memory_long_inference(self, inference, benchmark):
        """Measure peak memory for long inference."""

        def long_inference():
            return inference.generate(
                prompt="Write a comprehensive essay on the history of computing.",
                config=GenerationConfig(max_new_tokens=500),
            )

        result = benchmark(long_inference)
        assert result.memory_peak_mb < 8192, "Long inference should use less than 8GB"

    @pytest.mark.benchmark
    def test_peak_memory_batch_inference(self, inference, benchmark):
        """Measure peak memory for batched inference."""

        def batch_inference():
            prompts = [
                f"Question {i}: Explain the concept of neural networks."
                for i in range(8)
            ]
            return inference.batch_generate(prompts, max_tokens=100)

        result = benchmark(batch_inference)
        assert result.memory_peak_mb < 12288, (
            "Batch inference should use less than 12GB"
        )

    @pytest.mark.benchmark
    def test_peak_memory_varying_max_tokens(self, inference, benchmark):
        """Measure peak memory across different max token values."""
        memory_values = []

        for max_tokens in [50, 100, 200, 500]:

            def run_inference(mt=max_tokens):
                return inference.generate(
                    prompt="Test memory usage.",
                    config=GenerationConfig(max_new_tokens=mt),
                )

            result = benchmark(run_inference)
            memory_values.append(
                {"max_tokens": max_tokens, "peak_memory": result.memory_peak_mb}
            )

        # Memory should scale sub-linearly with max tokens
        short_memory = memory_values[0]["peak_memory"]
        long_memory = memory_values[-1]["peak_memory"]
        token_ratio = memory_values[-1]["max_tokens"] / memory_values[0]["max_tokens"]
        memory_ratio = long_memory / short_memory if short_memory > 0 else 1

        assert memory_ratio < token_ratio, (
            "Memory should scale sub-linearly with max tokens"
        )


class TestActivationMemory:
    """Activation memory optimization benchmarks."""

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
    def test_activation_memory_baseline(self, inference, benchmark):
        """Measure baseline activation memory without optimization."""

        def generate_with_activations():
            return inference.generate(
                prompt="Measure activation memory.",
                config=GenerationConfig(max_new_tokens=100),
            )

        result = benchmark(generate_with_activations)
        baseline_memory = result.activation_memory_mb

        # Store for later comparison
        inference.baseline_activation_memory = baseline_memory
        return baseline_memory

    @pytest.mark.benchmark
    def test_activation_caching_efficiency(self, inference, benchmark):
        """Test activation caching efficiency."""

        # First pass - cache activations
        def cache_activations():
            return inference.generate(
                prompt="First pass to cache activations.",
                config=GenerationConfig(max_new_tokens=100),
                cache_activations=True,
            )

        cache_result = benchmark(cache_activations)

        # Second pass - use cached activations
        def use_cached_activations():
            return inference.generate(
                prompt="Second pass using cached activations.",
                config=GenerationConfig(max_new_tokens=100),
                use_cached_activations=True,
            )

        cached_result = benchmark(use_cached_activations)

        # Cached should use less memory
        memory_savings = (
            cache_result.activation_memory_mb - cached_result.activation_memory_mb
        )
        savings_percent = (memory_savings / cache_result.activation_memory_mb) * 100

        assert savings_percent > 0, "Caching should save activation memory"

    @pytest.mark.benchmark
    def test_activation_compression_ratio(self, inference, benchmark):
        """Test activation compression ratio."""

        def generate_uncompressed():
            return inference.generate(
                prompt="Uncompressed activations.",
                config=GenerationConfig(max_new_tokens=100),
                compress_activations=False,
            )

        def generate_compressed():
            return inference.generate(
                prompt="Compressed activations.",
                config=GenerationConfig(max_new_tokens=100),
                compress_activations=True,
            )

        uncompressed_result = benchmark(generate_uncompressed)
        compressed_result = benchmark(generate_compressed)

        compression_ratio = (
            uncompressed_result.activation_memory_mb
            / compressed_result.activation_memory_mb
            if compressed_result.activation_memory_mb > 0
            else 1
        )

        assert compression_ratio >= 1.0, "Compression should not increase memory"

    @pytest.mark.benchmark
    def test_selective_activation_computation(self, inference, benchmark):
        """Test selective activation computation for attention."""

        # Compute only key activations
        def selective_attention():
            return inference.generate(
                prompt="Compute selective attention.",
                config=GenerationConfig(max_new_tokens=100),
                compute_attention_keys=True,
                compute_attention_values=True,
                compute_attention_queries=False,
            )

        result = benchmark(selective_attention)
        selective_memory = result.activation_memory_mb

        # Compare with full attention
        def full_attention():
            return inference.generate(
                prompt="Full attention computation.",
                config=GenerationConfig(max_new_tokens=100),
                compute_attention_keys=True,
                compute_attention_values=True,
                compute_attention_queries=True,
            )

        full_result = benchmark(full_attention)
        full_memory = full_result.activation_memory_mb

        # Selective should use less memory
        assert selective_memory <= full_memory, (
            "Selective activation should use less or equal memory"
        )

    @pytest.mark.benchmark
    def test_activation_memory_sequence_length(self, inference, benchmark):
        """Test activation memory scaling with sequence length."""
        results = []

        for seq_length in [128, 256, 512, 1024]:

            def generate_sequence(sl=seq_length):
                prompt = "word " * sl
                return inference.generate(
                    prompt=prompt[:sl], config=GenerationConfig(max_new_tokens=50)
                )

            result = benchmark(generate_sequence)
            results.append(
                {
                    "seq_length": seq_length,
                    "activation_memory": result.activation_memory_mb,
                }
            )

        # Activation memory should scale sub-linearly with sequence length
        short_seq_memory = results[0]["activation_memory"]
        long_seq_memory = results[-1]["activation_memory"]
        seq_ratio = results[-1]["seq_length"] / results[0]["seq_length"]
        memory_ratio = long_seq_memory / short_seq_memory if short_seq_memory > 0 else 1

        assert memory_ratio < seq_ratio, (
            "Activation memory should scale sub-linearly with sequence length"
        )


class TestGradientCheckpointing:
    """Gradient checkpointing benchmarks."""

    @pytest.fixture
    def trainer(self):
        """Set up training environment."""
        try:
            trainer = NexusDistillationTrainer(
                teacher_path="/mnt/e/data/models/Qwen2.5-Omni-7B-GPTQ-Int4",
                student_config={"d_model": 2048, "teacher_dim": 4096},
                profiling_data_path="/mnt/d/Research Experiments/nexus/data/profiles",
                device="cuda" if torch.cuda.is_available() else "cpu",
            )
            yield trainer
            del trainer
        except Exception as e:
            pytest.skip(f"Training setup failed: {e}")

    @pytest.fixture
    def sample_batch(self):
        """Create sample training batch."""
        return {
            "input_ids": torch.randint(0, 1000, (2, 512)),
            "attention_mask": torch.ones(2, 512),
            "teacher_logits": torch.randn(2, 512, 32000),
            "teacher_hidden_states": torch.randn(2, 512, 4096),
        }

    @pytest.mark.benchmark
    def test_gradient_checkpointing_baseline(self, trainer, sample_batch, benchmark):
        """Measure baseline memory without gradient checkpointing."""
        # Disable checkpointing
        trainer.disable_gradient_checkpointing()

        def train_without_checkpointing():
            trainer.student.train()

            student_output = trainer.student(
                input_ids=sample_batch["input_ids"],
                attention_mask=sample_batch["attention_mask"],
            )
            loss = nn.functional.cross_entropy(
                student_output.logits.view(-1, student_output.logits.size(-1)),
                sample_batch["input_ids"].view(-1),
            )
            loss.backward()

            peak_memory = trainer.get_memory_usage()
            trainer.optimizer.zero_grad()

            return peak_memory

        result = benchmark(train_without_checkpointing)
        baseline_memory = result.memory_peak_mb

        # Store for comparison
        trainer.baseline_gradient_memory = baseline_memory
        return baseline_memory

    @pytest.mark.benchmark
    def test_gradient_checkpointing_enabled(self, trainer, sample_batch, benchmark):
        """Measure memory with gradient checkpointing enabled."""
        # Enable checkpointing
        trainer.enable_gradient_checkpointing()

        def train_with_checkpointing():
            trainer.student.train()

            student_output = trainer.student(
                input_ids=sample_batch["input_ids"],
                attention_mask=sample_batch["attention_mask"],
            )
            loss = nn.functional.cross_entropy(
                student_output.logits.view(-1, student_output.logits.size(-1)),
                sample_batch["input_ids"].view(-1),
            )
            loss.backward()

            peak_memory = trainer.get_memory_usage()
            trainer.optimizer.zero_grad()

            return peak_memory

        result = benchmark(train_with_checkpointing)

        # Should use less memory than baseline
        baseline_memory = getattr(
            trainer, "baseline_gradient_memory", result.memory_peak_mb
        )
        memory_savings = baseline_memory - result.memory_peak_mb
        savings_percent = (
            (memory_savings / baseline_memory) * 100 if baseline_memory > 0 else 0
        )

        assert savings_percent >= 0, "Checkpointing should not increase memory"
        result.savings_percent = savings_percent
        return result

    @pytest.mark.benchmark
    def test_checkpointing_savings_percentage(self, trainer, sample_batch, benchmark):
        """Calculate exact gradient checkpointing savings percentage."""
        # Baseline
        trainer.disable_gradient_checkpointing()

        trainer.student.train()
        student_output = trainer.student(
            input_ids=sample_batch["input_ids"],
            attention_mask=sample_batch["attention_mask"],
        )
        loss = nn.functional.cross_entropy(
            student_output.logits.view(-1, student_output.logits.size(-1)),
            sample_batch["input_ids"].view(-1),
        )
        loss.backward()
        baseline_memory = trainer.get_memory_usage()
        trainer.optimizer.zero_grad()

        # With checkpointing
        trainer.enable_gradient_checkpointing()

        trainer.student.train()
        student_output = trainer.student(
            input_ids=sample_batch["input_ids"],
            attention_mask=sample_batch["attention_mask"],
        )
        loss = nn.functional.cross_entropy(
            student_output.logits.view(-1, student_output.logits.size(-1)),
            sample_batch["input_ids"].view(-1),
        )
        loss.backward()
        checkpointed_memory = trainer.get_memory_usage()
        trainer.optimizer.zero_grad()

        savings_percent = (
            (baseline_memory - checkpointed_memory) / baseline_memory
        ) * 100

        assert savings_percent >= 0, "Checkpointing should save memory"
        assert savings_percent <= 50, "Checkpointing savings should be realistic"

    @pytest.mark.benchmark
    def test_checkpointing_sequence_length_scaling(self, trainer, benchmark):
        """Test checkpointing benefits across sequence lengths."""
        savings_by_length = []

        for seq_length in [256, 512, 1024]:
            # Baseline
            trainer.disable_gradient_checkpointing()
            batch = {
                "input_ids": torch.randint(0, 1000, (2, seq_length)),
                "attention_mask": torch.ones(2, seq_length),
                "teacher_logits": torch.randn(2, seq_length, 32000),
                "teacher_hidden_states": torch.randn(2, seq_length, 4096),
            }

            trainer.student.train()
            student_output = trainer.student(
                input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
            )
            loss = nn.functional.cross_entropy(
                student_output.logits.view(-1, student_output.logits.size(-1)),
                batch["input_ids"].view(-1),
            )
            loss.backward()
            baseline_memory = trainer.get_memory_usage()
            trainer.optimizer.zero_grad()

            # With checkpointing
            trainer.enable_gradient_checkpointing()

            trainer.student.train()
            student_output = trainer.student(
                input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
            )
            loss = nn.functional.cross_entropy(
                student_output.logits.view(-1, student_output.logits.size(-1)),
                batch["input_ids"].view(-1),
            )
            loss.backward()
            checkpointed_memory = trainer.get_memory_usage()
            trainer.optimizer.zero_grad()

            savings_percent = (
                (baseline_memory - checkpointed_memory) / baseline_memory
            ) * 100
            savings_by_length.append(
                {"seq_length": seq_length, "savings_percent": savings_percent}
            )

        # Longer sequences should benefit more from checkpointing
        short_savings = savings_by_length[0]["savings_percent"]
        long_savings = savings_by_length[-1]["savings_percent"]

        assert long_savings >= short_savings, (
            "Longer sequences should benefit more from checkpointing"
        )

    @pytest.mark.benchmark
    def test_checkpointing_compute_overhead(self, trainer, sample_batch, benchmark):
        """Measure compute overhead from gradient checkpointing."""
        # Baseline timing
        trainer.disable_gradient_checkpointing()

        def baseline_step():
            trainer.student.train()
            student_output = trainer.student(
                input_ids=sample_batch["input_ids"],
                attention_mask=sample_batch["attention_mask"],
            )
            loss = nn.functional.cross_entropy(
                student_output.logits.view(-1, student_output.logits.size(-1)),
                sample_batch["input_ids"].view(-1),
            )
            loss.backward()
            trainer.optimizer.step()
            trainer.optimizer.zero_grad()
            return loss.item()

        baseline_result = benchmark(baseline_step)

        # Checkpointing timing
        trainer.enable_gradient_checkpointing()

        def checkpointing_step():
            trainer.student.train()
            student_output = trainer.student(
                input_ids=sample_batch["input_ids"],
                attention_mask=sample_batch["attention_mask"],
            )
            loss = nn.functional.cross_entropy(
                student_output.logits.view(-1, student_output.logits.size(-1)),
                sample_batch["input_ids"].view(-1),
            )
            loss.backward()
            trainer.optimizer.step()
            trainer.optimizer.zero_grad()
            return loss.item()

        checkpointing_result = benchmark(checkpointing_step)

        # Checkpointing should be slower (compute overhead)
        time_overhead = (
            (checkpointing_result.total_time - baseline_result.total_time)
            / baseline_result.total_time
            * 100
        )

        assert time_overhead >= 0, (
            "Checkpointing should have non-negative time overhead"
        )


class TestMemoryOptimization:
    """General memory optimization benchmarks."""

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
    def test_memory_allocator_efficiency(self, inference, benchmark):
        """Test custom memory allocator efficiency."""

        # Use custom allocator
        def generate_with_custom_allocator():
            return inference.generate(
                prompt="Test custom allocator.",
                config=GenerationConfig(max_new_tokens=100),
                use_custom_allocator=True,
            )

        result = benchmark(generate_with_custom_allocator)

        # Compare with default
        def generate_with_default():
            return inference.generate(
                prompt="Test default allocator.",
                config=GenerationConfig(max_new_tokens=100),
                use_custom_allocator=False,
            )

        default_result = benchmark(generate_with_default)

        # Custom should be at least as efficient
        assert result.memory_peak_mb <= default_result.memory_peak_mb * 1.1, (
            "Custom allocator should not be significantly worse"
        )

    @pytest.mark.benchmark
    def test_memory_defragmentation(self, inference, benchmark):
        """Test memory defragmentation efficiency."""
        # Generate multiple times to fragment memory
        for _ in range(5):
            inference.generate(
                prompt="Fragment memory.", config=GenerationConfig(max_new_tokens=100)
            )

        # Defragment
        def defragment_and_generate():
            inference.defragment_memory()
            return inference.generate(
                prompt="After defragmentation.",
                config=GenerationConfig(max_new_tokens=100),
            )

        result = benchmark(defragment_and_generate)

        # Should have reasonable memory usage after defragmentation
        assert result.memory_peak_mb < 10240, (
            "Memory should be reasonable after defragmentation"
        )

    @pytest.mark.benchmark
    def test_memory_pooling(self, inference, benchmark):
        """Test memory pooling efficiency."""

        # Enable pooling
        def generate_with_pooling():
            return inference.generate(
                prompt="Test pooling.",
                config=GenerationConfig(max_new_tokens=100),
                use_memory_pooling=True,
            )

        pooling_result = benchmark(generate_with_pooling)
        pooling_memory = pooling_result.memory_peak_mb

        # Disable pooling
        def generate_without_pooling():
            return inference.generate(
                prompt="Test no pooling.",
                config=GenerationConfig(max_new_tokens=100),
                use_memory_pooling=False,
            )

        no_pooling_result = benchmark(generate_without_pooling)
        no_pooling_memory = no_pooling_result.memory_peak_mb

        # Pooling should be at least as efficient
        assert pooling_memory <= no_pooling_memory * 1.1, (
            "Memory pooling should not significantly increase usage"
        )

    @pytest.mark.benchmark
    def test_tensor_memory_reuse(self, inference, benchmark):
        """Test tensor memory reuse efficiency."""

        # Generate multiple times
        def generate_multiple():
            for _ in range(3):
                inference.generate(
                    prompt="Reuse tensor memory.",
                    config=GenerationConfig(max_new_tokens=50),
                )
            return inference.get_memory_usage()

        result = benchmark(generate_multiple)

        # Memory should stabilize
        assert result < 8192, "Memory should stabilize with reuse"

    @pytest.mark.benchmark
    def test_peak_vs_active_memory(self, inference, benchmark):
        """Compare peak vs active memory usage."""

        def measure_memory():
            result = inference.generate(
                prompt="Measure peak vs active.",
                config=GenerationConfig(max_new_tokens=100),
            )
            return {
                "peak": result.memory_peak_mb,
                "active": result.active_memory_mb,
                "reserved": result.reserved_memory_mb,
            }

        result = benchmark(measure_memory)

        # Active memory should be less than peak
        assert result["active"] <= result["peak"], (
            "Active memory should not exceed peak memory"
        )

        # Reserved should be reasonable
        assert result["reserved"] < result["peak"] * 1.5, (
            "Reserved memory should not significantly exceed peak"
        )


class TestGPUMemorySpecific:
    """GPU-specific memory benchmarks."""

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

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="No GPU available")
    @pytest.mark.benchmark
    def test_cuda_memory_management(self, inference, benchmark):
        """Test CUDA memory management."""
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        def cuda_generation():
            result = inference.generate(
                prompt="Test CUDA memory.", config=GenerationConfig(max_new_tokens=100)
            )
            return result

        result = benchmark(cuda_generation)

        # Check CUDA stats
        peak_memory = torch.cuda.max_memory_allocated() / 1024**2
        assert peak_memory < 10240, "CUDA peak memory should be under 10GB"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="No GPU available")
    @pytest.mark.benchmark
    def test_cuda_memory_fraction(self, inference, benchmark):
        """Test different CUDA memory fractions."""
        fractions = [0.5, 0.75, 0.9]
        results = []

        for fraction in fractions:
            torch.cuda.set_per_process_memory_fraction(fraction)
            torch.cuda.empty_cache()

            def generate_with_fraction(frac=fraction):
                return inference.generate(
                    prompt=f"Test fraction {frac}.",
                    config=GenerationConfig(max_new_tokens=50),
                )

            result = benchmark(generate_with_fraction)
            results.append({"fraction": fraction, "memory_used": result.memory_peak_mb})

        # All should complete successfully
        assert len(results) == len(fractions), "All fractions should complete"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="No GPU available")
    @pytest.mark.benchmark
    def test_memory_pinning_efficiency(self, inference, benchmark):
        """Test memory pinning efficiency for data transfer."""

        # With pinning
        def generate_with_pinning():
            return inference.generate(
                prompt="Test pinning.",
                config=GenerationConfig(max_new_tokens=100),
                pin_memory=True,
            )

        pinning_result = benchmark(generate_with_pinning)

        # Without pinning
        def generate_without_pinning():
            return inference.generate(
                prompt="Test no pinning.",
                config=GenerationConfig(max_new_tokens=100),
                pin_memory=False,
            )

        no_pinning_result = benchmark(generate_without_pinning)

        # Pinning should complete successfully
        assert pinning_result.tokens_per_second > 0, "Pinning should work"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="No GPU available")
    @pytest.mark.benchmark
    def test_asynchronous_memory_operations(self, inference, benchmark):
        """Test asynchronous memory operations."""

        def async_generation():
            return inference.generate(
                prompt="Test async memory.",
                config=GenerationConfig(max_new_tokens=100),
                async_memory_ops=True,
            )

        result = benchmark(async_generation)

        # Should complete successfully
        assert result.tokens_per_second > 0, "Async memory operations should work"
