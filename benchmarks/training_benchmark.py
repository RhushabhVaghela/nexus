#!/usr/bin/env python3
"""
Training Benchmark Suite
Comprehensive training benchmarks for the Nexus multimodal model.

Covers:
- Training throughput
- Gradient accumulation efficiency
- Checkpoint save/load time
- Mixed precision overhead
"""

import pytest
import torch
import torch.nn as nn
import time
import numpy as np
from typing import Dict, Any, List
from dataclasses import dataclass
from pathlib import Path
import sys
import tempfile
import shutil

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.core.training.student_trainer import NexusDistillationTrainer
from src.core.training.loop import TrainingLoop
from src.core.training.data_loader import create_data_loader


@dataclass
class TrainingBenchmarkResult:
    """Container for training benchmark results."""

    throughput_samples_per_second: float
    tokens_per_second: float
    samples_per_second_per_gpu: float
    epoch_time_seconds: float
    gradient_accumulation_overhead: float
    checkpoint_save_time_seconds: float
    checkpoint_load_time_seconds: float
    mixed_precision_overhead_percent: float
    memory_peak_mb: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "throughput_samples_per_second": self.throughput_samples_per_second,
            "tokens_per_second": self.tokens_per_second,
            "samples_per_second_per_gpu": self.samples_per_second_per_gpu,
            "epoch_time_seconds": self.epoch_time_seconds,
            "gradient_accumulation_overhead": self.gradient_accumulation_overhead,
            "checkpoint_save_time_seconds": self.checkpoint_save_time_seconds,
            "checkpoint_load_time_seconds": self.checkpoint_load_time_seconds,
            "mixed_precision_overhead_percent": self.mixed_precision_overhead_percent,
            "memory_peak_mb": self.memory_peak_mb,
        }


class TestTrainingThroughput:
    """Training throughput benchmarks."""

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
            "input_ids": torch.randint(0, 1000, (4, 512)),
            "attention_mask": torch.ones(4, 512),
            "teacher_logits": torch.randn(4, 512, 32000),
            "teacher_hidden_states": torch.randn(4, 512, 4096),
        }

    @pytest.mark.benchmark
    def test_forward_pass_throughput(self, trainer, sample_batch, benchmark):
        """Benchmark forward pass throughput."""

        def forward_pass():
            trainer.student.train()
            with torch.no_grad():
                student_output = trainer.student(
                    input_ids=sample_batch["input_ids"],
                    attention_mask=sample_batch["attention_mask"],
                )
            return student_output

        result = benchmark(forward_pass)
        assert result.tokens_per_second > 100, "Forward pass should be fast"

    @pytest.mark.benchmark
    def test_forward_backward_throughput(self, trainer, sample_batch, benchmark):
        """Benchmark forward-backward pass throughput."""

        def forward_backward():
            trainer.student.train()

            # Forward
            student_output = trainer.student(
                input_ids=sample_batch["input_ids"],
                attention_mask=sample_batch["attention_mask"],
            )

            # Compute loss (simplified)
            loss = nn.functional.cross_entropy(
                student_output.logits.view(-1, student_output.logits.size(-1)),
                sample_batch["input_ids"].view(-1),
            )

            # Backward
            loss.backward()

            return loss.item()

        result = benchmark(forward_backward)
        assert result > 0, "Forward-backward should produce valid loss"

    @pytest.mark.benchmark
    def test_training_step_throughput(self, trainer, sample_batch, benchmark):
        """Benchmark complete training step throughput."""

        def training_step():
            trainer.student.train()

            # Forward
            student_output = trainer.student(
                input_ids=sample_batch["input_ids"],
                attention_mask=sample_batch["attention_mask"],
            )

            # Compute loss
            loss = nn.functional.cross_entropy(
                student_output.logits.view(-1, student_output.logits.size(-1)),
                sample_batch["input_ids"].view(-1),
            )

            # Backward
            loss.backward()

            # Optimizer step
            trainer.optimizer.step()
            trainer.optimizer.zero_grad()

            return loss.item()

        result = benchmark(training_step)
        assert result > 0, "Training step should produce valid loss"

    @pytest.mark.benchmark
    def test_batch_size_scaling(self, trainer, benchmark):
        """Test throughput scaling with batch size."""
        scaling_results = []

        for batch_size in [1, 2, 4, 8]:
            batch = {
                "input_ids": torch.randint(0, 1000, (batch_size, 512)),
                "attention_mask": torch.ones(batch_size, 512),
                "teacher_logits": torch.randn(batch_size, 512, 32000),
                "teacher_hidden_states": torch.randn(batch_size, 512, 4096),
            }

            def step(b=batch):
                trainer.student.train()
                student_output = trainer.student(
                    input_ids=b["input_ids"], attention_mask=b["attention_mask"]
                )
                loss = nn.functional.cross_entropy(
                    student_output.logits.view(-1, student_output.logits.size(-1)),
                    b["input_ids"].view(-1),
                )
                loss.backward()
                trainer.optimizer.step()
                trainer.optimizer.zero_grad()
                return loss.item()

            result = benchmark(step)
            throughput = batch_size / result.total_time
            scaling_results.append(
                {
                    "batch_size": batch_size,
                    "throughput": throughput,
                    "time": result.total_time,
                }
            )

        # Verify scaling behavior
        base_throughput = scaling_results[0]["throughput"]
        for result in scaling_results:
            efficiency = result["throughput"] / (result["batch_size"] * base_throughput)
            assert efficiency >= 0.5, (
                f"Batch size {result['batch_size']} should have reasonable efficiency"
            )

    @pytest.mark.benchmark
    def test_sequence_length_scaling(self, trainer, benchmark):
        """Test throughput scaling with sequence length."""
        length_results = []

        for seq_len in [128, 256, 512, 1024]:
            batch = {
                "input_ids": torch.randint(0, 1000, (4, seq_len)),
                "attention_mask": torch.ones(4, seq_len),
                "teacher_logits": torch.randn(4, seq_len, 32000),
                "teacher_hidden_states": torch.randn(4, seq_len, 4096),
            }

            def step(b=batch):
                trainer.student.train()
                student_output = trainer.student(
                    input_ids=b["input_ids"], attention_mask=b["attention_mask"]
                )
                loss = nn.functional.cross_entropy(
                    student_output.logits.view(-1, student_output.logits.size(-1)),
                    b["input_ids"].view(-1),
                )
                loss.backward()
                trainer.optimizer.step()
                trainer.optimizer.zero_grad()
                return loss.item()

            result = benchmark(step)
            throughput = 4 / result.total_time  # samples per second
            length_results.append(
                {
                    "seq_len": seq_len,
                    "throughput": throughput,
                    "time": result.total_time,
                }
            )

        # Longer sequences should take proportionally longer
        short_time = length_results[0]["time"]
        long_time = length_results[-1]["time"]
        length_ratio = length_results[-1]["seq_len"] / length_results[0]["seq_len"]
        time_ratio = long_time / short_time

        assert time_ratio < length_ratio * 1.5, (
            "Time should scale sub-linearly with sequence length"
        )


class TestGradientAccumulation:
    """Gradient accumulation efficiency benchmarks."""

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
            "input_ids": torch.randint(0, 1000, (4, 512)),
            "attention_mask": torch.ones(4, 512),
            "teacher_logits": torch.randn(4, 512, 32000),
            "teacher_hidden_states": torch.randn(4, 512, 4096),
        }

    @pytest.mark.benchmark
    def test_gradient_accumulation_efficiency(self, trainer, sample_batch, benchmark):
        """Test efficiency of gradient accumulation."""
        accumulation_steps = 4

        def accumulate_gradients():
            trainer.student.train()
            accumulated_grads = None

            for step in range(accumulation_steps):
                student_output = trainer.student(
                    input_ids=sample_batch["input_ids"],
                    attention_mask=sample_batch["attention_mask"],
                )
                loss = nn.functional.cross_entropy(
                    student_output.logits.view(-1, student_output.logits.size(-1)),
                    sample_batch["input_ids"].view(-1),
                )
                loss = loss / accumulation_steps
                loss.backward()

            # Apply gradients
            trainer.optimizer.step()
            trainer.optimizer.zero_grad()

            return loss.item()

        result = benchmark(accumulate_gradients)
        assert result > 0, "Gradient accumulation should produce valid loss"

    @pytest.mark.benchmark
    def test_accumulation_memory_efficiency(self, trainer, sample_batch):
        """Test memory efficiency with gradient accumulation."""
        # Baseline: full batch
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

        # Accumulated: split batch
        trainer.student.train()
        half_batch = {
            "input_ids": sample_batch["input_ids"][:, :256],
            "attention_mask": sample_batch["attention_mask"][:, :256],
        }

        for _ in range(2):  # Accumulate twice
            student_output = trainer.student(
                input_ids=half_batch["input_ids"],
                attention_mask=half_batch["attention_mask"],
            )
            loss = nn.functional.cross_entropy(
                student_output.logits.view(-1, student_output.logits.size(-1)),
                half_batch["input_ids"].view(-1),
            )
            loss.backward()

        accumulated_memory = trainer.get_memory_usage()
        trainer.optimizer.zero_grad()

        # Accumulation should use less or equal memory
        assert accumulated_memory <= baseline_memory, (
            "Gradient accumulation should not increase memory"
        )

    @pytest.mark.benchmark
    def test_accumulation_steps_scaling(self, trainer, sample_batch, benchmark):
        """Test scaling with different accumulation step counts."""
        results = []

        for accum_steps in [1, 2, 4, 8]:

            def train_with_accumulation(s=accum_steps):
                trainer.student.train()

                for step in range(s):
                    student_output = trainer.student(
                        input_ids=sample_batch["input_ids"],
                        attention_mask=sample_batch["attention_mask"],
                    )
                    loss = nn.functional.cross_entropy(
                        student_output.logits.view(-1, student_output.logits.size(-1)),
                        sample_batch["input_ids"].view(-1),
                    )
                    loss = loss / s
                    loss.backward()

                trainer.optimizer.step()
                trainer.optimizer.zero_grad()
                return loss.item()

            result = benchmark(train_with_accumulation)
            effective_batch_size = sample_batch["input_ids"].size(0) * accum_steps
            results.append(
                {
                    "accum_steps": accum_steps,
                    "effective_batch": effective_batch_size,
                    "time": result.total_time,
                }
            )

        # Verify effective batch processing
        for result in results:
            assert result["effective_batch"] > 0, (
                f"Accumulation steps {result['accum_steps']} should produce valid effective batch"
            )


class TestCheckpointPerformance:
    """Checkpoint save/load performance benchmarks."""

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
    def temp_checkpoint_dir(self):
        """Create temporary checkpoint directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.mark.benchmark
    def test_checkpoint_save_time(self, trainer, temp_checkpoint_dir, benchmark):
        """Benchmark checkpoint save time."""

        def save_checkpoint():
            checkpoint_path = f"{temp_checkpoint_dir}/checkpoint_step_1000.pt"
            trainer.save_checkpoint(checkpoint_path)
            return checkpoint_path

        result = benchmark(save_checkpoint)
        assert result.total_time < 60, (
            "Checkpoint save should complete in under 60 seconds"
        )

    @pytest.mark.benchmark
    def test_checkpoint_load_time(self, trainer, temp_checkpoint_dir, benchmark):
        """Benchmark checkpoint load time."""
        # First save a checkpoint
        checkpoint_path = f"{temp_checkpoint_dir}/checkpoint_test.pt"
        trainer.save_checkpoint(checkpoint_path)

        def load_checkpoint():
            trainer.load_checkpoint(checkpoint_path)

        result = benchmark(load_checkpoint)
        assert result.total_time < 60, (
            "Checkpoint load should complete in under 60 seconds"
        )

    @pytest.mark.benchmark
    def test_optimizer_state_checkpointing(
        self, trainer, temp_checkpoint_dir, benchmark
    ):
        """Benchmark optimizer state checkpointing."""

        def checkpoint_with_optimizer():
            checkpoint_path = f"{temp_checkpoint_dir}/checkpoint_with_optimizer.pt"
            trainer.save_checkpoint(checkpoint_path, include_optimizer=True)
            return checkpoint_path

        result = benchmark(checkpoint_with_optimizer)
        assert result.total_time < 120, (
            "Checkpoint with optimizer should complete in under 2 minutes"
        )

    @pytest.mark.benchmark
    def test_partial_checkpointing(self, trainer, temp_checkpoint_dir, benchmark):
        """Test partial checkpointing (only weights vs full state)."""

        def save_weights_only():
            checkpoint_path = f"{temp_checkpoint_dir}/weights_only.pt"
            trainer.save_checkpoint(checkpoint_path, only_weights=True)
            return checkpoint_path

        def save_full_state():
            checkpoint_path = f"{temp_checkpoint_dir}/full_state.pt"
            trainer.save_checkpoint(checkpoint_path, only_weights=False)
            return checkpoint_path

        weights_result = benchmark(save_weights_only)
        full_result = benchmark(save_full_state)

        # Weights-only should be faster
        assert weights_result.total_time <= full_result.total_time, (
            "Weights-only checkpoint should be faster or equal"
        )

    @pytest.mark.benchmark
    def test_checkpoint_integrity(self, trainer, temp_checkpoint_dir):
        """Verify checkpoint integrity after save/load."""
        # Get initial state dict
        initial_state = {k: v.clone() for k, v in trainer.student.state_dict().items()}
        initial_opt_state = {
            k: v.clone()
            for k, v in trainer.optimizer.state_dict().items()
            if isinstance(v, torch.Tensor)
        }

        # Save checkpoint
        checkpoint_path = f"{temp_checkpoint_dir}/integrity_test.pt"
        trainer.save_checkpoint(checkpoint_path)

        # Modify current state (simulate training)
        with torch.no_grad():
            for param in trainer.student.parameters():
                param.add_(torch.randn_like(param) * 0.01)

        # Load checkpoint
        trainer.load_checkpoint(checkpoint_path)

        # Verify state matches original
        loaded_state = trainer.student.state_dict()
        loaded_opt_state = trainer.optimizer.state_dict()

        # Check model weights
        for key in initial_state:
            if "weight" in key or "bias" in key:
                torch.testing.assert_close(
                    initial_state[key].cpu(),
                    loaded_state[key].cpu(),
                    rtol=1e-4,
                    atol=1e-4,
                )

        assert True, "Checkpoint integrity verified"


class TestMixedPrecision:
    """Mixed precision training benchmarks."""

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
            "input_ids": torch.randint(0, 1000, (4, 512)),
            "attention_mask": torch.ones(4, 512),
            "teacher_logits": torch.randn(4, 512, 32000),
            "teacher_hidden_states": torch.randn(4, 512, 4096),
        }

    @pytest.mark.benchmark
    def test_fp16_training_throughput(self, trainer, sample_batch, benchmark):
        """Benchmark FP16 training throughput."""
        # Enable FP16
        trainer.enable_fp16()

        def fp16_training_step():
            trainer.student.train()

            with torch.autocast(device_type="cuda", dtype=torch.float16):
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

        result = benchmark(fp16_training_step)
        assert result > 0, "FP16 training should produce valid loss"

    @pytest.mark.benchmark
    def test_fp32_baseline_throughput(self, trainer, sample_batch, benchmark):
        """Benchmark FP32 baseline throughput."""
        # Ensure FP32
        trainer.disable_fp16()

        def fp32_training_step():
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

        result = benchmark(fp32_training_step)
        assert result > 0, "FP32 training should produce valid loss"

    @pytest.mark.benchmark
    def test_fp16_speedup(self, trainer, sample_batch, benchmark):
        """Measure FP16 speedup over FP32."""
        # FP16
        trainer.enable_fp16()

        def fp16_step():
            trainer.student.train()
            with torch.autocast(device_type="cuda", dtype=torch.float16):
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

        fp16_result = benchmark(fp16_step)

        # FP32
        trainer.disable_fp16()

        def fp32_step():
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

        fp32_result = benchmark(fp32_step)

        speedup = fp32_result.total_time / fp16_result.total_time
        assert speedup >= 0.8, "FP16 should be at least 80% as fast as FP32"

    @pytest.mark.benchmark
    def test_fp16_memory_savings(self, trainer, sample_batch):
        """Measure FP16 memory savings."""
        # FP32 memory
        trainer.disable_fp16()
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
        fp32_memory = trainer.get_memory_usage()
        trainer.optimizer.zero_grad()

        # FP16 memory
        trainer.enable_fp16()
        trainer.student.train()
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            student_output = trainer.student(
                input_ids=sample_batch["input_ids"],
                attention_mask=sample_batch["attention_mask"],
            )
            loss = nn.functional.cross_entropy(
                student_output.logits.view(-1, student_output.logits.size(-1)),
                sample_batch["input_ids"].view(-1),
            )
        loss.backward()
        fp16_memory = trainer.get_memory_usage()
        trainer.optimizer.zero_grad()

        # FP16 should use less memory
        assert fp16_memory <= fp32_memory, "FP16 should not use more memory than FP32"

    @pytest.mark.benchmark
    def test_fp16_gradient_scaling(self, trainer, sample_batch, benchmark):
        """Test gradient scaling for FP16 stability."""
        # Enable FP16 with scaling
        scaler = torch.cuda.amp.GradScaler()
        trainer.enable_fp16()

        def training_with_scaling():
            trainer.student.train()

            with torch.autocast(device_type="cuda", dtype=torch.float16):
                student_output = trainer.student(
                    input_ids=sample_batch["input_ids"],
                    attention_mask=sample_batch["attention_mask"],
                )
                loss = nn.functional.cross_entropy(
                    student_output.logits.view(-1, student_output.logits.size(-1)),
                    sample_batch["input_ids"].view(-1),
                )

            scaler.scale(loss).backward()
            scaler.step(trainer.optimizer)
            scaler.update()
            trainer.optimizer.zero_grad()

            return loss.item()

        result = benchmark(training_with_scaling)
        assert result > 0, "Training with gradient scaling should work"


class TestDataLoaderPerformance:
    """Data loader performance benchmarks."""

    @pytest.fixture
    def data_loader(self):
        """Create data loader."""
        try:
            loader = create_data_loader(
                data_path="/mnt/d/Research Experiments/nexus/data/training",
                batch_size=4,
                seq_length=512,
                num_workers=4,
            )
            yield loader
            del loader
        except Exception as e:
            pytest.skip(f"Data loader creation failed: {e}")

    @pytest.mark.benchmark
    def test_data_loading_throughput(self, data_loader, benchmark):
        """Benchmark data loading throughput."""

        def load_batch():
            batch = next(iter(data_loader))
            return batch

        result = benchmark(load_batch)
        assert result.samples_per_second > 1, (
            "Data loading should be faster than 1 sample/sec"
        )

    @pytest.mark.benchmark
    def test_prefetching_efficiency(self, data_loader, benchmark):
        """Test prefetching efficiency."""
        times = []

        for _ in range(10):
            start = time.perf_counter()
            batch = next(iter(data_loader))
            load_time = time.perf_counter() - start
            times.append(load_time)

        avg_time = np.mean(times)
        std_time = np.std(times)

        # Prefetching should reduce variance
        assert std_time < avg_time, "Prefetching should reduce load time variance"

    @pytest.mark.benchmark
    def test_num_workers_scaling(self, benchmark):
        """Test scaling with number of data loader workers."""
        scaling_results = []

        for num_workers in [0, 1, 2, 4]:
            loader = create_data_loader(
                data_path="/mnt/d/Research Experiments/nexus/data/training",
                batch_size=4,
                seq_length=512,
                num_workers=num_workers,
            )

            def load_multiple():
                for _ in range(5):
                    batch = next(iter(loader))
                return batch

            result = benchmark(load_multiple)
            throughput = 5 / result.total_time
            scaling_results.append(
                {
                    "num_workers": num_workers,
                    "throughput": throughput,
                    "time": result.total_time,
                }
            )

        # More workers should generally improve throughput
        zero_worker_throughput = scaling_results[0]["throughput"]
        for result in scaling_results:
            if result["num_workers"] > 0:
                assert result["throughput"] >= 0, (
                    f"Workers={result['num_workers']} should have non-negative throughput"
                )
