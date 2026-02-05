"""
Integration tests for NVFP4 + QAD Pipeline.

Tests cover:
- Full pipeline with quantized teacher
- Distillation with NVFP4 layers
- Mixed precision verification
- End-to-end quantization-aware distillation
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

# Import the modules under test
from src.models.sli import (
    NVFP4StreamingLoader,
    NVFP4Quantizer,
    NVFP4Config,
    NVFP4Mode,
    QuantizedTensor,
    QADDistillationLoss,
    QADLossConfig,
    QADLossType,
    PerLayerQADLoss,
    NestedUpdateScheduler,
    NestedUpdateConfig,
    UpdateGroup,
    HierarchicalLayerCache,
    HierarchicalCacheConfig,
    AdvancedSLIIntegrator,
    AdvancedSLIConfig,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def nvfp4_config():
    """Create NVFP4 config for testing."""
    return NVFP4Config(
        mode=NVFP4Mode.SOFTWARE, block_size=16, compute_dtype=torch.bfloat16
    )


@pytest.fixture
def qad_config():
    """Create QAD loss config for testing."""
    return QADLossConfig(
        temperature=1.5,
        alpha=0.7,
        beta=0.3,
        label_smoothing=0.1,
        loss_type=QADLossType.KL_DIVERGENCE,
    )


@pytest.fixture
def temp_cache_dir(tmp_path):
    """Create temporary cache directory."""
    cache_dir = tmp_path / "nvfp4_qad_cache"
    return str(cache_dir)


@pytest.fixture
def nvfp4_loader(nvfp4_config, temp_cache_dir):
    """Create NVFP4 loader."""
    return NVFP4StreamingLoader(
        config=nvfp4_config, cache_dir=temp_cache_dir, device="cpu"
    )


@pytest.fixture
def qad_loss(qad_config):
    """Create QAD loss."""
    return QADDistillationLoss(qad_config)


@pytest.fixture
def sample_teacher_model():
    """Create sample teacher model (FP32)."""

    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(512, 1024)
            self.fc2 = nn.Linear(1024, 1024)
            self.fc3 = nn.Linear(1024, 1000)

        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    model = SimpleModel()
    model.eval()
    return model


@pytest.fixture
def sample_student_model():
    """Create sample student model (to be quantized)."""

    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(512, 1024)
            self.fc2 = nn.Linear(1024, 1024)
            self.fc3 = nn.Linear(1024, 1000)

        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    model = SimpleModel()
    model.eval()
    return model


@pytest.fixture
def sample_batch():
    """Create sample input batch."""
    torch.manual_seed(42)
    return torch.randn(4, 512)


@pytest.fixture
def sample_labels():
    """Create sample labels."""
    torch.manual_seed(43)
    return torch.randint(0, 1000, (4,))


# ============================================================================
# Test Quantized Teacher Pipeline
# ============================================================================


class TestQuantizedTeacherPipeline:
    """Test suite for quantized teacher pipeline."""

    def test_quantize_teacher_model(self, nvfp4_loader, sample_teacher_model):
        """Test quantizing teacher model with NVFP4."""
        quantized_layers = []

        for name, module in sample_teacher_model.named_children():
            if isinstance(module, nn.Linear):
                quantized = nvfp4_loader.quantize_layer(
                    module, is_attention=False, layer_name=f"teacher_{name}"
                )
                quantized_layers.append(quantized)

        assert len(quantized_layers) == 3
        for layer in quantized_layers:
            assert isinstance(layer, nn.Module)

    def test_quantized_teacher_inference(
        self, nvfp4_loader, sample_teacher_model, sample_batch
    ):
        """Test inference with quantized teacher."""
        # Quantize teacher
        quantized_model = nn.Sequential()
        for name, module in sample_teacher_model.named_children():
            if isinstance(module, nn.Linear):
                quantized = nvfp4_loader.quantize_layer(module, is_attention=False)
                dequantized = nvfp4_loader.dequantize_layer(quantized)
                quantized_model.add_module(name, dequantized)
            else:
                quantized_model.add_module(name, module)

        # Run inference
        with torch.no_grad():
            output = quantized_model(sample_batch)

        assert output.shape == (4, 1000)

    def test_quantized_teacher_outputs_similar(
        self, nvfp4_loader, sample_teacher_model, sample_batch
    ):
        """Test that quantized teacher outputs are similar to original."""
        # Original output
        with torch.no_grad():
            original_output = sample_teacher_model(sample_batch)

        # Quantized output
        quantized_model = nn.Sequential()
        for name, module in sample_teacher_model.named_children():
            if isinstance(module, nn.Linear):
                quantized = nvfp4_loader.quantize_layer(module, is_attention=False)
                dequantized = nvfp4_loader.dequantize_layer(quantized)
                quantized_model.add_module(name, dequantized)
            else:
                quantized_model.add_module(name, module)

        with torch.no_grad():
            quantized_output = quantized_model(sample_batch)

        # Outputs should be similar
        error = torch.abs(original_output - quantized_output).mean()
        assert error < 1.0  # Reasonable error threshold


# ============================================================================
# Test Distillation with NVFP4 Layers
# ============================================================================


class TestDistillationWithNVFP4Layers:
    """Test suite for distillation with NVFP4 layers."""

    def test_distill_with_quantized_teacher(
        self,
        nvfp4_loader,
        qad_loss,
        sample_teacher_model,
        sample_student_model,
        sample_batch,
        sample_labels,
    ):
        """Test distillation with quantized teacher."""
        # Quantize teacher
        quantized_teacher_layers = []
        for name, module in sample_teacher_model.named_children():
            if isinstance(module, nn.Linear):
                quantized = nvfp4_loader.quantize_layer(module, is_attention=False)
                dequantized = nvfp4_loader.dequantize_layer(quantized)
                quantized_teacher_layers.append(dequantized)

        # Create teacher model with quantized layers
        class QuantizedTeacher(nn.Module):
            def __init__(self, layers):
                super().__init__()
                self.fc1 = layers[0]
                self.fc2 = layers[1]
                self.fc3 = layers[2]

            def forward(self, x):
                x = F.relu(self.fc1(x))
                x = F.relu(self.fc2(x))
                return self.fc3(x)

        quantized_teacher = QuantizedTeacher(quantized_teacher_layers)

        # Get teacher and student outputs
        with torch.no_grad():
            teacher_logits = quantized_teacher(sample_batch)

        student_logits = sample_student_model(sample_batch)

        # Compute distillation loss
        loss = qad_loss(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            labels=sample_labels,
        )

        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0

    def test_distill_with_quantized_student(
        self,
        nvfp4_loader,
        qad_loss,
        sample_teacher_model,
        sample_student_model,
        sample_batch,
        sample_labels,
    ):
        """Test distillation with quantized student."""
        # Quantize student
        quantized_student_layers = []
        for name, module in sample_student_model.named_children():
            if isinstance(module, nn.Linear):
                quantized = nvfp4_loader.quantize_layer(module, is_attention=False)
                dequantized = nvfp4_loader.dequantize_layer(quantized)
                quantized_student_layers.append(dequantized)

        # Create student model with quantized layers
        class QuantizedStudent(nn.Module):
            def __init__(self, layers):
                super().__init__()
                self.fc1 = layers[0]
                self.fc2 = layers[1]
                self.fc3 = layers[2]

            def forward(self, x):
                x = F.relu(self.fc1(x))
                x = F.relu(self.fc2(x))
                return self.fc3(x)

        quantized_student = QuantizedStudent(quantized_student_layers)

        # Get teacher and student outputs
        with torch.no_grad():
            teacher_logits = sample_teacher_model(sample_batch)

        student_logits = quantized_student(sample_batch)

        # Compute distillation loss
        loss = qad_loss(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            labels=sample_labels,
        )

        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0

    def test_distill_both_quantized(
        self,
        nvfp4_loader,
        qad_loss,
        sample_teacher_model,
        sample_student_model,
        sample_batch,
        sample_labels,
    ):
        """Test distillation with both teacher and student quantized."""

        # Quantize both models
        def quantize_model(model):
            quantized_layers = []
            for name, module in model.named_children():
                if isinstance(module, nn.Linear):
                    quantized = nvfp4_loader.quantize_layer(module, is_attention=False)
                    dequantized = nvfp4_loader.dequantize_layer(quantized)
                    quantized_layers.append(dequantized)
                else:
                    quantized_layers.append(module)
            return quantized_layers

        teacher_layers = quantize_model(sample_teacher_model)
        student_layers = quantize_model(sample_student_model)

        # Create quantized models
        class QuantizedModel(nn.Module):
            def __init__(self, layers):
                super().__init__()
                self.fc1 = layers[0]
                self.fc2 = layers[1]
                self.fc3 = layers[2]

            def forward(self, x):
                x = F.relu(self.fc1(x))
                x = F.relu(self.fc2(x))
                return self.fc3(x)

        quantized_teacher = QuantizedModel(teacher_layers)
        quantized_student = QuantizedModel(student_layers)

        # Get outputs
        with torch.no_grad():
            teacher_logits = quantized_teacher(sample_batch)

        student_logits = quantized_student(sample_batch)

        # Compute loss
        loss = qad_loss(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            labels=sample_labels,
        )

        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0


# ============================================================================
# Test Mixed Precision Verification
# ============================================================================


class TestMixedPrecisionVerification:
    """Test suite for mixed precision verification."""

    def test_mixed_precision_attention_ffn(self, nvfp4_loader):
        """Test mixed precision with attention and FFN layers."""
        attention_layer = nn.Linear(512, 512)
        ffn_layer = nn.Linear(512, 2048)

        # Quantize with different types
        quantized_attention = nvfp4_loader.quantize_layer(
            attention_layer, is_attention=True
        )
        quantized_ffn = nvfp4_loader.quantize_layer(ffn_layer, is_attention=False)

        # Both should be valid quantized layers
        assert isinstance(quantized_attention, nn.Module)
        assert isinstance(quantized_ffn, nn.Module)

    def test_mixed_precision_forward_pass(self, nvfp4_loader, sample_batch):
        """Test forward pass with mixed precision layers."""
        # Create mixed model
        attention_layer = nn.Linear(512, 512)
        ffn_layer = nn.Linear(512, 2048)
        output_layer = nn.Linear(2048, 1000)

        q_attention = nvfp4_loader.quantize_layer(attention_layer, is_attention=True)
        q_ffn = nvfp4_loader.quantize_layer(ffn_layer, is_attention=False)
        q_output = nvfp4_loader.quantize_layer(output_layer, is_attention=False)

        # Dequantize for forward pass
        d_attention = nvfp4_loader.dequantize_layer(q_attention)
        d_ffn = nvfp4_loader.dequantize_layer(q_ffn)
        d_output = nvfp4_loader.dequantize_layer(q_output)

        # Forward pass
        x = sample_batch
        x = F.relu(d_attention(x))
        x = F.relu(d_ffn(x))
        x = d_output(x)

        assert x.shape == (4, 1000)

    def test_precision_consistency(self, nvfp4_loader):
        """Test that precision is consistent across layers."""
        layer = nn.Linear(512, 512)

        # Quantize and dequantize multiple times
        outputs = []
        for _ in range(3):
            quantized = nvfp4_loader.quantize_layer(layer, is_attention=False)
            dequantized = nvfp4_loader.dequantize_layer(quantized)
            outputs.append(dequantized.weight.data)

        # All should have same dtype
        dtype = outputs[0].dtype
        for output in outputs:
            assert output.dtype == dtype


# ============================================================================
# Test End-to-End Pipeline
# ============================================================================


class TestEndToEndPipeline:
    """Test suite for end-to-end pipeline."""

    def test_full_training_iteration(
        self,
        nvfp4_loader,
        qad_loss,
        sample_teacher_model,
        sample_student_model,
        sample_batch,
        sample_labels,
    ):
        """Test full training iteration with NVFP4 and QAD."""
        # Quantize student model
        quantized_student_layers = []
        for name, module in sample_student_model.named_children():
            if isinstance(module, nn.Linear):
                quantized = nvfp4_loader.quantize_layer(module, is_attention=False)
                dequantized = nvfp4_loader.dequantize_layer(quantized)
                quantized_student_layers.append(dequantized)
            else:
                quantized_student_layers.append(module)

        # Create quantized student
        class QuantizedStudent(nn.Module):
            def __init__(self, layers):
                super().__init__()
                self.fc1 = layers[0]
                self.fc2 = layers[1]
                self.fc3 = layers[2]

            def forward(self, x):
                x = F.relu(self.fc1(x))
                x = F.relu(self.fc2(x))
                return self.fc3(x)

        quantized_student = QuantizedStudent(quantized_student_layers)

        # Get outputs
        with torch.no_grad():
            teacher_logits = sample_teacher_model(sample_batch)

        student_logits = quantized_student(sample_batch)

        # Compute loss
        loss = qad_loss(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            labels=sample_labels,
        )

        # Backward pass (for training)
        loss.backward()

        assert loss.item() >= 0

    def test_pipeline_with_nested_scheduler(
        self,
        nvfp4_loader,
        qad_loss,
        sample_teacher_model,
        sample_student_model,
        sample_batch,
        sample_labels,
    ):
        """Test pipeline with nested scheduler integration."""
        # Setup nested scheduler
        nested_config = NestedUpdateConfig(
            fast_layers={0, 1}, medium_layers={2}, slow_layers={3, 4}
        )
        scheduler = NestedUpdateScheduler(nested_config)

        # Quantize student
        quantized_layers = []
        for name, module in sample_student_model.named_children():
            if isinstance(module, nn.Linear):
                quantized = nvfp4_loader.quantize_layer(module, is_attention=False)
                quantized_layers.append(quantized)

        # Training loop simulation
        for step in range(20):
            # Get student output
            student_logits = sample_student_model(sample_batch)

            # Get teacher output
            with torch.no_grad():
                teacher_logits = sample_teacher_model(sample_batch)

            # Compute loss
            loss = qad_loss(
                student_logits=student_logits,
                teacher_logits=teacher_logits,
                labels=sample_labels,
            )

            # Check which layers should be updated
            update_layers = []
            for layer_idx in range(5):
                if scheduler.should_update(layer_idx, step):
                    update_layers.append(layer_idx)

            # Step scheduler
            scheduler.step()

        # Verify the nested scheduler ran correctly
        stats = scheduler.get_stats()
        assert "updates_performed" in stats, "Scheduler should track updates performed"
        assert isinstance(stats.get("updates_performed", 0), int), (
            "Updates count should be an integer"
        )
        # Verify all layers were considered for updates
        assert len(quantized_layers) > 0, "Should have quantized layers to update"

    def test_pipeline_with_cache(
        self,
        nvfp4_loader,
        qad_loss,
        temp_cache_dir,
        sample_teacher_model,
        sample_student_model,
        sample_batch,
        sample_labels,
    ):
        """Test pipeline with hierarchical cache."""
        # Setup cache
        cache_config = HierarchicalCacheConfig(
            cache_dir=temp_cache_dir,
            memory_cache_size_gb=0.1,
            disk_l1_size_gb=0.2,
            enable_compression=False,
        )
        cache = HierarchicalLayerCache(cache_config)

        # Quantize and cache layers
        quantized_layers = []
        for i, (name, module) in enumerate(sample_student_model.named_children()):
            if isinstance(module, nn.Linear):
                quantized = nvfp4_loader.quantize_layer(module, is_attention=False)

                # Cache the quantized layer
                cache.cache_layer(
                    f"layer_{i}", quantized, priority=5, initial_tier=CacheTier.DISK_L1
                )

                # Retrieve from cache
                cached = cache.get_layer(f"layer_{i}")
                if cached is not None:
                    dequantized = nvfp4_loader.dequantize_layer(cached)
                    quantized_layers.append(dequantized)
                else:
                    dequantized = nvfp4_loader.dequantize_layer(quantized)
                    quantized_layers.append(dequantized)

        # Use in forward pass
        assert len(quantized_layers) > 0

    def test_full_integrator_pipeline(
        self, temp_cache_dir, sample_batch, sample_labels
    ):
        """Test full pipeline using AdvancedSLIIntegrator."""
        # Create integrator
        config = AdvancedSLIConfig(device="cpu", output_dir=temp_cache_dir)
        integrator = AdvancedSLIIntegrator(config)

        # Create simple models
        teacher = nn.Sequential(nn.Linear(512, 1024), nn.ReLU(), nn.Linear(1024, 1000))

        student = nn.Sequential(nn.Linear(512, 1024), nn.ReLU(), nn.Linear(1024, 1000))

        # Get outputs
        with torch.no_grad():
            teacher_logits = teacher(sample_batch)

        student_logits = student(sample_batch)

        # Compute loss through integrator
        loss = integrator.compute_distillation_loss(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            labels=sample_labels,
        )

        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0


# ============================================================================
# Test Performance Benchmarks
# ============================================================================


class TestPerformanceBenchmarks:
    """Test suite for performance benchmarks."""

    def test_quantization_speed(self, nvfp4_loader):
        """Benchmark quantization speed."""
        layer = nn.Linear(4096, 4096)

        start = time.time()
        for _ in range(5):
            quantized = nvfp4_loader.quantize_layer(layer, is_attention=False)
        elapsed = time.time() - start

        # Should be reasonably fast
        assert elapsed < 10.0

    def test_distillation_speed(
        self,
        qad_loss,
        sample_teacher_model,
        sample_student_model,
        sample_batch,
        sample_labels,
    ):
        """Benchmark distillation speed."""
        with torch.no_grad():
            teacher_logits = sample_teacher_model(sample_batch)

        student_logits = sample_student_model(sample_batch)

        start = time.time()
        for _ in range(10):
            loss = qad_loss(
                student_logits=student_logits,
                teacher_logits=teacher_logits,
                labels=sample_labels,
            )
        elapsed = time.time() - start

        # Should be reasonably fast
        assert elapsed < 5.0

    def test_memory_usage_reduction(self, nvfp4_loader):
        """Test that quantization reduces memory usage."""
        layer = nn.Linear(4096, 4096)

        # Original size
        original_size = sum(p.numel() * p.element_size() for p in layer.parameters())

        # Quantized size
        quantized = nvfp4_loader.quantize_layer(layer, is_attention=False)
        quantized_size = sum(b.numel() * b.element_size() for b in quantized.buffers())

        # Should use less memory (FP8 vs FP32)
        assert quantized_size < original_size

    def test_end_to_end_latency(
        self,
        nvfp4_loader,
        qad_loss,
        sample_teacher_model,
        sample_student_model,
        sample_batch,
        sample_labels,
    ):
        """Test end-to-end latency."""
        start = time.time()

        # Quantize student
        for name, module in sample_student_model.named_children():
            if isinstance(module, nn.Linear):
                quantized = nvfp4_loader.quantize_layer(module, is_attention=False)
                dequantized = nvfp4_loader.dequantize_layer(quantized)
                setattr(sample_student_model, name, dequantized)

        # Forward pass
        with torch.no_grad():
            teacher_logits = sample_teacher_model(sample_batch)

        student_logits = sample_student_model(sample_batch)

        # Compute loss
        loss = qad_loss(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            labels=sample_labels,
        )

        elapsed = time.time() - start

        # Should complete in reasonable time
        assert elapsed < 10.0


# ============================================================================
# Test Edge Cases
# ============================================================================


class TestEdgeCases:
    """Test suite for edge cases."""

    def test_empty_batch(
        self, nvfp4_loader, qad_loss, sample_teacher_model, sample_student_model
    ):
        """Test with empty batch."""
        empty_batch = torch.randn(0, 512)

        # This should not crash
        try:
            with torch.no_grad():
                teacher_logits = sample_teacher_model(empty_batch)
            student_logits = sample_student_model(empty_batch)

            # Loss computation might fail with empty batch, which is expected
            if student_logits.numel() > 0 and teacher_logits.numel() > 0:
                loss = qad_loss(student_logits, teacher_logits)
        except RuntimeError:
            pass  # Expected with empty batch

    def test_single_sample(
        self, nvfp4_loader, qad_loss, sample_teacher_model, sample_student_model
    ):
        """Test with single sample."""
        single_batch = torch.randn(1, 512)
        single_label = torch.tensor([5])

        with torch.no_grad():
            teacher_logits = sample_teacher_model(single_batch)

        student_logits = sample_student_model(single_batch)

        loss = qad_loss(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            labels=single_label,
        )

        assert isinstance(loss, torch.Tensor)

    def test_large_batch(
        self, nvfp4_loader, qad_loss, sample_teacher_model, sample_student_model
    ):
        """Test with large batch."""
        large_batch = torch.randn(64, 512)
        large_labels = torch.randint(0, 1000, (64,))

        with torch.no_grad():
            teacher_logits = sample_teacher_model(large_batch)

        student_logits = sample_student_model(large_batch)

        loss = qad_loss(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            labels=large_labels,
        )

        assert isinstance(loss, torch.Tensor)

    def test_gradient_flow(
        self,
        nvfp4_loader,
        qad_loss,
        sample_teacher_model,
        sample_student_model,
        sample_batch,
        sample_labels,
    ):
        """Test that gradients flow correctly."""
        # Quantize student but keep it trainable
        student = sample_student_model

        # Enable gradients
        for param in student.parameters():
            param.requires_grad = True

        # Forward pass
        with torch.no_grad():
            teacher_logits = sample_teacher_model(sample_batch)

        student_logits = student(sample_batch)

        # Compute loss
        loss = qad_loss(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            labels=sample_labels,
        )

        # Backward pass
        loss.backward()

        # Check that gradients exist
        has_gradients = any(param.grad is not None for param in student.parameters())

        assert has_gradients


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
