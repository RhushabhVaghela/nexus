"""
Integration tests for Advanced SLI Integrator.

Tests cover:
- AdvancedSLIIntegrator end-to-end
- Configuration presets (fast/balanced/quality)
- Full pipeline with all components
- Performance benchmarks within tests
"""

import pytest
import torch
import torch.nn as nn
import time
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

# Import the module under test
from src.nexus.models.sli.advanced_sli_integrator import (
    AdvancedSLIIntegrator,
    AdvancedSLIConfig,
    LayerInfo,
    AdvancedSLIError,
    create_advanced_integrator,
)
from src.nexus.models.sli.nvfp4_loader import (
    NVFP4Config,
    NVFP4Mode,
    NVFP4QuantizationError,
)
from src.nexus.models.sli.qad_loss import (
    QADLossConfig,
    QADLossType,
)
from src.nexus.models.sli.nested_scheduler import (
    NestedUpdateConfig,
    UpdateGroup,
)
from src.nexus.models.sli.hierarchical_cache import (
    HierarchicalCacheConfig,
    CacheTier,
)
from src.nexus.models.sli.exceptions import SLIError


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def temp_output_dir(tmp_path):
    """Create a temporary output directory."""
    output_dir = tmp_path / "advanced_sli_output"
    return str(output_dir)


@pytest.fixture
def default_config(temp_output_dir):
    """Create default advanced SLI config."""
    return AdvancedSLIConfig(
        device="cpu",
        output_dir=temp_output_dir
    )


@pytest.fixture
def integrator_default(temp_output_dir):
    """Create default advanced SLI integrator."""
    config = AdvancedSLIConfig(device="cpu", output_dir=temp_output_dir)
    return AdvancedSLIIntegrator(config)


@pytest.fixture
def sample_layer():
    """Create a sample layer for testing."""
    return nn.Linear(512, 256)


@pytest.fixture
def sample_student_logits():
    """Create sample student logits."""
    torch.manual_seed(42)
    return torch.randn(4, 1000)


@pytest.fixture
def sample_teacher_logits():
    """Create sample teacher logits."""
    torch.manual_seed(43)
    return torch.randn(4, 1000)


@pytest.fixture
def sample_labels():
    """Create sample labels."""
    torch.manual_seed(44)
    return torch.randint(0, 1000, (4,))


# ============================================================================
# Test AdvancedSLIConfig
# ============================================================================

class TestAdvancedSLIConfig:
    """Test suite for AdvancedSLIConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = AdvancedSLIConfig()
        
        assert config.enable_quantization is True
        assert config.enable_distillation is True
        assert config.enable_nested_updates is True
        assert config.enable_hierarchical_cache is True
        assert config.device == "cuda"
        assert config.output_dir == "./advanced_sli_output"
        
        # Should have default sub-configs
        assert config.nvfp4_config is not None
        assert config.qad_config is not None
        assert config.nested_config is not None
        assert config.cache_config is not None

    def test_post_init_creates_defaults(self):
        """Test that __post_init__ creates default configs."""
        config = AdvancedSLIConfig(
            nvfp4_config=None,
            qad_config=None,
            nested_config=None,
            cache_config=None
        )
        
        assert config.nvfp4_config is not None
        assert config.qad_config is not None
        assert config.nested_config is not None
        assert config.cache_config is not None

    def test_custom_config(self):
        """Test custom configuration values."""
        config = AdvancedSLIConfig(
            enable_quantization=False,
            enable_distillation=False,
            enable_nested_updates=False,
            enable_hierarchical_cache=False,
            device="cpu",
            output_dir="/custom/output"
        )
        
        assert config.enable_quantization is False
        assert config.enable_distillation is False
        assert config.enable_nested_updates is False
        assert config.enable_hierarchical_cache is False
        assert config.device == "cpu"
        assert config.output_dir == "/custom/output"

    def test_config_to_dict(self):
        """Test configuration serialization to dict."""
        config = AdvancedSLIConfig(
            enable_quantization=True,
            device="cpu"
        )
        config_dict = config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert config_dict['enable_quantization'] is True
        assert config_dict['enable_distillation'] is True
        assert config_dict['device'] == "cpu"
        assert config_dict['nvfp4_config'] is not None

    def test_config_from_dict(self):
        """Test configuration deserialization from dict."""
        data = {
            'enable_quantization': False,
            'enable_distillation': False,
            'enable_nested_updates': True,
            'enable_hierarchical_cache': True,
            'device': 'cpu',
            'output_dir': './test_output',
            'nvfp4_config': None,
            'qad_config': None,
            'nested_config': None,
            'cache_config': None
        }
        
        config = AdvancedSLIConfig.from_dict(data)
        
        assert config.enable_quantization is False
        assert config.enable_distillation is False
        assert config.device == "cpu"
        assert config.output_dir == "./test_output"


# ============================================================================
# Test AdvancedSLIIntegrator Initialization
# ============================================================================

class TestAdvancedSLIIntegratorInitialization:
    """Test suite for integrator initialization."""

    def test_initialization_all_enabled(self, temp_output_dir):
        """Test initialization with all features enabled."""
        config = AdvancedSLIConfig(
            enable_quantization=True,
            enable_distillation=True,
            enable_nested_updates=True,
            enable_hierarchical_cache=True,
            device="cpu",
            output_dir=temp_output_dir
        )
        
        integrator = AdvancedSLIIntegrator(config)
        
        assert integrator.config == config
        assert integrator.nvfp4_loader is not None
        assert integrator.qad_loss is not None
        assert integrator.nested_scheduler is not None
        assert integrator.hierarchical_cache is not None

    def test_initialization_quantization_disabled(self, temp_output_dir):
        """Test initialization with quantization disabled."""
        config = AdvancedSLIConfig(
            enable_quantization=False,
            enable_distillation=True,
            device="cpu",
            output_dir=temp_output_dir
        )
        
        integrator = AdvancedSLIIntegrator(config)
        
        assert integrator.nvfp4_loader is None
        assert integrator.qad_loss is not None

    def test_initialization_distillation_disabled(self, temp_output_dir):
        """Test initialization with distillation disabled."""
        config = AdvancedSLIConfig(
            enable_quantization=True,
            enable_distillation=False,
            device="cpu",
            output_dir=temp_output_dir
        )
        
        integrator = AdvancedSLIIntegrator(config)
        
        assert integrator.nvfp4_loader is not None
        assert integrator.qad_loss is None

    def test_initialization_all_disabled(self, temp_output_dir):
        """Test initialization with all features disabled."""
        config = AdvancedSLIConfig(
            enable_quantization=False,
            enable_distillation=False,
            enable_nested_updates=False,
            enable_hierarchical_cache=False,
            device="cpu",
            output_dir=temp_output_dir
        )
        
        integrator = AdvancedSLIIntegrator(config)
        
        assert integrator.nvfp4_loader is None
        assert integrator.qad_loss is None
        assert integrator.nested_scheduler is None
        assert integrator.hierarchical_cache is None

    def test_output_dir_created(self, temp_output_dir):
        """Test that output directory is created."""
        config = AdvancedSLIConfig(
            device="cpu",
            output_dir=temp_output_dir
        )
        
        integrator = AdvancedSLIIntegrator(config)
        
        assert Path(temp_output_dir).exists()


# ============================================================================
# Test Configuration Presets
# ============================================================================

class TestConfigurationPresets:
    """Test suite for configuration presets."""

    def test_fast_preset(self):
        """Test 'fast' preset configuration."""
        integrator = create_advanced_integrator(mode="fast", device="cpu")
        
        assert integrator.config.nvfp4_config.mode == NVFP4Mode.SOFTWARE
        assert integrator.config.qad_config.temperature == 2.0
        assert integrator.config.qad_config.alpha == 0.5
        assert integrator.config.nested_config.medium_interval == 20
        assert integrator.config.nested_config.slow_interval == 200

    def test_balanced_preset(self):
        """Test 'balanced' preset configuration."""
        integrator = create_advanced_integrator(mode="balanced", device="cpu")
        
        assert integrator.config.nvfp4_config.mode == NVFP4Mode.MIXED
        assert integrator.config.qad_config.temperature == 1.5
        assert integrator.config.qad_config.alpha == 0.7

    def test_quality_preset(self):
        """Test 'quality' preset configuration."""
        integrator = create_advanced_integrator(mode="quality", device="cpu")
        
        assert integrator.config.nvfp4_config.mode == NVFP4Mode.MIXED
        assert integrator.config.qad_config.temperature == 1.0
        assert integrator.config.qad_config.alpha == 0.9
        assert integrator.config.nested_config.medium_interval == 5
        assert integrator.config.nested_config.slow_interval == 50

    def test_preset_additional_kwargs(self):
        """Test preset with additional kwargs."""
        integrator = create_advanced_integrator(
            mode="balanced",
            device="cpu",
            enable_quantization=False
        )
        
        assert integrator.config.enable_quantization is False


# ============================================================================
# Test Load Layer Pipeline
# ============================================================================

class TestLoadLayerPipeline:
    """Test suite for load_layer pipeline."""

    def test_load_layer_with_quantization(self, integrator_default, sample_layer):
        """Test loading layer with quantization."""
        # Create a layer with weights
        weights = {
            "weight": sample_layer.weight.data,
            "bias": sample_layer.bias.data
        }
        
        layer = integrator_default.load_layer(
            model_id="test_model",
            layer_idx=0,
            layer_weights=weights,
            is_attention=False
        )
        
        assert layer is not None
        assert isinstance(layer, nn.Module)

    def test_load_layer_from_cache(self, integrator_default, sample_layer):
        """Test loading layer from hierarchical cache."""
        # First load to cache it
        weights = {
            "weight": sample_layer.weight.data,
            "bias": sample_layer.bias.data
        }
        
        layer1 = integrator_default.load_layer(
            "test_model",
            0,
            layer_weights=weights
        )
        
        # Second load should come from cache
        layer2 = integrator_default.load_layer(
            "test_model",
            0
        )
        
        assert layer2 is not None

    def test_load_layer_without_quantization(self, temp_output_dir):
        """Test loading layer without quantization."""
        config = AdvancedSLIConfig(
            enable_quantization=False,
            device="cpu",
            output_dir=temp_output_dir
        )
        integrator = AdvancedSLIIntegrator(config)
        
        weights = {
            "weight": torch.randn(256, 512),
            "bias": torch.randn(256)
        }
        
        layer = integrator.load_layer(
            "test_model",
            0,
            layer_weights=weights
        )
        
        assert layer is not None


# ============================================================================
# Test Quantize/Dequantize Operations
# ============================================================================

class TestQuantizeDequantizeOperations:
    """Test suite for quantize/dequantize operations."""

    def test_quantize_layer(self, integrator_default, sample_layer):
        """Test quantizing a layer."""
        quantized = integrator_default.quantize_layer(sample_layer, is_attention=False)
        
        assert quantized is not None
        assert isinstance(quantized, nn.Module)

    def test_dequantize_layer(self, integrator_default, sample_layer):
        """Test dequantizing a layer."""
        quantized = integrator_default.quantize_layer(sample_layer, is_attention=False)
        dequantized = integrator_default.dequantize_layer(quantized)
        
        assert dequantized is not None
        assert isinstance(dequantized, nn.Module)
        assert hasattr(dequantized, "weight")

    def test_quantize_without_loader(self, temp_output_dir):
        """Test that quantize without loader raises error."""
        config = AdvancedSLIConfig(
            enable_quantization=False,
            device="cpu",
            output_dir=temp_output_dir
        )
        integrator = AdvancedSLIIntegrator(config)
        
        with pytest.raises(AdvancedSLIError):
            integrator.quantize_layer(nn.Linear(100, 100))

    def test_dequantize_without_loader(self, temp_output_dir):
        """Test that dequantize without loader raises error."""
        config = AdvancedSLIConfig(
            enable_quantization=False,
            device="cpu",
            output_dir=temp_output_dir
        )
        integrator = AdvancedSLIIntegrator(config)
        
        with pytest.raises(AdvancedSLIError):
            integrator.dequantize_layer(nn.Linear(100, 100))


# ============================================================================
# Test Distillation Loss
# ============================================================================

class TestDistillationLoss:
    """Test suite for distillation loss computation."""

    def test_compute_distillation_loss(self, integrator_default, sample_student_logits, sample_teacher_logits, sample_labels):
        """Test computing distillation loss."""
        loss = integrator_default.compute_distillation_loss(
            student_logits=sample_student_logits,
            teacher_logits=sample_teacher_logits,
            labels=sample_labels
        )
        
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert loss.item() >= 0

    def test_compute_distillation_loss_with_hidden(self, integrator_default, sample_student_logits, sample_teacher_logits, sample_labels):
        """Test computing distillation loss with hidden states."""
        hidden_student = torch.randn(4, 128, 768)
        hidden_teacher = torch.randn(4, 128, 768)
        
        loss = integrator_default.compute_distillation_loss(
            student_logits=sample_student_logits,
            teacher_logits=sample_teacher_logits,
            labels=sample_labels,
            hidden_student=hidden_student,
            hidden_teacher=hidden_teacher
        )
        
        assert isinstance(loss, torch.Tensor)

    def test_compute_distillation_loss_without_loss_module(self, temp_output_dir):
        """Test that computing loss without module raises error."""
        config = AdvancedSLIConfig(
            enable_distillation=False,
            device="cpu",
            output_dir=temp_output_dir
        )
        integrator = AdvancedSLIIntegrator(config)
        
        with pytest.raises(AdvancedSLIError):
            integrator.compute_distillation_loss(
                torch.randn(4, 100),
                torch.randn(4, 100)
            )


# ============================================================================
# Test Nested Scheduler Integration
# ============================================================================

class TestNestedSchedulerIntegration:
    """Test suite for nested scheduler integration."""

    def test_should_update_with_scheduler(self, integrator_default):
        """Test should_update with nested scheduler."""
        # Setup scheduler
        integrator_default.nested_scheduler = MagicMock()
        integrator_default.nested_scheduler.should_update.return_value = True
        
        result = integrator_default.should_update(0, step=0)
        
        assert result is True

    def test_should_update_disabled(self, temp_output_dir):
        """Test should_update when nested updates disabled."""
        config = AdvancedSLIConfig(
            enable_nested_updates=False,
            device="cpu",
            output_dir=temp_output_dir
        )
        integrator = AdvancedSLIIntegrator(config)
        
        # Should always return True when disabled
        result = integrator.should_update(0, step=0)
        assert result is True

    def test_get_update_layers(self, integrator_default):
        """Test getting update layers."""
        # Setup scheduler
        integrator_default.nested_scheduler = MagicMock()
        integrator_default.nested_scheduler.get_update_layers.return_value = [0, 1, 2]
        
        layers = integrator_default.get_update_layers(step=0)
        
        assert layers == [0, 1, 2]

    def test_step_scheduler(self, integrator_default):
        """Test stepping the scheduler."""
        integrator_default.nested_scheduler = MagicMock()
        
        integrator_default.step_scheduler()
        
        integrator_default.nested_scheduler.step.assert_called_once()


# ============================================================================
# Test Cache Operations
# ============================================================================

class TestCacheOperations:
    """Test suite for cache operations."""

    def test_prefetch_layers(self, integrator_default):
        """Test prefetching layers."""
        integrator_default.hierarchical_cache = MagicMock()
        
        integrator_default.prefetch_layers("test_model", [0, 1, 2])
        
        integrator_default.hierarchical_cache.prefetch_layers.assert_called_once()

    def test_prefetch_without_cache(self, temp_output_dir):
        """Test prefetch without cache enabled."""
        config = AdvancedSLIConfig(
            enable_hierarchical_cache=False,
            device="cpu",
            output_dir=temp_output_dir
        )
        integrator = AdvancedSLIIntegrator(config)
        
        # Should not raise error
        integrator.prefetch_layers("test_model", [0, 1, 2])

    def test_get_layer_info(self, integrator_default):
        """Test getting layer info."""
        # Create mock entry
        mock_entry = MagicMock()
        mock_entry.tier = CacheTier.MEMORY
        mock_entry.size_bytes = 1024
        
        integrator_default.hierarchical_cache = MagicMock()
        integrator_default.hierarchical_cache._entries = {"test_model_layer_0": mock_entry}
        
        info = integrator_default.get_layer_info("test_model_layer_0")
        
        assert info is not None
        assert info.is_quantized is True
        assert info.tier == "memory"
        assert info.size_bytes == 1024

    def test_clear_cache(self, integrator_default):
        """Test clearing cache."""
        integrator_default.hierarchical_cache = MagicMock()
        integrator_default.nvfp4_loader = MagicMock()
        
        integrator_default.clear_cache()
        
        integrator_default.hierarchical_cache.clear.assert_called_once()
        integrator_default.nvfp4_loader.clear_cache.assert_called_once()


# ============================================================================
# Test Statistics and Reporting
# ============================================================================

class TestStatisticsAndReporting:
    """Test suite for statistics and reporting."""

    def test_get_stats(self, integrator_default):
        """Test getting integrator statistics."""
        # Setup mock components
        integrator_default.nvfp4_loader = MagicMock()
        integrator_default.nvfp4_loader.get_stats.return_value = {"layers_loaded": 10}
        
        integrator_default.qad_loss = MagicMock()
        integrator_default.qad_loss.get_stats.return_value = {"total_loss": 1.5}
        
        stats = integrator_default.get_stats()
        
        assert isinstance(stats, dict)
        assert "layers_loaded" in stats
        assert "nvfp4" in stats
        assert "qad" in stats

    def test_get_stats_partial_components(self, integrator_default):
        """Test getting stats with some components disabled."""
        integrator_default.nvfp4_loader = None
        integrator_default.qad_loss = MagicMock()
        integrator_default.qad_loss.get_stats.return_value = {"total_loss": 1.5}
        
        stats = integrator_default.get_stats()
        
        assert "qad" in stats
        assert "nvfp4" not in stats

    def test_save_config(self, integrator_default, temp_output_dir):
        """Test saving configuration."""
        integrator_default.save_config()
        
        config_path = Path(temp_output_dir) / "config.json"
        assert config_path.exists()
        
        # Verify it's valid JSON
        with open(config_path) as f:
            data = json.load(f)
            assert "enable_quantization" in data

    def test_save_config_custom_path(self, integrator_default, tmp_path):
        """Test saving config to custom path."""
        custom_path = tmp_path / "custom_config.json"
        
        integrator_default.save_config(str(custom_path))
        
        assert custom_path.exists()

    def test_export_model_profile(self, integrator_default):
        """Test exporting model profile."""
        integrator_default.nested_scheduler = MagicMock()
        integrator_default.nested_scheduler.get_group.return_value = UpdateGroup.FAST
        
        profile = integrator_default.export_model_profile("test_model", num_layers=5)
        
        assert isinstance(profile, dict)
        assert profile['model_id'] == "test_model"
        assert profile['num_layers'] == 5
        assert 'config' in profile
        assert 'layer_groups' in profile


# ============================================================================
# Test Inference Pipeline
# ============================================================================

class TestInferencePipeline:
    """Test suite for inference pipeline."""

    def test_run_inference_pipeline(self, integrator_default):
        """Test running full inference pipeline."""
        input_tensor = torch.randn(2, 512)
        
        def layer_factory(idx):
            return nn.Linear(512, 512)
        
        # Mock load_layer to return simple layers
        integrator_default.load_layer = MagicMock()
        integrator_default.load_layer.return_value = nn.Linear(512, 512)
        
        output = integrator_default.run_inference_pipeline(
            model_id="test_model",
            input_tensor=input_tensor,
            num_layers=3,
            layer_factory=layer_factory
        )
        
        assert output is not None
        assert isinstance(output, torch.Tensor)

    def test_run_inference_pipeline_prefetch(self, integrator_default):
        """Test that inference pipeline prefetches layers."""
        integrator_default.prefetch_layers = MagicMock()
        integrator_default.load_layer = MagicMock()
        integrator_default.load_layer.return_value = nn.Linear(512, 512)
        
        input_tensor = torch.randn(2, 512)
        
        integrator_default.run_inference_pipeline(
            model_id="test_model",
            input_tensor=input_tensor,
            num_layers=3,
            layer_factory=lambda x: nn.Linear(512, 512)
        )
        
        # Prefetch should have been called
        assert integrator_default.prefetch_layers.called


# ============================================================================
# Test End-to-End Integration
# ============================================================================

class TestEndToEndIntegration:
    """Test suite for end-to-end integration."""

    def test_full_training_workflow(self, integrator_default):
        """Test full training workflow."""
        batch_size = 4
        num_classes = 1000
        
        # Simulate training steps
        for step in range(5):
            # Generate random data
            student_logits = torch.randn(batch_size, num_classes)
            teacher_logits = torch.randn(batch_size, num_classes)
            labels = torch.randint(0, num_classes, (batch_size,))
            
            # Compute loss
            loss = integrator_default.compute_distillation_loss(
                student_logits=student_logits,
                teacher_logits=teacher_logits,
                labels=labels
            )
            
            assert isinstance(loss, torch.Tensor)
            assert loss.item() >= 0
            
            # Step scheduler
            integrator_default.step_scheduler()

    def test_layer_loading_and_caching_workflow(self, integrator_default):
        """Test layer loading and caching workflow."""
        model_id = "test_model"
        
        for layer_idx in range(5):
            # Create layer weights
            weights = {
                "weight": torch.randn(256, 512),
                "bias": torch.randn(256)
            }
            
            # Load layer
            layer = integrator_default.load_layer(
                model_id=model_id,
                layer_idx=layer_idx,
                layer_weights=weights,
                is_attention=(layer_idx == 0)
            )
            
            assert layer is not None
            
            # Check if layer should be updated
            should_update = integrator_default.should_update(layer_idx, step=0)
            assert isinstance(should_update, bool)

    def test_quantized_inference_workflow(self, integrator_default):
        """Test quantized inference workflow."""
        # Create and quantize layers
        layers = []
        for i in range(3):
            layer = nn.Linear(512, 512)
            quantized = integrator_default.quantize_layer(layer, is_attention=(i == 0))
            layers.append(quantized)
        
        # Run inference
        x = torch.randn(2, 512)
        for layer in layers:
            # Dequantize for inference
            dequantized = integrator_default.dequantize_layer(layer)
            x = dequantized(x)
        
        assert x.shape == (2, 512)

    def test_performance_benchmark(self, integrator_default):
        """Test performance benchmark."""
        start_time = time.time()
        
        # Run operations
        for _ in range(10):
            layer = nn.Linear(256, 256)
            weights = {"weight": layer.weight.data, "bias": layer.bias.data}
            
            loaded = integrator_default.load_layer("test", 0, weights)
            quantized = integrator_default.quantize_layer(loaded, is_attention=False)
            dequantized = integrator_default.dequantize_layer(quantized)
        
        elapsed = time.time() - start_time
        
        # Should complete in reasonable time
        assert elapsed < 30.0
        
        # Get stats
        stats = integrator_default.get_stats()
        assert "layers_loaded" in stats


# ============================================================================
# Test Error Handling
# ============================================================================

class TestErrorHandling:
    """Test suite for error handling."""

    def test_advanced_sli_error(self):
        """Test AdvancedSLIError creation."""
        error = AdvancedSLIError("Test error message")
        
        assert "Test error message" in str(error)
        assert isinstance(error, SLIError)

    def test_layer_info_creation(self):
        """Test LayerInfo dataclass."""
        info = LayerInfo(
            layer_idx=0,
            is_quantized=True,
            tier="memory",
            size_bytes=1024,
            load_time_ms=10.0
        )
        
        assert info.layer_idx == 0
        assert info.is_quantized is True
        assert info.tier == "memory"
        assert info.size_bytes == 1024
        assert info.load_time_ms == 10.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
