"""
tests/unit/test_quantization.py
Comprehensive tests for the quantization module.

Tests cover:
- QuantizationConfig initialization
- All quantization modes (INT8, NF4, FP4, etc.)
- LayerQuantizer functionality
- Adaptive quantization
- QuantizationRegistry
- Graceful degradation when bitsandbytes unavailable
"""

import pytest
import torch
import torch.nn as nn
from unittest.mock import patch, MagicMock

# Import the module under test
from src.nexus_final.sli.quantization import (
    QuantizationMode,
    QuantizationConfig,
    LayerQuantizer,
    AdaptiveQuantizer,
    QuantizationRegistry,
    get_int8_config,
    get_nf4_config,
    get_fp4_config,
    get_mixed_precision_config,
    quantize_layer,
    dequantize_layer,
    BITSANDBYTES_AVAILABLE,
)


class TestQuantizationMode:
    """Test QuantizationMode enum."""

    def test_mode_values(self):
        """Test quantization mode enum values."""
        assert QuantizationMode.NONE.value == "none"
        assert QuantizationMode.INT8.value == "int8"
        assert QuantizationMode.INT8_DYNAMIC.value == "int8_dynamic"
        assert QuantizationMode.NF4.value == "nf4"
        assert QuantizationMode.FP4.value == "fp4"
        assert QuantizationMode.INT4.value == "int4"

    def test_mode_from_string(self):
        """Test creating mode from string."""
        assert QuantizationMode("none") == QuantizationMode.NONE
        assert QuantizationMode("int8") == QuantizationMode.INT8
        assert QuantizationMode("nf4") == QuantizationMode.NF4


class TestQuantizationConfig:
    """Test QuantizationConfig dataclass."""

    def test_default_initialization(self):
        """Test config with default values."""
        config = QuantizationConfig()

        assert config.mode == QuantizationMode.NONE
        assert config.compute_dtype == torch.float16
        assert config.compress_statistics is True
        assert config.quant_storage_dtype == torch.uint8
        assert config.double_quant is True
        assert config.quant_type == "nf4"
        assert config.llm_int8_threshold == 6.0
        assert config.llm_int8_skip_modules is None

    def test_custom_initialization(self):
        """Test config with custom values."""
        config = QuantizationConfig(
            mode=QuantizationMode.INT8,
            compute_dtype=torch.float32,
            compress_statistics=False,
            quant_storage_dtype=torch.int8,
            double_quant=False,
            quant_type="fp4",
            llm_int8_threshold=4.0,
            llm_int8_skip_modules=["lm_head", "embed"]
        )

        assert config.mode == QuantizationMode.INT8
        assert config.compute_dtype == torch.float32
        assert config.compress_statistics is False
        assert config.quant_storage_dtype == torch.int8
        assert config.double_quant is False
        assert config.quant_type == "fp4"
        assert config.llm_int8_threshold == 4.0
        assert config.llm_int8_skip_modules == ["lm_head", "embed"]

    def test_to_dict(self):
        """Test converting config to dictionary."""
        config = QuantizationConfig(
            mode=QuantizationMode.INT8,
            llm_int8_skip_modules=["lm_head"]
        )

        data = config.to_dict()

        assert data['mode'] == "int8"
        assert 'compute_dtype' in data
        assert data['compress_statistics'] is True
        assert data['quant_type'] == "nf4"
        assert data['llm_int8_threshold'] == 6.0
        assert data['llm_int8_skip_modules'] == ["lm_head"]

    def test_from_dict(self):
        """Test creating config from dictionary."""
        data = {
            'mode': 'nf4',
            'compute_dtype': 'torch.float32',
            'compress_statistics': False,
            'double_quant': False,
            'quant_type': 'fp4',
            'llm_int8_threshold': 5.0,
            'llm_int8_skip_modules': ['layer1', 'layer2']
        }

        config = QuantizationConfig.from_dict(data)

        assert config.mode == QuantizationMode.NF4
        assert config.compute_dtype == torch.float32
        assert config.compress_statistics is False
        assert config.double_quant is False
        assert config.quant_type == "fp4"
        assert config.llm_int8_threshold == 5.0
        assert config.llm_int8_skip_modules == ['layer1', 'layer2']

    def test_from_dict_defaults(self):
        """Test from_dict with missing fields uses defaults."""
        data = {'mode': 'int8'}

        config = QuantizationConfig.from_dict(data)

        assert config.mode == QuantizationMode.INT8
        assert config.compute_dtype == torch.float16  # default
        assert config.compress_statistics is True  # default


class TestLayerQuantizer:
    """Test LayerQuantizer class."""

    def test_initialization_default_config(self):
        """Test quantizer with default config."""
        quantizer = LayerQuantizer()

        assert quantizer.config.mode == QuantizationMode.NONE

    def test_initialization_custom_config(self):
        """Test quantizer with custom config."""
        config = QuantizationConfig(mode=QuantizationMode.INT8)
        quantizer = LayerQuantizer(config)

        assert quantizer.config.mode == QuantizationMode.INT8

    def test_quantize_none_mode(self):
        """Test quantization with NONE mode returns layer unchanged."""
        config = QuantizationConfig(mode=QuantizationMode.NONE)
        quantizer = LayerQuantizer(config)

        layer = nn.Linear(100, 100)
        result = quantizer.quantize_layer(layer)

        assert result is layer

    def test_should_skip_layer(self):
        """Test layer skip logic."""
        config = QuantizationConfig(
            mode=QuantizationMode.INT8,
            llm_int8_skip_modules=["lm_head", "embed"]
        )
        quantizer = LayerQuantizer(config)

        assert quantizer._should_skip_layer("model.lm_head") is True
        assert quantizer._should_skip_layer("model.embed_tokens") is True
        assert quantizer._should_skip_layer("model.layers.0") is False

    def test_should_skip_layer_no_config(self):
        """Test layer skip with no skip modules configured."""
        config = QuantizationConfig(mode=QuantizationMode.INT8)
        quantizer = LayerQuantizer(config)

        assert quantizer._should_skip_layer("any_layer") is False

    def test_is_quantized_unquantized_layer(self):
        """Test is_quantized returns False for unquantized layer."""
        quantizer = LayerQuantizer()
        layer = nn.Linear(100, 100)

        assert quantizer.is_quantized(layer) is False

    def test_get_quantized_size_ratio_none(self):
        """Test size ratio for NONE mode."""
        config = QuantizationConfig(mode=QuantizationMode.NONE)
        quantizer = LayerQuantizer(config)
        layer = nn.Linear(100, 100)

        ratio = quantizer.get_quantized_size_ratio(layer)
        assert ratio == 1.0

    def test_get_quantized_size_ratio_int8(self):
        """Test size ratio for INT8 mode."""
        config = QuantizationConfig(mode=QuantizationMode.INT8)
        quantizer = LayerQuantizer(config)
        layer = nn.Linear(100, 100)

        ratio = quantizer.get_quantized_size_ratio(layer)
        assert ratio == 0.5

    def test_get_quantized_size_ratio_nf4(self):
        """Test size ratio for NF4 mode."""
        config = QuantizationConfig(mode=QuantizationMode.NF4)
        quantizer = LayerQuantizer(config)
        layer = nn.Linear(100, 100)

        ratio = quantizer.get_quantized_size_ratio(layer)
        assert ratio == 0.25

    def test_quantize_int8_dynamic(self):
        """Test INT8 dynamic quantization."""
        config = QuantizationConfig(mode=QuantizationMode.INT8_DYNAMIC)
        quantizer = LayerQuantizer(config)

        layer = nn.Linear(100, 100)
        result = quantizer.quantize_layer(layer)

        # Should return a quantized layer
        assert result is not None

    def test_dequantize_unquantized_layer(self):
        """Test dequantizing unquantized layer returns it unchanged."""
        quantizer = LayerQuantizer()
        layer = nn.Linear(100, 100)

        result = quantizer.dequantize_layer(layer)
        assert result is layer


class TestLayerQuantizerWithBitsAndBytes:
    """Test LayerQuantizer with bitsandbytes (if available)."""

    @pytest.mark.skipif(not BITSANDBYTES_AVAILABLE, reason="bitsandbytes not available")
    def test_quantize_int8_with_bitsandbytes(self):
        """Test INT8 quantization with bitsandbytes."""
        from bitsandbytes.nn import Linear8bitLt

        config = QuantizationConfig(mode=QuantizationMode.INT8)
        quantizer = LayerQuantizer(config)

        layer = nn.Linear(100, 100)
        result = quantizer.quantize_layer(layer)

        assert isinstance(result, Linear8bitLt)

    @pytest.mark.skipif(not BITSANDBYTES_AVAILABLE, reason="bitsandbytes not available")
    def test_quantize_nf4_with_bitsandbytes(self):
        """Test NF4 quantization with bitsandbytes."""
        from bitsandbytes.nn import Linear4bit

        config = QuantizationConfig(mode=QuantizationMode.NF4)
        quantizer = LayerQuantizer(config)

        layer = nn.Linear(100, 100)
        result = quantizer.quantize_layer(layer)

        assert isinstance(result, Linear4bit)

    @pytest.mark.skipif(not BITSANDBYTES_AVAILABLE, reason="bitsandbytes not available")
    def test_quantize_fp4_with_bitsandbytes(self):
        """Test FP4 quantization with bitsandbytes."""
        from bitsandbytes.nn import Linear4bit

        config = QuantizationConfig(mode=QuantizationMode.FP4)
        quantizer = LayerQuantizer(config)

        layer = nn.Linear(100, 100)
        result = quantizer.quantize_layer(layer)

        assert isinstance(result, Linear4bit)

    @pytest.mark.skipif(not BITSANDBYTES_AVAILABLE, reason="bitsandbytes not available")
    def test_dequantize_bitsandbytes_layer(self):
        """Test dequantizing bitsandbytes layer."""
        config = QuantizationConfig(mode=QuantizationMode.INT8)
        quantizer = LayerQuantizer(config)

        layer = nn.Linear(100, 100)
        quantized = quantizer.quantize_layer(layer)

        dequantized = quantizer.dequantize_layer(quantized)

        assert isinstance(dequantized, nn.Linear)
        assert dequantized.weight.dtype == torch.float16

    @pytest.mark.skipif(not BITSANDBYTES_AVAILABLE, reason="bitsandbytes not available")
    def test_is_quantized_bitsandbytes_layer(self):
        """Test detecting bitsandbytes quantized layer."""
        from bitsandbytes.nn import Linear8bitLt

        config = QuantizationConfig(mode=QuantizationMode.INT8)
        quantizer = LayerQuantizer(config)

        layer = nn.Linear(100, 100)
        quantized = quantizer.quantize_layer(layer)

        assert quantizer.is_quantized(quantized) is True


class TestLayerQuantizerGracefulDegradation:
    """Test graceful degradation when bitsandbytes unavailable."""

    def test_quantize_int8_without_bitsandbytes(self):
        """Test INT8 quantization falls back when bitsandbytes unavailable."""
        config = QuantizationConfig(mode=QuantizationMode.INT8)
        quantizer = LayerQuantizer(config)

        with patch.object(quantizer, '_check_dependencies'):
            # Force BITSANDBYTES_AVAILABLE to False for this test
            with patch('src.nexus_final.sli.quantization.BITSANDBYTES_AVAILABLE', False):
                layer = nn.Linear(100, 100)
                result = quantizer._quantize_int8(layer)

                # Should return layer unchanged
                assert result is layer

    def test_quantize_nf4_without_bitsandbytes(self):
        """Test NF4 quantization falls back when bitsandbytes unavailable."""
        config = QuantizationConfig(mode=QuantizationMode.NF4)
        quantizer = LayerQuantizer(config)

        with patch('src.nexus_final.sli.quantization.BITSANDBYTES_AVAILABLE', False):
            layer = nn.Linear(100, 100)
            result = quantizer._quantize_4bit(layer)

            # Should return layer unchanged
            assert result is layer

    def test_check_dependencies_raises_error(self):
        """Test dependency check raises error for bitsandbytes modes."""
        config = QuantizationConfig(mode=QuantizationMode.INT8)
        quantizer = LayerQuantizer(config)

        with patch('src.nexus_final.sli.quantization.BITSANDBYTES_AVAILABLE', False):
            with pytest.raises(ImportError) as exc_info:
                quantizer._check_dependencies()

            assert "bitsandbytes" in str(exc_info.value)

    def test_quantize_to_nf4_without_bitsandbytes(self):
        """Test _quantize_to_nf4 returns None when bitsandbytes unavailable."""
        config = QuantizationConfig(mode=QuantizationMode.NF4)
        quantizer = LayerQuantizer(config)

        with patch('src.nexus_final.sli.quantization.BITSANDBYTES_AVAILABLE', False):
            weight = torch.randn(100, 100)
            result = quantizer._quantize_to_nf4(weight)

            assert result is None


class TestAdaptiveQuantizer:
    """Test AdaptiveQuantizer class."""

    def test_initialization_default(self):
        """Test adaptive quantizer with default configs."""
        base_config = QuantizationConfig(mode=QuantizationMode.NF4)
        quantizer = AdaptiveQuantizer(base_config)

        assert quantizer.base_config == base_config
        assert quantizer.attention_config is not None
        assert quantizer.ffn_config == base_config

    def test_initialization_custom(self):
        """Test adaptive quantizer with custom configs."""
        base = QuantizationConfig(mode=QuantizationMode.NF4)
        attention = QuantizationConfig(mode=QuantizationMode.INT8)
        ffn = QuantizationConfig(mode=QuantizationMode.FP4)

        quantizer = AdaptiveQuantizer(
            base_config=base,
            attention_config=attention,
            ffn_config=ffn
        )

        assert quantizer.base_config == base
        assert quantizer.attention_config == attention
        assert quantizer.ffn_config == ffn

    def test_higher_precision_config_nf4(self):
        """Test upgrading NF4 to higher precision."""
        base = QuantizationConfig(mode=QuantizationMode.NF4)
        quantizer = AdaptiveQuantizer(base)

        higher = quantizer._higher_precision_config(base)

        assert higher.mode == QuantizationMode.INT8

    def test_higher_precision_config_int8(self):
        """Test upgrading INT8 to higher precision."""
        base = QuantizationConfig(mode=QuantizationMode.INT8)
        quantizer = AdaptiveQuantizer(base)

        higher = quantizer._higher_precision_config(base)

        assert higher.mode == QuantizationMode.INT8_DYNAMIC

    def test_higher_precision_config_int8_dynamic(self):
        """Test upgrading INT8_DYNAMIC to higher precision."""
        base = QuantizationConfig(mode=QuantizationMode.INT8_DYNAMIC)
        quantizer = AdaptiveQuantizer(base)

        higher = quantizer._higher_precision_config(base)

        assert higher.mode == QuantizationMode.NONE

    def test_get_config_for_layer_attention(self):
        """Test getting config for attention layer."""
        base = QuantizationConfig(mode=QuantizationMode.NF4)
        attention = QuantizationConfig(mode=QuantizationMode.INT8)
        quantizer = AdaptiveQuantizer(base, attention_config=attention)

        config = quantizer._get_config_for_layer("model.layers.0.attn")
        assert config.mode == QuantizationMode.INT8

        config = quantizer._get_config_for_layer("model.layers.0.query")
        assert config.mode == QuantizationMode.INT8

    def test_get_config_for_layer_ffn(self):
        """Test getting config for FFN layer."""
        base = QuantizationConfig(mode=QuantizationMode.NF4)
        ffn = QuantizationConfig(mode=QuantizationMode.FP4)
        quantizer = AdaptiveQuantizer(base, ffn_config=ffn)

        config = quantizer._get_config_for_layer("model.layers.0.ffn")
        assert config.mode == QuantizationMode.FP4

        config = quantizer._get_config_for_layer("model.layers.0.mlp")
        assert config.mode == QuantizationMode.FP4

    def test_get_config_for_layer_default(self):
        """Test getting config for regular layer."""
        base = QuantizationConfig(mode=QuantizationMode.NF4)
        quantizer = AdaptiveQuantizer(base)

        config = quantizer._get_config_for_layer("model.layers.0.norm")
        assert config.mode == QuantizationMode.NF4


class TestQuantizationRegistry:
    """Test QuantizationRegistry class."""

    def test_register_config(self):
        """Test registering a quantization config."""
        config = QuantizationConfig(mode=QuantizationMode.INT8)

        QuantizationRegistry.register("test_int8", config)

        assert "test_int8" in QuantizationRegistry.list_configs()

    def test_get_config(self):
        """Test retrieving a registered config."""
        config = QuantizationConfig(mode=QuantizationMode.NF4)

        QuantizationRegistry.register("test_nf4", config)
        retrieved = QuantizationRegistry.get("test_nf4")

        assert retrieved is config

    def test_get_nonexistent_config(self):
        """Test retrieving non-existent config returns None."""
        result = QuantizationRegistry.get("nonexistent_config_12345")

        assert result is None

    def test_list_configs(self):
        """Test listing registered configs."""
        # Clear existing configs first
        QuantizationRegistry._configs.clear()

        QuantizationRegistry.register("config1", QuantizationConfig())
        QuantizationRegistry.register("config2", QuantizationConfig())

        configs = QuantizationRegistry.list_configs()

        assert "config1" in configs
        assert "config2" in configs


class TestPredefinedConfigs:
    """Test predefined quantization configurations."""

    def test_get_int8_config(self):
        """Test predefined INT8 config."""
        config = get_int8_config()

        assert config.mode == QuantizationMode.INT8
        assert config.compute_dtype == torch.float16
        assert config.llm_int8_threshold == 6.0
        assert "lm_head" in config.llm_int8_skip_modules
        assert "embed_tokens" in config.llm_int8_skip_modules

    def test_get_nf4_config(self):
        """Test predefined NF4 config."""
        config = get_nf4_config()

        assert config.mode == QuantizationMode.NF4
        assert config.compute_dtype == torch.float16
        assert config.compress_statistics is True
        assert config.double_quant is True
        assert config.quant_type == "nf4"

    def test_get_fp4_config(self):
        """Test predefined FP4 config."""
        config = get_fp4_config()

        assert config.mode == QuantizationMode.FP4
        assert config.compute_dtype == torch.float16
        assert config.compress_statistics is True
        assert config.double_quant is True
        assert config.quant_type == "fp4"

    def test_get_mixed_precision_config(self):
        """Test mixed precision adaptive quantizer."""
        quantizer = get_mixed_precision_config()

        assert isinstance(quantizer, AdaptiveQuantizer)
        assert quantizer.base_config.mode == QuantizationMode.NF4
        assert quantizer.attention_config.mode == QuantizationMode.INT8
        assert quantizer.ffn_config.mode == QuantizationMode.NF4


class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_quantize_layer_with_string_mode(self):
        """Test quantize_layer with string mode."""
        layer = nn.Linear(100, 100)

        # Should not raise
        result = quantize_layer(layer, mode="int8_dynamic")

        assert result is not None

    def test_quantize_layer_with_enum_mode(self):
        """Test quantize_layer with enum mode."""
        layer = nn.Linear(100, 100)

        result = quantize_layer(layer, mode=QuantizationMode.INT8_DYNAMIC)

        assert result is not None

    def test_dequantize_layer(self):
        """Test dequantize_layer convenience function."""
        layer = nn.Linear(100, 100)

        result = dequantize_layer(layer, compute_dtype=torch.float32)

        assert result is not None


class TestQuantizationEdgeCases:
    """Test edge cases."""

    def test_quantize_layer_error_handling(self):
        """Test quantizer handles errors gracefully."""
        config = QuantizationConfig(mode=QuantizationMode.INT8)
        quantizer = LayerQuantizer(config)

        # Mock _quantize_int8 to raise exception
        with patch.object(quantizer, '_quantize_int8', side_effect=Exception("Test error")):
            layer = nn.Linear(100, 100)
            result = quantizer.quantize_layer(layer, "test_layer")

            # Should return original layer on error
            assert result is layer

    def test_quantize_complex_module(self):
        """Test quantizing complex module with submodules."""
        config = QuantizationConfig(mode=QuantizationMode.INT8_DYNAMIC)
        quantizer = LayerQuantizer(config)

        class ComplexModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = nn.Linear(100, 200)
                self.linear2 = nn.Linear(200, 100)
                self.norm = nn.LayerNorm(100)

            def forward(self, x):
                return self.linear2(self.linear1(x))

        module = ComplexModule()
        result = quantizer.quantize_layer(module)

        assert result is not None

    def test_quantize_non_linear_layer(self):
        """Test quantizing non-linear layer."""
        config = QuantizationConfig(mode=QuantizationMode.INT8_DYNAMIC)
        quantizer = LayerQuantizer(config)

        layer = nn.LayerNorm(100)
        result = quantizer.quantize_layer(layer)

        # LayerNorm doesn't support quantization, should return unchanged
        assert result is layer

    def test_skip_empty_layer_name(self):
        """Test skip check with empty layer name."""
        config = QuantizationConfig(
            mode=QuantizationMode.INT8,
            llm_int8_skip_modules=["lm_head"]
        )
        quantizer = LayerQuantizer(config)

        assert quantizer._should_skip_layer("") is False

    def test_config_immutability(self):
        """Test that config changes don't affect other instances."""
        config1 = QuantizationConfig(mode=QuantizationMode.INT8)
        config2 = QuantizationConfig.from_dict(config1.to_dict())

        config2.mode = QuantizationMode.NF4

        assert config1.mode == QuantizationMode.INT8
        assert config2.mode == QuantizationMode.NF4
