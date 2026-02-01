"""
tests/unit/test_custom_layer_registry.py
Comprehensive tests for custom layer registry methods.

Tests cover:
- register_custom_layer()
- get_layer_factory()
- unregister_custom_layer()
- list_custom_layers()
- clear_custom_layers()
- Error handling for duplicates
"""

import pytest
from unittest.mock import MagicMock

import torch
import torch.nn as nn

# Import the module under test
from src.nexus_final.sli.architecture_registry import ArchitectureRegistry, get_registry


class TestArchitectureRegistrySingleton:
    """Test ArchitectureRegistry singleton behavior."""

    def test_singleton_instance(self):
        """Test that registry is a singleton."""
        # Reset singleton state
        ArchitectureRegistry._instance = None
        ArchitectureRegistry._initialized = False

        registry1 = ArchitectureRegistry()
        registry2 = ArchitectureRegistry()

        assert registry1 is registry2

    def test_singleton_same_families(self):
        """Test that singleton has same families."""
        # Reset singleton state
        ArchitectureRegistry._instance = None
        ArchitectureRegistry._initialized = False

        registry1 = ArchitectureRegistry()
        registry2 = ArchitectureRegistry()

        assert registry1._families is registry2._families
        assert registry1._custom_layers is registry2._custom_layers

    def test_get_registry_function(self):
        """Test get_registry convenience function."""
        # Reset global registry
        import src.nexus_final.sli.architecture_registry as reg_module
        reg_module._registry = None

        registry = get_registry()

        assert isinstance(registry, ArchitectureRegistry)


class TestRegisterCustomLayer:
    """Test register_custom_layer() method."""

    @pytest.fixture
    def registry(self):
        """Create a fresh registry for each test."""
        # Reset singleton state
        ArchitectureRegistry._instance = None
        ArchitectureRegistry._initialized = False
        reg = ArchitectureRegistry()
        # Clear custom layers for clean test
        reg._custom_layers.clear()
        return reg

    def test_register_simple_factory(self, registry):
        """Test registering a simple layer factory."""
        def factory(config, layer_idx):
            return nn.Linear(100, 100)

        registry.register_custom_layer("simple_linear", factory)

        assert "simple_linear" in registry._custom_layers
        assert registry._custom_layers["simple_linear"] is factory

    def test_register_class_factory(self, registry):
        """Test registering a class as factory."""
        class CustomLayer(nn.Module):
            def __init__(self, config, layer_idx):
                super().__init__()
                self.linear = nn.Linear(100, 100)

        registry.register_custom_layer("custom_layer_class", CustomLayer)

        assert "custom_layer_class" in registry._custom_layers

    def test_register_lambda_factory(self, registry):
        """Test registering a lambda factory."""
        factory = lambda config, idx: nn.Linear(100, 100)

        registry.register_custom_layer("lambda_layer", factory)

        assert "lambda_layer" in registry._custom_layers

    def test_register_multiple_layers(self, registry):
        """Test registering multiple custom layers."""
        registry.register_custom_layer("layer1", lambda c, i: nn.Linear(10, 10))
        registry.register_custom_layer("layer2", lambda c, i: nn.Linear(20, 20))
        registry.register_custom_layer("layer3", lambda c, i: nn.Linear(30, 30))

        assert len(registry._custom_layers) == 3
        assert "layer1" in registry._custom_layers
        assert "layer2" in registry._custom_layers
        assert "layer3" in registry._custom_layers

    def test_register_duplicate_raises_error(self, registry):
        """Test that registering duplicate name raises ValueError."""
        registry.register_custom_layer("duplicate", lambda c, i: nn.Linear(10, 10))

        with pytest.raises(ValueError) as exc_info:
            registry.register_custom_layer("duplicate", lambda c, i: nn.Linear(20, 20))

        assert "already registered" in str(exc_info.value)
        assert "duplicate" in str(exc_info.value)

    def test_register_empty_name_raises_error(self, registry):
        """Test that registering with empty name raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            registry.register_custom_layer("", lambda c, i: nn.Linear(10, 10))

        assert "layer_name must be a non-empty string" in str(exc_info.value)

    def test_register_none_name_raises_error(self, registry):
        """Test that registering with None name raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            registry.register_custom_layer(None, lambda c, i: nn.Linear(10, 10))

        assert "layer_name must be a non-empty string" in str(exc_info.value)

    def test_register_non_callable_raises_error(self, registry):
        """Test that registering non-callable raises TypeError."""
        with pytest.raises(TypeError) as exc_info:
            registry.register_custom_layer("not_callable", "not a function")

        assert "layer_factory must be callable" in str(exc_info.value)

    def test_register_none_factory_raises_error(self, registry):
        """Test that registering None factory raises TypeError."""
        with pytest.raises(TypeError) as exc_info:
            registry.register_custom_layer("none_factory", None)

        assert "layer_factory must be callable" in str(exc_info.value)

    def test_register_with_naming_convention(self, registry):
        """Test registering with proper naming convention."""
        # Good naming: <family>_<layer_type>
        registry.register_custom_layer("llama_custom_attn", lambda c, i: nn.Linear(100, 100))
        registry.register_custom_layer("moe_expert_layer", lambda c, i: nn.Linear(100, 100))

        assert "llama_custom_attn" in registry._custom_layers
        assert "moe_expert_layer" in registry._custom_layers


class TestGetLayerFactory:
    """Test get_layer_factory() method."""

    @pytest.fixture
    def registry(self):
        """Create a fresh registry for each test."""
        ArchitectureRegistry._instance = None
        ArchitectureRegistry._initialized = False
        reg = ArchitectureRegistry()
        reg._custom_layers.clear()
        return reg

    def test_get_existing_factory(self, registry):
        """Test retrieving a registered factory."""
        factory = lambda c, i: nn.Linear(100, 100)
        registry.register_custom_layer("test_factory", factory)

        retrieved = registry.get_layer_factory("test_factory")

        assert retrieved is factory

    def test_get_nonexistent_raises_keyerror(self, registry):
        """Test retrieving non-existent factory raises KeyError."""
        with pytest.raises(KeyError) as exc_info:
            registry.get_layer_factory("nonexistent")

        assert "not found in registry" in str(exc_info.value)

    def test_get_shows_available_layers(self, registry):
        """Test error message shows available layers."""
        registry.register_custom_layer("available1", lambda c, i: None)
        registry.register_custom_layer("available2", lambda c, i: None)

        with pytest.raises(KeyError) as exc_info:
            registry.get_layer_factory("missing")

        error_msg = str(exc_info.value)
        assert "available1" in error_msg or "available2" in error_msg

    def test_get_empty_registry_shows_none(self, registry):
        """Test error message shows 'none' when registry empty."""
        with pytest.raises(KeyError) as exc_info:
            registry.get_layer_factory("missing")

        assert "none" in str(exc_info.value)

    def test_get_factory_can_create_layer(self, registry):
        """Test that retrieved factory can create a layer."""
        def factory(config, layer_idx):
            return nn.Linear(config.hidden_size, config.hidden_size)

        registry.register_custom_layer("linear_factory", factory)

        retrieved_factory = registry.get_layer_factory("linear_factory")

        mock_config = MagicMock()
        mock_config.hidden_size = 128

        layer = retrieved_factory(mock_config, 0)

        assert isinstance(layer, nn.Linear)
        assert layer.in_features == 128
        assert layer.out_features == 128


class TestUnregisterCustomLayer:
    """Test unregister_custom_layer() method."""

    @pytest.fixture
    def registry(self):
        """Create a fresh registry for each test."""
        ArchitectureRegistry._instance = None
        ArchitectureRegistry._initialized = False
        reg = ArchitectureRegistry()
        reg._custom_layers.clear()
        return reg

    def test_unregister_existing(self, registry):
        """Test unregistering an existing layer."""
        registry.register_custom_layer("to_remove", lambda c, i: nn.Linear(10, 10))

        result = registry.unregister_custom_layer("to_remove")

        assert result is True
        assert "to_remove" not in registry._custom_layers

    def test_unregister_nonexistent(self, registry):
        """Test unregistering a non-existent layer returns False."""
        result = registry.unregister_custom_layer("never_existed")

        assert result is False

    def test_unregister_then_re_register(self, registry):
        """Test that unregistered layer can be re-registered."""
        factory1 = lambda c, i: nn.Linear(10, 10)
        factory2 = lambda c, i: nn.Linear(20, 20)

        registry.register_custom_layer("reusable", factory1)
        registry.unregister_custom_layer("reusable")

        # Should be able to register again
        registry.register_custom_layer("reusable", factory2)

        retrieved = registry.get_layer_factory("reusable")
        assert retrieved is factory2


class TestListCustomLayers:
    """Test list_custom_layers() method."""

    @pytest.fixture
    def registry(self):
        """Create a fresh registry for each test."""
        ArchitectureRegistry._instance = None
        ArchitectureRegistry._initialized = False
        reg = ArchitectureRegistry()
        reg._custom_layers.clear()
        return reg

    def test_list_empty(self, registry):
        """Test listing when no custom layers registered."""
        layers = registry.list_custom_layers()

        assert layers == []

    def test_list_single_layer(self, registry):
        """Test listing with one custom layer."""
        registry.register_custom_layer("single", lambda c, i: nn.Linear(10, 10))

        layers = registry.list_custom_layers()

        assert layers == ["single"]

    def test_list_multiple_layers(self, registry):
        """Test listing with multiple custom layers."""
        registry.register_custom_layer("layer_a", lambda c, i: None)
        registry.register_custom_layer("layer_b", lambda c, i: None)
        registry.register_custom_layer("layer_c", lambda c, i: None)

        layers = registry.list_custom_layers()

        assert len(layers) == 3
        assert "layer_a" in layers
        assert "layer_b" in layers
        assert "layer_c" in layers

    def test_list_returns_copy(self, registry):
        """Test that list returns a copy, not reference."""
        registry.register_custom_layer("original", lambda c, i: None)

        layers = registry.list_custom_layers()
        layers.append("modified")

        # Original should be unchanged
        assert "modified" not in registry.list_custom_layers()

    def test_list_after_unregister(self, registry):
        """Test listing after unregistering."""
        registry.register_custom_layer("keep", lambda c, i: None)
        registry.register_custom_layer("remove", lambda c, i: None)

        registry.unregister_custom_layer("remove")

        layers = registry.list_custom_layers()

        assert "keep" in layers
        assert "remove" not in layers


class TestClearCustomLayers:
    """Test clear_custom_layers() method."""

    @pytest.fixture
    def registry(self):
        """Create a fresh registry for each test."""
        ArchitectureRegistry._instance = None
        ArchitectureRegistry._initialized = False
        reg = ArchitectureRegistry()
        reg._custom_layers.clear()
        return reg

    def test_clear_empty(self, registry):
        """Test clearing empty registry."""
        registry.clear_custom_layers()

        assert registry._custom_layers == {}

    def test_clear_single(self, registry):
        """Test clearing with one layer."""
        registry.register_custom_layer("single", lambda c, i: None)

        registry.clear_custom_layers()

        assert len(registry._custom_layers) == 0

    def test_clear_multiple(self, registry):
        """Test clearing with multiple layers."""
        for i in range(5):
            registry.register_custom_layer(f"layer_{i}", lambda c, i: None)

        registry.clear_custom_layers()

        assert len(registry._custom_layers) == 0
        assert registry.list_custom_layers() == []

    def test_clear_then_register(self, registry):
        """Test that layers can be registered after clearing."""
        registry.register_custom_layer("old", lambda c, i: None)
        registry.clear_custom_layers()

        registry.register_custom_layer("new", lambda c, i: None)

        assert "old" not in registry._custom_layers
        assert "new" in registry._custom_layers


class TestCustomLayerRegistryIntegration:
    """Test integration scenarios."""

    @pytest.fixture
    def registry(self):
        """Create a fresh registry for each test."""
        ArchitectureRegistry._instance = None
        ArchitectureRegistry._initialized = False
        reg = ArchitectureRegistry()
        reg._custom_layers.clear()
        return reg

    def test_full_lifecycle(self, registry):
        """Test complete lifecycle of a custom layer."""
        # Register
        factory = lambda c, i: nn.Linear(c.hidden_size, c.hidden_size)
        registry.register_custom_layer("lifecycle_test", factory)

        # List
        assert "lifecycle_test" in registry.list_custom_layers()

        # Get and use
        retrieved = registry.get_layer_factory("lifecycle_test")
        mock_config = MagicMock(hidden_size=64)
        layer = retrieved(mock_config, 0)
        assert isinstance(layer, nn.Linear)

        # Unregister
        registry.unregister_custom_layer("lifecycle_test")
        assert "lifecycle_test" not in registry.list_custom_layers()

    def test_multiple_families(self, registry):
        """Test registering layers for different families."""
        registry.register_custom_layer("llama_custom", lambda c, i: nn.Linear(100, 100))
        registry.register_custom_layer("gpt_custom", lambda c, i: nn.Linear(200, 200))
        registry.register_custom_layer("bert_custom", lambda c, i: nn.Linear(300, 300))

        layers = registry.list_custom_layers()

        assert len(layers) == 3
        assert all(l in layers for l in ["llama_custom", "gpt_custom", "bert_custom"])

    def test_factory_with_complex_layer(self, registry):
        """Test registering a factory that creates complex layers."""
        def complex_factory(config, layer_idx):
            class ComplexLayer(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.linear1 = nn.Linear(config.hidden_size, config.intermediate_size)
                    self.activation = nn.GELU()
                    self.linear2 = nn.Linear(config.intermediate_size, config.hidden_size)
                    self.norm = nn.LayerNorm(config.hidden_size)

                def forward(self, x):
                    h = self.linear1(x)
                    h = self.activation(h)
                    h = self.linear2(h)
                    return self.norm(x + h)

            return ComplexLayer()

        registry.register_custom_layer("complex_transformer", complex_factory)

        mock_config = MagicMock(hidden_size=128, intermediate_size=512)
        layer_factory = registry.get_layer_factory("complex_transformer")
        layer = layer_factory(mock_config, 0)

        assert hasattr(layer, 'linear1')
        assert hasattr(layer, 'linear2')
        assert hasattr(layer, 'norm')
        assert layer.linear1.in_features == 128
        assert layer.linear1.out_features == 512

    def test_registry_isolated_from_families(self, registry):
        """Test that custom layers don't interfere with family registration."""
        # Families should be pre-registered
        assert len(registry._families) > 0

        # Add custom layers
        registry.register_custom_layer("custom", lambda c, i: None)

        # Families should be unchanged
        assert len(registry._families) > 0
        assert "custom" not in registry._families


class TestCustomLayerRegistryEdgeCases:
    """Test edge cases."""

    @pytest.fixture
    def registry(self):
        """Create a fresh registry for each test."""
        ArchitectureRegistry._instance = None
        ArchitectureRegistry._initialized = False
        reg = ArchitectureRegistry()
        reg._custom_layers.clear()
        return reg

    def test_register_whitespace_name(self, registry):
        """Test registering with whitespace-only name."""
        # Whitespace-only should be considered valid (non-empty string)
        registry.register_custom_layer("   ", lambda c, i: None)
        assert "   " in registry._custom_layers

    def test_register_unicode_name(self, registry):
        """Test registering with unicode name."""
        registry.register_custom_layer("カスタムレイヤー", lambda c, i: None)
        registry.register_custom_layer("自定义层", lambda c, i: None)

        assert "カスタムレイヤー" in registry._custom_layers
        assert "自定义层" in registry._custom_layers

    def test_register_very_long_name(self, registry):
        """Test registering with very long name."""
        long_name = "a" * 1000
        registry.register_custom_layer(long_name, lambda c, i: None)

        assert long_name in registry._custom_layers

    def test_register_callable_objects(self, registry):
        """Test registering callable objects."""
        class CallableClass:
            def __call__(self, config, layer_idx):
                return nn.Linear(100, 100)

        callable_obj = CallableClass()
        registry.register_custom_layer("callable_obj", callable_obj)

        factory = registry.get_layer_factory("callable_obj")
        layer = factory(None, 0)
        assert isinstance(layer, nn.Linear)

    def test_register_bound_method(self, registry):
        """Test registering a bound method."""
        class Factory:
            def create_layer(self, config, layer_idx):
                return nn.Linear(100, 100)

        factory = Factory()
        registry.register_custom_layer("bound_method", factory.create_layer)

        retrieved = registry.get_layer_factory("bound_method")
        layer = retrieved(None, 0)
        assert isinstance(layer, nn.Linear)

    def test_unregister_during_iteration(self, registry):
        """Test that unregistering during iteration works correctly."""
        for i in range(5):
            registry.register_custom_layer(f"layer_{i}", lambda c, i: None)

        # This should not raise
        for name in list(registry._custom_layers.keys()):
            registry.unregister_custom_layer(name)

        assert len(registry._custom_layers) == 0

    def test_clear_during_use(self, registry):
        """Test that clear doesn't break ongoing operations."""
        registry.register_custom_layer("layer1", lambda c, i: nn.Linear(10, 10))

        factory = registry.get_layer_factory("layer1")
        registry.clear_custom_layers()

        # Factory reference still works even if removed from registry
        layer = factory(None, 0)
        assert isinstance(layer, nn.Linear)

    def test_thread_safety_register(self, registry):
        """Test thread-safe registration."""
        import threading
        errors = []

        def worker(n):
            try:
                registry.register_custom_layer(f"thread_layer_{n}", lambda c, i: None)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should have no errors
        assert len(errors) == 0
        # Should have all 10 layers (different names)
        assert len(registry.list_custom_layers()) == 10
