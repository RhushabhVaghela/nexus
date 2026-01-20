"""
Unit tests for capability_registry.py

Tests:
- Capability registration and lookup
- Modality validation
- Training order calculation
- VRAM estimation
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.capability_registry import CapabilityRegistry, Capability


class TestCapabilityRegistry:
    """Test CapabilityRegistry class."""
    
    @pytest.fixture
    def registry(self):
        """Create registry instance."""
        return CapabilityRegistry()
    
    def test_registry_initialization(self, registry):
        """Test registry initializes with capabilities."""
        assert registry is not None
        assert len(registry._capabilities) > 0
    
    def test_key_capabilities_registered(self, registry):
        """Verify key expected capabilities exist."""
        expected = [
            "cot",
            "reasoning", 
            "thinking",
            "tool-calling",
            "streaming",
            "omni",
            "podcast",
            "vision-qa",
            "video-understanding",
            "tri-streaming",
        ]
        
        for cap in expected:
            assert cap in registry._capabilities, f"Missing capability: {cap}"
    
    def test_capability_has_required_modalities(self, registry):
        """Test that each capability specifies required modalities."""
        for name, cap in registry._capabilities.items():
            assert cap.required_modalities is not None
            assert len(cap.required_modalities) > 0
            assert "text" in cap.required_modalities  # All need text
    
    def test_text_only_capabilities(self, registry):
        """Test text-only capabilities only require text."""
        text_only = ["chain-of-thought", "reasoning", "extended-thinking", "tool-calling", "streaming"]
        
        for name in text_only:
            cap = registry._capabilities.get(name)
            if cap:
                # Should only require text
                assert cap.required_modalities == {"text"} or \
                       cap.required_modalities == frozenset({"text"})


class TestCapabilityValidation:
    """Test capability validation logic."""
    
    @pytest.fixture
    def registry(self):
        return CapabilityRegistry()
    
    def test_validate_text_only_model(self, registry):
        """Test validation for text-only model."""
        model_modalities = {"text"}
        
        # Should pass for text-only capabilities
        cot = registry.get("cot")
        if cot:
            valid, missing = cot.validate(model_modalities)
            assert valid is True
        
        # Should fail for multimodal capabilities
        podcast = registry.get("podcast")
        if podcast:
            valid, missing = podcast.validate(model_modalities)
            assert valid is False
            assert len(missing) > 0
    
    def test_validate_omni_model(self, registry):
        """Test validation for full Omni model."""
        model_modalities = {"text", "vision", "audio_input", "audio_output", "video"}
        
        # Should pass for all non-generation capabilities
        for cap_name in ["cot", "podcast", "vision-qa", "tri-streaming"]:
            cap = registry.get(cap_name)
            if cap:
                valid, missing = cap.validate(model_modalities)
                assert valid is True, f"Failed for {cap_name}: missing {missing}"
    
    def test_get_nonexistent_capability(self, registry):
        """Test getting a non-existent capability."""
        result = registry.get("nonexistent")
        assert result is None


class TestTrainingOrder:
    """Test optimal training order calculation."""
    
    @pytest.fixture
    def registry(self):
        return CapabilityRegistry()
    
    def test_get_training_order(self, registry):
        """Test training order is returned."""
        all_caps = list(registry._capabilities.keys())
        order = registry.get_training_order(all_caps)
        assert isinstance(order, list)
        assert len(order) > 0
    
    def test_text_capabilities_come_first(self, registry):
        """Test that text-only capabilities appear early."""
        all_caps = list(registry._capabilities.keys())
        order = registry.get_training_order(all_caps)
        text_only = ["cot", "reasoning", "tool-calling"]
        
        # Get indices of text-only capabilities
        text_indices = [order.index(c) for c in text_only if c in order]
        
        # Should be in first half
        if text_indices:
            assert max(text_indices) < len(order) // 2 + 2
    
    def test_generation_capabilities_come_last(self, registry):
        """Test that generation capabilities appear last."""
        all_caps = list(registry._capabilities.keys())
        order = registry.get_training_order(all_caps)
        gen_caps = ["image-generation", "video-generation"]
        
        # Get indices
        gen_indices = [order.index(c) for c in gen_caps if c in order]
        
        # Should be in last positions
        if gen_indices:
            assert min(gen_indices) >= len(order) - 3


class TestVRAMEstimation:
    """Test VRAM estimation for capabilities."""
    
    @pytest.fixture
    def registry(self):
        return CapabilityRegistry()
    
    def test_vram_estimates_exist(self, registry):
        """Test that VRAM estimates are provided."""
        for name, cap in registry._capabilities.items():
            assert cap.estimated_vram_gb is not None
            assert cap.estimated_vram_gb > 0
    
    def test_vram_estimates_reasonable(self, registry):
        """Test VRAM estimates are reasonable."""
        for name, cap in registry._capabilities.items():
            # Should be between 4GB and 24GB
            assert 2 <= cap.estimated_vram_gb <= 24, f"Unreasonable VRAM for {name}"
    
    def test_text_capabilities_low_vram(self, registry):
        """Test text-only capabilities have lower VRAM requirements."""
        text_only = ["chain-of-thought", "reasoning", "tool-calling"]
        
        for name in text_only:
            cap = registry._capabilities.get(name)
            if cap:
                assert cap.estimated_vram_gb <= 12, f"{name} VRAM too high"


class TestCapabilityDataclass:
    """Test Capability dataclass."""
    
    def test_capability_creation(self):
        """Test creating a Capability."""
        cap = Capability(
            name="test",
            description="Test capability",
            required_modalities={"text"},
            training_script="test.py",
            estimated_vram_gb=8,
            estimated_time_hours=2.0,
            dataset_patterns=["test_*"],
        )
        
        assert cap.name == "test"
        assert "text" in cap.required_modalities
        assert cap.estimated_vram_gb == 8
    
    def test_capability_required_fields(self):
        """Test that required fields are enforced."""
        # Should work with minimal fields
        cap = Capability(
            name="minimal",
            description="Minimal test",
            required_modalities={"text"},
            training_script="min.py",
        )
        assert cap.name == "minimal"
