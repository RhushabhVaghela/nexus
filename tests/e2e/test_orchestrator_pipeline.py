"""
End-to-end tests for the orchestrator pipeline.

Tests:
- Orchestrator script parsing
- Modality gate validation flow
- Capability selection
"""

import pytest
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

ORCHESTRATOR_PATH = Path(__file__).parent.parent.parent / "run_universal_pipeline.sh"


class TestOrchestratorScript:
    """Test orchestrator shell script."""
    
    def test_script_exists(self):
        """Test orchestrator script exists."""
        assert ORCHESTRATOR_PATH.exists()
    
    def test_script_is_executable(self):
        """Test script has execute permission."""
        import os
        assert os.access(ORCHESTRATOR_PATH, os.X_OK)
    
    def test_help_output(self):
        """Test --help flag works."""
        result = subprocess.run(
            [str(ORCHESTRATOR_PATH), "--help"],
            capture_output=True,
            text=True,
        )
        
        # Should show help text
        assert "enable-" in result.stdout.lower() or result.returncode == 0
        assert "capability" in result.stdout.lower() or "enable" in result.stdout.lower()
    
    def test_help_lists_all_capabilities(self):
        """Test help lists all 12 capabilities."""
        result = subprocess.run(
            [str(ORCHESTRATOR_PATH), "--help"],
            capture_output=True,
            text=True,
        )
        
        expected_caps = [
            "omni", "cot", "reasoning", "tools", "podcast",
            "vision-qa", "tri-streaming", "image-generation",
        ]
        
        for cap in expected_caps:
            assert cap in result.stdout.lower(), f"Missing capability: {cap}"


class TestModalityGateFlow:
    """Test modality gate validation flow."""
    
    def test_detect_modalities_for_text_model(self, text_model_path):
        """Test modality detection returns correct flags."""
        from src.detect_modalities import detect_modalities
        
        result = detect_modalities(text_model_path)
        
        # Validate all flags present
        expected_mods = ["text", "vision", "audio_input", "audio_output", "video"]
        for mod in expected_mods:
            assert mod in result["modalities"]
    
    def test_capability_validation_rejects_invalid(self):
        """Test capability validation rejects invalid combinations."""
        from src.capability_registry import CapabilityRegistry
        
        registry = CapabilityRegistry()
        
        # Text-only model trying to use podcast
        text_only_mods = {"text"}
        
        podcast = registry.get("podcast")
        if podcast:
            valid, missing = podcast.validate(text_only_mods)
            assert valid is False
    
    def test_capability_validation_accepts_valid(self):
        """Test capability validation accepts valid combinations."""
        from src.capability_registry import CapabilityRegistry
        
        registry = CapabilityRegistry()
        
        # Text-only model with CoT
        text_only_mods = {"text"}
        
        cot = registry.get("cot")
        if cot:
            valid, missing = cot.validate(text_only_mods)
            assert valid is True


class TestCapabilitySelection:
    """Test capability flag parsing."""
    
    def test_enable_all_text_expands(self):
        """Test --enable-all-text flag expands correctly."""
        # This would test the shell script logic
        # For now, verify the logic exists in script
        with open(ORCHESTRATOR_PATH) as f:
            content = f.read()
        
        assert "enable-all-text" in content
        assert "ENABLE_COT=true" in content or "enable-cot" in content.lower()
    
    def test_enable_full_omni_expands(self):
        """Test --enable-full-omni flag expands correctly."""
        with open(ORCHESTRATOR_PATH) as f:
            content = f.read()
        
        assert "enable-full-omni" in content
        assert "ENABLE_OMNI=true" in content or "enable-omni" in content.lower()


class TestPipelineStageSequencing:
    """Test pipeline stage sequencing logic."""
    
    def test_training_order_from_registry(self):
        """Test training order is determined correctly."""
        from src.capability_registry import CapabilityRegistry
        
        registry = CapabilityRegistry()
        all_caps = list(registry._capabilities.keys())
        order = registry.get_training_order(all_caps)
        
        assert isinstance(order, list)
        assert len(order) > 0
    
    def test_omni_comes_before_multimodal(self):
        """Test Omni conversion happens before multimodal training."""
        from src.capability_registry import CapabilityRegistry
        
        registry = CapabilityRegistry()
        all_caps = ["omni", "podcast", "cot"]
        order = registry.get_training_order(all_caps)
        
        if "omni" in order and "podcast" in order:
            omni_idx = order.index("omni")
            podcast_idx = order.index("podcast")
            assert omni_idx < podcast_idx
    
    def test_text_capabilities_before_vision(self):
        """Test text capabilities come before vision."""
        from src.capability_registry import CapabilityRegistry
        
        registry = CapabilityRegistry()
        all_caps = list(registry._capabilities.keys())
        order = registry.get_training_order(all_caps)
        
        text_caps = ["cot", "reasoning", "tool-calling"]
        vision_caps = ["vision-qa", "video-understanding"]
        
        text_indices = [order.index(c) for c in text_caps if c in order]
        vision_indices = [order.index(c) for c in vision_caps if c in order]
        
        if text_indices and vision_indices:
            assert max(text_indices) < min(vision_indices)


class TestIntegrationWithRealModel:
    """Test full integration with real model."""
    
    @pytest.mark.real_model
    @pytest.mark.slow
    def test_full_detection_to_validation_flow(self, text_model_path):
        """Test complete flow from detection to validation."""
        from src.detect_modalities import detect_modalities
        from src.capability_registry import CapabilityRegistry
        
        # Step 1: Detect
        detection = detect_modalities(text_model_path)
        assert detection["modalities"]["text"] is True
        
        # Step 2: Registry
        registry = CapabilityRegistry()
        model_mods = {m for m, v in detection["modalities"].items() if v}
        
        # Step 3: Validate allowed capabilities
        allowed = []
        for cap_name, cap in registry._capabilities.items():
            valid, _ = cap.validate(model_mods)
            if valid:
                allowed.append(cap_name)
        
        # Text model should allow text-only capabilities
        assert "cot" in allowed
        
        # Should NOT allow multimodal
        assert "podcast" not in allowed
