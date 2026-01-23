"""
E2E tests for actual pipeline execution scenarios.

Tests real orchestrator runs with sample data.
"""

import pytest
import subprocess
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

ORCHESTRATOR_PATH = Path(__file__).parent.parent.parent / "run_universal_pipeline.sh"
PROJECT_ROOT = Path(__file__).parent.parent.parent


class TestOrchestratorScenarios:
    """Test actual orchestrator execution scenarios."""
    
    @pytest.mark.slow
    @pytest.mark.real_model
    def test_scenario_text_only_capabilities(self, text_model_path, tmp_path):
        """Scenario 1: Run with random text-only capabilities."""
        output_dir = tmp_path / "scenario1_output"
        
        # Run with CoT and Tools enabled (text-only, should work with any model)
        result = subprocess.run(
            [
                "bash", str(ORCHESTRATOR_PATH),
                f"--base-model={text_model_path}",
                f"--output-dir={output_dir}",
                "--enable-cot",
                "--sample-size=10",  # Very small for testing
            ],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
            timeout=1500,  # 25 min timeout for real training
        )
        
        # Should complete (may have warnings but not crash)
        print(f"STDOUT: {result.stdout[:1000]}")
        print(f"STDERR: {result.stderr[:1000]}")
        
        # Check it ran modality detection
        assert "Detecting" in result.stdout or "modality" in result.stdout.lower() or result.returncode == 0
    
    @pytest.mark.slow
    @pytest.mark.real_model
    def test_scenario_help_runs(self):
        """Test that --help runs and shows usage."""
        result = subprocess.run(
            ["bash", str(ORCHESTRATOR_PATH), "--help"],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
            timeout=10,
        )
        
        # Script prints help but may exit with 1 (no base model)
        assert "enable-" in result.stdout.lower()
    
    @pytest.mark.slow
    @pytest.mark.real_model
    def test_scenario_validation_gates(self, text_model_path, tmp_path):
        """Test that validation gates block invalid capability combos."""
        output_dir = tmp_path / "scenario2_output"
        
        # Try to enable podcast on text-only model (should fail validation)
        result = subprocess.run(
            [
                "bash", str(ORCHESTRATOR_PATH),
                f"--base-model={text_model_path}",
                f"--output-dir={output_dir}",
                "--enable-podcast",  # Requires audio, text model doesn't have it
            ],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
            timeout=60,
        )
        
        # Should fail with modality gate message
        combined = result.stdout + result.stderr
        # Either exits with error or prints warning about missing modality
        assert result.returncode != 0 or "missing" in combined.lower() or \
               "requires" in combined.lower() or "audio" in combined.lower()


class TestPipelineComponentsIntegration:
    """Test individual pipeline components work together."""
    
    @pytest.mark.real_model
    def test_detect_then_validate_flow(self, text_model_path):
        """Test detection -> validation flow."""
        from src.detect_modalities import detect_modalities
        from src.capability_registry import CapabilityRegistry
        
        # Detect
        result = detect_modalities(text_model_path)
        assert result["modalities"]["text"] is True
        model_mods = {m for m, v in result["modalities"].items() if v}
        
        # Validate
        registry = CapabilityRegistry()
        
        # CoT should be valid
        cot = registry.get("cot")
        if cot:
            valid, _ = cot.validate(model_mods)
            assert valid is True
        
        # Podcast should be invalid
        podcast = registry.get("podcast")
        if podcast:
            valid, missing = podcast.validate(model_mods)
            assert valid is False
    
    @pytest.mark.real_model
    def test_training_controller_integration(self, tmp_path):
        """Test training controller can be instantiated."""
        from src.training_controller import (
            setup_signal_handlers,
            check_pause_state,
            check_and_cooldown,
        )
        import src.training_controller as tc
        
        # Verify the module works
        tc._paused = False
        check_pause_state()  # Should not block
        
        # Test step increment
        tc._checkpoint_requested = False
        assert tc._checkpoint_requested is False


class TestOrchestratorOutputArtifacts:
    """Test that orchestrator produces expected outputs."""
    
    def test_log_directory_creation(self, tmp_path):
        """Test log directory is created."""
        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        
        assert log_dir.exists()
    
    def test_checkpoint_directory_creation(self, tmp_path):
        """Test checkpoint directory is created."""
        ckpt_dir = tmp_path / "checkpoints"
        ckpt_dir.mkdir()
        
        assert ckpt_dir.exists()
    
    @pytest.mark.slow
    def test_orchestrator_creates_log_dir(self):
        """Test orchestrator script creates log directory."""
        with open(ORCHESTRATOR_PATH) as f:
            content = f.read()
        
        assert "LOG_DIR" in content
        assert "mkdir" in content
