import pytest
import sys
import subprocess
import shutil
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.capability_registry import CapabilityRegistry
from src.metrics_tracker import get_capability_datasets

class TestRemotionIntegration:
    """Integration tests for Remotion capability (MOCKED)."""
    
    @pytest.fixture
    def fake_model_dir(self, tmp_path):
        """Create a fake model directory for pipeline testing."""
        model_dir = tmp_path / "fake-model"
        model_dir.mkdir()
        # Create a dummy config.json so the script validation passes
        config = {"model_type": "qwen2", "hidden_size": 4096}
        with open(model_dir / "config.json", "w") as f:
            json.dump(config, f)
        return str(model_dir)

    def test_dataset_discovery(self):
        """Verify that the remotion dataset is discovered."""
        datasets = get_capability_datasets("remotion-explainer")
        assert len(datasets) > 0
        assert any("remotion" in d.lower() for d in datasets)
    
    def test_pipeline_dry_run(self, fake_model_dir):
        """Verify that the pipeline can run a dry-run with remotion-explainer."""
        cmd = [
            "./run_universal_pipeline.sh",
            f"--base-model={fake_model_dir}",
            "--enable-remotion-explainer",
            "--dry-run"
        ]
        
        # Use the specific python from the env if needed
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # If the environment check fails in the shell script, we might get returncode 1
        # but here we want to ensure it reached the stages
        assert "remotion-explainer" in result.stdout
        assert "DRY-RUN" in result.stdout
        assert result.returncode == 0

    def test_nexus_lib_compilation(self):
        """Verify that the Remotion project can be initialized (node_modules check)."""
        project_root = Path(__file__).parent.parent.parent
        remotion_dir = project_root / "remotion"
        
        assert (remotion_dir / "package.json").exists()
        # In a real environment node_modules should exist if npm install was run
        # For CI/Tests we might want to skip this if not installed
        if not (remotion_dir / "node_modules").exists():
            pytest.skip("node_modules not found in remotion directory")
        
        assert (remotion_dir / "node_modules").exists()
