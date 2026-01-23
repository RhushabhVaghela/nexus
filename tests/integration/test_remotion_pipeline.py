import pytest
import sys
import subprocess
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.capability_registry import CapabilityRegistry
from src.metrics_tracker import get_capability_datasets

class TestRemotionIntegration:
    """Integration tests for Remotion capability."""
    
    def test_dataset_discovery(self):
        """Verify that the remotion dataset is discovered."""
        datasets = get_capability_datasets("remotion-explainer")
        assert len(datasets) > 0
        assert any("remotion" in d.lower() for d in datasets)
    
    def test_pipeline_dry_run(self):
        """Verify that the pipeline can run a dry-run with remotion-explainer."""
        model_path = "/mnt/e/data/models/Qwen2.5-Omni-7B-GPTQ-Int4"
        if not Path(model_path).exists():
            pytest.skip(f"Model not found at {model_path}")
            
        cmd = [
            "./run_universal_pipeline.sh",
            f"--base-model={model_path}",
            "--enable-remotion-explainer",
            "--dry-run"
        ]
        
        # Use the specific python from the env if needed, but the script handles it
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        assert result.returncode == 0
        assert "remotion-explainer" in result.stdout
        assert "Completed: remotion-explainer" in result.stdout

    def test_nexus_lib_compilation(self):
        """Verify that the Remotion project can be initialized (node_modules check)."""
        project_root = Path(__file__).parent.parent.parent
        remotion_dir = project_root / "remotion"
        
        assert (remotion_dir / "package.json").exists()
        # Check if npm install was run (node_modules exists)
        assert (remotion_dir / "node_modules").exists()
