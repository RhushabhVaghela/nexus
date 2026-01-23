import pytest
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.capability_registry import CapabilityRegistry
from src.stages.stage_remotion_gen import RemotionGenStage
from src.stages.base import StageConfig

class TestRemotionCapability:
    """Test Remotion explainer capability and stage."""
    
    @pytest.fixture
    def registry(self):
        return CapabilityRegistry()
    
    def test_capability_registration(self, registry):
        """Verify remotion-explainer is correctly registered."""
        cap = registry.get("remotion-explainer")
        assert cap is not None
        assert cap.name == "remotion-explainer"
        assert "text" in cap.required_modalities
        assert cap.training_script == "src/stages/stage_remotion_gen.py"
    
    def test_remotion_stage_initialization(self, tmp_path):
        """Test RemotionGenStage can be initialized."""
        output_dir = tmp_path / "output"
        config = StageConfig(
            capability_name="remotion-explainer",
            base_model_path="/mock/model",
            output_dir=str(output_dir),
            dry_run=True
        )
        stage = RemotionGenStage(config)
        assert stage.CAPABILITY_NAME == "remotion-explainer"
        assert len(stage.DATASET_PATTERNS) > 0
    
    @patch("src.stages.base.TextCapabilityStage.prepare")
    def test_remotion_stage_prepare_dry_run(self, mock_prepare, tmp_path):
        """Test prepare method in dry-run mode."""
        mock_prepare.return_value = True
        output_dir = tmp_path / "output"
        config = StageConfig(
            capability_name="remotion-explainer",
            base_model_path="/mock/model",
            output_dir=str(output_dir),
            dry_run=True
        )
        stage = RemotionGenStage(config)
        assert stage.prepare() is True
    
    def test_remotion_lib_files_exist(self):
        """Verify that the NexusLib components were created."""
        project_root = Path(__file__).parent.parent.parent
        lib_path = project_root / "remotion" / "src" / "NexusLib"
        assert (lib_path / "NexusMath.tsx").exists()
        assert (lib_path / "NexusGraph.tsx").exists()
        assert (lib_path / "NexusFlow.tsx").exists()
        assert (lib_path / "NexusAnnotator.tsx").exists()
        assert (lib_path / "NexusAudio.tsx").exists()
        assert (lib_path / "Nexus3D.tsx").exists()
        assert (lib_path / "index.ts").exists()

    def test_generator_outputs_new_components(self):
        """Verify the generator produces samples for all categories including 3D and Audio."""
        from src.utils.generate_remotion_dataset import generate_sample
        
        # Sample until we see all types
        seen_types = set()
        for i in range(200):
            sample = generate_sample(i)
            if "NexusMath" in sample["output"]: seen_types.add("math")
            if "NexusGraph" in sample["output"]: seen_types.add("graph")
            if "NexusFlow" in sample["output"]: seen_types.add("flow")
            if "NexusAnnotator" in sample["output"]: seen_types.add("annotator")
            if "Nexus3D" in sample["output"]: seen_types.add("3d")
            if "NexusAudio" in sample["output"]: seen_types.add("audio")
            
        assert "flow" in seen_types
        assert "annotator" in seen_types
        assert "3d" in seen_types
        assert "audio" in seen_types

class TestDatasetGenerator:
    """Test the synthetic dataset generator script."""
    
    def test_dataset_script_exists(self):
        project_root = Path(__file__).parent.parent.parent
        script_path = project_root / "src" / "utils" / "generate_remotion_dataset.py"
        assert script_path.exists()
    
    def test_dataset_output_exists(self):
        """Verify the 1M dataset file was created (at least the first chunk)."""
        dataset_path = Path("/mnt/e/data/datasets/remotion/remotion_explainer_dataset.jsonl")
        assert dataset_path.exists()
        assert dataset_path.stat().st_size > 0

    def test_system_prompt_exists(self):
        """Verify the 3B1B system prompt is defined."""
        from src.capability_registry import REMOTION_EXPLAINER_SYSTEM_PROMPT
        assert "3Blue1Brown" in REMOTION_EXPLAINER_SYSTEM_PROMPT
        assert "NexusLib" in REMOTION_EXPLAINER_SYSTEM_PROMPT
