import pytest
import os
from unittest.mock import MagicMock, patch
from pathlib import Path
from src.inference.remotion_engine import RemotionExplainerEngine

class TestRemotionEngine:
    
    @pytest.fixture
    def engine(self, tmp_path):
        # Mock dependencies
        with patch("src.inference.remotion_engine.OmniInference") as mock_inf, \
             patch("src.inference.remotion_engine.AssetManager"):
            
            remotion_dir = tmp_path / "remotion"
            src_dir = remotion_dir / "src"
            src_dir.mkdir(parents=True)
            
            # Create a mock Root.tsx
            root_tsx = src_dir / "Root.tsx"
            root_tsx.write_text("import { NexusGraph } from './NexusLib/NexusGraph';\nexport const RemotionRoot = () => (<></>);")
            
            engine = RemotionExplainerEngine(
                model_path="/mock/model",
                remotion_dir=str(remotion_dir),
                data_root=str(tmp_path / "data")
            )
            # Ensure the inference instance is a mock
            engine.inference = mock_inf.return_value
            return engine

    def test_extract_tsx(self, engine):
        text = "Here is the code:\n```tsx\nimport React from 'react';\nconst Scene = () => <div />;\n```"
        result = engine._extract_tsx(text)
        assert "import React" in result
        assert "const Scene" in result

    def test_extract_tsx_no_blocks(self, engine):
        text = "import React from 'react';\nconst Scene = () => <div />;"
        result = engine._extract_tsx(text)
        assert "import React" in result

    @patch("subprocess.run")
    def test_generate_video_flow(self, mock_run, engine):
        # Mock model response
        engine.inference.chat.return_value = "```tsx\nimport React from 'react';\nexport const Scene = () => <div />;\n```"
        
        # Mock subprocess
        mock_run.return_value = MagicMock(returncode=0)
        
        output = engine.generate_video("test prompt", "test.mp4")
        
        assert "test.mp4" in output
        assert (engine.remotion_dir / "src" / "GeneratedScene.tsx").exists()
        assert engine.inference.chat.called
