import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from pathlib import Path
import os
import json

from src.api.explainer_api import app

client = TestClient(app)

class TestDashboardFlow:
    """Tests the flow from prompt to video generation."""
    
    @patch("src.api.explainer_api.RemotionExplainerEngine")
    def test_end_to_end_generation_logic(self, mock_engine_class, tmp_path):
        """Verify the full generation flow triggered by the API."""
        # 1. Setup Mock Engine
        mock_engine = mock_engine_class.return_value
        mock_engine.generate_video.return_value = str(tmp_path / "explanation.mp4")
        mock_engine.remotion_dir = tmp_path / "remotion"
        
        # 2. Create mock GeneratedScene.tsx
        (tmp_path / "remotion" / "src").mkdir(parents=True)
        tsx_file = tmp_path / "remotion" / "src" / "GeneratedScene.tsx"
        tsx_file.write_text("import React from 'react';\nexport const Scene = () => <div />;")
        
        # 3. Call API
        with patch("src.api.explainer_api.engine", mock_engine):
            response = client.post("/generate", json={
                "prompt": "Explain the concept of entropy",
                "narrate": True
            })
            
        # 4. Assertions
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "entropy" in mock_engine.generate_video.call_args[1]["prompt"]
        assert mock_engine.generate_video.call_args[1]["narrate"] is True
        assert "import React" in data["tsx_preview"]

    def test_health_and_startup(self):
        """Verify the API health check."""
        response = client.get("/health")
        assert response.status_code == 200
        assert "status" in response.json()
