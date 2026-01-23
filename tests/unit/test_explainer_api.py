import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
from src.api.explainer_api import app

client = TestClient(app)

class TestExplainerAPI:
    
    def test_health_check(self):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

    @patch("src.api.explainer_api.RemotionExplainerEngine")
    def test_generate_endpoint(self, mock_engine_class):
        # Setup mock engine
        mock_engine = mock_engine_class.return_value
        mock_engine.generate_video.return_value = "explanation.mp4"
        mock_engine.remotion_dir = MagicMock()
        
        # Mock reading the TSX file
        with patch("builtins.open", pytest.raises(FileNotFoundError) if False else MagicMock()):
            # We need a more robust mock for open()
            pass
            
        # Simplified test for now
        with patch("src.api.explainer_api.engine", mock_engine):
            with patch("builtins.open", MagicMock(return_value=MagicMock(__enter__=lambda x: MagicMock(read=lambda: "import React from 'react';")))):
                response = client.post("/generate", json={"prompt": "test prompt"})
                
        assert response.status_code == 200
        assert response.json()["success"] is True
        assert "import React" in response.json()["tsx_preview"]

    def test_generate_missing_prompt(self):
        response = client.post("/generate", json={})
        assert response.status_code == 422 # Validation error
