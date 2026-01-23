import pytest
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from nexus_explain import main as cli_main

class TestExplainCLI:
    """Integration tests for the nexus_explain.py CLI tool."""
    
    def test_cli_help(self):
        """Verify the help command works."""
        with patch("sys.argv", ["nexus_explain.py", "--help"]):
            with pytest.raises(SystemExit):
                cli_main()

    @patch("src.inference.remotion_engine.RemotionExplainerEngine.generate_video")
    @patch("src.inference.remotion_engine.OmniInference")
    def test_cli_basic_prompt(self, mock_inf, mock_gen):
        """Verify the CLI accepts a basic prompt and enters the generation flow."""
        mock_gen.return_value = "explanation.mp4"
        with patch("sys.argv", ["nexus_explain.py", "Explain the unit circle"]):
            cli_main()
        assert mock_gen.called

    @patch("src.inference.remotion_engine.RemotionExplainerEngine.generate_video")
    @patch("src.inference.remotion_engine.OmniInference")
    def test_cli_narrate_flag(self, mock_inf, mock_gen):
        """Verify the narrate flag triggers the audio stage."""
        mock_gen.return_value = "explanation.mp4"
        with patch("sys.argv", ["nexus_explain.py", "Explain gravity", "--narrate"]):
            cli_main()
        assert mock_gen.called
        # Check if narrate=True was passed to generate_video
        args, kwargs = mock_gen.call_args
        assert kwargs["narrate"] is True

