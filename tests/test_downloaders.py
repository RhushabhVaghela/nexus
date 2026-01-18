
import unittest
import sys
from unittest.mock import MagicMock, patch, mock_open
from pathlib import Path
import runpy
import importlib

# Fix imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Mock DL Libs BEFORE import
sys.modules["unsloth"] = MagicMock()
sys.modules["transformers"] = MagicMock()
sys.modules["peft"] = MagicMock()
sys.modules["trl"] = MagicMock()
sys.modules["wandb"] = MagicMock()
sys.modules["torch"] = MagicMock()
sys.modules["torch.nn"] = MagicMock()
sys.modules["torch.nn.functional"] = MagicMock()

# Mock datasets library before importing modules
mock_datasets_lib = MagicMock()
sys.modules["datasets"] = mock_datasets_lib

# Import Multimodal
import multimodal.download as mm_dl
importlib.reload(mm_dl)

class TestDownloaders(unittest.TestCase):
    
    def setUp(self):
        mock_datasets_lib.reset_mock()
        
    def test_multimodal_vision_streaming(self):
        """Test that vision download uses validation and streaming."""
        mock_ds = MagicMock()
        mock_datasets_lib.load_dataset.return_value = mock_ds
        
        mock_take = MagicMock()
        mock_ds.take.return_value = mock_take
        mock_take.__iter__.return_value = iter([{"image": "mock", "text": "mock"}])
        
        mm_dl.download_vision_data("/tmp/mock_out")
        
        mock_datasets_lib.load_dataset.assert_called_with("HuggingFaceM4/WebSight", split="train", streaming=True)
        mock_ds.take.assert_called_with(10000)
    
    def test_multimodal_audio_streaming(self):
        """Test audio streaming."""
        mock_ds = MagicMock()
        mock_datasets_lib.load_dataset.return_value = mock_ds
        
        mock_take = MagicMock()
        mock_ds.take.return_value = mock_take
        mock_take.__iter__.return_value = iter([{"path": "mock", "sentence": "mock"}])
        
        mm_dl.download_audio_data("/tmp/mock_out")
        
        mock_datasets_lib.load_dataset.assert_called_with("mozilla-foundation/common_voice_17_0", "en", split="train", streaming=True, trust_remote_code=True)
        mock_ds.take.assert_called_with(1000)
    def test_multimodal_limit(self):
        """Test strict limit enforcement."""
        mock_ds = MagicMock()
        mock_datasets_lib.load_dataset.return_value = mock_ds
        
        mock_take = MagicMock()
        mock_ds.take.return_value = mock_take
        mock_take.__iter__.return_value = iter([{"image": "mock"}])
        
        # Call with explicit limit 50
        mm_dl.download_vision_data("/tmp/mock_out_limit", limit=50)
        
        # Verify take called with 50
        mock_ds.take.assert_called_with(50)
    def test_01_static_analysis(self):
        """Verify 01_download_real_datasets.py uses streaming=True via static analysis."""
        script_path = Path(__file__).parent.parent / "src/01_download_real_datasets.py"
        with open(script_path, 'r') as f:
            content = f.read()
        
        # Check for streaming=True usage in download_huggingface
        self.assertIn('load_dataset(source, streaming=True)', content)
        self.assertIn('MAX_RAW_SAMPLES', content)

if __name__ == '__main__':
    unittest.main()
