
import unittest
from unittest.mock import MagicMock, patch
import sys
import importlib
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parents[2] / "src"))

class TestMultimodalDownload(unittest.TestCase):
    
    def setUp(self):
        # Create a mock for the datasets module
        self.mock_datasets = MagicMock()
        self.mock_datasets.load_dataset = MagicMock()
        
        # Patch sys.modules to include our mock
        self.modules_patcher = patch.dict(sys.modules, {"datasets": self.mock_datasets})
        self.modules_patcher.start()
        
        # Force reload of the module to ensure it picks up the mocked datasets
        import multimodal.download
        importlib.reload(multimodal.download)
        self.module = multimodal.download

    def tearDown(self):
        self.modules_patcher.stop()

    def test_download_vision(self):
        mock_ds = MagicMock()
        self.mock_datasets.load_dataset.return_value = mock_ds
        
        output_dir = "/tmp/test_output"
        self.module.download_vision_data(output_dir)
        
        # Check call arguments
        self.mock_datasets.load_dataset.assert_any_call("HuggingFaceM4/WebSight", split="train[:10000]")
        mock_ds.save_to_disk.assert_called()

    def test_download_audio(self):
        mock_ds = MagicMock()
        self.mock_datasets.load_dataset.return_value = mock_ds
        
        output_dir = "/tmp/test_output"
        self.module.download_audio_data(output_dir)
        
        self.mock_datasets.load_dataset.assert_any_call("mozilla-foundation/common_voice_17_0", "en", split="train[:1000]", trust_remote_code=True)

    def test_download_video(self):
        mock_ds = MagicMock()
        self.mock_datasets.load_dataset.return_value = mock_ds
        
        output_dir = "/tmp/test_output"
        self.module.download_video_data(output_dir)
        
        self.mock_datasets.load_dataset.assert_any_call("HuggingFaceM4/FineVideo", split="train[:100]")

if __name__ == '__main__':
    unittest.main()
