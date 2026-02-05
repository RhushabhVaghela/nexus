
import unittest
from unittest.mock import MagicMock, patch
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parents[2] / "src"))

class TestOmniIntegration(unittest.TestCase):
    
    def setUp(self):
        self.mock_torch = MagicMock()
        self.mock_transformers = MagicMock()
        
        self.patcher = patch.dict(sys.modules, {
            "torch": self.mock_torch,
            "torch.nn": MagicMock(),
            "transformers": self.mock_transformers
        })
        self.patcher.start()
        
    def tearDown(self):
        self.patcher.stop()

    def test_decoders_sota_format(self):
        """Test proper SOTA processor IDs"""
        from src.multimodal.decoders import ImageDecoder, AudioDecoder
        
        img_decoder = ImageDecoder()
        res = img_decoder.decode("test.png")
        self.assertEqual(res["processor_id"], "/mnt/e/data/encoders/vision-encoders/siglip2-so400m-patch16-512")
        
        aud_decoder = AudioDecoder()
        res = aud_decoder.decode("test.mp3")
        self.assertEqual(res["processor_id"], "/mnt/e/data/encoders/audio-encoders/whisper-large-v3-turbo")

if __name__ == '__main__':
    unittest.main()
