
import unittest
import sys
import os
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add src to path
sys.path.append(str(Path(__file__).parents[2] / "src"))

class TestVoiceCore(unittest.TestCase):
    
    def setUp(self):
        self.storage_path = Path("/tmp/nexus_voice_test")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
    def tearDown(self):
        # Clean up temp files
        import shutil
        if self.storage_path.exists():
            shutil.rmtree(self.storage_path)

    def test_voice_registry_presets(self):
        from voice_engine.registry import VoiceRegistry
        registry = VoiceRegistry(storage_path=str(self.storage_path))
        
        voices = registry.list_voices()
        self.assertIn("NATF0", voices)
        self.assertEqual(voices["NATF0"]["type"], "preset")
        
        dna = registry.get_voice_dna("NATF0")
        self.assertEqual(dna, "builtin://NATF0")

    def test_voice_registry_custom(self):
        from voice_engine.registry import VoiceRegistry
        registry = VoiceRegistry(storage_path=str(self.storage_path))
        
        registry.register_voice("Custom1", "/path/to/dna.pt", "A test voice")
        
        voices = registry.list_voices()
        self.assertIn("Custom1", voices)
        self.assertEqual(voices["Custom1"]["type"], "cloned")
        
        dna = registry.get_voice_dna("Custom1")
        self.assertEqual(dna, "/path/to/dna.pt")

    def test_vibe_modulator_params(self):
        from voice_engine.vibe_modulator import VibeModulator
        modulator = VibeModulator()
        
        params = modulator.get_vibe_params("excited")
        self.assertEqual(params["pitch"], 1.1)
        self.assertEqual(params["emotion_id"], 1)
        
        # Test default
        params = modulator.get_vibe_params("unknown")
        self.assertEqual(params["pitch"], 1.0)

    @patch('torchaudio.load')
    @patch('torch.save')
    def test_voice_cloner(self, mock_save, mock_load):
        from voice_engine.cloner import VoiceCloner
        from voice_engine.registry import voice_registry
        
        # Mock torchaudio.load return value
        mock_load.return_value = (MagicMock(), 16000)
        
        cloner = VoiceCloner()
        # Mock encoder
        cloner.encoder = MagicMock()
        cloner.encoder.extract_features.return_value = MagicMock()
        
        # Create a dummy wav file
        dummy_wav = self.storage_path / "test.wav"
        dummy_wav.write_text("dummy audio data")
        
        with patch('voice_engine.cloner.Path.exists') as mock_exists:
            mock_exists.return_value = True
            dna_path = cloner.clone_voice(str(dummy_wav), "MyClonedVoice")
        
        self.assertIsNotNone(dna_path)
        self.assertIn("MyClonedVoice.pt", dna_path)
        
        # Verify it was registered
        self.assertIn("MyClonedVoice", voice_registry.list_voices())

if __name__ == '__main__':
    unittest.main()
