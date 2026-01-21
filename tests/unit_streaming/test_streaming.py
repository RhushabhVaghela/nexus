
import unittest
from unittest.mock import MagicMock, patch
import sys
from pathlib import Path
import importlib

# Add src to path
sys.path.append(str(Path(__file__).parents[2] / "src"))

class TestStreamingComponents(unittest.TestCase):
    
    def setUp(self):
        # Create a dummy Module class to inherit from
        class MockModule:
            def __init__(self):
                pass
            def __call__(self, *args, **kwargs):
                return self.forward(*args, **kwargs)

        # Setup Mock torch
        self.mock_torch = MagicMock()
        self.mock_nn = MagicMock()
        self.mock_nn.Module = MockModule
        self.mock_torch.nn = self.mock_nn
        
        # Patch sys.modules safely using context manager setup
        self.patcher = patch.dict(sys.modules, {
            "torch": self.mock_torch,
            "torch.nn": self.mock_nn,
            "sounddevice": MagicMock(),
            "pyaudio": MagicMock()
        })
        self.patcher.start()

    def tearDown(self):
        self.patcher.stop()

    def test_memory_init(self):
        from streaming.memory import StreamingMemory
        mem = StreamingMemory(sink_size=4, window_size=128)
        self.assertEqual(mem.sink_size, 4)
        
    def test_vision_buffer(self):
        from streaming.vision import VisionStreamBuffer
        vis = VisionStreamBuffer(max_frames=5)
        self.assertEqual(vis.buffer.maxlen, 5)
        # Add mock tensor
        mock_tensor = MagicMock()
        vis.add_frame(mock_tensor)
        ctx = vis.get_context()
        # Ensure torch.stack was called
        self.mock_torch.stack.assert_called()

    def test_tts_init(self):
        from streaming.tts import TTSStreamer
        tts = TTSStreamer(model_name="Test-Model")
        self.assertEqual(tts.model_name, "Test-Model")
        
    def test_joint_orchestrator(self):
        from streaming.joint import JointStreamingOrchestrator
        orch = JointStreamingOrchestrator()
        self.assertFalse(orch.is_active)
        self.assertIsNotNone(orch.vision)
        self.assertIsNotNone(orch.tts)

    def test_podcast_generator(self):
        from podcast.generator import generate_podcast_script
        mock_llm = MagicMock()
        # Mock valid JSON response as string
        mock_llm.return_value = '{"turns": [{"speaker": "Host A", "text": "Hello world"}]}'
        
        script = generate_podcast_script(["Some content"], llm=mock_llm)
        self.assertTrue(len(script.turns) > 0)
        self.assertEqual(script.turns[0].speaker, "Host A")

if __name__ == '__main__':
    unittest.main()
