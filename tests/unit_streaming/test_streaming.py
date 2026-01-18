
import unittest
from unittest.mock import MagicMock
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parents[2] / "src"))

# ROBUST MOCKING STRATEGY
# 1. Create a dummy Module class to inherit from
class MockModule:
    def __init__(self):
        pass
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

# 2. Setup Mock torch
mock_torch = MagicMock()
mock_nn = MagicMock()
mock_nn.Module = MockModule  # Crucial: Real class for inheritance
mock_torch.nn = mock_nn

# 3. Patch sys.modules
sys.modules["torch"] = mock_torch
sys.modules["torch.nn"] = mock_nn
sys.modules["sounddevice"] = MagicMock()
sys.modules["pyaudio"] = MagicMock()

# 4. Import modules AFTER patching
from streaming.memory import StreamingMemory
from streaming.vision import VisionStreamBuffer
from streaming.tts import TTSStreamer
from streaming.joint import JointStreamOrchestrator
from podcast.generator import PodcastGenerator

class TestStreamingComponents(unittest.TestCase):
    
    def test_memory_init(self):
        mem = StreamingMemory(sink_size=4, window_size=128)
        self.assertEqual(mem.sink_size, 4)
        
    def test_vision_buffer(self):
        vis = VisionStreamBuffer(max_frames=5)
        self.assertEqual(vis.buffer.maxlen, 5)
        # Add mock tensor
        mock_tensor = MagicMock()
        vis.add_frame(mock_tensor)
        ctx = vis.get_context()
        # Ensure torch.stack was called
        mock_torch.stack.assert_called()

    def test_tts_init(self):
        tts = TTSStreamer(model_name="Test-Model")
        self.assertEqual(tts.model_name, "Test-Model")
        
    def test_joint_orchestrator(self):
        orch = JointStreamOrchestrator()
        self.assertFalse(orch.is_active)
        self.assertIsNotNone(orch.vision)
        self.assertIsNotNone(orch.tts)

    def test_podcast_generator(self):
        mock_llm = MagicMock()
        gen = PodcastGenerator(mock_llm)
        script = gen.generate_script("Some content", duration_mins=2)
        self.assertTrue(len(script) > 0)
        self.assertIn("speaker", script[0])

if __name__ == '__main__':
    unittest.main()
