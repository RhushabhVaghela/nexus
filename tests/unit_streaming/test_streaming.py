
import unittest
from unittest.mock import MagicMock, patch
import sys
from pathlib import Path
import importlib

# Add src to path
sys.path.append(str(Path(__file__).parents[2] / "src"))

class TestStreamingComponents(unittest.TestCase):
    
    def setUp(self):
        # Patch sound-related sys.modules only to avoid breaking torch internals
        self.patcher = patch.dict(sys.modules, {
            "sounddevice": MagicMock(),
            "pyaudio": MagicMock()
        })
        self.patcher.start()

        # Mock objects for some tests if needed
        self.mock_torch = MagicMock()
        # Ensure it has a version if used
        self.mock_torch.__version__ = "2.5.1"

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
        with patch('torch.stack') as mock_stack:
            ctx = vis.get_context()
            # Ensure torch.stack was called
            mock_stack.assert_called()

    def test_tts_init(self):
        from streaming.tts import TTSStreamer
        tts = TTSStreamer(model_name="Test-Model")
        self.assertEqual(tts.model_name, "Test-Model")
        
    def test_joint_orchestrator(self):
        from streaming.joint import JointStreamingOrchestrator, VisionStreamBuffer, AudioStreamBuffer, UserEventBuffer
        
        vis_buf = VisionStreamBuffer()
        aud_buf = AudioStreamBuffer()
        user_buf = UserEventBuffer()
        
        orch = JointStreamingOrchestrator(
            vision_buffer=vis_buf,
            audio_buffer=aud_buf,
            user_buffer=user_buf
        )
        self.assertIsNotNone(orch.vision_buffer)
        self.assertIsNotNone(orch.audio_buffer)
        self.assertIsNotNone(orch.user_buffer)

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
