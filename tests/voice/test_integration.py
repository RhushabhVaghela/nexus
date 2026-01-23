
import unittest
import sys
import os
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add src to path
sys.path.append(str(Path(__file__).parents[2] / "src"))

class TestVoiceIntegration(unittest.TestCase):
    
    def test_podcast_synthesizer_personaplex_backend(self):
        from podcast.synthesizer import synthesize_tts
        
        # Mock the voice engine components
        with patch('voice_engine.registry.voice_registry.get_voice_dna') as mock_get_dna, \
             patch('voice_engine.vibe_modulator.vibe_modulator.get_vibe_params') as mock_get_params:
            
            mock_get_dna.return_value = "builtin://NATF0"
            mock_get_params.return_value = {"pitch": 1.1}
            
            out_dir = Path("/tmp/podcast_test")
            out_dir.mkdir(parents=True, exist_ok=True)
            
            audio_path = synthesize_tts(
                speaker="Host A",
                text="Hello, this is a test!",
                out_dir=out_dir,
                tts_backend="personaplex",
                voice_map={"Host A": "NATF0"}
            )
            
            self.assertTrue(audio_path.exists())
            self.assertEqual(audio_path.suffix, ".wav")
            
            # Clean up
            import shutil
            shutil.rmtree(out_dir)

    def test_streaming_orchestrator_dynamic_voice(self):
        from streaming.joint import JointStreamingOrchestrator, VisionStreamBuffer, AudioStreamBuffer, UserEventBuffer
        
        vis_buf = VisionStreamBuffer()
        aud_buf = AudioStreamBuffer()
        user_buf = UserEventBuffer()
        
        # Mock LLM and TTS
        mock_llm = MagicMock(return_value="Response")
        mock_tts = MagicMock()
        
        orch = JointStreamingOrchestrator(
            vision_buffer=vis_buf,
            audio_buffer=aud_buf,
            user_buffer=user_buf,
            llm_fn=mock_llm,
            tts_engine=mock_tts
        )
        
        # Initial state
        self.assertEqual(orch.active_voice, "NATM1")
        
        # Switch voice
        orch.switch_voice("NATF2")
        self.assertEqual(orch.active_voice, "NATF2")
        
        # Set vibe
        orch.set_vibe("excited")
        self.assertEqual(orch.active_vibe, "excited")
        
        # Simulate one loop iteration
        with patch('time.sleep', side_effect=lambda x: orch._stop_event.set()):
            orch._run_loop()
            
        # Verify TTS was called with correct voice/vibe
        mock_tts.synthesize.assert_called_with(
            text="Response",
            voice="NATF2",
            vibe="excited",
            sync_mode="high_fidelity"
        )

    def test_podcast_generator_vibe_output(self):
        from podcast.generator import generate_podcast_script
        
        mock_llm = MagicMock()
        # Ensure LLM returns the new JSON structure with 'vibe'
        mock_llm.return_value = json.dumps({
            "turns": [
                {"speaker": "Host A", "text": "Wow!", "vibe": "excited"},
                {"speaker": "Host B", "text": "Indeed.", "vibe": "thoughtful"}
            ]
        })
        
        script = generate_podcast_script(["Content"], llm=mock_llm)
        
        self.assertEqual(len(script.turns), 2)
        self.assertEqual(script.turns[0].vibe, "excited")
        self.assertEqual(script.turns[1].vibe, "thoughtful")

if __name__ == '__main__':
    unittest.main()
