"""
Interactive Podcast Player with VAD Interruption
"""
import time
import threading
from typing import List, Dict

# Mock VAD (Voice Activity Detection)
class VADMonitor:
    def is_user_speaking(self):
        # In reality: check microphone input level
        return False 

class PodcastPlayer:
    def __init__(self, tts_engine):
        self.tts = tts_engine
        self.vad = VADMonitor()
        self.is_playing = False
        self.interrupted = False

    def play_episode(self, script: List[Dict[str, str]]):
        """
        Plays the podcast script, handling interruptions.
        """
        self.is_playing = True
        print("\n🎧 Playing Podcast Episode...")
        
        for turn in script:
            if not self.is_playing: break
            
            speaker = turn["speaker"]
            text = turn["text"]
            
            # 1. Synthesize Audio
            print(f"[{speaker}]: {text}")
            # audio = self.tts.synthesize(text, voice_id=speaker)
            
            # 2. Check for Interruption (Simulated Playback Loop)
            # We break the audio into chunks to check VAD frequently
            duration_sim = len(text) * 0.05
            check_interval = 0.1
            elapsed = 0
            
            while elapsed < duration_sim:
                if self.vad.is_user_speaking():
                    self._handle_interruption()
                    break
                time.sleep(check_interval)
                elapsed += check_interval
                
            if self.interrupted:
                break
                
        self.is_playing = False
        print("⏹️  Podcast Ended.")

    def _handle_interruption(self):
        print("\n🔴 USER INTERRUPTED! Pausing Podcast...")
        self.is_playing = False
        self.interrupted = True
        # Logic to trigger LLM Chat would go here
