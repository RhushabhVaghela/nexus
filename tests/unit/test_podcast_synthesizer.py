import pytest
import os
import json
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch
from src.podcast.synthesizer import AudioTurn, synthesize_tts, PodcastPlayer, play_audio
from src.podcast.generator import PodcastScript, Turn

@pytest.fixture
def temp_audio_dir(tmp_path):
    d = tmp_path / "audio"
    d.mkdir()
    return d

def test_audio_turn_dataclass():
    path = Path("test.wav")
    turn = AudioTurn(speaker="Host A", text="Hi", audio_path=path)
    assert turn.speaker == "Host A"
    assert turn.text == "Hi"
    assert turn.audio_path == path

@patch("requests.post")
def test_synthesize_tts_http(mock_post, temp_audio_dir):
    mock_resp = MagicMock()
    mock_resp.content = b"fake audio data"
    mock_post.return_value = mock_resp
    
    path = synthesize_tts("Host A", "Hello", temp_audio_dir, tts_backend="http")
    
    assert path.exists()
    assert path.read_bytes() == b"fake audio data"
    assert mock_post.called

def test_synthesize_tts_personaplex(temp_audio_dir):
    # Mocking dependencies for personaplex
    with patch("voice_engine.registry.voice_registry.get_voice_dna", return_value="dna"), \
         patch("voice_engine.vibe_modulator.vibe_modulator.get_vibe_params", return_value={}):
        path = synthesize_tts("Host A", "Hello", temp_audio_dir, tts_backend="personaplex")
        assert path.exists()
        assert path.read_bytes().startswith(b"RIFF")

@patch("subprocess.run")
def test_synthesize_tts_cli(mock_run, temp_audio_dir):
    path = synthesize_tts("Host A", "Hello", temp_audio_dir, tts_backend="cli")
    assert mock_run.called
    # Note: cli doesn't write the file in mock mode, but the path is returned

@patch("subprocess.run")
@patch("shutil.which", return_value="aplay")
@patch("platform.system", return_value="Linux")
def test_play_audio_linux(mock_system, mock_which, mock_run, tmp_path):
    f = tmp_path / "test.wav"
    f.touch()
    play_audio(f)
    assert mock_run.called
    assert mock_run.call_args[0][0][0] == "aplay"

class TestPodcastPlayer:
    def test_player_init(self, temp_audio_dir):
        script = PodcastScript(turns=[Turn("Host A", "Hi")])
        player = PodcastPlayer(script, temp_audio_dir)
        assert player.script == script
        assert player.audio_dir == temp_audio_dir

    @patch("src.podcast.synthesizer.synthesize_tts")
    @patch("src.podcast.synthesizer.play_audio")
    def test_player_run_loop(self, mock_play, mock_synth, temp_audio_dir):
        mock_synth.return_value = temp_audio_dir / "fake.wav"
        (temp_audio_dir / "fake.wav").touch()
        
        script = PodcastScript(turns=[Turn("Host A", "Hi")])
        player = PodcastPlayer(script, temp_audio_dir)
        
        # Run in a thread or just call internal methods to avoid blocking
        player._enqueue_next_turns_if_needed()
        assert player._queue.qsize() == 1
        
        # Mock stop event to exit loop after one item
        player.start()
        time.sleep(0.5)
        player.stop()
        
        assert mock_play.called

    @patch("src.podcast.synthesizer.handle_user_interrupt")
    @patch("src.podcast.synthesizer.synthesize_tts")
    def test_player_interrupt(self, mock_synth, mock_interrupt, temp_audio_dir):
        mock_synth.return_value = temp_audio_dir / "inter.wav"
        (temp_audio_dir / "inter.wav").touch()
        
        mock_interrupt.return_value = PodcastScript(turns=[Turn("Host B", "Replied")])
        
        script = PodcastScript(turns=[Turn("Host A", "Hi")])
        player = PodcastPlayer(script, temp_audio_dir)
        
        player.on_user_text("Interrupt")
        
        assert len(player.script.turns) == 2
        assert player.script.turns[1].text == "Replied"
        assert player._queue.qsize() == 1
