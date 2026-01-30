"""
Unit tests for TTS Engine and TTSStreamer.

Tests the text-to-speech synthesis engine with support for:
- Multi-language synthesis
- Voice cloning
- Caching mechanism
- Streaming interface
- Audio format conversion
"""

import pytest
import torch
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch, Mock, mock_open
import time
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class TestVoiceConfig:
    """Tests for VoiceConfig dataclass."""
    
    def test_default_config(self):
        """Test default voice configuration."""
        from src.streaming.tts import VoiceConfig
        config = VoiceConfig()
        
        assert config.voice_id == "default"
        assert config.language == "en"
        assert config.speed == 1.0
        assert config.pitch == 1.0
        assert config.volume == 1.0
        assert config.emotion is None
        assert config.speaker_wav is None
    
    def test_custom_config(self):
        """Test custom voice configuration."""
        from src.streaming.tts import VoiceConfig
        config = VoiceConfig(
            voice_id="custom_voice",
            language="es",
            speed=1.2,
            pitch=0.9,
            volume=0.8,
            emotion="happy",
            speaker_wav="/path/to/voice.wav"
        )
        
        assert config.voice_id == "custom_voice"
        assert config.language == "es"
        assert config.speed == 1.2
        assert config.pitch == 0.9
        assert config.volume == 0.8
        assert config.emotion == "happy"
        assert config.speaker_wav == "/path/to/voice.wav"


class TestAudioOutput:
    """Tests for AudioOutput dataclass."""
    
    def test_default_output(self):
        """Test default audio output."""
        from src.streaming.tts import AudioOutput
        audio_array = np.array([0.1, 0.2, 0.3])
        output = AudioOutput(audio_array=audio_array, sample_rate=24000)
        
        assert np.array_equal(output.audio_array, audio_array)
        assert output.sample_rate == 24000
        assert output.format == "wav"
        assert output.duration_ms == 0.0
        assert output.text_hash == ""
        assert output.voice_id == ""


class TestTTSEngine:
    """Tests for TTSEngine class."""
    
    @pytest.fixture
    def engine(self, tmp_path):
        """Fixture for TTSEngine instance."""
        with patch('src.streaming.tts.COQUI_TTS_AVAILABLE', True):
            from src.streaming.tts import TTSEngine
            return TTSEngine(
                model_key="xtts_v2",
                device="cpu",
                cache_dir=str(tmp_path / "cache"),
                enable_cache=True
            )
    
    @pytest.fixture
    def mock_tts_model(self):
        """Fixture for mocked TTS model."""
        mock = MagicMock()
        mock.synthesizer = MagicMock()
        mock.synthesizer.output_sample_rate = 24000
        mock.tts.return_value = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        return mock
    
    def test_initialization(self, engine, tmp_path):
        """Test TTS engine initialization."""
        assert engine.model_key == "xtts_v2"
        assert engine.device == "cpu"
        assert engine.enable_cache is True
        assert engine.cache_dir == tmp_path / "cache"
        assert engine.cache_dir.exists()
        assert engine._model is None
    
    def test_initialization_auto_device(self, tmp_path):
        """Test TTS engine with auto device selection."""
        with patch('src.streaming.tts.COQUI_TTS_AVAILABLE', True), \
             patch('torch.cuda.is_available', return_value=True):
            
            from src.streaming.tts import TTSEngine
            engine = TTSEngine(cache_dir=str(tmp_path / "cache"))
            
            assert engine.device == "cuda"
    
    def test_initialization_cpu_fallback(self, tmp_path):
        """Test TTS engine falls back to CPU."""
        with patch('src.streaming.tts.COQUI_TTS_AVAILABLE', True), \
             patch('torch.cuda.is_available', return_value=False):
            
            from src.streaming.tts import TTSEngine
            engine = TTSEngine(cache_dir=str(tmp_path / "cache"))
            
            assert engine.device == "cpu"
    
    def test_initialization_without_coqui(self, tmp_path):
        """Test TTS engine raises error when Coqui TTS not available."""
        with patch('src.streaming.tts.COQUI_TTS_AVAILABLE', False):
            from src.streaming.tts import TTSEngine
            
            with pytest.raises(RuntimeError, match="Coqui TTS is required"):
                TTSEngine(cache_dir=str(tmp_path / "cache"))
    
    def test_supported_models(self):
        """Test supported models registry."""
        with patch('src.streaming.tts.COQUI_TTS_AVAILABLE', True):
            from src.streaming.tts import TTSEngine
            
            assert "xtts_v2" in TTSEngine.SUPPORTED_MODELS
            assert "tacotron2" in TTSEngine.SUPPORTED_MODELS
            assert "bark" in TTSEngine.SUPPORTED_MODELS
            
            # Check XTTS-v2 specifics
            xtts = TTSEngine.SUPPORTED_MODELS["xtts_v2"]
            assert xtts["supports_cloning"] is True
            assert "en" in xtts["languages"]
            assert "es" in xtts["languages"]
    
    @patch('src.streaming.tts.TTS')
    def test_load_model_success(self, mock_tts_class, engine, mock_tts_model):
        """Test successful model loading."""
        mock_tts_class.return_value = mock_tts_model
        
        engine.load_model()
        
        assert engine._model is mock_tts_model
        mock_tts_class.assert_called_once_with(
            "tts_models/multilingual/multi-dataset/xtts_v2"
        )
        mock_tts_model.to.assert_called_once_with("cpu")
    
    @patch('src.streaming.tts.TTS')
    def test_load_model_already_loaded(self, mock_tts_class, engine, mock_tts_model):
        """Test loading when model is already loaded."""
        mock_tts_class.return_value = mock_tts_model
        engine._model = mock_tts_model
        
        engine.load_model()
        
        # Should not call TTS again
        mock_tts_class.assert_not_called()
    
    @patch('src.streaming.tts.TTS')
    def test_load_model_failure(self, mock_tts_class, engine):
        """Test handling of model loading failure."""
        mock_tts_class.side_effect = Exception("Download failed")
        
        with pytest.raises(RuntimeError, match="Failed to load TTS model"):
            engine.load_model()
    
    def test_is_model_loaded(self, engine, mock_tts_model):
        """Test model loaded check."""
        assert engine.is_model_loaded() is False
        
        engine._model = mock_tts_model
        assert engine.is_model_loaded() is True
    
    @patch('src.streaming.tts.TTS')
    def test_ensure_loaded(self, mock_tts_class, engine, mock_tts_model):
        """Test lazy loading."""
        mock_tts_class.return_value = mock_tts_model
        
        engine.ensure_loaded()
        
        assert engine._model is mock_tts_model
    
    @patch('src.streaming.tts.TTS')
    def test_synthesize_standard(self, mock_tts_class, engine, mock_tts_model):
        """Test standard synthesis without voice cloning."""
        mock_tts_class.return_value = mock_tts_model
        engine._model = mock_tts_model
        
        from src.streaming.tts import VoiceConfig
        config = VoiceConfig(language="en")
        
        output = engine.synthesize("Hello world", config=config)
        
        assert isinstance(output.audio_array, np.ndarray)
        assert output.sample_rate == 24000
        assert output.text_hash != ""
        assert output.voice_id == "default"
    
    @patch('src.streaming.tts.TTS')
    def test_synthesize_with_cloning(self, mock_tts_class, engine, mock_tts_model, tmp_path):
        """Test synthesis with voice cloning."""
        mock_tts_class.return_value = mock_tts_model
        engine._model = mock_tts_model
        
        # Create a fake speaker wav file
        speaker_path = tmp_path / "speaker.wav"
        speaker_path.write_bytes(b"fake audio data")
        
        from src.streaming.tts import VoiceConfig
        config = VoiceConfig(language="en", speaker_wav=str(speaker_path))
        
        output = engine.synthesize("Hello world", config=config)
        
        assert isinstance(output.audio_array, np.ndarray)
        # Verify cloning was called
        mock_tts_model.tts.assert_called_once()
        call_kwargs = mock_tts_model.tts.call_args.kwargs
        assert "speaker_wav" in call_kwargs
    
    @patch('src.streaming.tts.TTS')
    def test_synthesize_caching(self, mock_tts_class, engine, mock_tts_model):
        """Test synthesis caching."""
        mock_tts_class.return_value = mock_tts_model
        engine._model = mock_tts_model
        
        from src.streaming.tts import VoiceConfig
        config = VoiceConfig()
        
        # First synthesis
        output1 = engine.synthesize("Hello world", config=config)
        
        # Second synthesis should use cache
        output2 = engine.synthesize("Hello world", config=config)
        
        # Model should only be called once
        assert mock_tts_model.tts.call_count == 1
        assert engine.stats["cache_hits"] == 1
        assert engine.stats["cache_misses"] == 1
    
    @patch('src.streaming.tts.TTS')
    def test_synthesize_cache_disabled(self, mock_tts_class, engine, mock_tts_model):
        """Test synthesis with cache disabled."""
        mock_tts_class.return_value = mock_tts_model
        engine._model = mock_tts_model
        engine.enable_cache = False
        
        from src.streaming.tts import VoiceConfig
        config = VoiceConfig()
        
        # Two syntheses
        engine.synthesize("Hello world", config=config)
        engine.synthesize("Hello world", config=config)
        
        # Model should be called twice
        assert mock_tts_model.tts.call_count == 2
    
    @patch('src.streaming.tts.TTS')
    def test_synthesize_different_params_no_cache_match(self, mock_tts_class, engine, mock_tts_model):
        """Test that different parameters create different cache keys."""
        mock_tts_class.return_value = mock_tts_model
        engine._model = mock_tts_model
        
        from src.streaming.tts import VoiceConfig
        
        # Synthesize with different languages
        config1 = VoiceConfig(language="en")
        config2 = VoiceConfig(language="es")
        
        engine.synthesize("Hello", config=config1)
        engine.synthesize("Hello", config=config2)
        
        # Should be cache miss for second
        assert engine.stats["cache_misses"] == 2
    
    @patch('src.streaming.tts.TTS')
    def test_synthesize_with_mp3_output(self, mock_tts_class, engine, mock_tts_model):
        """Test synthesis with MP3 output format."""
        mock_tts_class.return_value = mock_tts_model
        engine._model = mock_tts_model
        
        with patch('src.streaming.tts.PYDUB_AVAILABLE', True), \
             patch('src.streaming.tts.AudioSegment'):
            
            from src.streaming.tts import VoiceConfig
            config = VoiceConfig()
            
            output = engine.synthesize("Hello", config=config, output_format="mp3")
            
            assert output.format == "mp3"
    
    @patch('src.streaming.tts.TTS')
    def test_synthesize_stats_tracking(self, mock_tts_class, engine, mock_tts_model):
        """Test synthesis statistics tracking."""
        mock_tts_class.return_value = mock_tts_model
        engine._model = mock_tts_model
        
        from src.streaming.tts import VoiceConfig
        config = VoiceConfig()
        
        engine.synthesize("Hello", config=config)
        
        assert engine.stats["synthesis_count"] == 1
        assert engine.stats["total_latency_ms"] > 0
    
    def test_synthesize_without_model(self, engine):
        """Test synthesis without loaded model."""
        from src.streaming.tts import VoiceConfig
        config = VoiceConfig()
        
        with pytest.raises(RuntimeError, match="Synthesis failed"):
            engine.synthesize("Hello", config=config)
    
    @patch('src.streaming.tts.TTS')
    def test_clone_voice(self, mock_tts_class, engine, mock_tts_model, tmp_path):
        """Test voice cloning registration."""
        mock_tts_class.return_value = mock_tts_model
        engine._model = mock_tts_model
        
        # Create fake audio file
        audio_path = tmp_path / "reference.wav"
        audio_path.write_bytes(b"fake audio")
        
        voice_id = engine.clone_voice(str(audio_path), "MyVoice", language="en")
        
        assert voice_id.startswith("cloned_MyVoice_")
        assert voice_id in engine.voice_registry
        assert engine.voice_registry[voice_id]["name"] == "MyVoice"
        assert engine.voice_registry[voice_id]["language"] == "en"
    
    def test_clone_voice_file_not_found(self, engine):
        """Test cloning with non-existent file."""
        with pytest.raises(FileNotFoundError):
            engine.clone_voice("/nonexistent/voice.wav", "MyVoice")
    
    @patch('src.streaming.tts.TTS')
    def test_get_available_voices(self, mock_tts_class, engine, mock_tts_model):
        """Test getting available voices."""
        mock_tts_class.return_value = mock_tts_model
        engine._model = mock_tts_model
        
        voices = engine.get_available_voices()
        
        assert len(voices) >= 1
        assert voices[0]["id"] == "default"
        assert voices[0]["type"] == "builtin"
    
    @patch('src.streaming.tts.TTS')
    def test_get_available_voices_with_cloned(self, mock_tts_class, engine, mock_tts_model, tmp_path):
        """Test getting voices including cloned ones."""
        mock_tts_class.return_value = mock_tts_model
        engine._model = mock_tts_model
        
        # Add a cloned voice
        audio_path = tmp_path / "ref.wav"
        audio_path.write_bytes(b"audio")
        engine.clone_voice(str(audio_path), "TestVoice")
        
        voices = engine.get_available_voices()
        
        cloned_voices = [v for v in voices if v["type"] == "cloned"]
        assert len(cloned_voices) == 1
        assert cloned_voices[0]["name"] == "TestVoice"
    
    @patch('src.streaming.tts.TTS')
    def test_get_stats(self, mock_tts_class, engine, mock_tts_model):
        """Test getting engine statistics."""
        mock_tts_class.return_value = mock_tts_model
        engine._model = mock_tts_model
        
        from src.streaming.tts import VoiceConfig
        config = VoiceConfig()
        
        engine.synthesize("Hello", config=config)
        
        stats = engine.get_stats()
        
        assert stats["synthesis_count"] == 1
        assert "avg_latency_ms" in stats
        assert stats["avg_latency_ms"] > 0
        assert "cache_hit_rate" in stats
    
    def test_get_stats_no_synthesis(self, engine):
        """Test stats when no synthesis has occurred."""
        stats = engine.get_stats()
        
        assert stats["synthesis_count"] == 0
        assert stats["avg_latency_ms"] == 0.0
    
    @patch('src.streaming.tts.TTS')
    def test_clear_cache(self, mock_tts_class, engine, mock_tts_model, tmp_path):
        """Test clearing cache."""
        mock_tts_class.return_value = mock_tts_model
        engine._model = mock_tts_model
        
        # Add something to cache
        from src.streaming.tts import VoiceConfig, AudioOutput
        config = VoiceConfig()
        engine._cache["test"] = AudioOutput(
            audio_array=np.array([0.1, 0.2]),
            sample_rate=24000
        )
        
        # Create cache file
        cache_file = engine.cache_dir / "test.wav"
        cache_file.write_bytes(b"audio")
        
        engine.clear_cache()
        
        assert len(engine._cache) == 0
        assert not cache_file.exists()
    
    def test_clear_cache_empty(self, engine):
        """Test clearing empty cache."""
        # Should not raise error
        engine.clear_cache()


class TestTTSStreamer:
    """Tests for TTSStreamer class."""
    
    @pytest.fixture
    def mock_engine(self):
        """Fixture for mocked TTSEngine."""
        mock = MagicMock()
        
        from src.streaming.tts import AudioOutput
        mock.synthesize.return_value = AudioOutput(
            audio_array=np.array([0.1, 0.2, 0.3]),
            sample_rate=24000,
            duration_ms=100.0
        )
        
        return mock
    
    @pytest.fixture
    def streamer(self, mock_engine):
        """Fixture for TTSStreamer instance."""
        from src.streaming.tts import TTSStreamer
        return TTSStreamer(
            model_name="TestModel",
            engine=mock_engine,
            enable_streaming=True
        )
    
    def test_initialization(self, streamer, mock_engine):
        """Test TTSStreamer initialization."""
        assert streamer.model_name == "TestModel"
        assert streamer.engine == mock_engine
        assert streamer.enable_streaming is True
        assert streamer.is_running is False
    
    def test_initialization_default_engine(self):
        """Test TTSStreamer with default engine."""
        with patch('src.streaming.tts.TTSEngine') as mock_engine_class:
            mock_engine = MagicMock()
            mock_engine_class.return_value = mock_engine
            
            from src.streaming.tts import TTSStreamer
            streamer = TTSStreamer()
            
            mock_engine_class.assert_called_once_with(model_key="xtts_v2")
    
    def test_synthesize_stream(self, streamer):
        """Test streaming synthesis."""
        def text_generator():
            yield "Hello"
            yield " world"
            yield "!"
        
        streamer.synthesize_stream(text_generator())
        
        assert streamer.is_running is False  # Should be False after completion
        assert streamer.engine.synthesize.call_count == 1  # All combined
    
    def test_synthesize_stream_with_punctuation(self, streamer):
        """Test streaming with punctuation triggers."""
        def text_generator():
            yield "Hello."
            yield " How are you?"
        
        streamer.synthesize_stream(text_generator())
        
        # Should synthesize on each punctuation
        assert streamer.engine.synthesize.call_count == 2
    
    def test_synthesize_stream_error_handling(self, streamer):
        """Test error handling during streaming."""
        streamer.engine.synthesize.side_effect = Exception("Synthesis error")
        
        def text_generator():
            yield "Hello"
        
        # Should not raise, but handle gracefully
        streamer.synthesize_stream(text_generator())
    
    def test_get_audio_stream(self, streamer):
        """Test getting audio stream."""
        # Add some audio to queue
        streamer.audio_queue.put({
            "audio": np.array([0.1, 0.2]),
            "sample_rate": 24000,
            "text": "Hello",
            "duration_ms": 100.0
        })
        
        streamer.is_running = False  # Signal completion
        
        chunks = list(streamer.get_audio_stream())
        
        assert len(chunks) == 1
        assert chunks[0]["text"] == "Hello"
    
    @patch('src.streaming.tts.sf')
    def test_synthesize_to_file_wav(self, mock_sf, streamer, tmp_path):
        """Test synthesizing to WAV file."""
        output_path = tmp_path / "output.wav"
        
        result = streamer.synthesize_to_file("Hello world", str(output_path))
        
        assert result == str(output_path)
        assert Path(result).exists()
    
    @patch('src.streaming.tts.sf')
    def test_synthesize_to_file_mp3(self, mock_sf, streamer, tmp_path):
        """Test synthesizing to MP3 file."""
        output_path = tmp_path / "output.mp3"
        
        result = streamer.synthesize_to_file("Hello world", str(output_path))
        
        assert result == str(output_path)
        assert Path(result).suffix == ".mp3"
    
    @patch('src.streaming.tts.sf')
    def test_synthesize_to_file_invalid_format(self, mock_sf, streamer, tmp_path):
        """Test synthesizing with invalid format defaults to WAV."""
        output_path = tmp_path / "output.xyz"
        
        result = streamer.synthesize_to_file("Hello world", str(output_path))
        
        assert Path(result).suffix == ".wav"
    
    def test_clone_voice_passes_to_engine(self, streamer, mock_engine, tmp_path):
        """Test that clone_voice passes to engine."""
        audio_path = tmp_path / "ref.wav"
        audio_path.write_bytes(b"audio")
        
        streamer.clone_voice(str(audio_path), "TestVoice")
        
        mock_engine.clone_voice.assert_called_once_with(str(audio_path), "TestVoice")


class TestTTSEngineFormatConversion:
    """Tests for audio format conversion."""
    
    @pytest.fixture
    def engine_with_model(self):
        """Engine with mocked model."""
        with patch('src.streaming.tts.COQUI_TTS_AVAILABLE', True), \
             patch('src.streaming.tts.TTS') as mock_tts_class:
            
            mock_model = MagicMock()
            mock_model.synthesizer.output_sample_rate = 24000
            mock_model.tts.return_value = np.array([0.1, 0.2, 0.3])
            mock_tts_class.return_value = mock_model
            
            from src.streaming.tts import TTSEngine
            engine = TTSEngine(device="cpu")
            engine._model = mock_model
            
            return engine
    
    def test_convert_format_wav(self, engine_with_model):
        """Test WAV format conversion (no change)."""
        audio = np.array([0.1, 0.2, 0.3])
        result = engine_with_model._convert_format(audio, 24000, "wav")
        
        assert np.array_equal(result, audio)
    
    @patch('src.streaming.tts.PYDUB_AVAILABLE', True)
    @patch('src.streaming.tts.AudioSegment')
    def test_convert_format_mp3(self, mock_audio_segment, engine_with_model):
        """Test MP3 format conversion."""
        mock_segment = MagicMock()
        mock_audio_segment.from_wav.return_value = mock_segment
        
        mock_buffer = MagicMock()
        mock_buffer.getvalue.return_value = b"mp3data"
        mock_segment.export.return_value = mock_buffer
        
        audio = np.array([0.1, 0.2, 0.3])
        result = engine_with_model._convert_format(audio, 24000, "mp3")
        
        assert result == b"mp3data"
    
    @patch('src.streaming.tts.PYDUB_AVAILABLE', False)
    def test_convert_format_mp3_no_pydub(self, engine_with_model):
        """Test MP3 conversion falls back to WAV when pydub unavailable."""
        audio = np.array([0.1, 0.2, 0.3])
        result = engine_with_model._convert_format(audio, 24000, "mp3")
        
        # Should return original array
        assert np.array_equal(result, audio)
    
    @patch('src.streaming.tts.sf')
    def test_save_to_disk_cache(self, mock_sf, engine_with_model, tmp_path):
        """Test saving to disk cache."""
        engine_with_model.cache_dir = tmp_path
        
        from src.streaming.tts import AudioOutput
        output = AudioOutput(
            audio_array=np.array([0.1, 0.2, 0.3]),
            sample_rate=24000
        )
        
        engine_with_model._save_to_disk_cache("test_hash", output)
        
        cache_file = tmp_path / "test_hash.wav"
        assert cache_file.exists()


class TestFactoryFunctions:
    """Tests for factory functions."""
    
    @patch('src.streaming.tts.COQUI_TTS_AVAILABLE', True)
    @patch('src.streaming.tts.TTS')
    def test_create_tts_engine(self, mock_tts_class):
        """Test create_tts_engine factory."""
        from src.streaming.tts import create_tts_engine
        
        engine = create_tts_engine(model="xtts_v2", device="cpu")
        
        assert engine.model_key == "xtts_v2"
        assert engine.device == "cpu"
    
    @patch('src.streaming.tts.COQUI_TTS_AVAILABLE', True)
    @patch('src.streaming.tts.TTS')
    def test_create_streamer(self, mock_tts_class):
        """Test create_streamer factory."""
        from src.streaming.tts import create_streamer, TTSEngine
        
        mock_engine = MagicMock()
        streamer = create_streamer(engine=mock_engine)
        
        assert streamer.engine == mock_engine
    
    @patch('src.streaming.tts.COQUI_TTS_AVAILABLE', True)
    @patch('src.streaming.tts.TTS')
    def test_create_streamer_default(self, mock_tts_class):
        """Test create_streamer with default engine."""
        from src.streaming.tts import create_streamer
        
        streamer = create_streamer()
        
        assert streamer.engine is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
