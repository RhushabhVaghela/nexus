"""
Real-Time Text-to-Speech Interface
Models: Chatterbox-Turbo / Coqui XTTS-v2
Target Latency: <200ms
"""
import os
import io
import time
import threading
import queue
import hashlib
import warnings
from typing import Optional, Generator, Dict, Any, List, Union
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import torch
import torchaudio

# TTS Model imports (Coqui TTS)
try:
    from TTS.api import TTS
    from TTS.utils.manage import ModelManager
    COQUI_TTS_AVAILABLE = True
except ImportError:
    COQUI_TTS_AVAILABLE = False
    warnings.warn("Coqui TTS not installed. Run: pip install TTS")

# Audio format handling
try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except ImportError:
    SOUNDFILE_AVAILABLE = False

try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False


@dataclass
class VoiceConfig:
    """Configuration for voice synthesis."""
    voice_id: str = "default"
    language: str = "en"
    speed: float = 1.0
    pitch: float = 1.0
    volume: float = 1.0
    emotion: Optional[str] = None
    speaker_wav: Optional[str] = None  # For voice cloning


@dataclass
class AudioOutput:
    """Container for audio output with metadata."""
    audio_array: np.ndarray
    sample_rate: int
    format: str = "wav"
    duration_ms: float = 0.0
    text_hash: str = ""
    voice_id: str = ""


class TTSEngine:
    """
    Production-grade Text-to-Speech Engine.
    
    Supports:
    - Coqui TTS (XTTS-v2) for voice cloning
    - Multi-language synthesis
    - Voice caching for low-latency responses
    - Multiple output formats (WAV, MP3, PCM)
    """
    
    # Model registry
    SUPPORTED_MODELS = {
        "xtts_v2": {
            "model_name": "tts_models/multilingual/multi-dataset/xtts_v2",
            "supports_cloning": True,
            "languages": ["en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl", "cs", "ar", "zh", "ja", "hu", "ko"],
            "description": "Coqui XTTS-v2 - High quality multilingual TTS with voice cloning"
        },
        "tacotron2": {
            "model_name": "tts_models/en/ljspeech/tacotron2-DDC",
            "supports_cloning": False,
            "languages": ["en"],
            "description": "Tacotron2 - Fast English TTS"
        },
        "bark": {
            "model_name": "tts_models/multilingual/multi-dataset/bark",
            "supports_cloning": False,
            "languages": ["en", "zh", "fr", "de", "hi", "it", "ja", "ko", "pl", "pt", "ru", "es", "tr"],
            "description": "Bark - Multilingual TTS with expressive capabilities"
        }
    }
    
    def __init__(
        self,
        model_key: str = "xtts_v2",
        device: Optional[str] = None,
        cache_dir: Optional[str] = None,
        enable_cache: bool = True
    ):
        """
        Initialize TTS Engine.
        
        Args:
            model_key: Key from SUPPORTED_MODELS
            device: 'cuda', 'cpu', or None (auto)
            cache_dir: Directory for voice caching
            enable_cache: Whether to enable synthesis caching
        """
        self.model_key = model_key
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.enable_cache = enable_cache
        self._model = None
        self._model_manager = None
        
        # Cache setup
        if cache_dir:
            self.cache_dir = Path(cache_dir)
        else:
            self.cache_dir = Path.home() / ".cache" / "nexus_tts"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache: Dict[str, AudioOutput] = {}
        
        # Voice registry integration
        self.voice_registry: Dict[str, Dict[str, Any]] = {}
        
        # Stats
        self.stats = {
            "synthesis_count": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "total_latency_ms": 0.0
        }
        
        print(f"🔊 Initializing TTSEngine: {model_key}")
        print(f"   Device: {self.device}")
        print(f"   Cache: {self.cache_dir}")
        
        if not COQUI_TTS_AVAILABLE:
            raise RuntimeError(
                "Coqui TTS is required but not installed. "
                "Install with: pip install TTS"
            )
    
    def load_model(self, progress_bar: bool = False) -> None:
        """
        Load the TTS model into memory.
        
        Args:
            progress_bar: Show download progress
        """
        if self._model is not None:
            return
        
        model_info = self.SUPPORTED_MODELS.get(self.model_key)
        if not model_info:
            raise ValueError(f"Unknown model: {self.model_key}")
        
        model_name = model_info["model_name"]
        print(f"⬇️  Loading model: {model_name}")
        
        try:
            self._model = TTS(model_name).to(self.device)
            print(f"✅ Model loaded successfully on {self.device}")
        except Exception as e:
            raise RuntimeError(f"Failed to load TTS model: {e}")
    
    def is_model_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._model is not None
    
    def ensure_loaded(self) -> None:
        """Lazy load model if not already loaded."""
        if not self.is_model_loaded():
            self.load_model()
    
    def synthesize(
        self,
        text: str,
        config: Optional[VoiceConfig] = None,
        output_format: str = "wav",
        use_cache: Optional[bool] = None
    ) -> AudioOutput:
        """
        Synthesize text to speech.
        
        Args:
            text: Text to synthesize
            config: Voice configuration
            output_format: Output format (wav, mp3, pcm)
            use_cache: Override default cache setting
        
        Returns:
            AudioOutput containing audio array and metadata
        """
        start_time = time.time()
        config = config or VoiceConfig()
        use_cache = use_cache if use_cache is not None else self.enable_cache
        
        # Generate cache key
        text_hash = hashlib.md5(
            f"{text}:{config.voice_id}:{config.language}:{config.speed}".encode()
        ).hexdigest()
        
        # Check cache
        if use_cache and text_hash in self._cache:
            self.stats["cache_hits"] += 1
            cached = self._cache[text_hash]
            cached.text_hash = text_hash
            return cached
        
        self.ensure_loaded()
        self.stats["cache_misses"] += 1
        
        # Prepare synthesis parameters
        model_info = self.SUPPORTED_MODELS[self.model_key]
        
        try:
            if model_info["supports_cloning"] and config.speaker_wav:
                # Voice cloning synthesis
                audio_array = self._synthesize_cloning(text, config)
            else:
                # Standard synthesis
                audio_array = self._synthesize_standard(text, config)
            
            # Calculate duration
            sample_rate = self._model.synthesizer.output_sample_rate or 24000
            duration_ms = (len(audio_array) / sample_rate) * 1000
            
            # Convert format if needed
            if output_format != "pcm":
                audio_array = self._convert_format(audio_array, sample_rate, output_format)
            
            # Create output
            output = AudioOutput(
                audio_array=audio_array,
                sample_rate=sample_rate,
                format=output_format,
                duration_ms=duration_ms,
                text_hash=text_hash,
                voice_id=config.voice_id
            )
            
            # Cache result
            if use_cache:
                self._cache[text_hash] = output
                self._save_to_disk_cache(text_hash, output)
            
            # Update stats
            self.stats["synthesis_count"] += 1
            latency_ms = (time.time() - start_time) * 1000
            self.stats["total_latency_ms"] += latency_ms
            
            print(f"🔊 Synthesized: '{text[:50]}...' ({duration_ms:.0f}ms, {latency_ms:.0f}ms latency)")
            
            return output
            
        except Exception as e:
            raise RuntimeError(f"Synthesis failed: {e}")
    
    def _synthesize_standard(self, text: str, config: VoiceConfig) -> np.ndarray:
        """Standard TTS synthesis without voice cloning."""
        # Some models require language parameter
        model_info = self.SUPPORTED_MODELS[self.model_key]
        
        if config.language in model_info.get("languages", ["en"]):
            audio_array = self._model.tts(text, language=config.language)
        else:
            audio_array = self._model.tts(text)
        
        return np.array(audio_array)
    
    def _synthesize_cloning(self, text: str, config: VoiceConfig) -> np.ndarray:
        """TTS with voice cloning using speaker reference."""
        if not config.speaker_wav or not Path(config.speaker_wav).exists():
            # Fall back to standard synthesis
            return self._synthesize_standard(text, config)
        
        audio_array = self._model.tts(
            text=text,
            speaker_wav=config.speaker_wav,
            language=config.language
        )
        
        return np.array(audio_array)
    
    def _convert_format(
        self,
        audio_array: np.ndarray,
        sample_rate: int,
        output_format: str
    ) -> np.ndarray:
        """Convert audio to desired format."""
        if output_format == "wav":
            return audio_array
        elif output_format == "mp3":
            return self._to_mp3(audio_array, sample_rate)
        return audio_array
    
    def _to_mp3(self, audio_array: np.ndarray, sample_rate: int) -> bytes:
        """Convert numpy array to MP3 bytes."""
        if not PYDUB_AVAILABLE:
            warnings.warn("pydub not available, returning WAV format")
            return audio_array
        
        # Convert to bytes
        wav_buffer = io.BytesIO()
        if SOUNDFILE_AVAILABLE:
            sf.write(wav_buffer, audio_array, sample_rate, format="WAV")
        else:
            # Fallback using torchaudio
            tensor = torch.from_numpy(audio_array).unsqueeze(0)
            torchaudio.save(wav_buffer, tensor, sample_rate, format="wav")
        
        wav_buffer.seek(0)
        
        # Convert to MP3
        audio_segment = AudioSegment.from_wav(wav_buffer)
        mp3_buffer = io.BytesIO()
        audio_segment.export(mp3_buffer, format="mp3", bitrate="192k")
        
        return mp3_buffer.getvalue()
    
    def _save_to_disk_cache(self, text_hash: str, output: AudioOutput) -> None:
        """Save synthesized audio to disk cache."""
        cache_path = self.cache_dir / f"{text_hash}.wav"
        try:
            if isinstance(output.audio_array, np.ndarray):
                if SOUNDFILE_AVAILABLE:
                    sf.write(str(cache_path), output.audio_array, output.sample_rate)
                else:
                    tensor = torch.from_numpy(output.audio_array).unsqueeze(0)
                    torchaudio.save(str(cache_path), tensor, output.sample_rate)
        except Exception as e:
            warnings.warn(f"Failed to save cache: {e}")
    
    def clone_voice(
        self,
        audio_path: str,
        voice_name: str,
        language: str = "en"
    ) -> str:
        """
        Register a voice for cloning from an audio sample.
        
        Args:
            audio_path: Path to reference audio file
            voice_name: Name to register the voice under
            language: Primary language of the voice
        
        Returns:
            voice_id for use in synthesis
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        voice_id = f"cloned_{voice_name}_{hashlib.md5(voice_name.encode()).hexdigest()[:8]}"
        
        self.voice_registry[voice_id] = {
            "name": voice_name,
            "speaker_wav": str(audio_path),
            "language": language,
            "type": "cloned"
        }
        
        print(f"🎤 Voice cloned: {voice_name} (ID: {voice_id})")
        return voice_id
    
    def get_available_voices(self) -> List[Dict[str, Any]]:
        """Get list of available voices."""
        voices = []
        
        # Add default voices
        voices.append({
            "id": "default",
            "name": "Default",
            "type": "builtin",
            "supports_cloning": self.SUPPORTED_MODELS[self.model_key]["supports_cloning"]
        })
        
        # Add cloned voices
        for voice_id, voice_data in self.voice_registry.items():
            voices.append({
                "id": voice_id,
                "name": voice_data["name"],
                "type": "cloned",
                "language": voice_data.get("language", "en")
            })
        
        return voices
    
    def get_stats(self) -> Dict[str, Any]:
        """Get synthesis statistics."""
        stats = self.stats.copy()
        if stats["synthesis_count"] > 0:
            stats["avg_latency_ms"] = stats["total_latency_ms"] / stats["synthesis_count"]
        else:
            stats["avg_latency_ms"] = 0.0
        stats["cache_hit_rate"] = (
            stats["cache_hits"] / (stats["cache_hits"] + stats["cache_misses"])
            if (stats["cache_hits"] + stats["cache_misses"]) > 0 else 0.0
        )
        return stats
    
    def clear_cache(self) -> None:
        """Clear in-memory and disk cache."""
        self._cache.clear()
        if self.cache_dir.exists():
            for cache_file in self.cache_dir.glob("*.wav"):
                cache_file.unlink()
        print("🗑️  Cache cleared")


class TTSStreamer:
    """
    Async TTS Streamer for Real-Time Voice Synthesis.
    Integrates with TTSEngine for production use.
    """
    
    def __init__(
        self,
        model_name: str = "Chatterbox-Turbo",
        engine: Optional[TTSEngine] = None,
        enable_streaming: bool = True
    ):
        self.model_name = model_name
        self.audio_queue = queue.Queue()
        self.is_running = False
        self.enable_streaming = enable_streaming
        
        # Initialize engine
        if engine:
            self.engine = engine
        else:
            # Default to XTTS-v2 for best quality and voice cloning
            self.engine = TTSEngine(model_key="xtts_v2")
        
        print(f"🔊 Initializing TTS Streamer: {model_name}")
        print(f"   Engine: {self.engine.model_key}")
        print(f"   Streaming: {enable_streaming}")
    
    def synthesize_stream(self, text_iterator: Generator[str, None, None]):
        """
        Consumes text tokens from LLM and generates audio chunks ASAP.
        
        Args:
            text_iterator: Generator yielding text tokens
        """
        self.is_running = True
        buffer = ""
        chunk_count = 0
        
        try:
            for token in text_iterator:
                buffer += token
                
                # Heuristic: Synthesize on punctuation for natural phrasing
                if any(p in token for p in [".", "!", "?", "\n"]):
                    self._generate_audio(buffer.strip())
                    buffer = ""
                    chunk_count += 1
                elif "," in token and len(buffer) > 50:
                    # Also synthesize on long comma-separated phrases
                    self._generate_audio(buffer.strip())
                    buffer = ""
                    chunk_count += 1
            
            # Process remaining buffer
            if buffer.strip():
                self._generate_audio(buffer.strip())
                chunk_count += 1
                
        except Exception as e:
            print(f"❌ Streaming error: {e}")
            raise
        finally:
            self.is_running = False
            print(f"✅ Streaming complete: {chunk_count} chunks synthesized")
    
    def _generate_audio(self, text: str):
        """
        Internal: Call TTS engine to generate audio.
        
        Args:
            text: Text to synthesize
        """
        if not text.strip():
            return
        
        try:
            # Use the TTSEngine for synthesis
            config = VoiceConfig(language="en")
            output = self.engine.synthesize(text, config=config)
            
            # Put audio data in queue
            self.audio_queue.put({
                "audio": output.audio_array,
                "sample_rate": output.sample_rate,
                "text": text,
                "duration_ms": output.duration_ms
            })
            
        except Exception as e:
            print(f"❌ Synthesis error for text '{text[:50]}...': {e}")
            # Continue processing other chunks even if one fails
    
    def get_audio_stream(self) -> Generator[Dict[str, Any], None, None]:
        """
        Yields audio chunks for playback.
        
        Yields:
            Dict containing audio array, sample_rate, text, duration_ms
        """
        while self.is_running or not self.audio_queue.empty():
            try:
                chunk = self.audio_queue.get(timeout=0.1)
                yield chunk
            except queue.Empty:
                continue
    
    def synthesize_to_file(
        self,
        text: str,
        output_path: str,
        voice_config: Optional[VoiceConfig] = None
    ) -> str:
        """
        Synthesize text to audio file.
        
        Args:
            text: Text to synthesize
            output_path: Output file path
            voice_config: Voice configuration
        
        Returns:
            Path to saved audio file
        """
        config = voice_config or VoiceConfig()
        output_path = Path(output_path)
        
        # Determine format from extension
        output_format = output_path.suffix.lower().replace(".", "")
        if output_format not in ["wav", "mp3"]:
            output_format = "wav"
            output_path = output_path.with_suffix(".wav")
        
        # Synthesize
        output = self.engine.synthesize(text, config=config, output_format=output_format)
        
        # Save to file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if isinstance(output.audio_array, bytes):
            # MP3 format
            with open(output_path, "wb") as f:
                f.write(output.audio_array)
        else:
            # WAV format
            if SOUNDFILE_AVAILABLE:
                sf.write(str(output_path), output.audio_array, output.sample_rate)
            else:
                tensor = torch.from_numpy(output.audio_array).unsqueeze(0)
                torchaudio.save(str(output_path), tensor, output.sample_rate)
        
        print(f"💾 Audio saved: {output_path} ({output.duration_ms:.0f}ms)")
        return str(output_path)
    
    def clone_voice(self, audio_path: str, voice_name: str) -> str:
        """
        Clone a voice from audio sample.
        
        Args:
            audio_path: Path to reference audio
            voice_name: Name for the cloned voice
        
        Returns:
            voice_id for use in synthesis
        """
        return self.engine.clone_voice(audio_path, voice_name)


# Factory functions for easy instantiation
def create_tts_engine(
    model: str = "xtts_v2",
    device: Optional[str] = None,
    enable_cache: bool = True
) -> TTSEngine:
    """
    Factory function to create a TTSEngine.
    
    Args:
        model: Model key from SUPPORTED_MODELS
        device: Device to run on
        enable_cache: Enable synthesis caching
    
    Returns:
        Configured TTSEngine instance
    """
    return TTSEngine(
        model_key=model,
        device=device,
        enable_cache=enable_cache
    )


def create_streamer(
    engine: Optional[TTSEngine] = None,
    enable_streaming: bool = True
) -> TTSStreamer:
    """
    Factory function to create a TTSStreamer.
    
    Args:
        engine: Optional pre-configured engine
        enable_streaming: Enable streaming mode
    
    Returns:
        Configured TTSStreamer instance
    """
    return TTSStreamer(
        engine=engine,
        enable_streaming=enable_streaming
    )


# Example usage
if __name__ == "__main__":
    # Example: Basic synthesis
    print("=" * 60)
    print("TTS Engine Demo")
    print("=" * 60)
    
    # Create engine (downloads model on first run)
    engine = create_tts_engine(model="xtts_v2")
    engine.load_model()
    
    # Synthesize text
    text = "Hello, this is a test of the Nexus TTS engine."
    output = engine.synthesize(text)
    
    print(f"\nSynthesis complete:")
    print(f"  Duration: {output.duration_ms:.0f}ms")
    print(f"  Sample rate: {output.sample_rate}Hz")
    print(f"  Format: {output.format}")
    
    # Print stats
    print(f"\nEngine stats:")
    for key, value in engine.get_stats().items():
        print(f"  {key}: {value}")
