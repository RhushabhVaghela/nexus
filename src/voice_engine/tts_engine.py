"""
src/voice_engine/tts_engine.py

Text-to-Speech Engine with multiple backend support:
- Coqui TTS (primary)
- gTTS (Google Text-to-Speech fallback)
- OpenAI TTS (optional)

Features:
- Voice cloning integration via VoiceCloner
- Vibe modulation via VibeModulator
- Multi-language support
- Streaming audio generation
- Caching for repeated phrases
"""

import os
import re
import hashlib
import logging
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any, List, Union, Callable
from dataclasses import dataclass
from enum import Enum
import threading

import torch
import torchaudio
import numpy as np

# Configure logging
logger = logging.getLogger(__name__)


class TTSBackend(Enum):
    """Available TTS backends."""
    COQUI = "coqui"           # Coqui TTS (local, high quality)
    GTTS = "gtts"             # Google TTS (cloud, simple)
    OPENAI = "openai"         # OpenAI TTS (cloud, high quality)
    PIPELINE = "pipeline"     # Combined pipeline with vibe modulation


@dataclass
class TTSConfig:
    """Configuration for TTS generation."""
    backend: TTSBackend = TTSBackend.COQUI
    model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2"
    vocoder_name: Optional[str] = None
    sample_rate: int = 22050
    language: str = "en"
    speed: float = 1.0
    pitch: float = 1.0
    volume: float = 1.0
    use_gpu: bool = True
    cache_enabled: bool = True
    max_cache_size_mb: int = 500


@dataclass
class TTSResult:
    """Result from TTS generation."""
    audio_tensor: torch.Tensor
    sample_rate: int
    text: str
    backend_used: str
    generation_time_ms: float
    cached: bool = False
    metadata: Dict[str, Any] = None


class TTSEngine:
    """
    Text-to-Speech Engine with multiple backend support and voice cloning.
    
    Supports:
    - Coqui TTS for high-quality local synthesis
    - gTTS for quick cloud-based synthesis
    - OpenAI TTS for premium quality
    - Custom voice cloning via VoiceCloner
    - Vibe modulation for expressive speech
    """
    
    def __init__(self, config: Optional[TTSConfig] = None):
        """
        Initialize the TTS Engine.
        
        Args:
            config: TTS configuration. Uses defaults if not provided.
        """
        self.config = config or TTSConfig()
        self.device = "cuda" if (torch.cuda.is_available() and self.config.use_gpu) else "cpu"
        
        # Backend instances (lazy loaded)
        self._coqui_model = None
        self._coqui_vocoder = None
        self._gtts_instance = None
        
        # Voice cloning and modulation
        self._voice_cloner = None
        self._vibe_modulator = None
        
        # Cache for repeated phrases
        self._cache: Dict[str, TTSResult] = {}
        self._cache_lock = threading.RLock()
        self._cache_size_bytes = 0
        
        # Stats
        self._stats = {
            "requests": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "errors": 0,
            "total_generation_time_ms": 0
        }
        
        logger.info(f"TTSEngine initialized with device: {self.device}")
    
    def _get_cache_key(self, text: str, voice_id: Optional[str] = None, 
                       vibe: Optional[str] = None) -> str:
        """Generate a cache key for the TTS request."""
        key_data = f"{text}:{voice_id}:{vibe}:{self.config.language}:{self.config.speed}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _check_cache(self, cache_key: str) -> Optional[TTSResult]:
        """Check if result is in cache."""
        if not self.config.cache_enabled:
            return None
        
        with self._cache_lock:
            if cache_key in self._cache:
                logger.debug(f"Cache hit for key: {cache_key[:8]}...")
                self._stats["cache_hits"] += 1
                return self._cache[cache_key]
        
        self._stats["cache_misses"] += 1
        return None
    
    def _add_to_cache(self, cache_key: str, result: TTSResult):
        """Add result to cache with size management."""
        if not self.config.cache_enabled:
            return
        
        with self._cache_lock:
            # Estimate size
            audio_bytes = result.audio_tensor.element_size() * result.audio_tensor.nelement()
            
            # Evict old entries if needed
            max_bytes = self.config.max_cache_size_mb * 1024 * 1024
            while self._cache_size_bytes + audio_bytes > max_bytes and self._cache:
                oldest_key = next(iter(self._cache))
                oldest_result = self._cache.pop(oldest_key)
                oldest_bytes = (oldest_result.audio_tensor.element_size() * 
                               oldest_result.audio_tensor.nelement())
                self._cache_size_bytes -= oldest_bytes
            
            self._cache[cache_key] = result
            self._cache_size_bytes += audio_bytes
    
    def _load_coqui(self):
        """Lazy load Coqui TTS model."""
        if self._coqui_model is not None:
            return
        
        try:
            from TTS.api import TTS
            
            logger.info(f"Loading Coqui TTS model: {self.config.model_name}")
            
            # Initialize TTS model
            self._coqui_model = TTS(self.config.model_name).to(self.device)
            
            logger.info("Coqui TTS model loaded successfully")
            
        except ImportError:
            logger.error("Coqui TTS not installed. Install with: pip install TTS")
            raise
        except Exception as e:
            logger.error(f"Failed to load Coqui TTS: {e}")
            raise
    
    def _load_gtts(self):
        """Lazy load gTTS."""
        if self._gtts_instance is not None:
            return
        
        try:
            from gtts import gTTS
            self._gtts_class = gTTS
            logger.info("gTTS initialized")
        except ImportError:
            logger.error("gTTS not installed. Install with: pip install gtts")
            raise
    
    def _synthesize_coqui(self, text: str, speaker_wav: Optional[str] = None,
                          language: Optional[str] = None) -> torch.Tensor:
        """Synthesize using Coqui TTS."""
        self._load_coqui()
        
        lang = language or self.config.language
        
        try:
            # Create temporary output file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                output_path = tmp.name
            
            # Generate speech
            if speaker_wav and hasattr(self._coqui_model, 'tts_to_file'):
                # Voice cloning mode
                self._coqui_model.tts_to_file(
                    text=text,
                    speaker_wav=speaker_wav,
                    language=lang,
                    file_path=output_path
                )
            else:
                # Standard TTS
                wav = self._coqui_model.tts(text=text)
                # Save to file for consistent loading
                import scipy.io.wavfile as wavfile
                wavfile.write(output_path, self.config.sample_rate, wav)
            
            # Load the generated audio
            waveform, sample_rate = torchaudio.load(output_path)
            
            # Cleanup
            os.unlink(output_path)
            
            # Resample if needed
            if sample_rate != self.config.sample_rate:
                resampler = torchaudio.transforms.Resample(sample_rate, self.config.sample_rate)
                waveform = resampler(waveform)
            
            return waveform
            
        except Exception as e:
            logger.error(f"Coqui TTS synthesis failed: {e}")
            raise
    
    def _synthesize_gtts(self, text: str) -> torch.Tensor:
        """Synthesize using gTTS."""
        self._load_gtts()
        
        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
                output_path = tmp.name
            
            # Generate speech
            tts = self._gtts_class(text=text, lang=self.config.language, slow=False)
            tts.save(output_path)
            
            # Load and convert to tensor
            waveform, sample_rate = torchaudio.load(output_path)
            
            # Cleanup
            os.unlink(output_path)
            
            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Resample if needed
            if sample_rate != self.config.sample_rate:
                resampler = torchaudio.transforms.Resample(sample_rate, self.config.sample_rate)
                waveform = resampler(waveform)
            
            return waveform
            
        except Exception as e:
            logger.error(f"gTTS synthesis failed: {e}")
            raise
    
    def _apply_audio_effects(self, audio: torch.Tensor, 
                             speed: Optional[float] = None,
                             pitch: Optional[float] = None,
                             volume: Optional[float] = None) -> torch.Tensor:
        """
        Apply audio effects (speed, pitch, volume).
        
        Args:
            audio: Input audio tensor of shape (channels, samples)
            speed: Speed multiplier (1.0 = normal)
            pitch: Pitch multiplier (1.0 = normal)
            volume: Volume multiplier (1.0 = normal)
            
        Returns:
            Processed audio tensor
        """
        result = audio.clone()
        
        try:
            # Apply volume
            vol = volume if volume is not None else self.config.volume
            if abs(vol - 1.0) > 0.01:
                result = result * vol
                result = torch.clamp(result, -1.0, 1.0)  # Prevent clipping
            
            # Apply speed (time stretching)
            spd = speed if speed is not None else self.config.speed
            if abs(spd - 1.0) > 0.01:
                # Use interpolate for time stretching
                orig_len = result.shape[-1]
                new_len = int(orig_len / spd)
                result = torch.nn.functional.interpolate(
                    result.unsqueeze(1),
                    size=new_len,
                    mode='linear',
                    align_corners=False
                ).squeeze(1)
                
                # Pad or trim to maintain approximate length
                if new_len < orig_len:
                    padding = torch.zeros(result.shape[0], orig_len - new_len, 
                                         device=result.device, dtype=result.dtype)
                    result = torch.cat([result, padding], dim=1)
                elif new_len > orig_len:
                    result = result[:, :orig_len]
            
            # Apply pitch shifting
            pch = pitch if pitch is not None else self.config.pitch
            if abs(pch - 1.0) > 0.01:
                # Simple pitch shifting via resampling
                orig_len = result.shape[-1]
                new_len = int(orig_len / pch)
                result = torch.nn.functional.interpolate(
                    result.unsqueeze(1),
                    size=new_len,
                    mode='linear',
                    align_corners=False
                ).squeeze(1)
                # Resample back to original length
                result = torch.nn.functional.interpolate(
                    result.unsqueeze(1),
                    size=orig_len,
                    mode='linear',
                    align_corners=False
                ).squeeze(1)
            
            return result
            
        except Exception as e:
            logger.warning(f"Audio effects application failed: {e}. Returning original audio.")
            return audio
    
    def synthesize(self, text: str, 
                   voice_id: Optional[str] = None,
                   vibe: Optional[str] = None,
                   speaker_wav: Optional[str] = None,
                   backend: Optional[TTSBackend] = None,
                   **kwargs) -> TTSResult:
        """
        Synthesize text to speech.
        
        Args:
            text: Text to synthesize
            voice_id: Voice identifier for cloned voices
            vibe: Vibe/emotion to apply (neutral, excited, thoughtful, etc.)
            speaker_wav: Path to speaker reference audio for voice cloning
            backend: TTS backend to use (defaults to config)
            **kwargs: Additional parameters (speed, pitch, volume, language)
            
        Returns:
            TTSResult containing audio tensor and metadata
        """
        import time
        start_time = time.time()
        
        self._stats["requests"] += 1
        
        # Clean text
        text = text.strip()
        if not text:
            raise ValueError("Empty text provided for synthesis")
        
        # Check cache
        use_cache = kwargs.get('use_cache', True)
        cache_key = self._get_cache_key(text, voice_id, vibe)
        
        if use_cache:
            cached_result = self._check_cache(cache_key)
            if cached_result is not None:
                cached_result.cached = True
                return cached_result
        
        # Select backend
        selected_backend = backend or self.config.backend
        
        try:
            # Synthesize based on backend
            if selected_backend == TTSBackend.COQUI:
                audio = self._synthesize_coqui(text, speaker_wav=speaker_wav,
                                               language=kwargs.get('language'))
                backend_name = "coqui"
                
            elif selected_backend == TTSBackend.GTTS:
                audio = self._synthesize_gtts(text)
                backend_name = "gtts"
                
            elif selected_backend == TTSBackend.PIPELINE:
                # Use Coqui with voice cloning and vibe modulation
                audio = self._synthesize_coqui(text, speaker_wav=speaker_wav)
                backend_name = "pipeline"
                
                # Apply vibe modulation if specified
                if vibe and vibe != "neutral":
                    from .vibe_modulator import VibeModulator
                    modulator = VibeModulator()
                    audio = modulator.apply_vibe(audio, vibe)
            
            else:
                raise ValueError(f"Unknown backend: {selected_backend}")
            
            # Apply audio effects
            audio = self._apply_audio_effects(
                audio,
                speed=kwargs.get('speed'),
                pitch=kwargs.get('pitch'),
                volume=kwargs.get('volume')
            )
            
            generation_time = (time.time() - start_time) * 1000
            self._stats["total_generation_time_ms"] += generation_time
            
            result = TTSResult(
                audio_tensor=audio,
                sample_rate=self.config.sample_rate,
                text=text,
                backend_used=backend_name,
                generation_time_ms=generation_time,
                cached=False,
                metadata={
                    "voice_id": voice_id,
                    "vibe": vibe,
                    "language": kwargs.get('language', self.config.language)
                }
            )
            
            # Add to cache
            if use_cache:
                self._add_to_cache(cache_key, result)
            
            logger.info(f"Synthesized '{text[:50]}...' in {generation_time:.1f}ms using {backend_name}")
            
            return result
            
        except Exception as e:
            self._stats["errors"] += 1
            logger.error(f"TTS synthesis failed: {e}")
            raise
    
    def synthesize_streaming(self, text: str, 
                            chunk_size_ms: int = 100,
                            **kwargs) -> Iterator[torch.Tensor]:
        """
        Stream TTS generation in chunks.
        
        Args:
            text: Text to synthesize
            chunk_size_ms: Size of each chunk in milliseconds
            **kwargs: Additional synthesis parameters
            
        Yields:
            Audio tensor chunks
        """
        # For streaming, we need to split text into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        for sentence in sentences:
            if not sentence.strip():
                continue
            
            result = self.synthesize(sentence, **kwargs)
            
            # Calculate chunk size in samples
            chunk_samples = int(chunk_size_ms * self.config.sample_rate / 1000)
            
            # Split into chunks
            audio = result.audio_tensor
            for i in range(0, audio.shape[-1], chunk_samples):
                chunk = audio[:, i:i+chunk_samples]
                if chunk.shape[-1] > 0:
                    yield chunk
    
    def clone_voice(self, audio_path: str, voice_name: str, 
                    description: str = "") -> str:
        """
        Clone a voice from audio sample.
        
        Args:
            audio_path: Path to reference audio file
            voice_name: Name for the cloned voice
            description: Optional description
            
        Returns:
            Path to the cloned voice DNA file
        """
        from .cloner import VoiceCloner
        
        cloner = VoiceCloner()
        return cloner.clone_voice(audio_path, voice_name, description)
    
    def save_audio(self, result: TTSResult, output_path: str, 
                   format: str = "wav") -> str:
        """
        Save TTS result to file.
        
        Args:
            result: TTSResult to save
            output_path: Output file path
            format: Audio format (wav, mp3, flac)
            
        Returns:
            Path to saved file
        """
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Ensure correct extension
            if not output_path.suffix:
                output_path = output_path.with_suffix(f".{format}")
            
            # Save audio
            torchaudio.save(
                str(output_path),
                result.audio_tensor,
                result.sample_rate,
                format=format
            )
            
            logger.info(f"Audio saved to: {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Failed to save audio: {e}")
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        """Get TTS engine statistics."""
        return {
            **self._stats,
            "cache_size_entries": len(self._cache),
            "cache_size_mb": self._cache_size_bytes / (1024 * 1024),
            "avg_generation_time_ms": (
                self._stats["total_generation_time_ms"] / max(1, self._stats["requests"])
            )
        }
    
    def clear_cache(self):
        """Clear the TTS cache."""
        with self._cache_lock:
            self._cache.clear()
            self._cache_size_bytes = 0
        logger.info("TTS cache cleared")
    
    def list_voices(self) -> Dict[str, Dict[str, str]]:
        """List available voices (presets and cloned)."""
        from .registry import voice_registry
        return voice_registry.list_voices()


# Global instance for convenience
tts_engine = TTSEngine()


def synthesize_speech(text: str, **kwargs) -> TTSResult:
    """Convenience function for speech synthesis."""
    return tts_engine.synthesize(text, **kwargs)


def clone_voice(audio_path: str, voice_name: str, description: str = "") -> str:
    """Convenience function for voice cloning."""
    return tts_engine.clone_voice(audio_path, voice_name, description)


__all__ = [
    'TTSEngine',
    'TTSConfig',
    'TTSResult',
    'TTSBackend',
    'synthesize_speech',
    'clone_voice',
    'tts_engine'
]