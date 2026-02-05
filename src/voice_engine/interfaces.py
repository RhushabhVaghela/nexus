"""
Voice Engine Interfaces

Base classes and interfaces for voice pipeline components.
Provides abstractions for TTS engines, voice identity, and acoustic processing.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Generator, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import logging
import numpy as np


@dataclass
class AudioSegment:
    """Audio segment with metadata."""

    waveform: np.ndarray
    sample_rate: int
    duration_seconds: float
    text: str = ""
    voice_id: str = ""

    def __post_init__(self):
        if isinstance(self.waveform, list):
            self.waveform = np.array(self.waveform)


@dataclass
class BrainOutput:
    """Output from brain/reasoning engine."""

    text: str
    sentiment: Optional[str] = None
    intent: Optional[str] = None
    context: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.context is None:
            self.context = {}


class BaseReasoningEngine(ABC):
    """
    Base class for reasoning engines in voice pipeline.

    Processes text through LLM or other reasoning systems before TTS.
    """

    def __init__(self, model_name: str = "default", **kwargs):
        """
        Initialize reasoning engine.

        Args:
            model_name: Name of the reasoning model
            **kwargs: Additional engine-specific parameters
        """
        self.model_name = model_name
        self.logger = logging.getLogger(
            f"{self.__class__.__module__}.{self.__class__.__name__}"
        )
        self._configure(**kwargs)

    def _configure(self, **kwargs: Any) -> None:
        """
        Configure engine with parameters.

        Args:
            **kwargs: Configuration parameters
        """
        for key, value in kwargs.items():
            setattr(self, key, value)

    @abstractmethod
    def process(self, text: str, context: Optional[Dict] = None) -> BrainOutput:
        """
        Process text through reasoning engine.

        Args:
            text: Input text to process
            context: Optional context dictionary

        Returns:
            BrainOutput containing processed text and metadata

        Raises:
            ReasoningError: If processing fails
        """
        self.logger.debug(f"Processing text: {text[:100]}...")
        try:
            # Stub implementation - override in subclass
            return BrainOutput(
                text=text, sentiment="neutral", intent="process", context=context or {}
            )
        except Exception as e:
            self.logger.error(f"Reasoning processing failed: {e}")
            raise

    @abstractmethod
    def generate_response(self, query: str, context: Optional[Dict] = None) -> str:
        """
        Generate a response to a query.

        Args:
            query: User query
            context: Conversation context

        Returns:
            Generated response text

        Raises:
            ReasoningError: If generation fails
        """
        self.logger.debug(f"Generating response for query: {query[:100]}...")
        try:
            # Stub implementation - override in subclass
            # In real implementation, this would call LLM API
            return f"Response to: {query}"
        except Exception as e:
            self.logger.error(f"Response generation failed: {e}")
            raise


class BaseVoiceIdentity(ABC):
    """
    Base class for voice identity management.

    Manages voice characteristics and persona for consistent voice synthesis.
    """

    def __init__(self, default_voice: str = "NATM1", **kwargs):
        """
        Initialize voice identity manager.

        Args:
            default_voice: Default voice identifier
            **kwargs: Additional configuration parameters
        """
        self.default_voice = default_voice
        self.logger = logging.getLogger(
            f"{self.__class__.__module__}.{self.__class__.__name__}"
        )
        self._initialized = False
        self._configure(**kwargs)

    def _configure(self, **kwargs: Any) -> None:
        """
        Configure identity manager with parameters.

        Args:
            **kwargs: Configuration parameters
        """
        for key, value in kwargs.items():
            setattr(self, key, value)

    @abstractmethod
    def get_identity(self, voice_id: str) -> Dict[str, Any]:
        """
        Get voice identity characteristics.

        Args:
            voice_id: Voice identifier

        Returns:
            Dictionary of voice identity parameters including:
            - pitch: Fundamental frequency
            - speed: Speaking rate
            - timbre: Voice quality characteristics
            - emotion: Emotional parameters

        Raises:
            VoiceIdentityError: If voice_id not found
        """
        self.logger.debug(f"Getting identity for voice: {voice_id}")
        try:
            # Stub implementation - override in subclass
            return {
                "voice_id": voice_id,
                "pitch": 1.0,
                "speed": 1.0,
                "timbre": {},
                "emotion": {"neutral": 1.0},
            }
        except Exception as e:
            self.logger.error(f"Failed to get voice identity: {e}")
            raise

    @abstractmethod
    def apply_identity(self, audio: np.ndarray, voice_id: str) -> np.ndarray:
        """
        Apply voice identity characteristics to audio.

        Args:
            audio: Input audio waveform
            voice_id: Voice identifier

        Returns:
            Processed audio with voice identity applied

        Raises:
            VoiceIdentityError: If processing fails
        """
        self.logger.debug(f"Applying voice identity: {voice_id}")
        try:
            # Stub implementation - override in subclass
            # In real implementation, this would modify pitch, speed, etc.
            return audio.copy()
        except Exception as e:
            self.logger.error(f"Failed to apply voice identity: {e}")
            raise

    def clone_from_sample(self, audio: np.ndarray) -> Dict[str, Any]:
        """
        Clone voice identity from audio sample.

        Args:
            audio: Reference audio sample

        Returns:
            Voice profile dictionary

        Raises:
            VoiceIdentityError: If cloning fails
        """
        self.logger.debug("Cloning voice from sample")
        try:
            # Stub implementation - override in subclass
            return {
                "voice_id": "cloned",
                "pitch": 1.0,
                "speed": 1.0,
                "timbre": {},
                "emotion": {"neutral": 1.0},
                "source_sample_duration": len(audio) / 24000,
            }
        except Exception as e:
            self.logger.error(f"Voice cloning failed: {e}")
            raise


class BaseAcousticEngine(ABC):
    """
    Base class for acoustic processing engine.

    Handles text-to-speech synthesis with voice cloning capabilities.
    """

    def __init__(self, sample_rate: int = 24000, **kwargs):
        """
        Initialize acoustic engine.

        Args:
            sample_rate: Output audio sample rate in Hz
            **kwargs: Additional engine-specific parameters
        """
        self.sample_rate = sample_rate
        self.logger = logging.getLogger(
            f"{self.__class__.__module__}.{self.__class__.__name__}"
        )
        self._configure(**kwargs)

    def _configure(self, **kwargs: Any) -> None:
        """
        Configure engine with parameters.

        Args:
            **kwargs: Configuration parameters
        """
        for key, value in kwargs.items():
            setattr(self, key, value)

    @abstractmethod
    def synthesize(
        self, text: str, voice_params: Dict[str, Any], output_path: Optional[str] = None
    ) -> AudioSegment:
        """
        Synthesize text to speech with given voice parameters.

        Args:
            text: Text to synthesize
            voice_params: Voice parameters (voice_id, vibe, etc.)
            output_path: Optional path to save audio file

        Returns:
            AudioSegment with synthesized audio

        Raises:
            SynthesisError: If synthesis fails
        """
        self.logger.debug(f"Synthesizing text: {text[:100]}...")
        try:
            # Calculate duration based on text length (stub)
            word_count = len(text.split())
            duration = max(1.0, word_count * 0.3)  # Estimate 0.3s per word

            # Create placeholder silent audio
            samples = int(duration * self.sample_rate)
            waveform = np.zeros(samples, dtype=np.float32)

            audio = AudioSegment(
                waveform=waveform,
                sample_rate=self.sample_rate,
                duration_seconds=duration,
                text=text,
                voice_id=voice_params.get("voice_id", "default"),
            )

            # Save to file if path provided
            if output_path:
                self._save_audio(audio, output_path)

            return audio

        except Exception as e:
            self.logger.error(f"Speech synthesis failed: {e}")
            raise

    @abstractmethod
    def synthesize_streaming(
        self, text: str, voice_params: Dict[str, Any]
    ) -> Generator[np.ndarray, None, None]:
        """
        Synthesize text to speech in streaming mode.

        Args:
            text: Text to synthesize
            voice_params: Voice parameters

        Yields:
            Audio chunks as numpy arrays

        Raises:
            SynthesisError: If synthesis fails
        """
        self.logger.debug(f"Streaming synthesis for text: {text[:100]}...")
        try:
            # Stub implementation - override in subclass
            # In real implementation, this would yield chunks progressively
            word_count = len(text.split())
            chunk_size = max(1, word_count // 4)
            words = text.split()

            for i in range(0, len(words), chunk_size):
                chunk_words = words[i : i + chunk_size]
                chunk_text = " ".join(chunk_words)

                # Generate placeholder chunk
                chunk_samples = int(len(chunk_words) * 0.3 * self.sample_rate)
                chunk_audio = np.zeros(chunk_samples, dtype=np.float32)

                yield chunk_audio

        except Exception as e:
            self.logger.error(f"Streaming synthesis failed: {e}")
            raise

    def get_supported_voices(self) -> List[Dict[str, Any]]:
        """
        Get list of available voice profiles.

        Returns:
            List of voice profile dictionaries

        Raises:
            EngineError: If retrieval fails
        """
        self.logger.debug("Retrieving supported voices")
        try:
            # Stub implementation - override in subclass
            return [
                {
                    "voice_id": "NATM1",
                    "name": "Natural Male 1",
                    "gender": "male",
                    "description": "Standard natural male voice",
                },
                {
                    "voice_id": "NATF1",
                    "name": "Natural Female 1",
                    "gender": "female",
                    "description": "Standard natural female voice",
                },
            ]
        except Exception as e:
            self.logger.error(f"Failed to get supported voices: {e}")
            raise

    def _save_audio(self, audio: AudioSegment, output_path: str) -> None:
        """
        Save audio segment to file.

        Args:
            audio: AudioSegment to save
            output_path: Path to save audio file

        Raises:
            IOError: If saving fails
        """
        try:
            import soundfile as sf

            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(path), audio.waveform, audio.sample_rate)
            self.logger.info(f"Saved audio to: {output_path}")
        except ImportError:
            self.logger.warning("soundfile not available, skipping audio save")
        except Exception as e:
            self.logger.error(f"Failed to save audio: {e}")
            raise


class UniversalVoicePipeline:
    """
    Universal voice processing pipeline.

    Orchestrates voice synthesis through reasoning, identity, and acoustic engines.
    Provides a unified interface for voice synthesis with persona and vibe control.
    """

    def __init__(
        self,
        reasoning_engine: Optional[BaseReasoningEngine] = None,
        voice_identity: Optional[BaseVoiceIdentity] = None,
        acoustic_engine: Optional[BaseAcousticEngine] = None,
    ):
        """
        Initialize universal voice pipeline.

        Args:
            reasoning_engine: Engine for text reasoning/processing
            voice_identity: Engine for voice identity management
            acoustic_engine: Engine for speech synthesis
        """
        self.reasoning_engine = reasoning_engine
        self.voice_identity = voice_identity
        self.acoustic_engine = acoustic_engine

        # Default voice settings
        self.default_voice_id = "NATM1"
        self.default_vibe = "neutral"

        # State
        self.active_voice = self.default_voice_id
        self.active_vibe = self.default_vibe
        self.conversation_history: list = []

    def switch_voice(self, voice_id: str) -> None:
        """
        Switch to a different voice persona.

        Args:
            voice_id: New voice identifier
        """
        self.active_voice = voice_id

    def set_vibe(self, vibe: str) -> None:
        """
        Set the emotional vibe for voice synthesis.

        Args:
            vibe: Vibe name (e.g., 'excited', 'calm', 'professional')
        """
        self.active_vibe = vibe

    def process_turn(
        self,
        user_input: str,
        voice_id: Optional[str] = None,
        vibe: Optional[str] = None,
    ) -> Tuple[AudioSegment, BrainOutput]:
        """
        Process a complete conversation turn.

        Args:
            user_input: User's text input
            voice_id: Optional voice override
            vibe: Optional vibe override

        Returns:
            Tuple of (audio_segment, brain_output)
        """
        # Use active settings or overrides
        voice = voice_id or self.active_voice
        emotion = vibe or self.active_vibe

        # Step 1: Reasoning (if available)
        if self.reasoning_engine:
            brain_output = self.reasoning_engine.process(
                user_input, context={"history": self.conversation_history}
            )
            response_text = brain_output.text
        else:
            # Simple echo if no reasoning engine
            response_text = f"You said: {user_input}"
            brain_output = BrainOutput(text=response_text)

        # Step 2: Voice Synthesis (if acoustic engine available)
        if self.acoustic_engine:
            voice_params = {
                "voice_id": voice,
                "vibe": emotion,
            }
            audio = self.acoustic_engine.synthesize(response_text, voice_params)
        else:
            # Create silent audio if no acoustic engine
            audio = AudioSegment(
                waveform=np.zeros(24000, dtype=np.float32),
                sample_rate=24000,
                duration_seconds=1.0,
                text=response_text,
                voice_id=voice,
            )

        # Update history
        self.conversation_history.append(
            {
                "user": user_input,
                "assistant": response_text,
                "voice": voice,
                "vibe": emotion,
            }
        )

        return audio, brain_output

    def synthesize_text(
        self,
        text: str,
        voice_id: Optional[str] = None,
        vibe: Optional[str] = None,
        output_path: Optional[str] = None,
    ) -> AudioSegment:
        """
        Synthesize text directly to speech.

        Args:
            text: Text to synthesize
            voice_id: Optional voice override
            vibe: Optional vibe override
            output_path: Optional path to save audio

        Returns:
            AudioSegment with synthesized audio
        """
        voice = voice_id or self.active_voice
        emotion = vibe or self.active_vibe

        if self.acoustic_engine:
            voice_params = {
                "voice_id": voice,
                "vibe": emotion,
            }
            return self.acoustic_engine.synthesize(text, voice_params, output_path)
        else:
            # Return silent audio placeholder
            return AudioSegment(
                waveform=np.zeros(24000, dtype=np.float32),
                sample_rate=24000,
                duration_seconds=1.0,
                text=text,
                voice_id=voice,
            )

    def clear_history(self) -> None:
        """Clear conversation history."""
        self.conversation_history = []

    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        return {
            "active_voice": self.active_voice,
            "active_vibe": self.active_vibe,
            "conversation_turns": len(self.conversation_history),
            "has_reasoning": self.reasoning_engine is not None,
            "has_acoustic": self.acoustic_engine is not None,
            "has_identity": self.voice_identity is not None,
        }
