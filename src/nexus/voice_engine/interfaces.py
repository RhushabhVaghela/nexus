"""
Voice Engine Interfaces

Base classes and interfaces for voice pipeline components.
Provides abstractions for TTS engines, voice identity, and acoustic processing.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
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
    context: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.context is None:
            self.context = {}


class BaseReasoningEngine(ABC):
    """
    Base class for reasoning engines in voice pipeline.
    
    Processes text through LLM or other reasoning systems before TTS.
    """
    
    @abstractmethod
    def process(self, text: str, context: Optional[Dict] = None) -> BrainOutput:
        """
        Process text through reasoning engine.
        
        Args:
            text: Input text to process
            context: Optional context dictionary
            
        Returns:
            BrainOutput containing processed text and metadata
        """
        pass
    
    @abstractmethod
    def generate_response(self, query: str, context: Optional[Dict] = None) -> str:
        """
        Generate a response to a query.
        
        Args:
            query: User query
            context: Conversation context
            
        Returns:
            Generated response text
        """
        pass


class BaseVoiceIdentity(ABC):
    """
    Base class for voice identity management.
    
    Manages voice characteristics and persona for consistent voice synthesis.
    """
    
    @abstractmethod
    def get_identity(self, voice_id: str) -> Dict[str, Any]:
        """
        Get voice identity characteristics.
        
        Args:
            voice_id: Voice identifier
            
        Returns:
            Dictionary of voice identity parameters
        """
        pass
    
    @abstractmethod
    def apply_identity(self, audio: np.ndarray, voice_id: str) -> np.ndarray:
        """
        Apply voice identity characteristics to audio.
        
        Args:
            audio: Input audio waveform
            voice_id: Voice identifier
            
        Returns:
            Processed audio with voice identity applied
        """
        pass


class BaseAcousticEngine(ABC):
    """
    Base class for acoustic processing engine.
    
    Handles text-to-speech synthesis with voice cloning capabilities.
    """
    
    @abstractmethod
    def synthesize(
        self,
        text: str,
        voice_params: Dict[str, Any],
        output_path: Optional[str] = None
    ) -> AudioSegment:
        """
        Synthesize text to speech with given voice parameters.
        
        Args:
            text: Text to synthesize
            voice_params: Voice parameters (voice_id, vibe, etc.)
            output_path: Optional path to save audio file
            
        Returns:
            AudioSegment with synthesized audio
        """
        pass
    
    @abstractmethod
    def synthesize_streaming(
        self,
        text: str,
        voice_params: Dict[str, Any]
    ) -> np.ndarray:
        """
        Synthesize text to speech in streaming mode.
        
        Args:
            text: Text to synthesize
            voice_params: Voice parameters
            
        Returns:
            Audio waveform as numpy array
        """
        pass


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
        vibe: Optional[str] = None
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
                user_input,
                context={"history": self.conversation_history}
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
                voice_id=voice
            )
        
        # Update history
        self.conversation_history.append({
            "user": user_input,
            "assistant": response_text,
            "voice": voice,
            "vibe": emotion,
        })
        
        return audio, brain_output
    
    def synthesize_text(
        self,
        text: str,
        voice_id: Optional[str] = None,
        vibe: Optional[str] = None,
        output_path: Optional[str] = None
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
                voice_id=voice
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
