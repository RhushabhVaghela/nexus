
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import torch

class BaseReasoningEngine(ABC):
    """Interface for the 'Brain' of the system."""
    @abstractmethod
    def generate_response(self, context: str, **kwargs) -> Dict[str, Any]:
        """Returns text plus rich metadata (intent, emotion, etc.)"""
        pass

class BaseVoiceIdentity(ABC):
    """Interface for Persona management and Cloning."""
    @abstractmethod
    def get_embedding(self, voice_id: str) -> torch.Tensor:
        pass
    
    @abstractmethod
    def clone_from_audio(self, audio_path: str) -> str:
        pass

class BaseAcousticEngine(ABC):
    """Interface for Vibe/Speech Synthesis."""
    @abstractmethod
    def synthesize(self, text: str, dna: torch.Tensor, vibe_params: Dict[str, Any]) -> torch.Tensor:
        pass

class UniversalVoicePipeline:
    """
    The Orchestrator that connects a Brain to a Voice.
    Designed for 100% capability inheritance.
    """
    def __init__(self, brain: BaseReasoningEngine, identity: BaseVoiceIdentity, acoustic: BaseAcousticEngine):
        self.brain = brain
        self.identity = identity
        self.acoustic = acoustic

    def process_turn(self, user_input: str, voice_id: str):
        # 1. Get High-Fidelity Reasoning + Metadata
        brain_output = self.brain.generate_response(user_input)
        
        # 2. Get Voice DNA
        dna = self.identity.get_embedding(voice_id)
        
        # 3. Synthesize with full intelligence transfer
        audio = self.acoustic.synthesize(
            text=brain_output["text"],
            dna=dna,
            vibe_params=brain_output.get("metadata", {})
        )
        return audio, brain_output
