"""
Voice Engine Module for Nexus

Provides voice cloning, vibe modulation, and voice registry capabilities.
Integrates with TTS systems for high-quality voice synthesis.
"""

from .cloner import VoiceCloner, VoiceEncoder, VoiceDNA, voice_cloner
from .vibe_modulator import VibeModulator, VibeParams, EmotionID, vibe_modulator
from .registry import VoiceRegistry, VoiceInfo, VoiceType, voice_registry
from .interfaces import (
    UniversalVoicePipeline,
    BaseReasoningEngine,
    BaseVoiceIdentity,
    BaseAcousticEngine,
    AudioSegment,
    BrainOutput,
)

__all__ = [
    # Cloner
    "VoiceCloner",
    "VoiceEncoder",
    "VoiceDNA",
    "voice_cloner",
    # Vibe Modulator
    "VibeModulator",
    "VibeParams",
    "EmotionID",
    "vibe_modulator",
    # Registry
    "VoiceRegistry",
    "VoiceInfo",
    "VoiceType",
    "voice_registry",
    # Interfaces
    "UniversalVoicePipeline",
    "BaseReasoningEngine",
    "BaseVoiceIdentity",
    "BaseAcousticEngine",
    "AudioSegment",
    "BrainOutput",
]

__version__ = "1.0.0"
