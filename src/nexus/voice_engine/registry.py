"""
Voice Registry Module

Manages voice presets and custom voice registrations.
Integrates with PersonaPlex and other TTS systems.
"""

import json
from typing import Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum


class VoiceType(Enum):
    """Voice classification types."""
    PRESET = "preset"
    CLONED = "cloned"
    CUSTOM = "custom"
    SYSTEM = "system"


@dataclass
class VoiceInfo:
    """Information about a voice."""
    name: str
    description: str
    type: VoiceType
    dna_path: Optional[str] = None
    language: str = "en"
    gender: Optional[str] = None
    age_range: Optional[str] = None
    tags: list = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "type": self.type.value,
            "dna_path": self.dna_path,
            "language": self.language,
            "gender": self.gender,
            "age_range": self.age_range,
            "tags": self.tags,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VoiceInfo":
        """Create from dictionary."""
        data = data.copy()
        data["type"] = VoiceType(data.get("type", "custom"))
        return cls(**data)


class VoiceRegistry:
    """
    Registry for managing voice presets and custom voices.
    
    Provides a centralized registry for voice lookup and management.
    Integrates with PersonaPlex voice system and custom cloned voices.
    """
    
    # PersonaPlex preset voices
    PRESET_VOICES = {
        # Female voices
        "NATF0": VoiceInfo(
            name="NATF0",
            description="Natural Female - Baseline",
            type=VoiceType.PRESET,
            dna_path="builtin://NATF0",
            language="en",
            gender="female",
            age_range="25-35",
            tags=["natural", "neutral", "professional"]
        ),
        "NATF1": VoiceInfo(
            name="NATF1",
            description="Natural Female - Warm",
            type=VoiceType.PRESET,
            dna_path="builtin://NATF1",
            language="en",
            gender="female",
            age_range="30-40",
            tags=["warm", "friendly", "conversational"]
        ),
        "NATF2": VoiceInfo(
            name="NATF2",
            description="Natural Female - Energetic",
            type=VoiceType.PRESET,
            dna_path="builtin://NATF2",
            language="en",
            gender="female",
            age_range="20-30",
            tags=["energetic", "young", "dynamic"]
        ),
        
        # Male voices
        "NATM0": VoiceInfo(
            name="NATM0",
            description="Natural Male - Baseline",
            type=VoiceType.PRESET,
            dna_path="builtin://NATM0",
            language="en",
            gender="male",
            age_range="25-35",
            tags=["natural", "neutral", "professional"]
        ),
        "NATM1": VoiceInfo(
            name="NATM1",
            description="Natural Male - Authoritative",
            type=VoiceType.PRESET,
            dna_path="builtin://NATM1",
            language="en",
            gender="male",
            age_range="35-45",
            tags=["authoritative", "deep", "professional"]
        ),
        "NATM2": VoiceInfo(
            name="NATM2",
            description="Natural Male - Friendly",
            type=VoiceType.PRESET,
            dna_path="builtin://NATM2",
            language="en",
            gender="male",
            age_range="25-35",
            tags=["friendly", "approachable", "conversational"]
        ),
        
        # System voices
        "system_default": VoiceInfo(
            name="system_default",
            description="System Default Voice",
            type=VoiceType.SYSTEM,
            dna_path="builtin://system",
            language="en",
            tags=["system", "default"]
        ),
        "system_narrator": VoiceInfo(
            name="system_narrator",
            description="System Narrator Voice",
            type=VoiceType.SYSTEM,
            dna_path="builtin://narrator",
            language="en",
            tags=["narrator", "storytelling", "documentary"]
        ),
    }
    
    def __init__(self, storage_path: Optional[str] = None):
        """
        Initialize voice registry.
        
        Args:
            storage_path: Path to store custom voice registrations
        """
        self.storage_path = storage_path
        self.custom_voices: Dict[str, VoiceInfo] = {}
        
        # Load custom voices from storage
        if storage_path and Path(storage_path).exists():
            self._load_custom_voices()
    
    def _load_custom_voices(self):
        """Load custom voices from storage."""
        try:
            with open(self.storage_path, 'r') as f:
                data = json.load(f)
            
            for voice_id, voice_data in data.items():
                self.custom_voices[voice_id] = VoiceInfo.from_dict(voice_data)
        except Exception as e:
            print(f"Failed to load custom voices: {e}")
    
    def _save_custom_voices(self):
        """Save custom voices to storage."""
        if self.storage_path:
            try:
                data = {
                    voice_id: voice_info.to_dict()
                    for voice_id, voice_info in self.custom_voices.items()
                }
                
                Path(self.storage_path).parent.mkdir(parents=True, exist_ok=True)
                with open(self.storage_path, 'w') as f:
                    json.dump(data, f, indent=4)
            except Exception as e:
                print(f"Failed to save custom voices: {e}")
    
    def list_voices(self) -> Dict[str, Dict[str, Any]]:
        """
        List all available voices.
        
        Returns:
            Dictionary mapping voice IDs to voice information
        """
        voices = {}
        
        # Add preset voices
        for voice_id, voice_info in self.PRESET_VOICES.items():
            voices[voice_id] = voice_info.to_dict()
        
        # Add custom voices
        for voice_id, voice_info in self.custom_voices.items():
            voices[voice_id] = voice_info.to_dict()
        
        return voices
    
    def get_voice_dna(self, voice_id: str) -> Optional[str]:
        """
        Get voice DNA path by ID.
        
        Args:
            voice_id: Voice identifier
            
        Returns:
            Path to voice DNA or None if not found
        """
        if voice_id in self.PRESET_VOICES:
            return self.PRESET_VOICES[voice_id].dna_path
        
        if voice_id in self.custom_voices:
            return self.custom_voices[voice_id].dna_path
        
        return None
    
    def get_voice_info(self, voice_id: str) -> Optional[VoiceInfo]:
        """
        Get voice information by ID.
        
        Args:
            voice_id: Voice identifier
            
        Returns:
            VoiceInfo object or None if not found
        """
        if voice_id in self.PRESET_VOICES:
            return self.PRESET_VOICES[voice_id]
        
        if voice_id in self.custom_voices:
            return self.custom_voices[voice_id]
        
        return None
    
    def register_voice(
        self,
        voice_id: str,
        dna_path: str,
        description: str = "",
        language: str = "en",
        **metadata
    ) -> None:
        """
        Register a custom voice.
        
        Args:
            voice_id: Unique voice identifier
            dna_path: Path to voice DNA file
            description: Voice description
            language: Voice language code
            **metadata: Additional voice metadata
        """
        voice_info = VoiceInfo(
            name=voice_id,
            description=description,
            type=VoiceType.CLONED,
            dna_path=dna_path,
            language=language,
            metadata=metadata
        )
        
        self.custom_voices[voice_id] = voice_info
        self._save_custom_voices()
    
    def unregister_voice(self, voice_id: str) -> bool:
        """
        Unregister a custom voice.
        
        Args:
            voice_id: Voice identifier
            
        Returns:
            True if voice was removed, False otherwise
        """
        if voice_id in self.custom_voices:
            del self.custom_voices[voice_id]
            self._save_custom_voices()
            return True
        return False
    
    def find_voices_by_tag(self, tag: str) -> Dict[str, VoiceInfo]:
        """
        Find voices by tag.
        
        Args:
            tag: Tag to search for
            
        Returns:
            Dictionary of matching voices
        """
        results = {}
        
        for voice_id, voice_info in {**self.PRESET_VOICES, **self.custom_voices}.items():
            if tag in voice_info.tags:
                results[voice_id] = voice_info
        
        return results
    
    def find_voices_by_language(self, language: str) -> Dict[str, VoiceInfo]:
        """
        Find voices by language.
        
        Args:
            language: Language code (e.g., 'en', 'es')
            
        Returns:
            Dictionary of matching voices
        """
        results = {}
        
        for voice_id, voice_info in {**self.PRESET_VOICES, **self.custom_voices}.items():
            if voice_info.language == language:
                results[voice_id] = voice_info
        
        return results
    
    def is_preset(self, voice_id: str) -> bool:
        """Check if voice is a preset."""
        return voice_id in self.PRESET_VOICES
    
    def is_custom(self, voice_id: str) -> bool:
        """Check if voice is custom/cloned."""
        return voice_id in self.custom_voices


# Global instance for convenience
voice_registry = VoiceRegistry()
