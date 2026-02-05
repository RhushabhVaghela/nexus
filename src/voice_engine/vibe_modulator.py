"""
Vibe Modulator Module

Provides vibe/energy modulation for voice synthesis.
Maps emotional states to acoustic parameters.
"""

import json
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
from enum import IntEnum


class EmotionID(IntEnum):
    """Emotion identifiers for vibe modulation."""
    NEUTRAL = 0
    CALM = 1
    HAPPY = 2
    SAD = 3
    ANGRY = 4
    FEARFUL = 5
    DISGUST = 6
    SURPRISED = 7
    EXCITED = 8
    PROFESSIONAL = 9
    CASUAL = 10
    ENERGETIC = 11
    WARM = 12
    SERIOUS = 13


@dataclass
class VibeParams:
    """
    Acoustic parameters for voice modulation.
    
    All parameters are multipliers (1.0 = baseline):
    - pitch: Pitch modulation (0.5 = lower, 2.0 = higher)
    - energy: Energy/volume (0.5 = quieter, 2.0 = louder)
    - speed: Speaking rate (0.5 = slower, 2.0 = faster)
    - emotion_id: Categorical emotion identifier
    """
    pitch: float = 1.0
    energy: float = 1.0
    speed: float = 1.0
    emotion_id: int = 0
    
    # Extended parameters for fine control
    warmth: float = 1.0
    brightness: float = 1.0
    clarity: float = 1.0
    expressiveness: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VibeParams":
        """Create from dictionary."""
        return cls(**data)


class VibeModulator:
    """
    Modulates voice vibes (energy, emotion, tone) for voice synthesis.
    
    Maps emotion names to acoustic parameters for real-time voice modulation.
    Based on research from emotional speech datasets (RAVDESS, CREMA-D).
    """
    
    # Preset vibe configurations based on emotional speech research
    VIBE_PRESETS = {
        # Basic emotions
        "neutral": VibeParams(
            pitch=1.0, energy=1.0, speed=1.0,
            emotion_id=EmotionID.NEUTRAL,
            warmth=1.0, expressiveness=1.0
        ),
        "calm": VibeParams(
            pitch=0.9, energy=0.8, speed=0.85,
            emotion_id=EmotionID.CALM,
            warmth=1.1, expressiveness=0.8
        ),
        "happy": VibeParams(
            pitch=1.15, energy=1.2, speed=1.1,
            emotion_id=EmotionID.HAPPY,
            warmth=1.2, expressiveness=1.3
        ),
        "sad": VibeParams(
            pitch=0.85, energy=0.7, speed=0.8,
            emotion_id=EmotionID.SAD,
            warmth=0.9, expressiveness=0.7
        ),
        "angry": VibeParams(
            pitch=1.1, energy=1.4, speed=1.2,
            emotion_id=EmotionID.ANGRY,
            warmth=0.7, expressiveness=1.4
        ),
        "fearful": VibeParams(
            pitch=1.2, energy=0.9, speed=1.3,
            emotion_id=EmotionID.FEARFUL,
            warmth=0.8, expressiveness=1.2
        ),
        "disgust": VibeParams(
            pitch=0.8, energy=1.1, speed=0.9,
            emotion_id=EmotionID.DISGUST,
            warmth=0.6, expressiveness=1.1
        ),
        "surprised": VibeParams(
            pitch=1.25, energy=1.3, speed=1.15,
            emotion_id=EmotionID.SURPRISED,
            warmth=1.0, expressiveness=1.4
        ),
        
        # Extended vibes for podcast/presentation
        "excited": VibeParams(
            pitch=1.1, energy=1.3, speed=1.15,
            emotion_id=EmotionID.EXCITED,
            warmth=1.1, expressiveness=1.4
        ),
        "professional": VibeParams(
            pitch=1.0, energy=1.0, speed=0.95,
            emotion_id=EmotionID.PROFESSIONAL,
            warmth=0.9, expressiveness=0.9
        ),
        "casual": VibeParams(
            pitch=1.05, energy=0.9, speed=1.0,
            emotion_id=EmotionID.CASUAL,
            warmth=1.2, expressiveness=1.1
        ),
        "energetic": VibeParams(
            pitch=1.2, energy=1.4, speed=1.25,
            emotion_id=EmotionID.ENERGETIC,
            warmth=1.1, expressiveness=1.5
        ),
        "warm": VibeParams(
            pitch=0.95, energy=0.9, speed=0.9,
            emotion_id=EmotionID.WARM,
            warmth=1.3, expressiveness=1.1
        ),
        "serious": VibeParams(
            pitch=0.95, energy=1.0, speed=0.9,
            emotion_id=EmotionID.SERIOUS,
            warmth=0.8, expressiveness=0.8
        ),
    }
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize vibe modulator.
        
        Args:
            config_path: Path to custom vibe configuration file
        """
        self.config_path = config_path
        self.custom_vibes: Dict[str, VibeParams] = {}
        
        # Load custom vibes if config exists
        if config_path and Path(config_path).exists():
            self._load_custom_vibes()
    
    def _load_custom_vibes(self):
        """Load custom vibe configurations from file."""
        try:
            with open(self.config_path, 'r') as f:
                data = json.load(f)
            
            for name, params in data.items():
                self.custom_vibes[name] = VibeParams.from_dict(params)
        except Exception as e:
            print(f"Failed to load custom vibes: {e}")
    
    def get_vibe_params(self, vibe_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get vibe parameters by name.
        
        Args:
            vibe_name: Name of the vibe preset (e.g., 'excited', 'calm')
            
        Returns:
            Dictionary of vibe parameters
        """
        if vibe_name is None:
            vibe_name = "neutral"
        
        # Check custom vibes first
        if vibe_name in self.custom_vibes:
            return self.custom_vibes[vibe_name].to_dict()
        
        # Check presets
        if vibe_name in self.VIBE_PRESETS:
            return self.VIBE_PRESETS[vibe_name].to_dict()
        
        # Return neutral if not found
        return self.VIBE_PRESETS["neutral"].to_dict()
    
    def register_vibe(self, name: str, params: VibeParams) -> None:
        """
        Register a custom vibe preset.
        
        Args:
            name: Name for the vibe preset
            params: Vibe parameters
        """
        self.custom_vibes[name] = params
        
        # Save to config if path set
        if self.config_path:
            self._save_custom_vibes()
    
    def _save_custom_vibes(self):
        """Save custom vibes to configuration file."""
        try:
            data = {name: params.to_dict() for name, params in self.custom_vibes.items()}
            
            Path(self.config_path).parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_path, 'w') as f:
                json.dump(data, f, indent=4)
        except Exception as e:
            print(f"Failed to save custom vibes: {e}")
    
    def list_vibes(self) -> list:
        """List all available vibe presets."""
        presets = list(self.VIBE_PRESETS.keys())
        customs = list(self.custom_vibes.keys())
        return presets + customs
    
    def modulate_audio_params(self, base_params: Dict[str, float], vibe_name: str) -> Dict[str, float]:
        """
        Apply vibe modulation to base audio parameters.
        
        Args:
            base_params: Base audio parameters
            vibe_name: Vibe preset to apply
            
        Returns:
            Modulated parameters
        """
        vibe = self.get_vibe_params(vibe_name)
        
        modulated = base_params.copy()
        modulated["pitch"] = base_params.get("pitch", 1.0) * vibe["pitch"]
        modulated["energy"] = base_params.get("energy", 1.0) * vibe["energy"]
        modulated["speed"] = base_params.get("speed", 1.0) * vibe["speed"]
        
        return modulated
    
    def get_emotion_id(self, vibe_name: str) -> int:
        """Get emotion ID for a vibe name."""
        params = self.get_vibe_params(vibe_name)
        return params["emotion_id"]
    
    def blend_vibes(self, vibe1: str, vibe2: str, ratio: float = 0.5) -> VibeParams:
        """
        Blend two vibe presets.
        
        Args:
            vibe1: First vibe name
            vibe2: Second vibe name
            ratio: Blend ratio (0.0 = all vibe1, 1.0 = all vibe2)
            
        Returns:
            Blended vibe parameters
        """
        p1 = VibeParams.from_dict(self.get_vibe_params(vibe1))
        p2 = VibeParams.from_dict(self.get_vibe_params(vibe2))
        
        return VibeParams(
            pitch=p1.pitch * (1 - ratio) + p2.pitch * ratio,
            energy=p1.energy * (1 - ratio) + p2.energy * ratio,
            speed=p1.speed * (1 - ratio) + p2.speed * ratio,
            emotion_id=p1.emotion_id if ratio < 0.5 else p2.emotion_id,
            warmth=p1.warmth * (1 - ratio) + p2.warmth * ratio,
            brightness=p1.brightness * (1 - ratio) + p2.brightness * ratio,
            clarity=p1.clarity * (1 - ratio) + p2.clarity * ratio,
            expressiveness=p1.expressiveness * (1 - ratio) + p2.expressiveness * ratio,
        )


# Global instance for convenience
vibe_modulator = VibeModulator()
