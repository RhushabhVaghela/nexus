
import os
import json
from typing import Dict, List, Optional
from pathlib import Path

class VoiceRegistry:
    """
    Registry for PersonaPlex and VibeVoice identities.
    Manages Factory Defaults (NAT/VAR) and User-Cloned Voices.
    """
    
    # Preset Definitions from NVIDIA PersonaPlex
    PRESETS = {
        "NATF0": "Natural Female 0 - Conversational, steady",
        "NATF1": "Natural Female 1 - Warm, engaging",
        "NATF2": "Natural Female 2 - Professional, clear",
        "NATF3": "Natural Female 3 - Young, energetic",
        "NATM0": "Natural Male 0 - Deep, authoritative",
        "NATM1": "Natural Male 1 - Friendly, neighborly",
        "NATM2": "Natural Male 2 - Calm, analytical",
        "NATM3": "Natural Male 3 - Sophisticated, smooth",
        "VARF0": "Variety Female 0 - High pitch, expressive",
        "VARF1": "Variety Female 1 - Raspy, unique",
        "VARF2": "Variety Female 2 - Dynamic, emotional",
        "VARF3": "Variety Female 3 - Formal, strict",
        "VARF4": "Variety Female 4 - Soft, whispered",
        "VARM0": "Variety Male 0 - Intense, dramatic",
        "VARM1": "Variety Male 1 - Laid back, casual",
        "VARM2": "Variety Male 2 - Energetic, broadcast style",
        "VARM3": "Variety Male 3 - Gritty, mature",
        "VARM4": "Variety Male 4 - Character-like, whimsical",
    }

    def __init__(self, storage_path: str = "/mnt/e/data/models/voice_dna"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.custom_voices: Dict[str, Dict[str, str]] = {}
        self._load_custom_voices()

    def _load_custom_voices(self):
        """Load registered custom voice embeddings from disk."""
        index_file = self.storage_path / "voice_index.json"
        if index_file.exists():
            with open(index_file, "r") as f:
                self.custom_voices = json.load(f)

    def register_voice(self, name: str, embedding_path: str, description: str = ""):
        """Register a new cloned voice DNA."""
        self.custom_voices[name] = {
            "path": embedding_path,
            "description": description,
            "type": "cloned"
        }
        self._save_index()

    def _save_index(self):
        index_file = self.storage_path / "voice_index.json"
        with open(index_file, "w") as f:
            json.dump(self.custom_voices, f, indent=4)

    def list_voices(self) -> Dict[str, Dict[str, str]]:
        """Combine presets and custom voices."""
        all_voices = {k: {"description": v, "type": "preset"} for k, v in self.PRESETS.items()}
        all_voices.update(self.custom_voices)
        return all_voices

    def get_voice_dna(self, voice_id: str) -> Optional[str]:
        """Get the path to the voice DNA (embedding) for synthesis."""
        if voice_id in self.PRESETS:
            # Presets are built into the model weights or stored in a specific asset dir
            return f"builtin://{voice_id}"
        
        voice_data = self.custom_voices.get(voice_id)
        if voice_data:
            return voice_data.get("path")
        return None

voice_registry = VoiceRegistry()
