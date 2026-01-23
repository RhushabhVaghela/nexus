from typing import Dict, Any, List

class SchemaNormalizer:
    """
    Normalizes diverse dataset schemas into Nexus standard multimodal messages.
    """
    
    # Mapping of dataset folder names to their normalization logic
    MAPPINGS = {
        "google_MusicCaps": {
            "user": "Describe this music.",
            "assistant_key": "caption",
            "media_key": "ytid", # Requires path joining logic in loader
            "modality": "audio"
        },
        "LDJnr_Pure-Dove": {
            "user_key": "instruction",
            "assistant_key": "response",
            "modality": "text"
        },
        "LucasFang_JourneyDB-GoT": {
            "user": "What is in this image?",
            "assistant_key": "caption",
            "media_key": "image_path",
            "modality": "image"
        },
        "E-MM1-100M": {
            "user": "Describe this.",
            "assistant_key": "caption",
            "modality": "multimodal"
        }
    }

    @staticmethod
    def normalize(sample: Dict[str, Any], dataset_name: str) -> Dict[str, Any]:
        """
        Convert a raw sample into the standard Nexus schema.
        """
        mapping = SchemaNormalizer.MAPPINGS.get(dataset_name, {})
        
        # Standard Nexus schema
        normalized = {
            "id": str(sample.get("id", sample.get("hash", "unknown"))),
            "messages": [],
            "modalities": {}
        }
        
        # Determine content
        user_content = mapping.get("user", sample.get("instruction", sample.get("user", "Describe the content.")))
        assistant_content = sample.get(mapping.get("assistant_key", "caption"), sample.get("output", sample.get("assistant", "")))
        
        normalized["messages"] = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content}
        ]
        
        # Modality handling
        modality = mapping.get("modality", "text")
        if modality == "image" or "image_path" in sample:
            path = sample.get("image_path", sample.get("path", ""))
            if path:
                normalized["modalities"]["image"] = [{"path": str(path)}]
        elif modality == "audio" or "audio_path" in sample:
            path = sample.get("audio_path", sample.get("path", ""))
            if path:
                normalized["modalities"]["audio"] = [{"path": str(path)}]
        elif modality == "video" or "video_path" in sample:
            path = sample.get("video_path", sample.get("path", ""))
            if path:
                normalized["modalities"]["video"] = [{"path": str(path)}]
                
        return normalized
