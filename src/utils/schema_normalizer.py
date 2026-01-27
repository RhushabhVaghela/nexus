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
        },
        "openai_gsm8k": {
            "user_key": "question",
            "assistant_key": "answer",
            "modality": "text",
            "instruction": "Solve the following math problem step by step."
        },
        "cais_mmlu": {
            "user_key": "question",
            "choices_key": "choices",
            "answer_key": "answer",
            "modality": "text",
            "instruction": "Answer the multiple choice question."
        },
        "MiniMaxAI_OctoCodingBench": {
            "user_key": "question",
            "assistant_key": "answer",
            "modality": "code"
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
        
        # Special handling for MMLU (Multiple Choice)
        if "choices_key" in mapping and mapping["choices_key"] in sample:
            question = sample.get(mapping["user_key"], "")
            choices = sample.get(mapping["choices_key"], [])
            answer_idx = sample.get(mapping["answer_key"], 0)
            
            options_text = "\n".join([f"{chr(65+i)}. {choice}" for i, choice in enumerate(choices)])
            user_content = f"{mapping.get('instruction', '')}\n\n{question}\n{options_text}\nAnswer:"
            assistant_content = chr(65 + answer_idx)
            
            normalized["messages"] = [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": assistant_content}
            ]
            normalized["modalities"]["type"] = "multiple_choice"
            return normalized

        # Determine content for standard Q&A
        user_key = mapping.get("user_key", "instruction")
        assistant_key = mapping.get("assistant_key", "output")
        
        # Fallback to defaults if keys not in mapping
        if "user" in mapping:
             user_content = mapping["user"]
        else:
             user_content = sample.get(user_key, sample.get("question", sample.get("user", "Describe the content.")))
             if "instruction" in mapping:
                 user_content = f"{mapping['instruction']}\n\n{user_content}"

        assistant_content = sample.get(assistant_key, sample.get("response", sample.get("assistant", "")))
        
        normalized["messages"] = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content}
        ]
        
        # Modality handling
        modality = mapping.get("modality", "text")
        media_key = mapping.get("media_key")
        
        if modality == "image" or "image_path" in sample:
            path = sample.get(media_key) if media_key else sample.get("image_path", sample.get("path"))
            if path:
                normalized["modalities"]["image"] = [{"path": str(path)}]
        elif modality == "audio" or "audio_path" in sample:
            path = sample.get(media_key) if media_key else sample.get("audio_path", sample.get("path"))
            if path:
                normalized["modalities"]["audio"] = [{"path": str(path)}]
        elif modality == "video" or "video_path" in sample:
            path = sample.get(media_key) if media_key else sample.get("video_path", sample.get("path"))
            if path:
                normalized["modalities"]["video"] = [{"path": str(path)}]
                
        return normalized
