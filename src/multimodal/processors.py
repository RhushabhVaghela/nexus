"""
multimodal/processors.py
Multimodal processors for handling vision, audio, and text inputs.
Extends prompt repetition to multimodal contexts (Paper 2512.14982).
"""

from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass
from pathlib import Path
import base64
import io
import logging

logger = logging.getLogger(__name__)


@dataclass
class ModalityData:
    """Container for modality-specific data."""
    modality: str  # "text", "image", "audio", "video"
    content: Any
    metadata: Dict[str, Any]


class MultimodalRepetitionProcessor:
    """
    Extends prompt repetition to multimodal contexts.
    Based on paper 2512.14982 - Prompt Repetition across modalities.
    """
    
    def __init__(self, 
                 image_repetition_style: str = "descriptor",
                 audio_repetition_style: str = "transcript",
                 max_image_repetitions: int = 2,
                 max_audio_repetitions: int = 2):
        """
        Initialize multimodal repetition processor.
        
        Args:
            image_repetition_style: How to repeat image content ("descriptor", "duplicate", "detail")
            audio_repetition_style: How to repeat audio content ("transcript", "duplicate", "summary")
            max_image_repetitions: Maximum number of image repetitions
            max_audio_repetitions: Maximum number of audio repetitions
        """
        self.image_repetition_style = image_repetition_style
        self.audio_repetition_style = audio_repetition_style
        self.max_image_repetitions = max_image_repetitions
        self.max_audio_repetitions = max_audio_repetitions
    
    def process_text_repetition(self, text: str, repetition_factor: int = 1) -> str:
        """
        Apply text-based repetition.
        
        Args:
            text: Input text
            repetition_factor: Number of times to repeat (1 = baseline)
            
        Returns:
            Repeated text
        """
        if repetition_factor <= 1:
            return text
        
        # Build repetition with natural language connectors
        repetitions = [text] * repetition_factor
        
        if repetition_factor == 2:
            return f"{repetitions[0]} Let me repeat that: {repetitions[1]}"
        elif repetition_factor == 3:
            return (f"{repetitions[0]} Let me repeat that: {repetitions[1]} "
                   f"Let me repeat that one more time: {repetitions[2]}")
        else:
            return " ".join(repetitions)
    
    def process_image_repetition(self, 
                                 image_data: Union[str, bytes],
                                 image_description: str = "",
                                 repetition_factor: int = 1) -> Dict[str, Any]:
        """
        Apply repetition to image modality.
        
        Args:
            image_data: Image path, base64 string, or bytes
            image_description: Text description of the image
            repetition_factor: Number of times to reference the image
            
        Returns:
            Dictionary with repeated image references
        """
        if repetition_factor <= 1:
            return {
                "images": [image_data],
                "image_descriptions": [image_description] if image_description else []
            }
        
        # Cap repetitions
        repetition_factor = min(repetition_factor, self.max_image_repetitions)
        
        repeated_images = []
        repeated_descriptions = []
        
        for i in range(repetition_factor):
            repeated_images.append(image_data)
            
            if self.image_repetition_style == "descriptor":
                # Use the same description with index
                desc = image_description if image_description else f"Image view {i+1}"
                repeated_descriptions.append(desc)
            elif self.image_repetition_style == "detail":
                # Add detail level to description
                if i == 0:
                    desc = image_description if image_description else "Original image"
                elif i == 1:
                    desc = f"Same image again for emphasis: {image_description}"
                else:
                    desc = f"Repeated view {i+1}: {image_description}"
                repeated_descriptions.append(desc)
            elif self.image_repetition_style == "duplicate":
                # Just duplicate without additional descriptors
                repeated_descriptions.append(image_description)
        
        return {
            "images": repeated_images,
            "image_descriptions": repeated_descriptions,
            "repetition_metadata": {
                "factor": repetition_factor,
                "style": self.image_repetition_style
            }
        }
    
    def process_audio_repetition(self,
                                 audio_data: Union[str, bytes],
                                 transcript: str = "",
                                 repetition_factor: int = 1) -> Dict[str, Any]:
        """
        Apply repetition to audio modality.
        
        Args:
            audio_data: Audio path, base64 string, or bytes
            transcript: Text transcript of the audio
            repetition_factor: Number of times to reference the audio
            
        Returns:
            Dictionary with repeated audio references
        """
        if repetition_factor <= 1:
            return {
                "audio": [audio_data],
                "transcripts": [transcript] if transcript else []
            }
        
        # Cap repetitions
        repetition_factor = min(repetition_factor, self.max_audio_repetitions)
        
        repeated_audio = []
        repeated_transcripts = []
        
        for i in range(repetition_factor):
            repeated_audio.append(audio_data)
            
            if self.audio_repetition_style == "transcript":
                # Repeat with transcript emphasis
                if i == 0:
                    text = transcript if transcript else "Audio content"
                elif i == 1:
                    text = f"Replaying: {transcript}" if transcript else "Audio content (replay)"
                else:
                    text = f"Replay {i+1}: {transcript}" if transcript else f"Audio content (replay {i+1})"
                repeated_transcripts.append(text)
            elif self.audio_repetition_style == "summary":
                # Summarize for subsequent repetitions
                if i == 0:
                    text = transcript if transcript else "Audio content"
                else:
                    text = f"Same audio as above (replay {i+1})"
                repeated_transcripts.append(text)
            elif self.audio_repetition_style == "duplicate":
                repeated_transcripts.append(transcript)
        
        return {
            "audio": repeated_audio,
            "transcripts": repeated_transcripts,
            "repetition_metadata": {
                "factor": repetition_factor,
                "style": self.audio_repetition_style
            }
        }
    
    def process_multimodal_prompt(self,
                                  text: str,
                                  images: Optional[List[Union[str, bytes]]] = None,
                                  audio: Optional[List[Union[str, bytes]]] = None,
                                  image_descriptions: Optional[List[str]] = None,
                                  audio_transcripts: Optional[List[str]] = None,
                                  repetition_factor: int = 1) -> Dict[str, Any]:
        """
        Process a complete multimodal prompt with repetition across all modalities.
        
        Args:
            text: Text prompt
            images: List of image data
            audio: List of audio data
            image_descriptions: Descriptions for images
            audio_transcripts: Transcripts for audio
            repetition_factor: Repetition factor to apply
            
        Returns:
            Processed multimodal prompt with repetitions
        """
        result = {
            "text": self.process_text_repetition(text, repetition_factor),
            "modality_repetitions": {}
        }
        
        # Process image repetitions
        if images:
            img_result = self.process_image_repetition(
                images[0] if len(images) == 1 else images,
                image_descriptions[0] if image_descriptions else "",
                repetition_factor
            )
            result["images"] = img_result.get("images", images)
            result["image_descriptions"] = img_result.get("image_descriptions", image_descriptions or [])
            result["modality_repetitions"]["image"] = img_result.get("repetition_metadata", {})
        
        # Process audio repetitions
        if audio:
            aud_result = self.process_audio_repetition(
                audio[0] if len(audio) == 1 else audio,
                audio_transcripts[0] if audio_transcripts else "",
                repetition_factor
            )
            result["audio"] = aud_result.get("audio", audio)
            result["audio_transcripts"] = aud_result.get("transcripts", audio_transcripts or [])
            result["modality_repetitions"]["audio"] = aud_result.get("repetition_metadata", {})
        
        return result
    
    def encode_image_base64(self, image_path: str) -> str:
        """Encode image file to base64 string."""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode('utf-8')
    
    def encode_audio_base64(self, audio_path: str) -> str:
        """Encode audio file to base64 string."""
        with open(audio_path, "rb") as f:
            return base64.b64encode(f.read()).decode('utf-8')


class VisionPromptProcessor:
    """Specialized processor for vision-language tasks with repetition."""
    
    def __init__(self, processor: Optional[MultimodalRepetitionProcessor] = None):
        self.processor = processor or MultimodalRepetitionProcessor()
    
    def process_visual_qa(self,
                         question: str,
                         image_path: str,
                         repetition_factor: int = 1) -> Dict[str, Any]:
        """
        Process visual question answering with repetition.
        
        Args:
            question: The question about the image
            image_path: Path to the image
            repetition_factor: Repetition factor
            
        Returns:
            Processed prompt with repeated image references
        """
        return self.processor.process_multimodal_prompt(
            text=question,
            images=[image_path],
            repetition_factor=repetition_factor
        )
    
    def process_image_captioning(self,
                                 image_path: str,
                                 style: str = "detailed",
                                 repetition_factor: int = 1) -> Dict[str, Any]:
        """
        Process image captioning with repetition for emphasis.
        
        Args:
            image_path: Path to the image
            style: Captioning style ("detailed", "concise", "creative")
            repetition_factor: Repetition factor
            
        Returns:
            Processed prompt
        """
        prompt_templates = {
            "detailed": "Describe this image in detail:",
            "concise": "Provide a brief caption for this image:",
            "creative": "Write a creative description of this image:"
        }
        
        prompt = prompt_templates.get(style, prompt_templates["detailed"])
        
        return self.processor.process_multimodal_prompt(
            text=prompt,
            images=[image_path],
            repetition_factor=repetition_factor
        )


class AudioPromptProcessor:
    """Specialized processor for audio-language tasks with repetition."""
    
    def __init__(self, processor: Optional[MultimodalRepetitionProcessor] = None):
        self.processor = processor or MultimodalRepetitionProcessor()
    
    def process_audio_transcription(self,
                                    audio_path: str,
                                    language_hint: Optional[str] = None,
                                    repetition_factor: int = 1) -> Dict[str, Any]:
        """
        Process audio transcription with repetition.
        
        Args:
            audio_path: Path to the audio file
            language_hint: Optional language hint
            repetition_factor: Repetition factor
            
        Returns:
            Processed prompt
        """
        if language_hint:
            prompt = f"Transcribe this audio in {language_hint}:"
        else:
            prompt = "Transcribe this audio:"
        
        return self.processor.process_multimodal_prompt(
            text=prompt,
            audio=[audio_path],
            repetition_factor=repetition_factor
        )
    
    def process_audio_qa(self,
                        question: str,
                        audio_path: str,
                        transcript_hint: Optional[str] = None,
                        repetition_factor: int = 1) -> Dict[str, Any]:
        """
        Process audio question answering with repetition.
        
        Args:
            question: Question about the audio
            audio_path: Path to the audio
            transcript_hint: Optional transcript hint
            repetition_factor: Repetition factor
            
        Returns:
            Processed prompt
        """
        if transcript_hint:
            prompt = f"Based on this audio (transcript: {transcript_hint}), {question}"
        else:
            prompt = question
        
        return self.processor.process_multimodal_prompt(
            text=prompt,
            audio=[audio_path],
            repetition_factor=repetition_factor
        )


class MultimodalFusionPipeline:
    """
    Pipeline for fusing multimodal inputs with repetition.
    Integrates vision and audio repetition into a unified format.
    """
    
    def __init__(self):
        self.processor = MultimodalRepetitionProcessor()
        self.vision_processor = VisionPromptProcessor(self.processor)
        self.audio_processor = AudioPromptProcessor(self.processor)
    
    def create_fused_prompt(self,
                           text: str,
                           images: Optional[List[str]] = None,
                           audio: Optional[List[str]] = None,
                           repetition_factor: int = 1,
                           fusion_mode: str = "sequential") -> Dict[str, Any]:
        """
        Create a fused multimodal prompt with repetition.
        
        Args:
            text: Base text prompt
            images: List of image paths
            audio: List of audio paths
            repetition_factor: Repetition factor for all modalities
            fusion_mode: How to fuse modalities ("sequential", "parallel")
            
        Returns:
            Fused multimodal prompt
        """
        if fusion_mode == "sequential":
            # Process each modality in sequence
            return self.processor.process_multimodal_prompt(
                text=text,
                images=images,
                audio=audio,
                repetition_factor=repetition_factor
            )
        elif fusion_mode == "parallel":
            # Create parallel references to all modalities
            result = {"text": text, "fusion_mode": fusion_mode}
            
            if images:
                result["images"] = images * repetition_factor
                result["image_count"] = len(images) * repetition_factor
            
            if audio:
                result["audio"] = audio * repetition_factor
                result["audio_count"] = len(audio) * repetition_factor
            
            return result
        else:
            raise ValueError(f"Unknown fusion mode: {fusion_mode}")
    
    def process_batch(self,
                     prompts: List[Dict[str, Any]],
                     repetition_factor: int = 1) -> List[Dict[str, Any]]:
        """
        Process a batch of multimodal prompts.
        
        Args:
            prompts: List of prompt dictionaries
            repetition_factor: Repetition factor to apply to all
            
        Returns:
            List of processed prompts
        """
        results = []
        for prompt in prompts:
            processed = self.create_fused_prompt(
                text=prompt.get("text", ""),
                images=prompt.get("images"),
                audio=prompt.get("audio"),
                repetition_factor=repetition_factor,
                fusion_mode=prompt.get("fusion_mode", "sequential")
            )
            results.append(processed)
        return results