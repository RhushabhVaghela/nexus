#!/usr/bin/env python3
"""
Omni Model Inference
Generation pipeline for Qwen2.5-Omni models.

Supports:
- Text generation via thinker
- Audio generation via talker (if available)
- Streaming output

Usage:
    from src.omni.inference import OmniInference
    
    inference = OmniInference("/path/to/omni-model")
    response = inference.generate("Hello, how are you?")
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any, Union, Generator
from dataclasses import dataclass

import torch

logger = logging.getLogger(__name__)


@dataclass
class GenerationConfig:
    """Configuration for text generation."""
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    do_sample: bool = True
    repetition_penalty: float = 1.1
    stream: bool = False


class OmniInference:
    """
    Inference pipeline for Qwen2.5-Omni models.
    """
    
    def __init__(
        self,
        model_path: Union[str, Path],
        enable_audio: bool = False,
        device: str = "cuda",
    ):
        """
        Initialize Omni inference.
        
        Args:
            model_path: Path to Omni model
            enable_audio: Enable audio output generation
            device: Device for inference
        """
        from .loader import OmniModelLoader
        
        self.model_path = Path(model_path)
        self.enable_audio = enable_audio
        self.device = device
        
        loader = OmniModelLoader()
        mode = "full" if enable_audio else "thinker_only"
        
        self.model, self.tokenizer = loader.load_for_inference(
            model_path,
            enable_audio=enable_audio,
        )
        
        logger.info(f"Omni inference initialized (audio: {enable_audio})")
    
    def generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
    ) -> str:
        """
        Generate text response.
        
        Args:
            prompt: Input text prompt
            config: Generation configuration
        
        Returns:
            Generated text
        """
        if config is None:
            config = GenerationConfig()
        
        # Tokenize input
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=config.max_new_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                top_k=config.top_k,
                do_sample=config.do_sample,
                repetition_penalty=config.repetition_penalty,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode
        response = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
        
        return response
    
    def generate_stream(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
    ) -> Generator[str, None, None]:
        """
        Generate text response with streaming.
        
        Args:
            prompt: Input text prompt
            config: Generation configuration
        
        Yields:
            Generated text tokens
        """
        if config is None:
            config = GenerationConfig(stream=True)
        
        from transformers import TextIteratorStreamer
        from threading import Thread
        
        # Tokenize input
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # Setup streamer
        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_special_tokens=True,
            skip_prompt=True,
        )
        
        # Generate in separate thread
        generation_kwargs = dict(
            **inputs,
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
            do_sample=config.do_sample,
            repetition_penalty=config.repetition_penalty,
            streamer=streamer,
        )
        
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()
        
        # Yield tokens as they're generated
        for text in streamer:
            yield text
        
        thread.join()
    
    def generate_with_audio(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
    ) -> Dict[str, Any]:
        """
        Generate text and audio response.
        
        Args:
            prompt: Input text prompt
            config: Generation configuration
        
        Returns:
            Dict with "text" and "audio" (if enabled)
        """
        text_output = self.generate(prompt, config)

        if not self.enable_audio:
            return {"text": text_output, "audio": None}
        
        # For full Omni model with talker
        # This requires the talker component to be loaded
        if not hasattr(self.model, 'talker') and not hasattr(self.model, 'speech_generator'):
            logger.warning("Model does not have talker/speech component, returning text only")
            return {"text": text_output, "audio": None}

        # Audio Generation Logic
        try:
            # 1. Tokenize text for speech generation (if separated)
            # Some models use the same tokens, some need specific text-to-speech tokens
            
            # 2. Invoke Talker/Speech Generator
            # We support two common interfaces: .talker() and .generate_speech()
            audio_data = None
            
            if hasattr(self.model, 'talker'):
                # Interface A: Qwen-Omni style 'talker' submodule
                # Expects hidden states or text input
                # This is a hypothetical interface implementation based on common multimodal patterns
                logger.info("Generating audio with .talker component...")
                audio_out = self.model.talker.generate(
                    text_input=text_output, 
                    voice=kwargs.get("voice", "default")
                )
                audio_data = audio_out.audio_values if hasattr(audio_out, 'audio_values') else audio_out
                
            elif hasattr(self.model, 'generate_speech'):
                # Interface B: Unified generate_speech method
                audio_data = self.model.generate_speech(text_output)

            return {"text": text_output, "audio": audio_data}

        except Exception as e:
            logger.error(f"Failed to generate audio: {e}")
            logger.warning("Returning text-only due to audio generation failure.")
            return {"text": text_output, "audio": None}
    
    def chat(
        self,
        messages: list[Dict[str, str]],
        config: Optional[GenerationConfig] = None,
    ) -> str:
        """
        Chat interface for conversation.
        
        Args:
            messages: List of {"role": str, "content": str}
            config: Generation configuration
        
        Returns:
            Generated assistant response
        """
        # Format messages using chat template
        if hasattr(self.tokenizer, 'apply_chat_template'):
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            # Fallback formatting
            prompt = ""
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role == "system":
                    prompt += f"<|im_start|>system\n{content}<|im_end|>\n"
                elif role == "user":
                    prompt += f"<|im_start|>user\n{content}<|im_end|>\n"
                elif role == "assistant":
                    prompt += f"<|im_start|>assistant\n{content}<|im_end|>\n"
            prompt += "<|im_start|>assistant\n"
        
        return self.generate(prompt, config)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Omni Inference")
    parser.add_argument("model_path", help="Path to Omni model")
    parser.add_argument("--prompt", default="Hello, how are you?", help="Input prompt")
    parser.add_argument("--stream", action="store_true", help="Enable streaming")
    args = parser.parse_args()
    
    inference = OmniInference(args.model_path)
    
    if args.stream:
        for token in inference.generate_stream(args.prompt):
            print(token, end="", flush=True)
        print()
    else:
        response = inference.generate(args.prompt)
        print(response)
