#!/usr/bin/env python3
"""
Multimodal Processing Example - Nexus

Demonstrates how to work with multimodal models (vision, audio, video)
using the Nexus multimodal pipeline.

Usage:
    python examples/multimodal_example.py --mode vision --image path/to/image.jpg
    python examples/multimodal_example.py --mode audio --audio path/to/audio.wav
    python examples/multimodal_example.py --mode text --prompt "Describe this scene"

Features:
- Vision understanding (image Q&A)
- Audio processing (speech, music)
- Video understanding
- Multimodal fusion
"""

import argparse
import torch
from pathlib import Path
from typing import Optional
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class MultimodalProcessor:
    """Simple multimodal processor for vision, audio, and text."""
    
    def __init__(self, model_name: str = "microsoft/Phi-3-vision-128k-instruct"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.processor = None
        
    def load_model(self):
        """Load the multimodal model."""
        try:
            from transformers import AutoProcessor, AutoModelForVision2Seq
            
            print(f"Loading multimodal model: {self.model_name}")
            self.processor = AutoProcessor.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            self.model = AutoModelForVision2Seq.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True
            )
            print("✓ Model loaded successfully")
            return True
        except Exception as e:
            print(f"✗ Failed to load model: {e}")
            return False
    
    def process_vision(self, image_path: str, prompt: str = "Describe this image in detail.") -> str:
        """Process an image with a text prompt."""
        try:
            from PIL import Image
            
            if self.model is None:
                self.load_model()
            
            print(f"Loading image: {image_path}")
            image = Image.open(image_path).convert("RGB")
            
            messages = [{"role": "user", "content": f"<|image|>{prompt}"}]
            prompt_text = self.processor.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            inputs = self.processor(prompt_text, [image], return_tensors="pt").to(self.model.device)
            
            with torch.no_grad():
                outputs = self.model.generate(**inputs, max_new_tokens=256, temperature=0.7, do_sample=True)
            
            response = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
            return response
        except Exception as e:
            return f"Error processing image: {e}"
    
    def process_audio(self, audio_path: str, prompt: str = "Transcribe this audio.") -> str:
        """Process audio with a text prompt."""
        print(f"Audio processing: {audio_path}")
        return (
            "Audio processing requires a speech recognition model.\n"
            "Recommended: openai/whisper-large-v3, microsoft/wavlm-large"
        )
    
    def process_video(self, video_path: str, prompt: str = "Describe what happens in this video.") -> str:
        """Process video with a text prompt."""
        try:
            import av
            container = av.open(video_path)
            frames = []
            for frame in container.decode(video=0):
                frames.append(frame.to_image())
                if len(frames) >= 8:
                    break
            return f"Video: Extracted {len(frames)} frames. Use Qwen2-VL or LLaVA-Video for full understanding."
        except ImportError:
            return "Error: av library required (pip install av)"
        except Exception as e:
            return f"Error processing video: {e}"
    
    def process_text(self, prompt: str) -> str:
        """Process text-only input."""
        if self.model is None:
            self.load_model()
        
        messages = [{"role": "user", "content": prompt}]
        prompt_text = self.processor.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        inputs = self.processor(prompt_text, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=256, temperature=0.7, do_sample=True)
        
        return self.processor.batch_decode(outputs, skip_special_tokens=True)[0]


def create_sample_image(output_path: str = "sample_image.png"):
    """Create a sample image for testing."""
    try:
        from PIL import Image, ImageDraw
        img = Image.new('RGB', (400, 300), color='lightblue')
        draw = ImageDraw.Draw(img)
        draw.rectangle([50, 50, 150, 150], fill='red', outline='black', width=2)
        draw.ellipse([200, 50, 350, 200], fill='green', outline='black', width=2)
        img.save(output_path)
        print(f"✓ Created sample image: {output_path}")
        return output_path
    except ImportError:
        print("✗ PIL not available")
        return None


def main():
    parser = argparse.ArgumentParser(description="Multimodal Processing Example")
    parser.add_argument("--mode", type=str, choices=["vision", "audio", "video", "text", "all"], default="text")
    parser.add_argument("--model", type=str, default="microsoft/Phi-3-vision-128k-instruct")
    parser.add_argument("--image", type=str, help="Path to image file")
    parser.add_argument("--audio", type=str, help="Path to audio file")
    parser.add_argument("--video", type=str, help="Path to video file")
    parser.add_argument("--prompt", type=str, default="Describe this in detail.")
    
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print(f"NEXUS - Multimodal Processing Example")
    print(f"{'='*60}\n")
    
    processor = MultimodalProcessor(model_name=args.model)
    
    if args.mode in ["vision", "all"]:
        print("\n[Vision Mode]")
        image_path = args.image or create_sample_image()
        if image_path and Path(image_path).exists():
            result = processor.process_vision(image_path, args.prompt)
            print(f"Response:\n{result}")
    
    if args.mode in ["audio", "all"]:
        print("\n[Audio Mode]")
        result = processor.process_audio(args.audio or "sample_audio.wav", args.prompt)
        print(f"Response:\n{result}")
    
    if args.mode in ["video", "all"]:
        print("\n[Video Mode]")
        result = processor.process_video(args.video or "sample_video.mp4", args.prompt)
        print(f"Response:\n{result}")
    
    if args.mode == "text":
        print("\n[Text Mode]")
        result = processor.process_text(args.prompt)
        print(f"Prompt: {args.prompt}")
        print(f"Response:\n{result}")
    
    print(f"\n{'='*60}")
    print("Multimodal Example Complete!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
