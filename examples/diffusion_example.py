#!/usr/bin/env python3
"""
Nexus Diffusion Example

Demonstrates image generation using various diffusion models including:
- Stable Diffusion XL
- FLUX
- SD3
- Image-to-image
- Inpainting
"""

import torch
from PIL import Image
import argparse

from src.models.diffusion import ImagePipeline, PipelineConfig, DiffusionPipelineLoader


def example_text_to_image():
    """Generate image from text prompt."""
    print("=" * 60)
    print("Example 1: Text-to-Image Generation")
    print("=" * 60)
    
    # Configure for SDXL
    config = PipelineConfig(
        model_id="stabilityai/stable-diffusion-xl-base-1.0",
        model_type="sdxl",
        device="auto",
        dtype=torch.float16,
        enable_vae_slicing=True,
        default_steps=30,
    )
    
    with ImagePipeline(config) as pipeline:
        result = pipeline.generate(
            prompt="a beautiful landscape with mountains and a lake at sunset, highly detailed, 8k resolution, professional photography",
            negative_prompt="blur, low quality, distorted, ugly",
            height=1024,
            width=1024,
            num_images_per_prompt=1,
            seed=42
        )
        
        image = result["images"][0]
        image.save("output_landscape.png")
        print("✓ Generated: output_landscape.png")


def example_preset_loading():
    """Load model using preset."""
    print("\n" + "=" * 60)
    print("Example 2: Using Presets")
    print("=" * 60)
    
    loader = DiffusionPipelineLoader()
    
    # List available presets
    presets = loader.list_presets()
    print(f"Available presets: {', '.join(presets[:5])}...")
    
    # Load FLUX Schnell (fast generation)
    print("\nLoading FLUX Schnell (4 steps)...")
    pipeline = loader.load("flux-schnell")
    
    result = pipeline.generate(
        prompt="a futuristic city with flying cars and neon lights",
        seed=123
    )
    
    result["images"][0].save("output_flux_city.png")
    print("✓ Generated: output_flux_city.png")
    
    pipeline.unload()


def example_image_to_image():
    """Generate image variations."""
    print("\n" + "=" * 60)
    print("Example 3: Image-to-Image")
    print("=" * 60)
    
    # Create a simple input image
    print("Creating input image...")
    input_image = Image.new("RGB", (512, 512), color=(100, 150, 200))
    input_image.save("input_base.png")
    
    config = PipelineConfig(
        model_id="stabilityai/stable-diffusion-2-1",
        model_type="sd",
        dtype=torch.float16
    )
    
    with ImagePipeline(config) as pipeline:
        print("Generating variations...")
        result = pipeline.generate_variations(
            image=input_image,
            prompt="a beautiful underwater scene with coral reefs and tropical fish",
            strength=0.75,  # How much to change the image
            num_images=2,
            seed=42
        )
        
        for i, img in enumerate(result["images"]):
            img.save(f"output_variation_{i}.png")
            print(f"✓ Generated: output_variation_{i}.png")


def example_inpainting():
    """Inpaint masked regions."""
    print("\n" + "=" * 60)
    print("Example 4: Inpainting")
    print("=" * 60)
    
    # Create base image
    base_image = Image.new("RGB", (512, 512), color=(135, 206, 235))  # Sky blue
    base_image.save("input_inpaint.png")
    
    # Create mask (white = inpaint, black = keep)
    mask = Image.new("L", (512, 512), color=0)  # All black
    # Draw white rectangle in center
    from PIL import ImageDraw
    draw = ImageDraw.Draw(mask)
    draw.rectangle([150, 150, 350, 350], fill=255)
    mask.save("input_mask.png")
    
    config = PipelineConfig(
        model_id="stabilityai/stable-diffusion-2-inpainting",
        model_type="sd",
        dtype=torch.float16
    )
    
    with ImagePipeline(config) as pipeline:
        result = pipeline.inpaint(
            image=base_image,
            mask=mask,
            prompt="a beautiful garden with colorful flowers and butterflies",
            num_inference_steps=25,
            seed=42
        )
        
        result["images"][0].save("output_inpainted.png")
        print("✓ Generated: output_inpainted.png")


def example_quantized_loading():
    """Load quantized model."""
    print("\n" + "=" * 60)
    print("Example 5: Quantized Model Loading")
    print("=" * 60)
    
    loader = DiffusionPipelineLoader()
    
    print("Loading with FP8 quantization...")
    try:
        pipeline = loader.load_quantized(
            "sdxl-base",
            quantization="fp8",
            enable_vae_slicing=True
        )
        
        result = pipeline.generate(
            prompt="an abstract digital art piece with vibrant colors",
            seed=999
        )
        
        result["images"][0].save("output_quantized.png")
        print("✓ Generated: output_quantized.png")
        
        pipeline.unload()
    except Exception as e:
        print(f"⚠ FP8 requires specific hardware: {e}")


def example_batch_generation():
    """Generate multiple images at once."""
    print("\n" + "=" * 60)
    print("Example 6: Batch Generation")
    print("=" * 60)
    
    config = PipelineConfig(
        model_id="stabilityai/sdxl-turbo",
        model_type="sdxl",
        dtype=torch.float16,
        default_steps=4
    )
    
    prompts = [
        "a cat wearing sunglasses",
        "a dog playing in the park",
        "a bird sitting on a branch",
    ]
    
    with ImagePipeline(config) as pipeline:
        result = pipeline.generate(
            prompt=prompts,
            num_images_per_prompt=1,
            seed=42
        )
        
        for i, img in enumerate(result["images"]):
            img.save(f"output_batch_{i}.png")
            print(f"✓ Generated: output_batch_{i}.png (prompt: {prompts[i][:30]}...)")


def example_memory_optimization():
    """Demonstrate memory optimization techniques."""
    print("\n" + "=" * 60)
    print("Example 7: Memory Optimization")
    print("=" * 60)
    
    print("Configuring for low VRAM (8GB)...")
    
    config = PipelineConfig(
        model_id="stabilityai/stable-diffusion-xl-base-1.0",
        model_type="sdxl",
        dtype=torch.float16,
        # Memory optimizations
        enable_vae_slicing=True,      # Essential for 1024px+
        enable_vae_tiling=True,       # For very large images
        enable_cpu_offload=True,      # Move unused parts to CPU
        # Reduce resolution
        default_height=768,
        default_width=768,
    )
    
    print(f"Configuration:")
    print(f"  - VAE Slicing: {config.enable_vae_slicing}")
    print(f"  - VAE Tiling: {config.enable_vae_tiling}")
    print(f"  - CPU Offload: {config.enable_cpu_offload}")
    print(f"  - Resolution: {config.default_width}x{config.default_height}")
    
    with ImagePipeline(config) as pipeline:
        result = pipeline.generate(
            prompt="a peaceful mountain landscape",
            seed=1
        )
        
        result["images"][0].save("output_optimized.png")
        print("✓ Generated: output_optimized.png")


def main():
    parser = argparse.ArgumentParser(description="Nexus Diffusion Examples")
    parser.add_argument(
        "--example",
        type=int,
        choices=range(1, 8),
        help="Run specific example (1-7)"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all examples"
    )
    
    args = parser.parse_args()
    
    examples = {
        1: example_text_to_image,
        2: example_preset_loading,
        3: example_image_to_image,
        4: example_inpainting,
        5: example_quantized_loading,
        6: example_batch_generation,
        7: example_memory_optimization,
    }
    
    if args.all:
        for num in range(1, 8):
            try:
                examples[num]()
            except Exception as e:
                print(f"\n⚠ Example {num} failed: {e}")
    elif args.example:
        examples[args.example]()
    else:
        # Run first example by default
        example_text_to_image()
        print("\n" + "=" * 60)
        print("Run with --all to see all examples")
        print("Run with --example N to run specific example")
        print("=" * 60)


if __name__ == "__main__":
    main()
