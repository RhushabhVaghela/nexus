#!/usr/bin/env python3
"""
Nexus Video Generation Example

Demonstrates video generation using various models including:
- LTX-Video (text-to-video)
- Stable Video Diffusion (image-to-video)
- Frame-by-frame generation
- Temporal consistency
"""

import torch
from PIL import Image, ImageDraw
import numpy as np
import argparse

from src.models.video import VideoPipeline, VideoConfig
from src.models.video.frame_generator import FrameGenerator, FrameGenerationConfig
from src.models.video.temporal_consistency import TemporalConsistencyProcessor
from src.models.diffusion import ImagePipeline, PipelineConfig


def example_text_to_video():
    """Generate video from text prompt."""
    print("=" * 60)
    print("Example 1: Text-to-Video Generation")
    print("=" * 60)
    
    config = VideoConfig(
        model_id="Lightricks/LTX-Video",
        model_type="ltx-video",
        num_frames=24,
        fps=24,
        height=512,
        width=512,
        dtype=torch.bfloat16,
        enable_vae_slicing=True,
        enable_cpu_offload=True,
    )
    
    with VideoPipeline(config) as pipeline:
        print("Generating video... (this may take a while)")
        result = pipeline.generate(
            prompt="a car driving through a futuristic city at night, neon lights, cinematic",
            num_frames=24,
            num_inference_steps=50,
            guidance_scale=6.0,
            seed=42
        )
        
        frames = result["frames"]
        fps = result["fps"]
        
        # Save frames as images
        for i, frame in enumerate(frames):
            frame.save(f"output_video_frame_{i:04d}.png")
        
        print(f"✓ Generated {len(frames)} frames at {fps} fps")
        print("  Frames saved as: output_video_frame_*.png")
        
        # Optionally save as video (requires imageio or opencv)
        try:
            save_video(frames, "output_video.mp4", fps)
            print("  Video saved as: output_video.mp4")
        except ImportError:
            print("  Install imageio to save as video file")


def example_image_to_video():
    """Generate video from image."""
    print("\n" + "=" * 60)
    print("Example 2: Image-to-Video Generation")
    print("=" * 60)
    
    # Create input image
    print("Creating input image...")
    input_image = Image.new("RGB", (512, 512), color=(50, 100, 150))
    draw = ImageDraw.Draw(input_image)
    draw.ellipse([150, 150, 350, 350], fill=(200, 150, 100))
    input_image.save("input_video_image.png")
    
    config = VideoConfig(
        model_id="stabilityai/stable-video-diffusion-img2vid-xt",
        model_type="svd-xt",
        num_frames=25,
        fps=7,
        dtype=torch.float16,
    )
    
    with VideoPipeline(config) as pipeline:
        print("Generating video from image...")
        result = pipeline.generate_from_image(
            image=input_image,
            num_frames=25,
            motion_bucket_id=127,  # Motion intensity (1-255)
            noise_aug_strength=0.02,
            seed=42
        )
        
        frames = result["frames"]
        
        for i, frame in enumerate(frames):
            frame.save(f"output_i2v_frame_{i:04d}.png")
        
        print(f"✓ Generated {len(frames)} frames")


def example_frame_generation():
    """Generate frames using frame-by-frame approach."""
    print("\n" + "=" * 60)
    print("Example 3: Frame-by-Frame Generation")
    print("=" * 60)
    
    # Use image pipeline for frame generation
    img_config = PipelineConfig(
        model_id="stabilityai/sdxl-turbo",
        model_type="sdxl",
        dtype=torch.float16,
        default_steps=4
    )
    
    img_pipeline = ImagePipeline(img_config)
    img_pipeline.load()
    
    # Configure frame generator
    frame_config = FrameGenerationConfig(
        num_frames=12,
        overlap_frames=2,
        mode="overlap",
        guidance_scale=7.5,
        temporal_weight=0.8
    )
    
    generator = FrameGenerator(img_pipeline, frame_config)
    
    print("Generating frame sequence...")
    frames = generator.generate_sequence(
        prompt="an animated character walking through a forest",
        num_frames=12
    )
    
    for i, frame in enumerate(frames):
        frame.save(f"output_frame_{i:04d}.png")
    
    print(f"✓ Generated {len(frames)} frames")
    
    img_pipeline.unload()


def example_keyframe_animation():
    """Generate animation from keyframes."""
    print("\n" + "=" * 60)
    print("Example 4: Keyframe Animation")
    print("=" * 60)
    
    # Create keyframes
    keyframes = []
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # Red, Green, Blue
    
    for color in colors:
        frame = Image.new("RGB", (256, 256), color=color)
        keyframes.append(frame)
    
    print(f"Created {len(keyframes)} keyframes")
    
    # Configure frame generator
    img_config = PipelineConfig(
        model_id="stabilityai/sdxl-turbo",
        dtype=torch.float16,
        default_steps=4
    )
    
    img_pipeline = ImagePipeline(img_config)
    img_pipeline.load()
    
    frame_config = FrameGenerationConfig(
        num_frames=15,
        mode="keyframe"
    )
    
    generator = FrameGenerator(img_pipeline, frame_config)
    
    print("Interpolating between keyframes...")
    frames = generator.generate_sequence(
        prompt="smooth color transition",
        num_frames=15,
        keyframes=keyframes
    )
    
    for i, frame in enumerate(frames):
        frame.save(f"output_keyframe_{i:04d}.png")
    
    print(f"✓ Generated {len(frames)} frames from {len(keyframes)} keyframes")
    
    img_pipeline.unload()


def example_temporal_consistency():
    """Apply temporal consistency to frames."""
    print("\n" + "=" * 60)
    print("Example 5: Temporal Consistency")
    print("=" * 60)
    
    # Generate some test frames with variation
    frames = []
    for i in range(10):
        # Create frames with slight variations
        color_val = int(100 + i * 15)
        frame = Image.new("RGB", (256, 256), color=(color_val, color_val, 200))
        frames.append(frame)
    
    print(f"Created {len(frames)} test frames")
    
    # Apply temporal smoothing
    processor = TemporalConsistencyProcessor(
        consistency_weight=0.8,
        use_optical_flow=False  # Set True if OpenCV available
    )
    
    print("Applying temporal smoothing...")
    smoothed = processor.process_sequence(frames, mode="smooth", strength=0.5)
    
    for i, frame in enumerate(smoothed):
        frame.save(f"output_smoothed_{i:04d}.png")
    
    print(f"✓ Smoothed {len(smoothed)} frames")
    
    # Apply stabilization
    print("\nApplying motion stabilization...")
    stabilized = processor.process_sequence(frames, mode="stabilize", strength=0.3)
    
    for i, frame in enumerate(stabilized):
        frame.save(f"output_stabilized_{i:04d}.png")
    
    print(f"✓ Stabilized {len(stabilized)} frames")


def example_frame_interpolation():
    """Interpolate between frames."""
    print("\n" + "=" * 60)
    print("Example 6: Frame Interpolation")
    print("=" * 60)
    
    # Create two key frames
    frame1 = Image.new("RGB", (256, 256), color=(255, 0, 0))  # Red
    frame2 = Image.new("RGB", (256, 256), color=(0, 0, 255))  # Blue
    
    frame1.save("interp_start.png")
    frame2.save("interp_end.png")
    
    processor = TemporalConsistencyProcessor()
    
    print("Interpolating frames...")
    interpolated = processor.interpolate_frames(
        frame1=frame1,
        frame2=frame2,
        num_interpolated=5
    )
    
    # Save sequence: start, interp frames, end
    sequence = [frame1] + interpolated + [frame2]
    for i, frame in enumerate(sequence):
        frame.save(f"output_interp_{i:04d}.png")
    
    print(f"✓ Generated {len(interpolated)} interpolated frames")


def example_low_memory_video():
    """Generate video with memory optimizations."""
    print("\n" + "=" * 60)
    print("Example 7: Low Memory Video Generation")
    print("=" * 60)
    
    print("Configuring for 8GB VRAM...")
    
    config = VideoConfig(
        model_id="Lightricks/LTX-Video",
        num_frames=16,
        fps=24,
        height=384,
        width=384,  # Lower resolution
        dtype=torch.float16,
        # Memory optimizations
        enable_vae_slicing=True,
        enable_vae_tiling=True,
        enable_cpu_offload=True,
        num_inference_steps=30,  # Fewer steps
    )
    
    print(f"Configuration:")
    print(f"  - Resolution: {config.width}x{config.height}")
    print(f"  - Frames: {config.num_frames}")
    print(f"  - VAE Slicing: {config.enable_vae_slicing}")
    print(f"  - VAE Tiling: {config.enable_vae_tiling}")
    print(f"  - CPU Offload: {config.enable_cpu_offload}")
    
    with VideoPipeline(config) as pipeline:
        result = pipeline.generate(
            prompt="a peaceful nature scene with flowing water",
            seed=1
        )
        
        frames = result["frames"]
        for i, frame in enumerate(frames):
            frame.save(f"output_lowmem_{i:04d}.png")
        
        print(f"✓ Generated {len(frames)} frames with low memory settings")


def save_video(frames, output_path, fps=24):
    """Save frames as video file."""
    try:
        import imageio
        writer = imageio.get_writer(output_path, fps=fps)
        for frame in frames:
            writer.append_data(np.array(frame))
        writer.close()
    except ImportError:
        try:
            import cv2
            height, width = frames[0].size[1], frames[0].size[0]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            for frame in frames:
                cv_frame = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
                writer.write(cv_frame)
            writer.release()
        except ImportError:
            raise ImportError("Install imageio or opencv-python to save videos")


def main():
    parser = argparse.ArgumentParser(description="Nexus Video Generation Examples")
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
        1: example_text_to_video,
        2: example_image_to_video,
        3: example_frame_generation,
        4: example_keyframe_animation,
        5: example_temporal_consistency,
        6: example_frame_interpolation,
        7: example_low_memory_video,
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
        print(__doc__)
        print("\n" + "=" * 60)
        print("Run with --all to see all examples")
        print("Run with --example N to run specific example")
        print("=" * 60)


if __name__ == "__main__":
    main()
