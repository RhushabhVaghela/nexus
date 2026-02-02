# Video Generation in Nexus

Nexus provides unified video generation capabilities through the Diffusers library, supporting LTX-Video, Stable Video Diffusion (SVD), CogVideoX, and HunyuanVideo models.

## Supported Models

### LTX-Video (Lightricks)

- **Model**: `Lightricks/LTX-Video`
- **Strength**: High-quality video generation
- **Best for**: General video generation, text-to-video

### Stable Video Diffusion (SVD)

- **Models**:
  - `stabilityai/stable-video-diffusion-img2vid`
  - `stabilityai/stable-video-diffusion-img2vid-xt`
  - `stabilityai/stable-video-diffusion-img2vid-xt-1-1`
- **Strength**: Image-to-video conversion
- **Best for**: Animating still images

### CogVideoX (THUDM)

- **Models**:
  - `THUDM/CogVideoX-2b`
  - `THUDM/CogVideoX-5b`
  - `THUDM/CogVideoX-5b-I2V` (image-to-video)
- **Strength**: Efficient video generation
- **Best for**: Resource-constrained environments

### HunyuanVideo

- **Model**: `Tencent-Hunyuan/HunyuanVideo`
- **Strength**: High-resolution video
- **Best for**: Production-quality videos

## Quick Start

### Text-to-Video Generation

```python
from nexus.models.video import VideoPipeline, VideoConfig

# Configure pipeline
config = VideoConfig(
    model_id="Lightricks/LTX-Video",
    num_frames=49,
    fps=24,
    height=512,
    width=512,
    dtype=torch.bfloat16
)

# Generate video
with VideoPipeline(config) as pipeline:
    result = pipeline.generate(
        prompt="a car driving through a futuristic city, cinematic lighting",
        num_frames=24,
        num_inference_steps=50,
        guidance_scale=6.0,
        seed=42
    )
    
    frames = result["frames"]
    fps = result["fps"]
    
    # Save as video (requires additional library)
    save_video_frames(frames, "output.mp4", fps=fps)
```

### Image-to-Video

```python
from PIL import Image

# Load input image
input_image = Image.open("input.png")

# Configure for SVD
config = VideoConfig(
    model_id="stabilityai/stable-video-diffusion-img2vid-xt",
    model_type="svd",
    num_frames=25,
    fps=7
)

with VideoPipeline(config) as pipeline:
    result = pipeline.generate_from_image(
        image=input_image,
        num_frames=25,
        motion_bucket_id=127,  # Motion intensity (1-255)
        noise_aug_strength=0.02
    )
    
    frames = result["frames"]
```

### Frame-by-Frame Generation

```python
from nexus.models.video import FrameGenerator
from nexus.models.video.frame_generator import FrameGenerationConfig

# Configure frame generator
config = FrameGenerationConfig(
    num_frames=24,
    overlap_frames=4,        # Overlapping for consistency
    mode="overlap",          # 'overlap', 'keyframe', or 'autoregressive'
    temporal_weight=0.8
)

# Use with image pipeline for frame generation
from nexus.models.diffusion import ImagePipeline, PipelineConfig

img_config = PipelineConfig(model_id="stabilityai/sdxl-base")
img_pipeline = ImagePipeline(img_config)
img_pipeline.load()

generator = FrameGenerator(img_pipeline, config)

# Generate frame sequence
frames = generator.generate_sequence(
    prompt="an animated character walking",
    num_frames=24
)
```

## Temporal Consistency

### Applying Smoothing

```python
from nexus.models.video.temporal_consistency import TemporalConsistencyProcessor

# Create processor
processor = TemporalConsistencyProcessor(
    consistency_weight=0.8,
    flow_weight=0.5,
    use_optical_flow=True
)

# Process frames for smoothness
frames = [...]  # Your generated frames
smoothed_frames = processor.process_sequence(
    frames,
    mode="smooth",  # 'smooth', 'stabilize', or 'flow'
    strength=0.5
)
```

### Frame Interpolation

```python
# Generate intermediate frames
interpolated = processor.interpolate_frames(
    frame1=frames[0],
    frame2=frames[1],
    num_interpolated=3
)
```

## Configuration Options

### Memory Optimization

```python
config = VideoConfig(
    model_id="Lightricks/LTX-Video",
    
    # VAE optimization
    enable_vae_slicing=True,    # Essential for video
    enable_vae_tiling=True,     # For high resolution
    enable_cpu_offload=True,    # For low VRAM
    
    # Reduce memory
    dtype=torch.float16,
    height=512,
    width=512,
)
```

### Quality Settings

```python
config = VideoConfig(
    model_id="Lightricks/LTX-Video",
    
    # Higher quality
    num_inference_steps=50,
    guidance_scale=6.0,
    dtype=torch.bfloat16,
    height=1024,
    width=1024,
)
```

## Frame Generation Modes

### Overlap Mode

```python
config = FrameGenerationConfig(
    mode="overlap",
    num_frames=24,
    overlap_frames=4  # 4 overlapping frames between windows
)
```

Generates frames in overlapping windows for temporal consistency.

### Keyframe Mode

```python
config = FrameGenerationConfig(
    mode="keyframe",
    num_frames=24
)

# Provide keyframes
keyframes = [frame1, frame4, frame7, frame10]
frames = generator.generate_sequence(
    prompt="animation",
    num_frames=24,
    keyframes=keyframes
)
```

Generates keyframes and interpolates between them.

### Autoregressive Mode

```python
config = FrameGenerationConfig(
    mode="autoregressive",
    num_frames=24,
    overlap_frames=2
)
```

Uses previous frames as context for next frame generation.

## Performance Tips

### For Low VRAM (8-12GB)

```python
config = VideoConfig(
    model_id="Lightricks/LTX-Video",
    enable_vae_slicing=True,
    enable_vae_tiling=True,
    enable_cpu_offload=True,
    dtype=torch.float16,
    num_frames=16,
    height=512,
    width=512,
)
```

### For Fast Generation

```python
# Use CogVideoX (faster)
config = VideoConfig(
    model_id="THUDM/CogVideoX-2b",
    num_inference_steps=25,
    num_frames=16,
)
```

### For Best Quality

```python
# Use HunyuanVideo or LTX with high settings
config = VideoConfig(
    model_id="Lightricks/LTX-Video",
    num_inference_steps=50,
    guidance_scale=6.0,
    dtype=torch.bfloat16,
    height=1024,
    width=1024,
    num_frames=49,
)
```

## Saving Videos

### Using OpenCV

```python
import cv2
import numpy as np

def save_video_frames(frames, output_path, fps=24):
    """Save PIL frames as MP4 video."""
    height, width = frames[0].size[1], frames[0].size[0]
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for frame in frames:
        # Convert PIL to OpenCV format
        cv_frame = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
        writer.write(cv_frame)
    
    writer.release()
```

### Using imageio

```python
import imageio

def save_video_frames(frames, output_path, fps=24):
    """Save PIL frames as MP4 video using imageio."""
    writer = imageio.get_writer(output_path, fps=fps)
    
    for frame in frames:
        writer.append_data(np.array(frame))
    
    writer.close()
```

## Troubleshooting

### Out of Memory

- Enable all memory optimizations (`enable_vae_slicing`, `enable_vae_tiling`, `enable_cpu_offload`)
- Reduce resolution (try 512x512 instead of 1024x1024)
- Reduce number of frames
- Use float16 instead of bfloat16

### Flickering/Jittery Video

- Increase `overlap_frames` in FrameGenerator
- Apply TemporalConsistencyProcessor
- Use higher `num_inference_steps`
- Try keyframe mode with more keyframes

### Slow Generation

- Reduce `num_inference_steps` (25-30 is often sufficient)
- Use smaller resolution
- Use CogVideoX instead of larger models
- Reduce number of frames

## API Reference

### VideoPipeline

- `load()`: Load the pipeline
- `generate()`: Generate video from text (text-to-video models)
- `generate_from_image()`: Generate video from image (img2vid models)
- `interpolate_frames()`: Generate frames between two images
- `unload()`: Free memory

### VideoConfig

- `model_id`: HuggingFace model ID
- `model_type`: Model type ("ltx-video", "svd", "cogvideo", "hunyuan-video")
- `num_frames`: Number of frames to generate
- `fps`: Frames per second
- `height`, `width`: Video dimensions
- `num_inference_steps`: Denoising steps
- `guidance_scale`: CFG scale
- `enable_vae_slicing`: VAE slicing for memory
- `enable_vae_tiling`: VAE tiling for large videos
- `enable_cpu_offload`: CPU offloading for low VRAM

### FrameGenerator

- `generate_sequence()`: Generate sequence of frames
- `clear_buffer()`: Clear frame buffer

### TemporalConsistencyProcessor

- `process_sequence()`: Apply temporal consistency
- `interpolate_frames()`: Interpolate between frames
- `compute_temporal_loss()`: Compute consistency loss
