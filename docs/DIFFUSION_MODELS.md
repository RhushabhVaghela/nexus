# Diffusion Models in Nexus

Nexus provides comprehensive support for image generation through the Diffusers library, with unified interfaces for Stable Diffusion, SDXL, FLUX, Z-Image, Z-Image-Turbo, and HunyuanImage models.

## Supported Models

### Stable Diffusion Family

- **SD 1.5/2.1**: `stabilityai/stable-diffusion-2-1`
- **SDXL**: `stabilityai/stable-diffusion-xl-base-1.0`
- **SDXL Turbo**: `stabilityai/sdxl-turbo` (4 steps)
- **SD3**: `stabilityai/stable-diffusion-3-medium`
- **SD3.5**: `stabilityai/stable-diffusion-3.5-medium`, `stabilityai/stable-diffusion-3.5-large`

### FLUX Family

- **FLUX.1 Dev**: `black-forest-labs/FLUX.1-dev` (50 steps, guidance 3.5)
- **FLUX.1 Schnell**: `black-forest-labs/FLUX.1-schnell` (4 steps, no CFG)
- **FLUX.1 Fill**: `black-forest-labs/FLUX.1-fill-dev` (inpainting)

### Z-Image Family

- **Z-Image**: `stabilityai/z-image`
- **Z-Image Turbo**: `stabilityai/z-image-turbo` (4 steps)

### Hunyuan DiT

- **HunyuanDiT**: `Tencent-Hunyuan/HunyuanDiT-v1.2`

## Quick Start

### Basic Image Generation

```python
from nexus.models.diffusion import ImagePipeline, PipelineConfig

# Configure pipeline
config = PipelineConfig(
    model_id="stabilityai/stable-diffusion-xl-base-1.0",
    model_type="sdxl",
    device="auto",
    dtype=torch.float16
)

# Generate image
with ImagePipeline(config) as pipeline:
    result = pipeline.generate(
        prompt="a beautiful sunset over mountains, highly detailed, 8k",
        negative_prompt="blur, low quality",
        height=1024,
        width=1024,
        num_inference_steps=30,
        guidance_scale=7.5,
        seed=42
    )
    
    image = result["images"][0]
    image.save("sunset.png")
```

### Using Presets

```python
from nexus.models.diffusion import DiffusionPipelineLoader

# Quick load with preset
loader = DiffusionPipelineLoader()

# Available presets: "sdxl-base", "sdxl-turbo", "flux-dev", "flux-schnell", etc.
pipeline = loader.load("flux-schnell")

result = pipeline.generate("a futuristic city")
image = result["images"][0]
```

### Image-to-Image

```python
from PIL import Image

# Load base image
input_image = Image.open("input.png")

# Generate variation
result = pipeline.generate_variations(
    image=input_image,
    prompt="same scene, different lighting",
    strength=0.75,  # How much to change (0-1)
    num_images=4
)

for i, img in enumerate(result["images"]):
    img.save(f"variation_{i}.png")
```

### Inpainting

```python
from PIL import Image

# Load image and mask
image = Image.open("photo.png")
mask = Image.open("mask.png")  # White = inpaint, Black = keep

# Inpaint masked region
result = pipeline.inpaint(
    image=image,
    mask=mask,
    prompt="beautiful flowers in the foreground"
)

result["images"][0].save("inpainted.png")
```

## Configuration Options

### Memory Optimization

```python
config = PipelineConfig(
    model_id="stabilityai/stable-diffusion-xl-base-1.0",
    
    # VAE optimization
    enable_vae_slicing=True,      # For large images (>1024px)
    enable_vae_tiling=True,       # For very large images (>2048px)
    
    # CPU offloading
    enable_cpu_offload=True,      # For low VRAM GPUs
    
    # Data type
    dtype=torch.float16,          # or torch.bfloat16
)
```

### Quantization

```python
from nexus.models.diffusion import DiffusionPipelineLoader

loader = DiffusionPipelineLoader()

# Load with quantization
pipeline = loader.load_quantized(
    "sdxl-base",
    quantization="fp8"  # "fp8", "int8", or "int4"
)
```

## Model-Specific Parameters

### FLUX Models

```python
# FLUX Dev
config = PipelineConfig(
    model_id="black-forest-labs/FLUX.1-dev",
    default_steps=50,
    default_guidance_scale=3.5,  # FLUX uses lower guidance
)

# FLUX Schnell (faster, lower quality)
config = PipelineConfig(
    model_id="black-forest-labs/FLUX.1-schnell",
    default_steps=4,             # Much faster
    default_guidance_scale=0.0,  # No CFG needed
)
```

### SD3 Models

```python
config = PipelineConfig(
    model_id="stabilityai/stable-diffusion-3-medium",
    default_steps=28,
    default_guidance_scale=5.0,  # SD3 uses lower guidance
)
```

## Integration with Nexus Training

```python
from nexus.models.diffusion import DiffusionAdapter
from nexus.models.diffusion import ImagePipeline, PipelineConfig

# Create adapter for knowledge distillation
adapter = DiffusionAdapter(
    teacher_dim=2048,
    student_dim=1024,
    extract_features_from="unet"
)

# Attach pipeline
config = PipelineConfig(model_id="stabilityai/sdxl-base")
pipeline = ImagePipeline(config)
pipeline.load()

adapter.attach_pipeline(pipeline)

# Extract features for training
features = adapter.extract_features(
    prompt="a beautiful landscape",
    num_inference_steps=1  # Fast extraction
)
```

## Performance Tips

### For Low VRAM (8GB)

```python
config = PipelineConfig(
    model_id="stabilityai/sdxl-base",
    enable_vae_slicing=True,
    enable_cpu_offload=True,
    dtype=torch.float16,
    default_height=768,
    default_width=768,
)
```

### For Fast Generation

```python
# Use Turbo models or reduce steps
config = PipelineConfig(
    model_id="stabilityai/sdxl-turbo",
    default_steps=4,  # Turbo uses 4 steps
)
```

### For High Quality

```python
config = PipelineConfig(
    model_id="black-forest-labs/FLUX.1-dev",
    default_steps=50,
    default_guidance_scale=3.5,
    dtype=torch.bfloat16,
)
```

## Troubleshooting

### Out of Memory

- Enable `enable_vae_slicing=True`
- Enable `enable_cpu_offload=True`
- Reduce image resolution
- Use lower precision (`torch.float16`)

### Slow Generation

- Use Turbo models (4 steps vs 30-50)
- Enable VAE slicing for large images
- Use GPU if available

### Poor Quality

- Increase `num_inference_steps`
- Adjust `guidance_scale` (7.5 for SD, 3.5 for FLUX)
- Use higher precision (`torch.bfloat16`)

## API Reference

### ImagePipeline

- `load()`: Load the pipeline
- `generate()`: Generate images from text
- `generate_variations()`: Generate image variations
- `inpaint()`: Inpaint masked regions
- `unload()`: Free memory

### PipelineConfig

- `model_id`: HuggingFace model ID
- `model_type`: Auto-detected or specified ("sd", "sdxl", "sd3", "flux", etc.)
- `device`: "auto", "cuda", "mps", or "cpu"
- `dtype`: torch.float16, torch.bfloat16, or torch.float32
- `enable_vae_slicing`: Enable for large images
- `enable_vae_tiling`: Enable for very large images
- `enable_cpu_offload`: Enable for low VRAM

### DiffusionPipelineLoader

- `load(model_id_or_preset)`: Load with preset or model ID
- `load_quantized()`: Load with quantization
- `list_presets()`: List available presets
- `get_preset_info()`: Get preset configuration
