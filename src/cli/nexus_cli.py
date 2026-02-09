#!/usr/bin/env python3
"""
nexus CLI

Main entry point for the Nexus CLI with integrated polish features.
"""

import sys
import os
import argparse
from pathlib import Path
from typing import Optional, List

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def main(args: Optional[List[str]] = None) -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        prog='nexus',
        description='Nexus - Universal Model Training and Inference Platform'
    )
    
    parser.add_argument('--version', '-v', action='version', version='%(prog)s 6.1.0')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    parser.add_argument('--install-completion', action='store_true', help='Install shell completion')
    parser.add_argument('--shell', choices=['bash', 'zsh', 'fish', 'auto'], default='auto')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Diffusion command
    diffusion_parser = subparsers.add_parser('diffusion', help='Image generation commands')
    diffusion_parser.add_argument('action', choices=['generate', 'img2img', 'inpaint', 'list-presets'])
    diffusion_parser.add_argument('--model', '-m', default='sdxl-base', help='Model preset or ID')
    diffusion_parser.add_argument('--prompt', '-p', required=True, help='Text prompt')
    diffusion_parser.add_argument('--negative-prompt', '-n', default='', help='Negative prompt')
    diffusion_parser.add_argument('--output', '-o', default='output.png', help='Output path')
    diffusion_parser.add_argument('--width', type=int, default=1024, help='Image width')
    diffusion_parser.add_argument('--height', type=int, default=1024, help='Image height')
    diffusion_parser.add_argument('--steps', type=int, default=30, help='Inference steps')
    diffusion_parser.add_argument('--seed', type=int, default=None, help='Random seed')
    diffusion_parser.add_argument('--input', '-i', help='Input image for img2img/inpaint')
    diffusion_parser.add_argument('--mask', help='Mask image for inpainting')
    diffusion_parser.add_argument('--strength', type=float, default=0.75, help='Img2img strength')
    diffusion_parser.add_argument('--quantize', choices=['fp8', 'int8', 'int4'], help='Quantization')
    
    # Video command
    video_parser = subparsers.add_parser('video', help='Video generation commands')
    video_parser.add_argument('action', choices=['generate', 'img2vid', 'list-models'])
    video_parser.add_argument('--model', '-m', default='ltx-video', help='Model ID')
    video_parser.add_argument('--prompt', '-p', required=True, help='Text prompt')
    video_parser.add_argument('--output', '-o', default='output.mp4', help='Output path')
    video_parser.add_argument('--frames', type=int, default=24, help='Number of frames')
    video_parser.add_argument('--fps', type=int, default=24, help='Frames per second')
    video_parser.add_argument('--width', type=int, default=512, help='Video width')
    video_parser.add_argument('--height', type=int, default=512, help='Video height')
    video_parser.add_argument('--steps', type=int, default=50, help='Inference steps')
    video_parser.add_argument('--seed', type=int, default=None, help='Random seed')
    video_parser.add_argument('--input', '-i', help='Input image for img2vid')
    
    # GGUF command
    gguf_parser = subparsers.add_parser('gguf', help='GGUF model commands')
    gguf_parser.add_argument('action', choices=['run', 'convert', 'info', 'validate'])
    gguf_parser.add_argument('--model', '-m', required=True, help='Path to GGUF model')
    gguf_parser.add_argument('--prompt', '-p', help='Prompt for generation')
    gguf_parser.add_argument('--ctx', type=int, default=4096, help='Context size')
    gguf_parser.add_argument('--gpu-layers', type=int, default=-1, help='GPU layers (-1=all)')
    gguf_parser.add_argument('--threads', type=int, default=-1, help='CPU threads')
    gguf_parser.add_argument('--temperature', '-t', type=float, default=0.7, help='Temperature')
    gguf_parser.add_argument('--max-tokens', type=int, default=256, help='Max tokens')
    gguf_parser.add_argument('--source', '-s', help='Source model for conversion')
    gguf_parser.add_argument('--quantization', '-q', default='Q4_K_M', 
                            choices=['Q2_K', 'Q3_K_M', 'Q4_K_M', 'Q5_K_M', 'Q6_K', 'Q8_0'],
                            help='Quantization type')
    gguf_parser.add_argument('--chat', action='store_true', help='Chat mode')
    
    # Registry command
    registry_parser = subparsers.add_parser('registry', help='Model registry commands')
    registry_parser.add_argument('action', choices=['list', 'info', 'detect'])
    registry_parser.add_argument('--type', help='Filter by model type')
    registry_parser.add_argument('--tag', help='Filter by tag')
    registry_parser.add_argument('--model', '-m', help='Model key for info')
    registry_parser.add_argument('--architecture', '-a', help='Architecture for detection')
    
    args = parser.parse_args(args)
    
    if args.install_completion:
        from src.cli.completion import install_completion
        shell = None if args.shell == 'auto' else args.shell
        success = install_completion(shell, prog_name='nexus')
        return 0 if success else 1
    
    if args.command == 'diffusion':
        return handle_diffusion_command(args)
    elif args.command == 'video':
        return handle_video_command(args)
    elif args.command == 'gguf':
        return handle_gguf_command(args)
    elif args.command == 'registry':
        return handle_registry_command(args)
    
    print("Nexus CLI v6.1.0 - Use --help for available commands")
    print("\nCommands:")
    print("  diffusion    Image generation (SD, SDXL, FLUX)")
    print("  video        Video generation (LTX, SVD)")
    print("  gguf         GGUF model inference and conversion")
    print("  registry     Model registry management")
    return 0


def handle_diffusion_command(args):
    """Handle diffusion subcommand."""
    from src.models.diffusion import DiffusionPipelineLoader
    
    if args.action == 'list-presets':
        presets = DiffusionPipelineLoader.list_presets()
        print("Available presets:")
        for preset in presets:
            info = DiffusionPipelineLoader.get_preset_info(preset)
            print(f"  {preset}: {info.get('model_id', 'N/A')}")
        return 0
    
    print(f"Loading model: {args.model}")
    loader = DiffusionPipelineLoader()
    
    if args.quantize:
        pipeline = loader.load_quantized(args.model, quantization=args.quantize)
    else:
        pipeline = loader.load(args.model)
    
    if args.action == 'generate':
        result = pipeline.generate(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            height=args.height,
            width=args.width,
            num_inference_steps=args.steps,
            seed=args.seed
        )
        result['images'][0].save(args.output)
        print(f"✓ Generated: {args.output}")
    
    elif args.action == 'img2img':
        if not args.input:
            print("Error: --input required for img2img")
            return 1
        from PIL import Image
        input_image = Image.open(args.input)
        result = pipeline.generate_variations(
            image=input_image,
            prompt=args.prompt,
            strength=args.strength,
            seed=args.seed
        )
        result['images'][0].save(args.output)
        print(f"✓ Generated: {args.output}")
    
    elif args.action == 'inpaint':
        if not args.input or not args.mask:
            print("Error: --input and --mask required for inpainting")
            return 1
        from PIL import Image
        image = Image.open(args.input)
        mask = Image.open(args.mask)
        result = pipeline.inpaint(
            image=image,
            mask=mask,
            prompt=args.prompt,
            seed=args.seed
        )
        result['images'][0].save(args.output)
        print(f"✓ Generated: {args.output}")
    
    pipeline.unload()
    return 0


def handle_video_command(args):
    """Handle video subcommand."""
    from src.models.video import VideoPipeline, VideoConfig
    
    if args.action == 'list-models':
        models = VideoPipeline.MODEL_TYPE_MAP.keys()
        print("Supported video models:")
        for model in models:
            print(f"  {model}")
        return 0
    
    print(f"Loading video model: {args.model}")
    config = VideoConfig(
        model_id=args.model,
        num_frames=args.frames,
        fps=args.fps,
        height=args.height,
        width=args.width
    )
    
    with VideoPipeline(config) as pipeline:
        if args.action == 'generate':
            result = pipeline.generate(
                prompt=args.prompt,
                num_frames=args.frames,
                num_inference_steps=args.steps,
                seed=args.seed
            )
        elif args.action == 'img2vid':
            if not args.input:
                print("Error: --input required for img2vid")
                return 1
            from PIL import Image
            input_image = Image.open(args.input)
            result = pipeline.generate_from_image(
                image=input_image,
                prompt=args.prompt,
                num_frames=args.frames,
                seed=args.seed
            )
        
        print(f"Generated {len(result['frames'])} frames")
        for i, frame in enumerate(result['frames']):
            frame.save(f"frame_{i:04d}.png")
        print(f"✓ Frames saved to frame_*.png")
    
    return 0


def handle_gguf_command(args):
    """Handle GGUF subcommand."""
    from src.models.gguf import GGUfLoader, GGUFConfig, GGUFConverter
    
    if args.action == 'info':
        converter = GGUFConverter()
        metadata = converter.get_gguf_metadata(args.model)
        print(f"Model: {args.model}")
        print("Metadata:")
        for key, value in metadata.items():
            print(f"  {key}: {value}")
        return 0
    
    elif args.action == 'validate':
        converter = GGUFConverter()
        report = converter.validate_gguf(args.model)
        print(f"Valid: {report['valid']}")
        if report['error']:
            print(f"Error: {report['error']}")
        print(f"Tensors: {report['tensor_count']}")
        print(f"Size: {report['file_size']} bytes")
        return 0
    
    elif args.action == 'convert':
        if not args.source:
            print("Error: --source required for conversion")
            return 1
        converter = GGUFConverter()
        output = converter.pytorch_to_gguf(
            args.source,
            args.model,
            quantization=args.quantization
        )
        print(f"✓ Would create: {output}")
        return 0
    
    elif args.action == 'run':
        if not args.prompt:
            print("Error: --prompt required")
            return 1
        
        config = GGUFConfig(
            model_path=args.model,
            n_ctx=args.ctx,
            n_gpu_layers=args.gpu_layers,
            n_threads=args.threads,
            temperature=args.temperature
        )
        
        with GGUfLoader(config) as model:
            if args.chat:
                messages = [{"role": "user", "content": args.prompt}]
                result = model.chat(messages, max_tokens=args.max_tokens)
                print(result['content'])
            else:
                result = model.generate(args.prompt, max_tokens=args.max_tokens)
                print(result['text'])
    
    return 0


def handle_registry_command(args):
    """Handle registry subcommand."""
    from src.core.towers import registry
    
    if args.action == 'list':
        models = registry.TEACHER_REGISTRY
        print("Registered models:")
        for key, info in models.items():
            if args.type and info.get('type') != args.type:
                continue
            if args.tag and args.tag not in info.get('tags', []):
                continue
            print(f"  {key}: {info.get('model', 'N/A')} ({info.get('type', 'unknown')})")
        return 0
    
    elif args.action == 'info':
        if not args.model:
            print("Error: --model required")
            return 1
        info = registry.get_model_info(args.model)
        print(f"Model: {args.model}")
        for key, value in info.items():
            print(f"  {key}: {value}")
        return 0
    
    elif args.action == 'detect':
        if not args.architecture:
            print("Error: --architecture required")
            return 1
        detected = registry.detect_architecture(args.architecture)
        print(f"Detected type for '{args.architecture}': {detected}")
        return 0

if __name__ == '__main__':
    sys.exit(main())
