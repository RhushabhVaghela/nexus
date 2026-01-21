# Manus Model - Universal Multimodal Training Pipeline

A comprehensive pipeline for training and extending language models with multimodal capabilities including vision, audio, video understanding, and generation.

---

## Quick Start

```bash
# Activate environment
conda activate manus

# Run tests (109 tests, ~90s)
pytest tests/unit/ tests/integration/ tests/e2e/ -v

# Train with dry-run (preview stages)
./run_universal_pipeline.sh \
    --base-model=/mnt/e/data/models/Qwen2.5-0.5B \
    --enable-cot \
    --dry-run

# Real training
./run_universal_pipeline.sh \
    --base-model=/mnt/e/data/models/Qwen2.5-0.5B \
    --enable-all-text \
    --sample-size=1000
```

---

## Features

### 12 Trainable Capabilities

| Capability | Description | Required Modalities |
|------------|-------------|---------------------|
| **cot** | Chain-of-Thought reasoning | text |
| **reasoning** | Multi-level mathematical reasoning | text |
| **thinking** | Extended thinking/reflection | text |
| **tools** | Function/tool calling | text |
| **streaming** | Token streaming output | text |
| **omni** | Convert text → full Omni model | text |
| **podcast** | NotebookLM-style podcast generation | text, audio |
| **vision-qa** | Image understanding | text, vision |
| **video-understanding** | Video comprehension | text, vision, video |
| **tri-streaming** | Real-time multimodal streaming | ALL |
| **image-generation** | Text-to-image (SD3 projector) | text, vision_output |
| **video-generation** | Text-to-video (SVD projector) | text, video_output |

### Training Safety Features

- **Pause/Resume** - Signal-based training control
- **Automatic Cooldown** - Every 500 steps
- **GPU Temperature Protection** - Auto-pause at 83°C
- **Emergency Checkpoints** - SIGUSR2 for instant save

---

## Project Structure

```
manus_model/
├── run_universal_pipeline.sh    # Main orchestrator
├── src/
│   ├── detect_modalities.py     # Model capability detection
│   ├── capability_registry.py   # Capability definitions
│   ├── training_controller.py   # Pause/resume/cooldown
│   ├── stages/                  # Training stage scripts
│   │   ├── base.py              # BaseStage class
│   │   ├── stage_cot.py
│   │   ├── stage_reasoning.py
│   │   ├── stage_thinking.py
│   │   ├── stage_tools.py
│   │   ├── stage_streaming.py
│   │   ├── stage_image_gen.py
│   │   └── stage_video_gen.py
│   └── multimodal/              # Encoder/decoder modules
├── configs/
│   └── encoders.yaml            # Encoder/decoder paths
├── tests/                       # 109 tests
│   ├── unit/                    # 48 tests
│   ├── integration/             # 40 tests
│   └── e2e/                     # 21 tests
└── docs/
    ├── TEST_SUITE.md            # Test documentation
    ├── SHELL_SCRIPTS.md         # Script reference
    └── GUIDE.md                 # User guide
```

---

## Usage

### Basic Training

```bash
# Chain-of-Thought only
./run_universal_pipeline.sh --base-model=/path/to/model --enable-cot

# Multiple capabilities
./run_universal_pipeline.sh --base-model=/path/to/model \
    --enable-cot --enable-reasoning --enable-tools

# All text capabilities
./run_universal_pipeline.sh --base-model=/path/to/model --enable-all-text

# Convert to Omni + podcast
./run_universal_pipeline.sh --base-model=/path/to/model \
    --enable-omni --enable-podcast
```

### Training Options

| Option | Default | Description |
|--------|---------|-------------|
| `--base-model` | required | Path to base model |
| `--output-dir` | `/mnt/e/data/models/trained` | Output directory |
| `--sample-size` | 0 (all) | Limit samples per dataset |
| `--batch-size` | 1 | Training batch size |
| `--epochs` | 3 | Number of epochs |
| `--training-method` | sft | Training method (see below) |
| `--dry-run` | false | Preview without training |

### 10 SOTA Training Methods

| Method | Description | Use Case |
|--------|-------------|----------|
| `sft` | Supervised Fine-Tuning | Full weight updates |
| `lora` | Low-Rank Adaptation | Parameter-efficient |
| `qlora` | Quantized LoRA (4-bit) | Low-VRAM training |
| `dora` | Weight-Decomposed LoRA | Improved LoRA (2024) |
| `dpo` | Direct Preference Optimization | Alignment |
| `grpo` | Group Relative Policy (DeepSeek) | Reasoning |
| `orpo` | Odds Ratio Preference | Combined SFT+Preference |
| `ppo` | Proximal Policy (RLHF) | Classic alignment |
| `distillation` | Knowledge Distillation | Learn from teacher |
| `cpt` | Continued Pre-Training | Domain adaptation |

```bash
# Example: Train with QLoRA
./run_pipeline.sh train --training-method=qlora

# Example: DPO alignment
./run_pipeline.sh train --training-method=dpo
```

### Dry-Run Mode

Preview the training pipeline without executing:

```bash
./run_universal_pipeline.sh \
    --base-model=/mnt/e/data/models/Qwen2.5-0.5B \
    --enable-all-text \
    --dry-run
```

Output:

```
╔═══════════════════════════════════════════════════════════════╗
║         MANUS UNIVERSAL CAPABILITY PIPELINE                   ║
╚═══════════════════════════════════════════════════════════════╝

  Base Model:  Qwen2.5-0.5B
  Output:      /mnt/e/data/models/trained
  Mode:        DRY-RUN (no actual training)

[STAGE] Stage 0: Detecting Model Modalities
  Detected: text=✓ vision=false audio_in=false audio_out=false

[STAGE] Stage 2: Training Queue (5 stages)
  1. cot
  2. reasoning
  3. thinking
  4. tools
  5. streaming
```

---

## Pause/Resume Training

### Signal-Based Control

All training integrates with the training controller for runtime control:

```bash
# Find training PID
ps aux | grep stage_cot.py
# Output: user 12345 ... python stage_cot.py

# PAUSE training (toggle)
kill -USR1 12345

# RESUME training (same signal)
kill -USR1 12345

# Emergency CHECKPOINT (saves immediately)
kill -USR2 12345
```

### What Happens When Paused

- Training loop stops at next step boundary
- GPU memory is cleared
- Status shows: `⏸️ Paused at step 1234... (SIGUSR1 to resume)`
- Model state is preserved in memory
- Send SIGUSR1 again to resume

### Automatic Protections

| Feature | Trigger | Action |
|---------|---------|--------|
| **Cooldown** | Every 500 steps | 60s pause, clear cache |
| **Temperature** | GPU > 83°C | Pause until cooled |
| **Checkpoint** | SIGUSR2 signal | Immediate save |

### Monitoring Training

```bash
# Watch logs
tail -f logs/train_cot.log

# Monitor GPU
watch -n 1 nvidia-smi

# Check if paused
grep -i "paused" logs/train_cot.log
```

---

## Testing

### Run All Tests

```bash
# Full test suite (109 tests)
pytest tests/unit/ tests/integration/ tests/e2e/ -v

# With coverage report
pytest tests/ --cov=src --cov-report=html
```

### Test Categories

| Category | Tests | Duration | Description |
|----------|-------|----------|-------------|
| Unit | 48 | ~5s | Fast, isolated module tests |
| Integration | 40 | ~30s | Real model loading tests |
| E2E | 21 | ~60s | Full pipeline tests |

### Run Specific Tests

```bash
# Unit tests only
pytest tests/unit/ -v

# Integration tests only
pytest tests/integration/ -v

# E2E tests only
pytest tests/e2e/ -v

# Skip slow tests
pytest tests/ -v -m "not slow"

# Run specific test
pytest tests/unit/test_detect_modalities.py::TestDetectModalities::test_detect_text_only_model -v
```

---

## Configuration

### Encoder Paths (`configs/encoders.yaml`)

```yaml
encoders:
  vision:
    default: /mnt/e/data/encoders/vision-encoders/siglip2-so400m-patch16-512
  audio_input:
    default: /mnt/e/data/encoders/audio-encoders/whisper-large-v3-turbo
  audio_output:
    default: /mnt/e/data/encoders/audio-encoders/parakeet-tdt-0.6b-v3

decoders:
  vision_output:
    default: stabilityai/stable-diffusion-3.5-large
  video_output:
    default: stabilityai/stable-video-diffusion-img2vid-xt
```

### ⚡ Memory-Efficient Streaming (New)

Train on **limited RAM** with **unlimited dataset size**.

| Feature | Description | Support |
|---------|-------------|---------|
| **Streaming** | Iterate over datasets larger than RAM (500GB+) | All Scripts |
| **Giant Files** | Process single 40GB+ files (videos, logs) via chunking | All Scripts |
| **Auto-Detect** | Automatically switches to streaming for files >1GB | Built-in |

**Usage:**

```bash
# SFT with manual streaming (optional, auto-detects anyway)
python src/10_sft_training.py --use_streaming

# PPO with massive prompts file
python src/ppo_training.py --prompts_data /path/to/1tb_dataset
```

### 🔨 Development Tools

- `tests/unit_streaming/` - Validation for streaming logic
- `src/detect_modalities.py` - Probe model capabilities

### Environment Variables

```bash
# Optional overrides
export MANUS_OUTPUT_DIR=/custom/output/path
export MANUS_CHECKPOINT_DIR=/custom/checkpoints
export CUDA_VISIBLE_DEVICES=0
```

---

## Stage Scripts

Individual training stages in `src/stages/`:

| Stage | Script | Datasets |
|-------|--------|----------|
| cot | `stage_cot.py` | OpenThoughts-114k |
| reasoning | `stage_reasoning.py` | NuminaMath-CoT |
| thinking | `stage_thinking.py` | s1K-1.1 |
| tools | `stage_tools.py` | xlam-function-calling-60k |
| streaming | `stage_streaming.py` | (runtime feature) |
| image-gen | `stage_image_gen.py` | naruto-blip-captions |
| video-gen | `stage_video_gen.py` | webvid |

### Run Individual Stage

```bash
python src/stages/stage_cot.py \
    --base-model /mnt/e/data/models/Qwen2.5-0.5B \
    --output-dir /mnt/e/data/models/trained/cot \
    --epochs 3 \
    --sample-size 1000
```

---

## Hardware Requirements

| Training Type | VRAM | Notes |
|---------------|------|-------|
| Text capabilities | 8GB | CoT, reasoning, tools |
| Omni conversion | 14GB | Adds encoders |
| Image generation | 14GB | SD3 projector |
| Video generation | 14GB+ | SVD projector |

---

## Troubleshooting

### Common Issues

**GPU Out of Memory**

```bash
# Reduce batch size
./run_universal_pipeline.sh --batch-size=1 --enable-cot

# Use 8-bit quantization
# (configure in stage script)
```

**Training Paused Unexpectedly**

```bash
# Check GPU temperature
nvidia-smi --query-gpu=temperature.gpu --format=csv

# Check if paused
grep "Paused" logs/train_*.log
```

**Modality Validation Failed**

```bash
# Run modality detection
python src/detect_modalities.py /path/to/model

# Add --enable-omni to convert text model
./run_universal_pipeline.sh --enable-omni --enable-podcast
```

---

## Documentation

| Document | Description |
|----------|-------------|
| [TEST_SUITE.md](docs/TEST_SUITE.md) | Complete test documentation |
| [SHELL_SCRIPTS.md](docs/SHELL_SCRIPTS.md) | Script reference |
| [GUIDE.md](docs/GUIDE.md) | User guide |
| [datasets.md](docs/multimodal/datasets.md) | Dataset catalog |

---

## License

MIT License - See LICENSE file for details.
