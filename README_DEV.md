# Nexus Model - Universal Multimodal Training Pipeline

A comprehensive pipeline for training and extending language models with multimodal capabilities including vision, audio, video understanding, and generation.

---

## Quick Start

```bash
# Activate environment
conda activate nexus

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
| **remotion-explainer** | 3Blue1Brown-style explanatory video generation | text |

### Training Safety Features

- **Pause/Resume** - Signal-based training control
- **Automatic Cooldown** - Every 500 steps
- **GPU Temperature Protection** - Auto-pause at 83°C
- **Emergency Checkpoints** - SIGUSR2 for instant save

---

### Dataset Organization System

- **Automatic Sorting** - Moves raw datasets into categorized folders (`datasets/cot`, `datasets/tools`, etc.)
- **Content Detection** - Inspects JSON keys to identify dataset capability (e.g., `tool_calls` -> tools)
- **Model Management** - Organizes encoders/decoders by modality
- **Auto-Run** - Integrated into start of all pipelines

---

## Project Structure

```
nexus_model/
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

### 📚 Comprehensive Training Methods Reference

The pipeline supports **10 distinct training methodologies**, covering the full lifecycle of LLM development from pre-training to advanced reasoning alignment.

#### 1. Supervised Learning (The Foundation)

| Method | Script | Description | When to Use |
|--------|--------|-------------|-------------|
| **SFT** (Supervised Fine-Tuning) | `10_sft_training.py` | The standard approach. Updates model weights to minimize loss on target text. Supports full fine-tuning. | **Default choice**. Use for teaching new knowledge, formats, or styles (e.g., medical data, coding style). |
| **CPT** (Continued Pre-Training) | `11_continued_pretraining.py` | Like SFT but on massive unstructured corpora. Focuses on domain adaptation without specific Q&A structure. | When you have **gb/tb of raw text** (e.g., legal documents) and want the model to "understand" the domain before teaching it tasks. |
| **Distillation** | `23_multimodal_distillation.py` | Training a smaller "student" model to mimic a larger "teacher" model's outputs (or formatting raw multimodal data). | When deploying to **edge devices** or converting complex inputs (Vision/Audio) into standard formats without a massive model. |

#### 2. Parameter-Efficient Tuning (Low VRAM)

| Method | Script | Description | When to Use |
|--------|--------|-------------|-------------|
| **LoRA** (Low-Rank Adaptation) | `10_sft_training.py` | Freezes main weights and trains small "adapter" layers. Uses ~60% less VRAM than SFT. | **Always recommended** for single-GPU training. Matches SFT performance with fraction of the cost. |
| **QLoRA** (Quantized LoRA) | `10_sft_training.py` | Combines LoRA with 4-bit model quantization. Extreme memory efficiency (runs 70B models on consumer GPUs). | When you have **limited VRAM** (e.g., 24GB for a 30B model). Slight trade-off in precision for massive accessibility. |

#### 3. Alignment & Reasoning (Making it Smart)

| Method | Script | Description | When to Use |
|--------|--------|-------------|-------------|
| **DPO** (Direct Preference Opt) | `src/dpo_training.py` | Optimizes model to prefer "chosen" over "rejected" answers directly, skipping the Reward Model step of RLHF. | For **style alignment** or **reducing hallucinations**. Much more stable and faster than PPO. |
| **ORPO** (Odds Ratio PO) | `src/orpo_training.py` | Combines SFT and Alignment in one stage. Penalizes rejected answers during the generation phase. | **Efficiency**. Use if you want to SFT and Align simultaneously without a separate DPO stage. |
| **PPO** (Proximal Policy Opt) | `src/ppo_training.py` | The classic "RLHF" method. Uses a separate Reward Model to guide the policy via reinforcement learning. | For **complex, subjective tasks** where a Reward Model exists (e.g., "be helpful but not sycophantic"). Harder to tune but distinct results. |
| **GRPO** (Group Relative PO) | `src/12_grpo_training.py` | **DeepSeek-Style Reasoning**. Generates groups of outputs and reinforces the best ones based on rules/verifiers without a heavy critic model. | **Reasoning & Math**. The gold standard for "System 2" thinking properties. Forces model to self-correct and verify. |

#### 4. Safety & Robustness

| Method | Script | Description | When to Use |
|--------|--------|-------------|-------------|
| **Safety Tuning** | `13_safety_finetuning.py` | Targeted SFT on refusal/safety datasets to prevent harmful outputs or jailbreaks. | **Pre-deployment**. Critical for public-facing models to ensure compliance and safety. |
| **Anti-Refusal** | `14_anti_refusal_training.py` | Specialized training to *reduce* false refusals ("I cannot answer that") while maintaining core safety. | If your model has become **"too woke" or overly sensitive** and refuses harmless prompts. |

### Memory-Efficient Streaming & Long Context

- **Giant File Support**: Handles 500GB+ files via `ChunkedSampleProcessor` and streaming.
- **Long Context**: Scale to 128k+ tokens using `--long-context` (RoPE scaling).
- **Quick Validation**: Rapidly verify pipelines with `--quick` (tiny batches, 10 steps).

### Development Tools

- **Metrics Tracker**: Auto-logs training/benchmark stats to `results/*.csv`.
- **Dataset Organizer**: Manages data library (`src/utils/organize_datasets.py`).
- **Omni-Modal Loading**: Universal loader for any architecture.
- **Strict Real Models**: Zero mocks in production/testing.

---

## 🚀 "Nexus" Training Recipe (CoT + RL + Long Context)

To replicate a high-reasoning, long-context model like Nexus or Gemini 3 Pro:

### 1. SFT Stage (Thinking & Long Context)

Train on Chain-of-Thought (CoT) datasets with RoPE scaling enabled:

```bash
python src/10_sft_training.py \
    --long-context \
    --use_streaming \
    --learning_rate 2e-5
```

*Datasets: OpenO1, CoT-Collection (auto-detected)*

### 2. RL Stage (Refining Reasoning)

Apply GRPO (Reinforcement Learning) to reward correct reasoning paths:

```bash
python src/12_grpo_training.py \
    --stage 2
```

### 3. Verification

```bash
./run_universal_pipeline.sh --training-method grpo
```

---

## Usage

```bash
./run_universal_pipeline.sh \
    --base-model=/mnt/e/data/models/Qwen2.5-0.5B \
    --enable-all-text \
    --dry-run
```

Output:

```
╔═══════════════════════════════════════════════════════════════╗
║         NEXUS UNIVERSAL CAPABILITY PIPELINE                   ║
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
```

### ⚡ Advanced Test Parameters & Aliases

The test suite supports custom CLI flags for precise control:

| Flag | Alias | Description | Example |
|------|-------|-------------|---------|
| `--full-tests` | `-F` | Run **ALL** tests (including slow E2E/Integration) | `pytest -F` |
| `--full-benchmarks` | `-G` | Run **ALL** benchmarks (Global) | `pytest -G` |
| `--test` | `-T` | Filter specific tests (comma-separated, alias for `-k`) | `pytest -T "sft,lora"` |
| `--benchmark` | `-B` | Run specific benchmarks only | `pytest -B "mmlu,gsm8k"` |

```bash
# Example: Run full suite with benchmarks
pytest -F -G

# Example: Run only SFT-related tests
pytest -T sft
```

### Test Markers (`pytest -m ...`)

- `slow`: Long-running tests (skipped by default unless `-F` used).
- `gpu`: Tests requiring GPU.
- `real_model`: Tests loading actual weights.
- `omni`: Tests specific to Omni-modal features.

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
export NEXUS_OUTPUT_DIR=/custom/output/path
export NEXUS_CHECKPOINT_DIR=/custom/checkpoints
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
| remotion-explainer | `stage_remotion_gen.py` | remotion_explainer_dataset |

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
| Remotion Explainer | 12GB | Programmatic generation |

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
