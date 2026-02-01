# Nexus Unified CLI Reference

> **Version:** 6.2  
> **Last Updated:** 2026-02-01

The **Nexus Unified CLI** (`scripts/nexus.sh`) is the single entry point for all Nexus operations, consolidating all previous shell scripts into one comprehensive tool with extensive progress tracking and real-time monitoring capabilities.

---

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Command Reference](#command-reference)
  - [master](#master)
  - [universal](#universal)
  - [training-suite](#training-suite)
  - [setup-voice](#setup-voice)
  - [monitor](#monitor)
  - [status](#status)
  - [reset](#reset)
- [Progress Tracking Features](#progress-tracking-features)
- [Environment Requirements](#environment-requirements)
- [Troubleshooting](#troubleshooting)

---

## Installation

The Unified CLI is included in the Nexus repository. Ensure it's executable:

```bash
chmod +x scripts/nexus.sh
```

### Prerequisites

- Linux/macOS environment
- Conda with `nexus` environment
- Python 3.9+
- CUDA-capable GPU (recommended)

### Setup

```bash
# Clone and setup
git clone <nexus-repo>
cd nexus

# Create conda environment (if not exists)
conda create -n nexus python=3.9
conda activate nexus

# Install dependencies
pip install -r requirements.txt
```

---

## Quick Start

```bash
# Show all available commands
./scripts/nexus.sh help

# Run master pipeline with reset
./scripts/nexus.sh master --reset

# Check current status
./scripts/nexus.sh status

# Monitor in real-time
./scripts/nexus.sh monitor
```

---

## Command Reference

### `master`

Run the master self-driving pipeline with full automation.

**Usage:**

```bash
./scripts/nexus.sh master [OPTIONS]
```

**Options:**

| Option | Description |
|--------|-------------|
| `--reset` | Full reset: clear state and checkpoints |
| `--dry-run` | Simulate execution without compute |
| `--skip-non-llm` | Skip audio/vision/multimodal models |
| `--stage NAME` | Run only specific stage (profiling, knowledge_extraction, training, router_training) |
| `--models ID1,ID2` | Filter to specific teacher models |
| `--datasets NAME` | Filter datasets |
| `--sample_size N` | Sample size for training |
| `--epochs N` | Training epochs |
| `--lr FLOAT` | Learning rate |
| `--router-epochs N` | Router training epochs |
| `--router-lr FLOAT` | Router learning rate |
| `--embedding-model PATH` | Custom embedding model |
| `--use-unsloth` | Use Unsloth for faster training |
| `--packing` | Enable sequence packing |
| `--max-seq-length N` | Maximum sequence length |
| `--grpo` | Use GRPO training method |

**Examples:**

```bash
# Run full pipeline with reset
./scripts/nexus.sh master --reset

# Run specific models only
./scripts/nexus.sh master --models "coder,vision_main" --sample_size 5000

# Run specific stage only
./scripts/nexus.sh master --stage training --dry-run

# Use Unsloth for faster training
./scripts/nexus.sh master --use-unsloth --packing --max-seq-length 8192
```

---

### `universal`

Universal pipeline for training any combination of capabilities on any base model. Automatically validates modality requirements before training.

**Usage:**

```bash
./scripts/nexus.sh universal [OPTIONS]
```

**Capability Flags:**

| Flag | Description |
|------|-------------|
| `--enable-omni` | Convert text model to Omni (add vision/audio) |
| `--enable-cot` | Chain-of-Thought reasoning |
| `--enable-reasoning` | Multi-level reasoning |
| `--enable-thinking` | Extended thinking/reflection |
| `--enable-tools` | Function/tool calling |
| `--enable-streaming` | Token streaming output |
| `--enable-podcast` | NotebookLM-style podcast |
| `--enable-vision-qa` | Image understanding |
| `--enable-video-understanding` | Video comprehension |
| `--enable-tri-streaming` | Real-time multimodal streaming |
| `--enable-image-generation` | Text-to-image (requires SD3) |
| `--enable-video-generation` | Text-to-video (requires SVD) |
| `--enable-remotion-explainer` | 3Blue1Brown-style video generation |
| `--enable-all-text` | Enable all text-only capabilities |
| `--enable-full-omni` | Enable Omni + all capabilities |

**Repetition Control (arXiv:2512.14982):**

| Option | Description |
|--------|-------------|
| `--repetition-factor N` | Global default repetition factor (1, 2, 3) |
| `--repetition-style STYLE` | Global style (baseline, 2x, verbose, 3x) |
| `--repetition-<capability> N` | Per-capability override |

**General Options:**

| Option | Description |
|--------|-------------|
| `--base-model PATH` | Base model path (required) |
| `--output-dir PATH` | Output directory |
| `--sample-size N` | Limit samples per dataset (0=all) |
| `--batch-size N` | Training batch size (default: 1) |
| `--gradient-accumulation N` | Gradient accumulation steps (default: 8) |
| `--epochs N` | Training epochs (default: 3) |
| `--training-method METHOD` | sft\|lora\|qlora\|dpo\|grpo\|orpo\|ppo\|distillation\|cpt |
| `--dry-run` | Simulate training without executing |
| `--organize` | Auto-organize datasets before training |

**Examples:**

```bash
# Enable CoT on a base model
./scripts/nexus.sh universal --base-model=/path/to/model --enable-cot

# Convert to Omni + add podcast
./scripts/nexus.sh universal --base-model=/path/to/model --enable-omni --enable-podcast

# Full Omni with all capabilities (dry run first)
./scripts/nexus.sh universal --base-model=/path/to/model --enable-full-omni --dry-run

# Image generation with 2x repetition
./scripts/nexus.sh universal --base-model=/path/to/model --enable-image-generation --repetition-image-generation=2
```

---

### `training-suite`

Generate a suite of training scripts with progress tracking for various dataset sizes and optimization levels.

**Usage:**

```bash
./scripts/nexus.sh training-suite [OPTIONS]
```

**Options:**

| Option | Description |
|--------|-------------|
| `--sizes SIZES` | Comma-separated list of sample sizes (default: 1K,10K,50K,100K,500K,1M,5M,10M,FULL) |
| `--output-dir PATH` | Output directory for generated scripts (default: training-suite/) |

**Generated Scripts:**

For each size, two scripts are generated:

- `train_<size>_optimized.sh` - 3x speedup with DeepSpeed
- `train_<size>_ultra.sh` - 6x speedup with optimized DeepSpeed config

**Examples:**

```bash
# Generate all training scripts
./scripts/nexus.sh training-suite

# Generate only specific sizes
./scripts/nexus.sh training-suite --sizes "10K,100K,1M"

# Custom output directory
./scripts/nexus.sh training-suite --output-dir custom-suite/

# Run a generated script
cd training-suite
./train_10K_optimized.sh
```

---

### `setup-voice`

Downloads and sets up voice models (NVIDIA PersonaPlex-7b-v1 and Microsoft VibeVoice-ASR).

**Usage:**

```bash
./scripts/nexus.sh setup-voice
```

**Models Downloaded:**

- `nvidia/personaplex-7b-v1` → `/mnt/e/data/models/personaplex-7b-v1`
- `microsoft/VibeVoice-ASR` → `/mnt/e/data/models/VibeVoice-ASR`

**Examples:**

```bash
# Download voice models
./scripts/nexus.sh setup-voice
```

---

### `monitor`

Launch real-time monitoring dashboard showing GPU utilization, training progress, system resources, and live log tailing.

**Usage:**

```bash
./scripts/nexus.sh monitor
```

**Display:**

- System memory usage
- GPU utilization and memory (if available)
- Disk usage
- Active Nexus processes
- Pipeline state
- Recent log activity
- Lock file status

**Controls:**

- Press `Ctrl+C` to exit

**Examples:**

```bash
# Start monitoring
./scripts/nexus.sh monitor
```

---

### `status`

Show current pipeline status including active processes, pipeline state, disk usage, recent logs, and system resources.

**Usage:**

```bash
./scripts/nexus.sh status
```

**Display:**

- Pipeline lock status
- Pipeline state file contents
- System resources (memory, GPU)
- Disk usage
- Log file count and recent entries
- Recent activity timestamps

**Examples:**

```bash
# Check status
./scripts/nexus.sh status
```

---

### `reset`

Reset pipeline state and cleanup all processes, lock files, and temporary files.

**Usage:**

```bash
./scripts/nexus.sh reset [OPTIONS]
```

**Options:**

| Option | Description |
|--------|-------------|
| `--force` | Force reset without confirmation |

**Actions:**

- Kill all running Nexus processes
- Remove lock files
- Clear Python cache (**pycache**, *.pyc,*.pyo)
- Remove temporary files
- Reset pipeline state

**Examples:**

```bash
# Reset with confirmation
./scripts/nexus.sh reset

# Force reset without confirmation
./scripts/nexus.sh reset --force
```

---

### Additional Commands

#### `pipeline`

Run the main text/code training pipeline.

```bash
./scripts/nexus.sh pipeline [download|process|validate|train|distill|all] [OPTIONS]
```

#### `multimodal`

Run the multimodal pipeline (vision/audio/video).

```bash
./scripts/nexus.sh multimodal [download|distill|train|all] [OPTIONS]
```

#### `reasoning`

Run the reasoning training pipeline (CoT, GRPO).

```bash
./scripts/nexus.sh reasoning --base-model=/path/to/model --enable-cot
```

#### `distillation`

Run knowledge distillation from teacher to student.

```bash
./scripts/nexus.sh distillation --teacher=/path/to/teacher --student=/path/to/student
```

#### `niwt`

Run NIWT (Neural Information-Weighted Tower) profiling.

```bash
./scripts/nexus.sh niwt --model_name="microsoft/Phi-3-mini-4k-instruct"
```

#### `profiling`

Run performance profiling on a model.

```bash
./scripts/nexus.sh profiling --model=/path/to/model --batch-size=8
```

#### `tests`

Run the test suite with intelligent categorization.

```bash
./scripts/nexus.sh tests --unit-only
./scripts/nexus.sh tests --integration-only --verbose
./scripts/nexus.sh tests --coverage --report
```

#### `cleanup`

Clean up temporary files and caches.

```bash
./scripts/nexus.sh cleanup
```

---

## Progress Tracking Features

All commands in the Unified CLI include extensive progress tracking:

### Visual Elements

- **Progress Bars**: Visual indication of completion percentage
- **Animated Spinners**: Indicate active operations
- **Stage Separators**: Clear visual separation between pipeline stages
- **Color Coding**:
  - Blue [INFO] for general status
  - Green [✓] for success
  - Yellow [⚠] for warnings
  - Red [✗] for errors
  - Purple [STAGE] for stage headers

### Timing Features

- **Elapsed Time Tracking**: Total elapsed time in HH:MM:SS format
- **Stage Timers**: Individual stage duration tracking
- **ETA Calculations**: Estimated time remaining based on throughput

### System Monitoring

- **Memory Usage**: Real-time memory consumption
- **GPU Utilization**: GPU usage and memory (if available)
- **Throughput Metrics**: Samples/sec, tokens/sec during training

### Example Output

```
═══════════════════════════════════════════════════════════════
[STAGE] Stage 1/5: Training cot
═══════════════════════════════════════════════════════════════
[Training] ████████████████████████████████░░░░░░░░░░░░░░░░░░ 64% (32/50) ETA: 00:12:34
⏱️  Stage Time: 00:22:15
⏱️  Total Elapsed: 01:45:32
```

---

## Environment Requirements

### Required

- **Conda Environment**: All commands expect the `nexus` conda environment:

  ```bash
  conda activate nexus
  ```

- **Python Dependencies**:
  - torch
  - transformers
  - faiss
  - huggingface_hub

### Optional

- **GPU**: CUDA-capable GPU for accelerated training
- **NVIDIA Tools**: `nvidia-smi` for GPU monitoring

---

## Troubleshooting

### "Must be run in 'nexus' conda environment"

Activate the environment:

```bash
conda activate nexus
```

### "Another Nexus instance is running"

Use reset to clear stale locks:

```bash
./scripts/nexus.sh reset --force
```

### GPU not detected

Check CUDA installation:

```bash
nvidia-smi
```

If not available, training will fall back to CPU (much slower).

### Lock file issues

Check and clear locks:

```bash
./scripts/nexus.sh status    # Check lock status
./scripts/nexus.sh reset     # Clear all locks
```

### Out of memory

Reduce batch size or use gradient accumulation:

```bash
./scripts/nexus.sh universal --base-model=/path/to/model --enable-cot --batch-size=1 --gradient-accumulation=16
```

---

## Migration from Old Scripts

The Unified CLI consolidates the following legacy scripts:

| Old Script | New Command |
|------------|-------------|
| `run_nexus_master.sh` | `./scripts/nexus.sh master` |
| `run_universal_pipeline.sh` | `./scripts/nexus.sh universal` |
| `generate_training_scripts.sh` | `./scripts/nexus.sh training-suite` |
| `scripts/setup_voice_models.sh` | `./scripts/nexus.sh setup-voice` |
| `scripts/run_profiling.sh` | `./scripts/nexus.sh profiling` |
| `scripts/run_distillation.sh` | `./scripts/nexus.sh distillation` |

---

## Related Documentation

- [Usage Guide](NEXUS_USAGE_GUIDE.md)
- [Scripts Guide](SCRIPTS_GUIDE.md)
- [Universal SLI Guide](SLI_UNIVERSAL_GUIDE.md)
- [Testing Guide](TESTING.md)
- [Troubleshooting](TROUBLESHOOTING.md)
