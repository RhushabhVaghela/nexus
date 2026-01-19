# Manus Any-to-Any Omni Model

Production-ready any-to-any multimodal model with DFM connectors, optimized training suite, and comprehensive dataset support.

## 🚀 Quick Start

### 1. Test Setup (10 min)

```bash
cd training-suite
./train_1K_ultra.sh
```

### 2. Development (3 hours)

```bash
cd training-suite
./train_100K_ultra.sh
```

### 3. Production (6 days)

```bash
cd training-suite
./train_5M_ultra.sh
```

## 📁 Project Structure

```
manus_model/
├── src/                          # Source code
│   ├── multimodal/              # Multimodal components
│   │   ├── model.py             # OmniMultimodalLM (DFM-powered)
│   │   ├── connectors/          # DFM & Perceiver connectors
│   │   └── datasets/            # Dataset loaders
│   ├── utils/                   # Utilities
│   ├── 24_multimodal_training.py # Main training script
│   └── process_manual_datasets.py
├── training-suite/              # 18 training scripts
│   ├── README.md
│   ├── train_1K_ultra.sh       # Fast test
│   ├── train_5M_ultra.sh       # Production (recommended)
│   └── train_FULL_ultra.sh     # Complete dataset
├── config/                      # Training configurations
│   ├── training_config.yaml
│   ├── ds_config.json          # DeepSpeed ZeRO-2
│   └── ds_config_ultra.json    # DeepSpeed ZeRO-3
├── base-model/                 # Model weights
│   ├── gpt-oss-20b/
│   ├── siglip2-so400m-patch16-512/
│   ├── whisper-large-v3-turbo/
│   ├── PaDT_OVD_3B/
│   └── parakeet-tdt-0.6b-v3/
├── results/                    # Training results CSV
└── logs/                       # Training logs

## 🎯 Features

- **True Any-to-Any**: Image → Video, Audio → Text, etc.
- **DFM Connectors**: SOTA discrete flow matching (5-10% gains)
- **Ultra-Optimized**: 6x faster training (4-bit, ZeRO-3)
- **100M+ Samples**: E-MM1 + 10 manual datasets
- **Memory Efficient**: Fits 16GB VRAM + 32GB RAM
- **Auto Train/Val/Test**: 80/10/10 splits automatic

## � Pipeline Usage

The project uses two master shell scripts to handle the entire lifecycle.

### 1. Multimodal Pipeline (Vision/Audio/Video)

Use this for training the Omni-Modal model (Connectors, Projectors, Full Fine-tuning).

```bash
./run_multimodal_pipeline.sh [PHASE] [OPTIONS]
```

**Phases:**

- `download`: Download raw datasets (Script 22).
- `distill`: Run teacher distillation (Script 23).
- `train`: Run training loop (Script 24).
- `all`: Run full sequence (**Default**).

**Options (Flags)**

| Flag | Description | Valid Values | Default |
| :--- | :--- | :--- | :--- |
| `--modality` | Data type to process | `vision`, `audio`, `video` | `vision` |
| `--stage` | Training configuration | `1` (Projectors), `2` (Full Model) | `1` |
| `--sample-size` | **Limit samples per dataset** | Any Integer (e.g. `50`) | `0` (All) |
| `--limit` | Download limit (HuggingFace) | Any Integer | `1000` |
| `--teacher` | Teacher model for labelling | `mock-teacher`, `gpt-4v` | `mock-teacher` |

**Example:**

```bash
# Train Stage 1 with only 50 samples per dataset (Fast Test)
./run_multimodal_pipeline.sh train --stage=1 --sample-size=50
```

### 2. Text/Code Pipeline (SFT/RLHF)

Use this for standard LLM fine-tuning (Text, Code, Reasoning).

```bash
./run_pipeline.sh [PHASE] [OPTIONS]
```

**Phases:** `download`, `process`, `validate`, `train`, `all`.

**Options (Flags)**

| Flag | Description | Valid Values | Default |
| :--- | :--- | :--- | :--- |
| `--mode` | Safety alignment mode | `censored`, `uncensored` | `censored` |
| `--sample-size` | Download limit | Any Integer | `200000` |
| `--target-samples` | Premium data limit | Any Integer | `100000` |

## 🛠 Setup

```bash
conda activate manus
pip install -r requirements.txt
```

## 📈 Results

All experiments logged to `results/training_results.csv`:

- Training/val/test losses
- VRAM/RAM usage
- Training time
- Throughput

View results:

```bash
cat results/training_results.csv
```

## 🏗 Architecture

- **LLM**: GPT-OSS-20B (13GB, 4-bit quantized)
- **Vision**: SigLIP2 → DFM Connector
- **Audio**: Whisper V3 → DFM Connector
- **Video Decoder**: PaDT_OVD_3B
- **Speech Decoder**: Parakeet-TDT
- **Optimization**: DeepSpeed ZeRO-3, 4-bit QLoRA

## 📚 Documentation

See `training-suite/README.md` for detailed usage.

## ✅ Ready to Train
