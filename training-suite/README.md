# Training Suite: Comprehensive Experiment Scripts

This directory contains 18 training scripts for systematic experimentation across different sample sizes and optimization levels.

## 🆕 New: Universal Orchestrator

For capability-based training, use the new orchestrator instead:

```bash
# From project root
./run_universal_pipeline.sh --enable-cot --enable-tools

# See all options
./run_universal_pipeline.sh --help
```

The orchestrator handles modality detection, validation, and sequential training.

---

## Structure

```
training-suite/
├── README.md (this file)
├── train_1K_optimized.sh
├── train_1K_ultra.sh
├── train_10K_optimized.sh
├── train_10K_ultra.sh
├── train_50K_optimized.sh
├── train_50K_ultra.sh
├── train_100K_optimized.sh
├── train_100K_ultra.sh
├── train_500K_optimized.sh
├── train_500K_ultra.sh
├── train_1M_optimized.sh
├── train_1M_ultra.sh
├── train_5M_optimized.sh
├── train_5M_ultra.sh
├── train_10M_optimized.sh
├── train_10M_ultra.sh
├── train_FULL_optimized.sh
└── train_FULL_ultra.sh
```

## Quick Reference

| Samples | Optimized (3x) | Ultra (6x) | Time (Ultra) | Accuracy |
|:--------|:---------------|:-----------|:-------------|:---------|
| 1K | `train_1K_optimized.sh` | `train_1K_ultra.sh` | 10 min | ~70% |
| 10K | `train_10K_optimized.sh` | `train_10K_ultra.sh` | 17 min | ~82% |
| 50K | `train_50K_optimized.sh` | `train_50K_ultra.sh` | 1.4 hours | ~87% |
| 100K | `train_100K_optimized.sh` | `train_100K_ultra.sh` | 3 hours | ~88% |
| 500K | `train_500K_optimized.sh` | `train_500K_ultra.sh` | 14 hours | ~92% |
| 1M | `train_1M_optimized.sh` | `train_1M_ultra.sh` | 28 hours | ~90% |
| **5M** ⭐ | `train_5M_optimized.sh` | `train_5M_ultra.sh` | **6 days** | **~94%** |
| 10M | `train_10M_optimized.sh` | `train_10M_ultra.sh` | 12 days | ~95.5% |
| Full | `train_FULL_optimized.sh` | `train_FULL_ultra.sh` | 116 days | ~97.5% |

## Usage

### Quick Test (1K samples)

```bash
cd training-suite
./train_1K_ultra.sh
```

### Development (100K samples)

```bash
cd training-suite
./train_100K_ultra.sh
```

### Production (5M samples) - Recommended

```bash
cd training-suite  
./train_5M_ultra.sh
```

## Features

- **Train/Val/Test Splits:** Automatic 80/10/10 split
- **CSV Logging:** Results append to `../results/training_results.csv`
- **E-MM1 Shard Strategy:** Train uses shards 1-13, Val uses 14-15, Test uses 16
- **Memory Optimized:** Fits 16GB VRAM + 32GB RAM

## Results

View all experiment results:

```bash
cat ../results/training_results.csv
```

Compare experiments:

```bash
# Best validation loss
sort -t, -k11 -n ../results/training_results.csv | head -5

# Fastest training
sort -t, -k9 -n ../results/training_results.csv | head -5
```

## Recommended Workflow

1. **Validate setup:** `./train_1K_ultra.sh` (10 min)
2. **Test hyperparameters:** `./train_10K_ultra.sh` (17 min)
3. **Iteration:** `./train_100K_ultra.sh` (3 hours)
4. **Production:** `./train_5M_ultra.sh` (6 days)
5. **Maximum quality:** `./train_10M_ultra.sh` (12 days)

## Notes

- All scripts use DeepSpeed ZeRO for memory efficiency
- Ultra scripts use 4-bit quantization (6x faster)
- Optimized scripts use 8-bit quantization (3x faster)
- Results automatically logged with metrics
- For capability-based training, see `../run_universal_pipeline.sh`
