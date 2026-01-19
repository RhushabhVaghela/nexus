#!/bin/bash
# Launch Any-to-Any Omni Model Training with DFM Connector
# Stage 1: Train connectors (DFM) and decoders only, freeze LLM

set -e  # Exit on error

echo "🚀 Launching Any-to-Any Omni Model Training"
echo "============================================"
echo ""
echo "Configuration:"
echo "  - Base Model: gpt-oss-20b (13GB, 2.88B params)"
echo "  - Vision Encoder: SigLIP2 (4.3GB)"
echo "  - Audio Encoder: Whisper V3 Turbo (1.6GB)"
echo "  - Video Decoder: PaDT_OVD (7.2GB)"
echo "  - Speech Decoder: Parakeet-TDT (2.4GB)"
echo "  - Connector: DFM (SOTA, 398M params)"
echo "  - Dataset: E-MM1-100M (shards 1-3, 1000 samples)"
echo ""

# Activate conda environment
source /home/rhushabh/miniconda3/etc/profile.d/conda.sh
conda activate manus

# Set working directory
cd "/mnt/d/Research Experiments/manus_model"

# Create output directory
mkdir -p /mnt/e/models/omnimodal_any2any_dfm
mkdir -p logs

# Training configuration
export CUDA_VISIBLE_DEVICES=0  # Single GPU for now
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

echo "Starting Stage 1: DFM Connectors + Decoders Training..."
echo "⏱️  Estimated time: 30-60 minutes"
echo ""

# Launch training
python src/24_multimodal_training.py \
  --stage 1 \
  --data-path /mnt/e/data/downloaded/E-MM1-100M/data \
  --output-dir /mnt/e/models/omnimodal_any2any_dfm \
  2>&1 | tee logs/training_stage1_$(date +%Y%m%d_%H%M%S).log

echo ""
echo "✅ Stage 1 Training Complete!"
echo ""
echo "Next steps:"
echo "  1. Review checkpoints in: /mnt/e/models/omnimodal_any2any_dfm"
echo "  2. Launch Stage 2 for full model fine-tuning"
echo "  3. Run benchmarks on E-MM1 for validation"
