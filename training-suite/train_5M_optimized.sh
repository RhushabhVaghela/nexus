#!/bin/bash
# Training: 5M samples (Optimized, 3x faster)

set -e

echo "🚀 Training: 5M samples - Optimized"
echo "========================================"
echo "Sample distribution:"
echo "  Train: 4000000 samples"
echo "  Val:   500000 samples"
echo "  Test:  500000 samples"
echo ""
echo "Optimization: Optimized (3x speedup)"
echo ""

source /home/rhushabh/miniconda3/etc/profile.d/conda.sh
conda activate manus

cd "/mnt/d/Research Experiments/manus_model"

# Directories
mkdir -p /mnt/e/models/omni_5M_optimized
mkdir -p logs
mkdir -p results

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

echo "📊 Starting training..."
START_TIME=$(date +%s)

# Training
deepspeed --num_gpus=1 src/24_multimodal_training.py \
  --deepspeed ds_config.json \
  --stage 1 \
  --sample-size 5000000 \
  --data-path /mnt/e/data/downloaded/E-MM1-100M/data \
  --output-dir /mnt/e/models/omni_5M_optimized \
  --experiment-name "5M_optimized" \
  --log-results \
  2>&1 | tee logs/train_5M_optimized_$(date +%Y%m%d_%H%M%S).log

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo ""
echo "✅ Training complete!"
echo "Time: $((DURATION / 60)) minutes"
echo "Results saved to: results/training_results.csv"
