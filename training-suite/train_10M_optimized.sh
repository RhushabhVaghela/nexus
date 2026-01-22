#!/bin/bash
# Training with Progress Tracking
set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to display elapsed time
display_time() {
    local duration=$1
    local hours=$((duration / 3600))
    local minutes=$(((duration % 3600) / 60))
    local seconds=$((duration % 60))
    printf "%02d:%02d:%02d" $hours $minutes $seconds
}

echo -e "${BLUE}🚀 Training: 10M samples - Optimized${NC}"
echo "========================================"
echo -e "${GREEN}Sample distribution:${NC}"
echo "  Train: 8000000 samples"
echo "  Val:   1000000 samples"
echo "  Test:  1000000 samples"
echo ""
echo -e "${YELLOW}Optimization: Optimized (3x speedup)${NC}"
echo ""

source /home/rhushabh/miniconda3/etc/profile.d/conda.sh
conda activate nexus

cd "/mnt/d/Research Experiments/nexus"

mkdir -p /mnt/e/models/omni_10M_optimized
mkdir -p logs
mkdir -p results

# DeepSpeed manages GPU allocation, no need to set CUDA_VISIBLE_DEVICES
export PYTORCH_ALLOC_CONF=max_split_size_mb:512

echo -e "${BLUE}📊 Starting training...${NC}"
START_TIME=$(date +%s)

# Progress monitoring in background
(
    sleep 5
    while kill -0 $$ 2>/dev/null; do
        CURRENT_TIME=$(date +%s)
        ELAPSED=$((CURRENT_TIME - START_TIME))
        echo -ne "\r${YELLOW}⏱️  Elapsed: $(display_time $ELAPSED)${NC}"
        sleep 10
    done
) &
MONITOR_PID=$!

# Training
deepspeed --num_gpus=1 src/24_multimodal_training.py \
  --deepspeed ../config/ds_config.json \
  --stage 1 \
  --sample-size 10000000 \
  --data-path /mnt/e/data/datasets/E-MM1-100M/data \
  --output-dir /mnt/e/models/omni_10M_optimized \
  --experiment-name "10M_optimized" \
  --log-results \
  2>&1 | tee logs/train_10M_optimized_$(date +%Y%m%d_%H%M%S).log

# Stop monitor
kill $MONITOR_PID 2>/dev/null || true

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo ""
echo -e "${GREEN}✅ Training complete!${NC}"
echo -e "${BLUE}⏱️  Total time: $(display_time $DURATION) ($(($DURATION / 60)) minutes)${NC}"
echo -e "${GREEN}📊 Results saved to: results/training_results.csv${NC}"
