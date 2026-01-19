#!/bin/bash
# Training Script Generator
# Generates all 18 training scripts (9 sample sizes × 2 optimization levels)

cd "$(dirname "$0")"

# Sample sizes
SIZES=(1000 10000 50000 100000 500000 1000000 5000000 10000000 0)
SIZE_NAMES=("1K" "10K" "50K" "100K" "500K" "1M" "5M" "10M" "FULL")

# Function to create script
create_script() {
    local size=$1
    local size_name=$2
    local opt_level=$3  # "optimized" or "ultra"
    
    if [ "$opt_level" = "optimized" ]; then
        ds_config="ds_config.json"
        opt_name="Optimized"
        speedup="3x"
    else
        ds_config="ds_config_ultra.json"
        opt_name="Ultra-Optimized"
        speedup="6x"
    fi
    
    # Calculate train/val/test splits
    if [ "$size" = "0" ]; then
        train_size="0"  # Unlimited
        val_size="0"
        test_size="0"
        desc="Full dataset"
    else
        train_size=$((size * 80 / 100))
        val_size=$((size * 10 / 100))
        test_size=$((size * 10 / 100))
        desc="${size_name} samples"
    fi
    
    script_name="train_${size_name}_${opt_level}.sh"
    
    cat > "$script_name" << EOF
#!/bin/bash
# Training: ${desc} (${opt_name}, ${speedup} faster)

set -e

echo "🚀 Training: ${desc} - ${opt_name}"
echo "========================================"
echo "Sample distribution:"
echo "  Train: ${train_size} samples"
echo "  Val:   ${val_size} samples"
echo "  Test:  ${test_size} samples"
echo ""
echo "Optimization: ${opt_name} (${speedup} speedup)"
echo ""

source /home/rhushabh/miniconda3/etc/profile.d/conda.sh
conda activate manus

cd "/mnt/d/Research Experiments/manus_model"

# Directories
mkdir -p /mnt/e/models/omni_${size_name}_${opt_level}
mkdir -p logs
mkdir -p results

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

echo "📊 Starting training..."
START_TIME=\$(date +%s)

# Training
deepspeed --num_gpus=1 src/24_multimodal_training.py \\
  --deepspeed ${ds_config} \\
  --stage 1 \\
  --sample-size ${size} \\
  --data-path /mnt/e/data/downloaded/E-MM1-100M/data \\
  --output-dir /mnt/e/models/omni_${size_name}_${opt_level} \\
  --experiment-name "${size_name}_${opt_level}" \\
  --log-results \\
  2>&1 | tee logs/train_${size_name}_${opt_level}_\$(date +%Y%m%d_%H%M%S).log

END_TIME=\$(date +%s)
DURATION=\$((END_TIME - START_TIME))

echo ""
echo "✅ Training complete!"
echo "Time: \$((DURATION / 60)) minutes"
echo "Results saved to: results/training_results.csv"
EOF

    chmod +x "$script_name"
    echo "✓ Created $script_name"
}

echo "Generating training scripts..."
echo "==============================="

# Generate all scripts
for i in "${!SIZES[@]}"; do
    size="${SIZES[$i]}"
    size_name="${SIZE_NAMES[$i]}"
    
    create_script "$size" "$size_name" "optimized"
    create_script "$size" "$size_name" "ultra"
done

echo ""
echo "✅ Generated 18 training scripts!"
echo ""
echo "Usage:"
echo "  ./train_1K_optimized.sh      # 1000 samples, optimized"
echo "  ./train_1K_ultra.sh          # 1000 samples, ultra-optimized"
echo "  ./train_5M_optimized.sh      # 5M samples, optimized"
echo "  ./train_FULL_ultra.sh        # All samples, ultra-optimized"
echo ""
echo "Results will be logged to: results/training_results.csv"
