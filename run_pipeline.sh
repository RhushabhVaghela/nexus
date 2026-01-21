#!/bin/bash
# =============================================================================
# NEXUS PRIME - Master Pipeline Script
# =============================================================================
# 
# Main Pipeline (Text/Code):
#   01-03: Downloads (Real data, Benchmarks, Premium RLHF)
#   04:    Process real data
#   05-06: Generate synthetic data
#   07-09: Validate datasets
#   10-15: Training
#   16-21: Tools
#
#
# Usage:
#   ./run_pipeline.sh [phase] [options]
#
# Phases:
#   download    - Download text datasets (01-03)
#   process     - Process text data (04-06)
#   validate    - Validate text datasets (07-09)
#   train       - Run training pipeline (10-15)
#   distill     - Run distillation from teacher model
#   all         - Run complete TEXT pipeline (01-15)
#
# Options:
#   --mode=censored|uncensored  - Training mode (default: censored)
#   --target-samples=N          - For premium datasets (default: 100000)
#   --training-method=METHOD    - Training method (sft|lora|qlora|dpo|grpo|orpo|distillation)
#   --teacher-model=PATH        - Teacher model for distillation
#   --distillation-alpha=FLOAT  - Distillation alpha (default: 0.5)
#
# =============================================================================

set -e

# Enforce 'nexus' conda environment
if [ "$CONDA_DEFAULT_ENV" != "nexus" ]; then
    echo -e "\033[0;31m[ERROR] This script must be run in the 'nexus' conda environment.\033[0m"
    echo "Current environment: ${CONDA_DEFAULT_ENV:-None}"
    echo "Please run: conda activate nexus"
    exit 1
fi

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="${PROJECT_DIR}/src"
LOG_DIR="${PROJECT_DIR}/logs"

PHASE="${1:-all}"
MODE="censored"
TARGET_SAMPLES=100000
SAMPLE_SIZE=200000 # Default limit from python script
TRAINING_METHOD="sft"  # Default: supervised fine-tuning
TEACHER_MODEL=""
DISTILLATION_ALPHA=0.5

for arg in "$@"; do
    case $arg in
        --mode=*) MODE="${arg#*=}" ;;
        --target-samples=*) TARGET_SAMPLES="${arg#*=}" ;;
        --sample-size=*) SAMPLE_SIZE="${arg#*=}" ;;
        --training-method=*) TRAINING_METHOD="${arg#*=}" ;;
        --teacher-model=*) TEACHER_MODEL="${arg#*=}" ;;
        --distillation-alpha=*) DISTILLATION_ALPHA="${arg#*=}" ;;
    esac
done

mkdir -p "${LOG_DIR}"

GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }

# =============================================================================
# PHASE 1: DOWNLOAD (Text Only)
# =============================================================================
run_download() {
    log_info "Phase 1: Downloading text datasets..."
    
    python "${SRC_DIR}/01_download_real_datasets.py" --limit="${SAMPLE_SIZE}" 2>&1 | tee "${LOG_DIR}/01_download.log"
    python "${SRC_DIR}/02_download_benchmarks.py" 2>&1 | tee "${LOG_DIR}/02_benchmarks.log"
    python "${SRC_DIR}/03_load_premium_datasets.py" --mode="${MODE}" --target-samples="${TARGET_SAMPLES}" 2>&1 | tee "${LOG_DIR}/03_premium.log"
    
    log_success "Text download phase complete"
}

# =============================================================================
# PHASE 2: PROCESS
# =============================================================================
run_process() {
    log_info "Phase 2: Processing data..."
    
    python "${SRC_DIR}/04_process_real_datasets.py" 2>&1 | tee "${LOG_DIR}/04_process.log"
    python "${SRC_DIR}/05_generate_repetitive_dataset.py" 2>&1 | tee "${LOG_DIR}/05_repetitive.log"
    python "${SRC_DIR}/06_generate_preference_dataset.py" --mode="${MODE}" 2>&1 | tee "${LOG_DIR}/06_preferences.log"
    
    log_success "Process phase complete"
}

# =============================================================================
# PHASE 3: VALIDATE
# =============================================================================
run_validate() {
    log_info "Phase 3: Validating datasets..."
    
    python "${SRC_DIR}/07_validate_all_datasets.py" 2>&1 | tee "${LOG_DIR}/07_validate.log"
    python "${SRC_DIR}/08_validate_benchmarks.py" 2>&1 | tee "${LOG_DIR}/08_validate_benchmarks.log"
    python "${SRC_DIR}/09_validate_premium_datasets.py" --mode="${MODE}" 2>&1 | tee "${LOG_DIR}/09_validate_premium.log"
    
    log_success "Validation phase complete"
}

# =============================================================================

# Ensure 'nexus' environment
if [[ "$CONDA_DEFAULT_ENV" != "nexus" ]]; then
   if [ -f "/home/rhushabh/miniconda3/envs/nexus/bin/python" ]; then
       PYTHON_CMD="/home/rhushabh/miniconda3/envs/nexus/bin/python"
   else
       echo "Error: Must be in 'nexus' environment"
       exit 1
   fi
else
   PYTHON_CMD="python"
fi

# Auto-organize
echo "[INFO] Auto-organizing datasets..."
$PYTHON_CMD src/utils/organize_datasets.py --base-path /mnt/e/data --move || true

set -e

# ============ COLORS =================================================================
# PHASE 4: TRAIN
# =============================================================================
run_train() {
    log_info "Phase 4: Training (mode: ${MODE}, method: ${TRAINING_METHOD})..."
    
    # Pass training method to training scripts
    python "${SRC_DIR}/10_sft_training.py" \
        --mode="${MODE}" \
        --training-method="${TRAINING_METHOD}" \
        2>&1 | tee "${LOG_DIR}/10_sft.log"
    
    python "${SRC_DIR}/12_grpo_training.py" \
        --mode="${MODE}" \
        --training-method="${TRAINING_METHOD}" \
        2>&1 | tee "${LOG_DIR}/12_grpo.log"
    
    if [ "${MODE}" = "censored" ]; then
        python "${SRC_DIR}/13_safety_finetuning.py" 2>&1 | tee "${LOG_DIR}/13_safety.log"
    else
        python "${SRC_DIR}/14_anti_refusal_training.py" 2>&1 | tee "${LOG_DIR}/14_antirefusal.log"
    fi
    
    log_success "Training complete"
}

# =============================================================================
# PHASE 5: DISTILLATION (Optional)
# =============================================================================
run_distill() {
    if [ -z "${TEACHER_MODEL}" ]; then
        log_info "Skipping distillation: --teacher-model not specified"
        return
    fi
    
    log_info "Phase 5: Distillation from teacher: ${TEACHER_MODEL}"
    
    python "${SRC_DIR}/distillation.py" \
        --mode="${MODE}" \
        --teacher-model="${TEACHER_MODEL}" \
        --distillation-alpha="${DISTILLATION_ALPHA}" \
        2>&1 | tee "${LOG_DIR}/distillation.log"
    
    log_success "Distillation complete"
}

echo "==============================================="
echo "  NEXUS PRIME PIPELINE"
echo "  Phase: ${PHASE} | Mode: ${MODE}"
echo "  Training Method: ${TRAINING_METHOD}"
echo "==============================================="

case "${PHASE}" in
    download)   run_download ;;
    process)    run_process ;;
    validate)   run_validate ;;
    train)      run_train ;;
    distill)    run_distill ;;
    all)
        run_download
        run_process
        run_validate
        run_train
        run_distill
        ;;
    *)
        echo "Usage: ./run_pipeline.sh [download|process|validate|train|distill|all] [options]"
        echo ""
        echo "Training Methods: sft, lora, qlora, dpo, grpo, orpo, distillation"
        echo "Example: ./run_pipeline.sh train --training-method=qlora --mode=censored"
        exit 1
        ;;
esac

log_success "Pipeline execution finished!"
