# =============================================================================
# NEXUS PRIME - Multimodal Pipeline Script
# =============================================================================
# 
# Purpose: Convert ANY text model to OMNI (any-to-any multimodal)
#
# Intelligent Behavior:
#   - Detects base model's native modalities using detect_modalities.py
#   - SKIPS training if model is already Omni (all modalities present)
#   - Only adds MISSING modalities to partial multimodal models
#
# 2. Start Multimodal Training
echo "🔍 Checking base model capabilities..."

# Detect modalities using the python script
MODALITIES=$(python3 src/detect_modalities.py --json "$BASE_MODEL")
IS_ALREADY_OMNI=$(echo "$MODALITIES" | python3 -c "import sys, json; data=json.load(sys.stdin); mods=data.get('modalities', {}); print('true' if mods.get('vision') and mods.get('audio_input') and mods.get('video') else 'false')")

if [ "$IS_ALREADY_OMNI" == "true" ] && [ "$FORCE" == "false" ]; then
    echo "⚠️  Model is ALREADY Omni-capable (Vision + Audio + Video)."
    echo "   Skipping training to prevent degradation."
    echo "   Use --force to train anyway."
    exit 0
fi

if [ "$IS_ALREADY_OMNI" == "true" ]; then
    echo "ℹ️  Model is Omni, but --force is set. Proceeding with training..."
fi

echo "🚀 Starting Multimodal Pipeline..."
# Stages:
#   1. DOWNLOAD (Script 22) - Fetch raw Vision/Audio data
#   2. DISTILL  (Script 23) - Process and format multimodal data
#   3. TRAIN    (Script 24) - Train Omni-Modal Projectors
#
# Usage:
#   ./run_multimodal_pipeline.sh [phase] --base-model=/path/to/model [options]
#
# Phases:
#   download      - Run Script 22
#   distill       - Run Script 23 (requires --modality)
#   train         - Run Script 24 (requires --stage)
#   all           - Run full pipeline
#
# Options:
#   --base-model=PATH        Base model path (REQUIRED for train)
#   --modality=vision|audio|video
#   --stage=1|2
#   --teacher=mock-teacher|gpt-4v|gemini-pro
#   --force                  Force training even if model is Omni
#
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
MODALITY="vision"
STAGE=1
TEACHER="mock-teacher"
LIMIT=1000
SAMPLE_SIZE=0
BASE_MODEL=""
FORCE_TRAIN=false

for arg in "$@"; do
    case $arg in
        --base-model=*) BASE_MODEL="${arg#*=}" ;;
        --force) FORCE_TRAIN=true ;;
        --modality=*) MODALITY="${arg#*=}" ;;
        --stage=*) STAGE="${arg#*=}" ;;
        --teacher=*) TEACHER="${arg#*=}" ;;
        --limit=*) LIMIT="${arg#*=}" ;;
        --sample-size=*) SAMPLE_SIZE="${arg#*=}" ;;
    esac
done

mkdir -p "${LOG_DIR}"
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[⚠]${NC} $1"; }

YELLOW='\033[0;33m'

# =============================================================================
# MODALITY DETECTION - Check if model is already Omni
# =============================================================================
check_if_omni() {
    if [ -z "${BASE_MODEL}" ]; then
        log_warn "No --base-model specified, skipping Omni detection"
        return 1  # Not Omni (continue training)
    fi
    
    log_info "Detecting modalities for: ${BASE_MODEL}"
    
    # Run detect_modalities.py and capture result
    RESULT=$(python "${SRC_DIR}/detect_modalities.py" --json "${BASE_MODEL}" 2>/dev/null || echo '{}')
    
    # Check if is_omni flag is set
    IS_OMNI=$(echo "$RESULT" | python -c "import sys,json; d=json.load(sys.stdin); print(d.get('is_omni', False))" 2>/dev/null || echo "False")
    
    if [ "$IS_OMNI" = "True" ]; then
        return 0  # Is Omni
    else
        return 1  # Not Omni
    fi
}

# 1. DOWNLOAD
run_download() {
    log_info "Phase 1: Download Multimodal Data (Limit: ${LIMIT})..."
    python "${SRC_DIR}/22_multimodal_pipeline.py" --phase download --limit "${LIMIT}" 2>&1 | tee "${LOG_DIR}/22_multimodal_dl.log"
}

# 2. DISTILL
run_distill() {
    log_info "Phase 2: Distill Data (Teacher: ${TEACHER}, Modality: ${MODALITY})..."
    python "${SRC_DIR}/23_multimodal_distillation.py" \
        --modality "${MODALITY}" \
        --teacher "${TEACHER}" \
        2>&1 | tee "${LOG_DIR}/23_distill_${MODALITY}.log"
}

# 3. TRAIN
run_train() {
    # Determine data path based on modality default
    DATA_PATH="/mnt/e/data/datasets"
    
    log_info "Phase 3: Train Omni-Modal Model (Stage: ${STAGE})..."
    python "${SRC_DIR}/24_multimodal_training.py" \
        --stage "${STAGE}" \
        --data-path "${DATA_PATH}" \
        --sample-size "${SAMPLE_SIZE}" \
        2>&1 | tee "${LOG_DIR}/24_train_stage${STAGE}.log"
}

# MAIN
case "${PHASE}" in
    download) run_download ;;
    distill)  run_distill ;;
    train)    run_train ;;
    all)
        run_download
        run_distill
        run_train
        ;;
    *)
        echo "Usage: ./run_multimodal_pipeline.sh [download|distill|train|all] [options]"
        exit 1
        ;;
esac

log_success "Multimodal pipeline execution finished!"
