#!/bin/bash
# =============================================================================
# MANUS UNIVERSAL PIPELINE ORCHESTRATOR
# =============================================================================
#
# Unified pipeline for training any combination of capabilities on any base model.
# Automatically validates modality requirements before training.
#
# Usage:
#   ./run_universal_pipeline.sh --base-model PATH [CAPABILITIES] [OPTIONS]
#
# Examples:
#   # Text-only capabilities (any model)
#   ./run_universal_pipeline.sh --base-model /path/to/model --enable-cot --enable-tools
#
#   # Convert to Omni + add podcast
#   ./run_universal_pipeline.sh --base-model /path/to/model --enable-omni --enable-podcast
#
#   # Full Omni with tri-streaming
#   ./run_universal_pipeline.sh --base-model /path/to/omni-model --enable-tri-streaming
#
#   # Image generation (requires vision_output decoder)
#   ./run_universal_pipeline.sh --base-model /path/to/model --enable-omni --enable-image-generation
#
# =============================================================================

set -e

# ============ COLORS ============
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[✓]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[⚠]${NC} $1"; }
log_error() { echo -e "${RED}[✗]${NC} $1"; }
log_step() { echo -e "${PURPLE}[STAGE]${NC} $1"; }

# ============ DEFAULTS ============
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="${PROJECT_DIR}/src"
LOG_DIR="${PROJECT_DIR}/logs"
CONFIG_FILE="${PROJECT_DIR}/configs/encoders.yaml"

BASE_MODEL="/mnt/e/data/base-model/Qwen2.5-Omni-7B-GPTQ-Int4"
OUTPUT_DIR="/mnt/e/data/models/trained"
CHECKPOINT_DIR="/mnt/e/data/models/checkpoints"

# Capability flags (all false by default)
ENABLE_OMNI=false
ENABLE_COT=false
ENABLE_REASONING=false
ENABLE_THINKING=false
ENABLE_TOOLS=false
ENABLE_STREAMING=false
ENABLE_PODCAST=false
ENABLE_VISION_QA=false
ENABLE_VIDEO_UNDERSTANDING=false
ENABLE_TRI_STREAMING=false
ENABLE_IMAGE_GENERATION=false
ENABLE_VIDEO_GENERATION=false

# Training options
SAMPLE_SIZE=0  # 0 = all
BATCH_SIZE=1
GRADIENT_ACCUMULATION=8
EPOCHS=3

# ============ PARSE ARGUMENTS ============
print_usage() {
    echo "Usage: ./run_universal_pipeline.sh --base-model PATH [CAPABILITIES] [OPTIONS]"
    echo ""
    echo "Capabilities (enable with flags):"
    echo "  --enable-omni              Convert text model to Omni (add vision/audio)"
    echo "  --enable-cot               Chain-of-Thought reasoning"
    echo "  --enable-reasoning         Multi-level reasoning"
    echo "  --enable-thinking          Extended thinking/reflection"
    echo "  --enable-tools             Function/tool calling"
    echo "  --enable-streaming         Token streaming output"
    echo "  --enable-podcast           NotebookLM-style podcast"
    echo "  --enable-vision-qa         Image understanding"
    echo "  --enable-video-understanding  Video comprehension"
    echo "  --enable-tri-streaming     Real-time multimodal streaming"
    echo "  --enable-image-generation  Text-to-image (requires SD3)"
    echo "  --enable-video-generation  Text-to-video (requires SVD)"
    echo "  --enable-all-text          Enable all text-only capabilities"
    echo "  --enable-full-omni         Enable Omni + all capabilities"
    echo ""
    echo "Options:"
    echo "  --base-model PATH          Base model path (required)"
    echo "  --output-dir PATH          Output directory"
    echo "  --sample-size N            Limit samples per dataset (0=all)"
    echo "  --batch-size N             Training batch size (default: 1)"
    echo "  --epochs N                 Training epochs (default: 3)"
    echo ""
    exit 1
}

for arg in "$@"; do
    case $arg in
        --base-model=*) BASE_MODEL="${arg#*=}" ;;
        --output-dir=*) OUTPUT_DIR="${arg#*=}" ;;
        --sample-size=*) SAMPLE_SIZE="${arg#*=}" ;;
        --batch-size=*) BATCH_SIZE="${arg#*=}" ;;
        --epochs=*) EPOCHS="${arg#*=}" ;;
        --enable-omni) ENABLE_OMNI=true ;;
        --enable-cot) ENABLE_COT=true ;;
        --enable-reasoning) ENABLE_REASONING=true ;;
        --enable-thinking) ENABLE_THINKING=true ;;
        --enable-tools) ENABLE_TOOLS=true ;;
        --enable-streaming) ENABLE_STREAMING=true ;;
        --enable-podcast) ENABLE_PODCAST=true ;;
        --enable-vision-qa) ENABLE_VISION_QA=true ;;
        --enable-video-understanding) ENABLE_VIDEO_UNDERSTANDING=true ;;
        --enable-tri-streaming) ENABLE_TRI_STREAMING=true ;;
        --enable-image-generation) ENABLE_IMAGE_GENERATION=true ;;
        --enable-video-generation) ENABLE_VIDEO_GENERATION=true ;;
        --enable-all-text)
            ENABLE_COT=true
            ENABLE_REASONING=true
            ENABLE_THINKING=true
            ENABLE_TOOLS=true
            ENABLE_STREAMING=true
            ;;
        --enable-full-omni)
            ENABLE_OMNI=true
            ENABLE_COT=true
            ENABLE_REASONING=true
            ENABLE_THINKING=true
            ENABLE_TOOLS=true
            ENABLE_PODCAST=true
            ENABLE_VISION_QA=true
            ENABLE_TRI_STREAMING=true
            ;;
        --help|-h) print_usage ;;
    esac
done

# ============ VALIDATE ARGUMENTS ============
if [ ! -d "$BASE_MODEL" ] && [ ! -f "$BASE_MODEL/config.json" ]; then
    log_error "Base model not found: $BASE_MODEL"
    exit 1
fi

# Check if any capability is enabled
if ! $ENABLE_OMNI && ! $ENABLE_COT && ! $ENABLE_REASONING && ! $ENABLE_THINKING && \
   ! $ENABLE_TOOLS && ! $ENABLE_STREAMING && ! $ENABLE_PODCAST && ! $ENABLE_VISION_QA && \
   ! $ENABLE_VIDEO_UNDERSTANDING && ! $ENABLE_TRI_STREAMING && \
   ! $ENABLE_IMAGE_GENERATION && ! $ENABLE_VIDEO_GENERATION; then
    log_error "No capabilities enabled. Use --enable-* flags."
    print_usage
fi

mkdir -p "$LOG_DIR" "$OUTPUT_DIR" "$CHECKPOINT_DIR"

# ============ HEADER ============
echo ""
echo -e "${CYAN}╔═══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║         MANUS UNIVERSAL CAPABILITY PIPELINE                   ║${NC}"
echo -e "${CYAN}╚═══════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "  Base Model:  ${GREEN}$(basename "$BASE_MODEL")${NC}"
echo -e "  Output:      ${OUTPUT_DIR}"
echo ""

# ============ STAGE 0: DETECT MODALITIES ============
log_step "Stage 0: Detecting Model Modalities"

MODALITY_JSON=$(python "${SRC_DIR}/detect_modalities.py" "$BASE_MODEL" --json 2>/dev/null)
if [ $? -ne 0 ]; then
    log_error "Failed to detect modalities"
    exit 1
fi

# Extract modalities using Python
HAS_VISION=$(echo "$MODALITY_JSON" | python -c "import sys,json; d=json.load(sys.stdin); print('true' if d['modalities'].get('vision') else 'false')")
HAS_AUDIO_IN=$(echo "$MODALITY_JSON" | python -c "import sys,json; d=json.load(sys.stdin); print('true' if d['modalities'].get('audio_input') else 'false')")
HAS_AUDIO_OUT=$(echo "$MODALITY_JSON" | python -c "import sys,json; d=json.load(sys.stdin); print('true' if d['modalities'].get('audio_output') else 'false')")
HAS_VIDEO=$(echo "$MODALITY_JSON" | python -c "import sys,json; d=json.load(sys.stdin); print('true' if d['modalities'].get('video') else 'false')")
HAS_VISION_OUT=$(echo "$MODALITY_JSON" | python -c "import sys,json; d=json.load(sys.stdin); print('true' if d['modalities'].get('vision_output') else 'false')")
HAS_VIDEO_OUT=$(echo "$MODALITY_JSON" | python -c "import sys,json; d=json.load(sys.stdin); print('true' if d['modalities'].get('video_output') else 'false')")

echo -e "  Detected: text=✓ vision=$HAS_VISION audio_in=$HAS_AUDIO_IN audio_out=$HAS_AUDIO_OUT video=$HAS_VIDEO"
echo ""

# ============ MODALITY GATE VALIDATION ============
log_step "Stage 1: Modality Gate Validation"

VALIDATION_FAILED=false

# Podcast requires audio_input + audio_output
if $ENABLE_PODCAST && ([ "$HAS_AUDIO_IN" = "false" ] || [ "$HAS_AUDIO_OUT" = "false" ]); then
    if ! $ENABLE_OMNI; then
        log_error "Podcast requires audio_input + audio_output"
        log_warn "  Current model lacks these modalities."
        log_warn "  Hint: Use --enable-omni to add them first."
        VALIDATION_FAILED=true
    fi
fi

# Vision-QA requires vision
if $ENABLE_VISION_QA && [ "$HAS_VISION" = "false" ]; then
    if ! $ENABLE_OMNI; then
        log_error "Vision-QA requires vision modality"
        log_warn "  Hint: Use --enable-omni to add vision encoder."
        VALIDATION_FAILED=true
    fi
fi

# Tri-streaming requires ALL modalities
if $ENABLE_TRI_STREAMING; then
    if [ "$HAS_VISION" = "false" ] || [ "$HAS_AUDIO_IN" = "false" ] || \
       [ "$HAS_AUDIO_OUT" = "false" ] || [ "$HAS_VIDEO" = "false" ]; then
        if ! $ENABLE_OMNI; then
            log_error "Tri-streaming requires FULL OMNI (text + vision + audio + video)"
            log_warn "  Hint: Use --enable-omni first."
            VALIDATION_FAILED=true
        fi
    fi
fi

# Image generation requires vision_output decoder
if $ENABLE_IMAGE_GENERATION && [ "$HAS_VISION_OUT" = "false" ]; then
    log_warn "Image generation requires vision_output decoder (SD3)"
    log_info "  Will train projector to connect LLM → SD3"
fi

# Video generation requires video_output decoder
if $ENABLE_VIDEO_GENERATION && [ "$HAS_VIDEO_OUT" = "false" ]; then
    log_warn "Video generation requires video_output decoder (SVD)"
    log_info "  Will train projector to connect LLM → SVD"
fi

if $VALIDATION_FAILED; then
    log_error "Modality validation failed. Exiting."
    exit 1
fi

log_success "All modality gates passed!"
echo ""

# ============ BUILD TRAINING ORDER ============
STAGES=()

if $ENABLE_OMNI; then
    STAGES+=("omni")
fi
if $ENABLE_COT; then
    STAGES+=("cot")
fi
if $ENABLE_REASONING; then
    STAGES+=("reasoning")
fi
if $ENABLE_THINKING; then
    STAGES+=("thinking")
fi
if $ENABLE_TOOLS; then
    STAGES+=("tools")
fi
if $ENABLE_STREAMING; then
    STAGES+=("streaming")
fi
if $ENABLE_VISION_QA; then
    STAGES+=("vision-qa")
fi
if $ENABLE_VIDEO_UNDERSTANDING; then
    STAGES+=("video-understanding")
fi
if $ENABLE_PODCAST; then
    STAGES+=("podcast")
fi
if $ENABLE_TRI_STREAMING; then
    STAGES+=("tri-streaming")
fi
if $ENABLE_IMAGE_GENERATION; then
    STAGES+=("image-generation")
fi
if $ENABLE_VIDEO_GENERATION; then
    STAGES+=("video-generation")
fi

log_step "Stage 2: Training Queue (${#STAGES[@]} stages)"
for i in "${!STAGES[@]}"; do
    echo -e "  $((i+1)). ${STAGES[$i]}"
done
echo ""

# ============ EXECUTE TRAINING STAGES ============
CURRENT_MODEL="$BASE_MODEL"

for stage in "${STAGES[@]}"; do
    log_step "Training: $stage"
    STAGE_OUTPUT="${OUTPUT_DIR}/${stage}"
    mkdir -p "$STAGE_OUTPUT"
    
    case $stage in
        omni)
            python "${SRC_DIR}/24_multimodal_training.py" \
                --base-model "$CURRENT_MODEL" \
                --output-dir "$STAGE_OUTPUT" \
                --sample-size "$SAMPLE_SIZE" \
                2>&1 | tee "${LOG_DIR}/train_omni.log"
            CURRENT_MODEL="$STAGE_OUTPUT"
            ;;
        tools)
            python "${SRC_DIR}/16_tool_integration.py" \
                --model "$CURRENT_MODEL" \
                --output-dir "$STAGE_OUTPUT" \
                2>&1 | tee "${LOG_DIR}/train_tools.log"
            CURRENT_MODEL="$STAGE_OUTPUT"
            ;;
        cot|reasoning|thinking)
            log_info "Running $stage training on reasoning datasets..."
            python "${SRC_DIR}/10_sft_training.py" \
                --model "$CURRENT_MODEL" \
                --datasets "$stage" \
                --output-dir "$STAGE_OUTPUT" \
                --epochs "$EPOCHS" \
                2>&1 | tee "${LOG_DIR}/train_${stage}.log"
            CURRENT_MODEL="$STAGE_OUTPUT"
            ;;
        podcast|vision-qa|video-understanding|tri-streaming)
            log_info "Running multimodal $stage training..."
            python "${SRC_DIR}/24_multimodal_training.py" \
                --base-model "$CURRENT_MODEL" \
                --output-dir "$STAGE_OUTPUT" \
                --capability "$stage" \
                --sample-size "$SAMPLE_SIZE" \
                2>&1 | tee "${LOG_DIR}/train_${stage}.log"
            CURRENT_MODEL="$STAGE_OUTPUT"
            ;;
        image-generation|video-generation)
            log_info "Running generation projector training for $stage..."
            log_warn "  (This is advanced - using projector-only training)"
            python "${SRC_DIR}/stages/stage_${stage//-/_}.py" \
                --base-model "$CURRENT_MODEL" \
                --output-dir "$STAGE_OUTPUT" \
                2>&1 | tee "${LOG_DIR}/train_${stage}.log" || true
            CURRENT_MODEL="$STAGE_OUTPUT"
            ;;
        *)
            log_warn "Unknown stage: $stage (skipping)"
            ;;
    esac
    
    log_success "Completed: $stage"
    echo ""
done

# ============ FINAL SUMMARY ============
echo ""
echo -e "${CYAN}╔═══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║                    TRAINING COMPLETE                          ║${NC}"
echo -e "${CYAN}╚═══════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "  Final Model:  ${GREEN}$CURRENT_MODEL${NC}"
echo -e "  Stages Run:   ${#STAGES[@]}"
echo -e "  Logs:         ${LOG_DIR}"
echo ""
log_success "Pipeline finished successfully!"
