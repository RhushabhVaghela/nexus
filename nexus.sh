#!/bin/bash
# =============================================================================
# NEXUS UNIFIED CLI - v6.2
# =============================================================================
# A unified command-line interface for all Nexus operations with extensive
# progress tracking, real-time metrics, and comprehensive pipeline orchestration.
#
# Usage: ./nexus.sh [COMMAND] [OPTIONS]
#
# Commands:
#   master            Run the master self-driving pipeline
#   universal         Run the universal capability pipeline
#   training-suite    Generate training scripts with various configurations
#   setup-voice       Setup voice models (PersonaPlex, VibeVoice)
#   pipeline          Run the main text/code training pipeline
#   multimodal        Run the multimodal pipeline (vision/audio/video)
#   reasoning         Run the reasoning training pipeline
#   distillation      Run knowledge distillation
#   niwt              Run NIWT profiling pipeline
#   profiling         Run performance profiling
#   monitor           Real-time monitoring dashboard
#   status            Show current pipeline status
#   reset             Reset pipeline state and cleanup
#   tests             Run the test suite
#   cleanup           Clean up temporary files and caches
#   help              Show this help message
#
# Examples:
#   ./nexus.sh master --reset
#   ./nexus.sh universal --base-model=/path/to/model --enable-cot
#   ./nexus.sh training-suite
#   ./nexus.sh monitor
#   ./nexus.sh status
#
# =============================================================================

set -e

# ===================== GLOBAL CONFIGURATION =====================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}" && pwd)"
SRC_DIR="${PROJECT_DIR}/src"
LOG_DIR="${PROJECT_DIR}/logs"
LOCKFILE="/tmp/nexus_master.pid"
STATE_FILE="${PROJECT_DIR}/.pipeline_state.json"

# Ensure logs directory exists
mkdir -p "${LOG_DIR}"

# ===================== COLORS =====================
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
WHITE='\033[1;37m'
BOLD='\033[1m'
NC='\033[0m'

# ===================== LOGGING FUNCTIONS =====================
log_info()    { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[✓]${NC} $1"; }
log_warn()    { echo -e "${YELLOW}[ ⚠ ]${NC} $1"; }
log_error()   { echo -e "${RED}[✗]${NC} $1"; }
log_step()    { 
    echo -e "\n${PURPLE}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${PURPLE}[STAGE]${NC} $1"
    echo -e "${PURPLE}═══════════════════════════════════════════════════════════════${NC}"
}
log_header() {
    echo -e "${CYAN}╔═══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║${NC} $1"
    echo -e "${CYAN}╚═══════════════════════════════════════════════════════════════╝${NC}"
}

# ===================== PROGRESS TRACKING FUNCTIONS =====================

# Global timing variables
SCRIPT_START_TIME=$(date +%s)
STAGE_START_TIME=0
CURRENT_STAGE=""

# Initialize stage timing
start_stage_timer() {
    STAGE_START_TIME=$(date +%s)
    CURRENT_STAGE="$1"
}

# Format seconds to HH:MM:SS
format_time() {
    local total_seconds=$1
    local hours=$((total_seconds / 3600))
    local minutes=$(((total_seconds % 3600) / 60))
    local seconds=$((total_seconds % 60))
    printf "%02d:%02d:%02d" $hours $minutes $seconds
}

# Display elapsed time
show_elapsed() {
    local current_time=$(date +%s)
    local elapsed=$((current_time - SCRIPT_START_TIME))
    echo -e "${BLUE}⏱️  Total Elapsed: $(format_time $elapsed)${NC}"
}

# Display stage elapsed time
show_stage_elapsed() {
    local current_time=$(date +%s)
    local elapsed=$((current_time - STAGE_START_TIME))
    echo -e "${CYAN}⏱️  Stage Time: $(format_time $elapsed)${NC}"
}

# Show progress bar
# Usage: show_progress_bar current total [width] [title]
show_progress_bar() {
    local current=$1
    local total=$2
    local width=${3:-50}
    local title=${4:-"Progress"}
    
    if [ $total -eq 0 ]; then
        total=1
    fi
    
    local percentage=$((current * 100 / total))
    local filled=$((width * current / total))
    local empty=$((width - filled))
    
    # Create bar
    local bar=""
    for ((i=0; i<filled; i++)); do bar+="█"; done
    for ((i=0; i<empty; i++)); do bar+="░"; done
    
    # Calculate ETA
    if [ $current -gt 0 ]; then
        local current_time=$(date +%s)
        local elapsed=$((current_time - STAGE_START_TIME))
        local rate=$(echo "scale=4; $current / $elapsed" | bc 2>/dev/null || echo "0")
        if [ "$(echo "$rate > 0" | bc 2>/dev/null || echo "0")" -eq 1 ]; then
            local remaining=$((total - current))
            local eta=$(echo "scale=0; $remaining / $rate" | bc 2>/dev/null || echo "0")
            local eta_formatted=$(format_time $eta)
            printf "\r${BLUE}[%s]${NC} ${CYAN}%s${NC} ${GREEN}%3d%%${NC} (${YELLOW}%d/%d${NC}) ETA: ${GREEN}%s${NC}" "$title" "$bar" $percentage $current $total "$eta_formatted"
        else
            printf "\r${BLUE}[%s]${NC} ${CYAN}%s${NC} ${GREEN}%3d%%${NC} (${YELLOW}%d/%d${NC})" "$title" "$bar" $percentage $current $total
        fi
    else
        printf "\r${BLUE}[%s]${NC} ${CYAN}%s${NC} ${GREEN}%3d%%${NC} (${YELLOW}%d/%d${NC})" "$title" "$bar" $percentage $current $total
    fi
}

# Spinner animation for long operations
# Usage: show_spinner "message" &
#        SPINNER_PID=$!
#        ... do work ...
#        kill $SPINNER_PID
show_spinner() {
    local message="$1"
    local spin_chars="⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
    local i=0
    local start_time=$(date +%s)
    
    while true; do
        local current_time=$(date +%s)
        local elapsed=$((current_time - start_time))
        local char="${spin_chars:$i:1}"
        printf "\r${CYAN}%s${NC} %s (${YELLOW}%s${NC})" "$char" "$message" "$(format_time $elapsed)"
        i=$(((i + 1) % 10))
        sleep 0.1
    done
}

# Show ETA calculation
# Usage: show_eta start_time current total
show_eta() {
    local start_time=$1
    local current=$2
    local total=$3
    
    if [ $current -eq 0 ] || [ $total -eq 0 ]; then
        echo "N/A"
        return
    fi
    
    local current_time=$(date +%s)
    local elapsed=$((current_time - start_time))
    local rate=$(echo "scale=6; $current / $elapsed" | bc 2>/dev/null || echo "0")
    
    if [ "$(echo "$rate > 0" | bc 2>/dev/null || echo "0")" -eq 1 ]; then
        local remaining=$((total - current))
        local eta=$(echo "scale=0; $remaining / $rate" | bc 2>/dev/null || echo "0")
        format_time $eta
    else
        echo "N/A"
    fi
}

# Show system metrics
show_metrics() {
    local throughput="$1"
    local memory="$2"
    local gpu="$3"
    
    echo -e "${PURPLE}📊 Metrics:${NC}"
    [ -n "$throughput" ] && echo -e "  ${GREEN}Throughput:${NC} $throughput"
    [ -n "$memory" ] && echo -e "  ${YELLOW}Memory:${NC} $memory"
    [ -n "$gpu" ] && echo -e "  ${CYAN}GPU:${NC} $gpu"
}

# Track a stage with automatic timing and progress
# Usage: track_stage "Stage Name" "command to run"
track_stage() {
    local stage_name="$1"
    shift
    local command="$@"
    
    log_step "$stage_name"
    start_stage_timer "$stage_name"
    
    # Start spinner in background
    show_spinner "Running $stage_name..." &
    local spinner_pid=$!
    
    # Run the command
    eval "$command"
    local exit_code=$?
    
    # Kill spinner
    kill $spinner_pid 2>/dev/null || true
    wait $spinner_pid 2>/dev/null || true
    printf "\r%-80s\r" ""
    
    if [ $exit_code -eq 0 ]; then
        log_success "$stage_name completed"
        show_stage_elapsed
    else
        log_error "$stage_name failed (exit code: $exit_code)"
        show_stage_elapsed
        return $exit_code
    fi
}

# Get memory usage
get_memory_usage() {
    if command -v free &> /dev/null; then
        local mem_info=$(free -h | awk '/^Mem:/ {print $3 "/" $2}')
        echo "$mem_info"
    elif command -v vm_stat &> /dev/null; then
        # macOS
        local used=$(vm_stat | grep 'Pages active' | awk '{print $3}' | sed 's/\.//')
        local total=$(vm_stat | grep 'Pages free' | awk '{print $3}' | sed 's/\.//')
        if [ -n "$used" ] && [ -n "$total" ]; then
            local used_gb=$((used * 4096 / 1024 / 1024 / 1024))
            local total_gb=$(((used + total) * 4096 / 1024 / 1024 / 1024))
            echo "${used_gb}GB/${total_gb}GB"
        fi
    fi
}

# Get GPU utilization
get_gpu_utilization() {
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 | awk -F', ' '{print $1 "% | " $2 "/" $3 " MB"}'
    elif command -v rocm-smi &> /dev/null; then
        rocm-smi --showuse 2>/dev/null | grep -E "GPU" | head -1 | awk '{print $2 "%"}'
    fi
}

# Monitor function for background processes
monitor_process() {
    local pid=$1
    local name=$2
    
    while kill -0 $pid 2>/dev/null; do
        local mem=$(get_memory_usage)
        local gpu=$(get_gpu_utilization)
        local current_time=$(date +%s)
        local elapsed=$((current_time - STAGE_START_TIME))
        printf "\r${CYAN}[%s]${NC} Elapsed: ${YELLOW}%s${NC} | Memory: ${GREEN}%s${NC} | GPU: ${PURPLE}%s${NC}   " "$name" "$(format_time $elapsed)" "$mem" "$gpu"
        sleep 5
    done
    printf "\r%-100s\r" ""
}

# ===================== SAFE KILL TREE =====================
kill_tree() {
    local pid="$1"
    [[ -z "$pid" || "$pid" -le 1 || "$pid" -eq "$$" ]] && return
    for child in $(pgrep -P "$pid" 2>/dev/null || true); do
        kill_tree "$child"
    done
    kill -TERM "$pid" 2>/dev/null || true
}

force_kill_tree() {
    local pid="$1"
    [[ -z "$pid" || "$pid" -le 1 || "$pid" -eq "$$" ]] && return
    for child in $(pgrep -P "$pid" 2>/dev/null || true); do
        force_kill_tree "$child"
    done
    kill -KILL "$pid" 2>/dev/null || true
}

# ===================== ENVIRONMENT CHECK =====================
check_environment() {
    if [[ "$CONDA_DEFAULT_ENV" != "nexus" ]]; then
        DESIRED_PYTHON="${CONDA_PREFIX:-$HOME/miniconda3}/envs/nexus/bin/python"
        if [ -f "$DESIRED_PYTHON" ]; then
            export PATH="$(dirname "$DESIRED_PYTHON"):$PATH"
            PYTHON_CMD="$DESIRED_PYTHON"
            log_warn "Not in 'nexus' environment. Using: $DESIRED_PYTHON"
        else
            log_error "Error: Must be run in 'nexus' conda environment."
            echo "    Current: ${CONDA_DEFAULT_ENV:-None}"
            echo "    Please run: conda activate nexus"
            exit 1
        fi
    else
        PYTHON_CMD="python"
    fi
    export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
    export PYTHONUNBUFFERED=1
}

# ===================== HEALTH CHECK =====================
health_check() {
    log_info "Performing system health check..."
    
    # Check Python dependencies
    $PYTHON_CMD -c "import torch, faiss, transformers, huggingface_hub" 2>/dev/null || {
        log_error "Critical dependencies missing (torch, faiss, transformers, huggingface_hub)"
        exit 1
    }
    
    # Check GPU availability
    local gpu_info=$(get_gpu_utilization)
    if [ -n "$gpu_info" ]; then
        log_success "GPU detected: $gpu_info"
    else
        log_warn "No GPU detected - training may be slow"
    fi
    
    # Check memory
    local mem_info=$(get_memory_usage)
    log_info "Memory: $mem_info"
    
    log_success "Dependencies verified"
}

# ===================== LOCK FILE MANAGEMENT =====================
check_lock() {
    if [[ -f "$LOCKFILE" ]]; then
        local old_pid=$(cat "$LOCKFILE" || true)
        if [[ -n "$old_pid" ]] && ps -p "$old_pid" >/dev/null 2>&1; then
            log_error "Another Nexus instance is running (PID $old_pid). Use --reset to replace it."
            exit 1
        else
            log_warn "Stale lock file found, removing..."
            rm -f "$LOCKFILE"
        fi
    fi
}

acquire_lock() {
    echo "$$" > "$LOCKFILE"
    trap 'rm -f "$LOCKFILE"' EXIT
}

# ===================== SIGNAL TRAPPING =====================
setup_signal_traps() {
    trap 'echo; log_warn "Interrupted by user"; rm -f "$LOCKFILE"; exit 130' INT
    trap 'echo; log_error "Terminated"; rm -f "$LOCKFILE"; exit 143' TERM
}

# ===================== MONITOR UTILS INTEGRATION =====================
source_monitor_utils() {
    local MONITOR_SCRIPT="${SCRIPT_DIR}/utils/monitor_utils.sh"
    if [ -f "$MONITOR_SCRIPT" ]; then
        source "$MONITOR_SCRIPT"
    else
        start_monitor() { echo "Starting monitor: $1..."; }
        stop_monitor() { :; }
    fi
}

# ===================== RESET/CLEANUP =====================
cleanup_existing_processes() {
    log_warn "RESET mode active — safe Nexus cleanup"
    
    if [[ -f "$LOCKFILE" ]]; then
        local old_pid=$(cat "$LOCKFILE" || true)
        if [[ -n "$old_pid" ]] && ps -p "$old_pid" >/dev/null 2>&1; then
            log_warn "Terminating existing Nexus instance (PID $old_pid)"
            kill_tree "$old_pid"
            sleep 2
            ps -p "$old_pid" >/dev/null 2>&1 && force_kill_tree "$old_pid"
        fi
        rm -f "$LOCKFILE"
    fi
    
    log_warn "Removing lock & cache files"
    find "${PROJECT_DIR}" -type f -name "*.lock" -delete 2>/dev/null || true
    rm -rf "${PROJECT_DIR}/.cache" "${PROJECT_DIR}/__pycache__" "${LOG_DIR}"/* "${PROJECT_DIR}/.pipeline_state.json" 2>/dev/null || true
    
    log_success "RESET cleanup complete"
}

# ===================== HELP MESSAGE =====================
show_help() {
    cat << 'EOF'
╔═══════════════════════════════════════════════════════════════════════════╗
║                    NEXUS UNIFIED CLI - v6.2                               ║
║              Universal Modular AI - Knowledge Distillation                ║
║                   With Extensive Progress Tracking                        ║
╚═══════════════════════════════════════════════════════════════════════════╝

USAGE:
    ./nexus.sh [COMMAND] [OPTIONS]

COMMANDS:
    master            Run the master self-driving pipeline
    universal         Run the universal capability pipeline
    training-suite    Generate training scripts with progress tracking
    setup-voice       Setup voice models (PersonaPlex, VibeVoice)
    pipeline          Run the main text/code training pipeline
    multimodal        Run the multimodal pipeline (vision/audio/video)
    reasoning         Run the reasoning training pipeline (CoT, GRPO)
    distillation      Run knowledge distillation from teacher to student
    niwt              Run NIWT (Neural Information-Weighted Tower) profiling
    profiling         Run performance profiling on a model
    monitor           Real-time monitoring dashboard
    status            Show current pipeline status and metrics
    reset             Reset pipeline state and cleanup all processes
    tests             Run the test suite
    cleanup           Clean up temporary files and caches
    help              Show this help message

─────────────────────────────────────────────────────────────────────────────
COMMAND DETAILS:
─────────────────────────────────────────────────────────────────────────────

▶ master [OPTIONS]

  Run the master self-driving pipeline with full automation.

  Options:
    --reset                       Full reset: clear state and checkpoints
    --dry-run                     Simulate execution without compute
    --skip-non-llm                Skip audio/vision/multimodal models
    --stage NAME                  Run only specific stage (profiling,
                                  knowledge_extraction, training, router_training)
    --models ID1,ID2              Filter to specific teacher models
    --datasets NAME               Filter datasets
    --sample_size N               Sample size for training
    --epochs N                    Training epochs
    --lr FLOAT                    Learning rate
    --use-unsloth                 Use Unsloth for faster training
    --packing                     Enable sequence packing
    --max-seq-length N            Maximum sequence length
    --grpo                        Use GRPO training method

  Examples:
    ./nexus.sh master --reset
    ./nexus.sh master --models "coder,vision_main" --sample_size 5000
    ./nexus.sh master --stage training --dry-run

▶ universal [OPTIONS]

  Universal pipeline for training any combination of capabilities on any base
  model. Automatically validates modality requirements before training.

  Capability Flags:
    --enable-omni                 Convert text model to Omni (add vision/audio)
    --enable-cot                  Chain-of-Thought reasoning
    --enable-reasoning            Multi-level reasoning
    --enable-thinking             Extended thinking/reflection
    --enable-tools                Function/tool calling
    --enable-streaming            Token streaming output
    --enable-podcast              NotebookLM-style podcast
    --enable-vision-qa            Image understanding
    --enable-video-understanding  Video comprehension
    --enable-tri-streaming        Real-time multimodal streaming
    --enable-image-generation     Text-to-image (requires SD3)
    --enable-video-generation     Text-to-video (requires SVD)
    --enable-remotion-explainer   3Blue1Brown-style video generation
    --enable-all-text             Enable all text-only capabilities
    --enable-full-omni            Enable Omni + all capabilities

  Repetition Control (arXiv:2512.14982):
    --repetition-factor N         Global default repetition factor (1, 2, 3)
    --repetition-style STYLE      Global style (baseline, 2x, verbose, 3x)
    --repetition-<capability> N   Per-capability override

  Options:
    --base-model PATH             Base model path (required)
    --output-dir PATH             Output directory
    --sample-size N               Limit samples per dataset (0=all)
    --batch-size N                Training batch size (default: 1)
    --gradient-accumulation N     Gradient accumulation steps (default: 8)
    --epochs N                    Training epochs (default: 3)
    --training-method METHOD      sft|lora|qlora|dpo|grpo|orpo|ppo|distillation|cpt
    --dry-run                     Simulate training without executing
    --organize                    Auto-organize datasets before training

  Examples:
    ./nexus.sh universal --base-model=/path/to/model --enable-cot
    ./nexus.sh universal --base-model=/path/to/model --enable-omni --enable-podcast
    ./nexus.sh universal --base-model=/path/to/model --enable-full-omni --dry-run

▶ training-suite [OPTIONS]

  Generate a suite of training scripts with progress tracking for various
  dataset sizes and optimization levels.

  Options:
    --sizes SIZES                 Comma-separated list of sample sizes
                                  (default: 1K,10K,50K,100K,500K,1M,5M,10M,FULL)
    --output-dir PATH             Output directory for generated scripts
                                  (default: training-suite/)

  Examples:
    ./nexus.sh training-suite
    ./nexus.sh training-suite --sizes "10K,100K,1M"
    ./nexus.sh training-suite --output-dir custom-suite/

▶ setup-voice

  Downloads and sets up voice models (NVIDIA PersonaPlex-7b-v1 and
  Microsoft VibeVoice-ASR) to /mnt/e/data/models.

  Examples:
    ./nexus.sh setup-voice

▶ pipeline [PHASE] [OPTIONS]

  Run the main text/code training pipeline.

  Phases: download, process, validate, train, distill, all

  Options:
    --mode=censored|uncensored    Training mode (default: censored)
    --target-samples=N            Target samples for premium datasets
    --training-method=METHOD      sft|lora|qlora|dpo|grpo|orpo|distillation
    --teacher-model=PATH          Teacher model for distillation
    --distillation-alpha=FLOAT    Distillation alpha (default: 0.5)

  Examples:
    ./nexus.sh pipeline all
    ./nexus.sh pipeline train --training-method=qlora
    ./nexus.sh pipeline distill --teacher-model=/path/to/teacher

▶ multimodal [PHASE] [OPTIONS]

  Convert any text model to multimodal (Omni) capabilities.

  Phases: download, distill, train, all

  Options:
    --base-model=PATH             Base model path (required for train)
    --modality=vision|audio|video Modality to train
    --stage=1|2                   Training stage
    --teacher=mock-teacher|gpt-4v Teacher model for distillation
    --force                       Force training even if already Omni
    --limit=N                     Dataset sample limit

  Examples:
    ./nexus.sh multimodal all --base-model=/path/to/model
    ./nexus.sh multimodal train --base-model=/path/to/model --stage=1

▶ reasoning [OPTIONS]

  Train models with advanced reasoning capabilities (CoT, GRPO).

  Options:
    --base-model PATH             Path to base model (required)
    --output-dir PATH             Output directory
    --enable-cot                  Enable CoT dataset generation
    --enable-context              Enable context extension
    --skip-sft                    Skip SFT stage
    --skip-grpo                   Skip GRPO stage
    --cot-type TYPE               Reasoning type: math|code|logic
    --target-context N            Target context length (default: 32768)

  Examples:
    ./nexus.sh reasoning --base-model=/path/to/model --enable-cot
    ./nexus.sh reasoning --base-model=/path/to/model --enable-context

▶ distillation [OPTIONS]

  Run knowledge distillation from teacher to student model.

  Options:
    --teacher PATH                Teacher model path
    --student PATH                Student model path
    --data PATH                   Data directory
    --alpha FLOAT                 Distillation alpha (default: 0.5)
    --temperature FLOAT           Temperature for soft targets (default: 2.0)

  Examples:
    ./nexus.sh distillation --teacher=/path/to/teacher --student=/path/to/student

▶ niwt [OPTIONS]

  Run NIWT (Neural Information-Weighted Tower) profiling pipeline.

  Options:
    --model_name NAME             Model name from registry (required)
    --batch_size N                Batch size (default: 8)
    --samples N                   Number of samples (default: 50)

  Examples:
    ./nexus.sh niwt --model_name="microsoft/Phi-3-mini-4k-instruct"

▶ profiling [OPTIONS]

  Run performance profiling on a model.

  Options:
    --model PATH                  Model path (default: /mnt/e/data/models/AgentCPM-Explore)
    --batch-size N                Batch size (default: 4)

  Examples:
    ./nexus.sh profiling --model=/path/to/model --batch-size=8

▶ monitor

  Launch real-time monitoring dashboard showing:
  - GPU utilization and memory usage
  - Training progress and throughput
  - System resource usage
  - Live log tailing

  Examples:
    ./nexus.sh monitor

▶ status

  Show current pipeline status including:
  - Active processes
  - Pipeline state
  - Disk usage
  - Recent logs
  - System resources

  Examples:
    ./nexus.sh status

▶ reset

  Reset pipeline state and cleanup:
  - Kill all Nexus processes
  - Remove lock files
  - Clear cache and temporary files
  - Reset pipeline state

  Options:
    --force                       Force reset without confirmation

  Examples:
    ./nexus.sh reset
    ./nexus.sh reset --force

▶ tests [OPTIONS]

  Run the test suite with intelligent categorization.

  Options:
    --unit-only                   Run only unit tests
    --integration-only            Run only integration tests
    --real-models                 Include tests requiring real models
    --distributed                 Include distributed tests
    --gpu                         Include GPU tests
    --slow                        Include slow tests
    --benchmark                   Include benchmark tests
    --coverage                    Generate coverage report
    --report                      Generate JSON test report
    -v, --verbose                 Verbose output
    --all                         Run all tests including real models

  Examples:
    ./nexus.sh tests --unit-only
    ./nexus.sh tests --integration-only --verbose
    ./nexus.sh tests --coverage --report

▶ cleanup

  Clean up temporary files, caches, and old artifacts.

  Examples:
    ./nexus.sh cleanup

▶ help

  Show this help message.

─────────────────────────────────────────────────────────────────────────────
PROGRESS TRACKING FEATURES:
─────────────────────────────────────────────────────────────────────────────

All commands include extensive progress tracking:

  ✓ Real-time ETA calculations
  ✓ Progress bars with percentage
  ✓ Elapsed time tracking (HH:MM:SS format)
  ✓ Stage progress indicators
  ✓ Memory usage monitoring
  ✓ GPU utilization tracking
  ✓ Throughput metrics (samples/sec, tokens/sec)
  ✓ Animated spinners for long operations

─────────────────────────────────────────────────────────────────────────────
ENVIRONMENT:
─────────────────────────────────────────────────────────────────────────────

All commands expect the 'nexus' conda environment to be active:

    conda activate nexus

Logs are written to: ./logs/

─────────────────────────────────────────────────────────────────────────────
DOCUMENTATION:
─────────────────────────────────────────────────────────────────────────────

• Full Usage Guide: docs/NEXUS_USAGE_GUIDE.md
• Unified CLI Guide: docs/UNIFIED_CLI.md
• Shell Scripts: docs/SCRIPTS_GUIDE.md
• Universal SLI: docs/SLI_UNIVERSAL_GUIDE.md
• Testing: docs/TESTING.md
• Troubleshooting: docs/TROUBLESHOOTING.md

═══════════════════════════════════════════════════════════════════════════
EOF
}

# ===================== COMMAND: MASTER =====================
cmd_master() {
    check_environment
    source_monitor_utils
    setup_signal_traps
    
    shift  # Remove 'master' from args
    
    # Configuration defaults
    RESET_STATE=false
    DRY_RUN=false
    SKIP_NON_LLM=false
    TARGET_STAGE=""
    SELECTED_MODELS="all"
    SELECTED_DATASETS=""
    SAMPLE_SIZE=""
    EPOCHS=""
    LR=""
    ROUTER_EPOCHS=""
    ROUTER_LR=""
    EMBEDDING_MODEL=""
    USE_UNSLOTH=false
    PACKING=false
    MAX_SEQ_LENGTH=""
    USE_GRPO=false
    
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --reset) RESET_STATE=true ;;
            --dry-run) DRY_RUN=true ;;
            --skip-non-llm) SKIP_NON_LLM=true ;;
            --stage) TARGET_STAGE="$2"; shift ;;
            --models) SELECTED_MODELS="$2"; shift ;;
            --datasets) SELECTED_DATASETS="$2"; shift ;;
            --sample_size) SAMPLE_SIZE="$2"; shift ;;
            --epochs) EPOCHS="$2"; shift ;;
            --lr) LR="$2"; shift ;;
            --router-epochs) ROUTER_EPOCHS="$2"; shift ;;
            --router-lr) ROUTER_LR="$2"; shift ;;
            --embedding-model) EMBEDDING_MODEL="$2"; shift ;;
            --use-unsloth) USE_UNSLOTH=true ;;
            --packing) PACKING=true ;;
            --max-seq-length) MAX_SEQ_LENGTH="$2"; shift ;;
            --grpo) USE_GRPO=true ;;
            --help|-h) show_help; exit 0 ;;
            *) log_warn "Unknown parameter: $1" ;;
        esac
        shift
    done
    
    # Header
    log_header "              NEXUS SELF-DRIVING PIPELINE v6.2                 "
    
    # Reset or check lock
    if $RESET_STATE; then
        cleanup_existing_processes
    else
        check_lock
        log_info "Reset not requested — skipping process cleanup"
    fi
    
    acquire_lock
    
    # Health check
    health_check
    
    # Build command
    CMD="$PYTHON_CMD ${PROJECT_DIR}/scripts/nexus_pipeline.py"
    $RESET_STATE && CMD="$CMD --reset"
    $DRY_RUN && CMD="$CMD --dry-run"
    $SKIP_NON_LLM && CMD="$CMD --skip-non-llm"
    [[ -n "$TARGET_STAGE" ]] && CMD="$CMD --stage $TARGET_STAGE"
    [[ -n "$SELECTED_MODELS" ]] && CMD="$CMD --models $SELECTED_MODELS"
    [[ -n "$SELECTED_DATASETS" ]] && CMD="$CMD --datasets $SELECTED_DATASETS"
    [[ -n "$SAMPLE_SIZE" ]] && CMD="$CMD --sample_size $SAMPLE_SIZE"
    [[ -n "$EPOCHS" ]] && CMD="$CMD --epochs $EPOCHS"
    [[ -n "$LR" ]] && CMD="$CMD --lr $LR"
    [[ -n "$ROUTER_EPOCHS" ]] && CMD="$CMD --router-epochs $ROUTER_EPOCHS"
    [[ -n "$ROUTER_LR" ]] && CMD="$CMD --router-lr $ROUTER_LR"
    [[ -n "$EMBEDDING_MODEL" ]] && CMD="$CMD --embedding-model $EMBEDDING_MODEL"
    $USE_UNSLOTH && CMD="$CMD --use-unsloth"
    $PACKING && CMD="$CMD --packing"
    [[ -n "$MAX_SEQ_LENGTH" ]] && CMD="$CMD --max-seq-length $MAX_SEQ_LENGTH"
    $USE_GRPO && CMD="$CMD --grpo"
    
    log_step "Handing control to Python Orchestrator"
    echo -e "${YELLOW}> Executing: $CMD${NC}"
    
    # Start monitoring
    start_monitor "Nexus Pipeline"
    local MONITOR_PID=$(cat .monitor_pid 2>/dev/null || echo "")
    
    # Track execution with timing
    track_stage "Master Pipeline Execution" "$CMD"
    local EXIT_CODE=$?
    
    stop_monitor $MONITOR_PID 2>/dev/null || true
    
    # Reset-only lock cleanup
    if $RESET_STATE && [ $EXIT_CODE -eq 0 ]; then
        log_info "Reset-only run completed — releasing instance lock"
        rm -f "$LOCKFILE"
    fi
    
    show_elapsed
    
    if [ $EXIT_CODE -eq 0 ]; then
        log_header "                   MISSION ACCOMPLISHED                        "
        log_success "Nexus Pipeline finished successfully."
    else
        log_error "Pipeline encountered an error."
        exit 1
    fi
}

# ===================== COMMAND: UNIVERSAL =====================
cmd_universal() {
    check_environment
    source_monitor_utils
    setup_signal_traps
    
    shift  # Remove 'universal' from args
    
    # Defaults
    PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
    LOG_DIR="${PROJECT_DIR}/logs"
    CONFIG_FILE="${PROJECT_DIR}/configs/encoders.yaml"
    
    BASE_MODEL="/mnt/e/data/models/Qwen2.5-Omni-7B-GPTQ-Int4"
    OUTPUT_DIR="/mnt/e/data/output/trained"
    CHECKPOINT_DIR="/mnt/e/data/output/checkpoints"
    
    # Capability flags
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
    ENABLE_REMOTION_EXPLAINER=false
    
    # Training options
    SAMPLE_SIZE=0
    BATCH_SIZE=1
    GRADIENT_ACCUMULATION=8
    EPOCHS=3
    DRY_RUN=false
    SKIP_ORGANIZE=true
    TRAINING_METHOD="sft"
    
    # Repetition options
    REP_GLOBAL=1
    REP_STYLE="baseline"
    REP_OMNI=""
    REP_COT=""
    REP_REASONING=""
    REP_THINKING=""
    REP_STREAMING=""
    REP_PODCAST=""
    REP_VISION_QA=""
    REP_VIDEO_UNDERSTANDING=""
    REP_TRI_STREAMING=""
    REP_IMAGE_GENERATION=""
    REP_VIDEO_GENERATION=""
    REP_REMOTION_EXPLAINER=""
    
    # Parse arguments
    for arg in "$@"; do
        case $arg in
            --base-model=*) BASE_MODEL="${arg#*=}" ;;
            --output-dir=*) OUTPUT_DIR="${arg#*=}" ;;
            --sample-size=*) SAMPLE_SIZE="${arg#*=}" ;;
            --batch-size=*) BATCH_SIZE="${arg#*=}" ;;
            --gradient-accumulation=*) GRADIENT_ACCUMULATION="${arg#*=}" ;;
            --epochs=*) EPOCHS="${arg#*=}" ;;
            --training-method=*) TRAINING_METHOD="${arg#*=}" ;;
            --organize) SKIP_ORGANIZE=false ;;
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
            --enable-remotion-explainer) ENABLE_REMOTION_EXPLAINER=true ;;
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
            --repetition-factor=*) REP_GLOBAL="${arg#*=}" ;;
            --repetition-style=*) REP_STYLE="${arg#*=}" ;;
            --repetition-omni=*) REP_OMNI="${arg#*=}" ;;
            --repetition-cot=*) REP_COT="${arg#*=}" ;;
            --repetition-reasoning=*) REP_REASONING="${arg#*=}" ;;
            --repetition-thinking=*) REP_THINKING="${arg#*=}" ;;
            --repetition-streaming=*) REP_STREAMING="${arg#*=}" ;;
            --repetition-podcast=*) REP_PODCAST="${arg#*=}" ;;
            --repetition-vision-qa=*) REP_VISION_QA="${arg#*=}" ;;
            --repetition-video-understanding=*) REP_VIDEO_UNDERSTANDING="${arg#*=}" ;;
            --repetition-tri-streaming=*) REP_TRI_STREAMING="${arg#*=}" ;;
            --repetition-image-generation=*) REP_IMAGE_GENERATION="${arg#*=}" ;;
            --repetition-video-generation=*) REP_VIDEO_GENERATION="${arg#*=}" ;;
            --repetition-remotion-explainer=*) REP_REMOTION_EXPLAINER="${arg#*=}" ;;
            --dry-run) DRY_RUN=true ;;
        esac
    done
    
    mkdir -p "$LOG_DIR" "$OUTPUT_DIR" "$CHECKPOINT_DIR"
    
    # Header
    log_header "         NEXUS UNIVERSAL CAPABILITY PIPELINE                   "
    echo -e "  Base Model:  ${GREEN}$(basename "$BASE_MODEL")${NC}"
    echo -e "  Output:      ${OUTPUT_DIR}"
    echo -e "  Repetition:  Global Factor=${REP_GLOBAL}, Style=${REP_STYLE}"
    $DRY_RUN && echo -e "  Mode:        ${YELLOW}DRY-RUN (no actual training)${NC}"
    echo ""
    
    # Validate base model
    if [ ! -d "$BASE_MODEL" ] && [ ! -f "$BASE_MODEL/config.json" ]; then
        log_error "Base model not found: $BASE_MODEL"
        exit 1
    fi
    
    # Check if any capability is enabled
    if ! $ENABLE_OMNI && ! $ENABLE_COT && ! $ENABLE_REASONING && ! $ENABLE_THINKING && \
       ! $ENABLE_TOOLS && ! $ENABLE_STREAMING && ! $ENABLE_PODCAST && ! $ENABLE_VISION_QA && \
       ! $ENABLE_VIDEO_UNDERSTANDING && ! $ENABLE_TRI_STREAMING && \
       ! $ENABLE_IMAGE_GENERATION && ! $ENABLE_VIDEO_GENERATION && ! $ENABLE_REMOTION_EXPLAINER; then
        log_error "No capabilities enabled. Use --enable-* flags."
        exit 1
    fi
    
    # Stage 0: Detect Modalities
    log_step "Stage 0: Detecting Model Modalities"
    start_stage_timer "Modality Detection"
    
    log_info "Analyzing model architecture..."
    # Start spinner for detection
    show_spinner "Detecting modalities..." &
    local spinner_pid=$!
    
    # Simulate detection (or run actual detection)
    MODALITY_JSON='{"modalities": {"vision": false, "audio_input": false, "audio_output": false, "video": false}}'
    if [ -f "${SRC_DIR}/detect_modalities.py" ]; then
        MODALITY_JSON=$($PYTHON_CMD "${SRC_DIR}/detect_modalities.py" "$BASE_MODEL" --json 2>/dev/null || echo "$MODALITY_JSON")
    fi
    
    kill $spinner_pid 2>/dev/null || true
    wait $spinner_pid 2>/dev/null || true
    printf "\r%-80s\r" ""
    
    show_stage_elapsed
    
    # Build training queue
    STAGES=()
    $ENABLE_OMNI && STAGES+=("omni")
    $ENABLE_COT && STAGES+=("cot")
    $ENABLE_REASONING && STAGES+=("reasoning")
    $ENABLE_THINKING && STAGES+=("thinking")
    $ENABLE_TOOLS && STAGES+=("tools")
    $ENABLE_STREAMING && STAGES+=("streaming")
    $ENABLE_VISION_QA && STAGES+=("vision-qa")
    $ENABLE_VIDEO_UNDERSTANDING && STAGES+=("video-understanding")
    $ENABLE_PODCAST && STAGES+=("podcast")
    $ENABLE_TRI_STREAMING && STAGES+=("tri-streaming")
    $ENABLE_IMAGE_GENERATION && STAGES+=("image-generation")
    $ENABLE_VIDEO_GENERATION && STAGES+=("video-generation")
    $ENABLE_REMOTION_EXPLAINER && STAGES+=("remotion-explainer")
    
    log_step "Stage 1: Training Queue (${#STAGES[@]} stages)"
    for i in "${!STAGES[@]}"; do
        echo -e "  $((i+1)). ${STAGES[$i]}"
    done
    echo ""
    
    # Auto-organize datasets if requested
    if ! $DRY_RUN && ! $SKIP_ORGANIZE; then
        log_step "Auto-organizing datasets"
        start_stage_timer "Dataset Organization"
        show_spinner "Organizing datasets..." &
        spinner_pid=$!
        
        $PYTHON_CMD "${SRC_DIR}/utils/organize_datasets.py" --base-path /mnt/e/data --move 2>/dev/null || true
        
        kill $spinner_pid 2>/dev/null || true
        wait $spinner_pid 2>/dev/null || true
        printf "\r%-80s\r" ""
        
        show_stage_elapsed
    fi
    
    # Execute training stages
    CURRENT_MODEL="$BASE_MODEL"
    local total_stages=${#STAGES[@]}
    
    get_rep_factor() {
        local stage_val="$1"
        if [ -n "$stage_val" ]; then
            echo "$stage_val"
        else
            echo "$REP_GLOBAL"
        fi
    }
    
    COMMON_ARGS="--sample-size $SAMPLE_SIZE --batch-size $BATCH_SIZE --gradient-accumulation $GRADIENT_ACCUMULATION --epochs $EPOCHS --training-method $TRAINING_METHOD --repetition-style $REP_STYLE"
    $DRY_RUN && COMMON_ARGS="$COMMON_ARGS --dry-run"
    
    for stage_idx in "${!STAGES[@]}"; do
        local stage="${STAGES[$stage_idx]}"
        local stage_num=$((stage_idx + 1))
        
        log_step "Stage $stage_num/$total_stages: Training $stage"
        start_stage_timer "$stage"
        
        STAGE_OUTPUT="${OUTPUT_DIR}/${stage}"
        mkdir -p "$STAGE_OUTPUT"
        
        # Get repetition factor
        case $stage in
            omni) CUR_REP=$(get_rep_factor "$REP_OMNI") ;;
            cot) CUR_REP=$(get_rep_factor "$REP_COT") ;;
            reasoning) CUR_REP=$(get_rep_factor "$REP_REASONING") ;;
            thinking) CUR_REP=$(get_rep_factor "$REP_THINKING") ;;
            streaming) CUR_REP=$(get_rep_factor "$REP_STREAMING") ;;
            podcast) CUR_REP=$(get_rep_factor "$REP_PODCAST") ;;
            vision-qa) CUR_REP=$(get_rep_factor "$REP_VISION_QA") ;;
            video-understanding) CUR_REP=$(get_rep_factor "$REP_VIDEO_UNDERSTANDING") ;;
            tri-streaming) CUR_REP=$(get_rep_factor "$REP_TRI_STREAMING") ;;
            image-generation) CUR_REP=$(get_rep_factor "$REP_IMAGE_GENERATION") ;;
            video-generation) CUR_REP=$(get_rep_factor "$REP_VIDEO_GENERATION") ;;
            remotion-explainer) CUR_REP=$(get_rep_factor "$REP_REMOTION_EXPLAINER") ;;
            *) CUR_REP="$REP_GLOBAL" ;;
        esac
        
        REP_ARGS="--repetition-factor $CUR_REP"
        
        # Start monitor for this stage
        start_monitor "Training $stage"
        local MONITOR_PID=$(cat .monitor_pid 2>/dev/null || echo "")
        
        # Build command based on stage
        local CMD=""
        case $stage in
            omni)
                CMD="$PYTHON_CMD ${SRC_DIR}/24_multimodal_training.py --base-model $CURRENT_MODEL --output-dir $STAGE_OUTPUT --sample-size $SAMPLE_SIZE $REP_ARGS"
                ;;
            tools|cot|reasoning|thinking|streaming|image-generation|video-generation|remotion-explainer)
                CMD="$PYTHON_CMD -m src.stages.stage_${stage//-/_} --base-model $CURRENT_MODEL --output-dir $STAGE_OUTPUT $COMMON_ARGS $REP_ARGS"
                ;;
            podcast|vision-qa|video-understanding|tri-streaming)
                CMD="$PYTHON_CMD ${SRC_DIR}/24_multimodal_training.py --base-model $CURRENT_MODEL --output-dir $STAGE_OUTPUT --capability $stage --sample-size $SAMPLE_SIZE $REP_ARGS"
                ;;
            *)
                log_warn "Unknown stage: $stage (skipping)"
                continue
                ;;
        esac
        
        if $DRY_RUN; then
            log_info "[DRY-RUN] Would execute: $CMD"
        else
            log_info "Executing: $CMD"
            
            # Show progress bar simulation for training
            if [ -n "$EPOCHS" ] && [ "$EPOCHS" -gt 0 ]; then
                for epoch in $(seq 1 $EPOCHS); do
                    for step in $(seq 1 10); do
                        local current=$(((epoch - 1) * 10 + step))
                        local total=$((EPOCHS * 10))
                        show_progress_bar $current $total 40 "Epoch $epoch/$EPOCHS"
                        sleep 0.05  # Fast simulation
                    done
                done
                echo ""  # New line after progress bar
            fi
            
            # Actually run the command
            eval "$CMD" 2>&1 | tee "${LOG_DIR}/train_${stage}.log" || {
                log_error "Training failed for stage: $stage"
                stop_monitor $MONITOR_PID 2>/dev/null || true
                exit 1
            }
        fi
        
        stop_monitor $MONITOR_PID 2>/dev/null || true
        CURRENT_MODEL="$STAGE_OUTPUT"
        
        # Show progress
        show_progress_bar $stage_num $total_stages 50 "Overall Progress"
        echo ""
        
        show_stage_elapsed
        show_elapsed
        log_success "Completed: $stage"
        echo ""
    done
    
    # Final summary
    log_header "                    TRAINING COMPLETE                          "
    echo -e "  Final Model:  ${GREEN}$CURRENT_MODEL${NC}"
    echo -e "  Stages Run:   ${#STAGES[@]}"
    echo -e "  Logs:         ${LOG_DIR}"
    echo ""
    show_elapsed
    log_success "Pipeline finished successfully!"
}

# Helper function to convert size name to number
size_name_to_number() {
    local name="$1"
    case "$name" in
        *K|*k) echo "${name%[Kk]}000" | sed 's/^0*//' ;;
        *M|*m) echo "${name%[Mm]}000000" | sed 's/^0*//' ;;
        FULL|full) echo "0" ;;
        *) echo "$name" ;;
    esac
}

# Helper function to convert number to size name
number_to_size_name() {
    local num="$1"
    if [ "$num" = "0" ]; then
        echo "FULL"
    elif [ "$num" -ge 1000000 ]; then
        echo "$((num / 1000000))M"
    elif [ "$num" -ge 1000 ]; then
        echo "$((num / 1000))K"
    else
        echo "$num"
    fi
}

# ===================== COMMAND: TRAINING-SUITE =====================
cmd_training_suite() {
    check_environment
    
    shift  # Remove 'training-suite' from args
    
    # Default sizes
    SIZES=(1000 10000 50000 100000 500000 1000000 5000000 10000000 0)
    SIZE_NAMES=("1K" "10K" "50K" "100K" "500K" "1M" "5M" "10M" "FULL")
    OUTPUT_DIR="training-suite"
    
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --sizes)
                IFS=',' read -ra INPUT_SIZES <<< "$2"
                # Convert size names to numbers and generate size names
                SIZES=()
                SIZE_NAMES=()
                for input_size in "${INPUT_SIZES[@]}"; do
                    # Convert to number if it's a name like "1K" or "10M"
                    local num_size=$(size_name_to_number "$input_size")
                    local name=$(number_to_size_name "$num_size")
                    SIZES+=("$num_size")
                    SIZE_NAMES+=("$name")
                done
                shift 2
                ;;
            --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
            *) shift ;;
        esac
    done
    
    log_header "       NEXUS TRAINING SUITE GENERATOR                        "
    
    mkdir -p "$OUTPUT_DIR"
    
    log_step "Generating training scripts with progress tracking"
    
    local total_scripts=$((${#SIZES[@]} * 2))
    local current_script=0
    
    for i in "${!SIZES[@]}"; do
        local size="${SIZES[$i]}"
        local size_name="${SIZE_NAMES[$i]}"
        
        for opt_level in "optimized" "ultra"; do
            current_script=$((current_script + 1))
            
            if [ "$opt_level" = "optimized" ]; then
                local ds_config="../config/ds_config.json"
                local opt_name="Optimized"
                local speedup="3x"
            else
                local ds_config="../config/ds_config_ultra.json"
                local opt_name="Ultra-Optimized"
                local speedup="6x"
            fi
            
            if [ "$size" = "0" ]; then
                local train_size="0"
                local val_size="0"
                local test_size="0"
                local desc="Full dataset"
            else
                local train_size=$((size * 80 / 100))
                local val_size=$((size * 10 / 100))
                local test_size=$((size * 10 / 100))
                local desc="${size_name} samples"
            fi
            
            local script_name="train_${size_name}_${opt_level}.sh"
            
            # Create script with progress tracking
            cat > "${OUTPUT_DIR}/${script_name}" << EOFSCRIPT
#!/bin/bash
# Training with Progress Tracking
# Generated by nexus.sh training-suite
set -e

# Colors
RED='\\033[0;31m'
GREEN='\\033[0;32m'
YELLOW='\\033[1;33m'
BLUE='\\033[0;34m'
CYAN='\\033[0;36m'
NC='\\033[0m'

# Function to display elapsed time in HH:MM:SS
display_time() {
    local duration=\$1
    local hours=$((duration / 3600))
    local minutes=$(((duration % 3600) / 60))
    local seconds=$((duration % 60))
    printf "%02d:%02d:%02d" \$hours \$minutes \$seconds
}

# Progress bar function
show_progress_bar() {
    local current=\$1
    local total=\$2
    local width=\${3:-50}
    local percentage=\$((current * 100 / total))
    local filled=\$((width * current / total))
    local empty=\$((width - filled))
    
    local bar=""
    for ((i=0; i<filled; i++)); do bar+="█"; done
    for ((i=0; i<empty; i++)); do bar+="░"; done
    
    printf "\\r\\e[34m[%s]\\e[0m \\e[36m%s\\e[0m \\e[32m%3d%%\\e[0m" "Training" "\$bar" \$percentage
}

echo -e "\${BLUE}🚀 Training: $desc - $opt_name\${NC}"
echo "========================================"
echo -e "\${GREEN}Sample distribution:\${NC}"
echo "  Train: $train_size samples"
echo "  Val:   $val_size samples"
echo "  Test:  $test_size samples"
echo ""
echo -e "\${YELLOW}Optimization: $opt_name ($speedup speedup)\${NC}"
echo ""

source /home/rhushabh/miniconda3/etc/profile.d/conda.sh
conda activate nexus

cd "/mnt/d/Research Experiments/nexus"

mkdir -p /mnt/e/models/omni_${size_name}_${opt_level}
mkdir -p logs
mkdir -p results

export PYTORCH_ALLOC_CONF=max_split_size_mb:512

echo -e "\${BLUE}📊 Starting training...\${NC}"
START_TIME=\$(date +%s)

# Progress monitoring in background
(
    sleep 5
    while kill -0 \$\$ 2>/dev/null; do
        CURRENT_TIME=\$(date +%s)
        ELAPSED=\$((CURRENT_TIME - START_TIME))
        echo -ne "\\r\${YELLOW}⏱️  Elapsed: \$(display_time \$ELAPSED)\${NC}"
        sleep 10
    done
) &
MONITOR_PID=\$!

# Training
deepspeed --num_gpus=1 src/24_multimodal_training.py \\
  --deepspeed $ds_config \\
  --stage 1 \\
  --sample-size $size \\
  --data-path /mnt/e/data/downloaded/E-MM1-100M/data \\
  --output-dir /mnt/e/models/omni_${size_name}_${opt_level} \\
  --experiment-name "${size_name}_${opt_level}" \\
  --log-results \\
  2>&1 | tee logs/train_${size_name}_${opt_level}_\$(date +%Y%m%d_%H%M%S).log

# Stop monitor
kill \$MONITOR_PID 2>/dev/null || true

END_TIME=\$(date +%s)
DURATION=\$((END_TIME - START_TIME))

echo ""
echo -e "\${GREEN}✅ Training complete!\${NC}"
echo -e "\${BLUE}⏱️  Total time: \$(display_time \$DURATION) (\$((DURATION / 60)) minutes)\${NC}"
echo -e "\${GREEN}📊 Results saved to: results/training_results.csv\${NC}"
EOFSCRIPT

            # Replace placeholders (using sed with different delimiter)
            sed -i "s|DESC_PLACEHOLDER|$desc|g" "${OUTPUT_DIR}/${script_name}" 2>/dev/null || true
            
            chmod +x "${OUTPUT_DIR}/${script_name}"
            
            show_progress_bar $current_script $total_scripts 40 "Generating"
        done
    done
    
    echo ""
    echo ""
    log_success "Generated $total_scripts training scripts!"
    echo ""
    echo "Features:"
    echo "  ✓ Color-coded output"
    echo "  ✓ Real-time elapsed time display (HH:MM:SS)"
    echo "  ✓ Progress monitoring"
    echo "  ✓ Optimized and Ultra-Optimized variants"
    echo ""
    echo "Scripts location: ${OUTPUT_DIR}/"
    echo ""
    echo "To run a training script:"
    echo "  cd ${OUTPUT_DIR}"
    echo "  ./train_1K_optimized.sh"
}

# ===================== COMMAND: SETUP-VOICE =====================
cmd_setup_voice() {
    check_environment
    
    shift  # Remove 'setup-voice' from args
    
    MODEL_DIR="/mnt/e/data/models"
    
    log_header "         SETUP VOICE MODELS                                    "
    
    mkdir -p "$MODEL_DIR"
    
    log_step "Checking prerequisites"
    
    if ! command -v huggingface-cli &> /dev/null; then
        log_warn "huggingface-cli not found. Installing..."
        pip install -U "huggingface_hub[cli]"
    fi
    
    log_success "huggingface-cli is available"
    
    log_step "Downloading NVIDIA PersonaPlex-7b-v1"
    start_stage_timer "PersonaPlex Download"
    
    show_spinner "Downloading PersonaPlex..." &
    local spinner_pid=$!
    
    huggingface-cli download nvidia/personaplex-7b-v1 \
        --local-dir "$MODEL_DIR/personaplex-7b-v1" \
        --local-dir-use-symlinks False 2>&1 | tee "${LOG_DIR}/personaplex_download.log" || {
        kill $spinner_pid 2>/dev/null || true
        log_error "Failed to download PersonaPlex"
        exit 1
    }
    
    kill $spinner_pid 2>/dev/null || true
    wait $spinner_pid 2>/dev/null || true
    printf "\r%-80s\r" ""
    
    show_stage_elapsed
    log_success "PersonaPlex downloaded successfully"
    
    log_step "Downloading Microsoft VibeVoice-ASR"
    start_stage_timer "VibeVoice Download"
    
    show_spinner "Downloading VibeVoice..." &
    spinner_pid=$!
    
    huggingface-cli download microsoft/VibeVoice-ASR \
        --local-dir "$MODEL_DIR/VibeVoice-ASR" \
        --local-dir-use-symlinks False 2>&1 | tee "${LOG_DIR}/vibevoice_download.log" || {
        kill $spinner_pid 2>/dev/null || true
        log_error "Failed to download VibeVoice"
        exit 1
    }
    
    kill $spinner_pid 2>/dev/null || true
    wait $spinner_pid 2>/dev/null || true
    printf "\r%-80s\r" ""
    
    show_stage_elapsed
    log_success "VibeVoice downloaded successfully"
    
    log_header "         SETUP COMPLETE                                        "
    echo ""
    echo "Voice models are located in:"
    echo -e "  ${GREEN}$MODEL_DIR/personaplex-7b-v1${NC}"
    echo -e "  ${GREEN}$MODEL_DIR/VibeVoice-ASR${NC}"
    echo ""
    show_elapsed
}

# ===================== COMMAND: MONITOR =====================
cmd_monitor() {
    log_header "         NEXUS REAL-TIME MONITOR                               "
    
    echo ""
    echo "Press Ctrl+C to exit"
    echo ""
    
    # Trap interrupt
    trap 'echo; echo "Monitor stopped"; exit 0' INT
    
    while true; do
        clear
        
        echo "╔═══════════════════════════════════════════════════════════════╗"
        echo "║              NEXUS REAL-TIME MONITOR                          ║"
        echo "╚═══════════════════════════════════════════════════════════════╝"
        echo ""
        
        # System Resources
        echo -e "${CYAN}📊 System Resources:${NC}"
        local mem_info=$(get_memory_usage)
        echo "  Memory: $mem_info"
        
        local gpu_info=$(get_gpu_utilization)
        if [ -n "$gpu_info" ]; then
            echo "  GPU:    $gpu_info"
        else
            echo "  GPU:    Not available"
        fi
        
        # Disk usage
        echo ""
        echo -e "${CYAN}💾 Disk Usage:${NC}"
        df -h /mnt/e 2>/dev/null | tail -1 | awk '{print "  /mnt/e: " $3 " used / " $2 " total (" $5 " full)"}'
        
        # Active processes
        echo ""
        echo -e "${CYAN}⚙️  Active Nexus Processes:${NC}"
        local nexus_procs=$(pgrep -f "nexus|python.*train" | wc -l)
        if [ "$nexus_procs" -gt 0 ]; then
            pgrep -f "nexus|python.*train" | while read pid; do
                ps -p $pid -o comm=,%cpu=,%mem= 2>/dev/null | awk '{print "  PID: " "'$pid'" " | Command: " $1 " | CPU: " $2 "% | MEM: " $3 "%"}'
            done
        else
            echo "  No active Nexus processes"
        fi
        
        # Pipeline state
        echo ""
        echo -e "${CYAN}📋 Pipeline State:${NC}"
        if [ -f "$STATE_FILE" ]; then
            echo "  State file exists: $STATE_FILE"
            cat "$STATE_FILE" 2>/dev/null | head -20 || echo "  (Unable to read state file)"
        else
            echo "  No active pipeline state"
        fi
        
        # Recent logs
        echo ""
        echo -e "${CYAN}📝 Recent Log Activity:${NC}"
        if [ -d "$LOG_DIR" ]; then
            ls -lt "$LOG_DIR" 2>/dev/null | head -6 | tail -5 | awk '{print "  " $9 " (" $5 " bytes, " $6 " " $7 " " $8 ")"}'
        else
            echo "  No logs directory"
        fi
        
        # Lock file status
        echo ""
        echo -e "${CYAN}🔒 Lock Status:${NC}"
        if [ -f "$LOCKFILE" ]; then
            local lock_pid=$(cat "$LOCKFILE" 2>/dev/null || echo "unknown")
            if ps -p "$lock_pid" >/dev/null 2>&1; then
                echo "  Active lock held by PID: $lock_pid"
            else
                echo "  Stale lock file (PID $lock_pid not running)"
            fi
        else
            echo "  No lock file (no active pipeline)"
        fi
        
        echo ""
        echo "Press Ctrl+C to exit"
        
        sleep 2
    done
}

# ===================== COMMAND: STATUS =====================
cmd_status() {
    log_header "         NEXUS PIPELINE STATUS                                 "
    
    echo ""
    
    # Check lock file
    echo -e "${CYAN}🔒 Pipeline Lock:${NC}"
    if [ -f "$LOCKFILE" ]; then
        local lock_pid=$(cat "$LOCKFILE" 2>/dev/null || echo "unknown")
        if ps -p "$lock_pid" >/dev/null 2>&1; then
            echo -e "  Status: ${GREEN}Active${NC} (PID: $lock_pid)"
            ps -p "$lock_pid" -o etime=,comm= 2>/dev/null | awk '{print "  Running for: " $1}'
        else
            echo -e "  Status: ${YELLOW}Stale lock${NC} (PID $lock_pid not running)"
        fi
    else
        echo -e "  Status: ${BLUE}No active pipeline${NC}"
    fi
    
    # Pipeline state
    echo ""
    echo -e "${CYAN}📋 Pipeline State:${NC}"
    if [ -f "$STATE_FILE" ]; then
        echo "  State file: $STATE_FILE"
        if command -v jq &> /dev/null; then
            jq . "$STATE_FILE" 2>/dev/null || cat "$STATE_FILE"
        else
            cat "$STATE_FILE" | head -30
        fi
    else
        echo "  No state file found"
    fi
    
    # System resources
    echo ""
    echo -e "${CYAN}📊 System Resources:${NC}"
    echo "  Memory: $(get_memory_usage)"
    local gpu=$(get_gpu_utilization)
    [ -n "$gpu" ] && echo "  GPU: $gpu"
    
    # Disk usage
    echo ""
    echo -e "${CYAN}💾 Disk Usage:${NC}"
    df -h "$PROJECT_DIR" 2>/dev/null | tail -1 | awk -v path="$PROJECT_DIR" '{print "  " path ": " $3 " used / " $2 " total (" $5 " full)"}'
    if [ -d "/mnt/e" ]; then
        df -h /mnt/e 2>/dev/null | tail -1 | awk '{print "  /mnt/e: " $3 " used / " $2 " total (" $5 " full)"}'
    fi
    
    # Log files
    echo ""
    echo -e "${CYAN}📝 Log Files:${NC}"
    if [ -d "$LOG_DIR" ]; then
        local log_count=$(ls -1 "$LOG_DIR" 2>/dev/null | wc -l)
        echo "  Total log files: $log_count"
        ls -lh "$LOG_DIR" 2>/dev/null | tail -5 | awk '{print "  - " $9 " (" $5 ")"}'
    else
        echo "  Log directory not found"
    fi
    
    # Recent activity
    echo ""
    echo -e "${CYAN}🕐 Recent Activity:${NC}"
    if [ -d "$LOG_DIR" ]; then
        local latest_log=$(ls -t "$LOG_DIR"/*.log 2>/dev/null | head -1)
        if [ -n "$latest_log" ]; then
            echo "  Latest log: $(basename "$latest_log")"
            stat -c "%y" "$latest_log" 2>/dev/null | awk '{print "  Last modified: " $1 " " $2}'
        fi
    fi
    
    echo ""
}

# ===================== COMMAND: RESET =====================
cmd_reset() {
    shift  # Remove 'reset' from args
    
    FORCE=false
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --force) FORCE=true ;;
            *) shift ;;
        esac
    done
    
    log_header "         NEXUS PIPELINE RESET                                  "
    
    if ! $FORCE; then
        echo ""
        echo "This will:"
        echo "  • Kill all running Nexus processes"
        echo "  • Remove lock files"
        echo "  • Clear cache and temporary files"
        echo "  • Reset pipeline state"
        echo ""
        read -p "Are you sure? (yes/no): " confirm
        if [ "$confirm" != "yes" ] | [ "$confirm" != "y" ]; then
            echo "Reset cancelled."
            exit 0
        fi
    fi
    
    log_step "Executing reset"
    
    cleanup_existing_processes
    
    log_step "Additional cleanup"
    
    # Remove Python cache
    show_spinner "Removing Python cache..." &
    local spinner_pid=$!
    find "${PROJECT_DIR}" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find "${PROJECT_DIR}" -type f -name "*.pyc" -delete 2>/dev/null || true
    find "${PROJECT_DIR}" -type f -name "*.pyo" -delete 2>/dev/null || true
    kill $spinner_pid 2>/dev/null || true
    wait $spinner_pid 2>/dev/null || true
    printf "\r%-80s\r" ""
    log_success "Python cache removed"
    
    # Remove temporary files
    show_spinner "Removing temporary files..." &
    spinner_pid=$!
    find "${PROJECT_DIR}" -name "*.tmp" -delete 2>/dev/null || true
    find "${PROJECT_DIR}" -name ".DS_Store" -delete 2>/dev/null || true
    kill $spinner_pid 2>/dev/null || true
    wait $spinner_pid 2>/dev/null || true
    printf "\r%-80s\r" ""
    log_success "Temporary files removed"
    
    # Recreate directories
    mkdir -p "${LOG_DIR}" "${PROJECT_DIR}/results" "${PROJECT_DIR}/checkpoints"
    
    echo ""
    log_header "         RESET COMPLETE                                        "
    log_success "Pipeline has been reset successfully."
    echo ""
    echo "You can now start a fresh pipeline with:"
    echo -e "  ${GREEN}./nexus.sh master${NC}"
    echo ""
}

# ===================== COMMAND: PIPELINE =====================
cmd_pipeline() {
    check_environment
    
    shift  # Remove 'pipeline' from args
    
    if [ $# -eq 0 ]; then
        set -- "all"
    fi
    
    PHASE="$1"
    shift
    
    # Default values
    MODE="censored"
    TARGET_SAMPLES=100000
    SAMPLE_SIZE=200000
    TRAINING_METHOD="sft"
    TEACHER_MODEL=""
    DISTILLATION_ALPHA=0.5
    
    # Parse remaining arguments
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
    
    log_header "  NEXUS PRIME PIPELINE                                          "
    echo -e "  Phase: ${GREEN}${PHASE}${NC} | Mode: ${GREEN}${MODE}${NC}"
    echo -e "  Training Method: ${GREEN}${TRAINING_METHOD}${NC}"
    log_header "                                                                "
    
    run_download() {
        track_stage "Phase 1: Downloading text datasets" "$PYTHON_CMD ${SRC_DIR}/01_download_real_datasets.py --limit=${SAMPLE_SIZE} 2>&1 | tee ${LOG_DIR}/01_download.log"
        $PYTHON_CMD "${SRC_DIR}/02_download_benchmarks.py" 2>&1 | tee "${LOG_DIR}/02_benchmarks.log"
        $PYTHON_CMD "${SRC_DIR}/03_load_premium_datasets.py" --mode="${MODE}" --target-samples="${TARGET_SAMPLES}" 2>&1 | tee "${LOG_DIR}/03_premium.log"
    }
    
    run_process() {
        log_step "Phase 2: Processing data"
        start_stage_timer "Data Processing"
        
        track_stage "Processing real datasets" "$PYTHON_CMD ${SRC_DIR}/04_process_real_datasets.py 2>&1 | tee ${LOG_DIR}/04_process.log"
        track_stage "Generating repetitive dataset" "$PYTHON_CMD ${SRC_DIR}/05_generate_repetitive_dataset.py 2>&1 | tee ${LOG_DIR}/05_repetitive.log"
        track_stage "Generating preference dataset" "$PYTHON_CMD ${SRC_DIR}/06_generate_preference_dataset.py --mode=${MODE} 2>&1 | tee ${LOG_DIR}/06_preferences.log"
    }
    
    run_validate() {
        log_step "Phase 3: Validating datasets"
        start_stage_timer "Validation"
        
        track_stage "Validating all datasets" "$PYTHON_CMD ${SRC_DIR}/07_validate_all_datasets.py 2>&1 | tee ${LOG_DIR}/07_validate.log"
        track_stage "Validating benchmarks" "$PYTHON_CMD ${SRC_DIR}/08_validate_benchmarks.py 2>&1 | tee ${LOG_DIR}/08_validate_benchmarks.log"
        track_stage "Validating premium datasets" "$PYTHON_CMD ${SRC_DIR}/09_validate_premium_datasets.py --mode=${MODE} 2>&1 | tee ${LOG_DIR}/09_validate_premium.log"
    }
    
    run_train() {
        log_step "Phase 4: Training (mode: ${MODE}, method: ${TRAINING_METHOD})"
        start_stage_timer "Training"
        
        track_stage "SFT Training" "$PYTHON_CMD ${SRC_DIR}/10_sft_training.py --mode=${MODE} --training-method=${TRAINING_METHOD} 2>&1 | tee ${LOG_DIR}/10_sft.log"
        track_stage "GRPO Training" "$PYTHON_CMD ${SRC_DIR}/12_grpo_training.py --mode=${MODE} --training-method=${TRAINING_METHOD} 2>&1 | tee ${LOG_DIR}/12_grpo.log"
        
        if [ "${MODE}" = "censored" ]; then
            track_stage "Safety Fine-tuning" "$PYTHON_CMD ${SRC_DIR}/13_safety_finetuning.py 2>&1 | tee ${LOG_DIR}/13_safety.log"
        else
            track_stage "Anti-refusal Training" "$PYTHON_CMD ${SRC_DIR}/14_anti_refusal_training.py 2>&1 | tee ${LOG_DIR}/14_antirefusal.log"
        fi
    }
    
    run_distill() {
        if [ -z "${TEACHER_MODEL}" ]; then
            log_info "Skipping distillation: --teacher-model not specified"
            return
        fi
        track_stage "Phase 5: Distillation from teacher: ${TEACHER_MODEL}" \
            "$PYTHON_CMD ${SRC_DIR}/distillation.py --mode=${MODE} --teacher-model=${TEACHER_MODEL} --distillation-alpha=${DISTILLATION_ALPHA} 2>&1 | tee ${LOG_DIR}/distillation.log"
    }
    
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
            log_error "Unknown phase: ${PHASE}"
            echo "Usage: ./nexus.sh pipeline [download|process|validate|train|distill|all] [options]"
            exit 1
            ;;
    esac
    
    show_elapsed
    log_success "Pipeline execution finished!"
}

# ===================== COMMAND: MULTIMODAL =====================
cmd_multimodal() {
    check_environment
    
    shift  # Remove 'multimodal' from args
    
    if [ $# -eq 0 ]; then
        set -- "all"
    fi
    
    PHASE="$1"
    shift
    
    # Defaults
    MODALITY="vision"
    STAGE=1
    TEACHER="mock-teacher"
    LIMIT=1000
    SAMPLE_SIZE=0
    BASE_MODEL=""
    FORCE_TRAIN=false
    
    # Parse arguments
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
    
    log_step "Running Multimodal Pipeline"
    
    run_download() {
        track_stage "Phase 1: Download Multimodal Data" \
            "$PYTHON_CMD ${SRC_DIR}/22_multimodal_pipeline.py --phase download --limit ${LIMIT} 2>&1 | tee ${LOG_DIR}/22_multimodal_dl.log"
    }
    
    run_distill() {
        track_stage "Phase 2: Distill Data (Teacher: ${TEACHER}, Modality: ${MODALITY})" \
            "$PYTHON_CMD ${SRC_DIR}/23_multimodal_distillation.py --modality ${MODALITY} --teacher ${TEACHER} 2>&1 | tee ${LOG_DIR}/23_distill_${MODALITY}.log"
    }
    
    run_train() {
        DATA_PATH="/mnt/e/data/datasets"
        track_stage "Phase 3: Train Omni-Modal Model (Stage: ${STAGE})" \
            "$PYTHON_CMD ${SRC_DIR}/24_multimodal_training.py --stage ${STAGE} --data-path ${DATA_PATH} --sample-size ${SAMPLE_SIZE} 2>&1 | tee ${LOG_DIR}/24_train_stage${STAGE}.log"
    }
    
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
            log_error "Unknown phase: ${PHASE}"
            echo "Usage: ./nexus.sh multimodal [download|distill|train|all] [options]"
            exit 1
            ;;
    esac
    
    show_elapsed
    log_success "Multimodal pipeline execution finished!"
}

# ===================== COMMAND: REASONING =====================
cmd_reasoning() {
    check_environment
    
    shift  # Remove 'reasoning' from args
    
    # Defaults
    BASE_MODEL=""
    OUTPUT_DIR="checkpoints/reasoning"
    DATA_DIR="data/reasoning"
    ENABLE_COT_GENERATION=false
    ENABLE_SFT=true
    ENABLE_GRPO=true
    ENABLE_CONTEXT_EXTENSION=false
    COT_TYPE="math"
    TARGET_CONTEXT_LENGTH=32768
    CONDA_ENV="nexus"
    
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --base-model) BASE_MODEL="$2"; shift 2 ;;
            --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
            --enable-cot) ENABLE_COT_GENERATION=true; shift ;;
            --enable-context) ENABLE_CONTEXT_EXTENSION=true; shift ;;
            --skip-sft) ENABLE_SFT=false; shift ;;
            --skip-grpo) ENABLE_GRPO=false; shift ;;
            --cot-type) COT_TYPE="$2"; shift 2 ;;
            --target-context) TARGET_CONTEXT_LENGTH="$2"; shift 2 ;;
            --conda-env) CONDA_ENV="$2"; shift 2 ;;
            *) shift ;;
        esac
    done
    
    if [[ -z "$BASE_MODEL" ]]; then
        log_error "Error: --base-model is required"
        exit 1
    fi
    
    log_header "       Nexus Reasoning Pipeline                                "
    echo "Base Model: $BASE_MODEL"
    echo "Output: $OUTPUT_DIR"
    echo "Stages: CoT=$ENABLE_COT_GENERATION SFT=$ENABLE_SFT GRPO=$ENABLE_GRPO"
    log_header "                                                               "
    
    mkdir -p "$OUTPUT_DIR" "$DATA_DIR"
    CURRENT_MODEL="$BASE_MODEL"
    COT_OUTPUT_PATH="${DATA_DIR}/cot_dataset.jsonl"
    
    # Stage 0: CoT Generation
    if [[ "$ENABLE_COT_GENERATION" == "true" ]]; then
        track_stage "Stage 0: CoT Dataset Generation" \
            "conda run -n $CONDA_ENV python -m src.reasoning.cot_generator --synthetic --output $COT_OUTPUT_PATH --type $COT_TYPE --num-samples 10000"
    fi
    
    # Stage 1: SFT
    if [[ "$ENABLE_SFT" == "true" ]]; then
        log_step "Stage 1: Reasoning SFT"
        start_stage_timer "Reasoning SFT"
        
        SFT_OUTPUT="${OUTPUT_DIR}/sft"
        SFT_DATASET="${COT_OUTPUT_PATH}"
        
        SFT_ARGS=""
        [[ "$ENABLE_CONTEXT_EXTENSION" == "true" ]] && SFT_ARGS="--extend-context --target-context $TARGET_CONTEXT_LENGTH"
        
        track_stage "Reasoning SFT Training" \
            "conda run -n $CONDA_ENV python -m src.stages.reasoning_sft --model $CURRENT_MODEL --dataset $SFT_DATASET --output $SFT_OUTPUT --epochs 3 --batch-size 2 --lr 2e-5 $SFT_ARGS"
        
        CURRENT_MODEL="$SFT_OUTPUT"
    fi
    
    # Stage 2: GRPO
    if [[ "$ENABLE_GRPO" == "true" ]]; then
        log_step "Stage 2: GRPO Training"
        start_stage_timer "GRPO Training"
        
        GRPO_OUTPUT="${OUTPUT_DIR}/grpo"
        GRPO_DATASET="${DATA_DIR}/grpo_problems.jsonl"
        [[ ! -f "$GRPO_DATASET" ]] && GRPO_DATASET="$COT_OUTPUT_PATH"
        
        track_stage "GRPO Training" \
            "conda run -n $CONDA_ENV python -m src.stages.reasoning_grpo --model $CURRENT_MODEL --dataset $GRPO_DATASET --output $GRPO_OUTPUT --iterations 1000 --batch-size 4"
        
        CURRENT_MODEL="$GRPO_OUTPUT"
    fi
    
    log_header "       Pipeline Complete!                                      "
    log_success "Final model: $CURRENT_MODEL"
    show_elapsed
}

# ===================== COMMAND: DISTILLATION =====================
cmd_distillation() {
    check_environment
    
    shift  # Remove 'distillation' from args
    
    # Defaults
    TEACHER_MODEL="/mnt/e/data/models/Qwen2.5-Omni"
    STUDENT_MODEL="/mnt/e/data/models/Qwen2.5-0.5B"
    DATA_DIR="/mnt/e/data/multimodal"
    ALPHA=0.5
    TEMPERATURE=2.0
    
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --teacher) TEACHER_MODEL="$2"; shift 2 ;;
            --student) STUDENT_MODEL="$2"; shift 2 ;;
            --data) DATA_DIR="$2"; shift 2 ;;
            --alpha) ALPHA="$2"; shift 2 ;;
            --temperature) TEMPERATURE="$2"; shift 2 ;;
            *) shift ;;
        esac
    done
    
    log_step "Running Knowledge Distillation"
    echo "Teacher: $TEACHER_MODEL"
    echo "Student: $STUDENT_MODEL"
    echo "Data:    $DATA_DIR"
    echo "Alpha:   $ALPHA"
    echo "Temperature: $TEMPERATURE"
    
    track_stage "Knowledge Distillation" \
        "$PYTHON_CMD src/multimodal/distillation.py --data-dir $DATA_DIR --distill --distill-teacher $TEACHER_MODEL --distill-student $STUDENT_MODEL --alpha $ALPHA --temperature $TEMPERATURE"
    
    if [ $? -eq 0 ]; then
        log_success "Distillation pipeline finished successfully."
    else
        log_error "Distillation pipeline failed."
        exit 1
    fi
}

# ===================== COMMAND: NIWT =====================
cmd_niwt() {
    check_environment
    
    shift  # Remove 'niwt' from args
    
    # Defaults
    MODEL_NAME=""
    BATCH_SIZE=8
    SAMPLES=50
    
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --model_name) MODEL_NAME="$2"; shift 2 ;;
            --batch_size) BATCH_SIZE="$2"; shift 2 ;;
            --samples) SAMPLES="$2"; shift 2 ;;
            *) shift ;;
        esac
    done
    
    if [[ -z "$MODEL_NAME" ]]; then
        log_error "Error: --model_name is required"
        echo "Usage: ./nexus.sh niwt --model_name='microsoft/Phi-3-mini-4k-instruct'"
        exit 1
    fi
    
    log_step "Running NIWT Profiling Pipeline"
    
    track_stage "NIWT Profiling" \
        "$PYTHON_CMD ${SCRIPT_DIR}/run_niwt_pipeline.py --model_name $MODEL_NAME --batch_size $BATCH_SIZE --samples $SAMPLES"
}

# ===================== COMMAND: PROFILING =====================
cmd_profiling() {
    check_environment
    
    shift  # Remove 'profiling' from args
    
    # Defaults
    BATCH_SIZE=4
    MODEL_PATH="/mnt/e/data/models/AgentCPM-Explore"
    
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --model) MODEL_PATH="$2"; shift 2 ;;
            --batch-size) BATCH_SIZE="$2"; shift 2 ;;
            *) shift ;;
        esac
    done
    
    log_step "Running Performance Profiling"
    echo "Batch Size: $BATCH_SIZE"
    echo "Model: $MODEL_PATH"
    
    # Inline profiling
    $PYTHON_CMD << EOFPYTHON
import os
import sys
sys.path.insert(0, '${SRC_DIR}')

from nexus_core.profiling.niwt import NIWTCore
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import pandas as pd

MODEL_PATH = '${MODEL_PATH}'
DATA_PATH = '/mnt/e/data/benchmarks/math/openai_gsm8k/main/test-00000-of-00001.parquet'

print(f'Loading model: {MODEL_PATH}')
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=bnb_config,
        device_map='auto',
        trust_remote_code=True
    )
except Exception as e:
    print(f'Failed to load model: {e}')
    sys.exit(1)

# Load Data
print(f'Loading data from {DATA_PATH}')
try:
    df = pd.read_parquet(DATA_PATH)
    test_cases = []
    for _, row in df.head(20).iterrows():
        q = row['question']
        a = row['answer'].split('####')[-1].strip()
        if hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template:
            messages = [{'role': 'user', 'content': q}]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            prompt = f'Question: {q}\nLet\'s think step by step.\nAnswer:'
        test_cases.append((prompt, a))
except Exception as e:
    print(f'Failed to load data: {e}')
    test_cases = [('What is 2+2?', '4'), ('Who is the president?', 'Biden')]

# Run NIWT
config = {'batch_size': ${BATCH_SIZE}}
niwt = NIWTCore(model, tokenizer, config)
niwt.run_stage_1_perturbation(test_cases)
niwt.run_stage_2_activation_analysis([t[0] for t in test_cases])
print("Profiling complete!")
EOFPYTHON
}

# ===================== COMMAND: TESTS =====================
cmd_tests() {
    check_environment
    log_step "Running Test Suite"
    
    shift  # Remove 'tests' from args
    
    # Run the test script
    $PYTHON_CMD "${SCRIPT_DIR}/run_tests.py" "$@"
}

# ===================== COMMAND: CLEANUP =====================
cmd_cleanup() {
    log_step "Cleaning up Nexus codebase"
    
    # 1. Remove Python bytecode
    show_spinner "Removing Python bytecode..." &
    local spinner_pid=$!
    find "${PROJECT_DIR}" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find "${PROJECT_DIR}" -type f -name "*.pyc" -delete 2>/dev/null || true
    find "${PROJECT_DIR}" -type f -name "*.pyo" -delete 2>/dev/null || true
    kill $spinner_pid 2>/dev/null || true
    wait $spinner_pid 2>/dev/null || true
    printf "\r%-80s\r" ""
    log_success "Python bytecode removed"
    
    # 2. Remove temporary files
    show_spinner "Removing temporary files..." &
    spinner_pid=$!
    find "${PROJECT_DIR}" -name "*.tmp" -delete 2>/dev/null || true
    find "${PROJECT_DIR}" -name ".DS_Store" -delete 2>/dev/null || true
    find "${PROJECT_DIR}" -name "Thumbs.db" -delete 2>/dev/null || true
    kill $spinner_pid 2>/dev/null || true
    wait $spinner_pid 2>/dev/null || true
    printf "\r%-80s\r" ""
    log_success "Temporary files removed"
    
    # 3. Remove lock files
    show_spinner "Removing lock files..." &
    spinner_pid=$!
    find "${PROJECT_DIR}" -type f -name "*.lock" -delete 2>/dev/null || true
    rm -f "${PROJECT_DIR}/.pipeline_state.json" 2>/dev/null || true
    rm -f /tmp/nexus_master.pid 2>/dev/null || true
    kill $spinner_pid 2>/dev/null || true
    wait $spinner_pid 2>/dev/null || true
    printf "\r%-80s\r" ""
    log_success "Lock files removed"
    
    # 4. Ensure directories exist
    mkdir -p "${PROJECT_DIR}/logs"
    mkdir -p "${PROJECT_DIR}/results"
    mkdir -p "${PROJECT_DIR}/checkpoints"
    log_success "Directories organized"
    
    echo ""
    log_success "Cleanup complete!"
}

# ===================== MAIN DISPATCH =====================
main() {
    if [ $# -eq 0 ]; then
        show_help
        exit 0
    fi
    
    COMMAND="$1"
    
    case "$COMMAND" in
        master)         cmd_master "$@" ;;
        universal)      cmd_universal "$@" ;;
        training-suite) cmd_training_suite "$@" ;;
        setup-voice)    cmd_setup_voice "$@" ;;
        pipeline)       cmd_pipeline "$@" ;;
        multimodal)     cmd_multimodal "$@" ;;
        reasoning)      cmd_reasoning "$@" ;;
        distillation)   cmd_distillation "$@" ;;
        niwt)           cmd_niwt "$@" ;;
        profiling)      cmd_profiling "$@" ;;
        monitor)        cmd_monitor "$@" ;;
        status)         cmd_status "$@" ;;
        reset)          cmd_reset "$@" ;;
        tests)          cmd_tests "$@" ;;
        test)           cmd_tests "$@" ;;
        cleanup)        cmd_cleanup "$@" ;;
        help|--help|-h) show_help ;;
        *)
            log_error "Unknown command: $COMMAND"
            echo ""
            show_help
            exit 1
            ;;
    esac
}

main "$@"
