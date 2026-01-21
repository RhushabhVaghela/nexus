#!/usr/bin/env python3
"""
FILE: 00_environment_setup.sh
Purpose: Install all dependencies for Advanced Nexus Pipeline on RTX 5080
Runtime: ~10 minutes
Output: Conda environment 'nexus_training' with all dependencies
"""

#!/bin/bash
set -e

echo "═════════════════════════════════════════════════════════════════"
echo "🔧 Advanced Nexus 1.6 Max - Environment Setup for RTX 5080"
echo "═════════════════════════════════════════════════════════════════"
echo ""

# Check CUDA is available
echo "✓ Checking CUDA installation..."
if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ NVIDIA CUDA toolkit not found. Install CUDA 12.4 first."
    echo "   Download from: https://developer.nvidia.com/cuda-downloads"
    exit 1
fi

CUDA_VERSION=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
echo "✓ NVIDIA GPU detected: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "✓ CUDA Capability: $CUDA_VERSION"
echo ""

# Create conda environment
echo "📦 Creating conda environment 'nexus_training'..."
conda create -n nexus_training python=3.11 -y
conda activate nexus_training

echo "✓ Conda environment created"
echo ""

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip setuptools wheel -q
echo "✓ Pip upgraded"
echo ""

# PyTorch (CUDA 12.4 optimized)
echo "🔥 Installing PyTorch 2.4.0 with CUDA 12.4 support..."
pip install -q torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 \
    --index-url https://download.pytorch.org/whl/cu124
echo "✓ PyTorch installed"
echo ""

# Core ML libraries
echo "📚 Installing core ML libraries..."
pip install -q \
    transformers==4.42.0 \
    datasets==2.19.0 \
    peft==0.13.0 \
    bitsandbytes==0.43.0 \
    accelerate==0.32.0 \
    numpy==1.24.0
echo "✓ Core libraries installed"
echo ""

# Unsloth (optimized training)
echo "⚡ Installing Unsloth (optimized training)..."
pip install -q git+https://github.com/unslothai/unsloth.git
pip install -q xformers!=0.0.26
echo "✓ Unsloth installed"
echo ""

# TRL (GRPO, DPO)
echo "🎓 Installing TRL (training reward optimization)..."
pip install -q git+https://github.com/huggingface/trl.git@main
echo "✓ TRL installed"
echo ""

# Evaluation infrastructure
echo "📊 Installing evaluation frameworks..."
pip install -q \
    lm-eval==0.4.2 \
    deepeval==0.21.0 \
    vllm==0.6.3 \
    text-generation==0.17.0
echo "✓ Evaluation frameworks installed"
echo ""

# Utilities
echo "🛠️  Installing utility libraries..."
pip install -q \
    wandb==0.17.0 \
    requests==2.32.0 \
    tqdm==4.66.0 \
    openai==1.42.0 \
    pyyaml==6.0.1 \
    pandas==2.1.0 \
    matplotlib==3.8.0 \
    seaborn==0.13.0 \
    scikit-learn==1.3.0 \
    aiohttp==3.9.0
echo "✓ Utilities installed"
echo ""

# Development tools
echo "🔨 Installing development tools..."
pip install -q \
    jupyterlab==4.0.0 \
    ipython==8.18.0 \
    black==23.12.0 \
    pylint==3.0.0 \
    pytest==7.4.0
echo "✓ Development tools installed"
echo ""

# Create directories
echo "📁 Creating project structure..."
mkdir -p data/benchmarks data/trajectories evaluation_results
mkdir -p checkpoints/stage1_sft checkpoints/stage3_grpo checkpoints/stage4_tool_integration
mkdir -p logs configs utils
echo "✓ Directory structure created"
echo ""

# Verify installation
echo "✅ Verifying installation..."
python -c "import torch; print(f'✓ PyTorch version: {torch.__version__}')" 2>/dev/null || echo "❌ PyTorch verification failed"
python -c "import transformers; print(f'✓ Transformers version: {transformers.__version__}')" 2>/dev/null || echo "❌ Transformers verification failed"
python -c "import torch; print(f'✓ CUDA available: {torch.cuda.is_available()}')" 2>/dev/null || echo "❌ CUDA not available"
python -c "import torch; print(f'✓ GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')" 2>/dev/null || echo "❌ GPU detection failed"
python -c "from unsloth import FastLanguageModel; print(f'✓ Unsloth working')" 2>/dev/null || echo "❌ Unsloth verification failed"

echo ""
echo "═════════════════════════════════════════════════════════════════"
echo "✅ Environment setup complete!"
echo "═════════════════════════════════════════════════════════════════"
echo ""
echo "To activate environment:"
echo "  conda activate nexus_training"
echo ""
echo "Next steps:"
echo "  1. Set API key: export OPENAI_API_KEY='sk-...'"
echo "  2. Download benchmarks: python 01_download_benchmarks.py"
echo "  3. Generate trajectories: python 02_generate_trajectories.py"
echo ""
