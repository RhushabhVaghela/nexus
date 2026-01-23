#!/bin/bash
# scripts/setup_voice_models.sh
# Downloads PersonaPlex and VibeVoice models to /mnt/e/data/models

MODEL_DIR="/mnt/e/data/models"
mkdir -p "$MODEL_DIR"

echo "Checking for huggingface-cli..."
if ! command -v huggingface-cli &> /dev/null
then
    echo "huggingface-cli not found. Installing..."
    pip install -U "huggingface_hub[cli]"
fi

echo "--- Downloading NVIDIA PersonaPlex-7b-v1 ---"
# We primarily need the weights and the voice assets
huggingface-cli download nvidia/personaplex-7b-v1 --local-dir "$MODEL_DIR/personaplex-7b-v1" --local-dir-use-symlinks False

echo "--- Downloading Microsoft VibeVoice-ASR ---"
# Note: VibeVoice-ASR is the expressive synthesis/ASR model
huggingface-cli download microsoft/VibeVoice-ASR --local-dir "$MODEL_DIR/VibeVoice-ASR" --local-dir-use-symlinks False

echo "Done! Models are located in $MODEL_DIR"
