#!/bin/bash
# Script to run knowledge distillation for multimodal datasets

# Defaults
TEACHER_MODEL="/mnt/e/data/models/Qwen2.5-Omni"
STUDENT_MODEL="/mnt/e/data/models/Qwen2.5-0.5B"
DATA_DIR="/mnt/e/data/multimodal"

# Parse optional arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --teacher) TEACHER_MODEL="$2"; shift ;;
        --student) STUDENT_MODEL="$2"; shift ;;
        --data) DATA_DIR="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

echo "Starting Multimodal Knowledge Distillation..."
echo "Teacher: $TEACHER_MODEL"
echo "Student: $STUDENT_MODEL"
echo "Data:    $DATA_DIR"

# Ensure src is in python path
export PYTHONPATH=$PYTHONPATH:$(pwd)/src

python src/multimodal/distillation.py \
    --data-dir "$DATA_DIR" \
    --distill \
    --distill-teacher "$TEACHER_MODEL" \
    --distill-student "$STUDENT_MODEL"

if [ $? -eq 0 ]; then
    echo "✅ Distillation pipeline finished successfully."
else
    echo "❌ Distillation pipeline failed."
    exit 1
fi
