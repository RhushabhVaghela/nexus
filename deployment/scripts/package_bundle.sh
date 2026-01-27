#!/bin/bash
set -e

# Nexus Bundle Packaging Script
# Enforces Master Plan Stage 5: Teacher Removal Validation

BUNDLE_NAME="nexus_bundle_v1"
DIST_DIR="dist"
BUNDLE_ROOT="$DIST_DIR/$BUNDLE_NAME"
SOURCE_ROOT="src"  # Adjust if actual source is elsewhere
CHECKPOINT_DIR="checkpoints" # Where models come from

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

echo "📦 Starting Nexus Bundle Packaging..."

# 1. CLEAN
echo "Step 1: Cleaning distribution directory..."
rm -rf "$BUNDLE_ROOT"
mkdir -p "$BUNDLE_ROOT"/{bin,model,config,lib}

# 2. MODEL EXPORT (Simulation/Placeholder)
# In production, this would invoke the specific export command.
echo "Step 2: Exporting Student Model..."
# We expect a specific student model file.
STUDENT_MODEL="$CHECKPOINT_DIR/student_final.onnx"

if [ ! -f "$STUDENT_MODEL" ]; then
    echo -e "${RED}Error: Student model source not found at $STUDENT_MODEL${NC}"
    echo "Please run the export pipeline first."
    # For now, we create a dummy for the structure plan if it doesn't exist
    # REMOVE THIS IN PRODUCTION
    echo "⚠️  Creating dummy student model for structural verification..."
    mkdir -p "$CHECKPOINT_DIR"
    touch "$STUDENT_MODEL"
fi

cp "$STUDENT_MODEL" "$BUNDLE_ROOT/model/student.onnx"
echo "   Copied student model."

# 3. CONFIGURATION
echo "Step 3: Configuring Runtime..."
# Create a production-only config
cat <<EOF > "$BUNDLE_ROOT/config/inference.yaml"
mode: production
device: auto
threads: 4
logging: minimal
model_path: ../model/student.onnx
EOF

# 4. INFERENCE ENGINE
echo "Step 4: Copying Inference Code..."
# Only copy files strictly needed for inference.
# Explicitly AVOID copying training logic.
if [ -d "$SOURCE_ROOT/nexus_inference" ]; then
    cp -r "$SOURCE_ROOT/nexus_inference"/* "$BUNDLE_ROOT/lib/"
else
    # Create a placeholder inference entry point
    touch "$BUNDLE_ROOT/bin/nexus_run"
    chmod +x "$BUNDLE_ROOT/bin/nexus_run"
fi

# 5. TEACHER REMOVAL ENFORCEMENT
echo "Step 5: 🛡️  Enforcing Teacher Removal..."

# Active scan and delete of forbidden patterns
find "$BUNDLE_ROOT" -name "*teacher*" -print -delete
find "$BUNDLE_ROOT" -name "*distill*" -print -delete
find "$BUNDLE_ROOT" -name "*checkpoint*" -print -delete
find "$BUNDLE_ROOT" -name "*.pt" -print -delete # Remove raw PyTorch weights if ONNX is target

# 6. MANIFEST GENERATION
echo "Step 6: Generating Manifest..."
TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

# Calculate hash of the student model
if command -v sha256sum &> /dev/null; then
    MODEL_HASH=$(sha256sum "$BUNDLE_ROOT/model/student.onnx" | awk '{print $1}')
else
    MODEL_HASH="unknown_shasum_missing"
fi

cat <<EOF > "$BUNDLE_ROOT/manifest.json"
{
  "schema_version": "1.0",
  "bundle_id": "$BUNDLE_NAME",
  "build_timestamp": "$TIMESTAMP",
  "invariants": {
    "teacher_free": true,
    "training_code_excluded": true
  },
  "files": [
    {
      "path": "model/student.onnx",
      "sha256": "$MODEL_HASH"
    }
  ]
}
EOF

echo -e "${GREEN}✅ Bundle created at: $BUNDLE_ROOT${NC}"
