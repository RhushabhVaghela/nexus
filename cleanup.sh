#!/bin/bash
# Codebase Cleanup Script

cd "$(dirname "$0")"

echo "🧹 Cleaning codebase..."
echo "======================="

# 1. Remove Python bytecode
echo "Removing Python bytecode..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -type f -name "*.pyc" -delete 2>/dev/null
find . -type f -name "*.pyo" -delete 2>/dev/null
echo "✓ Python bytecode removed"

# 2. Remove old launch scripts (now in training-suite)
echo "Removing old launch scripts..."
rm -f launch_any2any_training.sh 2>/dev/null
rm -f launch_optimized.sh 2>/dev/null
rm -f launch_ultra_optimized.sh 2>/dev/null
echo "✓ Old launch scripts removed"

# 3. Remove duplicate/temporary files
echo "Removing temporary files..."
find . -name "*.tmp" -delete 2>/dev/null
find . -name ".DS_Store" -delete 2>/dev/null
find . -name "Thumbs.db" -delete 2>/dev/null
echo "✓ Temporary files removed"

# 4. Create/organize directories
echo "Organizing directory structure..."
mkdir -p logs
mkdir -p results
mkdir -p checkpoints
mkdir -p training-suite
echo "✓ Directories organized"

# 5. Remove old artifact processor (if exists)
echo "Removing obsolete files..."
rm -f src/additional_processors.py 2>/dev/null
echo "✓ Obsolete files removed"

# 6. Organize config files
echo "Organizing config files..."
mkdir -p config
mv ds_config.json config/ 2>/dev/null
mv ds_config_ultra.json config/ 2>/dev/null
echo "✓ Config files organized"

echo ""
echo "✅ Cleanup complete!"
echo ""
echo "Project structure:"
tree -L 2 -I '__pycache__|*.pyc' . 2>/dev/null || find . -maxdepth 2 -type d | head -20
