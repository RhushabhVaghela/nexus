#!/usr/bin/env python3
"""
Debug script to diagnose the tokenizer loading issue for Gemma Scope models.
"""

import os
from pathlib import Path

model_path = "/mnt/e/data/models/google_gemma-scope-2-27b-pt"

print("=" * 70)
print("DEBUG: Tokenizer Loading Issue for Gemma Scope Models")
print("=" * 70)

# 1. Check what files exist in the model path
print(f"\n1. Model path: {model_path}")
print(f"   Path exists: {os.path.exists(model_path)}")

if os.path.exists(model_path):
    files = os.listdir(model_path)
    print(f"\n2. Files in model directory:")
    for f in sorted(files):
        print(f"   - {f}")
    
    # Check for tokenizer files
    tokenizer_files = [
        "tokenizer.json", "tokenizer_config.json", "tokenizer.model",
        "spiece.model", "vocab.json", "sentencepiece.bpe.model",
        "config.json"
    ]
    
    print(f"\n3. Tokenizer-related files check:")
    found_any = False
    for tf in tokenizer_files:
        exists = os.path.exists(os.path.join(model_path, tf))
        status = "✓ EXISTS" if exists else "✗ MISSING"
        print(f"   {tf}: {status}")
        if exists:
            found_any = True
    
    if not found_any:
        print("\n   ⚠️  NO TOKENIZER FILES FOUND!")
        print("   This is a Gemma Scope SAE model, not a full language model.")

# 2. Try to load the tokenizer and see the exact error
print(f"\n4. Attempting to load tokenizer...")
try:
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    print("   SUCCESS: Tokenizer loaded!")
except Exception as e:
    print(f"   ERROR: {type(e).__name__}: {e}")
    import traceback
    print(f"\n   Full traceback:")
    traceback.print_exc()

# 3. Suggest the correct base model
print("\n" + "=" * 70)
print("DIAGNOSIS:")
print("=" * 70)
print("""
The path '/mnt/e/data/models/google_gemma-scope-2-27b-pt' contains:
- SAE (Sparse AutoEncoder) interpretability models
- These are used for analyzing Gemma model activations
- NO tokenizer files are present

To load a tokenizer for this model, you need to use the BASE MODEL:
  - google/gemma-2-27b (or google/gemma-3-27b-pt as referenced in configs)

SOLUTION OPTIONS:
1. Pass the base model name as the tokenizer source
2. Check if model path is a Scope/SAE model and handle accordingly
3. Load tokenizer from the base model referenced in SAE configs
""")
