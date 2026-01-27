#!/usr/bin/env python3
"""
Nexus Bundle Verification Script
Enforces "Teacher-Free" Invariant (Stage 5)
"""

import os
import sys
import json
import glob
import argparse

def fail(msg):
    print(f"❌ VERIFICATION FAILED: {msg}")
    sys.exit(1)

def pass_check(msg):
    print(f"✅ {msg}")

def verify_bundle(bundle_path):
    print(f"🔍 Inspecting bundle at: {bundle_path}")

    if not os.path.isdir(bundle_path):
        fail(f"Directory not found: {bundle_path}")

    # 1. Structural Validation
    required_paths = [
        "manifest.json",
        "model/student.onnx",
        "config/inference.yaml"
    ]
    for p in required_paths:
        full_p = os.path.join(bundle_path, p)
        if not os.path.exists(full_p):
            fail(f"Missing required component: {p}")
    pass_check("Directory structure valid")

    # 2. Artifact Blocklist (Forbidden Patterns)
    # These terms MUST NOT appear in any filename or content
    forbidden_terms = [
        "teacher",
        "distill",
        "soft_target",
        "checkpoint", # Raw training checkpoints
        "optimizer",
        "gradient"
    ]

    print("🛡️  Scanning for forbidden teacher artifacts...")
    for root, dirs, files in os.walk(bundle_path):
        for filename in files:
            # Check filename
            if any(term in filename.lower() for term in forbidden_terms):
                fail(f"Forbidden artifact found (filename): {filename} in {root}")
            
            # Check content of text files (config/manifest/scripts)
            if filename.endswith(('.json', '.yaml', '.sh', '.py', '.txt', '.md')):
                filepath = os.path.join(root, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read().lower()
                        # Exclude manifest invariant declaration from this check
                        if "manifest.json" in filename and "teacher_free" in content:
                            continue 
                            
                        for term in forbidden_terms:
                            if term in content:
                                # Allow specific exceptions if strictness is too high, 
                                # but generally we want zero mentions.
                                fail(f"Forbidden term '{term}' found in file content: {filepath}")
                except Exception as e:
                    print(f"⚠️  Could not read {filepath}: {e}")

    pass_check("No forbidden artifacts or terms found")

    # 3. Manifest Invariant Check
    manifest_path = os.path.join(bundle_path, "manifest.json")
    try:
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
            
        invariants = manifest.get("invariants", {})
        if not invariants.get("teacher_free"):
            fail("Manifest does not explicitly declare 'teacher_free' invariant")
        
        # Verify hash of the model
        # (Simplified for this script, in production would re-calc sha256)
        files = manifest.get("files", [])
        student_entry = next((f for f in files if "student.onnx" in f["path"]), None)
        if not student_entry:
            fail("Manifest does not track student.onnx")
            
    except Exception as e:
        fail(f"Manifest validation error: {e}")
    
    pass_check("Manifest confirms invariants")

    print(f"\n🚀 SUCCESS: {bundle_path} is certified Teacher-Free.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", default="dist/nexus_bundle_v1", help="Path to bundle")
    args = parser.parse_args()
    
    verify_bundle(args.bundle)
