import subprocess
import os
import sys

# Define datasets to test in order
datasets = [
    # Premium Text
    ("premium_text", "fineweb-edu", "HF"),
    ("premium_text", "cosmopedia", "HF"),
    ("premium_text", "code_alpaca", "HF"),
    
    # Vision
    ("vision", "websight", "HF"),
    ("vision", "llava_instruct", "HF"),
    
    # Audio (using HF for all, expecting CommonVoice failure or empty)
    ("audio", "librispeech", "HF"),
    ("audio", "common_voice", "HF/Blocked"),
    
    # Video
    ("video", "msr_vtt", "HF"),
    ("video", "vatex", "HF"),
    ("video", "fine_video", "HF"),
    
    # Benchmarks
    ("benchmarks", "mmlu", "HF/Kaggle"), # MMLU is tricky in registry
    ("benchmarks", "mmmu", "HF"),
    ("benchmarks", "gsm8k", "HF"),
    ("benchmarks", "scienceqa", "HF"),
    ("benchmarks", "mathvista", "HF"),
]

print("Starting Verification Loop (Sample=5)...")
os.makedirs("logs", exist_ok=True)

results = []

for modality, name, expected_source in datasets:
    print(f"Testing {modality}/{name}...", end=" ", flush=True)
    
    cmd = ["python", "src/mm_download_unified.py", "--dataset", name, "--sample", "5", "--output-dir", "data"]
    
    try:
        # Run with timeout to prevent hangs
        process = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        
        stdout = process.stdout
        stderr = process.stderr
        output = stdout + stderr
        
        status = "❌ FAIL"
        note = f"Return Code: {process.returncode}"
        
        if process.returncode == 0:
            if "Saved" in output or "samples to" in output:
                status = "✅ PASS"
                note = "Verified (HF Streaming)"
            else:
                status = "⚠️ OK?"
                note = "Exit 0 but 'Saved' msg not found"
        else:
            if "403" in output or "401" in output:
                 note = "Auth Error (401/403)"
            elif "Empty" in output:
                 note = "Dataset Empty"
            elif "Hang" in output: 
                 note = "Timeout/Hang"
            elif "All sources failed" in output: # From my logger
                 note = "All Sources Failed"
            else:
                 note = "Runtime Error (Check Log)"
                 
        print(status)
        results.append({
            "name": name,
            "modality": modality,
            "status": status,
            "note": note
        })
        
        # Save log for inspection
        with open(f"logs/global_verify_{name}.log", "w") as f:
            f.write(output)

    except subprocess.TimeoutExpired:
        print("⏳ TIMEOUT")
        results.append({
            "name": name,
            "modality": modality,
            "status": "⏳ TIMEOUT",
            "note": "Process timed out (>120s)."
        })

print("\n\n=== VERIFICATION REPORT ===")
print("| Dataset | Modality | Status | Note |")
print("|---|---|---|---|")
for r in results:
    print(f"| **{r['name']}** | {r['modality']} | {r['status']} | {r['note']} |")
