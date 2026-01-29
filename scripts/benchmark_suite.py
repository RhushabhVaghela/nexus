import sys
import os
import argparse
import json
import pandas as pd
from tqdm import tqdm

# Ensure we can import from src/
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE_DIR, 'src'))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_model(path, device="cuda"):
    print(f"Loading {path}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            path, 
            device_map=device, 
            trust_remote_code=True,
            torch_dtype=torch.float16,
            load_in_4bit=True, # Prevent OOM
            low_cpu_mem_usage=True
        )
        return model, tokenizer
    except Exception as e:
        print(f"Failed to load {path}: {e}")
        return None, None

def evaluate(model, tokenizer, questions, targets, device="cuda"):
    score = 0
    results = []
    
    for q, t in tqdm(zip(questions, targets), total=len(questions), desc="Evaluating"):
        inputs = tokenizer(q, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=100, do_sample=False)
        resp = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        correct = t.lower() in resp.lower()
        score += 1 if correct else 0
        results.append({"q": q, "response": resp, "target": t, "correct": correct})
        
    return score / len(questions), results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--student", default="nexus-release-v1/student_core")
    parser.add_argument("--teacher", required=True, help="Path to teacher model")
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()
    
    # Simple Benchmark Data (GSM8K-like)
    data = [
        ("What is 25 + 75?", "100"),
        ("Solve 2x + 5 = 15", "5"),
        ("If I have 3 apples and eat 1, how many left?", "2"),
        ("Write a python function to add two numbers.", "def add(a, b):")
    ] * (args.limit // 4 + 1)
    data = data[:args.limit]
    
    questions = [d[0] for d in data]
    targets = [d[1] for d in data]
    
    # Evaluate Teacher
    t_model, t_tok = load_model(args.teacher)
    if t_model:
        t_acc, t_res = evaluate(t_model, t_tok, questions, targets)
        del t_model, t_tok
        torch.cuda.empty_cache()
    else:
        t_acc = 0.0
        
    # Evaluate Student
    s_model, s_tok = load_model(args.student)
    if s_model:
        s_acc, s_res = evaluate(s_model, s_tok, questions, targets)
        del s_model, s_tok
    else:
        s_acc = 0.0
        
    print(f"\n=== BENCHMARK RESULTS ===")
    print(f"Teacher ({args.teacher}): {t_acc:.2%}")
    print(f"Student ({args.student}): {s_acc:.2%}")
    print(f"Gap: {s_acc - t_acc:.2%}")
    
    # Save Qualitative Comparison
    print("\nGenering Side-by-Side Comparison Report...")
    with open("results/benchmark_comparison.md", "w") as f:
        f.write("# Nexus Benchmark: Teacher vs Student\n\n")
        f.write(f"- **Teacher**: {args.teacher}\n")
        f.write(f"- **Student**: {args.student}\n")
        f.write(f"- **Accuracy Gap**: {s_acc - t_acc:.2%}\n\n")
        f.write("## Side-by-Side Samples\n\n")
        f.write("| Question | Teacher Output | Student Output | Correct? |\n")
        f.write("|----------|----------------|----------------|----------|\n")
        
        # We need to align results. Assuming sequential execution on same list.
        # t_res and s_res match index-wise.
        for i, (tr, sr) in enumerate(zip(t_res, s_res)):
            # Sanitize for markdown table
            sess_q = tr['q'].replace("\n", "<br>").replace("|", "\|")
            sess_t = tr['response'].replace("\n", "<br>").replace("|", "\|")[:200] + "..." # Truncate
            sess_s = sr['response'].replace("\n", "<br>").replace("|", "\|")[:200] + "..."
            mark = "✅" if sr['correct'] else "❌"
            
            f.write(f"| {sess_q} | {sess_t} | {sess_s} | {mark} |\n")
            
    print("✅ Full report saved to results/benchmark_comparison.md")

if __name__ == "__main__":
    main()
