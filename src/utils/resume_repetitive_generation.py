#!/usr/bin/env python3
"""
Resume Repetitive Dataset Generation
Continues from where generation stopped, with different random seed to avoid redundancy.
"""

import json
import logging
import random
import datetime
import string
from pathlib import Path
import sys
import os
import time

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/gen_repetitive.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# -------------------------------------------------------------------------
# PROCEDURAL GENERATORS (Same as original but with more variety)
# -------------------------------------------------------------------------

def random_string(length=8):
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

def random_ip():
    return f"{random.randint(1,255)}.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(0,255)}"

def random_date():
    start = datetime.date(2020, 1, 1)
    end = datetime.date(2025, 12, 31)
    delta = end - start
    random_days = random.randrange(delta.days)
    return (start + datetime.timedelta(days=random_days)).isoformat()

# --- Generator 1: Log Extraction ---
def gen_log_extraction():
    lines = []
    errors = []
    num_lines = random.randint(25, 60)  # More variety
    
    for _ in range(num_lines):
        ts = datetime.datetime.now().isoformat()
        level = random.choice(["INFO", "DEBUG", "WARN", "ERROR", "TRACE", "FATAL"])
        msg = random_string(random.randint(10, 25))
        
        if level in ["ERROR", "FATAL"]:
            code = f"E-{random.randint(100, 9999)}"  # Wider range
            line = f"[{ts}] {level} {code}: {msg}"
            errors.append(code)
        else:
            line = f"[{ts}] {level}: {msg}"
        lines.append(line)
        
    context = "\n".join(lines)
    query = random.choice([
        "List all error codes (e.g., E-XXX) found in the log.",
        "Extract all error codes from this log file.",
        "What error codes appear in the following log?",
        "Find and list every error code (format E-XXX) in the logs below.",
    ])
    
    if not errors:
        result = "No error codes found."
    else:
        result = json.dumps(list(set(errors)))  # Unique errors
        
    return query, context, result

# --- Generator 2: JSON Lookup ---
def gen_json_lookup():
    data = {}
    target_key = f"key_{random_string(random.randint(3, 8))}"
    target_value = f"val_{random_string(random.randint(4, 10))}"
    
    num_fields = random.randint(15, 40)
    for i in range(num_fields):
        k = f"field_{i}_{random_string(random.randint(2, 5))}"
        v = random.choice([
            random.randint(0, 10000), 
            random_string(random.randint(3, 8)), 
            random_ip(),
            random.uniform(0, 1000),
            random.choice([True, False]),
        ])
        data[k] = v
        
    data[target_key] = target_value
    
    context = json.dumps(data, indent=2)
    query = random.choice([
        f"What is the value associated with the key '{target_key}'?",
        f"Find the value for key '{target_key}' in this JSON.",
        f"Look up '{target_key}' in the following JSON data.",
    ])
    
    return query, context, target_value

# --- Generator 3: Phone Directory ---
def gen_directory_lookup():
    names = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis",
             "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzales", "Wilson", "Anderson",
             "Thomas", "Taylor", "Moore", "Jackson", "Martin", "Lee", "Perez", "Thompson"]
    firsts = ["James", "Mary", "John", "Patricia", "Robert", "Jennifer", "Michael", "Linda",
              "William", "Elizabeth", "David", "Barbara", "Richard", "Susan", "Joseph", "Jessica",
              "Thomas", "Sarah", "Charles", "Karen", "Christopher", "Lisa", "Daniel", "Nancy"]
    
    target_name = f"{random.choice(firsts)} {random.choice(names)}"
    target_ext = str(random.randint(1000, 9999))
    
    directory_list = []
    num_entries = random.randint(40, 80)
    
    for _ in range(num_entries):
        n = f"{random.choice(firsts)} {random.choice(names)}"
        e = str(random.randint(1000, 9999))
        directory_list.append(f"{n}..........{e}")
    
    directory_list.append(f"{target_name}..........{target_ext}")
    random.shuffle(directory_list)
    
    context = "Employee Directory:\n" + "\n".join(directory_list)
    query = random.choice([
        f"Find the extension number for '{target_name}'.",
        f"What is {target_name}'s extension?",
        f"Look up the phone extension for {target_name}.",
    ])
    
    return query, context, target_ext

# --- Generator 4: Table Lookup (NEW) ---
def gen_table_lookup():
    headers = ["ID", "Name", "Department", "Salary", "Start Date"]
    rows = []
    
    target_id = random.randint(1000, 9999)
    target_value = random.choice(["Engineering", "Sales", "Marketing", "HR", "Finance"])
    
    num_rows = random.randint(20, 50)
    for i in range(num_rows):
        row_id = random.randint(1000, 9999)
        name = f"{random_string(5)} {random_string(6)}"
        dept = random.choice(["Engineering", "Sales", "Marketing", "HR", "Finance"])
        salary = random.randint(50000, 150000)
        date = random_date()
        rows.append(f"| {row_id} | {name} | {dept} | ${salary:,} | {date} |")
    
    # Insert target row
    target_name = f"{random_string(5)} {random_string(6)}"
    rows.insert(random.randint(0, len(rows)), f"| {target_id} | {target_name} | {target_value} | ${random.randint(50000, 150000):,} | {random_date()} |")
    
    header_line = "| " + " | ".join(headers) + " |"
    separator = "|" + "|".join(["---"] * len(headers)) + "|"
    context = header_line + "\n" + separator + "\n" + "\n".join(rows)
    
    query = f"What department does employee ID {target_id} work in?"
    
    return query, context, target_value

# -------------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------------

def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=51150000, help="Starting sample index")
    parser.add_argument("--count", type=int, default=148850000, help="Total samples to generate")
    parser.add_argument("--seed", type=int, default=42424242, help="Random seed (different from original)")
    args = parser.parse_args()

    # Set different random seed to avoid redundancy
    random.seed(args.seed)
    
    TARGET_SCALE = args.count
    START_INDEX = args.start
    
    # Setup Directories - Use new E: drive path
    base_dir = Path("/mnt/e/repetitive-query-dataset")
    for split in ["train", "val", "test"]:
        os.makedirs(base_dir / split, exist_ok=True)
        
    logger.info("="*60)
    logger.info("🚀 RESUMING REPETITIVE PROMPT GENERATION")
    logger.info(f"   Starting from: {START_INDEX:,}")
    logger.info(f"   Target additional: {TARGET_SCALE:,}")
    logger.info(f"   Random seed: {args.seed} (different from original)")
    logger.info(f"   Output: {base_dir}")
    logger.info("="*60)
    
    generators = [gen_log_extraction, gen_json_lookup, gen_directory_lookup, gen_table_lookup]
    
    # File handles - start new chunk indices based on what exists
    CHUNK_SIZE = 1000000
    
    # Count existing files to determine starting chunk indices
    existing_train = len(list((base_dir / "train").glob("*.jsonl")))
    existing_val = len(list((base_dir / "val").glob("*.jsonl")))
    existing_test = len(list((base_dir / "test").glob("*.jsonl")))
    
    chunk_counters = {
        "train": existing_train, 
        "val": existing_val, 
        "test": existing_test
    }
    sample_counters = {"train": 0, "val": 0, "test": 0}
    file_handles = {}
    
    def open_new_file(split):
        idx = chunk_counters[split]
        path = base_dir / split / f"part_{idx:03d}.jsonl"
        file_handles[split] = open(path, "w")
        chunk_counters[split] += 1
        logger.info(f"�� Opened new file: {path}")
        return path

    for split in ["train", "val", "test"]:
        open_new_file(split)
    
    start_time = time.time()
    
    for i in range(TARGET_SCALE):
        # Determine Split (90/5/5)
        r = random.random()
        if r < 0.90:
            target = "train"
        elif r < 0.95:
            target = "val"
        else:
            target = "test"
            
        gen_func = random.choice(generators)
        query, context, answer = gen_func()
        
        user_content = f"{query}\n\n{query}\n\nContext:\n{context}"
        
        trajectory = {
            "messages": [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": answer}
            ],
            "domain": "repetitive_prompting",
            "technique": "procedural_generation",
            "split": target,
            "resume_batch": True  # Mark as from resume batch
        }
        
        file_handles[target].write(json.dumps(trajectory) + "\n")
        sample_counters[target] += 1
        
        # Rotate if chunk full
        if sample_counters[target] % CHUNK_SIZE == 0:
            file_handles[target].close()
            open_new_file(target)
            
        if i > 0 and i % 50000 == 0:
            elapsed = time.time() - start_time
            rate = i / elapsed
            remaining = TARGET_SCALE - i
            hours = (remaining / rate) / 3600
            total = START_INDEX + i
            logger.info(f"✓ Total: {total:,} ({rate:.0f}/sec) | Train: {sample_counters['train']:,} Val: {sample_counters['val']:,} Test: {sample_counters['test']:,} | ETA: {hours:.1f}h")

    # Close all
    for f in file_handles.values():
        f.close()
        
    total_time = time.time() - start_time
    logger.info("="*60)
    logger.info("✅ RESUME GENERATION COMPLETE")
    logger.info(f"   Additional samples: {TARGET_SCALE:,}")
    logger.info(f"   Total samples now: {START_INDEX + TARGET_SCALE:,}")
    logger.info(f"   Time: {total_time:.2f}s")
    logger.info("="*60)

if __name__ == "__main__":
    main()
