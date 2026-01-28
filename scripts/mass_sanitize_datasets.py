import os
import sys
import json
from tqdm import tqdm

# Ensure src is in path
sys.path.append(os.path.join(os.getcwd(), 'src'))

try:
    from nexus_core.data.sanitizer import UniversalSanitizer
    from nexus_core.towers.registry import DATASET_REGISTRY
except ImportError:
    print("[Error] Could not import necessary modules. Run from project root.")
    sys.exit(1)

def sanitize_file(input_path, output_path, limit=None):
    """Sanitizes a single JSONL file."""
    sanitizer = UniversalSanitizer()
    
    with open(input_path, 'r', encoding='utf-8') as f_in, \
         open(output_path, 'w', encoding='utf-8') as f_out:
        
        count = 0
        for line in f_in:
            if limit and count >= limit: break
            if not line.strip(): continue
            try:
                item = json.loads(line)
                # Use the sanitizer to extract ONLY the clean text
                # We normalize to a simple {"text": "..."} or {"messages": [...]} schema
                clean_text = sanitizer.sanitize(item)
                
                if clean_text:
                    # Output in standardized format
                    output_item = {
                        "messages": [{"role": "user", "content": clean_text}],
                        "original_metadata": {k: v for k, v in item.items() if not isinstance(v, (dict, list))} 
                    }
                    f_out.write(json.dumps(output_item) + "\n")
                    count += 1
            except:
                continue
    return count

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Nexus Mass Dataset Sanitizer")
    parser.add_argument("--dataset_name", type=str, help="Specific dataset to sanitize (registry key)")
    parser.add_argument("--limit_per_file", type=int, default=1000, help="Samples to process per file")
    parser.add_argument("--all", action="store_true", help="Sanitize all accessible datasets")
    parser.add_argument("--output_suffix", type=str, default=".sanitized", help="Suffix for output files")
    
    args = parser.parse_args()
    
    targets = []
    if args.dataset_name:
        if args.dataset_name in DATASET_REGISTRY:
            targets.append((args.dataset_name, DATASET_REGISTRY[args.dataset_name]))
        else:
            print(f"[Error] Dataset {args.dataset_name} not found in registry.")
            return
    elif args.all:
        targets = list(DATASET_REGISTRY.items())
    else:
        print("[Error] Specify --dataset_name or --all")
        return

    print(f"Starting mass sanitization for {len(targets)} targets...")
    
    for name, info in targets:
        path = info.get("local_path")
        if not path or not os.path.exists(path):
            continue
            
        print(f"\n[Processing] {name}...")
        
        # Look for data files
        data_files = []
        for root, _, filenames in os.walk(path):
            for f in filenames:
                if f.endswith('.jsonl') and not f.endswith('.sanitized.jsonl'):
                    data_files.append(os.path.join(root, f))
        
        if not data_files:
            print(f"  -> No .jsonl files found in {path}")
            continue
            
        for f_path in data_files:
            output_path = f_path.replace(".jsonl", f"{args.output_suffix}.jsonl")
            print(f"  -> Sanitizing: {os.path.basename(f_path)}")
            num_saved = sanitize_file(f_path, output_path, limit=args.limit_per_file)
            print(f"  -> Saved {num_saved} samples to {os.path.basename(output_path)}")

if __name__ == "__main__":
    main()
