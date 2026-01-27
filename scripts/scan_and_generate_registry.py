import os

DATASET_ROOT = "/mnt/e/data/datasets"
BENCHMARK_ROOT = "/mnt/e/data/benchmarks"

registry = {}
counts = {}

def scan_root(root_path, is_benchmark=False):
    if not os.path.exists(root_path):
        print(f"Error: {root_path} does not exist.")
        return

    # Level 1: Categories (e.g. reasoning, code)
    for category in sorted(os.listdir(root_path)):
        cat_path = os.path.join(root_path, category)
        if not os.path.isdir(cat_path):
            continue

        if category not in counts:
            counts[category] = 0

        # Level 2: Datasets (Directories OR Files)
        for ds_name in sorted(os.listdir(cat_path)):
            ds_path = os.path.join(cat_path, ds_name)
            
            is_valid_file = os.path.isfile(ds_path) and ds_name.endswith(('.jsonl', '.json', '.parquet', '.csv', '.tsv'))
            is_valid_dir = os.path.isdir(ds_path)

            if not (is_valid_file or is_valid_dir):
                continue
            
            # Key generation: lowercase, clean, remove extension
            clean_name = os.path.splitext(ds_name)[0]
            key = clean_name.lower().replace('.', '_')
            if is_benchmark:
                key = f"benchmark_{key}"

            # HF Path Heuristic
            if '_' in clean_name:
                parts = clean_name.split('_', 1)
                hf_path = f"{parts[0]}/{parts[1]}"
            else:
                hf_path = clean_name

            # Tags
            tags = [category]
            if is_benchmark:
                tags.append("benchmark")
            if "code" in category or "code" in key: tags.append("code")
            if "math" in category or "math" in key: tags.append("math")
            if "vision" in key or "image" in key or "video" in key or "multimodal" in category: tags.append("multimodal")
            if "audio" in key or "speech" in key: tags.append("audio")
            if "uncensored" in category or "nsfw" in key: tags.append("uncensored")

            # Remotion special case
            if category == "remotion":
                tags.append("remotion")
                # Fix path for readability
                if key == "remotion_explainer_dataset":
                     hf_path = "remotion/explainer_dataset"
                
            entry = {
                "path": hf_path,
                "local_path": ds_path,
                "desc": f"{'Benchmark' if is_benchmark else 'Dataset'}: {hf_path}",
                "tags": list(set(tags)) # dedup
            }

            registry[key] = entry
            counts[category] += 1

print(f"Scanning Datasets at {DATASET_ROOT}...")
scan_root(DATASET_ROOT, is_benchmark=False)

print(f"Scanning Benchmarks at {BENCHMARK_ROOT}...")
scan_root(BENCHMARK_ROOT, is_benchmark=True)

# Generate Python Output
print("\nDATASET_REGISTRY = {")

# Group by category for cleaner output
entries_by_cat = {}
for key, val in registry.items():
    # Use the first tag as the primary category for grouping
    # Check verified category list to align with folders
    cat = "general" # Default
    if "tags" in val:
        # Prioritize folder-based tags
        for t in ["reasoning", "code", "multimodal", "tools", "long_context", "remotion", "uncensored"]:
             if t in val["tags"]:
                 cat = t
                 break
    
    if cat not in entries_by_cat:
        entries_by_cat[cat] = []
    entries_by_cat[cat].append((key, val))

# Output sorted groups
category_order = ["reasoning", "code", "general", "multimodal", "tools", "long_context", "remotion", "uncensored"]
# Add any others found
for c in entries_by_cat:
    if c not in category_order:
        category_order.append(c)

for cat in category_order:
    if cat not in entries_by_cat:
         continue
         
    print(f"\n    # --- {cat.upper()} ({len(entries_by_cat[cat])}) ---")
    
    # Sort entries within category
    for key, val in sorted(entries_by_cat[cat]):
         val_str = str(val).replace("'", '"')
         print(f'    "{key}": {val_str},')

# Add Benchmarks section if any exist
benchmarks = {k: v for k, v in registry.items() if "benchmark" in v["tags"]}
if benchmarks:
    print(f"\n    # --- BENCHMARKS ({len(benchmarks)}) ---")
    # Note: Benchmarks are already printed above if they were categorized by their main tag. 
    # To avoid duplication, we rely on the main categories. 
    # If benchmarks are a separate root scan, they might be mixed in above.
    # The current logic just dumps everything by primary tag.
    pass

print("}")
