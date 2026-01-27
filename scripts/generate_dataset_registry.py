import os
import re

STRUCTURE_DIR = "/mnt/d/Research Experiments/nexus/new-plan-conversation-files"

CATEGORY_FILES = {
    "reasoning": "reasoning-dataset-structure.txt",
    "code": "coding-dataset-structure.txt",
    "general": "general-dataset-structure.txt",
    "multimodal": ["multimodal-dataset-structure-1.txt", "multimodal-dataset-structure-2.txt", "multimodal-dataset-structure-3.txt"],
    "remotion": "remotion-dataset-structure.txt",
    "uncensored": "uncensored-dataset-structure.txt",
    "long_context": "long_context-dataset-structure.txt",
    "tools": "tools-dataset-structure.txt"
}

dataset_registry = {}
dataset_counts = {}

def process_file(filepath, category):
    if not os.path.exists(filepath):
        print(f"Skipping {filepath} (not found)")
        return

    with open(filepath, 'r') as f:
        lines = f.readlines()

    datasets = set()
    
    # Heuristic: look for ./datasets/category/DatasetName lines
    # We want exactly that depth.
    # Pattern: ^\./datasets/[^/]+/[^/]+$ matches lines like ./datasets/reasoning/Foo_Bar
    
    # Adjust regex to capture category dynamicaly if possible, but we pass it in.
    # Actually the files seem to have ./datasets/uncensored/... so category is in path.
    
    for line in lines:
        line = line.strip()
        match = re.match(r'^\./datasets/([^/]+)/([^/]+)$', line)
        if match:
            # path_category = match.group(1) # should match category approx
            dataset_dir = match.group(2)
            
            # Skip hidden files or metadata directories if any (e.g. starting with .)
            if dataset_dir.startswith('.'):
                continue
            
            datasets.add(dataset_dir)
            
    # Add to registry
    for ds_dir in datasets:
        # Heuristic for HF ID: replace result of first underscore with / 
        # But some might not follow this. e.g. "gsm8k" (no underscore).
        # If no underscore, it might be a top-level dataset or a custom one.
        # Let's assume most are User/Dataset.
        
        if '_' in ds_dir:
            parts = ds_dir.split('_', 1)
            hf_id = f"{parts[0]}/{parts[1]}"
        else:
            hf_id = ds_dir # Fallback
            
        key = ds_dir.lower().replace('.', '_') # clean key
        
        # Tags
        tags = [category]
        if "uncensored" in key or "nsfw" in key:
            tags.append("uncensored")
        if "code" in key:
            tags.append("code")
        if "video" in key or "image" in key:
            tags.append("multimodal")
            
        dataset_registry[key] = {
            "path": hf_id,
            "local_path": f"/mnt/e/data/datasets/{category}/{ds_dir}", # Assuming this based on config
            "desc": f"{category.capitalize()} dataset: {hf_id}",
            "tags": tags
        }
        
    dataset_counts[category] = len(datasets)

for category, filename in CATEGORY_FILES.items():
    if isinstance(filename, list):
        for f in filename:
            process_file(os.path.join(STRUCTURE_DIR, f), category)
    else:
        process_file(os.path.join(STRUCTURE_DIR, filename), category)

print("DATASET_REGISTRY = {")
for key, val in sorted(dataset_registry.items()):
    val_str = str(val).replace("'", '"')
    print(f'    "{key}": {val_str},')
print("}")

print("\nCounts per category:")
total = 0
for cat, count in dataset_counts.items():
    print(f"{cat}: {count}")
    total += count
print(f"Total: {total}")
