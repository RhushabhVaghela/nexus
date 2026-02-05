#!/usr/bin/env python3
import os
from pathlib import Path

def replace_in_files(root_dir, replacements):
    for path in Path(root_dir).rglob('*'):
        if path.is_file() and not '.git' in str(path) and not '__pycache__' in str(path):
            try:
                # Read content
                with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                new_content = content
                changed = False
                for old_str, new_str in replacements.items():
                    if old_str in new_content:
                        print(f"Replacing '{old_str}' with '{new_str}' in: {path}")
                        new_content = new_content.replace(old_str, new_str)
                        changed = True
                
                if changed:
                    with open(path, 'w', encoding='utf-8') as f:
                        f.write(new_content)
            except Exception as e:
                print(f"Could not process {path}: {e}")

if __name__ == "__main__":
    replacements = {
        "/mnt/e/data/models/Qwen2.5-Omni-7B-GPTQ-Int4": "/mnt/e/data/models/Qwen2.5-Omni-7B-GPTQ-Int4",
        "/mnt/e/data/models/Qwen2.5-0.5B": "/mnt/e/data/models/Qwen2.5-0.5B",
        # Specifically handle the Instruct naming if needed
        "Qwen2.5-Omni-7B-GPTQ-Int4": "Qwen2.5-Omni-7B-GPTQ-Int4"
    }
    replace_in_files('.', replacements)
