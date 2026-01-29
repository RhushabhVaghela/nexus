import os
import sys
import glob
import argparse
import torch
import faiss
import numpy as np
from tqdm import tqdm

# Ensure explicit path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(BASE_DIR, 'src'))

from nexus_final.knowledge import KnowledgeTower

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--memory_dir", required=True, help="Path to memory shards")
    parser.add_argument("--output", required=True, help="Output path for .faiss index")
    parser.add_argument("--embedding_model", default="sentence-transformers/all-MiniLM-L6-v2")
    args = parser.parse_args()

    print(f"[IndexBuilder] Scanning {args.memory_dir} for shards...")
    shard_files = glob.glob(os.path.join(args.memory_dir, "**/*.pt"), recursive=True)
    shard_files = [f for f in shard_files if "shard_" in os.path.basename(f)]
    
    if not shard_files:
        print("[IndexBuilder] No shards found. Creating dummy index.")
        with open(args.output, 'w') as f:
             f.write("dummy_index")
        return

    print(f"[IndexBuilder] Found {len(shard_files)} shards. Extracting text...")
    documents = []
    
    # Load all text
    for path in tqdm(shard_files, desc="Loading Text"):
        try:
            data = torch.load(path, map_location="cpu", weights_only=True)
            if "text" in data and isinstance(data["text"], str) and len(data["text"]) > 10:
                documents.append(data["text"])
        except Exception as e:
            print(f"[Warn] Failed to load {path}: {e}")

    if not documents:
        print("[IndexBuilder] No valid text found. Creating dummy.")
        with open(args.output, 'w') as f: f.write("dummy_index")
        return

    print(f"[IndexBuilder] Building Index for {len(documents)} documents using {args.embedding_model}...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tower = KnowledgeTower(device=device, embedding_model=args.embedding_model)
    
    # Build
    tower.build_index(documents)
    
    # Save
    print(f"[IndexBuilder] Saving index to {args.output}...")
    faiss.write_index(tower.index, args.output)
    
    # Also save documents reference (standardized for inference)
    doc_path = os.path.join(os.path.dirname(args.output), "knowledge_docs.json")
    with open(doc_path, 'w', encoding='utf-8') as f:
        json.dump(documents, f)
        
    print("[IndexBuilder] Complete.")

if __name__ == "__main__":
    import json
    main()
