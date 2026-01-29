import os
import torch
import shutil
import json
from nexus_core.student.core import NexusStudentCore
from .knowledge import KnowledgeTower
from nexus_core.student.router import SparseIntentRouter

class NexusExporter:
    """
    Assembles the final Nexus Model for deployment.
    """
    def __init__(self, output_dir="nexus-release-v1"):
        self.output_dir = os.path.abspath(output_dir)
        os.makedirs(self.output_dir, exist_ok=True)
        
    def export(self, student_path, router_path, knowledge_index_path, vocab_size=None, hidden_size=None):
        print(f"[Exporter] Assembling Nexus Release in {self.output_dir}...")
        
        # 1. Export Student Core (The LLM)
        print("[Exporter] Saving Student Core...")
        # Direct export to parent folder
        student_dest = self.output_dir
        
        if os.path.isdir(student_path):
             # If it's a directory (HF format)
             shutil.copytree(student_path, student_dest, dirs_exist_ok=True)
        else:
             # If it's a .pt file
             # Handle Trainer Checkpoint (extract student_state)
             if os.path.exists(student_path):
                 try:
                     checkpoint = torch.load(student_path, map_location="cpu")
                     
                     state_dict = None
                     if isinstance(checkpoint, dict) and "student_state" in checkpoint:
                         print("[Exporter] Extracting student_state weights from trainer checkpoint.")
                         state_dict = checkpoint["student_state"]
                     else:
                         # Assume the whole file is the state dict
                         state_dict = checkpoint

                     try:
                         from safetensors.torch import save_file
                         print("[Exporter] Saving as model.safetensors (Modern Format)...")
                         save_file(state_dict, os.path.join(student_dest, "model.safetensors"))
                     except ImportError:
                         print("[Exporter] 'safetensors' not installed. Saving as pytorch_model.bin (Legacy)...")
                         torch.save(state_dict, os.path.join(student_dest, "pytorch_model.bin"))
                         
                 except Exception as e:
                     print(f"[Exporter] Warning: Failed to load checkpoint cleanly: {e}. Copying raw file...")
                     shutil.copy(student_path, os.path.join(student_dest, "pytorch_model.bin"))
             else:
                 print(f"[Exporter] Error: Student checkpoint not found at {student_path}")
                 print("[Exporter] Export Failed: Missing Student Core.")
                 import sys
                 sys.exit(1)
             
        # 1.5. Export Tokenizer Files
        print("[Exporter] Copying Tokenizer Assets...")
        tokenizer_files = ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.json", "merges.txt", "added_tokens.json"]
        source_dir = student_path if os.path.isdir(student_path) else os.path.dirname(student_path)
        copied_any = False
        for fname in tokenizer_files:
            src = os.path.join(source_dir, fname)
            if os.path.exists(src):
                shutil.copy(src, os.path.join(student_dest, fname))
                copied_any = True
        if not copied_any:
            print("[Exporter] Warning: No tokenizer files found locally.")
            print("[Exporter] Attempting to download base tokenizer (Unsloth Llama-3-8B)...")
            try:
                from transformers import AutoTokenizer
                # Use the same base as train.py
                tok = AutoTokenizer.from_pretrained("unsloth/llama-3-8b-bnb-4bit") 
                
                # Ensure Chat Template is set for GGUF compatibility
                if not tok.chat_template:
                    print("[Exporter] Injecting Llama-3 Chat Template...")
                    tok.chat_template = "{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}"
                
                tok.save_pretrained(student_dest)
                print("[Exporter] Success: Base tokenizer downloaded and saved (with Chat Template).")
            except Exception as e:
                print(f"[Exporter] Critical: Failed to download base tokenizer: {e}")
                print("[Exporter] Inference will forced to fallback to GPT-2 (Degraded).")

        # Create config
        config = {
            "model_type": "nexus_student",
            "vocab_size": vocab_size or 128256,
            "hidden_size": hidden_size or 2048
        }
        with open(os.path.join(student_dest, "config.json"), "w") as f:
            json.dump(config, f, indent=2)

        # 2. Export Router
        print("[Exporter] Saving Sparse Intent Router...")
        router_dest = os.path.join(self.output_dir, "router.pt")
        # In real scenario: model.save_pretrained()
        # Here just copy weights
        if os.path.exists(router_path):
            shutil.copy(router_path, router_dest)
            print(f"[Exporter] Router weights saved to {router_dest}")
        else:
            # Create dummy if missing (for dry run continuity)
            print(f"[Exporter] Warning: Router final weights not found at {router_path}. Creating dummy.")
            torch.save({"dummy": True}, router_dest)

        # 3. Export Knowledge Index
        print("[Exporter] Bundling Knowledge Tower...")
        index_dest = os.path.join(self.output_dir, "knowledge_index.faiss")
        if os.path.exists(knowledge_index_path):
            shutil.copy(knowledge_index_path, index_dest)
            # Standard: Also copy knowledge_docs.json if it exists alongside index
            docs_src = os.path.join(os.path.dirname(knowledge_index_path), "knowledge_docs.json")
            if os.path.exists(docs_src):
                shutil.copy(docs_src, os.path.join(self.output_dir, "knowledge_docs.json"))
                print(f"[Exporter] Knowledge index and docs mapping saved to {self.output_dir}")
            else:
                print(f"[Exporter] Knowledge index saved to {index_dest} (Warning: mapping missing)")
        else:
            # Create dummy
            print(f"[Exporter] Warning: Knowledge index not found at {knowledge_index_path}. Creating dummy.")
            with open(index_dest, "w") as f: f.write("dummy_index")

        # 4. Create Model Card
        self._create_model_card()
        
        print(f"[Exporter] Successfully exported Nexus to {self.output_dir}")

    def _create_model_card(self):
        content = """---
language: en
tags:
- nexus
- distillation
- mixture-of-experts
licenses: apache-2.0
---

# Nexus Model (v1.0)

This model was distilled from 15+ teacher models using the Nexus Pipeline.

## Components
- **Student Core**: 2B Parameter LLM
- **Router**: Sparse Intent Router for dynamic expert selection
- **Memory**: Activated Knowledge Tower (FAISS)

## Usage
```python
from nexus import NexusModel
model = NexusModel.from_pretrained("./nexus-release-v1")
response = model.generate("Explain quantum mechanics")
```
"""
        with open(os.path.join(self.output_dir, "README.md"), "w") as f:
            f.write(content)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--student", required=True, help="Path to student checkpoint")
    parser.add_argument("--router", default="checkpoints/router_final.pt", help="Path to router weights")
    parser.add_argument("--index", default="vector_index.faiss", help="Path to knowledge index")
    parser.add_argument("--output", default="nexus-release-v1", help="Export directory")
    parser.add_argument("--vocab_size", type=int, default=128256)
    parser.add_argument("--hidden_size", type=int, default=2048)
    args = parser.parse_args()
    
    exporter = NexusExporter(args.output)
    exporter.export(args.student, args.router, args.index, vocab_size=args.vocab_size, hidden_size=args.hidden_size)
