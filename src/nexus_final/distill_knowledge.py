import torch
import torch.nn as nn
import os
import json
import tqdm
from typing import List, Dict, Any, Optional
from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig
from .knowledge import KnowledgeTower

class KnowledgeDistiller:
    """
    Knowledge Extractor for NEXUS.
    Supports LLMs, Encoders (Vision/Audio), and Multimodal teachers.
    """
    def __init__(
        self, 
        tower: KnowledgeTower, 
        teacher_model_path: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.tower = tower
        self.device = device
        self.model_path = teacher_model_path
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(teacher_model_path)
        except Exception:
            # For some encoders, tokenizer might not be available or needed (e.g. CLIP/SigLIP might use processor)
            print(f"[Distiller] Warning: Could not load tokenizer for {teacher_model_path}. Using fallback.")
            self.tokenizer = None
        
        quantization_config = BitsAndBytesConfig(load_in_4bit=True)
        # Use AutoModel (generic) instead of AutoModelForCausalLM (specific)
        try:
            self.model = AutoModel.from_pretrained(
                teacher_model_path, 
                torch_dtype=torch.float16, 
                device_map=device,
                quantization_config=quantization_config,
                trust_remote_code=True
            )
        except Exception as e:
            print(f"[Distiller] Error loading model {teacher_model_path}: {e}")
            # Fallback to CausalLM if AutoModel failed for some reason
            from transformers import AutoModelForCausalLM
            self.model = AutoModelForCausalLM.from_pretrained(
                teacher_model_path,
                torch_dtype=torch.float16,
                device_map=device,
                quantization_config=quantization_config,
                trust_remote_code=True
            )
        self.model.eval()

    def extract_thematic_clusters(
        self, 
        data_source: List[str], 
        output_dir: str = "knowledge_shards/",
        batch_size: int = 4,
        shard_prefix: str = "shard"
    ):
        """
        Performs batch forward pass to extract hidden states (Mathematical Knowledge).
        Saves shards to SSD.
        """
        output_dir = os.path.abspath(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        print(f"[Distiller] Beginning mathematical extraction from {self.model_path}...")
        
        extracted_shards = []
        
        for i in tqdm.tqdm(range(0, len(data_source), batch_size)):
            batch = data_source[i : i + batch_size]
            inputs = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
                # Grab the final layer hidden states (the most semantic representation)
                hidden_states = outputs.hidden_states[-1] # [B, S, D]
                
                # Perform 'Mean Pooling' to get the shard's semantic essence
                mask = inputs['attention_mask'].unsqueeze(-1).expand(hidden_states.size())
                sum_h = torch.sum(hidden_states * mask, 1)
                count = torch.clamp(mask.sum(1), min=1e-9)
                shard_vectors = sum_h / count # [B, D]
                
            # Store Shard to SSD
            for b_idx, text in enumerate(batch):
                shard_id = f"{shard_prefix}_{i + b_idx}"
                shard_path = os.path.join(output_dir, f"{shard_id}.pt")
                
                # Save both the raw text and the mathematical representation
                # This allows the 'Librarian' (Student) to fetch facts from SSD.
                torch.save({
                    "text": text,
                    "hidden_state": shard_vectors[b_idx].cpu(),
                    "teacher": self.model_path
                }, shard_path)
                
                extracted_shards.append(text)
                
        # Update Tower Index with extracted knowledge
        print(f"[Distiller] Updating KnowledgeTower index with {len(extracted_shards)} shards...")
        self.tower.build_index(extracted_shards)
        print("[Distiller] Extraction complete.")



if __name__ == "__main__":
    import argparse
    parser.add_argument("--teacher", required=True, help="Path to teacher model")
    parser.add_argument("--output", required=True, help="Output directory for shards")
    parser.add_argument("--dataset", type=str, default="general/google_smol", help="Dataset path relative to /mnt/e/data/datasets")
    parser.add_argument("--limit", type=int, default=100, help="Max samples to extract")
    parser.add_argument("--shard_prefix", type=str, default="shard", help="Prefix for saved shards")
    args = parser.parse_args()
    
    # Mock KnowledgeTower for standalone run (or load real if path provided)
    from .knowledge import KnowledgeTower
    tower = KnowledgeTower(index_path="vector_index.faiss") # Dummy/Local
    
    distiller = KnowledgeDistiller(tower, args.teacher)
    
    # Use UniversalDataLoader
    from .data_loader import UniversalDataLoader
    loader = UniversalDataLoader()
    
    print(f"[Distiller] Loading data from {args.dataset} (limit={args.limit})...")
    dataset_gen = loader.load_dataset(args.dataset, limit=args.limit)
    
    prompts = []
    for sample in dataset_gen:
        # Extract user content as the prompt
        # Standard format: messages -> user content
        try:
            for msg in sample.get('messages', []):
                if msg['role'] == 'user':
                    prompts.append(msg['content'])
                    break
        except Exception:
            continue
            
    if not prompts:
        print("[Distiller] Warning: No prompts found in dataset. Using fallbacks.")
        prompts = ["Explain quantum mechanics.", "Solve 2x + 5 = 9"]

    print(f"[Distiller] Extracted {len(prompts)} prompts. Starting distillation...")
    distiller.extract_thematic_clusters(prompts, output_dir=args.output, shard_prefix=args.shard_prefix)
