import torch
import torch.nn as nn
import os
import json
import tqdm
from typing import List, Dict, Any, Optional
from transformers import AutoTokenizer, AutoModel, AutoConfig, BitsAndBytesConfig
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
            # fix_mistral_regex is needed for some newer models/tokenizers
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(teacher_model_path, fix_mistral_regex=True)
            except TypeError as te:
                print(f"[Distiller] Note: 'fix_mistral_regex' not supported by this transformers version ({te}). Loading standard tokenizer.")
                self.tokenizer = AutoTokenizer.from_pretrained(teacher_model_path)
        except Exception:
            # For some encoders, tokenizer might not be available or needed (e.g. CLIP/SigLIP might use processor)
            print(f"[Distiller] Warning: Could not load tokenizer for {teacher_model_path}. Using fallback.")
            self.tokenizer = None
        
        # Detect GPTQ/AWQ to avoid BitsAndBytes conflict
        is_quantized = "GPTQ" in teacher_model_path or "AWQ" in teacher_model_path
        if is_quantized:
            print(f"[Distiller] Detected Quantized Model ({teacher_model_path}). disabling BitsAndBytes.")
            quantization_config = None
        else:
            quantization_config = BitsAndBytesConfig(load_in_4bit=True)

        # Configure Logging to reduce console noise
        import logging
        logging.basicConfig(filename='distiller_debug.log', level=logging.INFO, 
                            format='%(asctime)s - %(levelname)s - %(message)s')
        
        from transformers import logging as hf_logging
        hf_logging.set_verbosity_error() # Suppress "Some weights..." to console
        hf_logging.enable_propagation()  # Allow propagation to our file handler

        # 0. Hardware Optimization (Beast Mode: RTX 5080)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

        # Datatype logic: Use BF16 for Blackwell/Ada hardware
        model_dtype = torch.bfloat16 if device == "cuda" else torch.float32
        if device == "cpu":
             pass 

        # Smart Model Loading Strategy
        try:
            # 1. Inspect config first
            config = AutoConfig.from_pretrained(teacher_model_path, trust_remote_code=True)
            model_type = getattr(config, "model_type", "")
            logging.info(f"Model Type Detected: {model_type}")

            if model_type == "qwen2_5_omni":
                print(f"[Distiller] Detected '{model_type}'. Using Qwen2ForCausalLM (Omni-Compatible).")
                from transformers import Qwen2ForCausalLM
                self.model = Qwen2ForCausalLM.from_pretrained(
                    teacher_model_path,
                    dtype=model_dtype,
                    device_map=device,
                    trust_remote_code=True
                )
            else:
                # 2. Standard Loading Path
                try:
                    self.model = AutoModel.from_pretrained(
                        teacher_model_path, 
                        dtype=model_dtype, 
                        device_map=device,
                        quantization_config=quantization_config,
                        trust_remote_code=True
                    )
                except Exception:
                    # 3. Fallback for models that demand xxxForCausalLM (e.g., standard Qwen)
                    print(f"[Distiller] AutoModel failed. Falling back to AutoModelForCausalLM.")
                    from transformers import AutoModelForCausalLM
                    self.model = AutoModelForCausalLM.from_pretrained(
                        teacher_model_path,
                        dtype=model_dtype,
                        device_map=device,
                        quantization_config=quantization_config,
                        trust_remote_code=True
                    )

        except Exception as e:
            logging.error(f"[Distiller] Fatal Load Error: {e}")
            print(f"[Distiller] Failed to load model {teacher_model_path}. See 'distiller_debug.log'.")
            raise e
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher", required=True, help="Path to teacher model")
    parser.add_argument("--output", required=True, help="Output directory for shards")
    parser.add_argument("--dataset", type=str, default="general/google_smol", help="Dataset path relative to /mnt/e/data/datasets")
    parser.add_argument("--limit", type=int, default=100, help="Max samples to extract")
    parser.add_argument("--shard_prefix", type=str, default="shard", help="Prefix for saved shards")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use (cuda/cpu)")
    parser.add_argument("--embedding_model", type=str, default="sentence-transformers/all-MiniLM-L6-v2", help="Embedding model for index (default: MiniLM-L6, optional: all-mpnet-base-v2)")
    args = parser.parse_args()
    
    # Mock KnowledgeTower for standalone run (or load real if path provided)
    from .knowledge import KnowledgeTower
    tower = KnowledgeTower(device="cpu", embedding_model=args.embedding_model) # Dummy/Local instantiation for extraction
    
    distiller = KnowledgeDistiller(tower, args.teacher, device=args.device)
    
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
