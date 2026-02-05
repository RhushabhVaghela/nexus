import os
import sys
import logging

# Configure logging early
logger = logging.getLogger(__name__)

# Ensure unsloth is imported before any other heavy libraries if possible
try:
    import unsloth
    logger.info("Unsloth library loaded successfully")
except ImportError:
    logger.warning(
        "Unsloth library not available. This is optional but recommended for optimized training. "
        "Install with: pip install unsloth"
    )

import torch
import torch.nn as nn
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

        # Load Model using Universal OmniModelLoader
        from src.omni.loader import OmniModelLoader
        
        print(f"[Distiller] Initializing Universal Loader for {teacher_model_path}...")
        loader = OmniModelLoader(teacher_model_path)
        
        try:
            load_kwargs = {
                "trust_remote_code": True,
                "torch_dtype": model_dtype
            }
            if quantization_config:
                load_kwargs["load_in_4bit"] = True
                
            self.model, self.tokenizer = loader.load(mode="full", **load_kwargs)
            self.processor = getattr(loader, "processor", None)
            print(f"[Distiller] Model loaded successfuly using class: {self.model.__class__.__name__}")
        except Exception as e:
            print(f"[Distiller] Universal Loader failed: {e}")
            logging.error(f"[Distiller] Fatal Load Error: {e}")
            raise e

        self.model.eval()

    def extract_thematic_clusters(
        self, 
        data_source: List[Dict[str, Any]], # Changed from List[str] to List[Dict]
        output_dir: str = "knowledge_shards/",
        batch_size: int = 2, # Reduced default for multimodal memory safety
        shard_prefix: str = "shard"
    ):
        """
        Performs batch forward pass to extract hidden states (Mathematical Knowledge).
        Supports multimodal inputs (Images/Audio).
        """
        from PIL import Image
        output_dir = os.path.abspath(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        print(f"[Distiller] Beginning mathematical extraction from {self.model_path}...")
        
        extracted_shards = []
        
        for i in tqdm.tqdm(range(0, len(data_source), batch_size)):
            batch_samples = data_source[i : i + batch_size]
            
            # 1. Prepare Inputs
            texts = []
            images = []
            audios = []
            
            for s in batch_samples:
                # Extract user prompt
                prompt = ""
                for m in s.get("messages", []):
                    if m["role"] == "user":
                        prompt = m["content"]
                        break
                texts.append(prompt)
                
                # Load Media if present
                if "images" in s and s["images"]:
                    try:
                        images.append(Image.open(s["images"][0]).convert("RGB"))
                    except Exception as e:
                        print(f"[Distiller] Failed to load image {s['images'][0]}: {e}")
                
                if "audio" in s and s["audio"]:
                    # For audio, we often pass the path directly to the processor 
                    # or load with torchaudio/librosa. 
                    # Qwen-Audio processor often handles paths or arrays.
                    audios.append(s["audio"][0])

            # 2. Tokenize / Process
            if self.processor:
                try:
                    # Multimodal Processing
                    process_kwargs = {"text": texts, "return_tensors": "pt", "padding": True}
                    if images: process_kwargs["images"] = images
                    if audios: process_kwargs["audios"] = audios # Check if model expects 'audio' or 'audios'
                    
                    inputs = self.processor(**process_kwargs).to(self.device)
                except Exception as e:
                    print(f"[Distiller] Processor failed: {e}. Falling back to tokenizer.")
                    inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
            else:
                inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
            
            # 3. Forward Pass
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
                # Grab the final layer hidden states (the most semantic representation)
                if hasattr(outputs, "hidden_states") and outputs.hidden_states:
                    hidden_states = outputs.hidden_states[-1] # [B, S, D]
                elif hasattr(outputs, "last_hidden_state"):
                    hidden_states = outputs.last_hidden_state
                else:
                    print("[Distiller] Error: No hidden states found in output.")
                    continue
                
                # Perform 'Mean Pooling' to get the shard's semantic essence
                mask = inputs.get('attention_mask', torch.ones(hidden_states.shape[:2])).to(self.device)
                if mask.dim() == 2:
                    mask = mask.unsqueeze(-1)
                
                # Ensure mask matches hidden_states sequence length (sometimes differs in multimodal models)
                if mask.size(1) != hidden_states.size(1):
                     mask = torch.ones((hidden_states.size(0), hidden_states.size(1), 1)).to(self.device)

                sum_h = torch.sum(hidden_states * mask, 1)
                count = torch.clamp(mask.sum(1), min=1e-9)
                shard_vectors = sum_h / count # [B, D]
                
            # 4. Store Shard to SSD
            for b_idx, sample in enumerate(batch_samples):
                text_id = f"{shard_prefix}_{i + b_idx}"
                shard_path = os.path.join(output_dir, f"{text_id}.pt")
                
                feat = shard_vectors[b_idx].cpu()
                # Dimension Unification... (existing logic)
                
                # Dimension Unification: Pad hidden_state to match student_dim if requested
                feat = shard_vectors[b_idx].cpu()
                if hasattr(self.tower, 'student_dim') and self.tower.student_dim > feat.shape[-1]:
                    padding = torch.zeros(self.tower.student_dim - feat.shape[-1])
                    feat = torch.cat([feat, padding], dim=-1)
                elif hasattr(self.tower, 'student_dim') and self.tower.student_dim < feat.shape[-1]:
                    feat = feat[:self.tower.student_dim]

                # Save both the raw text and the mathematical representation
                torch.save({
                    "text": text,
                    "hidden_state": feat,
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
    parser.add_argument("--student_dim", type=int, default=2048, help="Hidden dimension of the target student model")
    args = parser.parse_args()
    
    # Mock KnowledgeTower for standalone run (or load real if path provided)
    from .knowledge import KnowledgeTower
    tower = KnowledgeTower(student_dim=args.student_dim, device="cpu", embedding_model=args.embedding_model) # Dummy/Local instantiation for extraction
    
    distiller = KnowledgeDistiller(tower, args.teacher, device=args.device)
    
    # Use UniversalDataLoader
    from .data_loader import UniversalDataLoader
    loader = UniversalDataLoader()
    
    print(f"[Distiller] Loading data from {args.dataset} (limit={args.limit})...")
    dataset_gen = loader.load_dataset(args.dataset, limit=args.limit)
    
    samples = []
    for sample in dataset_gen:
        # Preserve full normalized sample for multimodal support
        if "messages" in sample:
            samples.append(sample)
            
    if not samples:
        print("[Distiller] Warning: No valid samples found in dataset. Using fallbacks.")
        samples = [{"messages": [{"role": "user", "content": "Explain quantum mechanics."}]}]

    print(f"[Distiller] Extracted {len(samples)} samples. Starting distillation...")
    distiller.extract_thematic_clusters(samples, output_dir=args.output, shard_prefix=args.shard_prefix)
