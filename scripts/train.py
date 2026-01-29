import argparse
import os
import sys

# Ensure unsloth is imported before any other heavy libraries if possible
try:
    # Disable Unsloth background telemetry which can hang during initial loading
    os.environ["UNSLOTH_OFFLINE"] = "1"
    import unsloth
except ImportError:
    pass

import json
import torch
import glob
import gc

# Ensure 'src' is in path before importing from it
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.join(BASE_DIR, 'src') not in sys.path:
    sys.path.append(os.path.join(BASE_DIR, 'src'))

from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer

# Import Core Modules
from nexus_core.student.core import NexusStudentCore, NexusStudentConfig
from nexus_final.distill import NexusTrainer
from utils.callbacks import KeyboardPauseCallback

class NexusDistillationDataset(Dataset):
    """
    Loads Knowledge Shards produced by Stage 1.5 (Extraction).
    """
    def __init__(self, data_dir, tokenizer, max_length=2048, hidden_size=2048):
        self.data_dir = data_dir
        self.files = glob.glob(os.path.join(self.data_dir, "**/*.pt"), recursive=True)
        # Filter for shard files (avoiding router weights etc)
        self.files = [f for f in self.files if "shard_" in os.path.basename(f)]
        
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.hidden_size = hidden_size
        
        print(f"[Dataset] Found {len(self.files)} knowledge shards in {data_dir}")

    def __len__(self):
        return max(1, len(self.files))

    def __getitem__(self, idx):
        if not self.files:
            # Yield dummy sample for verification
            print("[Dataset] Empty directory. Yielding dummy sample for verification...")
            input_ids = torch.zeros(self.max_length, dtype=torch.long)
            return {
                "input_ids": input_ids,
                "attention_mask": torch.ones(self.max_length, dtype=torch.long),
                "labels": input_ids.clone(),
                "teacher_features": {"base": torch.zeros(self.hidden_size)},
                "teacher_logits": torch.zeros(1)
            }
            
        path = self.files[idx]
        try:
            # Shard contains: {'text': str, 'hidden_state': Tensor, ...}
            # Use weights_only=True for safety and performance
            item = torch.load(path, map_location="cpu", weights_only=True)
            text = item.get("text", "")
            teacher_feat = item.get("hidden_state", None) # [D]
            
            # Tokenize
            tokens = self.tokenizer(
                text, 
                max_length=self.max_length, 
                padding="max_length", 
                truncation=True, 
                return_tensors="pt"
            )
            
            input_ids = tokens.input_ids.squeeze(0)
            attention_mask = tokens.attention_mask.squeeze(0)
            labels = input_ids.clone() # Causal LM
            
            # Standardize teacher dimension to 4096 (Universal Latent Buffer)
            # This allows batching shards from different teachers (e.g. 3584, 3072)
            MAX_DIM = 4096
            standardized_feat = torch.zeros(MAX_DIM)
            if teacher_feat is not None:
                d = min(teacher_feat.shape[0], MAX_DIM)
                standardized_feat[:d] = teacher_feat[:d]
            
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
                "teacher_features": {"base": standardized_feat}, 
                "teacher_logits": torch.zeros(1)
            }
        except Exception as e:
            print(f"Error loading {path}: {e}")
            # Return dummy to avoid crash
            return {
                "input_ids": torch.zeros(self.max_length, dtype=torch.long),
                "labels": torch.zeros(self.max_length, dtype=torch.long),
                "teacher_features": {}
            }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=2) # Micro-batch size (keep small for VRAM)
    parser.add_argument("--grad_accum", type=int, default=16) # Accumulate to effective batch 32
    parser.add_argument("--profile_path", type=str, default="")
    parser.add_argument("--data_dir", type=str, default="memory/")
    parser.add_argument("--hidden_size", type=int, default=2048)
    parser.add_argument("--vocab_size", type=int, default=128256)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--use_unsloth", action="store_true", help="Enable Unsloth for 3x speedup (if available and architecture supported)")
    parser.add_argument("--packing", action="store_true", help="Enable sequence packing for higher throughput")
    parser.add_argument("--max_seq_length", type=int, default=2048, help="Maximum sequence length")
    args = parser.parse_args()

    print(f"\n[Nexus Training] Starting Epochs: {args.epochs}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Hardware] Mode: {device}")

    # 0. Hardware & Precision Optimization (Beast Mode: RTX 5080)
    if device == "cuda":
        torch.set_float32_matmul_precision('high')
        torch.backends.cudnn.benchmark = True
    
    from nexus_final.utils.memory import check_memory_headroom, get_recommended_batch_size
    ok, status = check_memory_headroom(vram_headroom_gb=1.0, ram_headroom_gb=2.0)
    print(f"[Memory] Initial Check: {status}")

    # 1. Initialize Student (Low-Memory Mode)
    # RATIONALE: Attempt to use Unsloth if requested, otherwise fallback to NexusStudentCore
    student = None
    tokenizer = None
    
    if args.use_unsloth:
        try:
            from unsloth import FastLanguageModel
            print(f"[Unsloth] Initializing compatible base for Unsloth optimizations (Max Len: {args.max_seq_length})...")
            # For Unsloth, we typically load a known base model. 
            # If the user is doing nexus training, we use the Llama-3-8b base as the skeleton.
            model_name = "unsloth/llama-3-8b-bnb-4bit" # Optimized default
            student, tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_name,
                max_seq_length=args.max_seq_length,
                dtype=torch.bfloat16 if device == "cuda" else torch.float32,
                load_in_4bit=True,
                trust_remote_code=True
            )
            print("[Unsloth] FastLanguageModel loaded successfully.")
            
            # Add LoRA Adapters (Standard for Unsloth)
            student = FastLanguageModel.get_peft_model(
                student,
                r=64,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                lora_alpha=128,
                lora_dropout=0, # Unsloth optimized
                bias="none",
                use_gradient_checkpointing="unsloth",
                random_state=3407,
            )
        except Exception as e:
            print(f"[Warn] Unsloth failed to initialize: {e}. Falling back to standard Nexus core.")

    if student is None:
        config = NexusStudentConfig(
            vocab_size=args.vocab_size, 
            hidden_size=args.hidden_size,
            num_hidden_layers=16, 
            num_attention_heads=16,
            num_key_value_heads=4,
            num_adapters=1,
            torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
            attn_implementation="eager", # Custom architecture requires eager to avoid SDPA dispatch errors
            max_position_embeddings=args.max_seq_length
        )

        print("[Memory] Initializing Student on Accelerator (Beast Mode: BF16)...")
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        
        with torch.device(device):
            student = NexusStudentCore(config).to(dtype=dtype)
        
        # Enable Gradient Checkpointing for VRAM savings
        if device == "cuda":
            print("[Memory] Enabling Gradient Checkpointing...")
            student.gradient_checkpointing_enable()
            
            # 1.5 Compiled Execution (Skip by default, as 'reduce-overhead' causes CUDAGraph errors with custom loop)
            # if not args.use_unsloth:
            #     print("[Performance] Compiling Student Model (torch.compile)...")
            #     try:
            #         student = torch.compile(student, mode="reduce-overhead")
            #         print("[Performance] Kernel Fusion Active.")
            #     except Exception as e:
            #         print(f"[Warn] torch.compile failed: {e}. Falling back to eager.")
            if args.use_unsloth:
                print("[Performance] Unsloth Kernels Active (Skipping torch.compile).")

    print(f"[Model] Student Initialized on {device}.")

    # 2. Tokenizer (Using Llama-3-8B base)
    if tokenizer is None:
        try:
            tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
        except:
            # Fallback if Llama 3 not found locally/access token issue
            print("[Warn] Llama 3 Tokenizer not found. Using generic.")
            from transformers import GPT2Tokenizer
            tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            tokenizer.pad_token = tokenizer.eos_token

    # 3. Dataset
    dataset = NexusDistillationDataset(args.data_dir, tokenizer, max_length=args.max_seq_length, hidden_size=args.hidden_size)
    if len(dataset) == 0:
        print("[Warn] No data found. Generating dummy batch for validation.")
    
    # Dynamic Batch Size Recommendation - DISABLED for Stability
    batch_size = args.batch_size
    print(f"[Memory] Using Fixed Micro-Batch Size: {batch_size}")
    print(f"[Memory] Gradient Accumulation: {args.grad_accum} steps (Effective Batch: {batch_size * args.grad_accum})")

    # Aggressive Data Pipelining - RAM Optimized (Intel Ultra 9)
    loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        pin_memory=True if device == "cuda" else False,
        num_workers=0,
        persistent_workers=False
    )

    # 4. Optimizer: Use 8-bit AdamW for massive VRAM savings
    try:
        import bitsandbytes as bnb
        print(f"[Memory] Using 8-bit Paged AdamW Optimizer (LR: {args.lr})...")
        optimizer = bnb.optim.AdamW8bit(student.parameters(), lr=args.lr, is_paged=True)
    except ImportError:
        print(f"[Warn] bitsandbytes not found. Using standard AdamW (LR: {args.lr}).")
        print("       (Installing bitsandbytes is HIGHLY recommended for 16GB VRAM)")
        optimizer = AdamW(student.parameters(), lr=args.lr)

    # 5. Trainer Initialization
    # RATIONALE:
    # - 'adapters' are empty because NexusStudentCore (above) already integrates
    #   its own Internal Adapters and Routing. External mocking isn't needed here.
    # - 'offline_distillation' is False because the NexusDistillationDataset (above)
    #   already handles loading shards from SSD into the batch efficiently.
    adapters = {} 
    
    # 5.1 Load Distillation Profile (Critical Layers)
    critical_layers = []
    if args.profile_path and os.path.exists(args.profile_path):
        try:
            with open(args.profile_path, 'r') as f:
                profile_data = json.load(f)
                critical_layers = profile_data.get("critical_layers", [])
                print(f"[Distillation] Activation Anchoring active for layers: {critical_layers}")
        except Exception as e:
            print(f"[Warn] Failed to load profile from {args.profile_path}: {e}")
    else:
        print("[Distillation] Warning: No profile found. Proceeding with standard Mean-Squared Error distillation (No Anchoring).")

    trainer = NexusTrainer(
        student=student,
        adapters=adapters,
        train_loader=loader,
        val_loader=loader, 
        optimizer=optimizer,
        device=device,
        config={
            "offline_distillation": False, 
            "warmup_epochs": 1,
            "gradient_accumulation_steps": args.grad_accum,
            "critical_layers": critical_layers
        }
    )

    # 6. Run
    trainer.train(args.epochs)
    print("[Nexus Training] Run Complete.")

if __name__ == "__main__":
    main()
