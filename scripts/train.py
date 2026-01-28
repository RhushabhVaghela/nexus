import argparse
import os
import torch
import glob
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer

# Import Core Modules
from src.nexus_core.student.core import NexusStudentCore, NexusStudentConfig
from src.nexus_final.distill import NexusTrainer

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
            item = torch.load(path, map_location="cpu")
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
            
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
                "teacher_features": {"base": teacher_feat}, # Simple single-adapter map for now
                "teacher_logits": torch.zeros(1) # Stub for loss fn if needed
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
    parser.add_argument("--batch_size", type=int, default=1) # Low batch size for safety
    parser.add_argument("--profile_path", type=str, default="")
    parser.add_argument("--data_dir", type=str, default="memory/")
    parser.add_argument("--hidden_size", type=int, default=2048)
    parser.add_argument("--vocab_size", type=int, default=128256)
    args = parser.parse_args()

    print(f"\n[Nexus Training] Starting Epochs: {args.epochs}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Hardware] Mode: {device}")

    # 0. Hardware & Precision Optimization (Beast Mode: RTX 5080)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    
    from src.nexus_final.utils.memory import check_memory_headroom, get_recommended_batch_size
    ok, status = check_memory_headroom(vram_headroom_gb=1.0, ram_headroom_gb=2.0)
    print(f"[Memory] Initial Check: {status}")

    # 1. Initialize Student (Low-Memory Mode)
    # Right-Sized for 16GB VRAM (Sweet Spot for Distillation)
    # We slightly reduce layers and hidden size to ensure "and RAM" headroom is safe.
    config = NexusStudentConfig(
        vocab_size=args.vocab_size, 
        hidden_size=args.hidden_size,
        num_hidden_layers=16, 
        num_attention_heads=16,
        num_key_value_heads=4,
        num_adapters=1,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32
    )

    print("[Memory] Initializing Student on Accelerator (Beast Mode: BF16)...")
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    
    with torch.device(device):
        student = NexusStudentCore(config).to(dtype=dtype)
    
    # Enable Gradient Checkpointing for VRAM savings
    if device == "cuda":
        print("[Memory] Enabling Gradient Checkpointing...")
        student.gradient_checkpointing_enable()
        
        # 1.5 Compiled Execution (Beast Mode: Max Autotune)
        print("[Performance] Compiling Student Model (torch.compile)...")
        try:
            student = torch.compile(student, mode="reduce-overhead")
            print("[Performance] Kernel Fusion Active.")
        except Exception as e:
            print(f"[Warn] torch.compile failed: {e}. Falling back to eager.")

    print(f"[Model] Student Initialized (~1.6B Params) on {device}.")

    # 2. Tokenizer (Using Llama-3-8B base)
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
    dataset = NexusDistillationDataset(args.data_dir, tokenizer, hidden_size=args.hidden_size)
    if len(dataset) == 0:
        print("[Warn] No data found. Generating dummy batch for validation.")
    
    # Dynamic Batch Size Recommendation
    batch_size = args.batch_size
    if device == "cuda":
        rec_batch = get_recommended_batch_size(base_batch=args.batch_size, max_batch=8)
        if rec_batch > batch_size:
            print(f"[Memory] Increasing batch_size from {batch_size} to {rec_batch} (VRAM Headroom Detected).")
            batch_size = rec_batch

    # Aggressive Data Pipelining (Beast Mode: Intel Ultra 9)
    loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        pin_memory=True if device == "cuda" else False,
        num_workers=8 if device == "cuda" else 0,
        persistent_workers=True if device == "cuda" else False,
        prefetch_factor=4 if device == "cuda" else None
    )

    # 4. Optimizer: Use 8-bit AdamW for massive VRAM savings
    try:
        import bitsandbytes as bnb
        print("[Memory] Using 8-bit AdamW Optimizer...")
        optimizer = bnb.optim.AdamW8bit(student.parameters(), lr=1e-5)
    except ImportError:
        print("[Warn] bitsandbytes not found. Using standard AdamW.")
        optimizer = AdamW(student.parameters(), lr=1e-5)

    # 5. Trainer
    # Mock adapters for simple execution verification
    adapters = {} 
    
    trainer = NexusTrainer(
        student=student,
        adapters=adapters,
        train_loader=loader,
        val_loader=loader, # Use same for simple check
        optimizer=optimizer,
        device=device,
        config={
            "offline_distillation": False, # We are passing features directly via Dataset
            "warmup_epochs": 1
        }
    )

    # 6. Run
    trainer.train(args.epochs)
    print("[Nexus Training] Run Complete.")

if __name__ == "__main__":
    main()
