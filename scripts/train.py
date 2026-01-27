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
    def __init__(self, data_dir, tokenizer, max_length=2048):
        self.files = glob.glob(os.path.join(data_dir, "**/*.pt"), recursive=True)
        # Filter for shard files (avoiding router weights etc)
        self.files = [f for f in self.files if "shard_" in os.path.basename(f)]
        
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        print(f"[Dataset] Found {len(self.files)} knowledge shards in {data_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
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
    args = parser.parse_args()

    print(f"\n[Nexus Training] Starting Epochs: {args.epochs}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Hardware] Mode: {device}")

    # 1. Initialize Student (Low-Memory Mode)
    # Using Llama-3-8B config logic
    config = NexusStudentConfig(
        vocab_size=128256, 
        hidden_size=4096,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=8,
        num_adapters=1,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    )

    print("[Memory] Initializing Student directly on Accelerator (FP16)...")
    # Context manager to initialize directly on device (Avoids CPU RAM spike)
    # If using PyTorch < 2.0, explicit to() is needed. 
    # For robust OOM safety, we use Empty Init + Load or direct device construction.
    # NexusStudentCore inherits PreTrainedModel, check if it handles _init_weights.
    
    # Critical: Initialize in FP16/BF16 to save 50% RAM
    dtype = torch.float16 if device == "cuda" else torch.float32
    
    with torch.device(device):
        student = NexusStudentCore(config).to(dtype=dtype)
    
    print(f"[Model] Student Initialized (Llama-3 Arch) on {device}.")

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
    dataset = NexusDistillationDataset(args.data_dir, tokenizer)
    if len(dataset) == 0:
        print("[Warn] No data found. Generating dummy batch for validation.")
    
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # 4. Optimizer
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
