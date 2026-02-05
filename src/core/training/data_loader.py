import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from transformers import AutoTokenizer

class NexusDataset(Dataset):
    def __init__(self, parquet_path, tokenizer, max_length=1024):
        self.data = pd.read_parquet(parquet_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        question = row['question']
        answer = row['answer']
        
        # Format: Question + standard CoT or just raw?
        # For distillation, we ideally want the Teacher to generate the thought.
        # But for offline distillation (if we had traces), we'd load traces.
        # Here, do we force generation? 
        # No, "Activation-Guided Consolidation" usually implies run-time generation 
        # OR we just feed the Prompt and FORCE the teacher to generate? 
        # Activations are needed *per token* or *per sequence*.
        
        # If we just feed the question, the student predicts the first token of answer?
        # Or do we feed (Q + A) and teacher/student process it (Teacher Forcing)?
        # Yes, standard SFT style: Feed full seq, mask question loss.
        # Teacher provides activations for the full sequence.
        
        full_text = f"Question: {question}\nAnswer: {answer}"
        
        enc = self.tokenizer(
            full_text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        
        return {
            "input_ids": enc.input_ids.squeeze(0),
            "attention_mask": enc.attention_mask.squeeze(0)
        }

def get_dataloader(parquet_path, tokenizer_path, batch_size=2):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    dataset = NexusDataset(parquet_path, tokenizer)
    
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
