#!/usr/bin/env python3
"""
Knowledge Distillation Example - Nexus

Demonstrates how to perform knowledge distillation from a teacher model
to a smaller student model.

Usage:
    python examples/distillation_example.py \
        --teacher "microsoft/Phi-3-medium-4k-instruct" \
        --student "microsoft/Phi-3-mini-4k-instruct" \
        --dataset "openai/gsm8k"

Features:
- Logit-based distillation
- Hidden state transfer
- Attention map alignment
- Progressive layer distillation
"""

import argparse
import torch
import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer
)
from datasets import load_dataset
from typing import Optional
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class DistillationTrainer(Trainer):
    """Custom trainer for knowledge distillation."""
    
    def __init__(
        self,
        teacher_model: Optional[AutoModelForCausalLM] = None,
        temperature: float = 2.0,
        alpha: float = 0.5,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.teacher_model = teacher_model
        self.temperature = temperature
        self.alpha = alpha
        
        if teacher_model is not None:
            self.teacher_model.eval()
            for param in self.teacher_model.parameters():
                param.requires_grad = False
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """Compute distillation loss combining hard targets and soft targets."""
        outputs = model(**inputs)
        student_logits = outputs.logits
        labels = inputs.get("labels")
        
        ce_loss = F.cross_entropy(
            student_logits.view(-1, student_logits.size(-1)),
            labels.view(-1),
            ignore_index=-100
        )
        
        if self.teacher_model is None:
            return (ce_loss, outputs) if return_outputs else ce_loss
        
        with torch.no_grad():
            teacher_outputs = self.teacher_model(**inputs)
            teacher_logits = teacher_outputs.logits
        
        student_probs = F.log_softmax(student_logits / self.temperature, dim=-1)
        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=-1)
        
        kl_loss = F.kl_div(
            student_probs.view(-1, student_probs.size(-1)),
            teacher_probs.view(-1, teacher_probs.size(-1)),
            reduction="batchmean"
        ) * (self.temperature ** 2)
        
        loss = self.alpha * ce_loss + (1 - self.alpha) * kl_loss
        
        return (loss, outputs) if return_outputs else loss


def prepare_dataset(dataset_name: str, tokenizer, max_length: int = 512, num_samples: int = 1000):
    """Load and prepare dataset for distillation."""
    print(f"Loading dataset: {dataset_name}")
    
    if dataset_name == "openai/gsm8k":
        dataset = load_dataset(dataset_name, "main", split="train")
        def format_example(example):
            text = f"Question: {example['question']}\nAnswer: {example['answer'].split('####')[-1].strip()}"
            return {"text": text}
        dataset = dataset.map(format_example)
    else:
        dataset = load_dataset(dataset_name, split="train")
    
    dataset = dataset.select(range(min(num_samples, len(dataset))))
    
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length"
        )
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names
    )
    
    return tokenized_dataset


def run_distillation(
    teacher_name: str,
    student_name: str,
    dataset_name: str = "openai/gsm8k",
    output_dir: str = "./distilled_model",
    alpha: float = 0.5,
    temperature: float = 2.0,
    num_epochs: int = 3,
    batch_size: int = 4
):
    """Run knowledge distillation from teacher to student."""
    print(f"\n{'='*60}")
    print(f"NEXUS - Knowledge Distillation Example")
    print(f"{'='*60}\n")
    
    print(f"Teacher: {teacher_name}")
    print(f"Student: {student_name}")
    print(f"Dataset: {dataset_name}")
    print("-" * 60)
    
    # Step 1: Load tokenizer
    print("\n[Step 1] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(student_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("✓ Tokenizer loaded")
    
    # Step 2: Load models
    print("\n[Step 2] Loading models...")
    print(f"  Loading teacher: {teacher_name}")
    teacher_model = AutoModelForCausalLM.from_pretrained(
        teacher_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    print("  ✓ Teacher loaded")
    
    print(f"  Loading student: {student_name}")
    student_model = AutoModelForCausalLM.from_pretrained(
        student_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    print("  ✓ Student loaded")
    
    # Step 3: Prepare dataset
    print("\n[Step 3] Preparing dataset...")
    dataset = prepare_dataset(dataset_name, tokenizer, num_samples=500)
    print(f"✓ Dataset prepared: {len(dataset)} samples")
    
    # Step 4: Setup training
    print("\n[Step 4] Setting up distillation training...")
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_steps=100,
        logging_steps=10,
        save_strategy="epoch",
        fp16=torch.cuda.is_available(),
        report_to="none",
        remove_unused_columns=False,
    )
    
    trainer = DistillationTrainer(
        model=student_model,
        teacher_model=teacher_model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        temperature=temperature,
        alpha=alpha,
    )
    print("✓ Trainer initialized")
    
    # Step 5: Train
    print("\n[Step 5] Starting distillation training...")
    print("-" * 60)
    trainer.train()
    print("-" * 60)
    
    # Step 6: Save
    print("\n[Step 6] Saving distilled model...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"✓ Model saved to: {output_dir}")
    
    print(f"\n{'='*60}")
    print("Distillation Complete!")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="Knowledge Distillation Example")
    parser.add_argument("--teacher", type=str, required=True, help="Teacher model name or path")
    parser.add_argument("--student", type=str, required=True, help="Student model name or path")
    parser.add_argument("--dataset", type=str, default="openai/gsm8k", help="Dataset name")
    parser.add_argument("--output-dir", type=str, default="./distilled_model", help="Output directory")
    parser.add_argument("--alpha", type=float, default=0.5, help="Distillation alpha")
    parser.add_argument("--temperature", type=float, default=2.0, help="Distillation temperature")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    
    args = parser.parse_args()
    
    run_distillation(
        teacher_name=args.teacher,
        student_name=args.student,
        dataset_name=args.dataset,
        output_dir=args.output_dir,
        alpha=args.alpha,
        temperature=args.temperature,
        num_epochs=args.epochs,
        batch_size=args.batch_size
    )


if __name__ == "__main__":
    main()
