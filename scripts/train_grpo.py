#!/usr/bin/env python3
"""
scripts/train_grpo.py
Stage 3: GRPO (Group Relative Policy Optimization) for Reasoning.
Optimized for the Nexus Pipeline with Unsloth support.
"""

import argparse
import os
import sys

# Ensure unsloth is imported before any other heavy libraries if possible
try:
    from unsloth import FastLanguageModel, is_bfloat16_supported
    UNSLOTH_AVAILABLE = True
except ImportError:
    UNSLOTH_AVAILABLE = False

import torch
import logging
from pathlib import Path

# Ensure 'src' is in path before importing from it
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.join(BASE_DIR, 'src') not in sys.path:
    sys.path.append(os.path.join(BASE_DIR, 'src'))

try:
    from trl import GRPOConfig, GRPOTrainer
    from datasets import load_dataset
    from transformers import AutoTokenizer, TrainingArguments
except ImportError as e:
    print(f"[Error] Missing dependencies for GRPO: {e}")
    sys.exit(1)

# Import Reward Functions from src/12_grpo_training.py (simulated as standalone here for portability)
def correctness_reward(completions: list, answers: list, **kwargs) -> list:
    rewards = []
    for completion, answer in zip(completions, answers):
        completion_lower = completion.lower().strip()
        answer_lower = str(answer).lower().strip()
        if completion_lower == answer_lower or answer_lower in completion_lower:
            rewards.append(1.0)
        elif len(str(answer)) > 0 and any(word in completion_lower for word in str(answer).split()):
            rewards.append(0.7)
        else:
            rewards.append(0.0)
    return rewards

def quality_reward(completions: list, **kwargs) -> list:
    rewards = []
    for completion in completions:
        score = 0.0
        if "<think>" in completion: score += 0.3
        if "[Tool:" in completion or "[Final Answer]" in completion: score += 0.4
        words = len(completion.split())
        if 50 < words < 1000: score += 0.3
        rewards.append(min(1.0, score))
    return rewards

def combined_reward(completions: list, answers: list = None, **kwargs) -> list:
    correctness = correctness_reward(completions, answers or [""] * len(completions))
    quality = quality_reward(completions)
    return [(0.6 * c + 0.4 * q) for c, q in zip(correctness, quality)]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, default="rejection_sampled.jsonl")
    parser.add_argument("--output_dir", type=str, default="checkpoints/nexus_grpo")
    parser.add_argument("--max_seq_length", type=int, default=4096)
    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_generations", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--use_unsloth", action="store_true")
    args = parser.parse_args()

    print(f"\n[GRPO Trainer] Loading {args.model_path}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.use_unsloth and UNSLOTH_AVAILABLE:
        print("[Unsloth] Initializing GRPO with Unsloth speedups...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=args.model_path,
            max_seq_length=args.max_seq_length,
            dtype=torch.bfloat16 if is_bfloat16_supported() else torch.float32,
            load_in_4bit=True,
        )
        model = FastLanguageModel.get_peft_model(
            model,
            r=args.lora_rank,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_alpha=64,
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=3407,
        )
    else:
        print("[Transformers] Loading model for standard GRPO...")
        from transformers import AutoModelForCausalLM
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16 if is_bfloat16_supported() else torch.float32,
            device_map="auto",
            load_in_4bit=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    # Load and Prepare Data
    print(f"[Data] Loading {args.data_path}...")
    dataset = load_dataset("json", data_files=args.data_path, split="train")
    
    def prepare_grpo(sample):
        return {
            "prompt": f"Question: {sample.get('question', sample.get('prompt', ''))}\n\nLet me think:\n<think>",
            "answer": sample.get("answer", ""),
        }
    dataset = dataset.map(prepare_grpo)

    # GRPO Config
    grpo_args = GRPOConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=10,
        logging_steps=1,
        save_steps=50,
        bf16=is_bfloat16_supported(),
        max_grad_norm=1.0,
        num_generations=args.num_generations,
    )

    print("[Trainer] Starting GRPO loop...")
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[combined_reward],
        args=grpo_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )

    trainer.train()
    
    print(f"[Done] GRPO training complete. Saved to {args.output_dir}")
    model.save_pretrained(os.path.join(args.output_dir, "final"))
    tokenizer.save_pretrained(os.path.join(args.output_dir, "final"))

if __name__ == "__main__":
    main()
