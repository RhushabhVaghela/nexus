#!/usr/bin/env python3
"""
FILE 5: 04_sft_training.py
Stage 1: Supervised Fine-Tuning (6-8 hours on RTX 5080)
Purpose: Learn output format, reasoning structure, tool calling
"""

import os
import json
import torch
import logging
from pathlib import Path
from typing import Dict, Any
from datetime import datetime

from unsloth import FastLanguageModel, is_bfloat16_supported
from transformers import TrainingArguments
from trl import SFTTrainer
from datasets import load_dataset
import wandb

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/sft_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

CONFIG = {
    "model_name": "Qwen/Qwen2.5-72B-Instruct",
    "max_seq_length": 4096,
    "lora_rank": 32,
    "lora_alpha": 32,
    "batch_size": 1,
    "grad_accum_steps": 8,
    "learning_rate": 2e-5,
    "epochs": 1,
    "warmup_steps": 100,
    "output_dir": "checkpoints/stage1_sft",
    "logging_steps": 10,
    "save_steps": 100,
    "eval_strategy": "no",
}

def format_trajectory_for_training(sample: Dict[str, Any]) -> Dict[str, str]:
    """Convert trajectory JSON to chat format for SFT."""
    trajectory = sample.get("trajectory", [])
    reasoning_parts = []
    
    for step in trajectory:
        step_type = step.get("type", "unknown")
        
        if step_type == "think":
            reasoning_parts.append(f"<think>\n{step.get('content', '')}\n</think>")
        elif step_type == "action":
            tool = step.get("tool", "unknown")
            action_input = step.get("input", "")
            desc = step.get("description", "")
            reasoning_parts.append(f"\n[Tool: {tool}]\nInput: {action_input}\nDescription: {desc}")
        elif step_type == "observation":
            result = step.get("result", "")
            reasoning_parts.append(f"\n[Observation]\nResult: {result}")
        elif step_type == "error":
            error_type = step.get("error_type", "Error")
            error_msg = step.get("error_message", "")
            reasoning_parts.append(f"\n[{error_type}]\n{error_msg}")
        elif step_type == "recovery":
            content = step.get("content", "")
            action = step.get("action", "")
            reasoning_parts.append(f"\n[Recovery Strategy]\n{content}\n[Action]\n{action}")
        elif step_type == "final_answer":
            reasoning_parts.append(f"\n[Final Answer]\n{step.get('content', '')}")
    
    reasoning = "".join(reasoning_parts)
    
    messages = [
        {
            "role": "system",
            "content": "You are an advanced reasoning AI. Solve complex tasks step-by-step with explicit thinking, tool calls, error handling, and recovery patterns."
        },
        {
            "role": "user",
            "content": sample.get("user_query", "")
        },
        {
            "role": "assistant",
            "content": reasoning
        }
    ]
    
    return {"text": messages}

def main():
    logger.info("="*70)
    logger.info("🎓 STAGE 1: SUPERVISED FINE-TUNING (SFT)")
    logger.info("="*70)
    logger.info(f"Model: {CONFIG['model_name']}")
    logger.info(f"Max Sequence Length: {CONFIG['max_seq_length']}")
    logger.info(f"LoRA Rank: {CONFIG['lora_rank']}")
    logger.info(f"Batch Size: {CONFIG['batch_size']}")
    logger.info(f"Gradient Accumulation: {CONFIG['grad_accum_steps']}")
    logger.info(f"Effective Batch: {CONFIG['batch_size'] * CONFIG['grad_accum_steps']}")
    logger.info(f"Learning Rate: {CONFIG['learning_rate']}")
    logger.info(f"Expected Duration: 6-8 hours")
    logger.info("="*70)
    
    # Check prerequisites
    if not Path("cold_start_filtered.jsonl").exists():
        logger.error("❌ cold_start_filtered.jsonl not found!")
        logger.error("   Run: python 03_validate_trajectories.py")
        return
    
    # Load model
    logger.info("\n📦 Loading model...")
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=CONFIG["model_name"],
            max_seq_length=CONFIG["max_seq_length"],
            dtype=torch.bfloat16 if is_bfloat16_supported() else torch.float32,
            load_in_4bit=True,
            gpu_memory_utilization=0.75,
        )
        logger.info(f"✓ Model loaded successfully")
        logger.info(f"  Device: {next(model.parameters()).device}")
        logger.info(f"  Dtype: {model.dtype}")
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        return
    
    # Add LoRA
    logger.info("\n🔧 Adding LoRA adapters...")
    try:
        model = FastLanguageModel.get_peft_model(
            model,
            r=CONFIG["lora_rank"],
            lora_alpha=CONFIG["lora_alpha"],
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=3407,
        )
        logger.info(f"✓ LoRA adapters added (rank={CONFIG['lora_rank']})")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"  Total parameters: {total_params:,}")
        logger.info(f"  Trainable parameters: {trainable_params:,}")
        logger.info(f"  % Trainable: {100 * trainable_params / total_params:.2f}%")
    except Exception as e:
        logger.error(f"❌ Failed to add LoRA: {e}")
        return
    
    # Load data
    logger.info("\n📂 Loading training data...")
    try:
        dataset = load_dataset("json", data_files="cold_start_filtered.jsonl", split="train")
        logger.info(f"✓ Loaded {len(dataset)} trajectories")
        
        # Show samples
        logger.info(f"  Sample domains:")
        domains = {}
        for item in dataset:
            domain = item.get("domain", "unknown")
            domains[domain] = domains.get(domain, 0) + 1
        for domain, count in sorted(domains.items()):
            logger.info(f"    {domain}: {count}")
    except Exception as e:
        logger.error(f"❌ Failed to load dataset: {e}")
        return
    
    # Format data
    logger.info("\n🔄 Formatting data for SFT...")
    try:
        def format_for_sft(sample):
            messages = format_trajectory_for_training(sample)["text"]
            formatted = tokenizer.apply_chat_template(messages, tokenize=False)
            return {"text": formatted}
        
        dataset = dataset.map(
            format_for_sft,
            remove_columns=dataset.column_names,
            desc="Formatting trajectories"
        )
        logger.info(f"✓ Data formatted successfully")
    except Exception as e:
        logger.error(f"❌ Failed to format data: {e}")
        return
    
    # Training arguments
    logger.info("\n⚙️  Configuring training...")
    training_args = TrainingArguments(
        output_dir=CONFIG["output_dir"],
        num_train_epochs=CONFIG["epochs"],
        per_device_train_batch_size=CONFIG["batch_size"],
        gradient_accumulation_steps=CONFIG["grad_accum_steps"],
        learning_rate=CONFIG["learning_rate"],
        lr_scheduler_type="cosine",
        warmup_steps=CONFIG["warmup_steps"],
        logging_steps=CONFIG["logging_steps"],
        save_steps=CONFIG["save_steps"],
        save_total_limit=3,
        bf16=is_bfloat16_supported(),
        optim="adamw_8bit",
        seed=3407,
        report_to=["wandb"] if "WANDB_API_KEY" in os.environ else [],
        remove_unused_columns=False,
        max_grad_norm=1.0,
    )
    
    logger.info("✓ Training arguments configured")
    
    # Create trainer
    logger.info("\n🎓 Creating trainer...")
    try:
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            train_dataset=dataset,
            packing=False,
        )
        logger.info("✓ Trainer created")
    except Exception as e:
        logger.error(f"❌ Failed to create trainer: {e}")
        return
    
    # Train
    logger.info("\n🚀 Starting training...")
    logger.info("   Monitor GPU: nvidia-smi -l 1")
    logger.info("   Monitor loss: tail -f logs/sft_training.log")
    
    try:
        train_result = trainer.train()
        logger.info(f"\n✅ Training complete!")
        logger.info(f"   Final loss: {train_result.training_loss:.4f}")
        logger.info(f"   Training time: {train_result.training_time:.2f}s")
    except KeyboardInterrupt:
        logger.warning("⚠️  Training interrupted by user")
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        return
    
    # Save model
    logger.info("\n💾 Saving model...")
    try:
        output_path = f"{CONFIG['output_dir']}/final"
        model.save_pretrained(output_path)
        tokenizer.save_pretrained(output_path)
        logger.info(f"✓ Model saved to: {output_path}")
        
        # Save config
        config_path = Path(output_path) / "sft_config.json"
        with open(config_path, "w") as f:
            json.dump(CONFIG, f, indent=2)
        logger.info(f"✓ Config saved")
    except Exception as e:
        logger.error(f"❌ Failed to save model: {e}")
        return
    
    logger.info("\n" + "="*70)
    logger.info("✅ STAGE 1 COMPLETE!")
    logger.info("="*70)
    logger.info(f"Next: Run Stage 2 (Rejection Sampling)")
    logger.info(f"  python 05_rejection_sampling.py")
    logger.info("="*70)

if __name__ == "__main__":
    main()
