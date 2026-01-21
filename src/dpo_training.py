#!/usr/bin/env python3
"""
dpo_training.py
Direct Preference Optimization (DPO) Training

DPO is a simplified alternative to RLHF that directly optimizes on preference data
without requiring a separate reward model.

Usage:
    python src/dpo_training.py --mode=censored

References:
    - "Direct Preference Optimization" (Rafailov et al., 2023)
"""

import os
import sys
import torch
import logging
from pathlib import Path
from typing import Dict, Any

# Logging setup
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/dpo_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Import training method config
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.training_methods import TrainingMethod, get_training_config

# Import with error handling
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from datasets import load_dataset
except ImportError as e:
    logger.error(f"Missing dependency: {e}")
    logger.error("Install: pip install transformers trl peft datasets")
    sys.exit(1)

# Configuration
CONFIG = {
    "checkpoint": "/mnt/e/data/models/Qwen2.5-0.5B",  # Base or SFT checkpoint
    "preference_data": "/mnt/e/data/processed/preference",  # From 06_generate_preference_dataset.py
    "max_seq_length": 2048,
    "output_dir": "/mnt/e/data/output/dpo",
}

# Get DPO-specific config from training_methods
dpo_config = get_training_config(TrainingMethod.DPO)


def load_preference_dataset():
    """Load preference dataset with chosen/rejected pairs."""
    pref_path = Path(CONFIG["preference_data"])
    
    if not pref_path.exists():
        logger.warning(f"Preference data not found at {pref_path}")
        logger.info("Creating dummy preference dataset for testing...")
        
        # Create minimal test dataset
        return {
            "train": [
                {
                    "prompt": "What is 2+2?",
                    "chosen": "2+2 equals 4.",
                    "rejected": "I don't know."
                },
                {
                    "prompt": "Explain Python lists.",
                    "chosen": "Python lists are ordered, mutable collections that can hold items of any type.",
                    "rejected": "Lists are things."
                }
            ] * 50  # Replicate for minimum viable dataset
        }
    
    # Load real preference data
    files = list(pref_path.glob("*.jsonl"))
    if not files:
        raise ValueError(f"No preference data found in {pref_path}")
        
    # Check total size for streaming decision
    total_size = sum(f.stat().st_size for f in files)
    use_streaming = total_size > 1 * 1024 * 1024 * 1024  # >1GB auto-stream
    
    if use_streaming:
        logger.info(f"🌊 Streaming {len(files)} files ({total_size/1e9:.1f}GB)")
        try:
            from src.data.streaming_trainer import StreamingDatasetLoader, StreamingConfig
            loader = StreamingDatasetLoader([str(f) for f in files], StreamingConfig(buffer_size=5000))
            return {"train": loader.get_streaming_dataset()}
        except ImportError:
            logger.warning("Streaming loader not found. Loading full dataset.")

    dataset = load_dataset("json", data_files=[str(f) for f in files], split="train")
    logger.info(f"Loaded {len(dataset)} preference pairs from {pref_path}")
    return {"train": dataset}


def load_model_for_dpo():
    """Load model with optional LoRA for DPO training."""
    logger.info(f"Loading model: {CONFIG['checkpoint']}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        CONFIG["checkpoint"],
        trust_remote_code=True
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model in 4-bit for efficiency
    model = AutoModelForCausalLM.from_pretrained(
        CONFIG["checkpoint"],
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        load_in_4bit=True,
    )
    
    # Optional: Add LoRA for parameter efficiency
    model = prepare_model_for_kbit_training(model)
    
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    
    logger.info("Model loaded with LoRA adapters")
    return model, tokenizer


def main():
    logger.info("=" * 60)
    logger.info("🎓 DPO TRAINING - Direct Preference Optimization")
    logger.info("=" * 60)
    logger.info(f"Checkpoint: {CONFIG['checkpoint']}")
    logger.info(f"Beta: {dpo_config.beta}")
    logger.info(f"Learning Rate: {dpo_config.learning_rate}")
    logger.info("=" * 60)
    
    # Load model
    model, tokenizer = load_model_for_dpo()
    
    # Load preference data
    logger.info("Loading preference dataset...")
    datasets = load_preference_dataset()
    train_dataset = datasets["train"]
    
    if isinstance(train_dataset, list):
        from datasets import Dataset
        train_dataset = Dataset.from_list(train_dataset)
    
    logger.info(f"Training samples: {len(train_dataset)}")
    
    # DPO training config
    training_args = DPOConfig(
        output_dir=CONFIG["output_dir"],
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        num_train_epochs=dpo_config.epochs,
        learning_rate=dpo_config.learning_rate,
        beta=dpo_config.beta,  # KL penalty coefficient
        lr_scheduler_type="cosine",
        warmup_ratio=dpo_config.warmup_ratio,
        logging_steps=10,
        save_steps=100,
        bf16=True,
        max_length=CONFIG["max_seq_length"],
        max_prompt_length=512,
        report_to="none",
    )
    
    # Create DPO trainer
    logger.info("Creating DPO trainer...")
    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
    )
    
    # Train
    logger.info("Starting DPO training...")
    trainer.train()
    
    # Save
    logger.info("Saving model...")
    trainer.save_model(f"{CONFIG['output_dir']}/final")
    tokenizer.save_pretrained(f"{CONFIG['output_dir']}/final")
    
    logger.info("=" * 60)
    logger.info("✅ DPO TRAINING COMPLETE!")
    logger.info(f"Model saved to: {CONFIG['output_dir']}/final")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
