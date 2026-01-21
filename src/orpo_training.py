#!/usr/bin/env python3
"""
orpo_training.py
Odds Ratio Preference Optimization (ORPO) Training

ORPO combines SFT and preference optimization in a single step,
eliminating the need for a reference model.

Usage:
    python src/orpo_training.py --mode=censored

References:
    - "ORPO: Monolithic Preference Optimization without Reference Model" (2024)
"""

import os
import sys
import torch
import logging
from pathlib import Path

# Logging setup
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/orpo_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.training_methods import TrainingMethod, get_training_config

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import ORPOConfig, ORPOTrainer
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from datasets import load_dataset, Dataset
except ImportError as e:
    logger.error(f"Missing dependency: {e}")
    sys.exit(1)

CONFIG = {
    "checkpoint": "/mnt/e/data/models/Qwen2.5-0.5B",
    "preference_data": "/mnt/e/data/processed/preference",
    "max_seq_length": 2048,
    "output_dir": "/mnt/e/data/output/orpo",
}

orpo_config = get_training_config(TrainingMethod.ORPO)


def load_preference_dataset():
    """Load preference dataset with chosen/rejected pairs."""
    pref_path = Path(CONFIG["preference_data"])
    
    if pref_path.exists():
        files = list(pref_path.glob("*.jsonl"))
        if files:
            # Check total size for streaming decision
            total_size = sum(f.stat().st_size for f in files)
            use_streaming = total_size > 1 * 1024 * 1024 * 1024  # >1GB auto-stream
            
            if use_streaming:
                logger.info(f"🌊 Streaming {len(files)} files ({total_size/1e9:.1f}GB)")
                try:
                    from src.data.streaming_trainer import StreamingDatasetLoader, StreamingConfig
                    loader = StreamingDatasetLoader([str(f) for f in files], StreamingConfig(buffer_size=5000))
                    return loader.get_streaming_dataset()
                except ImportError:
                    logger.warning("Streaming loader not found. Loading full dataset.")
            
            dataset = load_dataset("json", data_files=[str(f) for f in files], split="train")
            logger.info(f"Loaded {len(dataset)} preference pairs")
            return dataset
    
    # Fallback test data
    logger.warning("Using dummy preference data for testing")
    return Dataset.from_list([
        {"prompt": "What is AI?", "chosen": "AI is artificial intelligence.", "rejected": "I don't know."},
        {"prompt": "Explain Python.", "chosen": "Python is a programming language.", "rejected": "It's a snake."},
    ] * 50)


def load_model():
    """Load model with LoRA for ORPO training."""
    logger.info(f"Loading: {CONFIG['checkpoint']}")
    
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["checkpoint"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        CONFIG["checkpoint"],
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        load_in_4bit=True,
    )
    
    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=16, lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.1, bias="none", task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    
    return model, tokenizer


def main():
    logger.info("=" * 60)
    logger.info("🎓 ORPO TRAINING - Odds Ratio Preference Optimization")
    logger.info("=" * 60)
    
    model, tokenizer = load_model()
    train_dataset = load_preference_dataset()
    
    training_args = ORPOConfig(
        output_dir=CONFIG["output_dir"],
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        num_train_epochs=orpo_config.epochs,
        learning_rate=orpo_config.learning_rate,
        beta=orpo_config.beta,  # ORPO lambda parameter
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        logging_steps=10,
        save_steps=100,
        bf16=True,
        max_length=CONFIG["max_seq_length"],
        report_to="none",
    )
    
    trainer = ORPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
    )
    
    logger.info("Starting ORPO training...")
    trainer.train()
    
    trainer.save_model(f"{CONFIG['output_dir']}/final")
    tokenizer.save_pretrained(f"{CONFIG['output_dir']}/final")
    
    logger.info("=" * 60)
    logger.info("✅ ORPO TRAINING COMPLETE!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
