#!/usr/bin/env python3
"""
15_continued_pretraining.py
Optional Step 2: Continued Pretraining on Code Corpus

Purpose: Transfer The Stack knowledge to gpt-oss-20b without downloading 6TB
Strategy: Stream & train on filtered subset (Python/JS/TS) for 50B tokens

This step is OPTIONAL but recommended for maximum code capability.
Skip if you're only doing instruction fine-tuning.

Reference: 
- https://arxiv.org/abs/2403.08763 (Continual Pre-training)
- https://arxiv.org/abs/2305.06161 (StarCoder training)
"""

import os
import sys
import yaml
import torch
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Iterator
from itertools import islice

# ═══════════════════════════════════════════════════════════════
# LOGGING
# ═══════════════════════════════════════════════════════════════

os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/continued_pretraining.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════

def load_config(config_path: str = "config/model_config.yaml") -> Dict:
    if Path(config_path).exists():
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    return {}

yaml_config = load_config()

CONFIG = {
    # Model
    "model_name": yaml_config.get("base_model", {}).get("name", "/mnt/e/data/models/Qwen2.5-Omni-7B-GPTQ-Int4"),
    
    # Continued Pretraining Specific
    "target_tokens": 50_000_000_000,  # 50B tokens target
    "languages": ["Python", "JavaScript", "TypeScript", "CSS", "HTML", "Dockerfile"],
    
    # Training
    "max_seq_length": 4096,
    "batch_size": 1,
    "grad_accum_steps": 16,  # Larger accumulation for CPT
    "learning_rate": 5e-5,   # Lower LR for CPT (not fine-tuning)
    "warmup_steps": 1000,
    "lr_scheduler": "cosine",
    
    # LoRA (use larger rank for CPT)
    "lora_rank": 256,  # Larger rank captures more knowledge
    "lora_alpha": 512,
    
    # Checkpointing
    "output_dir": "/mnt/e/models/nexus-prime-cpt",
    "save_steps": 5000,
    "logging_steps": 100,
    
    # Streaming (to avoid downloading 6TB)
    "stream_dataset": True,
    "dataset_name": "bigcode/the-stack-dedup",
}

# ═══════════════════════════════════════════════════════════════
# IMPORTS
# ═══════════════════════════════════════════════════════════════

try:
    from unsloth import FastLanguageModel, is_bfloat16_supported
    UNSLOTH_AVAILABLE = True
except ImportError:
    UNSLOTH_AVAILABLE = False

try:
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainingArguments,
        Trainer,
        DataCollatorForLanguageModeling
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from datasets import load_dataset, IterableDataset
except ImportError as e:
    logger.error(f"Missing dependency: {e}")
    sys.exit(1)

# ═══════════════════════════════════════════════════════════════
# STREAMING DATA LOADER
# ═══════════════════════════════════════════════════════════════

def create_streaming_dataset(
    languages: list,
    max_samples: Optional[int] = None,
    local_paths: Optional[list] = None
) -> IterableDataset:
    """
    Stream code data for pretraining.
    Supports:
    1. Local 500GB+ datasets (via StreamingDatasetLoader)
    2. HuggingFace Hub (bigcode/the-stack)
    """
    
    # 1. Local Streaming (Preferred for 500GB disks)
    if local_paths:
        logger.info(f"🌊 Streaming from local paths: {local_paths}")
        try:
            from src.data.streaming_trainer import StreamingDatasetLoader, StreamingConfig
            
            config = StreamingConfig(
                buffer_size=10000,
                max_samples=max_samples
            )
            loader = StreamingDatasetLoader(local_paths, config)
            dataset = loader.get_streaming_dataset()
            
            # Add simple language filter if metadata exists
            # Note: Local datasets might not have 'lang' column unless synthesized
            return dataset
            
        except ImportError:
            logger.warning("StreamingDatasetLoader not found, falling back to HF Hub")

    # 2. HF Hub Streaming (Fallback/Default)
    logger.info(f"🌊 Streaming from HF Hub: {CONFIG['dataset_name']} ({languages})")
    
    dataset = load_dataset(
        CONFIG["dataset_name"],
        data_dir="data",
        split="train",
        streaming=True,
        trust_remote_code=True
    )
    
    # Filter by language
    def filter_by_language(example):
        return example.get("lang", "") in languages
    
    dataset = dataset.filter(filter_by_language)
    
    if max_samples:
        dataset = dataset.take(max_samples)
    
    return dataset

def tokenize_for_cpt(examples: Dict, tokenizer) -> Dict:
    """Tokenize code for continued pretraining (causal LM objective)."""
    
    # Concatenate content
    texts = examples.get("content", [])
    if isinstance(texts, str):
        texts = [texts]
    
    # Tokenize
    tokenized = tokenizer(
        texts,
        truncation=True,
        max_length=CONFIG["max_seq_length"],
        padding=False,
        return_tensors=None
    )
    
    # For causal LM, labels = input_ids
    tokenized["labels"] = tokenized["input_ids"].copy()
    
    return tokenized

# ═══════════════════════════════════════════════════════════════
# MODEL LOADING
# ═══════════════════════════════════════════════════════════════

def load_model_for_cpt():
    """Load model for continued pretraining with larger LoRA."""
    
    logger.info(f"📦 Loading model: {CONFIG['model_name']}")
    
    if UNSLOTH_AVAILABLE:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=CONFIG["model_name"],
            max_seq_length=CONFIG["max_seq_length"],
            dtype=torch.bfloat16 if is_bfloat16_supported() else torch.float16,
            load_in_4bit=True,
        )
        
        # Larger LoRA for CPT
        model = FastLanguageModel.get_peft_model(
            model,
            r=CONFIG["lora_rank"],
            lora_alpha=CONFIG["lora_alpha"],
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.0,  # No dropout for CPT
            bias="none",
            use_gradient_checkpointing="unsloth",
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_name"])
        model = AutoModelForCausalLM.from_pretrained(
            CONFIG["model_name"],
            torch_dtype=torch.bfloat16,
            device_map="auto",
            load_in_4bit=True,
        )
        
        model = prepare_model_for_kbit_training(model)
        lora_config = LoraConfig(
            r=CONFIG["lora_rank"],
            lora_alpha=CONFIG["lora_alpha"],
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, lora_config)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer

# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    logger.info("=" * 60)
    logger.info("🔬 NEXUS PRIME: Continued Pretraining (CPT)")
    logger.info("=" * 60)
    logger.info(f"   Model: {CONFIG['model_name']}")
    logger.info(f"   Target: {CONFIG['target_tokens']:,} tokens")
    logger.info(f"   Languages: {CONFIG['languages']}")
    logger.info(f"   LoRA Rank: {CONFIG['lora_rank']} (larger for CPT)")
    logger.info("=" * 60)
    logger.info("")
    logger.info("⚠️  This step is OPTIONAL but recommended for max code ability.")
    logger.info("    Estimated time: 1-2 days on single GPU")
    logger.info("    Skip if you only need instruction fine-tuning.")
    logger.info("")
    
    # Confirm with user
    response = input("Continue with CPT? (y/N): ").strip().lower()
    if response != 'y':
        logger.info("Skipping CPT. Run 14_sft_training.py for instruction tuning only.")
        return
    
    # Load model
    logger.info("\n📦 Loading model for CPT...")
    model, tokenizer = load_model_for_cpt()
    logger.info("✓ Model loaded")
    
    # Create streaming dataset
    logger.info("\n🌊 Setting up streaming dataset...")
    dataset = create_streaming_dataset(
        languages=CONFIG["languages"],
        max_samples=1_000_000  # Start with 1M samples for testing
    )
    
    # Tokenize
    dataset = dataset.map(
        lambda x: tokenize_for_cpt(x, tokenizer),
        remove_columns=["content", "lang", "path", "size", "license"]
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # Causal LM, not masked
    )
    
    # Training args (CPT specific)
    training_args = TrainingArguments(
        output_dir=CONFIG["output_dir"],
        per_device_train_batch_size=CONFIG["batch_size"],
        gradient_accumulation_steps=CONFIG["grad_accum_steps"],
        learning_rate=CONFIG["learning_rate"],
        warmup_steps=CONFIG["warmup_steps"],
        lr_scheduler_type=CONFIG["lr_scheduler"],
        logging_steps=CONFIG["logging_steps"],
        save_steps=CONFIG["save_steps"],
        bf16=True,
        optim="adamw_8bit",
        max_steps=100000,  # Adjust based on target tokens
        save_total_limit=3,
        report_to="none",
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )
    
    # Train
    logger.info("\n🚀 Starting continued pretraining...")
    trainer.train()
    
    # Save
    logger.info("\n💾 Saving CPT checkpoint...")
    trainer.save_model(CONFIG["output_dir"])
    tokenizer.save_pretrained(CONFIG["output_dir"])
    
    logger.info("\n" + "=" * 60)
    logger.info("✅ Continued Pretraining complete!")
    logger.info(f"   Checkpoint: {CONFIG['output_dir']}")
    logger.info("   Next: Run 14_sft_training.py for instruction tuning")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()
