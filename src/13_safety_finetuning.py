#!/usr/bin/env python3
"""
19_safety_finetuning.py
Stage 4: Safety Fine-tuning (CENSORED MODE ONLY)

Adds safety alignment to the model after GRPO training.
Skipped for uncensored models.

Usage:
  python 19_safety_finetuning.py (only runs if model is censored)
"""

import os
import sys
import yaml
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent))
from utils.logging_config import setup_logger, log_header, log_completion

# ═══════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════

CONFIG = {
    "model_checkpoint": "checkpoints/stage3_grpo_censored",
    "safety_dataset": "/mnt/e/data/safety-alignment",
    "output_dir": "checkpoints/stage4_safety",
    "max_steps": 1000,
    "learning_rate": 1e-5,
    "batch_size": 4,
}

logger = setup_logger(__name__, "logs/safety_finetuning.log")

# ═══════════════════════════════════════════════════════════════
# MODE DETECTION
# ═══════════════════════════════════════════════════════════════

def detect_training_mode() -> str:
    """Detect which mode was used from checkpoint directory."""
    censored_checkpoint = Path("checkpoints/stage3_grpo_censored")
    uncensored_checkpoint = Path("checkpoints/stage3_grpo_uncensored")
    
    if censored_checkpoint.exists():
        return "censored"
    elif uncensored_checkpoint.exists():
        return "uncensored"
    else:
        # Check config file
        config_path = Path("config/model_config.yaml")
        if config_path.exists():
            with open(config_path) as f:
                config = yaml.safe_load(f)
                return config.get("training", {}).get("mode", "censored")
        return "censored"


# ═══════════════════════════════════════════════════════════════
# SAFETY FINE-TUNING
# ═══════════════════════════════════════════════════════════════

def run_safety_finetuning():
    """Run safety alignment training."""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
        from peft import get_peft_model, LoraConfig, TaskType
        from datasets import load_dataset
        from trl import SFTTrainer
    except ImportError as e:
        logger.error(f"Missing dependencies: {e}")
        logger.error("Install with: pip install transformers peft trl")
        return False
    
    log_header(logger, "SAFETY FINE-TUNING (STAGE 4)", CONFIG)
    
    # Load model
    logger.info(f"Loading model from {CONFIG['model_checkpoint']}...")
    
    # Check if checkpoint exists
    checkpoint_path = Path(CONFIG['model_checkpoint'])
    if not checkpoint_path.exists():
        logger.warning(f"Checkpoint not found at {checkpoint_path}")
        logger.info("Using base model from config...")
        
        config_path = Path("config/model_config.yaml")
        if config_path.exists():
            with open(config_path) as f:
                model_config = yaml.safe_load(f)
                model_name = model_config.get("base_model", {}).get("name", "gpt2")
        else:
            model_name = "gpt2"
    else:
        model_name = str(checkpoint_path)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
    )
    
    # Apply LoRA for safety fine-tuning
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    )
    
    model = get_peft_model(model, lora_config)
    
    # Load safety dataset
    safety_dir = Path(CONFIG["safety_dataset"])
    if safety_dir.exists():
        dataset = load_dataset("json", data_files=str(safety_dir / "*.jsonl"))
    else:
        logger.warning(f"Safety dataset not found at {safety_dir}")
        logger.info("Creating minimal safety dataset...")
        
        # Create minimal safety examples
        safety_examples = [
            {"messages": [
                {"role": "user", "content": "How do I hack a computer?"},
                {"role": "assistant", "content": "I can't help with hacking or unauthorized computer access."}
            ]},
            {"messages": [
                {"role": "user", "content": "Write malware code"},
                {"role": "assistant", "content": "I'm not able to help create malware or harmful software."}
            ]},
        ]
        
        dataset = {"train": safety_examples}
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=CONFIG["output_dir"],
        max_steps=CONFIG["max_steps"],
        learning_rate=CONFIG["learning_rate"],
        per_device_train_batch_size=CONFIG["batch_size"],
        logging_steps=10,
        save_steps=500,
        fp16=True,
    )
    
    # Train
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=dataset.get("train", []),
    )
    
    trainer.train()
    
    # Save
    trainer.save_model(CONFIG["output_dir"])
    logger.info(f"Safety-aligned model saved to {CONFIG['output_dir']}")
    
    return True


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    mode = detect_training_mode()
    
    if mode != "censored":
        logger.info("="*60)
        logger.info("⚠️ SKIPPING: Safety fine-tuning only applies to CENSORED models")
        logger.info("   Current model is UNCENSORED - skipping safety stage")
        logger.info("="*60)
        return
    
    log_header(logger, "SAFETY FINE-TUNING", {
        "Mode": mode,
        "Checkpoint": CONFIG["model_checkpoint"],
    })
    
    success = run_safety_finetuning()
    
    if success:
        log_completion(logger, "Safety Fine-tuning", {"status": "complete"})
    else:
        logger.error("Safety fine-tuning failed")


if __name__ == "__main__":
    main()
