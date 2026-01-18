#!/usr/bin/env python3
"""
14_sft_training.py
Stage 1: Supervised Fine-Tuning (SFT)

ARCHITECTURE-AGNOSTIC: Works with ANY HuggingFace model.
Configure via config/model_config.yaml

Features:
- Universal model loading (GPT, LLaMA, Qwen, DeepSeek, etc.)
- LoRA/QLoRA fine-tuning
- Mixed data training (30% real + 70% synthetic)
- Gradient checkpointing for memory efficiency
- WandB integration (optional)
"""

import os
import sys
import yaml
import torch
import logging
import glob
from pathlib import Path
from typing import Dict, Any, Optional

# ═══════════════════════════════════════════════════════════════
# LOGGING SETUP
# ═══════════════════════════════════════════════════════════════

os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/sft_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════
# CONFIGURATION LOADING
# ═══════════════════════════════════════════════════════════════

def load_config(config_path: str = "config/model_config.yaml") -> Dict:
    """Load configuration from YAML file."""
    if Path(config_path).exists():
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    else:
        logger.warning(f"Config file not found: {config_path}, using defaults")
        return {}

# Load config
yaml_config = load_config()

# Merge with defaults
CONFIG = {
    # Model - from YAML or default
    "model_name": yaml_config.get("base_model", {}).get("name", "openai/gpt-oss-20b"),
    "torch_dtype": yaml_config.get("base_model", {}).get("torch_dtype", "auto"),
    "trust_remote_code": yaml_config.get("base_model", {}).get("trust_remote_code", True),
    
    # LoRA
    "lora_enabled": yaml_config.get("lora", {}).get("enabled", True),
    "lora_rank": yaml_config.get("lora", {}).get("rank", 64),
    "lora_alpha": yaml_config.get("lora", {}).get("alpha", 128),
    "lora_dropout": yaml_config.get("lora", {}).get("dropout", 0.05),
    "target_modules": yaml_config.get("lora", {}).get("target_modules", 
        ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]),
    
    # Training
    "epochs": yaml_config.get("training", {}).get("num_epochs", 3),
    "batch_size": yaml_config.get("training", {}).get("per_device_train_batch_size", 2),
    "grad_accum_steps": yaml_config.get("training", {}).get("gradient_accumulation_steps", 8),
    "learning_rate": yaml_config.get("training", {}).get("learning_rate", 2e-4),
    "max_seq_length": yaml_config.get("training", {}).get("max_seq_length", 4096),
    "warmup_ratio": yaml_config.get("training", {}).get("warmup_ratio", 0.03),
    "gradient_checkpointing": yaml_config.get("training", {}).get("gradient_checkpointing", True),
    "bf16": yaml_config.get("training", {}).get("bf16", True),
    "logging_steps": yaml_config.get("training", {}).get("logging_steps", 1),
    "save_steps": yaml_config.get("training", {}).get("save_steps", 5),
    "eval_steps": yaml_config.get("training", {}).get("eval_steps", 500),
    
    # Data
    "mixed_data_dir": yaml_config.get("data", {}).get("mixed_data_dir", "/mnt/e/data/mixed-training"),
    "synthetic_data_dir": "/mnt/e/data/finetuned-fullstack-dataset",
    
    # Output
    "output_dir": yaml_config.get("output", {}).get("dir", "/mnt/e/models/manus-prime"),
    
    # WandB
    "wandb_enabled": yaml_config.get("wandb", {}).get("enabled", False),
    "wandb_project": yaml_config.get("wandb", {}).get("project", "manus-prime"),
}

# ═══════════════════════════════════════════════════════════════
# GPU DETECTION & VRAM-BASED ADJUSTMENTS
# ═══════════════════════════════════════════════════════════════

if torch.cuda.is_available():
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    logger.info(f"✅ GPU detected: {torch.cuda.get_device_name(0)} ({vram_gb:.1f}GB VRAM)")
    
    # Auto-adjust context based on available VRAM
    if vram_gb >= 80:  # H100/A100 80GB
        CONFIG["max_seq_length"] = min(CONFIG["max_seq_length"], 131072)
        CONFIG["batch_size"] = 4
        logger.info("🚀 High VRAM mode: 128K context, batch=4")
    elif vram_gb >= 24:  # A10G/RTX 4090
        CONFIG["max_seq_length"] = min(CONFIG["max_seq_length"], 65536)
        CONFIG["batch_size"] = 2
        logger.info("🚀 Medium VRAM mode: 64K context, batch=2")
    elif vram_gb >= 16:  # RTX 5080/4080
        CONFIG["max_seq_length"] = min(CONFIG["max_seq_length"], 32768)
        CONFIG["batch_size"] = 1
        logger.info("🚀 Standard mode: 32K context, batch=1")
    else:  # < 16GB
        CONFIG["max_seq_length"] = min(CONFIG["max_seq_length"], 8192)
        CONFIG["batch_size"] = 1
        logger.info("⚠️ Low VRAM mode: 8K context, batch=1")
else:
    logger.error("❌ No CUDA GPU detected. Training requires GPU.")
    sys.exit(1)

# ═══════════════════════════════════════════════════════════════
# IMPORTS (After config to handle missing libs gracefully)
# ═══════════════════════════════════════════════════════════════

try:
    from unsloth import FastLanguageModel, is_bfloat16_supported
    UNSLOTH_AVAILABLE = True
except ImportError:
    UNSLOTH_AVAILABLE = False
    logger.warning("Unsloth not available, using standard transformers")

try:
    from transformers import (
        AutoModelForCausalLM, 
        AutoTokenizer, 
        TrainingArguments,
        Trainer
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from trl import SFTTrainer
    from datasets import load_dataset, Dataset
except ImportError as e:
    logger.error(f"Missing dependency: {e}")
    logger.error("Please install: pip install transformers peft trl datasets")
    sys.exit(1)

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# ═══════════════════════════════════════════════════════════════
# DATA FORMATTING
# ═══════════════════════════════════════════════════════════════

SYSTEM_PROMPTS = {
    "fullstack": (
        "You are an expert Fullstack Architect and AI Engineer. "
        "Build production-grade applications using deep research and reasoned architectural decisions. "
        "Output your reasoning and actions as structured steps."
    ),
    "architecture": (
        "You are a System Architecture expert. "
        "Design scalable, maintainable systems with clear reasoning about trade-offs. "
        "Explain your decisions step by step."
    ),
    "qa": (
        "You are a QA Engineering expert specializing in security and testing. "
        "Identify vulnerabilities, write tests, and ensure code quality. "
        "Explain your findings with detailed reasoning."
    ),
    "uiux": (
        "You are a UI/UX Design expert. "
        "Create beautiful, accessible interfaces with Tailwind CSS and React. "
        "Explain your design decisions and accessibility considerations."
    ),
    "devops": (
        "You are a DevOps Engineer expert. "
        "Design infrastructure, CI/CD pipelines, and deployment configurations. "
        "Explain your choices for scalability and reliability."
    ),
    "default": (
        "You are an advanced AI Assistant. "
        "Solve tasks step-by-step with clear reasoning. "
        "Show your thinking process before providing solutions."
    )
}

def get_system_prompt(domain: str) -> str:
    """Get appropriate system prompt for domain."""
    for key, prompt in SYSTEM_PROMPTS.items():
        if key in domain.lower():
            return prompt
    return SYSTEM_PROMPTS["default"]

def format_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
    """Format sample for training with system prompt injection."""
    messages = sample.get("messages", [])
    if not messages:
        return {"text": ""}
    
    domain = sample.get("domain", "general")
    sys_prompt = get_system_prompt(domain)
    
    # Inject system prompt if not present
    if messages[0].get("role") != "system":
        messages = [{"role": "system", "content": sys_prompt}] + messages
    
    return {"text": messages}

# ═══════════════════════════════════════════════════════════════
# MODEL LOADING (Architecture-Agnostic)
# ═══════════════════════════════════════════════════════════════

def load_model_and_tokenizer():
    """Load model and tokenizer in architecture-agnostic way."""
    
    if UNSLOTH_AVAILABLE:
        logger.info(f"📦 Loading model with Unsloth: {CONFIG['model_name']}")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=CONFIG["model_name"],
            max_seq_length=CONFIG["max_seq_length"],
            dtype=torch.bfloat16 if is_bfloat16_supported() else torch.float16,
            load_in_4bit=True,
            trust_remote_code=CONFIG["trust_remote_code"],
        )
        
        if CONFIG["lora_enabled"]:
            logger.info(f"🔧 Adding LoRA adapters (rank={CONFIG['lora_rank']})...")
            model = FastLanguageModel.get_peft_model(
                model,
                r=CONFIG["lora_rank"],
                lora_alpha=CONFIG["lora_alpha"],
                target_modules=CONFIG["target_modules"],
                lora_dropout=CONFIG["lora_dropout"],
                bias="none",
                use_gradient_checkpointing="unsloth" if CONFIG["gradient_checkpointing"] else False,
                random_state=42,
            )
    else:
        logger.info(f"📦 Loading model with Transformers: {CONFIG['model_name']}")
        tokenizer = AutoTokenizer.from_pretrained(
            CONFIG["model_name"],
            trust_remote_code=CONFIG["trust_remote_code"]
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            CONFIG["model_name"],
            torch_dtype=torch.bfloat16 if CONFIG["bf16"] else torch.float16,
            device_map="auto",
            trust_remote_code=CONFIG["trust_remote_code"],
            load_in_4bit=True,
        )
        
        if CONFIG["lora_enabled"]:
            logger.info(f"🔧 Adding LoRA adapters (rank={CONFIG['lora_rank']})...")
            model = prepare_model_for_kbit_training(model)
            
            lora_config = LoraConfig(
                r=CONFIG["lora_rank"],
                lora_alpha=CONFIG["lora_alpha"],
                target_modules=CONFIG["target_modules"],
                lora_dropout=CONFIG["lora_dropout"],
                bias="none",
                task_type="CAUSAL_LM"
            )
            model = get_peft_model(model, lora_config)
    
    # Ensure tokenizer has padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer

# ═══════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════

def load_datasets() -> Dict[str, Dataset]:
    """Load train and validation datasets (Real Data High Priority)."""
    
    # Base path for Real Processed Data (from 04_process_real_datasets.py)
    real_data_base = Path("/mnt/e/data/processed")
    train_dir = real_data_base / "train"
    val_dir = real_data_base / "val"
    
    datasets = {}
    
    # 1. Train Data
    if train_dir.exists():
        files = list(train_dir.glob("*.jsonl"))
        if files:
            logger.info(f"📂 Loading REAL TRAIN data from {train_dir}")
            logger.info(f"   Found {len(files)} files")
            datasets["train"] = load_dataset("json", data_files=[str(f) for f in files], split="train")
            
    if "train" not in datasets:
        # Fallback to mixed if real not found (for legacy support, but warn)
        logger.warning("⚠️ No REAL training data found in processed/train. Checking legacy paths...")
        mixed_dir = Path(CONFIG["mixed_data_dir"]) / "train"
        if mixed_dir.exists():
            files = list(mixed_dir.glob("*.jsonl"))
            if files:
                datasets["train"] = load_dataset("json", data_files=[str(f) for f in files], split="train")
    
    # 2. Eval Data
    if val_dir.exists():
        files = list(val_dir.glob("*.jsonl"))
        if files:
            logger.info(f"📂 Loading REAL EVAL data from {val_dir}")
            datasets["test"] = load_dataset("json", data_files=[str(f) for f in files], split="train") # It's 'train' split of the json loader
            
    if "train" not in datasets:
        logger.error("❌ No training data found! Run 'run_pipeline.sh process' first.")
        sys.exit(1)
        
    return datasets
    
    logger.error("❌ No training data found!")
    logger.error("   Run generators first or use utils/data_mixer.py")
    sys.exit(1)

# ═══════════════════════════════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════════════════════════════

def main():
    logger.info("=" * 60)
    logger.info("🎓 MANUS PRIME: Supervised Fine-Tuning (SFT)")
    logger.info("=" * 60)
    logger.info(f"   Model: {CONFIG['model_name']}")
    logger.info(f"   LoRA Rank: {CONFIG['lora_rank']}")
    logger.info(f"   Max Seq Length: {CONFIG['max_seq_length']}")
    logger.info(f"   Learning Rate: {CONFIG['learning_rate']}")
    logger.info(f"   Epochs: {CONFIG['epochs']}")
    logger.info("=" * 60)
    
    # Enforce 'manus' conda environment
    if os.environ.get("CONDA_DEFAULT_ENV") != "manus":
        sys.exit("\033[0;31m[ERROR] Must be run in 'manus' conda environment.\033[0m")

    # Initialize WandB
    if CONFIG["wandb_enabled"] and WANDB_AVAILABLE:
        wandb.init(project=CONFIG["wandb_project"], name="sft-training")
    
    # Load model
    logger.info("\n📦 Loading model...")
    model, tokenizer = load_model_and_tokenizer()
    logger.info("✓ Model loaded successfully")
    
    # Load data
    logger.info("\n📂 Loading datasets...")
    datasets = load_datasets()
    train_dataset = datasets["train"]
    eval_dataset = datasets.get("test")
    
    logger.info(f"✓ Training samples: {len(train_dataset)}")
    if eval_dataset:
        logger.info(f"✓ Evaluation samples: {len(eval_dataset)}")
    
    # Format dataset
    logger.info("\n🔄 Formatting data...")
    train_dataset = train_dataset.map(format_sample, remove_columns=train_dataset.column_names)
    if eval_dataset:
        eval_dataset = eval_dataset.map(format_sample, remove_columns=eval_dataset.column_names)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=CONFIG["output_dir"],
        num_train_epochs=CONFIG["epochs"],
        per_device_train_batch_size=CONFIG["batch_size"],
        gradient_accumulation_steps=CONFIG["grad_accum_steps"],
        learning_rate=CONFIG["learning_rate"],
        warmup_ratio=CONFIG["warmup_ratio"],
        lr_scheduler_type="cosine",
        logging_steps=CONFIG["logging_steps"],
        save_steps=CONFIG["save_steps"],
        eval_strategy="steps" if eval_dataset else "no",
        eval_steps=CONFIG["eval_steps"] if eval_dataset else None,
        bf16=CONFIG["bf16"],
        optim="adamw_8bit",
        save_total_limit=3,
        report_to="wandb" if (CONFIG["wandb_enabled"] and WANDB_AVAILABLE) else "none",
    )
    
    # Create trainer
    logger.info("\n🚀 Starting training...")
    
    if UNSLOTH_AVAILABLE:
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            args=training_args,
            max_seq_length=CONFIG["max_seq_length"],
        )
    else:
        trainer = Trainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            args=training_args,
        )
    
    # Train
    trainer.train()
    
    # Save
    logger.info("\n💾 Saving model...")
    trainer.save_model(CONFIG["output_dir"])
    tokenizer.save_pretrained(CONFIG["output_dir"])
    
    # Save LoRA adapter separately
    if CONFIG["lora_enabled"]:
        lora_dir = Path(CONFIG["output_dir"]) / "lora_adapter"
        model.save_pretrained(str(lora_dir))
        logger.info(f"✓ LoRA adapter saved to {lora_dir}")
    
    logger.info("\n" + "=" * 60)
    logger.info("✅ Training complete!")
    logger.info(f"   Model saved to: {CONFIG['output_dir']}")
    logger.info("=" * 60)
    
    if CONFIG["wandb_enabled"] and WANDB_AVAILABLE:
        wandb.finish()

if __name__ == "__main__":
    main()
