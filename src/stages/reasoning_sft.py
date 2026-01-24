#!/usr/bin/env python3
"""
Reasoning SFT Training Stage

Supervised Fine-Tuning on Chain-of-Thought datasets with <think>...</think> formatting.
Integrated with Universal Dataset Manager for domain-based loading.
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List

import torch
from torch.utils.data import Dataset

from src.utils.repetition import PromptRepetitionEngine

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import Universal Data Management
try:
    from src.data.universal_manager import UniversalDatasetManager
except ImportError:
    logger.warning("UniversalDatasetManager not found, falling back to basic loading if needed (but strongly recommended to fix python path)")


@dataclass
class ReasoningSFTConfig:
    model_path: str = ""
    output_dir: str = "checkpoints/reasoning_sft"
    mode: str = "censored"
    
    # Data Config
    dataset_path: str = "" # Legacy single path
    dataset_categories: List[str] = field(default_factory=list) # e.g. ["reasoning", "math"]
    dataset_names: List[str] = field(default_factory=list) # Specific datasets
    
    max_seq_length: int = 4096
    num_epochs: int = 3
    batch_size: int = 2
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    use_lora: bool = True
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
    ])
    think_start_token: str = "<think>"
    think_end_token: str = "</think>"
    add_special_tokens: bool = True
    extend_context: bool = False
    target_context_length: int = 32768
    gradient_checkpointing: bool = True
    bf16: bool = True
    logging_steps: int = 10
    save_steps: int = 500
    
    # Repetition Config (arXiv:2512.14982)
    repetition_factor: int = 1
    repetition_style: str = "baseline"


class ReasoningDataset(Dataset):
    def __init__(self, config: ReasoningSFTConfig, tokenizer: Any):
        self.tokenizer = tokenizer
        self.max_length = config.max_seq_length
        self.repetition_factor = config.repetition_factor
        self.repetition_style = config.repetition_style
        self.samples = self._load_data(config)
        print(f"DEBUG: ReasoningDataset loaded {len(self.samples)} samples")
    
    def _load_data(self, config: ReasoningSFTConfig) -> List[Dict[str, Any]]:
        # Universal Loader Integration
        # If legacy dataset_path is used and it's a file, load it directly
        # Otherwise use UniversalManager
        
        explicit_path = Path(config.dataset_path) if config.dataset_path and config.dataset_path.strip() else None
        
        if explicit_path and explicit_path.exists() and explicit_path.is_file():
            logger.info(f"Loading single dataset file: {explicit_path}")
            return self._load_file(explicit_path)
            
        # Use Universal Manager
        manager = UniversalDatasetManager(mode=config.mode) # Defaults to /mnt/e/data
        
        # If dataset_path was a folder name, treat as dataset name
        names = config.dataset_names.copy()
        if config.dataset_path and not (explicit_path and explicit_path.is_file()):
             names.append(config.dataset_path)
             
        try:
            logger.info(f"Loading data via UniversalManager. Categories={config.dataset_categories}, Names={names}")
            hf_dataset = manager.get_unified_train_dataset(
                enabled_categories=config.dataset_categories, 
                included_datasets=names
            )
            logger.info(f"Loaded {len(hf_dataset)} samples from unified loader")
            return hf_dataset 
            
        except Exception as e:
            logger.error(f"Failed to load via UniversalManager: {e}")
            if explicit_path:
                 logger.warning("Universal loading failed, trying legacy path load...")
                 return self._load_file(explicit_path)
            raise ValueError(f"Could not load datasets. Error: {e}")

    def _load_file(self, path: Path) -> List[Dict[str, Any]]:
        samples = []
        if path.suffix == ".jsonl":
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        samples.append(json.loads(line))
        elif path.suffix == ".json":
            with open(path, 'r', encoding='utf-8') as f:
                samples = json.load(f)
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        
        text = ""
        if "messages" in sample:
            text = self._format_messages(sample["messages"])
        else:
            # For non-message samples, we check if we can extract prompt/response
            prompt = sample.get("prompt", sample.get("instruction", ""))
            response = sample.get("response", sample.get("output", ""))
            
            if prompt and response:
                # Apply Repetition to prompt
                if self.repetition_factor > 1:
                    prompt = PromptRepetitionEngine.apply_repetition(
                        prompt, 
                        factor=self.repetition_factor,
                        style=self.repetition_style
                    )
                text = f"User: {prompt}\n\nAssistant: {response}"
            else:
                text = sample.get("text", str(sample))
        
        encoding = self.tokenizer(text, max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt")
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": input_ids.clone()}
    
    def _format_messages(self, messages: List[Dict[str, str]]) -> str:
        # Handle Prompt Repetition in message format (repeat first user message)
        formatted = []
        for i, m in enumerate(messages):
            role = m.get('role', 'user').title()
            content = m.get('content', '')
            
            if i == 0 and m.get('role') == 'user' and self.repetition_factor > 1:
                content = PromptRepetitionEngine.apply_repetition(
                    content,
                    factor=self.repetition_factor,
                    style=self.repetition_style
                )
            
            formatted.append(f"{role}: {content}")
            
        return "\n\n".join(formatted)


class ReasoningSFTTrainer:
    def __init__(self, config: ReasoningSFTConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
    
    def setup(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_path, trust_remote_code=True)
        if self.config.add_special_tokens:
            self.tokenizer.add_special_tokens({
                "additional_special_tokens": [self.config.think_start_token, self.config.think_end_token]
            })
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        model_kwargs = {"trust_remote_code": True, "torch_dtype": torch.bfloat16, "device_map": "auto"}
        
        if self.config.extend_context:
            from src.reasoning.context_extension import create_context_extender
            extender = create_context_extender(target_length=self.config.target_context_length, scaling_type="yarn")
            model_kwargs.update(extender.get_model_kwargs())
        
        self.model = AutoModelForCausalLM.from_pretrained(self.config.model_path, **model_kwargs)
        if self.config.add_special_tokens:
            self.model.resize_token_embeddings(len(self.tokenizer))
        if self.config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        if self.config.use_lora:
            self._apply_lora()
        logger.info("Model setup complete")
    
    def _apply_lora(self):
        try:
            from peft import LoraConfig, get_peft_model, TaskType
            lora_config = LoraConfig(
                r=self.config.lora_r, lora_alpha=self.config.lora_alpha, lora_dropout=self.config.lora_dropout,
                target_modules=self.config.lora_target_modules, bias="none", task_type=TaskType.CAUSAL_LM
            )
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()
        except ImportError:
            logger.warning("PEFT not installed, training full model")
    
    def train(self):
        from transformers import Trainer, TrainingArguments
        
        dataset = ReasoningDataset(self.config, self.tokenizer)
        training_args = TrainingArguments(
            output_dir=self.config.output_dir, num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size, gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate, weight_decay=self.config.weight_decay, warmup_ratio=self.config.warmup_ratio,
            logging_steps=self.config.logging_steps, save_steps=self.config.save_steps,
            bf16=self.config.bf16, gradient_checkpointing=self.config.gradient_checkpointing,
            optim="adamw_torch", report_to=["tensorboard"], save_total_limit=3
        )
        trainer = Trainer(model=self.model, args=training_args, train_dataset=dataset, processing_class=self.tokenizer)
        trainer.train()
        trainer.save_model(self.config.output_dir)
        self.tokenizer.save_pretrained(self.config.output_dir)
        logger.info(f"Training complete. Model saved to {self.config.output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Reasoning SFT Training")
    parser.add_argument("--model", required=True, help="Base model path")
    
    # Dataset arguments (Updated)
    parser.add_argument("--dataset", default="", help="Legacy path or specific dataset name")
    parser.add_argument("--reasoning", action="store_true", help="Enable reasoning category datasets")
    parser.add_argument("--math", action="store_true", help="Enable math category datasets")
    parser.add_argument("--code", action="store_true", help="Enable code category datasets")
    parser.add_argument("--tools", action="store_true", help="Enable tool/agent category datasets")
    parser.add_argument("--general", action="store_true", help="Enable general category datasets")
    
    parser.add_argument("--output", default="checkpoints/reasoning_sft", help="Output directory")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--lora-r", type=int, default=64)
    parser.add_argument("--extend-context", action="store_true")
    parser.add_argument("--target-context", type=int, default=32768)
    parser.add_argument("--no-bf16", action="store_true", help="Disable bf16 training")
    parser.add_argument("--mode", choices=["censored", "uncensored"], default="censored")

    # Repetition args
    parser.add_argument("--repetition-factor", type=int, default=1, help="Prompt repetition factor")
    parser.add_argument("--repetition-style", type=str, default="baseline", help="Repetition style")

    # Modality Check
    parser.add_argument("--check-modality", action="store_true", help="Check model modality and exit")
    args = parser.parse_args()

    if args.check_modality:
        from src.utils.model_utils import check_modality
        if not check_modality(args.model, "text"):
            sys.exit(1)
        sys.exit(0)
    
    # Map flags to categories
    categories = []
    if args.reasoning: categories.append("reasoning")
    if args.math: categories.append("math")
    if args.code: categories.append("code")
    if args.tools: categories.append("tools")
    if args.general: categories.append("general")
    
    config = ReasoningSFTConfig(
        model_path=args.model, 
        dataset_path=args.dataset,
        dataset_categories=categories,
        output_dir=args.output,
        num_epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.lr,
        max_seq_length=args.max_length, lora_r=args.lora_r,
        extend_context=args.extend_context, target_context_length=args.target_context,
        bf16=not args.no_bf16,
        mode=args.mode,
        repetition_factor=args.repetition_factor,
        repetition_style=args.repetition_style
    )
    trainer = ReasoningSFTTrainer(config)
    trainer.setup()
    trainer.train()


if __name__ == "__main__":
    main()
