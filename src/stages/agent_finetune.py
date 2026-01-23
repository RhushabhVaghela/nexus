#!/usr/bin/env python3
"""
Agent Fine-tuning Stage

Trains models for agentic capabilities using Universal Dataset Manager.
"""

import logging
import argparse
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from pathlib import Path
import sys

import torch
from torch import Tensor

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from src.data.universal_manager import UniversalDatasetManager
except ImportError:
    pass

@dataclass
class AgentFinetuneConfig:
    # Model
    model_path: str = ""
    output_dir: str = "./output/agent_finetuned"
    mode: str = "censored"
    
    # Data
    dataset_categories: List[str] = field(default_factory=list)
    dataset_names: List[str] = field(default_factory=list)
    max_seq_length: int = 2048
    
    # Training
    learning_rate: float = 1e-5
    batch_size: int = 4
    gradient_accumulation_steps: int = 8
    num_epochs: int = 3
    warmup_ratio: float = 0.1
    
    # Agent-specific
    enable_tool_use: bool = True
    enable_planning: bool = True
    enable_code_execution: bool = True
    enable_web_browsing: bool = False
    
    # Tool format
    tool_format: str = "json"  # "json", "xml", "function_call"
    max_tool_calls: int = 10
    
    # Planning
    max_plan_steps: int = 20
    plan_format: str = "structured"  # "structured", "freeform"
    
    # LoRA settings
    use_lora: bool = True
    lora_r: int = 32
    lora_alpha: int = 64
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
    ])


class AgentDataset:
    def __init__(self, config: AgentFinetuneConfig, tokenizer: Any):
        self.config = config
        self.tokenizer = tokenizer
        self.max_length = config.max_seq_length
        self.samples = self._load_data()
    
    def _load_data(self):
        manager = UniversalDatasetManager(mode=self.config.mode)
        
        # Default agent categories if none specified
        categories = self.config.dataset_categories
        if not categories:
            categories = ["tools", "code"] 
            
        logger.info(f"Loading Agent data via UniversalManager. Categories={categories}")
        try:
            hf_dataset = manager.get_unified_train_dataset(
                enabled_categories=categories,
                included_datasets=self.config.dataset_names
            )
            samples = []
            for item in hf_dataset:
                samples.append(item)
            return samples
        except Exception as e:
            logger.error(f"Failed load: {e}")
            return []
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        if "messages" in sample:
            text = self._format_messages(sample["messages"])
        else:
            text = sample.get("text", str(sample))
        
        encoding = self.tokenizer(text, max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt")
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": input_ids.clone()}
    
    def _format_messages(self, messages: List[Dict[str, str]]) -> str:
        return "\n\n".join([f"{m.get('role', 'user').title()}: {m.get('content', '')}" for m in messages])


# ... (AgentFinetuner classes remain similar, just need to update usage example mostly)
# For brevity, I'm replacing the whole file to ensure clean config integration
# but reusing the Trainer/Finetuner implementations as they were mostly method-based.

class AgentFinetuner:
    def __init__(self, config: AgentFinetuneConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.tools: List[Dict[str, Any]] = []
    
    def setup(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        logger.info(f"Loading model from {self.config.model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        special_tokens = ["<think>", "</think>", "<tool>", "</tool>", "<plan>", "</plan>"]
        self.tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_path,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
        self.model.resize_token_embeddings(len(self.tokenizer))
        
        if self.config.use_lora:
            self._apply_lora()
            
    def _apply_lora(self):
        try:
            from peft import LoraConfig, get_peft_model
            lora_config = LoraConfig(
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                target_modules=self.config.lora_target_modules,
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
            )
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()
        except ImportError:
            logger.warning("peft not available, skipping LoRA")
            
    def register_tools(self, tools: List[Dict[str, Any]]):
        self.tools = tools
        
    def train(self, dataset: AgentDataset):
        from transformers import TrainingArguments, Trainer
        
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            warmup_ratio=self.config.warmup_ratio,
            fp16=True,
            logging_steps=10,
            save_steps=500,
            save_total_limit=3,
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            processing_class=self.tokenizer,
        )
        trainer.train()
        trainer.save_model(self.config.output_dir)
        self.tokenizer.save_pretrained(self.config.output_dir)
        logger.info(f"Model saved to {self.config.output_dir}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--tools", action="store_true", help="Enable tool datasets")
    parser.add_argument("--code", action="store_true", help="Enable code datasets")
    parser.add_argument("--web", action="store_true", help="Enable web agent datasets")
    parser.add_argument("--output", default="./output/agent_finetuned")
    parser.add_argument("--num-epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--mode", choices=["censored", "uncensored"], default="censored")
    parser.add_argument("--check-modality", action="store_true", help="Check model modality and exit")
    
    args = parser.parse_args()
    
    if args.check_modality:
        from src.utils.model_utils import check_modality
        if not check_modality(args.model, "text"): # Agents usually text based unless specialized
             sys.exit(1)
        sys.exit(0)
    
    categories = []
    if args.tools: categories.append("tools")
    if args.code: categories.append("code")
    if args.web: categories.append("tools") # Web usually falls into tools or specific web datasets key
    
    config = AgentFinetuneConfig(
        model_path=args.model,
        output_dir=args.output,
        dataset_categories=categories,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        max_seq_length=args.max_length,
        mode=args.mode
    )
    
    finetuner = AgentFinetuner(config)
    finetuner.setup()
    dataset = AgentDataset(config, finetuner.tokenizer)
    finetuner.train(dataset)

if __name__ == "__main__":
    main()
