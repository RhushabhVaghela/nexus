import torch
import os
import logging
from pathlib import Path
from typing import Dict, Any, List
from datasets import load_dataset
from transformers import TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from .base import TextCapabilityStage, StageConfig
from src.utils.repetition import PromptRepetitionEngine

class RemotionGenStage(TextCapabilityStage):
    """Remotion explanatory video generation training with LoRA support."""
    
    CAPABILITY_NAME = "remotion-explainer"
    
    DATASET_PATTERNS = [
        "*remotion*",
        "*explainer*",
    ]
    
    def prepare(self) -> bool:
        """Prepare Remotion training."""
        if not super().prepare():
            return False
            
        if self.config.dry_run:
            self.logger.info("[DRY-RUN] Would load Remotion datasets")
            return True
            
        # Load datasets
        self.logger.info("Loading Remotion explainer datasets...")
        self.train_dataset = self.load_dynamic_datasets()
        
        if self.train_dataset:
            self.logger.info(f"Total training samples: {len(self.train_dataset)}")
        else:
            self.logger.warning("No datasets loaded for Remotion explainer")
            
        # Add LoRA adapter if not already present
        if self.model and not hasattr(self.model, "peft_config"):
            self.logger.info("Adding LoRA adapter to model...")
            self.model = prepare_model_for_kbit_training(self.model)
            lora_config = LoraConfig(
                r=64,
                lora_alpha=128,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM"
            )
            self.model = get_peft_model(self.model, lora_config)
            
        return True
    
    def train(self) -> Dict[str, Any]:
        """Run Remotion training loop using SFTTrainer-like logic."""
        if self.config.dry_run:
            return super().train()
        
        if self.train_dataset is None or len(self.train_dataset) == 0:
            self.logger.warning("No training data, skipping")
            return {"success": True, "steps": 0, "skipped": True}
            
        self.logger.info("Starting Remotion explainer training (LoRA SFT)...")
        if self.config.repetition_factor > 1:
            self.logger.info(f"Using Prompt Repetition: {self.config.repetition_factor}x ({self.config.repetition_style})")
        
        from src.capability_registry import REMOTION_EXPLAINER_SYSTEM_PROMPT

        def tokenize_function(sample):
            # Format with 3B1B System Prompt
            instruction = sample['instruction']
            
            # Apply Prompt Repetition to the user instruction
            if self.config.repetition_factor > 1:
                instruction = PromptRepetitionEngine.apply_repetition(
                    instruction,
                    factor=self.config.repetition_factor,
                    style=self.config.repetition_style
                )
            
            prompt = f"<|im_start|>system\n{REMOTION_EXPLAINER_SYSTEM_PROMPT}<|im_end|>\n"
            prompt += f"<|im_start|>user\n{instruction}<|im_end|>\n"
            prompt += f"<|im_start|>assistant\n"
            
            full_text = prompt + sample['output'] + self.tokenizer.eos_token
            
            tokenized = self.tokenizer(
                full_text,
                truncation=True,
                max_length=4096,
                padding=False
            )
            
            # Mask the prompt from loss
            prompt_ids = self.tokenizer(prompt, truncation=True, max_length=4096)["input_ids"]
            labels = [-100] * len(prompt_ids) + tokenized["input_ids"][len(prompt_ids):]
            tokenized["labels"] = labels
            
            return tokenized
            
        tokenized_dataset = self.train_dataset.map(
            tokenize_function, 
            batched=False, 
            remove_columns=self.train_dataset.column_names
        )
        
        from transformers import DataCollatorForSeq2Seq
        data_collator = DataCollatorForSeq2Seq(
            self.tokenizer, 
            model=self.model, 
            padding=True
        )

        training_args = TrainingArguments(
            output_dir=str(self.checkpoint_dir),
            per_device_train_batch_size=self.config.batch_size,
            gradient_accumulation_steps=4,
            learning_rate=2e-4,
            num_train_epochs=self.config.epochs,
            bf16=True,
            logging_steps=5,
            save_steps=50,
            save_total_limit=2,
            optim="adamw_8bit",
            remove_unused_columns=False,
            report_to="none"
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=data_collator,
        )
        
        trainer.train()
        
        # Save the LoRA adapter
        adapter_path = Path(self.config.output_dir) / "remotion_explainer_adapter"
        self.model.save_pretrained(str(adapter_path))
        self.tokenizer.save_pretrained(str(adapter_path))
        
        return {
            "success": True,
            "adapter_path": str(adapter_path),
            "steps": trainer.state.global_step
        }

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Remotion Gen Training Stage")
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--sample-size", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    # Repetition args
    parser.add_argument("--repetition-factor", type=int, default=1, help="Prompt repetition factor")
    parser.add_argument("--repetition-style", type=str, default="baseline", help="Repetition style")
    
    args = parser.parse_args()
    
    config = StageConfig(
        capability_name="remotion-explainer",
        base_model_path=args.base_model,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        sample_size=args.sample_size,
        dry_run=args.dry_run,
        repetition_factor=args.repetition_factor,
        repetition_style=args.repetition_style,
    )
    
    stage = RemotionGenStage(config)
    results = stage.run()
    return 0 if results.get("success") else 1

if __name__ == "__main__":
    exit(main())
