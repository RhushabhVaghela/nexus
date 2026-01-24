#!/usr/bin/env python3
"""
stage_thinking.py
Extended thinking/reflection training stage.

Trains model on extended reasoning with internal monologue.
"""

import torch
from typing import Dict, Any
from datasets import load_dataset

from .base import BaseStage, StageConfig
from src.utils.repetition import PromptRepetitionEngine


class ThinkingStage(BaseStage):
    \"\"\"Extended thinking/reflection training with complete implementation.\"\"\"
    
    CAPABILITY_NAME = \"thinking\"
    
    DATASET_PATTERNS = [
        \"simplescaling/s1K-1.1\",
        \"open-thoughts/OpenThoughts-114k\",
    ]
    
    # Thinking markers for training
    THINKING_TOKENS = {
        \"start\": \"<think>\",
        \"end\": \"</think>\",
        \"reconsider\": \"<reconsider>\",
        \"alternative\": \"<alternative>\",
    }
    
    def prepare(self) -> bool:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        self.logger.info(f\"Loading model from {self.config.base_model_path}\")
        
        if self.config.dry_run:
            self.logger.info(\"[DRY-RUN] Would load thinking datasets\")
            return True
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.base_model_path,
                trust_remote_code=True,
            )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.base_model_path,
                torch_dtype=torch.float16,
                device_map=\"auto\",
                trust_remote_code=True,
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Add thinking tokens
            special_tokens = {
                \"additional_special_tokens\": list(self.THINKING_TOKENS.values())
            }
            self.tokenizer.add_special_tokens(special_tokens)
            self.model.resize_token_embeddings(len(self.tokenizer))
            
            self.logger.info(\"Loading thinking datasets dynamic...\")
            self.train_dataset = self.load_dynamic_datasets()
            
            if self.train_dataset:
                self.logger.info(f\"Loaded: {len(self.train_dataset)} samples\")
            else:
                self.logger.warning(\"No datasets loaded\")
            
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f\"Failed: {e}\")
            return False
    
    def _format_thinking(self, sample: Dict) -> str:
        \"\"\"Format sample with thinking markers and optional repetition.\"\"\"
        text = \"\"
        if \"text\" in sample:
            text = sample[\"text\"]
            # Wrap in thinking tokens if not already
            if not text.startswith(\"<think>\"):
                text = f\"<think>\n{text}\n</think>\"
                
        elif \"input\" in sample and \"output\" in sample:
            question = sample['input']
            if self.config.repetition_factor > 1:
                question = PromptRepetitionEngine.apply_repetition(
                    question, 
                    factor=self.config.repetition_factor,
                    style=self.config.repetition_style
                )
            text = f\"Question: {question}\n<think>\n{sample['output']}\n</think>\"
            
        return text
    
    def train(self) -> Dict[str, Any]:
        if self.config.dry_run:
            self.logger.info(\"[DRY-RUN] Simulating thinking training...\")
            for epoch in range(self.config.epochs):
                self.logger.info(f\"[DRY-RUN] Epoch {epoch+1}/{self.config.epochs}\")
                for step in range(10):
                    self.current_step += 1
            return {\"success\": True, \"dry_run\": True, \"steps\": self.current_step}
        
        if self.train_dataset is None:
            self.logger.warning(\"No training data, skipping\")
            return {\"success\": True, \"steps\": 0, \"skipped\": True}
        
        self.logger.info(\"Starting extended thinking training...\")
        self.logger.info(f\"Using tokens: {list(self.THINKING_TOKENS.values())}\")
        if self.config.repetition_factor > 1:
            self.logger.info(f\"Using Prompt Repetition: {self.config.repetition_factor}x ({self.config.repetition_style})\")
        
        from src.training_controller import training_step_hook
        
        total_loss = 0.0
        
        for epoch in range(self.config.epochs):
            self.logger.info(f\"Epoch {epoch + 1}/{self.config.epochs}\")
            
            for sample in self.train_dataset:
                self.current_step += 1
                
                training_step_hook(
                    self.model, self.optimizer, self.current_step,
                    str(self.checkpoint_dir)
                )
                
                text = self._format_thinking(sample)
                if not text:
                    continue
                
                inputs = self.tokenizer(
                    text,
                    return_tensors=\"pt\",
                    truncation=True,
                    max_length=4096,  # Extended for thinking
                    padding=True,
                )
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
                inputs[\"labels\"] = inputs[\"input_ids\"].clone()
                
                outputs = self.model(**inputs)
                loss = outputs.loss
                total_loss += loss.item()
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                if self.current_step % 100 == 0:
                    avg = total_loss / self.current_step
                    self.logger.info(f\"Step {self.current_step}, Avg Loss: {avg:.4f}\")
        
        return {
            \"success\": True,
            \"steps\": self.current_step,
            \"final_loss\": total_loss / max(self.current_step, 1),
        }


def main():
    import argparse
    parser = argparse.ArgumentParser(description=\"Extended Thinking Training\")
    parser.add_argument(\"--base-model\", required=True)
    parser.add_argument(\"--output-dir\", required=True)
    parser.add_argument(\"--dry-run\", action=\"store_true\")
    parser.add_argument(\"--sample-size\", type=int, default=0)
    parser.add_argument(\"--batch-size\", type=int, default=1)
    parser.add_argument(\"--epochs\", type=int, default=3)
    # Repetition args
    parser.add_argument(\"--repetition-factor\", type=int, default=1, help=\"Prompt repetition factor\")
    parser.add_argument(\"--repetition-style\", type=str, default=\"baseline\", help=\"Repetition style\")
    
    args = parser.parse_args()
    
    config = StageConfig(
        capability_name=\"thinking\",
        base_model_path=args.base_model,
        output_dir=args.output_dir,
        sample_size=args.sample_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        dry_run=args.dry_run,
        repetition_factor=args.repetition_factor,
        repetition_style=args.repetition_style,
    )
    stage = ThinkingStage(config)
    return 0 if stage.run().get(\"success\") else 1

if __name__ == \"__main__\":
    exit(main())
