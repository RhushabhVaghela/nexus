#!/usr/bin/env python3
"""
stage_tools.py
Tool/function calling training stage.

Trains model on function calling format with tool definitions.
"""

import torch
import json
from typing import Dict, Any, List
from datasets import load_dataset

from .base import BaseStage, StageConfig


class ToolDefinition:
    """Represents a tool/function definition."""
    
    def __init__(self, name: str, description: str, parameters: Dict):
        self.name = name
        self.description = description
        self.parameters = parameters
    
    def to_json(self) -> str:
        return json.dumps({
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }, indent=2)


class ToolsStage(BaseStage):
    """Tool calling capability training with complete implementation."""
    
    CAPABILITY_NAME = "tool-calling"
    
    DATASET_PATTERNS = [
        "argilla/Synth-APIGen-v0.1",
        "Salesforce/xlam-function-calling-60k",
        "*tool*",
        "*function*call*",
    ]
    
    # Tool calling format tokens
    TOOL_TOKENS = {
        "tools_start": "<tools>",
        "tools_end": "</tools>",
        "call_start": "<tool_call>",
        "call_end": "</tool_call>",
        "result_start": "<tool_result>",
        "result_end": "</tool_result>",
    }
    
    def prepare(self) -> bool:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        self.logger.info(f"Loading model from {self.config.base_model_path}")
        
        if self.config.dry_run:
            self.logger.info("[DRY-RUN] Would load tool calling datasets")
            return True
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.base_model_path,
                trust_remote_code=True,
            )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.base_model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Add tool tokens
            special_tokens = {
                "additional_special_tokens": list(self.TOOL_TOKENS.values())
            }
            self.tokenizer.add_special_tokens(special_tokens)
            self.model.resize_token_embeddings(len(self.tokenizer))
            
            self.logger.info("Loading tool calling datasets...")
            
            try:
                ds = load_dataset(
                    "Salesforce/xlam-function-calling-60k",
                    split="train",
                    trust_remote_code=True,
                )
                if self.config.sample_size > 0:
                    ds = ds.select(range(min(self.config.sample_size, len(ds))))
                self.train_dataset = ds
                self.logger.info(f"Loaded: {len(ds)} samples")
            except Exception as e:
                self.logger.warning(f"Could not load dataset: {e}")
                self.train_dataset = None
            
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed: {e}")
            return False
    
    def _format_tool_call(self, sample: Dict) -> str:
        """Format sample in tool calling format."""
        # Handle different dataset formats
        if "tools" in sample and "query" in sample:
            tools = sample.get("tools", [])
            query = sample.get("query", "")
            response = sample.get("response", "")
            
            tools_str = "<tools>\n"
            if isinstance(tools, list):
                for tool in tools:
                    if isinstance(tool, dict):
                        tools_str += json.dumps(tool) + "\n"
            tools_str += "</tools>\n\n"
            
            text = f"{tools_str}User: {query}\n\n<tool_call>\n{response}\n</tool_call>"
            return text
        elif "text" in sample:
            return sample["text"]
        elif "messages" in sample:
            return str(sample["messages"])
        return ""
    
    def train(self) -> Dict[str, Any]:
        if self.config.dry_run:
            self.logger.info("[DRY-RUN] Simulating tool calling training...")
            for epoch in range(self.config.epochs):
                self.logger.info(f"[DRY-RUN] Epoch {epoch+1}/{self.config.epochs}")
                for step in range(10):
                    self.current_step += 1
            return {"success": True, "dry_run": True, "steps": self.current_step}
        
        if self.train_dataset is None:
            self.logger.warning("No training data, skipping")
            return {"success": True, "steps": 0, "skipped": True}
        
        self.logger.info("Starting tool calling training...")
        self.logger.info(f"Using tokens: {list(self.TOOL_TOKENS.values())}")
        
        from src.training_controller import training_step_hook
        
        total_loss = 0.0
        
        for epoch in range(self.config.epochs):
            self.logger.info(f"Epoch {epoch + 1}/{self.config.epochs}")
            
            for sample in self.train_dataset:
                self.current_step += 1
                
                training_step_hook(
                    self.model, self.optimizer, self.current_step,
                    str(self.checkpoint_dir)
                )
                
                text = self._format_tool_call(sample)
                if not text:
                    continue
                
                inputs = self.tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=2048,
                    padding=True,
                )
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
                inputs["labels"] = inputs["input_ids"].clone()
                
                outputs = self.model(**inputs)
                loss = outputs.loss
                total_loss += loss.item()
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                if self.current_step % 100 == 0:
                    avg = total_loss / self.current_step
                    self.logger.info(f"Step {self.current_step}, Avg Loss: {avg:.4f}")
        
        return {
            "success": True,
            "steps": self.current_step,
            "final_loss": total_loss / max(self.current_step, 1),
        }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Tool Calling Training")
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--sample-size", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()
    
    config = StageConfig(
        capability_name="tool-calling",
        base_model_path=args.base_model,
        output_dir=args.output_dir,
        sample_size=args.sample_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        dry_run=args.dry_run,
    )
    stage = ToolsStage(config)
    return 0 if stage.run().get("success") else 1

if __name__ == "__main__":
    exit(main())
