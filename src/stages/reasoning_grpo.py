#!/usr/bin/env python3
"""
Reasoning GRPO Training Stage

Group Relative Policy Optimization for emergent reasoning capabilities.
Integrated with Universal Dataset Manager.
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import Universal Data Management
try:
    from src.data.universal_manager import UniversalDatasetManager
except ImportError:
    pass

@dataclass
class ReasoningGRPOConfig:
    model_path: str = ""
    output_dir: str = "checkpoints/reasoning_grpo"
    mode: str = "censored"
    
    dataset_path: str = ""
    dataset_categories: List[str] = field(default_factory=list)
    dataset_names: List[str] = field(default_factory=list)
    
    max_seq_length: int = 4096
    group_size: int = 4
    num_iterations: int = 1000
    samples_per_iteration: int = 64
    batch_size: int = 4
    learning_rate: float = 1e-6
    kl_coef: float = 0.1
    clip_ratio: float = 0.2
    correctness_weight: float = 0.4
    format_weight: float = 0.2
    max_new_tokens: int = 2048
    temperature: float = 1.0
    top_p: float = 0.95
    gradient_checkpointing: bool = True
    bf16: bool = True
    logging_steps: int = 10
    save_steps: int = 100


class GRPODataset(Dataset):
    def __init__(self, config: ReasoningGRPOConfig, tokenizer: Any):
        self.tokenizer = tokenizer
        self.max_length = config.max_seq_length
        self.problems = self._load_problems(config)
    
    def _load_problems(self, config: ReasoningGRPOConfig) -> List[Dict[str, Any]]:
        # Universal Loader Integration
        # Prioritize unified loader
        
        manager = UniversalDatasetManager(mode=config.mode)
        explicit_path = Path(config.dataset_path) if config.dataset_path and config.dataset_path.strip() else None
        
        # Load from Universal Manager if categories provided or not a file path
        if config.dataset_categories or (config.dataset_path and not (explicit_path and explicit_path.is_file())):
            names = config.dataset_names.copy()
            if config.dataset_path and not (explicit_path and explicit_path.is_file()):
                names.append(config.dataset_path)
                
            try:
                logger.info(f"Loading GRPO data via UniversalManager. Categories={config.dataset_categories}")
                hf_dataset = manager.get_unified_train_dataset(
                    enabled_categories=config.dataset_categories, 
                    included_datasets=names
                )
                
                # Convert HF dataset to problems list
                # GRPO expects "prompt"/"question" and "reference"/"answer"
                problems = []
                for item in hf_dataset:
                    problems.append(item)
                logger.info(f"Loaded {len(problems)} problems via UniversalManager")
                return problems
            except Exception as e:
                logger.error(f"Failed universal load: {e}")
                
        # Fallback to single file
        if explicit_path and explicit_path.exists():
            return self._load_file(explicit_path)
            
        return []

    def _load_file(self, path: Path) -> List[Dict[str, Any]]:
        problems = []
        if path.suffix == ".jsonl":
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        problems.append(json.loads(line))
        elif path.suffix == ".json":
            with open(path, 'r', encoding='utf-8') as f:
                 data = json.load(f)
                 if isinstance(data, list): problems = data
                 else: problems = [data]
        return problems
    
    def __len__(self) -> int:
        return len(self.problems)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        problem = self.problems[idx]
        # Robust prompt extraction
        prompt = problem.get("question", problem.get("problem", problem.get("input", problem.get("prompt", ""))))
        # If no explicit prompt, maybe it's in messages format?
        if not prompt and "messages" in problem:
            # Extract last user message or construct prompt
             prompt = "\n".join([m['content'] for m in problem['messages'] if m['role'] == 'user'])
             
        reference = str(problem.get("answer", problem.get("solution", problem.get("reference", ""))))
        return {"prompt": prompt, "reference": reference}


class GRPOTrainer:
    def __init__(self, config: ReasoningGRPOConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.ref_model = None
        self.reward_fn = None
        self.optimizer = None
    
    def setup(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from src.reasoning.reward_functions import create_reward_function, RewardConfig
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        model_kwargs = {"trust_remote_code": True, "torch_dtype": torch.bfloat16, "device_map": "auto"}
        self.model = AutoModelForCausalLM.from_pretrained(self.config.model_path, **model_kwargs)
        self.ref_model = AutoModelForCausalLM.from_pretrained(self.config.model_path, **model_kwargs)
        self.ref_model.eval()
        for param in self.ref_model.parameters():
            param.requires_grad = False
        
        if self.config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        
        reward_config = RewardConfig(correctness_weight=self.config.correctness_weight, format_weight=self.config.format_weight)
        self.reward_fn = create_reward_function("combined", reward_config)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.learning_rate)
        logger.info("GRPO trainer setup complete")
    
    def generate_responses(self, prompts: List[str], num_responses: int = 4) -> List[List[str]]:
        all_responses = []
        self.model.eval()
        with torch.no_grad():
            for prompt in prompts:
                if not prompt: continue 
                responses = []
                inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=self.config.max_seq_length // 2).to(self.model.device)
                for _ in range(num_responses):
                    outputs = self.model.generate(**inputs, max_new_tokens=self.config.max_new_tokens, temperature=self.config.temperature, top_p=self.config.top_p, do_sample=True, pad_token_id=self.tokenizer.pad_token_id)
                    response = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
                    responses.append(response)
                all_responses.append(responses)
        self.model.train()
        return all_responses
    
    def compute_rewards(self, prompts: List[str], responses: List[List[str]], references: List[str]) -> torch.Tensor:
        rewards = []
        for prompt, resps, ref in zip(prompts, responses, references):
            resp_rewards = [self.reward_fn.compute(resp, ref, prompt).reward for resp in resps]
            rewards.append(resp_rewards)
        return torch.tensor(rewards)
    
    def compute_advantages(self, rewards: torch.Tensor) -> torch.Tensor:
        group_mean = rewards.mean(dim=1, keepdim=True)
        group_std = rewards.std(dim=1, keepdim=True) + 1e-8
        return (rewards - group_mean) / group_std
    
    def train_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        prompts, references = batch["prompts"], batch["references"]
        responses = self.generate_responses(prompts, self.config.group_size)
        if not responses: return {"loss": 0.0, "mean_reward": 0.0, "max_reward": 0.0}
        
        rewards = self.compute_rewards(prompts, responses, references)
        advantages = self.compute_advantages(rewards)
        
        self.optimizer.zero_grad()
        total_loss = 0.0
        for i, (prompt, resps) in enumerate(zip(prompts, responses)):
            for j, resp in enumerate(resps):
                full_text = prompt + resp
                inputs = self.tokenizer(full_text, return_tensors="pt", truncation=True, max_length=self.config.max_seq_length).to(self.model.device)
                outputs = self.model(**inputs)
                logits = outputs.logits
                prompt_len = len(self.tokenizer.encode(prompt))
                
                # Safety check for prompt/response split
                if prompt_len >= inputs.input_ids.shape[1]: continue
                
                log_probs = F.log_softmax(logits[:, prompt_len-1:-1], dim=-1)
                token_log_probs = torch.gather(log_probs, 2, inputs.input_ids[:, prompt_len:].unsqueeze(-1)).squeeze(-1)
                advantage = advantages[i, j].to(self.model.device)
                total_loss += -advantage * token_log_probs.mean()
        
        loss = total_loss / max(len(prompts) * self.config.group_size, 1)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        return {"loss": loss.item(), "mean_reward": rewards.mean().item(), "max_reward": rewards.max().item()}
    
    def train(self):
        from tqdm import tqdm
        
        dataset = GRPODataset(self.config, self.tokenizer)
        if len(dataset) == 0:
            logger.error("No data found for GRPO training!")
            return

        dataloader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True, collate_fn=lambda b: {"prompts": [x["prompt"] for x in b], "references": [x["reference"] for x in b]})
        
        global_step = 0
        best_reward = -float('inf')
        
        for iteration in tqdm(range(self.config.num_iterations), desc="GRPO"):
            for batch in dataloader:
                metrics = self.train_step(batch)
                global_step += 1
                
                if global_step % self.config.logging_steps == 0:
                    logger.info(f"Step {global_step}: loss={metrics['loss']:.4f}, reward={metrics['mean_reward']:.4f}")
                
                if global_step % self.config.save_steps == 0:
                    self._save_checkpoint(global_step, metrics['mean_reward'])
                
                if metrics['mean_reward'] > best_reward:
                    best_reward = metrics['mean_reward']
                    self._save_checkpoint("best", best_reward)
                
                if global_step >= self.config.num_iterations:
                    break
            if global_step >= self.config.num_iterations:
                break
        
        self._save_checkpoint("final", best_reward)
        logger.info(f"GRPO complete. Best reward: {best_reward:.4f}")
    
    def _save_checkpoint(self, name: str, reward: float):
        save_path = Path(self.config.output_dir) / f"checkpoint-{name}"
        save_path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        with open(save_path / "metadata.json", 'w') as f:
            json.dump({"step": str(name), "reward": reward}, f)
        logger.info(f"Saved checkpoint to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Reasoning GRPO Training")
    parser.add_argument("--model", required=True)
    
    # Dataset Args
    parser.add_argument("--dataset", default="")
    parser.add_argument("--reasoning", action="store_true")
    parser.add_argument("--math", action="store_true")
    parser.add_argument("--code", action="store_true")
    
    parser.add_argument("--output", default="checkpoints/reasoning_grpo")
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--kl-coef", type=float, default=0.1)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--mode", choices=["censored", "uncensored"], default="censored")
    parser.add_argument("--check-modality", action="store_true", help="Check model modality and exit")
    args = parser.parse_args()
    
    if args.check_modality:
        from src.utils.model_utils import check_modality
        if not check_modality(args.model, "text"):
            sys.exit(1)
        sys.exit(0)
    
    categories = []
    if args.reasoning: categories.append("reasoning")
    if args.math: categories.append("math")
    if args.code: categories.append("code")
    
    config = ReasoningGRPOConfig(
        model_path=args.model, 
        dataset_path=args.dataset,
        dataset_categories=categories,
        output_dir=args.output,
        num_iterations=args.iterations, batch_size=args.batch_size, group_size=args.group_size,
        learning_rate=args.lr, kl_coef=args.kl_coef,
        mode=args.mode
    )
    trainer = GRPOTrainer(config)
    trainer.setup()
    trainer.train()


if __name__ == "__main__":
    main()
