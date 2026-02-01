#!/usr/bin/env python3
"""
ppo_training.py
Proximal Policy Optimization (PPO) Training for RLHF

PPO is the classic RLHF approach using a reward model to optimize
the policy through reinforcement learning.

Usage:
    python src/ppo_training.py --mode=censored

References:
    - "Training language models to follow instructions with human feedback" (InstructGPT)
"""

import os
import sys
import torch
import logging
from pathlib import Path

os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/ppo_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.training_methods import TrainingMethod, get_training_config

def check_env():
    """Verify environment dependencies."""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification
        from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead
        from peft import LoraConfig, get_peft_model
        from datasets import load_dataset, Dataset
    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        return False
        
    if not torch.cuda.is_available():
        logger.warning("⚠️ No CUDA GPU detected. PPO requires GPU.")
        return False
    return True

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification
    from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead
    from peft import LoraConfig, get_peft_model
    from datasets import load_dataset, Dataset
except ImportError:
    pass

CONFIG = {
    "model_checkpoint": "/mnt/e/data/models/Qwen2.5-0.5B",
    "reward_model": None,  # Use rule-based reward if None
    "prompts_data": "/mnt/e/data/processed/train",
    "max_seq_length": 512,
    "output_dir": "/mnt/e/data/output/ppo",
}

ppo_config = get_training_config(TrainingMethod.PPO)


def load_prompts():
    """Load prompts for PPO training."""
    prompts_path = Path(CONFIG["prompts_data"])
    
    if prompts_path.exists():
        files = list(prompts_path.glob("*.jsonl"))
        if files:
            dataset = load_dataset("json", data_files=[str(f) for f in files], split="train")
            
            # Extract prompts from messages format
            prompts = []
            for sample in dataset:
                messages = sample.get("messages", [])
                for msg in messages:
                    if msg.get("role") == "user":
                        prompts.append({"query": msg.get("content", "")})
                        break
            
            if prompts:
                return Dataset.from_list(prompts[:1000])  # Limit for PPO

    # Check for streaming request or large file
    # If path is a directory or large file, try streaming
    if prompts_path.exists():
        is_large = prompts_path.is_file() and prompts_path.stat().st_size > 1 * 1024**3
        is_dir = prompts_path.is_dir()
        
        if is_large or is_dir:
            logger.info("🌊 Using StreamingDatasetLoader for prompts")
            try:
                from src.data.streaming_trainer import StreamingDatasetLoader, StreamingConfig
                loader = StreamingDatasetLoader([str(prompts_path)], StreamingConfig(buffer_size=10000))
                # For PPO we need text, so we map the streaming dataset
                dataset = loader.get_streaming_dataset()
                # PPO implementation in trl expects a Dataset, not IterableDataset easily for some versions
                # But we can try to take a subset if strict PPO is needed, or allow streaming if trl supports it.
                # Standard PPO often needs random access for epochs. 
                # For now, we'll take a large sample if streaming is forced, to avoid OOM but allow training.
                logger.info("Streaming enabled: Taking 10,000 samples for PPO epoch efficiency")
                return Dataset.from_list(list(dataset.take(10000)))
            except ImportError:
                pass
    
    # Fallback
    logger.warning("Using dummy prompts for testing")
    return Dataset.from_list([
        {"query": "What is machine learning?"},
        {"query": "Explain neural networks."},
        {"query": "How does Python work?"},
    ] * 100)


def rule_based_reward(responses):
    """Rule-based reward function when no reward model is available."""
    rewards = []
    for response in responses:
        score = 0.0
        
        # Length reward (not too short, not too long)
        words = len(response.split())
        if 20 < words < 200:
            score += 0.3
        
        # Structure reward
        if any(kw in response.lower() for kw in ["because", "therefore", "first", "second"]):
            score += 0.3
        
        # Helpfulness indicators
        if any(kw in response.lower() for kw in ["example", "for instance", "such as"]):
            score += 0.2
        
        # Penalty for refusals
        if any(kw in response.lower() for kw in ["i cannot", "i can't", "i'm sorry"]):
            score -= 0.3
        
        rewards.append(max(0.0, min(1.0, score)))
    
    return rewards


def main():
    if not check_env():
         logger.error("❌ Environment check failed (missing dependencies or GPU).")
         sys.exit(1)
         
    logger.info("=" * 60)
    logger.info("🎓 PPO TRAINING - Proximal Policy Optimization (RLHF)")
    logger.info("=" * 60)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        CONFIG["model_checkpoint"],
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with value head for PPO
    logger.info("Loading model with value head...")
    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        CONFIG["model_checkpoint"],
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    
    # Reference model (frozen copy)
    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(
        CONFIG["model_checkpoint"],
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    
    # Load prompts
    dataset = load_prompts()
    logger.info(f"Loaded {len(dataset)} prompts")
    
    # PPO config
    training_args = PPOConfig(
        output_dir=CONFIG["output_dir"],
        learning_rate=ppo_config.learning_rate,
        batch_size=4,
        mini_batch_size=1,
        gradient_accumulation_steps=4,
        log_with="tensorboard",
        ppo_epochs=4,
        kl_penalty="kl",
        target_kl=0.1,
    )
    
    # Create PPO trainer
    trainer = PPOTrainer(
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        config=training_args,
        dataset=dataset,
    )
    
    logger.info("Starting PPO training loop...")
    
    # Training loop
    for epoch in range(ppo_config.epochs):
        logger.info(f"Epoch {epoch + 1}/{ppo_config.epochs}")
        
        for batch in trainer.dataloader:
            queries = batch["query"]
            
            # Tokenize
            query_tensors = [tokenizer.encode(q, return_tensors="pt").squeeze() for q in queries]
            
            # Generate responses
            response_tensors = []
            for query in query_tensors:
                response = trainer.generate(query.unsqueeze(0), max_new_tokens=128)
                response_tensors.append(response.squeeze()[len(query):])
            
            # Decode responses
            responses = [tokenizer.decode(r, skip_special_tokens=True) for r in response_tensors]
            
            # Get rewards
            rewards = rule_based_reward(responses)
            reward_tensors = [torch.tensor([r]) for r in rewards]
            
            # PPO step
            stats = trainer.step(query_tensors, response_tensors, reward_tensors)
            
            logger.info(f"Mean reward: {sum(rewards)/len(rewards):.3f}")
    
    # Save
    trainer.save_pretrained(f"{CONFIG['output_dir']}/final")
    
    logger.info("=" * 60)
    logger.info("✅ PPO TRAINING COMPLETE!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
