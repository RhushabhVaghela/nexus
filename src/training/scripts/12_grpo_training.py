#!/usr/bin/env python3
"""
12_grpo_training.py
Stage 3: GRPO Training (5-7 days) - MAIN TRAINING STAGE

Group Relative Policy Optimization for reasoning optimization.
Implements group-based reward estimation and policy gradient updates.

Output: checkpoints/stage3_grpo/final/ (PRODUCTION MODEL)

Usage:
    python 12_grpo_training.py

Requirements:
    - unsloth: For optimized model loading
    - trl: For GRPOTrainer
    - transformers: For base model support
    - datasets: For data loading
"""

import os
import sys

# torch will be imported in main or check_env
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False

# Initialize logger first to avoid NoneType errors
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create logs directory if it doesn't exist
try:
    os.makedirs("logs", exist_ok=True)
    # Setup logging with file handler
    file_handler = logging.FileHandler("logs/grpo_training.log")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(file_handler)
except Exception as e:
    logger.warning(f"Could not create logs directory: {e}. Logging to console only.")


def check_env():
    """Verify environment dependencies."""
    try:
        from unsloth import FastLanguageModel, is_bfloat16_supported
        from trl import GRPOConfig, GRPOTrainer
        from datasets import load_dataset
        from transformers import TrainingArguments
    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        return False

    if torch and not torch.cuda.is_available():
        if logger:
            logger.warning("⚠️ No CUDA GPU detected. GRPO requires significant VRAM.")
        else:
            print("⚠️ No CUDA GPU detected. GRPO requires significant VRAM.")
        return False
    return True


try:
    from unsloth import FastLanguageModel, is_bfloat16_supported
    from trl import GRPOConfig, GRPOTrainer
    from datasets import load_dataset
    from transformers import TrainingArguments
except ImportError as e:
    logger.warning(
        f"Optional imports not available: {e}. Install with: pip install unsloth trl"
    )

CONFIG = {
    "checkpoint": "checkpoints/stage1_sft/final",
    "max_seq_length": 4096,
    "lora_rank": 32,
    "batch_size": 1,
    "grad_accum_steps": 4,
    "learning_rate": 5e-6,
    "epochs": 1,
    "num_generations": 4,
    "output_dir": "checkpoints/stage3_grpo",
}


def correctness_reward(
    completions: List[str], answers: List[str], **kwargs
) -> List[float]:
    """
    Reward based on answer correctness.

    Args:
        completions: List of model-generated completions
        answers: List of expected answers
        **kwargs: Additional arguments (unused)

    Returns:
        List of reward scores (0.0-1.0) for each completion
    """
    rewards = []
    for completion, answer in zip(completions, answers):
        completion_lower = completion.lower().strip()
        answer_lower = str(answer).lower().strip()

        if completion_lower == answer_lower or answer_lower in completion_lower:
            rewards.append(1.0)
        elif len(str(answer)) > 0 and any(
            word in completion_lower for word in str(answer).split()
        ):
            rewards.append(0.7)
        else:
            rewards.append(0.0)

    return rewards


def quality_reward(completions: List[str], **kwargs) -> List[float]:
    """
    Reward based on code/reasoning quality indicators.

    Args:
        completions: List of model-generated completions
        **kwargs: Additional arguments (unused)

    Returns:
        List of reward scores (0.0-1.0) for each completion
    """
    rewards = []
    for completion in completions:
        score = 0.0

        # Has structure
        if "<think>" in completion:
            score += 0.3
        if "[Tool:" in completion or "[Error:" in completion:
            score += 0.2
        if "[Final Answer]" in completion:
            score += 0.2

        # Not too long/short
        words = len(completion.split())
        if 50 < words < 1000:
            score += 0.3

        rewards.append(min(1.0, score))

    return rewards


def integration_reward(completions: List[str], **kwargs) -> List[float]:
    """
    Reward based on API/integration correctness indicators.

    Args:
        completions: List of model-generated completions
        **kwargs: Additional arguments (unused)

    Returns:
        List of reward scores (0.0-1.0) for each completion
    """
    rewards = []
    for completion in completions:
        score = 0.0

        keywords = ["api", "database", "auth", "integration", "webhook", "stripe"]
        found = sum(1 for kw in keywords if kw in completion.lower())
        score = min(0.5, found * 0.1)

        rewards.append(score)

    return rewards


def stack_compliance_reward(completions: List[str], **kwargs) -> List[float]:
    """
    Reward for using correct tech stack keywords based on context.

    Args:
        completions: List of model-generated completions
        **kwargs: Additional arguments (unused)

    Returns:
        List of reward scores (0.0-1.0) for each completion
    """
    rewards = []
    # Keywords that imply high-quality modern stack
    required_stack = [
        "next.js",
        "supabase",
        "shadcn",
        "tailwind",
        "typescript",
        "vite",
        "fastapi",
        "docker",
    ]

    for completion in completions:
        c_lower = completion.lower()
        # Count how many of the 'premium' stack items are mentioned
        found_count = sum(1 for item in required_stack if item in c_lower)
        # Cap at 0.5 bonus
        score = min(0.5, found_count * 0.1)
        rewards.append(score)
    return rewards


def replica_feature_reward(completions: List[str], **kwargs) -> List[float]:
    """
    Reward for implementing specific replica features (Artifacts, Preview, etc).

    Args:
        completions: List of model-generated completions
        **kwargs: Additional arguments (unused)

    Returns:
        List of reward scores (0.0-1.0) for each completion
    """
    rewards = []
    features = [
        "artifact",
        "preview",
        "iframe",
        "websocket",
        "crdt",
        "collaboration",
        "sandbox",
    ]

    for completion in completions:
        c_lower = completion.lower()
        found = any(f in c_lower for f in features)
        rewards.append(0.3 if found else 0.0)
    return rewards


def combined_reward(
    completions: List[str], answers: Optional[List[str]] = None, **kwargs
) -> List[float]:
    """
    Combined reward function combining multiple reward components.

    Combines correctness, quality, integration, stack compliance, and replica
    feature rewards with weighted averaging.

    Args:
        completions: List of model-generated completions
        answers: Optional list of expected answers for correctness scoring
        **kwargs: Additional arguments passed to reward functions

    Returns:
        List of combined reward scores (0.0-1.0) for each completion
    """
    correctness = correctness_reward(completions, answers or [""] * len(completions))
    quality = quality_reward(completions)
    integration = integration_reward(completions)
    stack = stack_compliance_reward(completions)
    replica = replica_feature_reward(completions)

    # Weighted combination
    combined = [
        0.3 * c + 0.2 * q + 0.2 * i + 0.15 * s + 0.15 * r
        for c, q, i, s, r in zip(correctness, quality, integration, stack, replica)
    ]

    return combined


def main():
    if not check_env():
        logger.error("❌ Environment check failed (missing dependencies or GPU).")
        sys.exit(1)

    logger.info("=" * 70)
    logger.info("🎓 STAGE 3: GRPO TRAINING (MAIN TRAINING)")
    logger.info("=" * 70)
    logger.info(f"Checkpoint: {CONFIG['checkpoint']}")
    logger.info(f"Learning rate: {CONFIG['learning_rate']}")
    logger.info(f"Num generations: {CONFIG['num_generations']}")
    logger.info(f"Expected duration: 5-7 days")
    logger.info("=" * 70)

    # Check input
    if not Path("rejection_sampled.jsonl").exists():
        logger.error("❌ rejection_sampled.jsonl not found")
        logger.error("   Run: python 05_rejection_sampling.py")
        return

    # Load model
    logger.info("\n📦 Loading model...")
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=CONFIG["checkpoint"],
            max_seq_length=CONFIG["max_seq_length"],
            dtype=torch.bfloat16 if is_bfloat16_supported() else torch.float32,
            load_in_4bit=True,
            gpu_memory_utilization=0.75,
        )
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        return

    # Add LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r=CONFIG["lora_rank"],
        lora_alpha=32,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )
    FastLanguageModel.for_training(model)
    logger.info("✓ Model loaded with LoRA")

    # Load data
    logger.info("\n📂 Loading rejection-sampled data...")

    # Check for streaming request or large file
    use_streaming = (
        os.path.getsize("rejection_sampled.jsonl") > 1 * 1024 * 1024 * 1024
    )  # >1GB auto-stream

    if use_streaming:
        logger.info("🌊 Using StreamingDatasetLoader for large dataset")
        try:
            from src.data.streaming_trainer import (
                StreamingDatasetLoader,
                StreamingConfig,
            )

            loader = StreamingDatasetLoader(
                "rejection_sampled.jsonl", StreamingConfig(buffer_size=1000)
            )
            dataset = loader.get_streaming_dataset()
        except ImportError:
            logger.warning("Streaming loader not found. Loading full dataset.")
            dataset = load_dataset(
                "json", data_files="rejection_sampled.jsonl", split="train"
            )
    else:
        dataset = load_dataset(
            "json", data_files="rejection_sampled.jsonl", split="train"
        )

    logger.info("✓ Data loaded")

    def prepare_grpo(sample):
        return {
            "prompt": f"Question: {sample['question']}\n\nLet me think:\n<think>",
            "answer": sample.get("answer", ""),
        }

    dataset = dataset.map(prepare_grpo)

    # GRPO config
    logger.info("\n⚙️  Configuring GRPO...")
    grpo_args = GRPOConfig(
        output_dir=CONFIG["output_dir"],
        per_device_train_batch_size=CONFIG["batch_size"],
        num_train_epochs=CONFIG["epochs"],
        learning_rate=CONFIG["learning_rate"],
        lr_scheduler_type="cosine",
        warmup_steps=50,
        logging_steps=5,
        save_steps=50,
        bf16=is_bfloat16_supported(),
        max_grad_norm=1.0,
        num_generations=CONFIG["num_generations"],
        report_to=["wandb"] if "WANDB_API_KEY" in os.environ else [],
    )

    # Trainer
    logger.info("\n🚀 Creating GRPO trainer...")
    # Add Callback for Pause/Resume
    from utils.callbacks import KeyboardPauseCallback

    pause_callback = KeyboardPauseCallback(
        flag_dir="flags", output_dir="logs", trainer_ref=None
    )

    # Trainer
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[combined_reward],  # Changed from reward_model to reward_funcs
        args=grpo_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        callbacks=[pause_callback],  # Add pause callback
    )

    # Link trainer reference
    pause_callback.trainer_ref = trainer

    # Check for checkpoint to resume
    last_checkpoint = None
    if os.path.isdir(CONFIG["output_dir"]):
        checkpoints = [
            d for d in os.listdir(CONFIG["output_dir"]) if d.startswith("checkpoint-")
        ]
        if checkpoints:
            checkpoints.sort(key=lambda x: int(x.split("-")[1]))
            last_checkpoint = os.path.join(CONFIG["output_dir"], checkpoints[-1])
            logger.info(f"🔄 Resuming from checkpoint: {last_checkpoint}")

    # Train
    logger.info("\n⚙️ Starting GRPO training...")
    logger.info(
        "💡 Tip: Run 'python3 utils/control_training.py --flag-dir flags' to pause"
    )
    logger.info("   This is the main training stage (5-7 days)")
    logger.info("   Monitor: nvidia-smi -l 1")
    logger.info("   Monitor: tail -f logs/grpo_training.log")

    try:
        train_result = trainer.train(resume_from_checkpoint=last_checkpoint)
        logger.info(f"\n✅ Training complete!")
    except KeyboardInterrupt:
        logger.warning("⚠️  Training interrupted")
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        return

    # Save
    logger.info("\n💾 Saving model...")
    model.save_pretrained(f"{CONFIG['output_dir']}/final")
    tokenizer.save_pretrained(f"{CONFIG['output_dir']}/final")
    logger.info(f"✓ Saved to: {CONFIG['output_dir']}/final")

    logger.info("\n" + "=" * 70)
    logger.info("✅ STAGE 3 COMPLETE!")
    logger.info("=" * 70)
    logger.info(f"Your production model: checkpoints/stage3_grpo/final/")
    logger.info(f"Next: Optional Stage 4 (Tool Integration)")
    logger.info(f"  python 07_tool_integration.py")


if __name__ == "__main__":
    main()
