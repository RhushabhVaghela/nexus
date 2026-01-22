#!/usr/bin/env python3
"""
26_distributed_training.py

Distributed training support with DeepSpeed and FSDP for multi-GPU/multi-node training.

Supports:
- DeepSpeed ZeRO Stage 1/2/3
- PyTorch FSDP (Fully Sharded Data Parallel)
- Multi-node training with SLURM
- Gradient checkpointing
- Mixed precision (FP16/BF16)
- FlashAttention 2 integration
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, Optional, Any
from dataclasses import dataclass, field, asdict

try:
    import torch
    import torch.distributed as dist
    from torch.utils.data import DataLoader, DistributedSampler
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainingArguments,
        Trainer,
        BitsAndBytesConfig,
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    import deepspeed
    DEEPSPEED_AVAILABLE = True
except ImportError:
    DEEPSPEED_AVAILABLE = False

sys.path.insert(0, str(Path(__file__).parent))
from utils.logging_config import setup_logger, log_header, log_completion

logger = setup_logger(__name__, "logs/distributed_training.log")


# ═══════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════

@dataclass
class DistributedConfig:
    """Distributed training configuration."""
    
    # Model
    model_name: str = "/mnt/e/data/models/Qwen2.5-Omni-7B-GPTQ-Int4"
    max_seq_length: int = 4096
    
    # Distributed backend
    backend: str = "deepspeed"  # "deepspeed", "fsdp", "ddp"
    zero_stage: int = 3  # 0, 1, 2, 3 for DeepSpeed
    
    # Hardware
    num_gpus: int = 1
    num_nodes: int = 1
    master_addr: str = "localhost"
    master_port: int = 29500
    
    # Training
    batch_size_per_gpu: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-5
    num_epochs: int = 3
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    
    # Precision
    fp16: bool = False
    bf16: bool = True
    
    # Memory optimization
    gradient_checkpointing: bool = True
    cpu_offload: bool = False
    nvme_offload: bool = False
    offload_dir: str = "/tmp/offload"
    
    # LoRA
    use_lora: bool = True
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    lora_target_modules: list = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"])
    
    # Data
    train_data_path: str = "/mnt/e/data/training/train.jsonl"
    eval_data_path: str = "/mnt/e/data/training/eval.jsonl"
    
    # Output
    output_dir: str = "./checkpoints/distributed"
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500
    
    # Checkpointing
    save_total_limit: int = 3
    resume_from_checkpoint: Optional[str] = None


# ═══════════════════════════════════════════════════════════════
# DEEPSPEED CONFIGS
# ═══════════════════════════════════════════════════════════════

def get_deepspeed_config(config: DistributedConfig) -> Dict:
    """Generate DeepSpeed configuration based on settings."""
    
    ds_config = {
        "train_batch_size": config.batch_size_per_gpu * config.num_gpus * config.gradient_accumulation_steps,
        "train_micro_batch_size_per_gpu": config.batch_size_per_gpu,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": config.learning_rate,
                "betas": [0.9, 0.95],
                "eps": 1e-8,
                "weight_decay": config.weight_decay,
            },
        },
        
        "scheduler": {
            "type": "WarmupDecayLR",
            "params": {
                "warmup_min_lr": 0,
                "warmup_max_lr": config.learning_rate,
                "warmup_num_steps": "auto",
                "total_num_steps": "auto",
            },
        },
        
        "gradient_clipping": 1.0,
        "prescale_gradients": False,
        "wall_clock_breakdown": False,
    }
    
    # Precision
    if config.bf16:
        ds_config["bf16"] = {"enabled": True}
    elif config.fp16:
        ds_config["fp16"] = {
            "enabled": True,
            "loss_scale": 0,
            "loss_scale_window": 1000,
            "initial_scale_power": 16,
            "hysteresis": 2,
            "min_loss_scale": 1,
        }
    
    # ZeRO configuration
    if config.zero_stage == 0:
        ds_config["zero_optimization"] = {"stage": 0}
    elif config.zero_stage == 1:
        ds_config["zero_optimization"] = {
            "stage": 1,
            "reduce_bucket_size": 5e8,
        }
    elif config.zero_stage == 2:
        ds_config["zero_optimization"] = {
            "stage": 2,
            "offload_optimizer": {
                "device": "cpu" if config.cpu_offload else "none",
            },
            "contiguous_gradients": True,
            "overlap_comm": True,
            "reduce_bucket_size": 5e8,
            "reduce_scatter": True,
        }
    elif config.zero_stage == 3:
        ds_config["zero_optimization"] = {
            "stage": 3,
            "offload_optimizer": {
                "device": "cpu" if config.cpu_offload else "none",
                "pin_memory": True,
            },
            "offload_param": {
                "device": "cpu" if config.cpu_offload else "none",
                "pin_memory": True,
            },
            "overlap_comm": True,
            "contiguous_gradients": True,
            "sub_group_size": 1e9,
            "reduce_bucket_size": "auto",
            "stage3_prefetch_bucket_size": "auto",
            "stage3_param_persistence_threshold": "auto",
            "stage3_max_live_parameters": 1e9,
            "stage3_max_reuse_distance": 1e9,
            "stage3_gather_16bit_weights_on_model_save": True,
        }
        
        if config.nvme_offload:
            ds_config["zero_optimization"]["offload_optimizer"]["nvme_path"] = config.offload_dir
            ds_config["zero_optimization"]["offload_param"]["nvme_path"] = config.offload_dir
            ds_config["zero_optimization"]["aio"] = {
                "block_size": 262144,
                "queue_depth": 32,
                "thread_count": 1,
                "single_submit": False,
                "overlap_events": True,
            }
    
    # Gradient checkpointing
    if config.gradient_checkpointing:
        ds_config["activation_checkpointing"] = {
            "partition_activations": True,
            "cpu_checkpointing": config.cpu_offload,
            "contiguous_memory_optimization": True,
            "number_checkpoints": None,
            "synchronize_checkpoint_boundary": False,
            "profile": False,
        }
    
    return ds_config


def save_deepspeed_config(config: DistributedConfig, path: Path):
    """Save DeepSpeed config to JSON file."""
    ds_config = get_deepspeed_config(config)
    with open(path, "w") as f:
        json.dump(ds_config, f, indent=2)
    logger.info(f"DeepSpeed config saved to {path}")


# ═══════════════════════════════════════════════════════════════
# FSDP CONFIG
# ═══════════════════════════════════════════════════════════════

def get_fsdp_config(config: DistributedConfig) -> Dict:
    """Generate FSDP configuration."""
    
    from torch.distributed.fsdp import (
        FullyShardedDataParallel as FSDP,
        MixedPrecision,
        BackwardPrefetch,
        ShardingStrategy,
        CPUOffload,
    )
    from torch.distributed.fsdp.wrap import (
        transformer_auto_wrap_policy,
        size_based_auto_wrap_policy,
    )
    
    # Mixed precision
    if config.bf16:
        mp_policy = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        )
    elif config.fp16:
        mp_policy = MixedPrecision(
            param_dtype=torch.float16,
            reduce_dtype=torch.float16,
            buffer_dtype=torch.float16,
        )
    else:
        mp_policy = None
    
    fsdp_config = {
        "sharding_strategy": ShardingStrategy.FULL_SHARD,
        "cpu_offload": CPUOffload(offload_params=config.cpu_offload),
        "mixed_precision": mp_policy,
        "backward_prefetch": BackwardPrefetch.BACKWARD_PRE,
        "use_orig_params": True,
        "limit_all_gathers": True,
    }
    
    return fsdp_config


# ═══════════════════════════════════════════════════════════════
# TRAINING SETUP
# ═══════════════════════════════════════════════════════════════

def setup_distributed(config: DistributedConfig) -> int:
    """Initialize distributed training environment."""
    
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch not available")
    
    # Set environment variables for multi-node
    os.environ.setdefault("MASTER_ADDR", config.master_addr)
    os.environ.setdefault("MASTER_PORT", str(config.master_port))
    
    # Check if already initialized (e.g., by torchrun)
    if dist.is_initialized():
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        return local_rank
    
    # Initialize process group
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = config.num_gpus * config.num_nodes
    
    if config.backend == "deepspeed":
        deepspeed.init_distributed()
    else:
        dist.init_process_group(
            backend="nccl",
            world_size=world_size,
            rank=local_rank,
        )
    
    torch.cuda.set_device(local_rank)
    
    return local_rank


def load_model_distributed(config: DistributedConfig, local_rank: int):
    """Load model with distributed configuration."""
    
    if not TRANSFORMERS_AVAILABLE:
        raise RuntimeError("transformers not available")
    
    logger.info(f"Loading model {config.model_name} on rank {local_rank}")
    
    # Quantization config for QLoRA
    bnb_config = None
    if config.use_lora:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if config.bf16 else torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16 if config.bf16 else torch.float16,
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
        device_map={"": local_rank} if config.use_lora else None,
    )
    
    # Enable gradient checkpointing
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    
    # Apply LoRA
    if config.use_lora:
        model = prepare_model_for_kbit_training(model)
        
        lora_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            target_modules=config.lora_target_modules,
            lora_dropout=config.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    
    return model


def get_training_arguments(config: DistributedConfig) -> TrainingArguments:
    """Build HuggingFace TrainingArguments."""
    
    args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size_per_gpu,
        per_device_eval_batch_size=config.batch_size_per_gpu,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        lr_scheduler_type="cosine",
        
        # Precision
        fp16=config.fp16,
        bf16=config.bf16,
        
        # Logging
        logging_steps=config.logging_steps,
        logging_first_step=True,
        report_to="tensorboard",
        
        # Saving
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        save_strategy="steps",
        
        # Evaluation
        eval_strategy="steps" if config.eval_data_path else "no",
        eval_steps=config.eval_steps,
        
        # Distributed
        ddp_find_unused_parameters=False,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        
        # Checkpointing
        gradient_checkpointing=config.gradient_checkpointing,
        resume_from_checkpoint=config.resume_from_checkpoint,
        
        # DeepSpeed
        deepspeed=None,  # Set later if using DeepSpeed
    )
    
    # Add DeepSpeed config
    if config.backend == "deepspeed":
        ds_config_path = Path(config.output_dir) / "ds_config.json"
        ds_config_path.parent.mkdir(parents=True, exist_ok=True)
        save_deepspeed_config(config, ds_config_path)
        args.deepspeed = str(ds_config_path)
    
    return args


# ═══════════════════════════════════════════════════════════════
# SLURM INTEGRATION
# ═══════════════════════════════════════════════════════════════

def generate_slurm_script(config: DistributedConfig, script_path: Path):
    """Generate SLURM submission script for multi-node training."""
    
    slurm_script = f"""#!/bin/bash
#SBATCH --job-name=nexus_training
#SBATCH --nodes={config.num_nodes}
#SBATCH --ntasks-per-node={config.num_gpus}
#SBATCH --gpus-per-node={config.num_gpus}
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

# Load modules (adjust for your cluster)
module load cuda/12.1
module load python/3.11

# Activate environment
source ~/.bashrc
conda activate nexus_training

# Set environment variables
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT={config.master_port}
export WORLD_SIZE=$((SLURM_NNODES * SLURM_GPUS_PER_NODE))
export NODE_RANK=$SLURM_NODEID

# Set CUDA devices
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Enable NCCL debugging
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=2

echo "=== Training Info ==="
echo "Nodes: $SLURM_NNODES"
echo "GPUs per node: $SLURM_GPUS_PER_NODE"
echo "Master: $MASTER_ADDR:$MASTER_PORT"
echo "===================="

# Run training
srun torchrun \\
    --nnodes=$SLURM_NNODES \\
    --nproc_per_node=$SLURM_GPUS_PER_NODE \\
    --rdzv_id=$SLURM_JOB_ID \\
    --rdzv_backend=c10d \\
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \\
    src/26_distributed_training.py \\
    --model {config.model_name} \\
    --backend {config.backend} \\
    --zero-stage {config.zero_stage} \\
    --train-data {config.train_data_path} \\
    --output-dir {config.output_dir}

echo "Training complete!"
"""
    
    with open(script_path, "w") as f:
        f.write(slurm_script)
    
    logger.info(f"SLURM script saved to {script_path}")
    logger.info(f"Submit with: sbatch {script_path}")


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Distributed training for Nexus Model")
    
    # Model
    parser.add_argument("--model", type=str, default="/mnt/e/data/models/Qwen2.5-Omni-7B-GPTQ-Int4")
    parser.add_argument("--max-seq-length", type=int, default=4096)
    
    # Distributed
    parser.add_argument("--backend", type=str, default="deepspeed",
                        choices=["deepspeed", "fsdp", "ddp"])
    parser.add_argument("--zero-stage", type=int, default=3, choices=[0, 1, 2, 3])
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--num-nodes", type=int, default=1)
    
    # Training
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=3)
    
    # Memory
    parser.add_argument("--gradient-checkpointing", action="store_true", default=True)
    parser.add_argument("--cpu-offload", action="store_true")
    parser.add_argument("--nvme-offload", action="store_true")
    
    # LoRA
    parser.add_argument("--use-lora", action="store_true", default=True)
    parser.add_argument("--lora-r", type=int, default=64)
    parser.add_argument("--lora-alpha", type=int, default=128)
    
    # Data
    parser.add_argument("--train-data", type=str, required=False)
    parser.add_argument("--eval-data", type=str, required=False)
    
    # Output
    parser.add_argument("--output-dir", type=str, default="./checkpoints/distributed")
    parser.add_argument("--resume", type=str, default=None)
    
    # Utilities
    parser.add_argument("--generate-slurm", action="store_true",
                        help="Generate SLURM script only")
    parser.add_argument("--generate-ds-config", action="store_true",
                        help="Generate DeepSpeed config only")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print config without training")
    
    args = parser.parse_args()
    
    # Build config
    config = DistributedConfig(
        model_name=args.model,
        max_seq_length=args.max_seq_length,
        backend=args.backend,
        zero_stage=args.zero_stage,
        num_gpus=args.num_gpus,
        num_nodes=args.num_nodes,
        batch_size_per_gpu=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        num_epochs=args.epochs,
        gradient_checkpointing=args.gradient_checkpointing,
        cpu_offload=args.cpu_offload,
        nvme_offload=args.nvme_offload,
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        train_data_path=args.train_data or "/mnt/e/data/training/train.jsonl",
        eval_data_path=args.eval_data,
        output_dir=args.output_dir,
        resume_from_checkpoint=args.resume,
    )
    
    # Handle utility modes
    if args.generate_slurm:
        generate_slurm_script(config, Path("submit_training.slurm"))
        return
    
    if args.generate_ds_config:
        save_deepspeed_config(config, Path(args.output_dir) / "ds_config.json")
        return
    
    if args.dry_run:
        print("\n=== Distributed Training Config ===")
        for key, value in asdict(config).items():
            print(f"  {key}: {value}")
        print("\n=== DeepSpeed Config ===")
        ds_config = get_deepspeed_config(config)
        print(json.dumps(ds_config, indent=2))
        return
    
    # Run training
    log_header(
        logger,
        "DISTRIBUTED TRAINING",
        {
            "Model": config.model_name,
            "Backend": config.backend,
            "ZeRO Stage": config.zero_stage,
            "GPUs": f"{config.num_gpus} x {config.num_nodes} nodes",
            "LoRA": config.use_lora,
        },
    )
    
    # Setup distributed
    local_rank = setup_distributed(config)
    logger.info(f"Initialized rank {local_rank}")
    
    # Load model
    model = load_model_distributed(config, local_rank)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Get training arguments
    training_args = get_training_arguments(config)
    
    # Load datasets
    from datasets import load_dataset as hf_load_dataset
    
    if Path(config.train_data_path).exists():
        train_dataset = hf_load_dataset(
            "json",
            data_files=config.train_data_path,
            split="train",
        )
    else:
        logger.warning(f"Train data not found: {config.train_data_path}")
        logger.info("Using dummy dataset for demonstration")
        train_dataset = hf_load_dataset(
            "tatsu-lab/alpaca",
            split="train[:1000]",
        )
    
    eval_dataset = None
    if config.eval_data_path and Path(config.eval_data_path).exists():
        eval_dataset = hf_load_dataset(
            "json",
            data_files=config.eval_data_path,
            split="train",
        )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )
    
    # Train
    logger.info("Starting training...")
    trainer.train(resume_from_checkpoint=config.resume_from_checkpoint)
    
    # Save final model
    trainer.save_model(config.output_dir)
    logger.info(f"Model saved to {config.output_dir}")
    
    log_completion(
        logger,
        "Distributed Training",
        {"Output": config.output_dir},
    )


if __name__ == "__main__":
    main()
