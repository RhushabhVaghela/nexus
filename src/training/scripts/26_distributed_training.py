#!/usr/bin/env python3
"""
26_distributed_training.py

Distributed training support with DeepSpeed and FSDP for multi-GPU/multi-node training.

Supports:
- DeepSpeed ZeRO Stage 1/2/3
- PyTorch DDP (DistributedDataParallel) with optimizations
- PyTorch FSDP (Fully Sharded Data Parallel) v2
- Multi-node training with SLURM
- Gradient checkpointing
- Mixed precision (FP16/BF16)
- FlashAttention 2 integration
- Gradient synchronization optimizations
- Checkpoint sharding for distributed setup
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, Optional, Any, List, Callable
from dataclasses import dataclass, field, asdict
from contextlib import contextmanager
import time

# Globals to be initialized in main()
logger = None
TORCH_AVAILABLE = False
TRANSFORMERS_AVAILABLE = False
DEEPSPEED_AVAILABLE = False
FSDP_AVAILABLE = False


def check_env():
    """Verify environment dependencies and import conditional libraries."""
    global TORCH_AVAILABLE, TRANSFORMERS_AVAILABLE, DEEPSPEED_AVAILABLE, FSDP_AVAILABLE
    try:
        import torch
        import torch.distributed as _dist
        TORCH_AVAILABLE = True
    except ImportError:
        TORCH_AVAILABLE = False
        return False
        
    try:
        from transformers import Trainer
        TRANSFORMERS_AVAILABLE = True
    except ImportError:
        TRANSFORMERS_AVAILABLE = False
        return False
        
    try:
        import deepspeed
        DEEPSPEED_AVAILABLE = True
    except ImportError:
        DEEPSPEED_AVAILABLE = False
        # Deepspeed is optional for some modes
        
    # Check FSDP availability (PyTorch 2.0+)
    try:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        FSDP_AVAILABLE = True
    except ImportError:
        FSDP_AVAILABLE = False
        
    if not TORCH_AVAILABLE:
        print("[ERROR] PyTorch not available.")
        return False
        
    if not torch.cuda.is_available():
        print("⚠️ No CUDA GPU detected.")
        return False
    return True

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
    
    # NEW: Distributed optimizations
    ddp_bucket_cap_mb: float = 25.0  # Gradient bucket size for DDP
    gradient_as_bucket_view: bool = True  # Memory optimization for DDP
    static_graph: bool = False  # For models with unused parameters
    
    # NEW: FSDP specific
    fsdp_sharding_strategy: str = "FULL_SHARD"  # FULL_SHARD, SHARD_GRAD_OP, NO_SHARD, HYBRID_SHARD
    fsdp_backward_prefetch: str = "BACKWARD_PRE"  # BACKWARD_PRE, BACKWARD_POST
    fsdp_cpu_offload: bool = False
    fsdp_limit_all_gathers: bool = True
    fsdp_use_orig_params: bool = True  # Required for gradient checkpointing
    fsdp_sync_module_states: bool = True  # Sync module states across ranks
    
    # NEW: Checkpoint sharding
    enable_checkpoint_sharding: bool = True
    shard_size_gb: float = 5.0  # Size of each checkpoint shard
    
    # NEW: Gradient synchronization
    gradient_sync_every_n_steps: int = 1  # Sync every N steps (for large models)
    delay_grad_reduce: bool = False  # Delay gradient reduction
    

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
            "gradient_predivide_factor": "auto",
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
            "round_robin_gradients": True,  # Better load balancing
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
    
    # Gradient synchronization optimizations
    if config.gradient_sync_every_n_steps > 1:
        ds_config["gradient_accumulation_steps"] *= config.gradient_sync_every_n_steps
    
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
# FSDP CONFIG V2 (PyTorch 2.0+)
# ═══════════════════════════════════════════════════════════════

def get_fsdp_config(config: DistributedConfig) -> Dict:
    """Generate FSDP configuration with v2 optimizations."""
    
    if not FSDP_AVAILABLE:
        raise RuntimeError("FSDP not available. Requires PyTorch 2.0+")
    
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
    
    # Sharding strategy mapping
    sharding_strategies = {
        "FULL_SHARD": ShardingStrategy.FULL_SHARD,
        "SHARD_GRAD_OP": ShardingStrategy.SHARD_GRAD_OP,
        "NO_SHARD": ShardingStrategy.NO_SHARD,
        "HYBRID_SHARD": ShardingStrategy.HYBRID_SHARD,
        "HYBRID_SHARD_ZERO2": ShardingStrategy._HYBRID_SHARD_ZERO2,
    }
    
    # Backward prefetch mapping
    backward_prefetches = {
        "BACKWARD_PRE": BackwardPrefetch.BACKWARD_PRE,
        "BACKWARD_POST": BackwardPrefetch.BACKWARD_POST,
    }
    
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
        "sharding_strategy": sharding_strategies.get(config.fsdp_sharding_strategy, ShardingStrategy.FULL_SHARD),
        "cpu_offload": CPUOffload(offload_params=config.fsdp_cpu_offload),
        "mixed_precision": mp_policy,
        "backward_prefetch": backward_prefetches.get(config.fsdp_backward_prefetch, BackwardPrefetch.BACKWARD_PRE),
        "use_orig_params": config.fsdp_use_orig_params,
        "limit_all_gathers": config.fsdp_limit_all_gathers,
        "sync_module_states": config.fsdp_sync_module_states,
        # NEW: Device mesh for multi-node
        "device_mesh": None,  # Will be set up during initialization
    }
    
    return fsdp_config


# ═══════════════════════════════════════════════════════════════
# DDP OPTIMIZATIONS
# ═══════════════════════════════════════════════════════════════

class DDPModelWrapper:
    """
    Enhanced DDP wrapper with gradient synchronization optimizations.
    """
    
    def __init__(
        self,
        model: nn.Module,
        device_ids: Optional[List[int]] = None,
        output_device: Optional[int] = None,
        bucket_cap_mb: float = 25.0,
        gradient_as_bucket_view: bool = True,
        static_graph: bool = False,
        delay_grad_reduce: bool = False,
        gradient_sync_every_n_steps: int = 1,
    ):
        """
        Initialize DDP with optimizations.
        
        Args:
            model: The model to wrap
            device_ids: List of GPU IDs
            output_device: Device for output
            bucket_cap_mb: Bucket size in MB for gradient all-reduce
            gradient_as_bucket_view: Save memory by using gradient buckets as views
            static_graph: For models with unused parameters
            delay_grad_reduce: Delay gradient reduction for better overlap
            gradient_sync_every_n_steps: Sync gradients every N steps
        """
        import torch.nn.parallel as parallel
        
        self.model = parallel.DistributedDataParallel(
            model,
            device_ids=device_ids,
            output_device=output_device,
            bucket_cap_mb=bucket_cap_mb,
            gradient_as_bucket_view=gradient_as_bucket_view,
            static_graph=static_graph,
            find_unused_parameters=not static_graph,
            # Performance optimizations
            delay_all_reduce=delay_grad_reduce,
        )
        
        self.gradient_sync_every_n_steps = gradient_sync_every_n_steps
        self._step_count = 0
        
    def forward(self, *args, **kwargs):
        """Forward pass through DDP model."""
        return self.model(*args, **kwargs)
    
    def backward(self, loss: torch.Tensor):
        """Backward pass with gradient synchronization control."""
        loss.backward()
        
        self._step_count += 1
        
        # Delay gradient synchronization if configured
        if self._step_count % self.gradient_sync_every_n_steps == 0:
            self.model.reducer.prepare_for_backward([])
    
    @contextmanager
    def no_sync(self):
        """Context manager to temporarily disable gradient synchronization."""
        with self.model.no_sync():
            yield


def setup_ddp(
    model: nn.Module,
    config: DistributedConfig,
    local_rank: int,
    find_unused_parameters: bool = False
) -> DDPModelWrapper:
    """
    Setup DDP with optimizations.
    
    Args:
        model: Model to wrap
        config: Distributed configuration
        local_rank: Local rank
        find_unused_parameters: Whether to find unused parameters
        
    Returns:
        DDPModelWrapper with optimizations
    """
    logger.info(f"Setting up DDP with bucket_cap={config.ddp_bucket_cap_mb}MB")
    
    return DDPModelWrapper(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        bucket_cap_mb=config.ddp_bucket_cap_mb,
        gradient_as_bucket_view=config.gradient_as_bucket_view,
        static_graph=config.static_graph and not find_unused_parameters,
        delay_grad_reduce=config.delay_grad_reduce,
        gradient_sync_every_n_steps=config.gradient_sync_every_n_steps,
    )


# ═══════════════════════════════════════════════════════════════
# CHECKPOINT SHARDING
# ═══════════════════════════════════════════════════════════════

class CheckpointSharder:
    """
    Handles sharded checkpointing for distributed training.
    
    Features:
    - Shards large models into smaller files
    - Saves/loads checkpoints efficiently across ranks
    - Supports partial checkpoint loading for fine-tuning
    """
    
    def __init__(self, shard_size_gb: float = 5.0, world_size: int = 1, rank: int = 0):
        """
        Initialize checkpoint sharder.
        
        Args:
            shard_size_gb: Maximum size of each shard in GB
            world_size: Total number of processes
            rank: Current process rank
        """
        self.shard_size_gb = shard_size_gb
        self.shard_size_bytes = int(shard_size_gb * 1024 * 1024 * 1024)
        self.world_size = world_size
        self.rank = rank
        
    def shard_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> List[Dict[str, torch.Tensor]]:
        """
        Split state dict into shards based on size.
        
        Args:
            state_dict: Full model state dict
            
        Returns:
            List of sharded state dicts
        """
        shards = []
        current_shard = {}
        current_size = 0
        
        for key, tensor in state_dict.items():
            tensor_size = tensor.numel() * tensor.element_size()
            
            # If single tensor exceeds shard size, it gets its own shard
            if tensor_size > self.shard_size_bytes and current_shard:
                shards.append(current_shard)
                current_shard = {}
                current_size = 0
            
            # Start new shard if current would exceed size
            if current_size + tensor_size > self.shard_size_bytes and current_shard:
                shards.append(current_shard)
                current_shard = {}
                current_size = 0
            
            current_shard[key] = tensor
            current_size += tensor_size
        
        # Add remaining shard
        if current_shard:
            shards.append(current_shard)
        
        return shards
    
    def save_checkpoint_sharded(
        self,
        state_dict: Dict[str, torch.Tensor],
        output_dir: Path,
        prefix: str = "model_shard"
    ):
        """
        Save state dict as sharded checkpoint.
        
        Args:
            state_dict: Model state dict
            output_dir: Directory to save shards
            prefix: Prefix for shard filenames
        """
        if self.rank != 0:
            return  # Only rank 0 saves full model
            
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Shard the state dict
        shards = self.shard_state_dict(state_dict)
        
        # Save each shard
        shard_info = {
            "num_shards": len(shards),
            "shard_size_gb": self.shard_size_gb,
            "world_size": self.world_size,
        }
        
        for i, shard in enumerate(shards):
            shard_path = output_dir / f"{prefix}_{i:05d}_of_{len(shards):05d}.pt"
            torch.save(shard, shard_path)
            shard_info[f"shard_{i}"] = {
                "path": str(shard_path),
                "num_params": len(shard),
            }
        
        # Save shard metadata
        metadata_path = output_dir / f"{prefix}_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(shard_info, f, indent=2)
        
        logger.info(f"Saved {len(shards)} checkpoint shards to {output_dir}")
    
    def load_checkpoint_sharded(
        self,
        checkpoint_dir: Path,
        prefix: str = "model_shard",
        map_location: str = "cpu"
    ) -> Dict[str, torch.Tensor]:
        """
        Load sharded checkpoint.
        
        Args:
            checkpoint_dir: Directory containing shards
            prefix: Shard filename prefix
            map_location: Device to map tensors to
            
        Returns:
            Reconstructed state dict
        """
        checkpoint_dir = Path(checkpoint_dir)
        
        # Load metadata
        metadata_path = checkpoint_dir / f"{prefix}_metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Shard metadata not found: {metadata_path}")
        
        with open(metadata_path) as f:
            metadata = json.load(f)
        
        # Load all shards
        state_dict = {}
        for i in range(metadata["num_shards"]):
            shard_path = checkpoint_dir / f"{prefix}_{i:05d}_of_{metadata['num_shards']:05d}.pt"
            if not shard_path.exists():
                raise FileNotFoundError(f"Shard not found: {shard_path}")
            
            shard = torch.load(shard_path, map_location=map_location, weights_only=False)
            state_dict.update(shard)
            logger.debug(f"Loaded shard {i+1}/{metadata['num_shards']}")
        
        logger.info(f"Loaded checkpoint with {len(state_dict)} parameters from {metadata['num_shards']} shards")
        return state_dict
    
    def load_shard_for_rank(
        self,
        checkpoint_dir: Path,
        param_assignments: Dict[str, int],
        prefix: str = "model_shard",
        map_location: str = "cpu"
    ) -> Dict[str, torch.Tensor]:
        """
        Load only the shards needed for current rank.
        
        Args:
            checkpoint_dir: Directory containing shards
            param_assignments: Dict mapping parameter names to rank assignments
            prefix: Shard filename prefix
            map_location: Device to map tensors to
            
        Returns:
            State dict with parameters assigned to this rank
        """
        checkpoint_dir = Path(checkpoint_dir)
        
        # Determine which shards we need
        needed_shards = set()
        for param_name, assigned_rank in param_assignments.items():
            if assigned_rank == self.rank:
                # Calculate which shard contains this parameter
                shard_idx = self._get_shard_index(param_name)
                needed_shards.add(shard_idx)
        
        # Load only needed shards
        state_dict = {}
        for shard_idx in needed_shards:
            shard_path = checkpoint_dir / f"{prefix}_{shard_idx:05d}_of_*.pt"
            # Find actual shard file
            matching_shards = list(checkpoint_dir.glob(f"{prefix}_{shard_idx:05d}_of_*.pt"))
            if matching_shards:
                shard = torch.load(matching_shards[0], map_location=map_location, weights_only=False)
                # Filter to only parameters for this rank
                for param_name, tensor in shard.items():
                    if param_assignments.get(param_name) == self.rank:
                        state_dict[param_name] = tensor
        
        return state_dict
    
    def _get_shard_index(self, param_name: str) -> int:
        """Calculate shard index for a parameter (simple hash-based)."""
        return hash(param_name) % 1000  # Will be determined from actual metadata


# ═══════════════════════════════════════════════════════════════
# DISTRIBUTED TRAINING SETUP
# ═══════════════════════════════════════════════════════════════

def setup_distributed(config: DistributedConfig) -> int:
    """Initialize distributed training environment."""
    
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch not available")
    
    import torch.distributed as dist
    
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
    
    if config.backend == "deepspeed" and DEEPSPEED_AVAILABLE:
        deepspeed.init_distributed()
    else:
        dist.init_process_group(
            backend="nccl",
            world_size=world_size,
            rank=local_rank,
        )
    
    torch.cuda.set_device(local_rank)
    
    # Log setup info
    logger.info(f"Distributed setup complete:")
    logger.info(f"  Rank: {local_rank}/{world_size}")
    logger.info(f"  Backend: {config.backend}")
    logger.info(f"  Device: cuda:{local_rank}")
    
    return local_rank


def load_model_distributed(config: DistributedConfig, local_rank: int):
    """Load model with distributed configuration."""
    
    if not TRANSFORMERS_AVAILABLE:
        raise RuntimeError("transformers not available")
    
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
    
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
    
    # Wrap with appropriate distributed wrapper
    if config.backend == "ddp":
        model = setup_ddp(model, config, local_rank)
        logger.info("Model wrapped with DDP")
    elif config.backend == "fsdp" and FSDP_AVAILABLE:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        fsdp_config = get_fsdp_config(config)
        
        # Auto-wrap policy for transformer models
        auto_wrap_policy = get_transformer_auto_wrap_policy(model)
        if auto_wrap_policy:
            fsdp_config["auto_wrap_policy"] = auto_wrap_policy
        
        model = FSDP(model, **fsdp_config)
        logger.info("Model wrapped with FSDP")
    
    return model


def get_transformer_auto_wrap_policy(model):
    """Get auto wrap policy for transformer models."""
    try:
        from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
        
        # Common transformer layer classes
        transformer_layer_classes = (
            "LlamaDecoderLayer",
            "Qwen2DecoderLayer", 
            "MistralDecoderLayer",
            "GPTNeoXLayer",
            "OPTDecoderLayer",
        )
        
        # Find which class exists in the model
        for layer_class_name in transformer_layer_classes:
            if hasattr(model, "model") and hasattr(model.model, "model"):
                for name, module in model.model.model.named_modules():
                    if module.__class__.__name__ == layer_class_name:
                        return transformer_auto_wrap_policy(
                            {layer_class_name: module.__class__},
                            recurse=True,
                        )
    except Exception as e:
        logger.warning(f"Could not set up auto wrap policy: {e}")
    
    return None


def get_training_arguments(config: DistributedConfig) -> Any:
    """Build HuggingFace TrainingArguments."""
    
    from transformers import TrainingArguments
    
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
        
        # NEW: DDP optimizations
        ddp_bucket_cap_mb=config.ddp_bucket_cap_mb,
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

# Performance optimizations
export NCCL_NSOCKS_PERTHREAD=4
export NCCL_SOCKET_NTHREADS=2
export OMP_NUM_THREADS=8

echo "=== Training Info ==="
echo "Nodes: $SLURM_NNODES"
echo "GPUs per node: $SLURM_GPUS_PER_NODE"
echo "Master: $MASTER_ADDR:$MASTER_PORT"
echo "Backend: {config.backend}"
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
    --ddp-bucket-cap {config.ddp_bucket_cap_mb} \\
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
    if not check_env():
         sys.exit(1)
         
    global logger
    logger = setup_logger(__name__, "logs/distributed_training.log")

    import torch
    import torch.distributed as dist
    from transformers import TrainingArguments, AutoModelForCausalLM, AutoTokenizer, Trainer
    from transformers import BitsAndBytesConfig
    from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
    from datasets import load_dataset as hf_load_dataset

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
    
    # NEW: DDP optimizations
    parser.add_argument("--ddp-bucket-cap", type=float, default=25.0,
                        help="DDP gradient bucket size in MB")
    parser.add_argument("--gradient-as-bucket-view", action="store_true", default=True,
                        help="Use gradient buckets as views to save memory")
    parser.add_argument("--static-graph", action="store_true",
                        help="Enable static graph for DDP (faster but requires no unused params)")
    parser.add_argument("--delay-grad-reduce", action="store_true",
                        help="Delay gradient reduction for better compute/communication overlap")
    parser.add_argument("--grad-sync-every-n-steps", type=int, default=1,
                        help="Sync gradients every N steps (for large models)")
    
    # NEW: FSDP options
    parser.add_argument("--fsdp-sharding", type=str, default="FULL_SHARD",
                        choices=["FULL_SHARD", "SHARD_GRAD_OP", "NO_SHARD", "HYBRID_SHARD"])
    parser.add_argument("--fsdp-cpu-offload", action="store_true",
                        help="Offload parameters to CPU in FSDP")
    
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
    
    # NEW: Checkpoint sharding
    parser.add_argument("--shard-checkpoint", action="store_true",
                        help="Save checkpoints as shards for large models")
    parser.add_argument("--shard-size-gb", type=float, default=5.0,
                        help="Size of each checkpoint shard in GB")
    
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
        # NEW options
        ddp_bucket_cap_mb=args.ddp_bucket_cap,
        gradient_as_bucket_view=args.gradient_as_bucket_view,
        static_graph=args.static_graph,
        delay_grad_reduce=args.delay_grad_reduce,
        gradient_sync_every_n_steps=args.grad_sync_every_n_steps,
        fsdp_sharding_strategy=args.fsdp_sharding,
        fsdp_cpu_offload=args.fsdp_cpu_offload,
        enable_checkpoint_sharding=args.shard_checkpoint,
        shard_size_gb=args.shard_size_gb,
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
        
        if config.backend == "fsdp":
            print("\n=== FSDP Config ===")
            fsdp_config = get_fsdp_config(config)
            print(json.dumps({k: str(v) for k, v in fsdp_config.items()}, indent=2))
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
            "DDP Bucket": f"{config.ddp_bucket_cap_mb}MB",
            "Gradient Sync": f"every {config.gradient_sync_every_n_steps} steps",
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
    
    # Initialize checkpoint sharder if needed
    sharder = None
    if config.enable_checkpoint_sharding:
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        sharder = CheckpointSharder(
            shard_size_gb=config.shard_size_gb,
            world_size=world_size,
            rank=local_rank
        )
        logger.info(f"Checkpoint sharding enabled ({config.shard_size_gb}GB shards)")
    
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
    if config.enable_checkpoint_sharding and sharder and local_rank == 0:
        logger.info("Saving sharded checkpoint...")
        state_dict = model.state_dict()
        sharder.save_checkpoint_sharded(state_dict, Path(config.output_dir) / "sharded")
    
    trainer.save_model(config.output_dir)
    logger.info(f"Model saved to {config.output_dir}")
    
    log_completion(
        logger,
        "Distributed Training",
        {"Output": config.output_dir},
    )


if __name__ == "__main__":
    main()
