# Nexus Pipeline Guide

## Overview

The Nexus platform implements a sophisticated pipeline architecture designed for high-performance inference and training operations. This guide provides comprehensive documentation on pipeline configuration, stage chaining, checkpoint management, and error handling mechanisms. Understanding the pipeline system is essential for optimizing model inference, managing training workflows, and implementing custom pipeline stages.

The pipeline system is built around three core concepts: stages that perform specific operations, checkpoints that enable fault tolerance and resumption, and workers that execute pipeline stages in parallel. The architecture supports both synchronous and asynchronous pipeline execution, with dynamic batching, layer skipping, and KV cache optimization available as configurable pipeline stages.

This guide covers the inference pipeline for model serving, the training pipeline for fine-tuning operations, and the benchmarking pipeline for performance evaluation. Each section includes configuration options, stage chaining patterns, checkpoint management strategies, and error handling approaches. For deployment information, see the Deployment Guide.

## Installation

### Prerequisites

```bash
# Python 3.10 or higher
python --version  # Must be >= 3.10

# PyTorch with CUDA support (recommended)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# DeepSpeed for distributed training
pip install deepspeed

# Transformers and related libraries
pip install transformers datasets accelerate peft

# Nexus core dependencies
pip install nexus-core
```

### Additional Dependencies

```bash
# For KV cache optimization
pip install nexus-kvcache

# For layer fusion operations
pip install nexus-fusion

# For benchmarking
pip install nexus-benchmarks

# For monitoring integration
pip install nexus-monitoring prometheus-client
```

## Usage

### Basic Pipeline Execution

```python
from nexus.pipeline import InferencePipeline, TrainingPipeline
from nexus.stages import (
    PreprocessStage,
    ModelStage,
    PostprocessStage,
    KVCacheStage,
    LayerSkipStage
)

# Create an inference pipeline with optimizations
pipeline = InferencePipeline(
    stages=[
        PreprocessStage(
            tokenize=True,
            max_length=4096,
            padding="max_length"
        ),
        KVCacheStage(
            enabled=True,
            cache_size=10000,
            compression_level=2
        ),
        ModelStage(
            model_name="meta-llama/Llama-3-8b-instruct",
            torch_dtype="bfloat16",
            device_map="auto"
        ),
        LayerSkipStage(
            enabled=True,
            confidence_threshold=0.85,
            early_exit_layers=[16, 24, 32]
        ),
        PostprocessStage(
            detokenize=True,
            skip_special_tokens=True
        )
    ],
    config={
        "batch_timeout_ms": 50,
        "max_batch_size": 32,
        "enable_streaming": True
    }
)

# Execute pipeline
result = pipeline.execute(
    inputs=["Explain quantum computing"]
)
print(result[0].output_text)
```

### Training Pipeline Setup

```python
from nexus.pipeline import TrainingPipeline
from nexus.training.stages import (
    DataLoadStage,
    DPOStage,
    GradientAccumulationStage,
    CheckpointStage,
    EvaluationStage
)

training_pipeline = TrainingPipeline(
    stages=[
        DataLoadStage(
            dataset_name="nexus-preference-dataset",
            split="train",
            batch_size=4,
            max_seq_length=4096
        ),
        DPOStage(
            model_name="meta-llama/Llama-3-8b-instruct",
            learning_rate=5e-7,
            beta=0.1,
            max_grad_norm=1.0
        ),
        GradientAccumulationStage(
            steps=4
        ),
        CheckpointStage(
            every_n_steps=500,
            directory="/checkpoints",
            include_optimizer=True
        ),
        EvaluationStage(
            eval_dataset="nexus-preference-dataset",
            eval_steps=1000
        )
    ],
    config={
        "epochs": 3,
        "warmup_steps": 100,
        "logging_steps": 50,
        "save_strategy": "steps"
    }
)

# Run training
training_pipeline.run()
```

## Pipeline Configuration

### Inference Pipeline Configuration

```python
from nexus.pipeline.config import PipelineConfig, BatchConfig, CacheConfig

config = PipelineConfig(
    # Pipeline execution mode
    mode="async",  # "sync", "async", "streaming"
    
    # Batch processing configuration
    batch=BatchConfig(
        max_batch_size=32,
        batch_timeout_ms=50,
        min_batch_size=1,
        dynamic_batching=True,
        priority_queue=True
    ),
    
    # KV cache configuration
    cache=CacheConfig(
        enabled=True,
        cache_backend="redis",  # "memory", "redis", "disk"
        max_cache_size=10000,
        eviction_policy="lru",
        compression=True,
        compression_level=2,
        warmup_steps=100
    ),
    
    # Layer skipping configuration
    layer_skip=LayerSkipConfig(
        enabled=True,
        strategy="confidence",  # "confidence", "threshold", "adaptive"
        confidence_threshold=0.85,
        early_exit_layers=[16, 24, 32],
        fallback_to_full=True
    ),
    
    # Memory optimization
    memory=MemoryConfig(
        enable_gradient_checkpointing=False,
        reduce_memory_footprint=True,
        offload_to_cpu=False,
        mixed_precision=True,
        optimizer="adamw"
    ),
    
    # Streaming configuration
    streaming=StreamingConfig(
        enabled=True,
        chunk_size=32,  # tokens per chunk
        prefetch_batches=2
    )
)
```

### Training Pipeline Configuration

```python
from nexus.pipeline.config import TrainingPipelineConfig, DeepSpeedConfig

training_config = TrainingPipelineConfig(
    # Distributed training configuration
    distributed=DistributedConfig(
        backend="nccl",
        world_size=1,  # Set by launcher
        rank=0,  # Set by launcher
        local_rank=0  # Set by launcher
    ),
    
    # DeepSpeed configuration
    deepspeed=DeepSpeedConfig(
        stage=2,  # ZeRO stage (1, 2, or 3)
        offload_optimizer=True,
        offload_param=True,
        nvme_path="/nvme/offload",
        overlap_communication=True,
        partition_activations=True,
        contiguous_checkpointing=True,
        checkpoint_num_layers=1
    ),
    
    # Mixed precision training
    mixed_precision=MixedPrecisionConfig(
        enabled=True,
        dtype="bfloat16",
        initial_scale=1.0,
        dynamic_scale=True
    ),
    
    # Gradient checkpointing
    gradient_checkpointing=GradientCheckpointingConfig(
        enabled=True,
        checkpoint_every_n_layers=1,
        offload_to_cpu=False
    ),
    
    # Learning rate scheduling
    lr_scheduler=LrSchedulerConfig(
        type="cosine",
        num_warmup_steps=100,
        num_training_steps=10000,
        min_lr_ratio=0.1,
        cycle_ratio=1.0
    ),
    
    # Optimizer configuration
    optimizer=OptimizerConfig(
        type="adamw",
        lr=5e-7,
        weight_decay=0.01,
        betas=[0.9, 0.999],
        eps=1e-8,
        fused=True
    ),
    
    # Logging and monitoring
    logging=LoggingConfig(
        steps=50,
        tensorboard=True,
        wandb=True,
        wandb_project="nexus-training",
        log_dir="/logs"
    )
)
```

## Stage Chaining

### Stage Types

```python
from nexus.stages.base import BaseStage
from nexus.stages.inference import (
    InputStage,
    PreprocessStage,
    TokenizeStage,
    EmbedStage,
    ModelStage,
    GenerateStage,
    DecodeStage,
    PostprocessStage,
    OutputStage
)
from nexus.stages.optimization import (
    KVCacheStage,
    LayerSkipStage,
    QuantizeStage,
    FuseStage,
    CacheStage,
    CompressStage
)
from nexus.stages.training import (
    DataLoadStage,
    ForwardStage,
    ComputeLossStage,
    BackwardStage,
    OptimizerStage,
    LrScheduleStage,
    CheckpointStage,
    EvaluationStage
)
```

### Inference Stage Chain

```
┌─────────────────────────────────────────────────────────────────┐
│              Inference Pipeline Stage Chain                      │
└─────────────────────────────────────────────────────────────────┘

┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐
│  Input   │──▶│Preprocess│──▶│ Tokenize │──▶│  Embed   │──▶│ KV Cache │
│  Stage   │   │  Stage   │   │  Stage   │   │  Stage   │   │  Stage   │
└──────────┘   └──────────┘   └──────────┘   └──────────┘   └──────────┘
                                                                   │
                                                                   ▼
┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐
│  Output  │◀──│Postprocess│◀──│  Decode  │◀──│ Generate │◀──│ Model    │
│  Stage   │   │  Stage   │   │  Stage   │   │  Stage   │   │  Stage   │
└──────────┘   └──────────┘   └──────────┘   └──────────┘   └──────────┘
```

### Training Stage Chain

```
┌─────────────────────────────────────────────────────────────────┐
│              Training Pipeline Stage Chain                       │
└─────────────────────────────────────────────────────────────────┘

┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐
│   Data   │──▶│ Forward  │──▶│   Loss   │──▶│ Backward │
│   Load   │   │  Stage   │   │  Compute │   │  Stage   │
│  Stage   │   │          │   │  Stage   │   │          │
└──────────┘   └──────────┘   └──────────┘   └──────────┘
      │                                     │
      │                                     ▼
      │                              ┌──────────┐   ┌──────────┐
      │                              │Gradient  │──▶│Optimizer │
      │                              │Accumul-  │   │  Stage   │
      │                              │  ation   │   │          │
      │                              └──────────┘   └──────────┘
      │                                                   │
      │                                                   ▼
      │                              ┌──────────┐   ┌──────────┐
      │                              │ LrSched- │──▶│Checkpoint│
      │                              │  uler    │   │  Stage   │
      │                              │  Stage   │   │          │
      │                              └──────────┘   └──────────┘
      │                                                   │
      │                                                   ▼
      │                                             ┌──────────┐
      │                                             │Evaluat-  │
      │                                             │  ion     │
      │                                             │ Stage    │
      │                                             └──────────┘
      │                                                   │
      └───────────────────────────────────────────────────┘
```

### Custom Stage Implementation

```python
from nexus.stages.base import BaseStage, StageOutput
from typing import Dict, Any, Optional
import torch

class CustomPreprocessStage(BaseStage):
    """Custom preprocessing stage for data transformation."""
    
    def __init__(
        self,
        normalize: bool = True,
        max_length: int = 4096,
        custom_transform: Optional[callable] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.normalize = normalize
        self.max_length = max_length
        self.custom_transform = custom_transform
        
    def forward(self, inputs: Dict[str, Any]) -> StageOutput:
        """Execute preprocessing transformation.
        
        Args:
            inputs: Dictionary containing input data
            
        Returns:
            StageOutput with preprocessed data
        """
        # Extract input tensors
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask", None)
        
        # Apply normalization if enabled
        if self.normalize:
            input_ids = self._normalize_input_ids(input_ids)
        
        # Truncate to max length
        if input_ids.size(1) > self.max_length:
            input_ids = input_ids[:, :self.max_length]
            if attention_mask is not None:
                attention_mask = attention_mask[:, :self.max_length]
        
        # Apply custom transform
        if self.custom_transform:
            input_ids = self.custom_transform(input_ids)
        
        # Return processed output
        return StageOutput(
            data={
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "original_length": inputs["input_ids"].size(1)
            },
            metadata={
                "stage": "custom_preprocess",
                "truncated": input_ids.size(1) < inputs["input_ids"].size(1)
            }
        )
    
    def _normalize_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Normalize input IDs to valid range."""
        # Clip to valid token range
        input_ids = torch.clamp(input_ids, min=0, max=32000)
        return input_ids
    
    def get_config(self) -> Dict[str, Any]:
        """Return stage configuration."""
        return {
            "normalize": self.normalize,
            "max_length": self.max_length,
            "custom_transform": self.custom_transform is not None
        }
```

### Parallel Stage Execution

```python
from nexus.pipeline.parallel import ParallelStage, PipelineParallelism

# Configure pipeline parallelism
pipeline_parallel = PipelineParallelism(
    num_stages=4,
    num_micro_batches=8,
    stage_devices={
        0: [0, 1],      # Stages 0-1 on GPU 0-1
        1: [2, 3],      # Stages 2-3 on GPU 2-3
        2: [4, 5],
        3: [6, 7]
    }
)

# Create parallel stage group
parallel_stage = ParallelStage(
    stages=[
        PreprocessStage(),
        TokenizeStage()
    ],
    mode="data_parallel",  # "data_parallel", "tensor_parallel"
    device_placement="auto"
)
```

## Checkpoint Management

### Checkpoint Configuration

```python
from nexus.checkpoint import (
    CheckpointManager,
    CheckpointConfig,
    CheckpointStorage,
    CheckpointStrategy
)

checkpoint_config = CheckpointConfig(
    # Checkpoint strategy
    strategy=CheckpointStrategy.STEPS,
    save_steps=500,
    save_epoch=1,
    
    # Storage configuration
    storage=CheckpointStorage(
        type="distributed",  # "local", "distributed", "cloud"
        path="/checkpoints",
        backend="s3",  # "s3", "gcs", "azure"
        bucket="nexus-checkpoints",
        region="us-east-1"
    ),
    
    # What to include
    include=CheckpointInclude(
        model=True,
        optimizer=True,
        scheduler=True,
        random_state=True,
        epoch=1,
        step=True
    ),
    
    # Checkpoint format
    format="safetensors",  # "safetensors", "torch", "huggingface"
    compression="gz",  # "gz", "zstd", None
    
    # Checkpoint limits
    max_checkpoints=10,
    keep_last_n=3,
    
    # Async saving
    async_save=True,
    save_buffer_size=1024,  # MB
    num_save_workers=4
)
```

### Checkpoint Manager Usage

```python
from nexus.checkpoint import CheckpointManager

manager = CheckpointManager(
    config=checkpoint_config,
    model=model,
    optimizer=optimizer,
    scheduler=scheduler
)

# Save checkpoint
checkpoint_path = manager.save(
    step=1000,
    epoch=2,
    metrics={"loss": 0.5234}
)
print(f"Checkpoint saved: {checkpoint_path}")

# List checkpoints
checkpoints = manager.list_checkpoints()
for cp in checkpoints:
    print(f"Step {cp.step}: {cp.path} (loss={cp.metrics['loss']})")

# Load checkpoint
manager.load("/checkpoints/checkpoint-1000")
model = manager.get_model()
optimizer = manager.get_optimizer()

# Delete old checkpoints
manager.cleanup(keep_last=3)

# Get latest checkpoint
latest = manager.get_latest()
```

### Checkpoint Structure

```
checkpoints/
├── checkpoint-500/
│   ├── model/
│   │   ├── pytorch_model.bin
│   │   ├── config.json
│   │   └── training_args.json
│   ├── optimizer/
│   │   ├── optimizer.pt
│   │   └── random_states_0.pkl
│   ├── scheduler/
│   │   └── scheduler.pt
│   ├── trainer_state.json
│   └── train_results.json
├── checkpoint-1000/
│   ├── model/
│   │   ├── pytorch_model.bin
│   │   ├── config.json
│   │   └── adapter_model.safetensors  # LoRA adapter
│   ├── optimizer/
│   │   └── optimizer.pt
│   ├── scheduler/
│   │   └── scheduler.pt
│   ├── trainer_state.json
│   └── train_results.json
└── latest_checkpoint -> checkpoint-1000
```

### Distributed Checkpointing

```python
from nexus.checkpoint import DistributedCheckpointManager

# For multi-GPU training
dist_manager = DistributedCheckpointManager(
    world_size=8,
    rank=0,
    backend="nccl",
    fsdp=True  # Fully Sharded Data Parallel
)

# Save with sharding
dist_manager.save_sharded(
    model=model,
    optimizer=optimizer,
    path="/checkpoints/sharded",
    shard_size_gb=10
)

# Load sharded checkpoints
dist_manager.load_sharded(
    path="/checkpoints/sharded",
    model=model
)

# Zero-redundancy optimizer checkpoint
dist_manager.save_zero(
    optimizer,
    path="/checkpoints/zero"
)
```

## Error Handling

### Error Types and Recovery

```python
from nexus.exceptions import (
    NexusError,
    PipelineError,
    StageError,
    CheckpointError,
    OOMError,
    TimeoutError,
    ValidationError
)
from nexus.recovery import RecoveryManager, FallbackStrategy

# Configure error handling
error_config = ErrorConfig(
    retry_strategy=RetryStrategy(
        max_retries=3,
        initial_delay=1.0,
        max_delay=60.0,
        exponential_base=2.0,
        retry_on=[OOMError, TimeoutError, ConnectionError]
    ),
    
    fallback_strategy=FallbackStrategy(
        enabled=True,
        on_oom="reduce_batch",
        on_timeout="retry_shorter",
        on_error="log_and_continue"
    ),
    
    recovery=RecoveryConfig(
        enabled=True,
        auto_recovery=True,
        checkpoint_recovery=True,
        max_recovery_attempts=2
    ),
    
    circuit_breaker=CircuitBreakerConfig(
        enabled=True,
        failure_threshold=5,
        recovery_timeout=60.0,
        half_open_requests=3
    )
)
```

### Error Handling in Pipelines

```python
from nexus.pipeline.error_handling import PipelineErrorHandler

handler = PipelineErrorHandler(config=error_config)

# Wrap pipeline execution with error handling
try:
    result = pipeline.execute(inputs)
except PipelineError as e:
    # Get error context
    error_info = handler.get_error_info(e)
    print(f"Stage: {error_info.stage}")
    print(f"Error: {error_info.message}")
    print(f"Recovery: {error_info.recovery_action}")
    
    # Attempt recovery
    if error_info.recoverable:
        recovered = handler.recover(e, pipeline)
        result = recovered.execute(inputs)
    else:
        # Fallback to backup model
        result = handler.fallback(e, backup_pipeline)
        
except OOMError as e:
    # Handle out of memory
    reduced = pipeline.with_config(
        max_batch_size=1,
        enable_layer_skipping=True
    )
    result = reduced.execute(inputs)
    
except TimeoutError as e:
    # Handle timeout
    shorter_pipeline = pipeline.with_config(
        max_tokens=min(current_max, 256)
    )
    result = shorter_pipeline.execute(inputs)
```

### Validation and Sanitization

```python
from nexus.validation import InputValidator, OutputValidator

# Configure input validation
input_validator = InputValidator(
    max_input_length=32768,
    max_batch_size=64,
    allowed_content_types=["text", "image", "audio"],
    sanitize_inputs=True,
    filter_prompt_injection=True,
    max_requests_per_minute=1000
)

# Configure output validation
output_validator = OutputValidator(
    max_output_length=4096,
    filter_sensitive=True,
    content_moderation=True,
    validate_format=True
)

# Apply validation to pipeline
pipeline = InferencePipeline(
    stages=[
        ValidationStage(validator=input_validator),
        ModelStage(...),
        ValidationStage(validator=output_validator)
    ]
)
```

### Logging and Monitoring

```python
from nexus.monitoring import PipelineMonitor, AlertManager

monitor = PipelineMonitor(
    metrics_backend="prometheus",
    log_level="INFO",
    track_performance=True,
    track_memory=True,
    track_errors=True
)

# Configure alerts
alerts = AlertManager(
    rules=[
        AlertRule(
            name="high_error_rate",
            condition="error_rate > 0.05",
            severity="warning",
            notify=["slack", "email"]
        ),
        AlertRule(
            name="high_latency",
            condition="p99_latency > 5000",
            severity="warning",
            notify=["pagerduty"]
        ),
        AlertRule(
            name="oom_errors",
            condition="oom_count > 10",
            severity="critical",
            notify=["pagerduty", "slack"]
        )
    ]
)

# Monitor pipeline execution
with monitor.track(pipeline):
    result = pipeline.execute(inputs)
    
# Get metrics
metrics = monitor.get_metrics()
print(f"Success rate: {metrics.success_rate}")
print(f"P95 latency: {metrics.p95_latency_ms}")
print(f"Error breakdown: {metrics.error_breakdown}")
```

## Optimization Stages

### KV Cache Optimization

```python
from nexus.stages.optimization import KVCacheStage

kv_cache_stage = KVCacheStage(
    enabled=True,
    
    # Cache backend
    backend="redis",  # "memory", "redis", "disk"
    
    # Cache size management
    max_cache_entries=10000,
    max_cache_size_gb=50,
    
    # Cache key configuration
    cache_by=["inputs", "model", "max_tokens"],
    ignore_cache_params=["temperature", "seed"],
    
    # Cache optimization
    compression=True,
    compression_level=2,
    quantization="fp8",
    
    # Cache warming
    warmup_on_start=True,
    warmup_samples=100,
    
    # Cache eviction
    eviction_policy="lru",  # "lru", "lfu", "fifo"
    ttl_seconds=3600,
    
    # Hit rate tracking
    track_hit_rate=True,
    reset_stats_interval=3600
)
```

### Layer Skipping

```python
from nexus.stages.optimization import LayerSkipStage

layer_skip_stage = LayerSkipStage(
    enabled=True,
    
    # Skipping strategy
    strategy="confidence",  # "confidence", "threshold", "adaptive"
    
    # Confidence-based skipping
    confidence_threshold=0.85,
    confidence_metric="entropy",  # "entropy", "probability", "variance",
    
    # Early exit layers
    exit_layers=[16, 24, 32],
    use_last_n_layers=8,
    
    # Fallback configuration
    fallback_to_full=True,
    fallback_threshold=0.7,
    
    # Adaptive skipping
    adaptive=AdaptiveConfig(
        enabled=True,
        learning_rate=0.001,
        update_interval=100,
        target_skip_ratio=0.3
    ),
    
    # State management
    save_state=True,
    state_path="/state/layer_skip"
)
```

### Quantization Stage

```python
from nexus.stages.optimization import QuantizeStage

quantize_stage = QuantizeStage(
    enabled=True,
    
    # Quantization method
    method="awq",  # "awq", "gptq", "bitsandbytes", "fp8", "int8"
    
    # Model quantization
    quantize_model=True,
    quantization_bits=4,
    group_size=128,
    damp_percent=0.01,
    
    # KV cache quantization
    quantize_kv_cache=True,
    kv_cache_bits=8,
    kv_cache_group_size=64,
    
    # Activation quantization
    quantize_activations=True,
    activation_bits=8,
    
    # Calibration
    calibration_dataset="wikitext",
    calibration_samples=128,
    calibration_method="minmax",  # "minmax", "percentile", "mse"
    
    # Optimization
    fused_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    exllama_config=None  # For EXL2 compatibility
)
```

### Layer Fusion Stage

```python
from nexus.stages.optimization import LayerFusionStage

fusion_stage = LayerFusionStage(
    enabled=True,
    
    # Fusion strategy
    strategy="auto",  # "auto", "manual", "aggressive"
    
    # Fuse operations
    fuse_gelu=True,
    fuse_bias_add=True,
    fuse_layer_norm=True,
    fuse_qkv=True,
    fuse_attention=True,
    fuse_mlp=True,
    fuse_embed=True,
    
    # Performance tuning
    use_cuda_graph=True,
    kernel_optimization=True,
    memory_format="channels_last",
    
    # Debugging
    export_fusion_graph=False,
    fusion_log_path="/logs/fusion"
)
```

## Examples

### Complete Inference Pipeline

```python
from nexus.pipeline import InferencePipeline
from nexus.stages import (
    InputStage,
    PreprocessStage,
    TokenizeStage,
    KVCacheStage,
    ModelStage,
    LayerSkipStage,
    GenerateStage,
    DecodeStage,
    PostprocessStage,
    OutputStage
)

pipeline = InferencePipeline(
    stages=[
        InputStage(
            max_batch_size=32,
            priority_queue=True
        ),
        PreprocessStage(
            normalize=True,
            max_length=4096
        ),
        TokenizeStage(
            padding="max_length",
            truncation=True,
            max_length=4096
        ),
        KVCacheStage(
            enabled=True,
            backend="redis",
            max_cache_entries=10000,
            compression=True
        ),
        ModelStage(
            model_name="meta-llama/Llama-3-70b-instruct",
            torch_dtype="bfloat16",
            device_map="auto",
            load_in_8bit=False,
            attn_implementation="flash_attention_2"
        ),
        LayerSkipStage(
            enabled=True,
            confidence_threshold=0.85,
            strategy="adaptive"
        ),
        GenerateStage(
            max_new_tokens=2048,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1
        ),
        DecodeStage(
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        ),
        PostprocessStage(
            remove_trailing_whitespace=True
        ),
        OutputStage(
            format="json",
            stream=False
        )
    ],
    config={
        "batch_timeout_ms": 50,
        "enable_streaming": False,
        "memory_optimization": True
    }
)

# Execute
results = pipeline.execute([
    "Explain quantum computing",
    "What is machine learning?",
    "Describe the solar system"
])
```

### Complete Training Pipeline

```python
from nexus.pipeline import TrainingPipeline
from nexus.training.stages import (
    DataLoadStage,
    DPOStage,
    GradientCheckpointStage,
    GradientAccumulationStage,
    OptimizerStage,
    LrScheduleStage,
    CheckpointStage,
    EvaluationStage,
    LoggingStage
)

pipeline = TrainingPipeline(
    stages=[
        DataLoadStage(
            dataset="nexus-preference-dataset",
            split="train",
            batch_size=4,
            max_seq_length=4096,
            shuffle=True,
            num_workers=4
        ),
        DPOStage(
            model_name="meta-llama/Llama-3-8b-instruct",
            beta=0.1,
            loss_type="sigmoid",  # "sigmoid", "hinge", "ipo"
            max_grad_norm=1.0,
            sigma=0.1
        ),
        GradientCheckpointStage(
            enabled=True,
            checkpoint_every_n_layers=1
        ),
        GradientAccumulationStage(
            steps=4
        ),
        OptimizerStage(
            optimizer="adamw",
            lr=5e-7,
            betas=[0.9, 0.999],
            weight_decay=0.01,
            fused=True
        ),
        LrScheduleStage(
            scheduler="cosine",
            num_warmup_steps=100,
            num_training_steps=10000,
            min_lr_ratio=0.1
        ),
        CheckpointStage(
            every_n_steps=500,
            directory="/checkpoints",
            include_optimizer=True
        ),
        EvaluationStage(
            eval_dataset="nexus-preference-eval",
            eval_steps=1000,
            metrics=["accuracy", "margin"]
        ),
        LoggingStage(
            log_dir="/logs",
            tensorboard=True,
            wandb=True,
            wandb_project="dpo-training"
        )
    ],
    config={
        "epochs": 3,
        "seed": 42,
        "mixed_precision": True,
        "distributed_backend": "nccl"
    }
)

# Run training
pipeline.run()
```

### Custom Pipeline with Monitoring

```python
from nexus.pipeline import CustomPipeline
from nexus.monitoring import PipelineMonitor, MetricsCollector

# Create monitor
monitor = PipelineMonitor(
    metrics_backend="prometheus",
    track_performance=True,
    track_memory=True,
    track_gpu=True
)

# Create custom pipeline
pipeline = CustomPipeline(
    stages=[
        ("input", InputStage(max_batch_size=16)),
        ("preprocess", PreprocessStage(normalize=True)),
        ("tokenize", TokenizeStage(max_length=2048)),
        ("model", ModelStage(model="custom-model")),
        ("output", OutputStage(format="json"))
    ],
    error_handler=PipelineErrorHandler(
        retry_strategy=RetryStrategy(max_retries=3),
        fallback_strategy=FallbackStrategy(enabled=True)
    ),
    monitor=monitor
)

# Execute with monitoring
with monitor.track_pipeline(pipeline, name="custom-inference"):
    results = pipeline.execute(inputs)
    
# Export metrics
monitor.export_metrics("/metrics/nexus")

# Check pipeline health
health = pipeline.health_check()
print(f"Pipeline status: {health.status}")
print(f"GPU memory: {health.gpu_memory_used}GB")
print(f"Queue length: {health.queue_length}")
```

## API Reference

### Pipeline Classes

```python
class BasePipeline(ABC):
    """Base class for all pipelines."""
    
    @abstractmethod
    def execute(self, inputs: Any) -> Any:
        """Execute the pipeline."""
        
    @abstractmethod
    def add_stage(self, stage: BaseStage, position: Optional[int] = None) -> None:
        """Add a stage to the pipeline."""
        
    @abstractmethod
    def remove_stage(self, stage_name: str) -> None:
        """Remove a stage from the pipeline."""
        
    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """Get pipeline configuration."""
        
    @abstractmethod
    def health_check(self) -> PipelineHealth:
        """Check pipeline health status."""

class InferencePipeline(BasePipeline):
    """Pipeline for model inference operations."""
    
    def __init__(
        self,
        stages: List[BaseStage],
        config: Optional[PipelineConfig] = None,
        error_handler: Optional[PipelineErrorHandler] = None,
        monitor: Optional[PipelineMonitor] = None
    ):
        """Initialize inference pipeline.
        
        Args:
            stages: List of pipeline stages
            config: Pipeline configuration
            error_handler: Error handling strategy
            monitor: Monitoring backend
        """
        
    def execute(
        self,
        inputs: List[str],
        stream: bool = False,
        **kwargs
    ) -> List[InferenceResult]:
        """Execute inference pipeline.
        
        Args:
            inputs: List of input prompts
            stream: Enable streaming output
            **kwargs: Additional arguments
            
        Returns:
            List of inference results
        """
        
    def add_optimization(
        self,
        optimization_type: str,
        **config
    ) -> None:
        """Add optimization stage.
        
        Args:
            optimization_type: Type of optimization
            **config: Optimization configuration
        """
        
    def set_batch_config(
        self,
        max_batch_size: int = 32,
        batch_timeout_ms: int = 50
    ) -> None:
        """Configure batch processing.
        
        Args:
            max_batch_size: Maximum batch size
            batch_timeout_ms: Batch timeout in milliseconds
        """

class TrainingPipeline(BasePipeline):
    """Pipeline for model training operations."""
    
    def __init__(
        self,
        stages: List[BaseStage],
        config: TrainingPipelineConfig,
        error_handler: Optional[PipelineErrorHandler] = None
    ):
        """Initialize training pipeline.
        
        Args:
            stages: List of training stages
            config: Training configuration
            error_handler: Error handling strategy
        """
        
    def run(
        self,
        resume_from: Optional[str] = None,
        eval_only: bool = False
    ) -> TrainingResult:
        """Execute training pipeline.
        
        Args:
            resume_from: Checkpoint path to resume from
            eval_only: Run evaluation only, no training
            
        Returns:
            Training result with metrics
        """
        
    def evaluate(self, dataset: str) -> EvaluationResult:
        """Run evaluation on dataset.
        
        Args:
            dataset: Dataset name or path
            
        Returns:
            Evaluation metrics
        """
        
    def get_current_state(self) -> TrainingState:
        """Get current training state.
        
        Returns:
            Current epoch, step, and metrics
        """
```

### Stage Classes

```python
class BaseStage(ABC):
    """Base class for all pipeline stages."""
    
    def __init__(self, name: Optional[str] = None, **kwargs):
        """Initialize stage.
        
        Args:
            name: Stage name (auto-generated if not provided)
            **kwargs: Additional configuration
        """
        
    @abstractmethod
    def forward(self, inputs: Dict[str, Any]) -> StageOutput:
        """Execute stage forward pass.
        
        Args:
            inputs: Stage input dictionary
            
        Returns:
            StageOutput with processed data
        """
        
    def backward(self, gradients: Dict[str, Any]) -> None:
        """Execute stage backward pass (for training stages).
        
        Args:
            gradients: Gradient dictionary
        """
        
    def get_config(self) -> Dict[str, Any]:
        """Get stage configuration."""
        
    def get_input_schema(self) -> Dict[str, type]:
        """Get expected input types."""
        
    def get_output_schema(self) -> Dict[str, type]:
        """Get output types."""
```

### Checkpoint Manager Classes

```python
class CheckpointManager:
    """Manager for training checkpoints."""
    
    def __init__(
        self,
        config: CheckpointConfig,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[_LRScheduler] = None
    ):
        """Initialize checkpoint manager.
        
        Args:
            config: Checkpoint configuration
            model: Model to checkpoint
            optimizer: Optimizer to checkpoint
            scheduler: LR scheduler to checkpoint
        """
        
    def save(
        self,
        step: int,
        epoch: int,
        metrics: Optional[Dict[str, float]] = None,
        path: Optional[str] = None
    ) -> str:
        """Save checkpoint.
        
        Args:
            step: Current training step
            epoch: Current epoch
            metrics: Training metrics
            path: Custom path (auto-generated if not provided)
            
        Returns:
            Checkpoint path
        """
        
    def load(
        self,
        path: str,
        load_optimizer: bool = True,
        load_scheduler: bool = True
    ) -> int:
        """Load checkpoint.
        
        Args:
            path: Checkpoint path
            load_optimizer: Load optimizer state
            load_scheduler: Load scheduler state
            
        Returns:
            Step number loaded
        """
        
    def list_checkpoints(self) -> List[CheckpointInfo]:
        """List all checkpoints.
        
        Returns:
            List of checkpoint information
        """
        
    def get_latest(self) -> Optional[CheckpointInfo]:
        """Get latest checkpoint.
        
        Returns:
            Latest checkpoint info or None
        """
        
    def cleanup(self, keep_last: int = 3) -> None:
        """Remove old checkpoints.
        
        Args:
            keep_last: Number of recent checkpoints to keep
        """
```

## See Also

- **[Architecture Overview](ARCHITECTURE.md)** - System architecture details
- **[API Reference](API_REFERENCE.md)** - Detailed API documentation
- **[Security Documentation](SECURITY.md)** - Authentication and authorization
- **[Deployment Guide](DEPLOYMENT.md)** - Production deployment instructions
- **[Training Methods](TRAINING_METHODS.md)** - Training pipeline details
- **[Configuration Guide](../configs/README.md)** - Configuration options
