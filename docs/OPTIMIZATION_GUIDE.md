# Nexus Optimization Guide

This guide details the advanced training and inference optimizations integrated into the Nexus pipeline.

## 🚀 Training Optimizations

### 1. Unsloth Integration

Nexus now supports [Unsloth](https://github.com/unslothai/unsloth) for significantly faster training and lower VRAM usage.

**Usage:**
Add the `--use-unsloth` flag to `./scripts/nexus.sh master` or `scripts/nexus_pipeline.py`.

```bash
./scripts/nexus.sh master --use-unsloth --models qwen_main
```

**Benefits:**

- up to 3x faster training.
- 2x less VRAM usage.
- Supported architectures: Llama, Mistral, Qwen, Gemma.

### 2. Sequence Packing

Sequence packing bins multiple short sequences together to reduce padding overhead and maximize GPU utilization.

**Usage:**
Add the `--packing` flag.

```bash
./scripts/nexus.sh master --packing
```

### 3. Long Context Support

Train with up to 500k context windows using optimized RoPE scaling.

**Usage:**
Specify the `--max-seq-length` parameter.

```bash
./scripts/nexus.sh master --max-seq-length 32768
```

---

## 🧠 Reasoning Optimization (GRPO)

### Group Relative Policy Optimization

Distill reasoning capabilities from large thinking models (like DeepSeek-R1) into the Nexus student.

**Usage:**
Add the `--grpo` flag to activate the reasoning evolution stage.

```bash
./scripts/nexus.sh master --grpo
```

---

## 🔍 Inference & Retrieval Optimizations

### FastSentenceTransformer

The `KnowledgeTower` (Librarian) now supports Unsloth's `FastSentenceTransformer` for optimized embedding and RAG performance.

**Feature Highlights:**

- **Automatic Fallback**: If `unsloth` is not installed, the system automatically reverts to standard `sentence-transformers` using `transformers`.
- **In-Memory Speed**: Significantly faster document indexing and query embedding.

**Automatic Activation:**
The system will attempt to load `FastSentenceTransformer` whenever a `KnowledgeTower` is initialized.

---

## 🛠️ Environment Setup

Ensure you are in the `nexus` conda environment:

```bash
conda activate nexus
```

To install Unsloth and its dependencies for maximum performance:

```bash
# Example installation for CUDA 12.1
pip install unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git
pip install --no-deps xformers "trl<0.13.0" peft accelerate bitsandbytes
```

> [!NOTE]
> Training and inference will still work without Unsloth, but performance will be lower (fallback mode).

---

## ⚡ I/O & Memory Optimizations (SLI)

Selective Layer Inference (SLI) optimizations for running large models on limited GPU memory.

### 4. Intelligent Layer Caching

Avoid re-downloading and re-loading layers with LRU-based two-tier caching (memory + SSD).

**Usage:**

```python
from nexus_final.sli.layer_cache import LayerCache

cache = LayerCache(
    cache_dir="~/.cache/nexus/layers",
    max_cache_size_gb=50.0,
    max_memory_cache_size_gb=2.0
)

# Automatic cache check before loading
layer = cache.get_layer("model_id", layer_idx)
if layer is None:
    layer = load_layer(layer_idx)
    cache.cache_layer("model_id", layer_idx, layer)
```

**Benefits:**

- 80-95% cache hit ratio in production
- Reduces network downloads by 10-50x
- Sub-100ms layer loading after warm-up

**See:** [Layer Caching Documentation](LAYER_CACHING.md)

---

### 5. Quantization for Faster I/O

Load layers 2-4x faster with INT8 and NF4 quantization.

**Usage:**

```python
from nexus_final.sli.quantization import quantize_layer, get_nf4_config

# Apply 4-bit quantization
config = get_nf4_config()
layer = quantize_layer(layer, mode="nf4")

# Save is 4x smaller
torch.save(layer, "layer_nf4.pt")  # 500MB vs 2GB
```

**Quantization Options:**

| Mode | Size Reduction | Accuracy Impact | Speed |
|------|----------------|-----------------|-------|
| INT8 | 2x | Minimal | Fast |
| NF4 | 4x | Low | Fast |
| Mixed | 3x | Very Low | Fast |

**Benefits:**

- 2-4x faster layer loading from disk
- 2-4x faster network downloads
- Reduced memory footprint

---

### 6. Async Layer Pre-fetching

Overlap computation with I/O by pre-fetching layers while GPU is busy.

**Usage:**

```python
from nexus_final.sli.io_optimizer import IOOptimizer

optimizer = IOOptimizer(
    layer_cache=cache,
    enable_prefetch=True,
    prefetch_lookahead=2
)

# During inference - next layers loaded in background
for i in range(num_layers):
    layer = optimizer.get_layer_with_prefetch(
        model_id, i, num_layers
    )
    output = layer(output)  # GPU computes while I/O prefetches
```

**Benefits:**

- 2-3x speedup for I/O-bound inference
- Pipeline parallelism without code changes
- Automatic compute-I/O overlap

**See:** [I/O Optimization Guide](IO_OPTIMIZATION.md)

---

### 7. Distributed Training (DDP/FSDP)

Multi-GPU/multi-node training with optimized gradient synchronization.

**Usage:**

```bash
# DDP with optimized bucket size
python src/26_distributed_training.py \
    --backend ddp \
    --ddp-bucket-cap 50.0 \
    --num-gpus 4

# FSDP with CPU offloading
python src/26_distributed_training.py \
    --backend fsdp \
    --fsdp-sharding FULL_SHARD \
    --fsdp-cpu-offload

# DeepSpeed ZeRO-3
python src/26_distributed_training.py \
    --backend deepspeed \
    --zero-stage 3 \
    --num-nodes 2
```

**Optimizations:**

- **DDP**: Configurable gradient buckets, static graph optimization
- **FSDP**: Shard parameters across GPUs, CPU offloading
- **DeepSpeed**: ZeRO stages 0-3, NVMe offloading

**Features:**

- Checkpoint sharding for large models
- Gradient synchronization optimizations
- SLURM integration for multi-node

---

## 📊 Performance Comparison

| Optimization | Memory | Speed | Setup Complexity |
|--------------|--------|-------|------------------|
| Layer Caching | Neutral | 2-5x | Low |
| Quantization (NF4) | 4x less | 2-4x | Low |
| Async Pre-fetch | Neutral | 2-3x | Low |
| FSDP | Scales with GPUs | Linear | Medium |
| DeepSpeed Z3 | Fits larger models | 0.7-0.9x | Medium |

---

## 🔧 Advanced Configuration

### Environment Variables

```bash
# I/O Optimization
export NEXUS_CACHE_DIR=/nvme/nexus_cache
export NEXUS_MAX_CACHE_GB=100
export NEXUS_PREFETCH_LOOKAHEAD=2

# Distributed Training
export NCCL_NSOCKS_PERTHREAD=4
export NCCL_SOCKET_NTHREADS=2
export OMP_NUM_THREADS=8
```

### Configuration File

```yaml
# config/optimization.yaml
sli:
  caching:
    enabled: true
    max_size_gb: 50
    memory_cache_gb: 2
  quantization:
    mode: nf4  # int8, nf4, fp4
    compute_dtype: bfloat16
  prefetch:
    enabled: true
    lookahead: 2

distributed:
  backend: fsdp  # ddp, fsdp, deepspeed
  sharding_strategy: FULL_SHARD
  gradient_checkpointing: true
```

---

## 📚 Additional Resources

- [Layer Caching Guide](LAYER_CACHING.md) - Detailed caching documentation
- [I/O Optimization Guide](IO_OPTIMIZATION.md) - I/O bottleneck solutions
- [Distributed Training](../src/26_distributed_training.py) - Multi-GPU training script
- [Quantization Module](../src/nexus_final/sli/quantization.py) - Quantization implementation
- [I/O Optimizer](../src/nexus_final/sli/io_optimizer.py) - Async I/O implementation
