# Nexus Troubleshooting Runbook

This guide provides solutions to common issues encountered when using Nexus.

## Table of Contents

- [Installation Issues](#installation-issues)
- [CUDA/GPU Issues](#cudagpu-issues)
- [Memory Issues](#memory-issues)
- [Model Loading Issues](#model-loading-issues)
- [Training Issues](#training-issues)
- [Inference Issues](#inference-issues)
- [Performance Issues](#performance-issues)
- [Distributed Training Issues](#distributed-training-issues)
- [Debugging Procedures](#debugging-procedures)

---

## Installation Issues

### Package Conflicts

**Problem:** `pip install` fails with dependency conflicts

**Solution:**

```bash
# Use a fresh virtual environment
conda create -n nexus python=3.10
conda activate nexus
pip install --upgrade pip
pip install -r requirements.txt
```

### CUDA Version Mismatch

**Problem:** `RuntimeError: CUDA error: no kernel image is available`

**Solution:**

```bash
# Check CUDA version
nvidia-smi

# Install matching PyTorch version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### AudioCraft Installation Fails

**Problem:** `audiocraft` requires ffmpeg

**Solution:**

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install ffmpeg

# macOS
brew install ffmpeg

# Then install audiocraft
pip install git+https://github.com/facebookresearch/audiocraft.git
```

---

## CUDA/GPU Issues

### GPU Not Detected

**Problem:** `torch.cuda.is_available()` returns `False`

**Diagnostic:**

```bash
# Check NVIDIA driver
nvidia-smi

# Check PyTorch CUDA
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA version: {torch.version.cuda}')"
```

**Solutions:**

1. **Driver issues:**

   ```bash
   # Reinstall NVIDIA drivers
   sudo apt-get purge nvidia*
   sudo apt-get install nvidia-driver-535
   sudo reboot
   ```

2. **Environment issues:**

   ```bash
   export CUDA_VISIBLE_DEVICES=0
   export CUDA_HOME=/usr/local/cuda
   export PATH=$CUDA_HOME/bin:$PATH
   export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
   ```

### CUDA Out of Memory

**Problem:** `RuntimeError: CUDA out of memory`

**Immediate Fix:**

```bash
# Clear cache
python -c "import torch; torch.cuda.empty_cache()"

# Kill zombie processes
nvidia-smi | grep python
kill -9 <PID>
```

**Configuration Fixes:**

```python
# In your training script
import torch

# Enable gradient checkpointing
model.gradient_checkpointing_enable()

# Reduce batch size
batch_size = 1  # Start small

# Use mixed precision
from torch.cuda.amp import autocast
with autocast():
    outputs = model(inputs)
```

**Environment Variables:**

```bash
# Limit GPU memory fraction
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Enable memory efficient attention
export ATTN_IMPLEMENTATION=flash_attention_2
```

---

## Memory Issues

### System RAM Exhaustion

**Problem:** Process killed by OOM killer

**Diagnostic:**

```bash
# Monitor memory
watch -n 1 free -h

# Check process memory
ps aux --sort=-%mem | head -10
```

**Solutions:**

1. **Reduce data loader workers:**

   ```python
   dataloader = DataLoader(
       dataset,
       batch_size=1,
       num_workers=2,  # Reduce from default
       pin_memory=False
   )
   ```

2. **Use streaming datasets:**

   ```python
   from datasets import load_dataset
   
   dataset = load_dataset(
       "name",
       streaming=True,  # Don't load entire dataset
       split="train"
   )
   ```

3. **Enable swap (temporary):**

   ```bash
   sudo fallocate -l 16G /swapfile
   sudo chmod 600 /swapfile
   sudo mkswap /swapfile
   sudo swapon /swapfile
   ```

### Memory Leaks

**Problem:** Memory usage grows over time

**Diagnostic:**

```python
import tracemalloc

tracemalloc.start()
# ... your code ...
snapshot = tracemalloc.take_snapshot()
top_stats = snapshot.statistics('lineno')

for stat in top_stats[:10]:
    print(stat)
```

**Common Causes & Fixes:**

1. **Not detaching tensors:**

   ```python
   # Wrong
   loss = model(input).loss
   
   # Correct
   loss = model(input).loss.detach()
   ```

2. **Accumulating gradients:**

   ```python
   # Wrong
   for batch in dataloader:
       loss = model(batch)
       loss.backward()  # Accumulates!
   
   # Correct
   optimizer.zero_grad()
   for batch in dataloader:
       loss = model(batch)
       loss.backward()
       optimizer.step()
       optimizer.zero_grad()
   ```

---

## Model Loading Issues

### Model Download Fails

**Problem:** `HTTPError: 403 Forbidden` or connection timeout

**Solutions:**

1. **Use HuggingFace token:**

   ```bash
   huggingface-cli login
   # Or set token
   export HF_TOKEN=your_token_here
   ```

2. **Use mirror/endpoint:**

   ```bash
   export HF_ENDPOINT=https://hf-mirror.com
   ```

3. **Manual download:**

   ```bash
   # Download manually
   git lfs install
   git clone https://huggingface.co/model_name ./local_path
   ```

### Model Format Not Recognized

**Problem:** `ValueError: Unrecognized model architecture`

**Solution:**

```python
from src.omni.loader import OmniModelLoader

# Use universal loader
loader = OmniModelLoader()
model = loader.load(
    model_path,
    trust_remote_code=True,  # For custom architectures
    use_safetensors=True
)
```

### Checkpoint Corruption

**Problem:** `RuntimeError: PytorchStreamReader failed reading zip archive`

**Solution:**

```bash
# Verify file integrity
sha256sum model.safetensors

# Re-download if corrupted
rm -rf ~/.cache/huggingface/hub/models--model-name

# Load with relaxed constraints
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(
    path,
    local_files_only=True,
    low_cpu_mem_usage=True
)
```

---

## Training Issues

### Loss Not Decreasing

**Problem:** Training loss stays constant or increases

**Diagnostic Checklist:**

- [ ] Learning rate too high/low?
- [ ] Data preprocessing correct?
- [ ] Labels aligned with inputs?
- [ ] Gradient flow working?

**Debugging:**

```python
# Check gradients
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: {param.grad.norm()}")

# Verify data
batch = next(iter(dataloader))
print(batch.keys())
print(batch['input_ids'].shape)
print(batch['labels'].shape)
```

**Solutions:**

1. **Adjust learning rate:**

   ```python
   from transformers import get_linear_schedule_with_warmup
   
   optimizer = AdamW(model.parameters(), lr=1e-5)  # Lower LR
   scheduler = get_linear_schedule_with_warmup(
       optimizer,
       num_warmup_steps=100,
       num_training_steps=total_steps
   )
   ```

2. **Check label masking:**

   ```python
   # Ensure -100 is used for ignored positions
   labels = batch['input_ids'].clone()
   labels[labels == tokenizer.pad_token_id] = -100
   ```

### NaN Loss

**Problem:** Loss becomes NaN during training

**Causes & Solutions:**

1. **Learning rate too high:**

   ```python
   # Use gradient clipping
   torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
   ```

2. **Mixed precision overflow:**

   ```python
   # Use full precision for loss
   from torch.cuda.amp import autocast
   
   with autocast():
       outputs = model(**inputs)
       loss = outputs.loss.float()  # Cast to float32
   ```

3. **Bad data points:**

   ```python
   # Add NaN checking
   if torch.isnan(loss):
       print(f"NaN detected at step {step}")
       print(f"Inputs: {inputs}")
       continue  # Skip batch
   ```

### Distributed Training Hangs

**Problem:** Multi-GPU training freezes

**Diagnostic:**

```bash
# Check NCCL
NCCL_DEBUG=INFO python -m torch.distributed.launch ...

# Check network
nccl-tests/build/all_reduce_perf -b 8 -e 128M -f 2 -g 1
```

**Solutions:**

1. **Set proper network:**

   ```bash
   export NCCL_SOCKET_IFNAME=eth0
   export NCCL_IB_DISABLE=1
   ```

2. **Increase timeout:**

   ```python
   import torch.distributed as dist
   dist.init_process_group(
       backend='nccl',
       timeout=timedelta(hours=1)
   )
   ```

---

## Inference Issues

### Slow Generation

**Problem:** Text generation is too slow

**Optimizations:**

```python
# Use KV cache
model.generation_config.use_cache = True

# Batch inputs
inputs = tokenizer(prompts, padding=True, return_tensors="pt")

# Optimize generation
outputs = model.generate(
    **inputs,
    max_new_tokens=100,
    do_sample=True,
    temperature=0.7,
    use_cache=True,
    pad_token_id=tokenizer.eos_token_id
)
```

### Incorrect/Unexpected Outputs

**Problem:** Model produces gibberish or wrong format

**Diagnostic:**

```python
# Check tokenization
encoded = tokenizer(prompt, return_tensors="pt")
decoded = tokenizer.decode(encoded['input_ids'][0])
print(f"Original: {prompt}")
print(f"Decoded: {decoded}")

# Check generation config
print(model.generation_config)
```

**Solutions:**

1. **Adjust temperature/top-p:**

   ```python
   outputs = model.generate(
       **inputs,
       temperature=0.3,  # Lower for deterministic
       top_p=0.9,
       repetition_penalty=1.1
   )
   ```

2. **Check prompt format:**

   ```python
   # Ensure proper chat template
   prompt = tokenizer.apply_chat_template(
       messages,
       tokenize=False,
       add_generation_prompt=True
   )
   ```

---

## Performance Issues

### Slow Data Loading

**Problem:** GPU utilization drops between batches

**Solutions:**

```python
# Increase workers
dataloader = DataLoader(
    dataset,
    batch_size=32,
    num_workers=8,  # Adjust based on CPU cores
    pin_memory=True,  # For GPU
    persistent_workers=True
)

# Prefetch data
from torch.utils.data import prefetch_generator
```

### Low GPU Utilization

**Problem:** GPU usage below 50%

**Diagnostic:**

```bash
# Monitor GPU
watch -n 1 nvidia-smi

# Profile code
nsys profile -o report python train.py
```

**Solutions:**

1. **Increase batch size**
2. **Use gradient accumulation:**

   ```python
   accumulation_steps = 4
   for i, batch in enumerate(dataloader):
       loss = model(batch) / accumulation_steps
       loss.backward()
       
       if (i + 1) % accumulation_steps == 0:
           optimizer.step()
           optimizer.zero_grad()
   ```

3. **Use faster kernels:**

   ```bash
   # Install Flash Attention
   pip install flash-attn --no-build-isolation
   ```

---

## Debugging Procedures

### Enable Verbose Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Transformers logging
from transformers import logging as transformers_logging
transformers_logging.set_verbosity_debug()
```

### Profile Memory Usage

```python
from torch.profiler import profile, ProfilerActivity

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True
) as prof:
    # Your code here
    model(inputs)

print(prof.key_averages().table(sort_by="cuda_memory_usage"))
```

### Generate Debug Report

```python
from src.utils.health import generate_debug_report

report = generate_debug_report()
print(report)
```

### Emergency Debugging

If all else fails:

1. **Isolate the issue:**

   ```bash
   # Run minimal example
   python -c "import torch; print(torch.cuda.is_available())"
   ```

2. **Check versions:**

   ```bash
   python -c "import torch; print(torch.__version__)"
   python -c "import transformers; print(transformers.__version__)"
   nvidia-smi
   ```

3. **Clean environment:**

   ```bash
   pip cache purge
   rm -rf ~/.cache/huggingface
   conda clean --all
   ```

---

## Getting Help

If issues persist:

1. Check [GitHub Issues](https://github.com/yourusername/nexus/issues)
2. Join our [Discord community](https://discord.gg/nexus)
3. Email: <support@nexus.ai>
4. Create a minimal reproducible example

### Report Template

```markdown
**Environment:**
- OS: Ubuntu 22.04
- Python: 3.10.12
- PyTorch: 2.1.0
- CUDA: 12.1
- GPU: RTX 4090 (24GB)

**Problem:**
Clear description of the issue

**Reproduction:**
```python
# Minimal code to reproduce
```

**Error Message:**
Full traceback

**Attempts:**
What you've already tried

```
