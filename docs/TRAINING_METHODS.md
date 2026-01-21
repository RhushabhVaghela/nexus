# Training Methods Documentation

Comprehensive guide to all training methods available in the Nexus Model framework.

---

## Quick Reference

| Method | Type | Use Case | VRAM | Learning Rate |
|--------|------|----------|------|---------------|
| **SFT** | Full weights | General fine-tuning | High | 2e-5 |
| **LoRA** | PEFT | Memory-efficient | Low | 1e-4 |
| **QLoRA** | PEFT + Quant | Very large models | Very Low | 2e-4 |
| **DoRA** | PEFT | Improved LoRA (2024) | Low | 1e-4 |
| **DPO** | Preference | RLHF alternative | Medium | 5e-7 |
| **GRPO** | Preference | DeepSeek method | Medium | 1e-6 |
| **ORPO** | Preference | Combined SFT+Pref | Medium | 8e-6 |
| **PPO** | RL | Classic RLHF | High | 1e-6 |
| **Distillation** | Transfer | Learn from teacher | Medium | 2e-5 |
| **CPT** | Pre-training | Domain adaptation | High | 5e-6 |

---

## Detailed Method Descriptions

### 1. SFT (Supervised Fine-Tuning)

**What it is:**  
Full-weight supervised fine-tuning where all model parameters are updated during training.

**When to use:**

- You have sufficient GPU memory for full model training
- Maximum quality is required
- Training dataset is large and diverse

**Configuration:**

```python
from src.training_methods import TrainingMethod, get_training_config

config = get_training_config(TrainingMethod.SFT)
# learning_rate: 2e-5
# epochs: 3
```

**Command:**

```bash
./run_universal_pipeline.sh --base-model /path/to/model --training-method=sft --enable-cot
```

---

### 2. LoRA (Low-Rank Adaptation)

**What it is:**  
Parameter-efficient fine-tuning that adds trainable low-rank matrices to attention layers. Only ~1% of parameters are trained.

**When to use:**

- Limited GPU memory (< 24GB for 7B models)
- Need to maintain multiple task-specific adapters
- Quick experimentation with different configurations

**Key parameters:**

- `lora_r`: Rank of decomposition (default: 16)
- `lora_alpha`: Scaling factor (default: 32)
- `lora_target_modules`: Which layers to adapt (`q_proj`, `v_proj`, `k_proj`, `o_proj`)

**Configuration:**

```python
config = get_training_config(TrainingMethod.LORA)
# use_peft: True
# lora_r: 16
# lora_alpha: 32
# learning_rate: 1e-4
```

---

### 3. QLoRA (Quantized LoRA)

**What it is:**  
Combines 4-bit quantization with LoRA for extreme memory efficiency. Enables fine-tuning 70B+ models on consumer GPUs.

**When to use:**

- Very limited GPU memory (< 16GB)
- Training very large models (30B+)
- Willing to trade some quality for memory efficiency

**Key parameters:**

- `quantization_bits`: 4 (default)
- `lora_r`: 64 (higher rank compensates for quantization)
- `lora_alpha`: 16

**Configuration:**

```python
config = get_training_config(TrainingMethod.QLORA)
# use_peft: True
# use_quantization: True
# quantization_bits: 4
# lora_r: 64
# learning_rate: 2e-4
```

---

### 4. DoRA (Weight-Decomposed LoRA)

**What it is:**  
2024 improvement to LoRA that decomposes pre-trained weights into magnitude and direction, applying LoRA only to direction component.

**When to use:**

- Same scenarios as LoRA
- Want improved performance without extra memory cost
- Using latest training techniques

**Configuration:**

```python
config = get_training_config(TrainingMethod.DORA)
# use_peft: True
# learning_rate: 1e-4
```

---

### 5. DPO (Direct Preference Optimization)

**What it is:**  
Simplified RLHF that directly optimizes model on preference pairs without a separate reward model. Uses contrastive loss between chosen/rejected responses.

**When to use:**

- Aligning model to human preferences
- Have paired preference data (chosen vs rejected)
- Don't want complexity of PPO/reward model

**Key parameters:**

- `beta`: Regularization strength (default: 0.1)
- Higher beta = closer to reference model

**Data format:**

```json
{
  "prompt": "What is the capital of France?",
  "chosen": "The capital of France is Paris.",
  "rejected": "France's capital is Lyon."
}
```

**Configuration:**

```python
config = get_training_config(TrainingMethod.DPO)
# use_preference_data: True
# beta: 0.1
# learning_rate: 5e-7
```

---

### 6. GRPO (Group Relative Policy Optimization)

**What it is:**  
DeepSeek's method that optimizes policy using group-level comparisons rather than pairwise. More sample-efficient than DPO.

**When to use:**

- Training reasoning models (like DeepSeek-R1)
- Have multiple responses per prompt to compare
- Want better mathematical/logical reasoning

**Key parameters:**

- `beta`: 0.1 (regularization)

**Configuration:**

```python
config = get_training_config(TrainingMethod.GRPO)
# use_preference_data: True
# beta: 0.1
# learning_rate: 1e-6
```

---

### 7. ORPO (Odds Ratio Preference Optimization)

**What it is:**  
Combines SFT and preference optimization in a single training phase. Uses odds ratio for computing preference scores.

**When to use:**

- Want to do SFT + alignment in one pass
- Limited compute budget
- Preference data available from the start

**Configuration:**

```python
config = get_training_config(TrainingMethod.ORPO)
# use_preference_data: True
# beta: 0.1
# learning_rate: 8e-6
```

---

### 8. PPO (Proximal Policy Optimization / RLHF)

**What it is:**  
Classic reinforcement learning from human feedback. Requires a trained reward model and uses PPO to optimize policy.

**When to use:**

- Maximum control over reward signal
- Have resources for reward model training
- Need fine-grained behavioral control

**Architecture:**

```
User Prompt → Policy Model → Response → Reward Model → Score → PPO Update
```

**Configuration:**

```python
config = get_training_config(TrainingMethod.PPO)
# use_preference_data: True
# learning_rate: 1e-6
```

---

### 9. Distillation (Knowledge Distillation)

**What it is:**  
Transfer knowledge from a larger "teacher" model to a smaller "student" model by training on soft targets (logits) rather than hard labels.

**When to use:**

- Compressing large models for deployment
- Have access to powerful teacher model (GPT-4, Claude)
- Want to replicate capabilities in smaller model

**Key parameters:**

- `temperature`: Softening factor (default: 2.0)
- `distillation_alpha`: Weight for soft vs hard targets (default: 0.5)

**Loss formula:**

```
Loss = α * KL(soft_student, soft_teacher) + (1-α) * CE(student, hard_labels)
```

**Configuration:**

```python
config = get_training_config(TrainingMethod.DISTILLATION)
# use_distillation: True
# temperature: 2.0
# distillation_alpha: 0.5
# learning_rate: 2e-5
```

**Usage:**

```bash
python src/multimodal/distillation.py --distill \
    --distill-teacher gpt-4o \
    --distill-student /path/to/student
```

---

### 10. CPT (Continued Pre-Training)

**What it is:**  
Continue pre-training on domain-specific corpus using next-token prediction. Adapts model to new domains without forgetting general capabilities.

**When to use:**

- Adapting to specialized domain (legal, medical, code)
- Have large unlabeled corpus in target domain
- Want to improve base capabilities before SFT

**Configuration:**

```python
config = get_training_config(TrainingMethod.CPT)
# learning_rate: 5e-6 (lower than SFT)
# epochs: 1 (usually single pass through data)
```

---

## Method Selection Flowchart

```
                    ┌──────────────────┐
                    │ What's your goal?│
                    └────────┬─────────┘
                             │
           ┌─────────────────┼─────────────────┐
           ▼                 ▼                 ▼
    ┌──────────────┐ ┌───────────────┐ ┌──────────────┐
    │ Follow       │ │ Align to      │ │ Domain       │
    │ instructions │ │ preferences   │ │ adaptation   │
    └──────┬───────┘ └───────┬───────┘ └──────┬───────┘
           │                 │                 │
           ▼                 ▼                 ▼
   ┌────────────────────────────────────┐     CPT
   │ Limited GPU memory?                │
   └────────────────────────────────────┘
           │                 │
      Yes  │            No   │
           ▼                 ▼
    ┌──────────────┐ ┌──────────────┐
    │ QLoRA/LoRA   │ │     SFT      │
    └──────────────┘ └──────┬───────┘
                           │
                           ▼
                   ┌──────────────────────┐
                   │ Have preference data?│
                   └──────────┬───────────┘
                         Yes  │
                              ▼
                   ┌───────────────────────┐
                   │ Want simplicity? → DPO│
                   │ DeepSeek style? → GRPO│
                   │ Combined? → ORPO      │
                   │ Max control? → PPO    │
                   └───────────────────────┘
```

---

## Training Method Comparison

| Aspect | SFT | LoRA | DPO | Distillation |
|--------|-----|------|-----|--------------|
| Memory usage | High | Low | Medium | Medium |
| Training speed | Slow | Fast | Medium | Medium |
| Quality ceiling | Highest | Good | Highest | Depends on teacher |
| Data requirement | Labeled | Labeled | Preference pairs | Teacher outputs |
| Use case | General | Efficient | Alignment | Compression |

---

## Shell Script Usage

All training methods are available via `run_universal_pipeline.sh`:

```bash
# SFT (default)
./run_universal_pipeline.sh --base-model /path/to/model --training-method=sft --enable-cot

# LoRA
./run_universal_pipeline.sh --base-model /path/to/model --training-method=lora --enable-reasoning

# QLoRA (for large models)
./run_universal_pipeline.sh --base-model /path/to/model --training-method=qlora --enable-tools

# DPO
./run_universal_pipeline.sh --base-model /path/to/model --training-method=dpo --enable-thinking

# Distillation
./run_universal_pipeline.sh --base-model /path/to/model --training-method=distillation --enable-cot
```

---

## Related Files

| File | Description |
|------|-------------|
| `src/training_methods.py` | Training method configurations |
| `src/stages/base.py` | Base stage with distillation support |
| `src/multimodal/distillation.py` | Distillation engine |
| `run_universal_pipeline.sh` | Main orchestration script |
