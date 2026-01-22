# Reasoning Training Guide

Complete guide for training models with advanced reasoning capabilities like DeepSeek-R1, Gemini Thinking, and Claude Opus Thinking.

---

## Quick Start

```bash
# Full pipeline with math reasoning
./run_reasoning_pipeline.sh --base-model /path/to/Qwen2.5-7B --enable-cot

# With 128K context extension
./run_reasoning_pipeline.sh --base-model /path/to/model \
    --enable-context --target-context 131072
```

---

## Training Stages

### Stage 1: CoT Dataset Generation

Generate Chain-of-Thought datasets with `<think>...</think>` formatting:

```python
from src.reasoning import CoTGenerator, ReasoningType

generator = CoTGenerator()

# Convert existing dataset
generator.generate_from_dataset(
    "data/math_problems.jsonl",
    "data/cot_math.jsonl",
    reasoning_type=ReasoningType.MATH
)

# Generate synthetic math data
generator.generate_synthetic_math("data/synthetic_math.jsonl", num_samples=10000)
```

**Reasoning Types:** `MATH`, `CODE`, `LOGIC`, `PLANNING`, `TOOL_USE`, `GENERAL`

---

### Stage 2: Reasoning SFT

Supervised fine-tuning on CoT data:

```bash
python -m src.stages.reasoning_sft \
    --model /path/to/base-model \
    --dataset data/cot_math.jsonl \
    --output checkpoints/reasoning_sft \
    --epochs 3 \
    --lora-r 64
```

**Key Options:**

- `--extend-context` - Enable RoPE/YaRN scaling
- `--target-context 131072` - Target context length
- `--no-lora` - Train full model (more VRAM)

---

### Stage 3: GRPO Training

Group Relative Policy Optimization for emergent reasoning:

```bash
python -m src.stages.reasoning_grpo \
    --model checkpoints/reasoning_sft \
    --dataset data/math_problems.jsonl \
    --output checkpoints/reasoning_grpo \
    --iterations 1000 \
    --group-size 4
```

**How GRPO Works:**

1. Generate multiple responses per problem
2. Score each with reward functions
3. Compute group-relative advantages
4. Update policy to favor higher-reward responses

---

## Reward Functions

The GRPO trainer uses multiple reward functions:

| Reward | Weight | Description |
|--------|--------|-------------|
| **Correctness** | 0.4 | Verify answer accuracy |
| **Format** | 0.2 | Check `<think>` structure |
| **Consistency** | 0.2 | Logical coherence |
| **Length** | 0.1 | Penalize verbosity |
| **Process** | 0.1 | Score intermediate steps |

```python
from src.reasoning import create_reward_function, RewardConfig

config = RewardConfig(correctness_weight=0.5, format_weight=0.3)
reward_fn = create_reward_function("combined", config)

result = reward_fn.compute(
    response="<think>Let me solve...</think>\n42",
    reference="42",
    problem="What is 6 * 7?"
)
print(f"Reward: {result.reward:.2f}")
```

---

## Context Extension

Extend model context using YaRN, RoPE scaling, or NTK:

```python
from src.reasoning import create_context_extender

# Create extender for 128K context
extender = create_context_extender(
    target_length=131072,
    original_length=32768,
    scaling_type="yarn"
)

# Get model loading kwargs
kwargs = extender.get_model_kwargs()
model = AutoModelForCausalLM.from_pretrained(path, **kwargs)
```

**Scaling Types:**

- `linear` - Simple linear scaling
- `dynamic` - Adjusts based on sequence length
- `yarn` - YaRN (recommended for long context)
- `ntk` - NTK-aware interpolation

---

## Data Format

CoT datasets should use this format:

```json
{
  "messages": [
    {"role": "user", "content": "Solve: 23 × 17"},
    {"role": "assistant", "content": "<think>\nLet me break this down...\n23 × 17 = 391\n</think>\n\nThe answer is 391."}
  ],
  "reasoning_type": "math"
}
```

---

## Recommended Datasets

| Dataset | Use Case | Source |
|---------|----------|--------|
| GSM8K | Math word problems | HuggingFace |
| MATH | Competition math | HuggingFace |
| HumanEval | Code reasoning | OpenAI |
| FOLIO | Logical reasoning | HuggingFace |
| UltraFeedback | Preference pairs | HuggingFace |

---

## Hardware Requirements

| Stage | GPU Memory | Time (7B model) |
|-------|------------|-----------------|
| SFT | 24GB+ | ~2 hours |
| GRPO | 40GB+ | ~8 hours |
| Context Extension | +8GB per 2x | - |

---

## Troubleshooting

**OOM during GRPO:**

- Reduce `--group-size` (default: 4)
- Use gradient checkpointing (default: on)
- Reduce `--batch-size`

**Poor reasoning quality:**

- Train longer SFT stage first
- Increase CoT dataset quality
- Tune reward weights

**Context extension issues:**

- Start with smaller target (64K)
- Use `yarn` scaling type
- Enable flash attention
