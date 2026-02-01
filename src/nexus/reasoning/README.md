# Reasoning Module

The `reasoning` module provides advanced reasoning capabilities for the Nexus ecosystem, implementing Chain-of-Thought (CoT) generation, context extension, and reward functions for reinforcement learning-based reasoning training.

## Overview

This module enables Nexus to perform sophisticated reasoning across multiple domains:

- **Chain-of-Thought Generation**: Create structured thinking traces for training
- **Mathematical Reasoning**: Step-by-step math problem solving
- **Code Reasoning**: Debugging and algorithmic thinking
- **Long Context**: Extended context window support via ring attention
- **RL Training**: Reward functions for GRPO/ORPO training

## Module Structure

```
reasoning/
├── cot_generator.py         # Chain-of-Thought dataset generator
├── context_extension.py     # Context length extension utilities
├── reward_functions.py      # RL reward functions for reasoning
├── ring_attention.py        # Ring attention for long sequences
└── bookmark_indexation.py   # Bookmark-based context indexation
```

## Chain-of-Thought Generator

The [`CoTGenerator`](src/reasoning/cot_generator.py:83) creates reasoning datasets with `<think>...</think>` formatting for training models with advanced reasoning capabilities like DeepSeek-R1 and Claude Thinking.

### Basic Usage

```python
from src.reasoning.cot_generator import CoTGenerator, ReasoningType

# Initialize generator
generator = CoTGenerator()

# Convert problem to CoT format
trace = generator.convert_to_cot(
    problem="If a train travels 120 km in 2 hours, what is its speed?",
    answer="60 km/h",
    reasoning_type=ReasoningType.MATH
)

# Access formatted output
print(trace.thinking)
# Output: Let me break this problem down step by step.
#         ... reasoning steps ...
#         Therefore, the answer is: 60 km/h

# Convert to training format
messages = trace.to_messages()
print(messages)
# [
#   {"role": "user", "content": "If a train travels..."},
#   {"role": "assistant", "content": "<think>\n...\n</think>\n\n60 km/h"}
# ]
```

### Supported Reasoning Types

| Type | Description | Example Use Case |
|------|-------------|------------------|
| `MATH` | Mathematical reasoning | Algebra, arithmetic, word problems |
| `CODE` | Code reasoning | Debugging, algorithm design |
| `LOGIC` | Logical reasoning | Puzzles, deductions |
| `SCIENCE` | Scientific reasoning | Physics, chemistry problems |
| `GENERAL` | General reasoning | Open-ended questions |
| `PLANNING` | Task planning | Step-by-step task decomposition |
| `TOOL_USE` | Tool utilization | API calling, function usage |

### Generate from Dataset

```python
# Convert existing dataset to CoT format
generated = generator.generate_from_dataset(
    dataset_path="data/questions.jsonl",
    output_path="data/cot_dataset.jsonl",
    reasoning_type=ReasoningType.MATH,
    sample_size=1000,
    problem_key="question",
    answer_key="answer"
)

print(f"Generated {generated} CoT samples")
```

### Synthetic Math Generation

```python
# Generate synthetic math problems
generated = generator.generate_synthetic_math(
    output_path="data/synthetic_math.jsonl",
    num_samples=10000
)

# Supports:
# - Arithmetic (addition, subtraction, multiplication)
# - Word problems
# - Simple algebra (solve for x)
```

### Configuration

```python
from src.reasoning.cot_generator import CoTConfig

config = CoTConfig(
    think_start_token="<think>",
    think_end_token="</think>",
    max_thinking_length=2048,
    min_thinking_steps=2,
    max_thinking_steps=10,
    temperature=0.7,
    top_p=0.9,
    output_format="jsonl"
)

generator = CoTGenerator(config=config)
```

## Reward Functions

The [`reward_functions.py`](src/reasoning/reward_functions.py) module provides reward computation for RL training methods (GRPO, ORPO, PPO).

### Usage

```python
from src.reasoning.reward_functions import (
    compute_math_reward,
    compute_code_reward,
    compute_format_reward
)

# Math reward (exact match + partial credit)
reward = compute_math_reward(
    predicted="60 km/h",
    ground_truth="60",
    reasoning_trace="..."
)

# Code reward (execution-based)
reward = compute_code_reward(
    code="def add(a, b): return a + b",
    test_cases=[("add(2, 3)", 5), ("add(-1, 1)", 0)]
)

# Format reward (CoT structure)
reward = compute_format_reward(
    response="<think>...reasoning...</think>\n\nFinal answer",
    required_tags=["<think>", "</think>"]
)
```

## Ring Attention

Ring attention implementation for processing sequences beyond standard context limits.

```python
from src.reasoning.ring_attention import RingAttention

# Initialize ring attention
ring_attn = RingAttention(
    chunk_size=2048,
    overlap_size=256
)

# Process long sequence
output = ring_attn.process(
    hidden_states=long_sequence,
    attention_mask=mask
)
```

## Context Extension

Utilities for extending model context length.

```python
from src.reasoning.context_extension import (
    apply_rope_scaling,
    extend_position_embeddings
)

# Apply RoPE scaling for longer contexts
model = apply_rope_scaling(
    model,
    scaling_factor=4.0,  # 4x context extension
    scaling_type="linear"
)

# Extend position embeddings
model = extend_position_embeddings(
    model,
    new_max_position=128000  # 128k context
)
```

## Bookmark Indexation

Bookmark-based context indexation for efficient retrieval from long documents.

```python
from src.reasoning.bookmark_indexation import BookmarkIndexer

# Initialize indexer
indexer = BookmarkIndexer()

# Index long document
bookmarks = indexer.index_document(
    text=long_document,
    bookmark_interval=1000  # tokens per bookmark
)

# Retrieve relevant section
section = indexer.retrieve_section(
    bookmarks=bookmarks,
    query="specific topic",
    surrounding_context=500  # tokens of context
)
```

## Integration with Training

The reasoning module integrates with training methods:

### GRPO Training

```bash
python src/12_grpo_training.py \
    --reasoning_type math \
    --cot_dataset data/cot_math.jsonl \
    --reward_function math_exact
```

### ORPO Training

```bash
python src/orpo_training.py \
    --reasoning_data data/cot_dataset.jsonl \
    --enable_thinking
```

### SFT with CoT

```bash
python src/10_sft_training.py \
    --cot_data data/cot_dataset.jsonl \
    --thinking_format deepseek
```

## Command-Line Usage

### Generate CoT Dataset

```bash
# From existing dataset
python src/reasoning/cot_generator.py \
    --input data/questions.jsonl \
    --output data/cot_output.jsonl \
    --type math

# Synthetic generation
python src/reasoning/cot_generator.py \
    --synthetic \
    --type math \
    --num-samples 10000 \
    --output data/synthetic_math.jsonl
```

## Testing

Run reasoning module tests:

```bash
# Test CoT generation
python -m pytest tests/unit/test_cot_generator.py -v

# Test reward functions
python -m pytest tests/unit/test_reward_functions.py -v

# Test context extension
python -m pytest tests/unit/test_context_extension.py -v
```

## Performance Considerations

| Feature | Memory | Speed | Best For |
|---------|--------|-------|----------|
| Standard CoT | Low | Fast | General reasoning |
| Synthetic Generation | Medium | Medium | Math training data |
| Ring Attention | Low | Slower | >32k context |
| Full Context | High | Fastest | <32k context |

## Examples

### Example 1: Math Problem with Steps

```python
from src.reasoning.cot_generator import CoTGenerator, ReasoningType

generator = CoTGenerator()
trace = generator.convert_to_cot(
    problem="A rectangle has length 8 and width 5. What is its area?",
    answer="40",
    reasoning_type=ReasoningType.MATH
)

print(trace.to_dict())
```

**Output:**

```json
{
  "messages": [
    {"role": "user", "content": "A rectangle has length 8 and width 5..."},
    {"role": "assistant", "content": "<think>\nLet me break this problem down step by step.\n\nAnalyzing the question:\nGiven information:\n- Number: 8\n- Number: 5\n\nSetting up the equation:\nSolving step by step:\nThe area of a rectangle is length × width = 8 × 5 = 40\n\nTherefore, the answer is: 40\n</think>\n\n40"}
  ],
  "reasoning_type": "math",
  "metadata": {}
}
```

### Example 2: Code Reasoning

```python
trace = generator.convert_to_cot(
    problem="Debug this function: def add(a, b): return a - b",
    answer="Change - to +",
    reasoning_type=ReasoningType.CODE
)
```

## References

- DeepSeek-R1: [DeepSeek AI](https://github.com/deepseek-ai/DeepSeek-R1)
- Chain-of-Thought Paper: [Wei et al. 2022](https://arxiv.org/abs/2201.11903)
- Ring Attention: [Liu et al. 2023](https://arxiv.org/abs/2310.01889)

## License

MIT License - See [LICENSE](../../LICENSE) for details.
