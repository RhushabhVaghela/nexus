# RULER Benchmark Guide

Evaluate the **true** context length of LLMs using NVIDIA's RULER benchmark.

---

## Quick Start

```bash
# Run RULER on your model
python -m src.benchmarks.ruler_benchmark \
    --model /path/to/model \
    --lengths 4096,8192,16384,32768 \
    --samples 20

# Save results
python -m src.benchmarks.ruler_benchmark \
    --model /path/to/model \
    --output results/ruler_eval.json
```

---

## What is RULER?

RULER measures **actual** context utilization, not just claimed context window. Many models claiming 128K context only effectively use 16-32K.

### Task Categories

| Category | Tasks | What It Tests |
|----------|-------|---------------|
| **NIAH Retrieval** | Single, Multi-Key, Multi-Value, Multi-Query | Find information in noise |
| **Multi-hop** | Variable Tracing, Chain Following | Follow reasoning chains |
| **Aggregation** | Word Count, Frequent Word | Synthesize info |

---

## Task Descriptions

### 1. Single NIAH

```
Context: [noise] The secret code is X. [noise]
Question: What is the secret code?
Expected: X
```

### 2. Multi-Key NIAH

```
Context: [noise] Code for Paris is A. [noise] Code for London is B. [noise]
Question: What is the code for Paris?
Expected: A (must filter distractor)
```

### 3. Variable Tracing

```
Context: [noise] X = 10 [noise] Y = X + 5 [noise]
Question: What is Y?
Expected: 15 (must trace variables)
```

### 4. Common Word Count

```
Context: Text with APPLE appearing N times
Question: How many times does APPLE appear?
Expected: N
```

---

## Interpreting Results

```
📊 RULER Results

Context  | NIAH | Multi-hop | Aggregation | Overall
---------|------|-----------|-------------|--------
4K       | 98%  | 95%       | 92%         | 95% ✅
8K       | 92%  | 85%       | 80%         | 86% ✅
16K      | 82%  | 65%       | 55%         | 67% ❌
32K      | 65%  | 40%       | 35%         | 47% ❌

EFFECTIVE CONTEXT: 8,192 tokens (>70% threshold)
```

**Interpretation:** Despite 32K claimed context, this model only reliably uses 8K.

---

## API Usage

```python
from src.benchmarks.ruler_benchmark import RULERBenchmark, RULERConfig

config = RULERConfig(
    model_path="/path/to/model",
    context_lengths=[4096, 8192, 16384],
    samples_per_task=50,
    accuracy_threshold=0.7,
)

benchmark = RULERBenchmark(config)
benchmark.setup()
result = benchmark.run()

print(f"Effective context: {result.effective_context:,} tokens")
```

---

## Integration with Bookmark Indexation

Test if your Bookmark Indexation system extends effective context:

```bash
# Without Bookmark
python -m src.benchmarks.ruler_benchmark --model /path/to/model

# With Bookmark (once integrated)
python -m src.benchmarks.ruler_benchmark --model /path/to/model --use-bookmark
```

---

## Your Hardware (RTX 5080 16GB + 32GB RAM)

| Mode | Expected Effective Context |
|------|---------------------------|
| Standard | ~32K tokens |
| With Bookmark Indexation | ~128K-512K tokens |

RULER will help you validate the improvement.
