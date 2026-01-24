# Prompt Repetition (arXiv:2512.14982)

## Overview
This feature implements **Prompt Repetition** based on the paper *"Prompt Repetition Improves Non-Reasoning LLMs"* (Leviathan et al., 2025). The core idea is that repeating the user query or context allows causal language models (which cannot look ahead) to "attend to future tokens" by seeing them a second time in the sequence.

While the paper found neutral results for reasoning tasks (CoT), we have implemented this feature across **ALL modalities** with granular control, allowing for experimentation with "Embedding Repetition" for Vision and Audio.

## Usage

### Global Enable
To enable repetition globally for all stages:
```bash
./run_universal_pipeline.sh --base-model /path/to/model --enable-omni --repetition-factor 2
```

### Granular Control
You can enable repetition for specific capabilities while disabling it for others:
```bash
./run_universal_pipeline.sh \
    --base-model /path/to/model \
    --enable-vision-qa --repetition-vision-qa 2 \
    --enable-reasoning --repetition-reasoning 1
```

### Styles
The feature supports the styles defined in the paper:
*   `baseline`: No repetition (Default)
*   `2x`: Simple concatenation (`QUERY QUERY`)
*   `verbose`: Natural language bridge (`QUERY Let me repeat that: QUERY`)
*   `3x`: Triple repetition (`QUERY ... Let me repeat that one more time: QUERY`)

Set the style via `--repetition-style`:
```bash
./run_universal_pipeline.sh --repetition-factor 2 --repetition-style verbose
```

## Implementation Details

### 1. Core Logic (`src/utils/repetition.py`)
The `PromptRepetitionEngine` class handles string formatting.

### 2. Text Repetition
For text-based stages (Reasoning, Podcast, Tools, etc.), the user prompt is intercepted during dataset formatting and repeated.

### 3. Embedding Repetition (Experimental)
For the Omni model (`src/multimodal/model.py`), we implement **Embedding Repetition**.
*   **Vision**: If `visual_repetition_factor > 1`, the visual tokens (output of the connector) are repeated in the sequence *before* being concatenated with text embeddings.
*   **Audio**: Similar logic for audio tokens.

This simulates the model "re-reading" the image or audio clip.

## Benchmarks
A benchmark suite is available to verify performance impact:
```bash
python src/benchmarks/benchmark_repetition.py --model-path /path/to/model --iterations 10
```
This tests:
*   **Text Retrieval**: NameIndex task.
*   **Visual Retrieval**: VisualNameIndex task.
*   **Latency**: Verifies the "no latency penalty" claim (for prefill-heavy workloads).
