# Repetition Implementation Plan

## 1. Core Utility (`src/utils/repetition.py`)
- Extract `PromptRepetitionEngine` logic here.
- Support styles: `baseline`, `2x`, `verbose`, `3x`.

## 2. Model Modification (`src/multimodal/model.py`)
- Update `ModularMultimodalWrapper` to accept `visual_repetition_factor` (int, default 1) and `audio_repetition_factor` (int, default 1).
- In `forward`:
  - If factor > 1, repeat the `tokens` tensor along the sequence dimension (dim 1) before concatenation.
  - `tokens = tokens.repeat(1, factor, 1)` (assuming batch, seq, dim).

## 3. Training Integration (`src/24_multimodal_training.py`)
- Update `OmniDataset` to import `PromptRepetitionEngine`.
- Add a chance (configurable) to apply text repetition to the `text` field of multimodal samples.

## 4. Benchmarking (`src/benchmarks/benchmark_repetition.py`)
- **Text Task**: NameIndex (50 names, retrieve 25th).
- **Visual Task**:
  - Generate image with PIL containing a list of 10 items.
  - Task: "What is the 5th item in this list?"
  - Metric: Exact Match Accuracy.
- **Audio Task**:
  - Use dummy audio tensor (zeros/noise) or existing wav.
  - Since we can't easily generate semantic audio without a TTS model loaded, we might skip the *content* verification for audio and just verify the *mechanism* (latency/throughput) in the benchmark, or use a pre-existing file if available.
- **Comparison**: Run 50 iterations of Baseline vs 2x vs 3x. Report accuracy and avg latency.

## 5. Tests
- `tests/test_repetition_logic.py`: Verify string outputs.
- `tests/test_multimodal_repetition.py`:
  - Load `OmniMultimodalLM` (wrapper).
  - Pass dummy inputs with `visual_repetition_factor=2`.
  - Assert output logits shape or embedding shape matches expectation (seq_len increases).
