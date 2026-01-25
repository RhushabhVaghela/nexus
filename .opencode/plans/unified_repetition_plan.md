# Unified Repetition Implementation Plan

This plan implements Prompt Repetition (arXiv:2512.14982) across all requested modalities (excluding CoT/Reasoning as requested).

## 1. Core Infrastructure (`src/utils/repetition.py`)
- **Action**: Create a centralized `PromptRepetitionEngine` class.
- **Features**: Support `baseline`, `2x`, `verbose`, `3x` styles.
- **Usage**: Importable by all stages and loaders.

## 2. Model Architecture (`src/multimodal/model.py`)
- **Action**: Update `ModularMultimodalWrapper` to support **Embedding Repetition**.
- **New Args**: `visual_repetition_factor` (int), `audio_repetition_factor` (int).
- **Logic**: If factor > 1, repeat the latent tokens (e.g., 64 -> 128) in the `forward` pass before concatenation.

## 3. Stage Integration (Training)
We will inject textual repetition into the dataset formatting logic for the following stages:

| Stage File | Capability | Injection Point |
| :--- | :--- | :--- |
| `src/24_multimodal_training.py` | **Omni** | `OmniDataset._process_sample` (Text field) |
| `src/stages/stage_podcast.py` | **Podcast** | `_format_sample` (Podcast source/dialogue) |
| `src/stages/stage_vision_qa.py` | **Vision QA** | Training loop (Question text) |
| `src/stages/stage_video.py` | **Video Understanding** | Training loop (Question/Prompt) |
| `src/stages/stage_tri_streaming.py` | **Tri-Streaming** | `[USER]` turn text |
| `src/stages/stage_image_gen.py` | **Image Gen** | `caption` field |
| `src/stages/stage_video_gen.py` | **Video Gen** | `caption` field |
| `src/stages/stage_remotion_gen.py` | **Remotion** | `instruction` field |
| `src/stages/stage_streaming.py` | **Streaming** | Training loop (User prompt) |

*Note: CoT, Reasoning, and Thinking stages are explicitly EXCLUDED.*

## 4. Benchmarking Suite (`src/benchmarks/benchmark_repetition.py`)
- **Action**: Create a unified benchmark script.
- **Metric**: Comparison of **Baseline** vs **2x Repetition**.
- **Tasks**:
    1.  **Text**: `NameIndex` (Retrieve Nth item).
    2.  **Vision**: `VisualNameIndex` (Retrieve item from generated image).
    3.  **Audio**: `AudioRetrieval` (Retrieve keyword from audio stream - simulated).
    4.  **Generation**: `PromptAdherence` (Simulated score for Image/Video gen).
- **Output**: JSON report with Accuracy and Latency (tokens/sec).

## 5. Testing
- **Unit Tests** (`tests/test_repetition_core.py`): Verify string formatting for all styles.
- **Integration Tests** (`tests/test_repetition_integration.py`):
    - Load `OmniMultimodalLM` with `visual_repetition_factor=2`.
    - Verify forward pass output shape reflects doubled tokens.
    - Verify `PodcastStage` and others produce repeated text in `dry_run` mode.
