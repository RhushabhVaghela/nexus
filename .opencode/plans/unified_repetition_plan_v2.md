# Unified Repetition Implementation Plan (Revised)

This plan implements Prompt Repetition (arXiv:2512.14982) across **ALL** modalities (including Reasoning/CoT) with granular user control.

## 1. Core Infrastructure (`src/utils/repetition.py`)
- **Action**: Create `PromptRepetitionEngine`.
- **Features**: Support `baseline`, `2x`, `verbose`, `3x`.
- **Control**: Add `repetition_factor` (int) mapping (1=baseline, 2=2x/verbose, 3=3x).

## 2. Configuration (`src/stages/base.py`)
- **Action**: Update `StageConfig` to include:
    - `repetition_factor: int = 1`
    - `repetition_style: str = "baseline"`

## 3. Stage Integration (Granular Control)
Update each stage's `main()` to accept `--repetition-factor` and `--repetition-style` args.

| Stage File | Capability | Repetition Type | Logic Location |
| :--- | :--- | :--- | :--- |
| `src/24_multimodal_training.py` | **Omni** | **Embedding** (Visual/Audio) & **Text** | `OmniDataset` & `ModularMultimodalWrapper` |
| `src/stages/stage_reasoning.py` | **Reasoning** | **Text** (Problem) | `_format_reasoning` |
| `src/stages/stage_cot.py` | **CoT** | **Text** (Question) | `_format_cot` |
| `src/stages/stage_thinking.py` | **Thinking** | **Text** (Prompt) | `_format_thinking` |
| `src/stages/stage_podcast.py` | **Podcast** | **Text** (Source/Dialogue) | `_format_sample` |
| `src/stages/stage_vision_qa.py` | **Vision QA** | **Text** (Question) | Training Loop |
| `src/stages/stage_video.py` | **Video Und.** | **Text** (Question) | Training Loop |
| `src/stages/stage_tri_streaming.py` | **Tri-Streaming** | **Text** (User Turn) | Training Loop |
| `src/stages/stage_image_gen.py` | **Image Gen** | **Text** (Caption) | Training Loop |
| `src/stages/stage_video_gen.py` | **Video Gen** | **Text** (Caption) | Training Loop |
| `src/stages/stage_remotion_gen.py` | **Remotion** | **Text** (Instruction) | `tokenize_function` |
| `src/stages/stage_streaming.py` | **Streaming** | **Text** (Prompt) | Training Loop |

## 4. Model Architecture (`src/multimodal/model.py`)
- **Action**: Update `ModularMultimodalWrapper` to support `visual_repetition_factor`.
- **Logic**: If `visual_repetition_factor > 1`, duplicate the visual tokens in the stream.

## 5. Benchmarking (`src/benchmarks/benchmark_repetition.py`)
- **Action**: Create a unified benchmark suite.
- **Tasks**:
    - **Text**: `NameIndex` (Reasoning/Factual).
    - **Vision**: `VisualNameIndex`.
    - **Audio**: `AudioRetrieval`.
- **Control**: Benchmarks will run with `factor=[1, 2, 3]` to generate comparative reports.

## 6. Testing
- **Unit Tests**: `tests/test_repetition_core.py`
- **Integration Tests**: `tests/test_repetition_integration.py`
    - Verify arg parsing works for all stages.
    - Verify `OmniMultimodalLM` accepts repeated inputs.
    - Verify `ReasoningStage` produces repeated text when configured.
