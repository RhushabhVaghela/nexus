# Enhanced Integration Roadmap

## Overview
This roadmap outlines the specific integration tasks required to implement the "Missing Features" identified by the Explore Agent. The focus is on enhancing the `nexus_final` core components (`profiler`, `architect`, `distill`) and integrating multimodal decoders.

**Project Type:** BACKEND

## Success Criteria
- [ ] `profiler.py` successfully measures sensitivity via weight perturbation.
- [ ] `profiler.py` generates and verifies a causal graph of model components.
- [ ] `architect.py` can synthesize "Bridge Layers" to connect disparate modalities.
- [ ] `distill.py` enforces diversity in router selection (avoiding collapse).
- [ ] `distill.py` includes a robust validation loop during distillation.
- [ ] Multimodal decoders are correctly integrated into the pipeline.

## File Structure
Target directory: `src/nexus_final/`
- `profiler.py`: Performance and sensitivity analysis.
- `architect.py`: Neural architecture search and component synthesis.
- `distill.py`: Knowledge distillation and router training.
- `models/`: Storage for multimodal decoder definitions (or `src/multimodal/decoders.py`).

## Task Breakdown

### Wave 1: Profiler Enhancements (Perturbation & Causal)
**Agent:** `backend-specialist` or `research-engineer`

*   **Task 1.1: Implement Weight Perturbation**
    *   **Input:** `src/nexus_final/profiler.py`
    *   **Action:** Add `perturb_weights(model, noise_scale=1e-4)` method.
    *   **Details:** Inject random noise into specific layers to measure output variance/sensitivity.
    *   **Verify:** Unit test where output changes proportionally to noise scale.

*   **Task 1.2: Implement Causal Verification**
    *   **Input:** `src/nexus_final/profiler.py`
    *   **Action:** Add `verify_causal_graph(components)` method.
    *   **Details:** Check if removing a component (zeroing weights) breaks the expected dependency chain.
    *   **Verify:** Test case identifying a broken dependency.

### Wave 2: Architect Enhancements (Bridge Layer)
**Agent:** `backend-specialist`

*   **Task 2.1: Synthesize Bridge Layer**
    *   **Input:** `src/nexus_final/architect.py`
    *   **Action:** Add `synthesize_bridge_layer(input_dim, output_dim, modality_type)` method.
    *   **Details:** Create a projection layer (Linear + Non-linearity) to map embedding spaces.
    *   **Verify:** `test_bridge_layer_shapes` confirms correct tensor dimensions.

### Wave 3: Distill Enhancements (Router & Validation)
**Agent:** `backend-specialist`

*   **Task 3.1: Router Diversity**
    *   **Input:** `src/nexus_final/distill.py`
    *   **Action:** Add `enforce_router_diversity(router_logits)` loss term.
    *   **Details:** Penalize low entropy in router decisions (force usage of all experts).
    *   **Verify:** Distillation run showing >80% expert utilization.

*   **Task 3.2: Validation Loop**
    *   **Input:** `src/nexus_final/distill.py`
    *   **Action:** Implement `validation_loop(teacher, student, val_loader)` method.
    *   **Details:** Periodically evaluate student against teacher on holdout set.
    *   **Verify:** Logs showing validation metrics during training.

### Wave 4: Multimodal Decoder Integration
**Agent:** `backend-specialist`

*   **Task 4.1: Decoder Integration**
    *   **Input:** `src/nexus_final/distill.py` (or `models/`)
    *   **Action:** Connect `src/multimodal/decoders.py` to the distillation loop.
    *   **Details:** Ensure student can output tokens/embeddings compatible with specific decoders (e.g., Image Decoder).
    *   **Verify:** End-to-end pipeline run generating dummy image embeddings.

## Phase X: Final Verification
- [ ] Run `pytest tests/unit/test_niwt_profiler.py` (update to include new profiler tests)
- [ ] Run `pytest tests/unit/test_distillation.py` (create/update for diversity checks)
- [ ] Run `python src/nexus_final/profiler.py --test-perturb` (if script runnable)
