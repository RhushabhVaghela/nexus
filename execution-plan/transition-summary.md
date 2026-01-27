# Nexus Transition Summary: From "Ensemble" to "Master Execution Plan"

## 1. The Core Pivot
We are abandoning the **"Ensemble of Teachers"** architecture. 
*   **Old Way:** A router dynamically selecting between full-sized teacher models (7B, 27B, etc.) at inference time.
*   **New Way:** **"Backbone + Adapters" (Teacher-Free Inference)**. A single modest backbone (400-550M params) with distilled adapters (reasoning, code, voice, etc.).

## 2. Key Architecture Changes
| Feature | Old Plan | Master Execution Plan |
| :--- | :--- | :--- |
| **Inference** | Requires loading Teachers (Multiple GBs) | **Teacher-Free** (Backbone + Adapters only) |
| **Modularity** | Dynamic Routing to Models | **Adapter Selection** via Lightweight Router |
| **Distillation** | Variable / Unclear | **NIWT** (Non-Interactive Weight Transfer) + Activation Anchoring |
| **Storage** | High (Teacher Weights) | **Low** (Compressed PCA profiles, no raw activations) |
| **Hardware** | Datacenter / High-End | **Consumer** (RTX 5080 / 16GB VRAM) |

## 3. Immediate Process Enforcements
1.  **Strict Staging:** We will follow Stages 0-6 sequentially. No jumping ahead.
2.  **Single Source of Truth:** `retention_contracts.md` defines quality gates.
3.  **Constraint:** Teachers come ONLY from `ModelName-Parameters-Category-BestFeature.csv`.

## 4. Current Status
*   **Plan Documents:** Locked (`execution-plan/plan-details.md`).
*   **Teacher List:** Verified (`new-plan-conversation-files/ModelName-Parameters-Category-BestFeature.csv`).
*   **Next Steps:** Executing Stage 0 (Scope Lock) and Stage 1 (Registry Generation).
