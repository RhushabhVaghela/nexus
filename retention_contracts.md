# Nexus Retention Contracts
**Single Source of Truth for Quality Gates**

## 1. Global Invariants
- **Hardware Target:** RTX 5080 (16GB VRAM)
- **Minimum Retention:** 0.97 (97%) of Teacher Performance unless specified.

## 2. Capability Contracts

### Text & Reasoning
| Capability | Benchmark | Metric | Retain Fraction |
| :--- | :--- | :--- | :--- |
| **Reasoning** | GSM8K | Accuracy | 0.97 |
| **Knowledge** | MMLU | Accuracy | 0.97 |
| **Code** | HumanEval | Pass@1 | 0.95 |
| **Agent** | GAIA / Custom | Success Rate | 0.97 |

### Vision (Multimodal)
| Capability | Benchmark | Metric | Retain Fraction |
| :--- | :--- | :--- | :--- |
| **VQA** | VQAv2 | Accuracy | 0.97 |
| **Image Gen** | FID / CLIP Score | vs Baseline | 0.95 |

### Audio (Voice)
| Capability | Benchmark | Metric | Retain Fraction |
| :--- | :--- | :--- | :--- |
| **ASR** | Librispeech | WER (Lower is better) | 1.03 (allow 3% degradation) |
| **TTS/Cloning** | Cosine Sim | vs Original | 0.85 |

## 4. Resource & Policy Invariants
- **Activation Anchoring in Protected Subspaces:** (Formerly Gradient Surgery). The primary mechanism for penalizing divergence from teacher activations in identified "Critical Layers."
- **Loss-spike rollback + anchor weight escalation:** The automated safety protocol for handling training instability.
- **MAX_RANK_PER_CAPABILITY:** 
    - **Tier 1 & 2 (Logic/Code/Agent):** Capped at **512**.
    - **Tier 3 (Multimodal/High-Entropy):** Capped at **1024**.
    - This ensures VRAM compatibility for the RTX 5080 (16GB) target.
- **Teacher-Free Inference:** NO teacher weights are permitted in the final `nexus_bundle_v1`.
*
