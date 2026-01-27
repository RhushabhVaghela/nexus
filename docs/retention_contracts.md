# Nexus Retention Contracts (Stage 0 & 6)

**Status:** Authoritative  
**Context:** Master Plan Stage 0 (Definition) and Stage 6 (Release Verification).

## Overview
This document defines the "No Retention Loss" contracts for the Nexus ecosystem. Each "Teacher" model integrated into the system has a corresponding capability metric. The "Student" model (Nexus) must maintain a **Retention Ratio > 0.95** (95%) of the Teacher's baseline performance on these specific metrics.

## Teacher Model Contracts

| Model Name | Category | Primary Capability | Retention Metric | Target Benchmark |
| :--- | :--- | :--- | :--- | :--- |
| **AgentCPM-Explore** | Agent (LLM-based) | Long-horizon Planning | **Success Rate (SR)** | GAIA / HotpotQA |
| **google_gemma-scope-2-27b-pt** | Language (General) | General Text Generation | **Perplexity / MMLU Score** | MMLU |
| **stepfun-ai_Step3-VL-10B** | Vision-language | VisualQA / Reasoning | **Accuracy** | MMMU / VQAv2 |
| **microsoft_VibeVoice-ASR** | Audio (ASR) | Long-form Speech-to-Text | **Word Error Rate (WER)** | Librispeech (Long) |
| **nvidia_personaplex-7b-v1** | Audio (Conversational) | Role-play / Turn-taking | **Turn Latency & coherence** | Moshi Benchmark |
| **zai-org/GLM-4.7-Flash** | Language (MoE) | Coding / Complex Logic | **Pass@1** | HumanEval / MBPP |
| **parakeet-tdt-0.6b-v3** | Audio (Multilingual) | Multilingual ASR | **WER (Avg across langs)** | CommonVoice |
| **Qwen_Qwen3-TTS-12Hz-1.7B-CustomVoice** | Audio (TTS) | Voice Cloning | **Speaker Similarity (SIM)** | Internal Validation |
| **Qwen_Qwen3-TTS-12Hz-1.7B-VoiceDesign** | Audio (TTS) | Instruction-based TTS | **Instruction Following %** | Internal Validation |
| **stabilityai_stable-diffusion-3-medium-diffusers** | Image Generation | Text-to-Image Synthesis | **CLIP Score / FID** | COCO |
| **stabilityai_stable-video-diffusion-img2vid-xt-1-1** | Video Generation | Image-to-Video | **FVD (Frechet Video Dist)** | UCF101 |
| **Qwen_Qwen3-TTS-Tokenizer-12Hz** | Audio Tokenizer | Acoustic Compression | **Reconstruction Loss** | MUSHRA / Internal |
| **siglip2-so400m-patch16-512** | Vision Encoder | Image Classification/Retrieval | **Zero-shot Accuracy** | ImageNet-1k |
| **MCG-NJU_videomae-large** | Video Encoder | Video Understanding | **Top-1 Accuracy** | Kinetics-400 |

## Verification Protocol (Stage 6)

1.  **Baseline Establishment:** Run evaluation on the original Teacher model using the specific metric.
2.  **Nexus Evaluation:** Run evaluation on the distilled Nexus Student model (or specific Adapter) using the same test set.
3.  **Calculation:**
    ```
    Retention Ratio = (Nexus Score / Teacher Score)
    ```
    *(Note: For metrics like WER or FID where lower is better, the ratio is inverted: Teacher Score / Nexus Score)*
4.  **Pass Criteria:** Retention Ratio >= **0.95**.
