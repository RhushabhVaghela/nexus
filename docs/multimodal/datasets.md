# Multimodal Datasets - Complete Reference

> **Updated:** January 2026

---

## 📊 Dataset Summary

| Category | Datasets | Total Samples |
|----------|----------|---------------|
| **Reasoning** | 6 | ~500K |
| **Tool-Calling** | 6 | ~150K |
| **Podcast/Audio** | 4 | ~200K |
| **Image Generation** | 3 | ~220K |
| **Video Generation** | 4 | ~1.1M |
| **Vision/Video Understanding** | 5 | ~300K |

---

## 🧠 Reasoning Datasets

| Dataset | Samples | Quality | Use |
|---------|---------|---------|-----|
| `kaist-ai_CoT-Collection` | ~180K | ⭐⭐⭐ | High-quality CoT |
| `O1-OPEN_OpenO1-SFT-Pro` | ~100K | ⭐⭐⭐ | Deep reasoning |
| `O1-OPEN_OpenO1-SFT-Ultra` | ~100K | ⭐⭐⭐ | Ultra reasoning |
| `dipta007_APIGen-MT-5k-with-think` | ~5K | ⭐⭐ | Tool + thinking |
| `openai_gsm8k` | ~8K | ⭐⭐⭐ | Math reasoning |
| `tatsu-lab_alpaca` | ~52K | ⭐ | Basic instructions |

---

## 🔧 Tool-Calling Datasets

| Dataset | Samples | Quality | Use |
|---------|---------|---------|-----|
| `gorilla-llm_Berkeley-Function-Calling-Leaderboard` | ~5K | ⭐⭐⭐ | Benchmark |
| `gorilla-llm_gorilla-openfunctions-v2` | ~20K | ⭐⭐⭐ | Function calling |
| `Salesforce_xlam-function-calling-60k` | ~60K | ⭐⭐⭐ | Large scale |
| `NousResearch_hermes-function-calling-v1` | ~10K | ⭐⭐ | Hermes format |
| `argilla_apigen-function-calling` | ~15K | ⭐⭐ | API generation |
| `hiyouga_glaive-function-calling-v2-sharegpt` | ~40K | ⭐⭐ | ShareGPT format |

---

## 🎙️ Podcast/Audio Datasets

| Dataset | Samples | Quality | Use |
|---------|---------|---------|-----|
| `olewave_OleSpeech-IV-2025-EN-AR-100` | ~100K | ⭐⭐⭐ | Primary podcast |
| `blitt_SPoRC` | ~50K | ⭐⭐⭐ | Spoken conversations |
| `spawn99_CornellMovieDialogCorpus` | ~220K | ⭐⭐ | Movie dialogue |
| `IVLLab_MultiDialog` | ~30K | ⭐⭐ | Multi-turn dialog |

---

## 🎨 Image Generation Datasets

| Dataset | Samples | Size | Use |
|---------|---------|------|-----|
| `LucasFang_JourneyDB-GoT` | ~120K | ~50GB | ⭐ Midjourney-style, grounding |
| `LucasFang_Laion-Aesthetics-High-Resolution-GoT` | ~60K | ~30GB | ⭐ High-quality aesthetics |
| `LucasFang_OmniEdit-GoT` | ~40K | ~20GB | Image editing prompts |

**Total:** ~220K samples, ~100GB

---

## 🎬 Video Generation Datasets

| Dataset | Samples | Size | Use |
|---------|---------|------|-----|
| `XiangpengYang_VideoCoF-50k` | ~50K | ~40GB | ⭐ Text-to-video |
| `VLM2Vec_MSR-VTT` | ~10K | ~10GB | Video-text alignment |
| `qingy2024_VaTeX` | ~40K | ~30GB | Video captioning |
| `remotion_explainer_dataset` | ~1M | ~15GB | ⭐ Programmatic 3B1B-style |

**Total:** ~1.1M samples, ~95GB

---

## 👁️ Vision/Video Understanding

| Dataset | Samples | Use |
|---------|---------|-----|
| `OpenGVLab_ShareGPT-4o` | ~50K | Vision-text conversations |
| `CASIA-IVA-Lab_valor-32k-annotations` | ~32K | Video annotations |
| `mvp-lab_LLaVA-OneVision-1.5-RL-Data` | ~100K | Vision RL alignment |
| `fullstack__stargate_s04e01_100topkdiverse_text2vid` | ~100 | Text-to-video samples |
| `Mozilla_Common-Voice` | ~20K | Speech diversity |

---

## 📥 Download Commands

```bash
# Image Generation (100GB)
huggingface-cli download LucasFang/JourneyDB-GoT --local-dir /mnt/e/data/datasets/JourneyDB-GoT
huggingface-cli download LucasFang/Laion-Aesthetics-High-Resolution-GoT --local-dir /mnt/e/data/datasets/Laion-GoT
huggingface-cli download LucasFang/OmniEdit-GoT --local-dir /mnt/e/data/datasets/OmniEdit-GoT

# Video Generation (50GB)
huggingface-cli download XiangpengYang/VideoCoF-50k --local-dir /mnt/e/data/datasets/VideoCoF-50k
```

---

## 🗂️ Local Path Convention

All datasets stored in:

```
/mnt/e/data/datasets/<dataset-name>/
```

Configured in: `configs/encoders.yaml`
