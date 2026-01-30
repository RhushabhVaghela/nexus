# 🧠 Nexus Pipeline v6.1: The Complete Technical Manual

This document consolidates all technical insights, optimizations, and design decisions implemented during the restoration and stabilization of the Nexus Self-Driving Pipeline.

---

## 🚀 1. Universal "Sticky" Persistence

The pipeline now features a fully stateful configuration engine. This ensures that a resumed run is identical to the initial run without needing repetitive CLI flags.

### What is Persisted?

- **Teacher Selection**: List of targeted experts (e.g., `coder`, `translation`).
- **Dataset Mapping**: Recursive paths to locally discovered datasets.
- **Optimization States**: `use_unsloth`, `packing`, and `max_seq_length`.
- **Hyperparameters**: `lr`, `epochs`, `sample_size`, and Router configurations.

> [!NOTE]
> Persistence is managed via `.pipeline_state.json`. Manually editing this file allows for precision "Surgery" on pipeline state.

---

## 🧪 2. Distillation Loss Theory

Nexus uses **Importance-Weighted Activation Anchoring** (protected subspaces).

### Why is the Loss "High"? (e.g., 14.0)

Unlike standard training, Nexus loss is a multi-objective sum:

1. **Output Logit Match**: Ensures the student's *answers* match the teacher.
2. **Hidden State MSE**: Ensures the student's *internal logic* matches.
3. **Critical Layer Surge**: We apply a **10x multiplier** to specific "Soul" layers of the teacher model to prevent logic drift.

### The Scaling Paradox

- **Fewer Samples**: Lower loss, but high risk of "Parroting" (overfitting).
- **More Samples**: Higher loss, but creates a "Self-Driving Specialist" that generalizes to new prompts.

---

## 🛡️ 3. Memory & Stability Guardrails

### Inference RAM Management

Traditional LLM loading greedily allocates massive RAM. We implemented:

- **Meta-Device Initialization**: The model is created as a 0-byte skeleton.
- **Direct-to-GPU Streaming**: Weights are streamed from disk directly to the RTX 5080, bypassing RAM altogether.

### Training Stability

- **Eager Attention**: Fixed `ValueError` by bypassing incompatible SDPA implementations in custom student layers.
- **CUDAGraph Safety**: Disabled `torch.compile(reduce-overhead)` for the distillation loop to prevent runtime crashes during backpropagation.

---

## ⏱️ 4. Process Integrity

### Persistent Timer Fix

A `trap` handler was integrated into `run_nexus_master.sh`.

- **Logic**: Senses `SIGINT`, `SIGTERM`, and `EXIT` signals.
- **Action**: Immediately terminates the background monitoring thread and clears the terminal progress line, preventing log leakage after the script exits.

---

## 📈 5. Scaling Roadmap

To achieve "Production Grade" intelligence, use the following scaling ladder:

| Mode | Command Flag | Benefit |
| :--- | :--- | :--- |
| **Debug** | `--sample_size 50` | Proves the math works in minutes. |
| **Standard** | `--sample_size 500` | Inherits basic stylistic traits. |
| **Production** | `--sample_size 5000+` | Full inheritance of Teacher's Reasoning IQ. |

---

## 🔄 6. Universal Model Loader (OmniModelLoader)

The Nexus pipeline includes a sophisticated universal model loader that supports **50+ model architectures** across diverse model categories, enabling seamless loading of teacher models, encoders, decoders, and specialized models.

### 6.1 Architecture Support Overview

The [`OmniModelLoader`](src/omni/loader.py:76) supports the following model categories:

| Category | Architectures | Examples |
|----------|--------------|----------|
| **Text LLMs** | 130+ | Llama, Qwen, Mistral, Gemma, Phi, DeepSeek, Falcon |
| **Vision Encoders** | 10+ | SigLIP, CLIP, DINOv2, VideoMAE, ViT |
| **ASR Models** | 4+ | Whisper, Speech2Text, Wav2Vec2 |
| **Diffusers** | Full Support | Stable Diffusion, SDXL, Flux |
| **SAE Models** | Detection + Tokenizer Fallback | Gemma Scope, Custom SAEs |
| **Multimodal** | 20+ | LLaVA, Qwen-VL, CogVLM, MiniCPM |

### 6.2 Key Features

#### Automatic Model Category Detection

The loader automatically detects model types without manual configuration:

```python
from src.omni.loader import OmniModelLoader

# Detects model category automatically
category = OmniModelLoader._detect_model_category("/path/to/model")
# Returns: "transformers", "vision_encoder", "asr", "diffusers", or "sae"
```

**Detection Methods:**

- **Transformers**: Config-based detection using `config.json`
- **Vision Encoders**: Architecture matching (SigLIPModel, CLIPVisionModel, etc.)
- **ASR**: WhisperForConditionalGeneration, Speech2TextForConditionalGeneration
- **Diffusers**: `model_index.json` or unet/vae directory structure
- **SAE**: Directory indicators (resid_post, mlp_out, attn_out, transcoder)

#### SAE Model Support with Tokenizer Fallback

Sparse AutoEncoder (SAE) models like Gemma Scope lack tokenizers. The loader automatically detects SAE models and loads tokenizers from their base models:

```python
# SAE detection
is_sae = OmniModelLoader._is_sae_model("/path/to/gemma-scope")
# Returns: True if SAE directories exist without tokenizer files

# Base model extraction
base_model = OmniModelLoader._get_sae_base_model("/path/to/sae")
# Returns: "google/gemma-2b-it" (from SAE config)
```

**SAE Indicators:**

- `resid_post` - Residual post-attention activations
- `mlp_out` - MLP output activations  
- `attn_out` - Attention output activations
- `transcoder` - Transcoder models
- `resid_post_all` - All residual streams

#### Custom Architecture Registration

The loader automatically registers custom model types not in the standard Transformers library:

```python
# Model types with automatic registration:
- "glm4_moe_lite" → Glm4MoeForCausalLM
- "step_robotics" → Step3VL10BForCausalLM
- "qwen3" → Qwen3ForCausalLM
- "agent_cpm" → Qwen3ForCausalLM
```

### 6.3 Model Loading Strategies

The loader implements a cascading strategy system for maximum compatibility:

```python
# Strategy priority order:
1. AutoModelForCausalLM      # Most LLMs
2. AutoModelForVision2Seq    # Vision-language models
3. AutoModelForImageTextToText  # Image-text models
4. AutoModel                  # Fallback for encoders
5. AutoModelForSpeechSeq2Seq  # ASR models
6. AutoModelForSeq2SeqLM     # Seq2seq models
```

### 6.4 Self-Healing Patches

The loader applies runtime patches to handle common model loading issues:

1. **Submodule Resolution**: Patches `get_submodule` for malformed checkpoints
2. **Buffer Registration**: Fixes `persistent` argument in register_buffer lambda
3. **Attribute Setting**: Sanitizes parameter names with dots
4. **Fuzzy Resolver**: Resolves parameter/buffer name mismatches
5. **Quantization Safety**: Handles missing quantization state keys

### 6.5 Safe Loading API

```python
from src.omni.loader import OmniModelLoader, load_omni_model

# Method 1: Using the loader class
loader = OmniModelLoader("/path/to/model")
model, tokenizer = loader.load(mode="thinker_only")

# Method 2: Safe loading with error handling
result = OmniModelLoader.load_model_safe(
    "/path/to/model",
    mode="thinker_only",
    skip_on_error=True  # Returns None instead of raising
)
if result:
    model, tokenizer = result

# Method 3: Convenience function
model, tokenizer = load_omni_model(
    "/path/to/model",
    mode="full",
    skip_on_error=False
)
```

### 6.6 Model Support Verification

Check if a model is supported before loading:

```python
support_info = OmniModelLoader.is_model_supported("/path/to/model")
# Returns:
# {
#     "supported": True,
#     "category": "transformers",
#     "architecture": "LlamaForCausalLM",
#     "model_type": "llama",
#     "has_custom_files": False,
#     "error": None
# }
```

### 6.7 Teacher Registry Coverage

The loader supports all 14 models from the teacher registry:

| Model | Architecture | Category | Special Handling |
|-------|-------------|----------|------------------|
| AgentCPM-Explore | Qwen3ForCausalLM | transformers | Custom registration |
| GLM-4.7-Flash | Glm4MoeForCausalLM | transformers | Model type mapping |
| Step3-VL-10B | Step3VL10BForCausalLM | vision-language | Custom registration |
| Gemma Scope | SAE | sae | Tokenizer fallback |
| Stable Diffusion | DiffusersPipeline | diffusers | Pipeline loading |
| SigLIP | SigLIPModel | vision_encoder | Encoder loading |
| VideoMAE | VideoMAEModel | vision_encoder | Video encoder |
| Whisper/VibeVoice | WhisperForConditionalGeneration | asr | Processor loading |

### 6.8 Error Handling and Graceful Degradation

The loader implements comprehensive error handling:

1. **Missing Config**: Returns informative error message
2. **Malformed JSON**: Graceful handling with error logging
3. **Unsupported Architecture**: Clear error with suggestions
4. **Missing Tokenizer**: Fallback to base model (SAE) or common tokenizers
5. **Loading Failures**: Cascading strategy attempts before failure

### 6.9 Test Coverage

The loader includes comprehensive test coverage:

| Test Type | Count | Coverage |
|-----------|-------|----------|
| **Unit Tests** | 90+ | Detection, categorization, error handling |
| **Integration Tests** | 40+ | Real model loading scenarios |
| **Benchmarks** | 45+ | Performance and regression testing |

**Key Test Categories:**

- Architecture detection for 50+ model types
- SAE model detection and tokenizer fallback
- Model category detection (5 categories)
- Custom architecture registration
- Safe loading with error handling
- Edge cases (malformed configs, missing files)

### 6.10 Adding Support for New Architectures

To add support for a new architecture:

1. **Add to Architecture Lists** (if standard Transformers architecture):

```python
# In src/omni/loader.py
SUPPORTED_ARCHITECTURES = [
    # ... existing ...
    "NewArchitectureForCausalLM",
]
```

1. **Add Model Type Mapping** (if custom model type):

```python
MODEL_TYPE_MAPPINGS = {
    # ... existing ...
    "new_model_type": {
        "architecture": "NewArchitectureForCausalLM",
        "config_class": "NewConfig"
    },
}
```

1. **Add Category Detection** (if new category):

```python
@staticmethod
def _is_new_category(model_path: Path) -> bool:
    # Detection logic
    pass
```

1. **Add Specialized Loader** (if needed):

```python
def _load_new_category(self, model_path: Path, **kwargs):
    # Loading logic
    pass
```

1. **Update Detection Priority**:

```python
def _detect_model_category(self, model_path: Path) -> str:
    if self._is_new_category(model_path):
        return "new_category"
    # ... existing ...
```

---
*Created by Antigravity for Nexus v6.1 Implementation.*
*Updated: 2026-01-30 with Universal Loader documentation*.
