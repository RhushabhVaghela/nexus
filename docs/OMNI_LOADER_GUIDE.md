# OmniModelLoader Developer Guide

A comprehensive guide for using and extending the Universal Model Loader in the Nexus pipeline.

---

## Table of Contents

1. [Introduction](#introduction)
2. [Quick Start](#quick-start)
3. [Architecture Support](#architecture-support)
4. [Model Categories](#model-categories)
5. [Loading Modes](#loading-modes)
6. [API Reference](#api-reference)
7. [Error Handling](#error-handling)
8. [Extending the Loader](#extending-the-loader)
9. [Testing](#testing)
10. [Benchmarking](#benchmarking)

---

## Introduction

The [`OmniModelLoader`](src/omni/loader.py:76) is a universal model loading system designed to handle the diverse ecosystem of transformer models, encoders, decoders, and specialized architectures used in the Nexus pipeline.

### Key Capabilities

- **50+ Supported Architectures**: From Llama to Qwen, SigLIP to Whisper
- **5 Model Categories**: Automatic detection and specialized handling
- **SAE Model Support**: Automatic tokenizer fallback for Sparse AutoEncoders
- **Custom Architecture Registration**: Automatic handling of non-standard models
- **Self-Healing Patches**: Runtime fixes for common loading issues
- **Comprehensive Error Handling**: Graceful degradation with informative messages

---

## Quick Start

### Basic Usage

```python
from src.omni.loader import OmniModelLoader, load_omni_model

# Load a model
loader = OmniModelLoader("/path/to/model")
model, tokenizer = loader.load(mode="thinker_only")

# Or use the convenience function
model, tokenizer = load_omni_model("/path/to/model", mode="full")
```

### Checking Model Support

```python
# Check if a model is supported before loading
support_info = OmniModelLoader.is_model_supported("/path/to/model")

if support_info["supported"]:
    print(f"Category: {support_info['category']}")
    print(f"Architecture: {support_info['architecture']}")
else:
    print(f"Error: {support_info['error']}")
```

### Safe Loading

```python
# Load with error handling
result = OmniModelLoader.load_model_safe(
    "/path/to/model",
    mode="thinker_only",
    skip_on_error=True
)

if result is None:
    print("Model loading failed gracefully")
else:
    model, tokenizer = result
```

---

## Architecture Support

### Supported Architectures (130+)

The loader maintains comprehensive lists of supported architectures:

#### Causal Language Models

```python
SUPPORTED_ARCHITECTURES = [
    # Llama family
    "LlamaForCausalLM", "Llama4ForCausalLM", "LlamaBidirectionalModel",
    
    # Qwen family
    "Qwen2ForCausalLM", "Qwen2MoeForCausalLM", "Qwen3ForCausalLM",
    "Qwen3MoeForCausalLM", "Qwen3NextForCausalLM",
    
    # Mistral family
    "MistralForCausalLM",
    
    # Gemma family
    "GemmaForCausalLM", "Gemma2ForCausalLM", "Gemma3ForCausalLM",
    "Gemma3ForConditionalGeneration",
    
    # Phi family
    "PhiForCausalLM", "Phi3ForCausalLM", "PhiMoEForCausalLM",
    
    # DeepSeek
    "DeepseekForCausalLM",
    
    # And 100+ more...
]
```

#### Vision Encoders (10+)

```python
VISION_ENCODER_ARCHITECTURES = [
    "SigLIPModel", "SigLIPVisionModel",
    "CLIPModel", "CLIPVisionModel",
    "DINOv2Model", "VideoMAEModel",
    "ViTModel", "ViTMAEModel", "ViTMSNModel",
    "DeiTModel", "BeitModel",
    "ConvNextModel", "ConvNextV2Model",
]
```

#### Audio Encoders (6+)

```python
AUDIO_ENCODER_ARCHITECTURES = [
    "Wav2Vec2Model", "Wav2Vec2ForCTC",
    "HubertModel", "WavLMModel",
    "UniSpeechSatModel", "Data2VecAudioModel",
]
```

#### ASR Models (4+)

```python
ASR_ARCHITECTURES = [
    "WhisperForConditionalGeneration", "WhisperModel",
    "Speech2TextForConditionalGeneration",
    "SpeechEncoderDecoderModel",
]
```

### Architecture Aliases

Some architectures have aliases for compatibility:

```python
ARCHITECTURE_ALIASES = {
    "Glm4MoeLiteForCausalLM": "Glm4MoeForCausalLM",
    "Step3VL10BForCausalLM": "AutoModelForCausalLM",
    "Qwen3ForCausalLM": "AutoModelForCausalLM",
    "Qwen3MoeForCausalLM": "AutoModelForCausalLM",
    "Qwen3NextForCausalLM": "AutoModelForCausalLM",
}
```

### Model Type Mappings

Custom model types are automatically mapped to their implementations:

```python
MODEL_TYPE_MAPPINGS = {
    "glm4_moe_lite": {
        "architecture": "Glm4MoeForCausalLM",
        "config_class": "Glm4Config"
    },
    "step_robotics": {
        "architecture": "Step3VL10BForCausalLM",
        "config_class": "Step3VL10BConfig"
    },
    "qwen3": {
        "architecture": "Qwen3ForCausalLM",
        "config_class": "Qwen3Config"
    },
    "agent_cpm": {
        "architecture": "Qwen3ForCausalLM",
        "config_class": "Qwen3Config"
    },
}
```

---

## Model Categories

The loader automatically categorizes models into 5 types:

### 1. Transformers (LLMs)

Standard causal language models and sequence-to-sequence models.

**Detection**: Based on `config.json` architecture field.

**Loading Strategy**: Cascading through AutoModel classes.

```python
# Example models
- LlamaForCausalLM
- Qwen2ForCausalLM
- MistralForCausalLM
```

### 2. Vision Encoders

Image and video encoding models.

**Detection**: Architecture matching against VISION_ENCODER_ARCHITECTURES.

**Loading Strategy**: AutoModel with optional tokenizer.

```python
# Example models
- SigLIPModel
- CLIPVisionModel
- DINOv2Model
```

### 3. ASR Models

Automatic Speech Recognition models.

**Detection**: Architecture matching against ASR_ARCHITECTURES.

**Loading Strategy**: AutoModelForSpeechSeq2Seq with processor.

```python
# Example models
- WhisperForConditionalGeneration
- Speech2TextForConditionalGeneration
```

### 4. Diffusers Models

Image and video generation models (Stable Diffusion, etc.).

**Detection**: Presence of `model_index.json` or `unet`/`vae` directories.

**Loading Strategy**: DiffusionPipeline.from_pretrained().

```python
# Example models
- StableDiffusionPipeline
- StableDiffusionXLPipeline
```

### 5. SAE Models

Sparse AutoEncoder models (Gemma Scope, etc.).

**Detection**: Presence of SAE directories (`resid_post`, `mlp_out`, `attn_out`) without tokenizer files.

**Loading Strategy**: Load base model tokenizer, skip model loading.

```python
# Example models
- Gemma Scope
- Custom SAE models
```

### Category Detection

```python
# Automatic detection
category = OmniModelLoader._detect_model_category("/path/to/model")

# Manual checks
is_diffusers = OmniModelLoader._is_diffusers_model(path)
is_vision = OmniModelLoader._is_vision_encoder(path)
is_asr = OmniModelLoader._is_asr_model(path)
is_sae = OmniModelLoader._is_sae_model(path)
```

---

## Loading Modes

### Mode: `"full"`

Loads the complete model including all components (thinker, talker, encoders).

```python
model, tokenizer = loader.load(mode="full")
```

**Use Case**: Inference with all capabilities enabled.

### Mode: `"thinker_only"`

Loads only the language model component (default).

```python
model, tokenizer = loader.load(mode="thinker_only")
```

**Use Case**: Training, efficient inference without talker overhead.

### Mode: `"talker_only"`

Loads only the audio/text-to-speech component.

```python
model, tokenizer = loader.load(mode="talker_only")
```

**Use Case**: Specialized TTS inference.

---

## API Reference

### Class: `OmniModelLoader`

#### Constructor

```python
loader = OmniModelLoader(model_path: Optional[Union[str, Path]])
```

#### Methods

##### `load(mode="full", **kwargs)`

Load a model with the specified mode.

**Parameters:**

- `mode`: Loading mode ("full", "thinker_only", "talker_only")
- `trust_remote_code`: Allow custom modeling files (default: True)
- `device_map`: Device mapping strategy (default: "auto")
- `torch_dtype`: Data type (default: "auto")
- `load_in_8bit`: Enable 8-bit quantization
- `load_in_4bit`: Enable 4-bit quantization

**Returns:** Tuple of (model, tokenizer)

##### `load_for_training(model_path, freeze_talker=True, **kwargs)`

Load a model optimized for training.

**Parameters:**

- `freeze_talker`: Freeze talker parameters (default: True)
- Additional kwargs passed to `load()`

**Returns:** Tuple of (model, tokenizer)

##### `is_omni_model(model_path)` (static)

Check if a model is an Omni model.

**Returns:** bool

##### `get_model_info(model_path)` (static)

Get information about a model without loading it.

**Returns:** Dict with keys:

- `name`: Model name
- `architecture`: Architecture class name
- `model_type`: Model type from config
- `is_quantized`: Whether model is quantized
- `has_talker`: Whether model has talker config
- `is_supported`: Whether architecture is supported
- `has_custom_files`: Whether custom modeling files exist
- `error`: Error message if applicable

##### `is_model_supported(model_path)` (static)

Check if a model is supported.

**Returns:** Dict with keys:

- `supported`: bool
- `category`: Model category
- `architecture`: Architecture name
- `model_type`: Model type
- `has_custom_files`: bool
- `error`: Error message if not supported

##### `load_model_safe(model_path, mode, skip_on_error, **kwargs)` (static)

Safely load a model with error handling.

**Parameters:**

- `skip_on_error`: If True, returns None on error instead of raising

**Returns:** Tuple of (model, tokenizer) or None

### Function: `load_omni_model()`

Convenience function for loading models.

```python
load_omni_model(
    path: Union[str, Path],
    mode: str = "thinker_only",
    skip_on_error: bool = False,
    **kwargs
) -> Union[Tuple[Any, Any], None]
```

---

## Error Handling

### Common Errors and Solutions

#### "No config.json found"

**Cause**: Model directory is missing configuration file.

**Solution**: Verify model path and ensure it's a valid model directory.

#### "Architecture not recognized"

**Cause**: Model architecture is not in the supported list.

**Solution**:

- Check if model has custom modeling files (`modeling_*.py`)
- Add architecture to `SUPPORTED_ARCHITECTURES`
- Add model type mapping to `MODEL_TYPE_MAPPINGS`

#### "Tokenizer dependency missing"

**Cause**: Tokenizer files are missing or corrupted.

**Solution**:

- For SAE models: Ensure base model is accessible
- For other models: Re-download tokenizer files

#### "Failed to load model"

**Cause**: Model loading failed after trying all strategies.

**Solution**:

- Check model files are complete
- Verify sufficient memory/disk space
- Check compatibility with transformers version

### Graceful Degradation

The loader implements multiple fallback mechanisms:

1. **Strategy Cascading**: Tries multiple AutoModel classes
2. **Tokenizer Fallback**: Falls back to base model for SAEs, or common tokenizers
3. **Safe Loading**: Returns None instead of crashing
4. **Custom Registration**: Auto-registers unknown model types

---

## Extending the Loader

### Adding a New Architecture

1. Add to architecture list:

```python
# In src/omni/loader.py
SUPPORTED_ARCHITECTURES = [
    # ... existing architectures ...
    "NewArchitectureForCausalLM",
]
```

1. Add model type mapping (if needed):

```python
MODEL_TYPE_MAPPINGS = {
    # ... existing mappings ...
    "new_model_type": {
        "architecture": "NewArchitectureForCausalLM",
        "config_class": "NewConfig"
    },
}
```

1. Add tests:

```python
# In tests/unit/test_omni_loader.py
def test_new_architecture_support(self):
    """Test support for NewArchitecture."""
    self.assertIn("NewArchitectureForCausalLM", 
                  OmniModelLoader.SUPPORTED_ARCHITECTURES)
```

### Adding a New Category

1. Create detection method:

```python
@staticmethod
def _is_new_category(model_path: Path) -> bool:
    """Check if model is a new category."""
    # Detection logic
    return (model_path / "indicator_file.json").exists()
```

1. Create loader method:

```python
def _load_new_category(self, model_path: Path, **kwargs):
    """Load a new category model."""
    # Loading logic
    pass
```

1. Update detection priority:

```python
def _detect_model_category(self, model_path: Path) -> str:
    if self._is_new_category(model_path):
        return "new_category"
    # ... existing checks ...
```

1. Update support check:

```python
def is_model_supported(model_path):
    # ... existing checks ...
    elif result["category"] == "new_category":
        result["supported"] = True
```

---

## Testing

### Running Tests

```bash
# Run all loader tests
pytest tests/unit/test_omni_loader.py tests/integration/test_model_loading.py -v

# Run with coverage
pytest tests/unit/test_omni_loader.py --cov=src.omni.loader --cov-report=html

# Run specific test category
pytest tests/unit/test_omni_loader.py::TestSAEModelDetection -v
```

### Test Categories

| Category | Count | Description |
|----------|-------|-------------|
| Unit Tests | 90+ | Isolated component tests |
| Integration Tests | 40+ | End-to-end loading tests |
| Edge Cases | 15+ | Error handling, malformed inputs |

### Writing Tests

```python
# Example unit test
def test_new_feature():
    """Test description."""
    # Setup
    model_path = create_mock_model(config)
    
    # Execute
    result = OmniModelLoader.get_model_info(model_path)
    
    # Assert
    assert result["supported"] is True
    assert result["category"] == "transformers"
```

---

## Benchmarking

### Running Benchmarks

```bash
# Run all benchmarks
python benchmarks/test_omni_loader_benchmark.py --all

# Run specific category
python benchmarks/test_omni_loader_benchmark.py --category detection

# Save results
python benchmarks/test_omni_loader_benchmark.py --output results.json
```

### Benchmark Categories

| Category | Tests | Description |
|----------|-------|-------------|
| Architecture Detection | 7+ | Model info retrieval performance |
| SAE Detection | 6+ | SAE identification speed |
| Tokenizer Loading | 4+ | Tokenizer load times |
| Category Detection | 5+ | Category classification |
| Support Checking | 6+ | Support verification |

### Performance Targets

| Operation | Target | Acceptable |
|-----------|--------|------------|
| Architecture Detection | < 1ms | 0.1-2ms |
| Category Detection | < 0.5ms | 0.05-1ms |
| SAE Detection | < 0.3ms | 0.05-0.5ms |
| Support Check | < 1.5ms | 0.5-3ms |

---

## Architecture Compatibility Matrix

### Teacher Registry Models (14 Models)

| # | Model | Architecture | Category | Loading Strategy | Status |
|---|-------|--------------|----------|-----------------|--------|
| 1 | AgentCPM-Explore | Qwen3ForCausalLM | transformers | AutoModelForCausalLM | ✅ Supported |
| 2 | GLM-4.7-Flash | Glm4MoeForCausalLM | transformers | Custom Registration | ✅ Supported |
| 3 | Step3-VL-10B | Step3VL10BForCausalLM | vision-language | AutoModelForVision2Seq | ✅ Supported |
| 4 | Gemma Scope | SAE | sae | Tokenizer Fallback | ✅ Supported |
| 5 | Stable Diffusion | DiffusersPipeline | diffusers | DiffusionPipeline | ✅ Supported |
| 6 | SigLIP | SigLIPModel | vision_encoder | AutoModel | ✅ Supported |
| 7 | VideoMAE | VideoMAEModel | vision_encoder | AutoModel | ✅ Supported |
| 8 | Whisper/VibeVoice | WhisperForConditionalGeneration | asr | AutoModelForSpeechSeq2Seq | ✅ Supported |
| 9 | Llama Family | LlamaForCausalLM | transformers | AutoModelForCausalLM | ✅ Supported |
| 10 | Qwen2 Family | Qwen2ForCausalLM | transformers | AutoModelForCausalLM | ✅ Supported |
| 11 | Mistral | MistralForCausalLM | transformers | AutoModelForCausalLM | ✅ Supported |
| 12 | Gemma Family | GemmaForCausalLM | transformers | AutoModelForCausalLM | ✅ Supported |
| 13 | Phi Family | Phi3ForCausalLM | transformers | AutoModelForCausalLM | ✅ Supported |
| 14 | DeepSeek | DeepseekForCausalLM | transformers | AutoModelForCausalLM | ✅ Supported |

### Legend

- ✅ **Supported**: Full support with tested loading path
- ⚠️ **Partial**: Supported with limitations
- ❌ **Not Supported**: Not currently supported

---

## Summary

The OmniModelLoader provides a robust, extensible system for loading diverse model architectures in the Nexus pipeline. With comprehensive architecture support, automatic category detection, SAE model handling, and extensive testing, it enables seamless integration of new models and architectures.

### Key Takeaways

1. **Automatic Detection**: No manual configuration needed for supported models
2. **Extensible**: Easy to add new architectures and categories
3. **Robust**: Comprehensive error handling and graceful degradation
4. **Well-Tested**: 130+ tests covering all functionality
5. **Performance**: Benchmarked with defined performance targets

### Next Steps

- Review [Test Coverage](tests/OMNI_LOADER_TEST_COVERAGE.md) for detailed test information
- Check [Benchmark Report](benchmarks/LOADER_BENCHMARK_REPORT.md) for performance data
- See [Technical Manual](NEXUS_V6_TECHNICAL_MANUAL.md) for pipeline integration

---

*Document Version: 1.0*
*Last Updated: 2026-01-30*
*Maintainer: Nexus Documentation Team*
