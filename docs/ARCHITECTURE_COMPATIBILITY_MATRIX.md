# OmniModelLoader Architecture Compatibility Matrix

Complete compatibility matrix for the Nexus Universal Model Loader.

---

## Overview

The [`OmniModelLoader`](src/omni/loader.py:76) supports **50+ model architectures** across **5 model categories**. This matrix documents all supported architectures, their loading strategies, and compatibility status.

---

## Teacher Registry Models (14 Models)

These models from the teacher registry are fully supported:

| # | Model | Architecture | Category | Loading Strategy | Status | Notes |
|---|-------|--------------|----------|-----------------|--------|-------|
| 1 | **AgentCPM-Explore** | Qwen3ForCausalLM | transformers | AutoModelForCausalLM | ✅ Supported | Custom registration for qwen3 type |
| 2 | **GLM-4.7-Flash** | Glm4MoeForCausalLM | transformers | AutoModelForCausalLM | ✅ Supported | Model type: glm4_moe_lite |
| 3 | **Step3-VL-10B** | Step3VL10BForCausalLM | vision-language | AutoModelForVision2Seq | ✅ Supported | Custom registration for step_robotics |
| 4 | **Gemma Scope** | SAE | sae | Tokenizer Fallback | ✅ Supported | Loads tokenizer from base model |
| 5 | **Stable Diffusion** | DiffusersPipeline | diffusers | DiffusionPipeline | ✅ Supported | Full diffusers support |
| 6 | **SigLIP** | SigLIPModel | vision_encoder | AutoModel | ✅ Supported | Vision encoder loading |
| 7 | **VideoMAE** | VideoMAEModel | vision_encoder | AutoModel | ✅ Supported | Video encoder |
| 8 | **Whisper/VibeVoice** | WhisperForConditionalGeneration | asr | AutoModelForSpeechSeq2Seq | ✅ Supported | ASR with processor |
| 9 | **Llama Family** | LlamaForCausalLM | transformers | AutoModelForCausalLM | ✅ Supported | Llama, Llama 2, Llama 3 |
| 10 | **Qwen2 Family** | Qwen2ForCausalLM | transformers | AutoModelForCausalLM | ✅ Supported | Qwen 1.5, 2, 2.5 |
| 11 | **Mistral** | MistralForCausalLM | transformers | AutoModelForCausalLM | ✅ Supported | Mistral 7B, Mixtral |
| 12 | **Gemma Family** | GemmaForCausalLM | transformers | AutoModelForCausalLM | ✅ Supported | Gemma, Gemma 2, Gemma 3 |
| 13 | **Phi Family** | Phi3ForCausalLM | transformers | AutoModelForCausalLM | ✅ Supported | Phi, Phi 2, Phi 3, Phi 4 |
| 14 | **DeepSeek** | DeepseekForCausalLM | transformers | AutoModelForCausalLM | ✅ Supported | DeepSeek models |

---

## Full Architecture Support Lists

### Causal Language Models (130+ architectures)

```
✅ AfmoeForCausalLM
✅ ApertusForCausalLM
✅ ArceeForCausalLM
✅ ArcticForCausalLM
✅ AudioFlamingo3ForConditionalGeneration
✅ BaiChuanForCausalLM
✅ BaichuanForCausalLM
✅ BailingMoeForCausalLM
✅ BailingMoeV2ForCausalLM
✅ BambaForCausalLM
✅ BertForMaskedLM
✅ BertForSequenceClassification
✅ BertModel
✅ BitnetForCausalLM
✅ BloomForCausalLM
✅ BloomModel
✅ CamembertModel
✅ ChameleonForCausalLM
✅ ChameleonForConditionalGeneration
✅ ChatGLMForConditionalGeneration
✅ ChatGLMModel
✅ CodeShellForCausalLM
✅ CogVLMForCausalLM
✅ Cohere2ForCausalLM
✅ CohereForCausalLM
✅ DbrxForCausalLM
✅ DeciLMForCausalLM
✅ DeepseekForCausalLM
✅ DistilBertForMaskedLM
✅ DistilBertForSequenceClassification
✅ DistilBertModel
✅ Dots1ForCausalLM
✅ DreamModel
✅ Ernie4_5ForCausalLM
✅ Ernie4_5_ForCausalLM
✅ Ernie4_5_MoeForCausalLM
✅ Exaone4ForCausalLM
✅ ExaoneForCausalLM
✅ ExaoneMoEForCausalLM
✅ FalconForCausalLM
✅ FalconH1ForCausalLM
✅ FalconMambaForCausalLM
✅ GPT2LMHeadModel
✅ GPTBigCodeForCausalLM
✅ GPTNeoXForCausalLM
✅ GPTRefactForCausalLM
✅ Gemma2ForCausalLM
✅ Gemma3ForCausalLM
✅ Gemma3ForConditionalGeneration
✅ Gemma3TextModel
✅ Gemma3nForCausalLM
✅ Gemma3nForConditionalGeneration
✅ GemmaForCausalLM
✅ Glm4ForCausalLM
✅ Glm4MoeForCausalLM
✅ Glm4MoeLiteForCausalLM
✅ Glm4vForConditionalGeneration
✅ Glm4vMoeForConditionalGeneration
✅ GlmForCausalLM
✅ GlmasrModel
✅ GptOssForCausalLM
✅ GraniteForCausalLM
✅ GraniteMoeForCausalLM
✅ GraniteMoeHybridForCausalLM
✅ GraniteMoeSharedForCausalLM
✅ Grok1ForCausalLM
✅ GrokForCausalLM
✅ GroveMoeForCausalLM
✅ HunYuanDenseV1ForCausalLM
✅ HunYuanMoEV1ForCausalLM
✅ Idefics3ForConditionalGeneration
✅ InternLM2ForCausalLM
✅ InternLM3ForCausalLM
✅ InternVisionModel
✅ JAISLMHeadModel
✅ JambaForCausalLM
✅ JanusForConditionalGeneration
✅ JinaBertForMaskedLM
✅ JinaBertModel
✅ KORMoForCausalLM
✅ KimiVLForConditionalGeneration
✅ LFM2ForCausalLM
✅ LLaDAMoEModel
✅ LLaDAMoEModelLM
✅ LLaDAModelLM
✅ Lfm2AudioForConditionalGeneration
✅ Lfm2ForCausalLM
✅ Lfm2Model
✅ Lfm2MoeForCausalLM
✅ Lfm2VlForConditionalGeneration
✅ LightOnOCRForConditionalGeneration
✅ Llama4ForCausalLM
✅ Llama4ForConditionalGeneration
✅ LlamaBidirectionalModel
✅ LlavaStableLMEpochForCausalLM
✅ MPTForCausalLM
✅ MT5ForConditionalGeneration
✅ MaincoderForCausalLM
✅ Mamba2ForCausalLM
✅ MambaForCausalLM
✅ MambaLMHeadModel
✅ MiMoV2FlashForCausalLM
✅ MiniCPM3ForCausalLM
✅ MiniCPMForCausalLM
✅ MiniMaxM2ForCausalLM
✅ Mistral3ForConditionalGeneration
✅ ModernBertForMaskedLM
✅ ModernBertForSequenceClassification
✅ ModernBertModel
✅ NemotronForCausalLM
✅ NemotronHForCausalLM
✅ NeoBERT
✅ NeoBERTForSequenceClassification
✅ NeoBERTLMHead
✅ NomicBertModel
✅ OLMoForCausalLM
✅ Olmo2ForCausalLM
✅ Olmo3ForCausalLM
✅ OlmoForCausalLM
✅ OlmoeForCausalLM
✅ OpenELMForCausalLM
✅ OrionForCausalLM
✅ PLMForCausalLM
✅ PLaMo2ForCausalLM
✅ PLaMo3ForCausalLM
✅ PanguEmbeddedForCausalLM
✅ Phi3ForCausalLM
✅ PhiForCausalLM
✅ PhiMoEForCausalLM
✅ Plamo2ForCausalLM
✅ Plamo3ForCausalLM
✅ PlamoForCausalLM
✅ QWenLMHeadModel
✅ Qwen2AudioForConditionalGeneration
✅ Qwen2ForCausalLM
✅ Qwen2Model
✅ Qwen2MoeForCausalLM
✅ Qwen2OmniTalkerForConditionalGeneration
✅ Qwen2VLForConditionalGeneration
✅ Qwen2VLModel
✅ Qwen2_5OmniForConditionalGeneration
✅ Qwen2_5OmniModel
✅ Qwen2_5_VLForConditionalGeneration
✅ Qwen3ForCausalLM
✅ Qwen3MoeForCausalLM
✅ Qwen3NextForCausalLM
✅ Qwen3OmniForConditionalGeneration
✅ Qwen3TTSForConditionalGeneration
✅ Qwen3VLForConditionalGeneration
✅ Qwen3VLMoeForConditionalGeneration
✅ RND1
✅ RWForCausalLM
✅ RWKV6Qwen2ForCausalLM
✅ RWKV7ForCausalLM
✅ RobertaForSequenceClassification
✅ RobertaModel
✅ Rwkv6ForCausalLM
✅ Rwkv7ForCausalLM
✅ RwkvHybridForCausalLM
✅ SeedOssForCausalLM
✅ SmallThinkerForCausalLM
✅ SmolLM3ForCausalLM
✅ SmolVLMForConditionalGeneration
✅ SolarOpenForCausalLM
✅ StableLMEpochForCausalLM
✅ StableLmForCausalLM
✅ Starcoder2ForCausalLM
✅ T5EncoderModel
✅ T5ForConditionalGeneration
✅ T5WithLMHeadModel
✅ UMT5ForConditionalGeneration
✅ UMT5Model
✅ UltravoxModel
✅ VoxtralForConditionalGeneration
✅ WavTokenizerDec
✅ XLMRobertaForSequenceClassification
✅ XLMRobertaModel
✅ XverseForCausalLM
✅ YoutuVLForConditionalGeneration
✅ modeling_grove_moe.GroveMoeForCausalLM
```

### Vision Encoder Architectures (10+)

| Architecture | Status | Example Models |
|--------------|--------|----------------|
| SigLIPModel | ✅ | SigLIP-Large, SigLIP-SO400M |
| SigLIPVisionModel | ✅ | SigLIP Vision variants |
| CLIPModel | ✅ | CLIP-Base, CLIP-Large |
| CLIPVisionModel | ✅ | CLIP Vision encoders |
| DINOv2Model | ✅ | DINOv2-Small, DINOv2-Base, DINOv2-Large |
| VideoMAEModel | ✅ | VideoMAE-Base, VideoMAE-Large |
| ViTModel | ✅ | Vision Transformer |
| ViTMAEModel | ✅ | MAE pre-trained ViT |
| ViTMSNModel | ✅ | MSN pre-trained ViT |
| DeiTModel | ✅ | Data-efficient Image Transformer |
| BeitModel | ✅ | BEiT models |
| ConvNextModel | ✅ | ConvNeXt models |
| ConvNextV2Model | ✅ | ConvNeXt V2 models |

### Audio Encoder Architectures (6+)

| Architecture | Status | Example Models |
|--------------|--------|----------------|
| Wav2Vec2Model | ✅ | wav2vec 2.0 |
| Wav2Vec2ForCTC | ✅ | wav2vec 2.0 with CTC head |
| HubertModel | ✅ | HuBERT models |
| WavLMModel | ✅ | WavLM models |
| UniSpeechSatModel | ✅ | UniSpeech-SAT |
| Data2VecAudioModel | ✅ | data2vec-audio |

### ASR Architectures (4+)

| Architecture | Status | Example Models |
|--------------|--------|----------------|
| WhisperForConditionalGeneration | ✅ | Whisper Tiny, Base, Small, Medium, Large |
| WhisperModel | ✅ | Whisper encoder-decoder |
| Speech2TextForConditionalGeneration | ✅ | Speech2Text models |
| SpeechEncoderDecoderModel | ✅ | Generic encoder-decoder ASR |

### Custom Model Type Mappings

These model types are automatically registered when encountered:

| Model Type | Maps To | Config Class | Example Models |
|------------|---------|--------------|----------------|
| glm4_moe_lite | Glm4MoeForCausalLM | Glm4Config | GLM-4.7-Flash |
| step_robotics | Step3VL10BForCausalLM | Step3VL10BConfig | Step3-VL-10B |
| qwen3 | Qwen3ForCausalLM | Qwen3Config | AgentCPM-Explore |
| qwen3_moe | Qwen3MoeForCausalLM | Qwen3MoeConfig | Qwen3 MoE variants |
| agent_cpm | Qwen3ForCausalLM | Qwen3Config | AgentCPM models |

---

## Category Detection Priority

When detecting model categories, the loader uses the following priority:

1. **Diffusers** (Highest) - Checks for `model_index.json` or `unet`/`vae` directories
2. **SAE** - Checks for SAE directories (`resid_post`, `mlp_out`, etc.) without tokenizer
3. **Vision Encoder** - Checks architecture against VISION_ENCODER_ARCHITECTURES
4. **ASR** - Checks architecture against ASR_ARCHITECTURES
5. **Transformers** (Default) - Standard LLM loading

This priority ensures that models with multiple characteristics are handled correctly.

---

## Loading Strategies

The loader implements a cascading strategy system for maximum compatibility:

| Priority | Strategy | Use Case |
|----------|----------|----------|
| 1 | AutoModelForCausalLM | Most text generation models |
| 2 | AutoModelForVision2Seq | Vision-language models |
| 3 | AutoModelForImageTextToText | Image-text models |
| 4 | AutoModel | Generic fallback for encoders |
| 5 | AutoModelForSpeechSeq2Seq | ASR models |
| 6 | AutoModelForSeq2SeqLM | Sequence-to-sequence models |

If a strategy fails, the loader automatically tries the next one until all are exhausted.

---

## Test Coverage by Category

| Category | Unit Tests | Integration Tests | Total |
|----------|------------|-------------------|-------|
| Architecture Detection | 25+ | 10+ | 35+ |
| SAE Detection | 14 | 2 | 16 |
| Tokenizer Loading | 8 | 5 | 13 |
| Category Detection | 17 | 5 | 22 |
| Model Info | 9 | 3 | 12 |
| Custom Registration | 4 | 2 | 6 |
| Error Handling | 6 | 6 | 12 |
| **Total** | **90+** | **40+** | **130+** |

---

## Performance Targets

| Operation | Target | Acceptable | Notes |
|-----------|--------|------------|-------|
| Architecture Detection | < 1ms | 0.1-2ms | Config parsing only |
| Category Detection | < 0.5ms | 0.05-1ms | File system checks |
| SAE Detection | < 0.3ms | 0.05-0.5ms | Directory listing |
| Support Check | < 1.5ms | 0.5-3ms | Combined operations |
| Tokenizer Load | < 2ms | 1-5ms | With fallback |

---

## Adding New Architectures

To add support for a new architecture:

1. **Add to SUPPORTED_ARCHITECTURES** (if standard Transformers architecture):

   ```python
   SUPPORTED_ARCHITECTURES = [
       # ... existing ...
       "NewArchitectureForCausalLM",
   ]
   ```

2. **Add Model Type Mapping** (if custom model type):

   ```python
   MODEL_TYPE_MAPPINGS = {
       # ... existing ...
       "new_model_type": {
           "architecture": "NewArchitectureForCausalLM",
           "config_class": "NewConfig"
       },
   }
   ```

3. **Add Tests**:
   - Unit test for detection
   - Integration test for loading
   - Update architecture count in docs

---

## Version Information

- **Document Version**: 1.0
- **Last Updated**: 2026-01-30
- **Loader Version**: Compatible with Nexus v6.1+
- **Test Coverage**: 130+ tests (90+ unit, 40+ integration)
- **Benchmarks**: 45+ performance benchmarks

---

## See Also

- [Omni Loader Guide](OMNI_LOADER_GUIDE.md) - Complete developer guide
- [NEXUS_V6_TECHNICAL_MANUAL](NEXUS_V6_TECHNICAL_MANUAL.md) - Technical manual
- [Test Coverage Report](../tests/OMNI_LOADER_TEST_COVERAGE.md) - Detailed test documentation
- [Benchmark Report](../benchmarks/LOADER_BENCHMARK_REPORT.md) - Performance benchmarks
