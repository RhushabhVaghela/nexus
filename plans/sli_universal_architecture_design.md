# Universal SLI Architecture Design

## Supporting 130+ Model Types

**Version:** 1.0  
**Date:** 2026-01-31  
**Status:** Design Specification

---

## Executive Summary

The current SLI (Sequential Layer Ingestion) implementation is tightly coupled to the Llama architecture, limiting its ability to handle the diverse landscape of transformer models. This document presents a universal architecture design that can handle all 130+ model types supported by the Nexus framework.

---

## 1. Current State Analysis

### 1.1 Hardcoded Llama-Specific Parts in `sli_integrator.py`

| Line | Code | Issue |
|------|------|-------|
| 23 | `prefix = f"model.layers.{layer_idx}."` | Hardcoded layer naming convention |
| 129-131 | `shard_names` lookup with `input_layernorm.weight` | Llama-specific weight names |
| 138 | `embed_tokens.weight` | Hardcoded embedding name |
| 141 | `nn.Embedding(self.config.vocab_size...)` | No embedding factory |
| 154-156 | `from transformers.models.llama...LlamaDecoderLayer` | Only Llama decoder layer |
| 209-211 | `model.embed_tokens` / `model.model.embed_tokens` | Limited embedding access patterns |

### 1.2 Critical Limitations

1. **Weight Naming Convention:** Assumes `model.layers.{idx}` structure
2. **Layer Class:** Only instantiates `LlamaDecoderLayer`
3. **Embedding Access:** Only handles `embed_tokens` or `word_embeddings`
4. **No MoE Support:** Cannot handle Mixture of Experts layers
5. **No Architecture Detection:** Does not use config to determine architecture family
6. **Limited Weight Formats:** Only safetensors, no `.bin` or `.pt` support

### 1.3 Architecture Family Gaps

| Family | Current Support | Needed Support |
|--------|-----------------|----------------|
| Llama-based | ✅ Partial | Full |
| GPT-based | ❌ None | Full |
| ChatGLM-based | ❌ None | Full |
| T5/Enc-Dec | ❌ None | Full |
| BLOOM | ❌ None | Full |
| OPT | ❌ None | Full |
| Mamba/SSM | ❌ None | Full |
| MoE Architectures | ❌ None | Full |

---

## 2. Architecture Taxonomy (130+ Models)

### 2.1 By Architecture Family

#### **Family 1: Llama-Based (30+ models)**

- **Base:** Llama, Llama2, Llama3
- **Variants:** Mistral, Mixtral, Qwen2, Qwen2.5, Yi, DeepSeek, Codellama
- **Key Characteristics:**
  - Decoder-only architecture
  - Pre-norm RMSNorm
  - SwiGLU activation
  - Rotary Position Embeddings (RoPE)
  - Weight naming: `model.layers.{idx}.self_attn.*`, `model.layers.{idx}.mlp.*`
  - Layer class: `LlamaDecoderLayer`

#### **Family 2: GPT-Based (15+ models)**

- **Models:** GPT-2, GPT-J, GPT-Neo, GPT-NeoX, Pythia, Falcon
- **Key Characteristics:**
  - Decoder-only
  - Post-layer norm or pre-layer norm variants
  - GELU activation (GPT-2), GeLU variants
  - Learned positional embeddings
  - Weight naming: `transformer.h.{idx}.attn.*`, `transformer.h.{idx}.mlp.*`
  - Layer class: `GPT2Block`, `GPTJBlock`, `GPTNeoXLayer`

#### **Family 3: ChatGLM-Based (8+ models)**

- **Models:** ChatGLM, ChatGLM2, ChatGLM3, GLM-4, GLM-4.7
- **Key Characteristics:**
  - Prefix decoder (GLM) or causal decoder (ChatGLM)
  - Special RoPE implementation
  - Multi-Query Attention (MQA) or Grouped Query Attention (GQA)
  - Weight naming: `transformer.encoder.layers.{idx}.*`
  - Layer class: `GLMBlock`, `ChatGLMBlock`

#### **Family 4: T5-Based (10+ models)**

- **Models:** T5, T5v1.1, FLAN-T5, UL2, LongT5, ByT5
- **Key Characteristics:**
  - Encoder-Decoder architecture
  - Relative positional bias
  - Pre-norm layer structure
  - Weight naming: `encoder.block.{idx}.*`, `decoder.block.{idx}.*`
  - Layer classes: `T5Block` (contains both encoder and decoder)

#### **Family 5: BLOOM-Based (5+ models)**

- **Models:** BLOOM, BLOOMZ, BLOOM-1b7, etc.
- **Key Characteristics:**
  - Decoder-only
  - ALiBi positional encodings
  - Pre-norm layer structure
  - Weight naming: `transformer.h.{idx}.self_attention.*`, `transformer.h.{idx}.mlp.*`
  - Layer class: `BloomBlock`

#### **Family 6: OPT-Based (5+ models)**

- **Models:** OPT, OPT-IML, etc.
- **Key Characteristics:**
  - Decoder-only
  - Learned positional embeddings
  - Pre-norm layer structure
  - Weight naming: `model.decoder.layers.{idx}.self_attn.*`, `model.decoder.layers.{idx}.fc1.*`
  - Layer class: `OPTDecoderLayer`

#### **Family 7: Mamba/State Space Models (10+ models)**

- **Models:** Mamba, Mamba2, Falcon-Mamba, Jamba, Zamba
- **Key Characteristics:**
  - State Space Model (SSM) layers instead of attention
  - Selective state spaces
  - Hardware-aware parallel scan
  - Weight naming: `backbone.layers.{idx}.mixer.*`, `backbone.layers.{idx}.norm.*`
  - Layer classes: `MambaBlock`, `Mamba2Block`, `JambaMambaLayer`

#### **Family 8: MoE Architectures (15+ models)**

- **Models:** Mixtral, GLM-4.7, Qwen2-MoE, DeepSeek-MoE, Grok
- **Key Characteristics:**
  - Sparse MoE layers with router
  - Expert selection mechanism
  - Load balancing
  - Weight naming: `model.layers.{idx}.block_sparse_moe.*` or `model.layers.{idx}.mlp.gate.*`
  - Layer classes: `MixtralDecoderLayer`, `Qwen2MoeDecoderLayer`, `DeepseekMoeLayer`

#### **Family 9: Multimodal Architectures (20+ models)**

- **Models:** LLaVA, Qwen-VL, CogVLM, InternVL, Idefics
- **Key Characteristics:**
  - Vision encoder + LLM backbone
  - Cross-modal projection layers
  - Special token handling for images
  - Layer classes: Often use base LLM layers with vision components

#### **Family 10: Encoder-Only (15+ models)**

- **Models:** BERT, RoBERTa, DeBERTa, ModernBERT, JinaBERT
- **Key Characteristics:**
  - Bidirectional attention
  - MLM training objective
  - No autoregressive generation
  - Layer classes: `BertLayer`, `RobertaLayer`

---

## 3. Universal SLI Design Specification

### 3.1 Core Architecture Components

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      UniversalSLIIntegrator                             │
├─────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────┐ │
│  │ Architecture    │  │ LayerFactory    │  │ WeightLoader            │ │
│  │ Registry        │  │                 │  │                         │ │
│  │                 │  │                 │  │                         │ │
│  │ - Detect family │  │ - Create layers │  │ - Load shards           │ │
│  │ - Get metadata  │  │ - Map weights   │  │ - Cache management      │ │
│  │ - Normalize     │  │ - Handle MoE    │  │ - Format support        │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────────────┘ │
│           │                   │                     │                   │
│           └───────────────────┼─────────────────────┘                   │
│                               ▼                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    Family-Specific Handlers                      │   │
│  ├─────────────┬─────────────┬─────────────┬───────────────────────┤   │
│  │ LlamaHandler│ GPTHandler  │ ChatGLM     │ T5Handler  │ ...      │   │
│  │             │             │ Handler     │            │          │   │
│  │ - Layer     │ - Layer     │ - Layer     │ - Encoder  │          │   │
│  │   naming    │   naming    │   naming    │ - Decoder  │          │   │
│  │ - Weight    │ - Weight    │ - Weight    │ - Weight   │          │   │
│  │   mapping   │   mapping   │   mapping   │   mapping  │          │   │
│  └─────────────┴─────────────┴─────────────┴───────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Key Design Patterns

#### Pattern 1: Architecture Registry

```python
class ArchitectureRegistry:
    """Registry for all supported architecture families."""
    
    _families: Dict[str, ArchitectureFamily] = {}
    
    @classmethod
    def register(cls, family_id: str, family: ArchitectureFamily):
        cls._families[family_id] = family
    
    @classmethod
    def detect_family(cls, config: PretrainedConfig) -> ArchitectureFamily:
        """Auto-detect architecture family from config."""
        model_type = getattr(config, "model_type", "").lower()
        architectures = getattr(config, "architectures", [])
        
        # Detection logic based on model_type and architectures
        for family in cls._families.values():
            if family.matches(model_type, architectures):
                return family
        
        raise UnsupportedArchitectureError(f"Cannot detect family for {model_type}")
```

#### Pattern 2: Layer Factory

```python
class UniversalLayerFactory:
    """Factory for creating layer instances from any architecture."""
    
    def create_layer(
        self,
        family: ArchitectureFamily,
        config: PretrainedConfig,
        layer_idx: int,
        layer_type: str = "decoder"
    ) -> nn.Module:
        """Create a layer instance for the given architecture family."""
        
        handler = self._get_handler(family)
        return handler.create_layer(config, layer_idx, layer_type)
    
    def get_weight_mapping(
        self,
        family: ArchitectureFamily,
        layer_idx: int
    ) -> Dict[str, str]:
        """Get weight name mapping for the layer."""
        
        handler = self._get_handler(family)
        return handler.get_weight_mapping(layer_idx)
```

#### Pattern 3: Weight Loader Strategy

```python
class UniversalWeightLoader:
    """Universal weight loading with format detection and caching."""
    
    SUPPORTED_FORMATS = [".safetensors", ".bin", ".pt", ".pth"]
    
    def __init__(self, cache_dir: str):
        self.cache_dir = cache_dir
        self.loaded_shards = {}
        self.weight_map = None
        self.format = None
    
    def discover_format(self, model_id: str) -> str:
        """Auto-discover weight format from available files."""
        # Check for index files first
        for ext in ["safetensors", "bin"]:
            index_url = f"{model_id}/resolve/main/model.{ext}.index.json"
            if self._check_exists(index_url):
                self.format = ext
                return ext
        
        # Fallback: check for single file
        for ext in self.SUPPORTED_FORMATS:
            url = f"{model_id}/resolve/main/pytorch_model.{ext}"
            if self._check_exists(url):
                self.format = ext
                return ext
        
        raise ValueError(f"No supported weight format found for {model_id}")
```

---

## 4. Architecture-Specific Specifications

### 4.1 Weight Naming Conventions by Family

| Family | Layer Prefix Pattern | Embedding Pattern | LM Head Pattern |
|--------|---------------------|-------------------|-----------------|
| Llama | `model.layers.{idx}.` | `model.embed_tokens` | `lm_head` |
| GPT-2 | `transformer.h.{idx}.` | `transformer.wte` | `lm_head` |
| GPT-J | `transformer.h.{idx}.` | `transformer.wte` | `lm_head` |
| GPT-NeoX | `gpt_neox.layers.{idx}.` | `gpt_neox.embed_in` | `embed_out` |
| ChatGLM | `transformer.encoder.layers.{idx}.` | `transformer.embedding` | `transformer.output_layer` |
| T5 | `encoder.block.{idx}.` / `decoder.block.{idx}.` | `shared` | `lm_head` |
| BLOOM | `transformer.h.{idx}.` | `transformer.word_embeddings` | `lm_head` |
| OPT | `model.decoder.layers.{idx}.` | `model.decoder.embed_tokens` | `lm_head` |
| Mamba | `backbone.layers.{idx}.` | `backbone.embeddings` | `backbone.lm_head` |
| Mixtral | `model.layers.{idx}.` | `model.embed_tokens` | `lm_head` |

### 4.2 Layer Class Mapping

| Family | Layer Class Path | Module Import |
|--------|-----------------|---------------|
| Llama | `transformers.models.llama.modeling_llama.LlamaDecoderLayer` | `LlamaDecoderLayer` |
| Mistral | `transformers.models.mistral.modeling_mistral.MistralDecoderLayer` | `MistralDecoderLayer` |
| Mixtral | `transformers.models.mixtral.modeling_mixtral.MixtralDecoderLayer` | `MixtralDecoderLayer` |
| Qwen2 | `transformers.models.qwen2.modeling_qwen2.Qwen2DecoderLayer` | `Qwen2DecoderLayer` |
| GPT-2 | `transformers.models.gpt2.modeling_gpt2.GPT2Block` | `GPT2Block` |
| GPT-J | `transformers.models.gptj.modeling_gptj.GPTJBlock` | `GPTJBlock` |
| GPT-NeoX | `transformers.models.gpt_neox.modeling_gpt_neox.GPTNeoXLayer` | `GPTNeoXLayer` |
| ChatGLM | `modeling_chatglm.GLMBlock` | `GLMBlock` (trust_remote_code) |
| T5 | `transformers.models.t5.modeling_t5.T5Block` | `T5Block` |
| BLOOM | `transformers.models.bloom.modeling_bloom.BloomBlock` | `BloomBlock` |
| OPT | `transformers.models.opt.modeling_opt.OPTDecoderLayer` | `OPTDecoderLayer` |
| Mamba | `transformers.models.mamba.modeling_mamba.MambaBlock` | `MambaBlock` |
| Mamba2 | `transformers.models.mamba2.modeling_mamba2.Mamba2Block` | `Mamba2Block` |
| DeepSeek | `transformers.models.deepseek.modeling_deepseek.DeepseekDecoderLayer` | `DeepseekDecoderLayer` |
| Qwen2MoE | `transformers.models.qwen2_moe.modeling_qwen2_moe.Qwen2MoeDecoderLayer` | `Qwen2MoeDecoderLayer` |

### 4.3 MoE Layer Handling

#### MoE Detection Strategy

```python
def detect_moe_architecture(config: PretrainedConfig) -> MoEConfig:
    """Detect MoE configuration from model config."""
    
    moe_config = MoEConfig()
    
    # Check for MoE-specific attributes
    if hasattr(config, "num_local_experts"):
        # Mixtral-style
        moe_config.num_experts = config.num_local_experts
        moe_config.top_k = getattr(config, "num_experts_per_tok", 2)
        moe_config.moe_type = "mixtral"
    
    elif hasattr(config, "n_routed_experts"):
        # DeepSeek-style
        moe_config.num_experts = config.n_routed_experts
        moe_config.top_k = getattr(config, "num_experts_per_tok", 6)
        moe_config.moe_type = "deepseek"
        moe_config.has_shared_experts = hasattr(config, "n_shared_experts")
    
    elif hasattr(config, "moe_intermediate_size"):
        # Qwen2-MoE style
        moe_config.num_experts = getattr(config, "num_experts", 8)
        moe_config.top_k = getattr(config, "num_experts_per_tok", 2)
        moe_config.moe_type = "qwen2_moe"
    
    return moe_config
```

#### MoE Weight Sharding Strategy

```python
class MoEWeightSharding:
    """Handle MoE expert weight loading and sharding."""
    
    def load_expert_weights(
        self,
        layer_idx: int,
        expert_indices: List[int],
        weight_map: Dict[str, str]
    ) -> Dict[str, torch.Tensor]:
        """
        Load only required expert weights for a layer.
        
        For inference, we may only need top-k experts per layer.
        For training, we may need all experts.
        """
        expert_weights = {}
        
        for expert_idx in expert_indices:
            # Expert-specific weight patterns
            patterns = self._get_expert_patterns(layer_idx, expert_idx)
            
            for pattern in patterns:
                if pattern in weight_map:
                    shard_name = weight_map[pattern]
                    weights = self._load_shard(shard_name)
                    expert_weights[pattern] = weights[pattern]
        
        return expert_weights
```

---

## 5. Implementation Plan

### Phase 1: Foundation (Week 1)

#### 5.1.1 Create Architecture Registry

**File:** `src/nexus_final/sli/architecture_registry.py`

```python
class ArchitectureFamily:
    """Base class for architecture families."""
    
    family_id: str
    model_types: List[str]
    architectures: List[str]
    
    def matches(self, model_type: str, architectures: List[str]) -> bool:
        """Check if config matches this family."""
        return (
            model_type in self.model_types or
            any(arch in self.architectures for arch in architectures)
        )
    
    def get_layer_prefix(self, layer_idx: int, layer_type: str = "decoder") -> str:
        """Get weight prefix for layer."""
        raise NotImplementedError
    
    def create_layer(self, config, layer_idx: int) -> nn.Module:
        """Create layer instance."""
        raise NotImplementedError
```

#### 5.1.2 Create Family Implementations

- `LlamaFamilyHandler`
- `GPTFamilyHandler`
- `ChatGLMFamilyHandler`
- `T5FamilyHandler`
- `BLOOMFamilyHandler`
- `OPTFamilyHandler`
- `MambaFamilyHandler`
- `MoEFamilyHandler`

### Phase 2: Universal Layer Factory (Week 2)

#### 5.2.1 Layer Factory Core

**File:** `src/nexus_final/sli/layer_factory.py`

```python
class UniversalLayerFactory:
    """Factory for creating architecture-agnostic layers."""
    
    def __init__(self):
        self.registry = ArchitectureRegistry()
        self._handler_cache = {}
    
    def create_layer(
        self,
        config: PretrainedConfig,
        layer_idx: int,
        layer_type: str = "decoder"
    ) -> nn.Module:
        """Create layer from config."""
        family = self.registry.detect_family(config)
        handler = self._get_handler(family)
        return handler.create_layer(config, layer_idx, layer_type)
    
    def get_weight_prefix(
        self,
        config: PretrainedConfig,
        layer_idx: int,
        layer_type: str = "decoder"
    ) -> str:
        """Get weight prefix for layer."""
        family = self.registry.detect_family(config)
        handler = self._get_handler(family)
        return handler.get_layer_prefix(layer_idx, layer_type)
```

#### 5.2.2 Weight Mapping System

**File:** `src/nexus_final/sli/weight_mapper.py`

```python
class WeightNameMapper:
    """Maps weight names between different architectures."""
    
    def __init__(self, family: ArchitectureFamily):
        self.family = family
        self._mapping_cache = {}
    
    def normalize_weight_name(self, name: str) -> str:
        """Normalize weight name to canonical form."""
        # Remove family-specific prefixes
        for prefix in self.family.get_layer_prefixes():
            if name.startswith(prefix):
                return name[len(prefix):]
        return name
    
    def denormalize_weight_name(self, name: str, layer_idx: int) -> str:
        """Add family-specific prefix."""
        prefix = self.family.get_layer_prefix(layer_idx)
        return f"{prefix}{name}"
```

### Phase 3: Enhanced Weight Loader (Week 2-3)

#### 5.3.1 Multi-Format Support

**File:** `src/nexus_final/sli/weight_loader.py`

```python
class UniversalWeightLoader:
    """Load weights from multiple formats."""
    
    FORMAT_LOADERS = {
        ".safetensors": load_file,
        ".bin": torch.load,
        ".pt": torch.load,
        ".pth": torch.load,
    }
    
    def __init__(self, cache_dir: str, model_id: str):
        self.cache_dir = cache_dir
        self.model_id = model_id
        self.format = self._detect_format()
        self.weight_map = self._load_weight_map()
        self.loaded_shards = {}
    
    def _detect_format(self) -> str:
        """Auto-detect weight format."""
        # Implementation
    
    def load_layer_weights(
        self,
        layer_idx: int,
        family: ArchitectureFamily
    ) -> Dict[str, torch.Tensor]:
        """Load weights for specific layer."""
        prefix = family.get_layer_prefix(layer_idx)
        
        # Find shards containing this prefix
        needed_shards = set()
        for weight_name, shard_name in self.weight_map.items():
            if weight_name.startswith(prefix):
                needed_shards.add(shard_name)
        
        # Load and extract weights
        layer_weights = {}
        for shard_name in needed_shards:
            weights = self._load_shard(shard_name)
            for k, v in weights.items():
                if k.startswith(prefix):
                    layer_weights[k.replace(prefix, "")] = v
        
        return layer_weights
```

### Phase 4: MoE Support (Week 3)

#### 5.4.1 MoE Detection and Routing

**File:** `src/nexus_final/sli/moe_handler.py`

```python
class MoEHandler:
    """Handle MoE-specific operations."""
    
    def __init__(self, config: PretrainedConfig):
        self.config = self._parse_moe_config(config)
    
    def _parse_moe_config(self, config) -> MoEConfig:
        """Parse MoE configuration from model config."""
        # Implementation
    
    def get_expert_weight_pattern(self, layer_idx: int, expert_idx: int) -> str:
        """Get weight pattern for specific expert."""
        # Family-specific patterns
    
    def should_load_expert(self, layer_idx: int, expert_idx: int, routing_weights: torch.Tensor) -> bool:
        """Determine if expert should be loaded based on routing."""
        # Load only top-k experts
        top_k = self.config.top_k
        top_experts = routing_weights.topk(top_k).indices
        return expert_idx in top_experts
```

### Phase 5: Integration and Refactoring (Week 4)

#### 5.5.1 Refactor SLIIntegrator

**File:** `src/nexus_final/sli_integrator_v2.py`

```python
class UniversalSLIIntegrator:
    """Universal Sequential Layer Ingestion."""
    
    def __init__(
        self,
        model_id: str,
        output_dir: str = "profiles/sli_profile",
        cache_dir: str = "temp_sli_shards",
        activation_cache_dir: str = "activation_cache",
        device: str = "cuda"
    ):
        self.model_id = model_id
        self.device = device
        
        # Load config
        self.config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        
        # Initialize architecture components
        self.registry = ArchitectureRegistry()
        self.family = self.registry.detect_family(self.config)
        self.factory = UniversalLayerFactory()
        self.weight_loader = UniversalWeightLoader(cache_dir, model_id)
        
        # MoE support
        if self._is_moe_model():
            self.moe_handler = MoEHandler(self.config)
    
    def run_sli(self, dataset: List[str]):
        """Run SLI pipeline."""
        num_layers = self._get_num_layers()
        
        # Process embeddings
        current_act_path = self._process_embeddings(dataset)
        
        for layer_idx in range(num_layers):
            print(f"\n[SLI] Processing Layer {layer_idx+1}/{num_layers}...")
            
            # Create layer
            layer = self.factory.create_layer(self.config, layer_idx)
            
            # Load weights
            layer_weights = self.weight_loader.load_layer_weights(
                layer_idx, self.family
            )
            layer.load_state_dict(layer_weights, strict=False)
            
            # Forward pass
            next_act_path = self._forward_layer(current_act_path, layer, layer_idx)
            
            # Cleanup
            current_act_path = next_act_path
            del layer
            torch.cuda.empty_cache()
```

### Phase 6: Testing and Validation (Week 5)

#### 5.6.1 Test Matrix

| Architecture | Model | Test Status |
|--------------|-------|-------------|
| Llama | meta-llama/Llama-3.2-1B | ⬜ |
| Mistral | mistralai/Mistral-7B-v0.1 | ⬜ |
| Mixtral | mistralai/Mixtral-8x7B-v0.1 | ⬜ |
| Qwen2 | Qwen/Qwen2-7B | ⬜ |
| GPT-2 | gpt2 | ⬜ |
| ChatGLM | THUDM/chatglm3-6b | ⬜ |
| T5 | google/flan-t5-base | ⬜ |
| BLOOM | bigscience/bloom-560m | ⬜ |
| OPT | facebook/opt-125m | ⬜ |
| Mamba | state-spaces/mamba-370m | ⬜ |

---

## 6. Migration Guide

### 6.1 API Changes

| Old API | New API | Notes |
|---------|---------|-------|
| `SequentialLayerIntegrator` | `UniversalSLIIntegrator` | Drop-in replacement |
| `_create_layer(idx)` | `factory.create_layer(config, idx)` | Factory-based |
| `loader.load_layer_weights(idx)` | `loader.load_layer_weights(idx, family)` | Family-aware |
| `_process_embeddings(dataset)` | `_process_embeddings(dataset, family)` | Family-aware |

### 6.2 Backward Compatibility

```python
# Legacy compatibility wrapper
class SequentialLayerIntegrator(UniversalSLIIntegrator):
    """Backward-compatible wrapper."""
    
    def __init__(self, *args, **kwargs):
        # Force Llama family for legacy behavior
        super().__init__(*args, **kwargs)
        self.family = LlamaFamilyHandler()
```

---

## 7. Performance Considerations

### 7.1 Memory Optimization

1. **Shard Caching:** Keep frequently accessed shards in memory
2. **Expert Pruning:** For MoE, only load active experts
3. **Lazy Loading:** Defer weight loading until needed
4. **Activation Checkpointing:** Save activations to SSD

### 7.2 Speed Optimization

1. **Parallel Shard Loading:** Download multiple shards concurrently
2. **Weight Pre-fetching:** Pre-load next layer weights during computation
3. **Format Selection:** Prefer safetensors for faster loading
4. **Cache Warmup:** Maintain persistent weight cache

---

## 8. Future Extensions

### 8.1 Planned Features

1. **Quantization Awareness:** Support loading quantized weights directly
2. **Speculative Decoding:** Integrate with speculative execution
3. **Multi-GPU Sharding:** Distribute layers across multiple GPUs
4. **Dynamic Batching:** Batch multiple sequences efficiently

### 8.2 Architecture Roadmap

| Quarter | Target Architectures |
|---------|---------------------|
| Q1 2026 | All 130+ current architectures |
| Q2 2026 | New MoE variants (DeepSeek v3, etc.) |
| Q3 2026 | Multimodal architectures (LLaVA, etc.) |
| Q4 2026 | State Space Models v2 |

---

## 9. Appendix

### 9.1 Complete Architecture List (130+ models)

See `src/nexus_final/sli/supported_architectures.json` for machine-readable list.

### 9.2 Weight Map Examples

See `docs/weight_naming_conventions.md` for detailed examples per family.

### 9.3 Troubleshooting Guide

See `docs/sli_troubleshooting.md` for common issues and solutions.

---

**Document End**
