# Encoder-Only Model Support Documentation

## Table of Contents

- [Overview](#overview)
- [Encoder vs Decoder Architectures](#encoder-vs-decoder-architectures)
- [Supported Models](#supported-models)
- [Limitations](#limitations)
- [Quick Start](#quick-start)
- [API Reference](#api-reference)
- [Use Cases](#use-cases)
- [Examples](#examples)
- [Comparison](#comparison)

---

## Overview

Nexus SLI (Selective Layer Inference) provides **limited support** for encoder-only architectures through the [`BERTFamilyHandler`](src/nexus_final/sli/architecture_registry.py:569). While decoder architectures (GPT, Llama, etc.) are fully supported for generative tasks, encoder-only models serve different purposes and have distinct operational characteristics.

### What Are Encoder-Only Models?

Encoder-only models process input bi-directionally to create contextualized representations. They excel at:

- Understanding input context
- Classification tasks
- Embedding extraction
- Semantic similarity

Unlike decoder models, they **do not generate sequences autoregressively**.

```
┌─────────────────────────────────────────────────────────────────┐
│              Encoder vs Decoder Architecture                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ENCODER (BERT-style)           DECODER (GPT-style)             │
│  ┌─────────────────────┐       ┌─────────────────────┐         │
│  │  Input Tokens       │       │  Input Tokens       │         │
│  │  [The][cat][sat]    │       │  [The][cat]         │         │
│  └──────────┬──────────┘       └──────────┬──────────┘         │
│             ▼                              ▼                    │
│  ┌─────────────────────┐       ┌─────────────────────┐         │
│  │  Bi-directional     │       │  Uni-directional    │         │
│  │  Attention          │       │  (Causal) Attention │         │
│  │                     │       │                     │         │
│  │  The ↔ cat ↔ sat    │       │  The → cat → [?]    │         │
│  │  (See all context)  │       │  (Only past tokens) │         │
│  └──────────┬──────────┘       └──────────┬──────────┘         │
│             ▼                              ▼                    │
│  ┌─────────────────────┐       ┌─────────────────────┐         │
│  │  Contextualized     │       │  Generate Next      │         │\n│  │  Representations    │       │  Token              │         │
│  │  [CLS] embedding    │       │  "sat"              │         │
│  └─────────────────────┘       └─────────────────────┘         │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Encoder vs Decoder Architectures

### Key Differences

| Aspect | Encoder (BERT) | Decoder (GPT) |
|--------|----------------|---------------|
| **Attention** | Bi-directional | Uni-directional (causal) |
| **Use Case** | Understanding | Generation |
| **Typical Tasks** | Classification, NER, QA | Text completion, chat |
| **Output** | Context vectors | Next token probabilities |
| **SLI Support** | Limited | Full |
| **Layer Type** | Encoder only | Decoder only |

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    BERT Encoder Layer                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Input Embeddings ──▶ Self-Attention ──▶ Feed Forward ──▶ Output│
│                        (Bi-directional)                          │
│                                                                  │
│  All tokens attend to all other tokens                           │
│  [The] [cat] [sat] [on] [mat]                                   │
│    ↕     ↕     ↕     ↕     ↕                                    │
│  (Full cross-attention within sequence)                          │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                   GPT Decoder Layer                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Input Embeddings ──▶ Masked Self-Attn ──▶ Feed Fwd ──▶ Output  │
│                        (Causal/Auto-regressive)                  │
│                                                                  │
│  Tokens only attend to previous tokens                           │
│  [The] [cat] [sat] [on] [mat]                                   │
│    ↓     ↓     ↓     ↓     ↓                                    │
│  (Each position only sees past positions)                        │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Why SLI is Different for Encoders

```
┌─────────────────────────────────────────────────────────────────┐
│              SLI Flow Comparison                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  DECODER (Full SLI Support)              ENCODER (Limited)       │
│                                                                  │
│  ┌──────────────┐                        ┌──────────────┐       │
│  │ Load Layer 0 │                        │ Load All     │       │
│  │ Process      │                        │ Encoders     │       │
│  │ Evict        │                        │ Process      │       │
│  │ Load Layer 1 │                        │ (No evict)   │       │
│  │ Process      │                        └──────────────┘       │
│  │ Evict        │                                                │
│  │ ...          │         Encoders need ALL layers for           │
│  │ Load Layer N │         bi-directional attention!              │
│  │ Generate     │                                                │
│  └──────────────┘                                                │
│                                                                  │
│  ✓ Layer-by-layer efficient          ✗ Must load entire encoder │
│  ✓ Memory scales with 1 layer        ✗ Memory scales with all   │
│  ✓ Great for generation              ✓ Still useful for caching │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Supported Models

The [`BERTFamilyHandler`](src/nexus_final/sli/architecture_registry.py:569) supports the following encoder-only architectures:

### Core BERT Family

| Model | Model Type | Architecture | Description |
|-------|------------|--------------|-------------|
| **BERT** | `bert` | `BertModel`, `BertForMaskedLM`, `BertForSequenceClassification` | Original BERT |
| **RoBERTa** | `roberta` | `RobertaModel`, `RobertaForSequenceClassification` | Robustly optimized BERT |
| **DeBERTa** | `deberta`, `deberta_v2` | `DebertaModel`, `DebertaForSequenceClassification` | Decoding-enhanced BERT |

### Distilled/Optimized Variants

| Model | Model Type | Architecture | Description |
|-------|------------|--------------|-------------|
| **DistilBERT** | `distilbert` | `DistilBertModel`, `DistilBertForMaskedLM` | 40% smaller, 60% faster |
| **ALBERT** | `albert` | `AlbertModel`, `AlbertForMaskedLM` | Parameter sharing, lighter |
| **ModernBERT** | `modernbert` | `ModernBertModel` | Efficient modern variant |

### Specialized Variants

| Model | Model Type | Architecture | Description |
|-------|------------|--------------|-------------|
| **JinaBERT** | `jinabert` | `JinaBertModel` | Embedding-optimized |
| **Nomic BERT** | `nomic_bert` | `NomicBertModel` | Contrastive training |
| **NeoBERT** | `neobert` | `NeoBERT`, `NeoBERTForSequenceClassification` | Long context support |

### Multilingual Variants

| Model | Model Type | Architecture | Description |
|-------|------------|--------------|-------------|
| **XLM-RoBERTa** | `xlm_roberta` | `XLMRobertaModel` | Cross-lingual |
| **CamemBERT** | `camembert` | `CamembertModel` | French optimized |
| **ELECTRA** | `electra` | `ElectraModel` | Discriminator-style |

### Supported Model List

```python
# All supported model types
model_types = [
    "bert",           # Original BERT
    "roberta",        # Robustly optimized BERT
    "deberta",        # DeBERTa v1
    "deberta_v2",     # DeBERTa v2
    "distilbert",     # Distilled BERT
    "albert",         # Lite BERT
    "modernbert",     # Modern BERT
    "jinabert",       # Jina embeddings
    "nomic_bert",     # Nomic embeddings
    "neobert",        # NeoBERT long context
    "electra",        # ELECTRA discriminator
    "xlm_roberta",    # XLM-RoBERTa
    "camembert",      # CamemBERT (French)
]
```

---

## Limitations

### Why SLI is Limited for Encoders

1. **Bi-directional Attention Requirement**

   ```
   Encoder layers require access to ALL tokens simultaneously.
   You cannot process layer-by-layer while evicting previous layers
   because every layer needs full context.
   ```

2. **No Autoregressive Generation**

   ```
   Encoders don't generate text token-by-token.
   They produce fixed-size context vectors.
   The "selective" aspect of SLI doesn't apply the same way.
   ```

3. **Memory Requirements**

   ```
   For a 24-layer BERT encoder:
   - Decoder SLI: Keep 1-2 layers in memory at a time
   - Encoder: Must load all 24 layers for inference
   ```

### Specific Limitations

| Feature | Decoder Support | Encoder Support | Reason |
|---------|-----------------|-----------------|--------|
| Layer-by-layer inference | ✅ Full | ⚠️ Limited | Bi-directional attention |
| KV-cache optimization | ✅ Yes | ❌ N/A | No generation |
| Streaming generation | ✅ Yes | ❌ N/A | Fixed-size output |
| Memory scaling O(1) | ✅ Yes | ❌ O(N) | Need all layers |
| Multi-layer eviction | ✅ Yes | ❌ No | Context dependencies |

### What Works Well

Despite limitations, encoder SLI provides value for:

| Use Case | Benefit |
|----------|---------|
| **Model caching** | Quantized encoder layers cached efficiently |
| **Batch processing** | Reuse cached encoder for multiple inputs |
| **Embedding extraction** | Load encoder once, extract many embeddings |
| **Fine-tuning** | SLI can help with parameter-efficient training |

---

## Quick Start

### Basic Usage

```python
from src.nexus_final.sli.architecture_registry import BERTFamilyHandler
from transformers import AutoConfig

# 1. Create handler
handler = BERTFamilyHandler()

# 2. Load model config
config = AutoConfig.from_pretrained("bert-base-uncased")

# 3. Check if handler matches
is_match = handler.matches("bert", ["BertModel"], config)
print(f"Handler matches: {is_match}")  # True

# 4. Get layer information
num_layers = handler.get_num_layers(config)
hidden_size = handler.get_hidden_size(config)
print(f"Layers: {num_layers}, Hidden size: {hidden_size}")
# Layers: 12, Hidden size: 768
```

### Creating Encoder Layers

```python
# Create a single encoder layer
layer = handler.create_layer(config, layer_idx=0, layer_type="encoder")
print(f"Created layer: {type(layer).__name__}")
# Created layer: BertLayer

# Get layer weight prefix
prefix = handler.get_layer_prefix(0)
print(f"Weight prefix: {prefix}")
# Weight prefix: encoder.layer.0.
```

### Working with Different Model Types

```python
# RoBERTa
roberta_config = AutoConfig.from_pretrained("roberta-base")
handler.create_layer(roberta_config, 0)
prefix = handler.get_layer_prefix(0)
print(f"RoBERTa prefix: {prefix}")
# RoBERTa prefix: roberta.encoder.layer.0.

# DistilBERT
distil_config = AutoConfig.from_pretrained("distilbert-base-uncased")
handler.create_layer(distil_config, 0)
prefix = handler.get_layer_prefix(0)
print(f"DistilBERT prefix: {prefix}")
# DistilBERT prefix: transformer.layer.0.

# ALBERT
albert_config = AutoConfig.from_pretrained("albert-base-v2")
handler.create_layer(albert_config, 0)
prefix = handler.get_layer_prefix(13)  # ALBERT shares parameters
print(f"ALBERT prefix: {prefix}")
# ALBERT prefix: albert.encoder.albert_layer_group.1.albert_layers.1.
```

---

## API Reference

### BERTFamilyHandler

#### Constructor

```python
handler = BERTFamilyHandler()
```

Creates a handler for BERT-based encoder architectures.

#### Attributes

| Attribute | Type | Value | Description |
|-----------|------|-------|-------------|
| `family_id` | `str` | `"bert"` | Unique family identifier |
| `family_name` | `str` | `"BERT-Based Encoder Architectures"` | Human-readable name |
| `model_types` | `List[str]` | 13 types | Supported model type strings |
| `architectures` | `List[str]` | 20+ architectures | Supported architecture classes |
| `trust_remote_code` | `bool` | `False` | Whether remote code is needed |

#### Methods

##### matches()

```python
def matches(
    self, 
    model_type: str, 
    architectures: List[str], 
    config: Optional[Any] = None
) -> bool
```

Check if config matches this family.

**Parameters:**

- `model_type`: The model_type from config
- `architectures`: List of architecture names from config
- `config`: Optional full config object

**Returns:** `True` if this family handles the given config

**Example:**

```python
config = AutoConfig.from_pretrained("bert-base-uncased")
is_match = handler.matches(
    config.model_type, 
    config.architectures,
    config
)
```

##### get_layer_prefix()

```python
def get_layer_prefix(
    self, 
    layer_idx: int, 
    layer_type: str = "encoder"
) -> str
```

Get weight prefix for encoder layer.

**Note:** Encoder-only models only support `"encoder"` layer type.

**Parameters:**

- `layer_idx`: Layer index
- `layer_type`: Type of layer (must be `"encoder"`)

**Returns:** Weight name prefix for the layer

**Examples:**

```python
# BERT
handler.get_layer_prefix(0)  # "encoder.layer.0."
handler.get_layer_prefix(5)  # "encoder.layer.5."

# RoBERTa (after create_layer)
handler.get_layer_prefix(0)  # "roberta.encoder.layer.0."

# DistilBERT
handler.get_layer_prefix(0)  # "transformer.layer.0."

# ALBERT (parameter sharing)
handler.get_layer_prefix(13)  # Wraps around: "albert_layer_group.1..."
```

##### create_layer()

```python
def create_layer(
    self, 
    config: PretrainedConfig, 
    layer_idx: int,
    layer_type: str = "encoder"
) -> nn.Module
```

Create an encoder layer for this architecture family.

**Note:** Encoder-only models only have encoder layers, no decoder.

**Parameters:**

- `config`: Model configuration
- `layer_idx`: Layer index
- `layer_type`: Type of layer (must be `"encoder"`)

**Returns:** Instantiated layer module

**Example:**

```python
from transformers import BertConfig

config = BertConfig()
layer = handler.create_layer(config, layer_idx=0, layer_type="encoder")
```

##### get_embedding_name()

```python
def get_embedding_name(self) -> str
```

Get the embedding weight name.

**Returns:** Embedding layer path

**Examples:**

```python
# BERT
handler.get_embedding_name()  # "embeddings"

# RoBERTa
handler.get_embedding_name()  # "roberta.embeddings"

# DistilBERT
handler.get_embedding_name()  # "distilbert.embeddings"

# ALBERT
handler.get_embedding_name()  # "albert.embeddings"
```

##### get_lm_head_name()

```python
def get_lm_head_name(self) -> str
```

Get the LM head weight name.

**Note:** Encoder-only models don't have LM heads for generation. Returns `None`.

**Returns:** `None`

##### is_encoder_only()

```python
def is_encoder_only(self) -> bool
```

Indicate this is an encoder-only architecture.

**Returns:** `True`

**Example:**

```python
handler.is_encoder_only()  # True
```

##### Inherited Methods

From [`ArchitectureFamily`](src/nexus_final/sli/architecture_registry.py:22):

| Method | Description |
|--------|-------------|
| `get_num_layers(config)` | Get number of layers |
| `get_hidden_size(config)` | Get hidden dimension |
| `get_vocab_size(config)` | Get vocabulary size |

---

## Use Cases

### When to Use Encoder Models

#### ✅ Recommended Use Cases

1. **Text Classification**

   ```python
   # Sentiment analysis, spam detection, topic classification
   from transformers import AutoTokenizer, AutoModelForSequenceClassification
   
   tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
   model = AutoModelForSequenceClassification.from_pretrained(
       "bert-base-uncased",
       num_labels=2
   )
   ```

2. **Named Entity Recognition (NER)**

   ```python
   # Extract entities: persons, organizations, locations
   from transformers import AutoModelForTokenClassification
   
   model = AutoModelForTokenClassification.from_pretrained(
       "dslim/bert-base-NER"
   )
   ```

3. **Semantic Similarity**

   ```python
   # Compare sentence meanings
   from sentence_transformers import SentenceTransformer
   
   model = SentenceTransformer('all-MiniLM-L6-v2')  # Based on BERT
   embeddings = model.encode(["text 1", "text 2"])
   similarity = cosine_similarity(embeddings[0], embeddings[1])
   ```

4. **Question Answering (Extractive)**

   ```python
   # Find answer span in context
   from transformers import AutoModelForQuestionAnswering
   
   model = AutoModelForQuestionAnswering.from_pretrained(
       "deepset/bert-base-cased-squad2"
   )
   ```

5. **Embedding Extraction**

   ```python
   # Get contextualized embeddings for downstream tasks
   outputs = model(**inputs)
   embeddings = outputs.last_hidden_state  # [batch, seq, hidden]
   pooled = outputs.pooler_output          # [batch, hidden]
   ```

#### ❌ Not Recommended

- **Text generation** (use decoder models)
- **Chat/conversation** (use decoder models)
- **Code completion** (use code-specific decoder models)
- **Long-form writing** (use decoder models)

### SLI Benefits for Encoders

Even with limited support, SLI provides:

```python
# 1. Efficient Caching
from src.nexus_final.sli.layer_cache import LayerCache

cache = LayerCache(max_cache_size_gb=10)
# Cache quantized encoder layers for fast loading

# 2. Quantized Storage
from src.nexus_final.sli.quantization import get_int8_config

config = get_int8_config()
# Store encoder in 8-bit for 50% memory savings

# 3. Batch Processing
# Load encoder once, process multiple inputs
encoder = load_encoder()
for batch in data_loader:
    embeddings = encoder(batch)
```

---

## Examples

### Example 1: Basic Classification

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from src.nexus_final.sli.architecture_registry import BERTFamilyHandler
import torch

# Load model
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=2
)

# Classify text
text = "This movie was absolutely fantastic!"
inputs = tokenizer(text, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)
    predictions = torch.softmax(outputs.logits, dim=-1)
    predicted_class = torch.argmax(predictions, dim=-1).item()

print(f"Sentiment: {'Positive' if predicted_class == 1 else 'Negative'}")
print(f"Confidence: {predictions[0][predicted_class]:.2%}")
```

### Example 2: Extract Embeddings

```python
from transformers import AutoTokenizer, AutoModel
from src.nexus_final.sli.architecture_registry import get_registry
import torch

# Load with SLI registry
registry = get_registry()
config = AutoConfig.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

# Detect family
family = registry.detect_family(config)
print(f"Detected family: {family.family_id}")  # "bert"

# Load model
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

# Extract embeddings
texts = [
    "The cat sat on the mat.",
    "A feline rested on the rug.",
]

inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)
    # Mean pooling
    attention_mask = inputs['attention_mask']
    token_embeddings = outputs.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).float()
    sum_embeddings = (token_embeddings * input_mask_expanded).sum(dim=1)
    embeddings = sum_embeddings / input_mask_expanded.sum(dim=1).clamp(min=1e-9)

print(f"Embeddings shape: {embeddings.shape}")
# Embeddings shape: torch.Size([2, 384])

# Compute similarity
similarity = torch.cosine_similarity(embeddings[0], embeddings[1], dim=0)
print(f"Similarity: {similarity:.3f}")
```

### Example 3: Named Entity Recognition

```python
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch

# Load NER model
tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")

# Process text
text = "Apple Inc. is headquartered in Cupertino, California."
inputs = tokenizer(text, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=-1)

# Decode predictions
tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
predicted_labels = [model.config.id2label[p.item()] for p in predictions[0]]

# Extract entities
entities = []
for token, label in zip(tokens, predicted_labels):
    if label != "O":  # Not "Outside"
        entities.append((token, label))

print("Entities found:")
for token, label in entities:
    print(f"  {token}: {label}")
# Entities found:
#   Apple: B-ORG
#   Inc: I-ORG
#   Cupertino: B-LOC
#   California: B-LOC
```

### Example 4: Cross-Encoder for Reranking

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Cross-encoder for relevance scoring
tokenizer = AutoTokenizer.from_pretrained("cross-encoder/ms-marco-MiniLM-L-6-v2")
model = AutoModelForSequenceClassification.from_pretrained(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

# Query and documents
query = "What is machine learning?"
documents = [
    "Machine learning is a subset of artificial intelligence.",
    "The capital of France is Paris.",
    "Deep learning uses neural networks with many layers.",
]

# Score each document
scores = []
for doc in documents:
    inputs = tokenizer(
        query, doc, 
        padding=True, 
        truncation=True, 
        return_tensors="pt"
    )
    with torch.no_grad():
        outputs = model(**inputs)
        score = torch.sigmoid(outputs.logits).item()
        scores.append((doc, score))

# Sort by relevance
scores.sort(key=lambda x: x[1], reverse=True)

print("Documents ranked by relevance:")
for doc, score in scores:
    print(f"  [{score:.3f}] {doc[:50]}...")
```

### Example 5: Using with Layer Cache

```python
from src.nexus_final.sli.layer_cache import LayerCache
from src.nexus_final.sli.quantization import get_int8_config
from transformers import AutoModel, AutoConfig
import torch

# Create cache
cache = LayerCache(
    cache_dir="/path/to/encoder_cache",
    max_cache_size_gb=5
)

# Load model
model_name = "sentence-transformers/all-MiniLM-L6-v2"
config = AutoConfig.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Cache each encoder layer
for idx, layer in enumerate(model.encoder.layer):
    cache.cache_layer(
        model_id=model_name,
        layer_index=idx,
        layer=layer,
        quantization_mode="int8"
    )

print(f"Cached {len(model.encoder.layer)} encoder layers")

# Later: Load from cache
loaded_layers = []
for idx in range(len(model.encoder.layer)):
    layer = cache.get_layer(model_name, idx)
    loaded_layers.append(layer)

print(f"Loaded {len(loaded_layers)} layers from cache")
```

---

## Comparison

### Encoder vs Decoder for Different Tasks

| Task | Encoder | Decoder | Recommendation |
|------|---------|---------|----------------|
| **Sentiment Analysis** | ✅ Excellent | ⚠️ Overkill | **Encoder** - Direct classification |
| **Text Classification** | ✅ Excellent | ⚠️ Overkill | **Encoder** - Purpose-built |
| **Named Entity Recognition** | ✅ Excellent | ❌ Poor | **Encoder** - Token-level predictions |
| **Semantic Search** | ✅ Excellent | ⚠️ Possible | **Encoder** - Embedding similarity |
| **Question Answering (Extractive)** | ✅ Excellent | ⚠️ Possible | **Encoder** - Span extraction |
| **Text Generation** | ❌ Cannot | ✅ Excellent | **Decoder** - Autoregressive |
| **Chat/Conversation** | ❌ Cannot | ✅ Excellent | **Decoder** - Conversational |
| **Code Completion** | ❌ Cannot | ✅ Excellent | **Decoder** - Generate code |
| **Summarization** | ⚠️ Possible | ✅ Excellent | **Decoder** - Generate summaries |
| **Translation** | ⚠️ Possible | ✅ Excellent | **Encoder-Decoder** | Seq2Seq |

### Performance Comparison

| Metric | BERT (Encoder) | GPT (Decoder) | Notes |
|--------|----------------|---------------|-------|
| **Parameters** | 110M-340M | 125M-175B | Similar range available |
| **Inference Speed** | Fast (single pass) | Slower (token-by-token) | Encoder = O(1), Decoder = O(N) |
| **Memory (SLI)** | O(N) - all layers | O(1) - one layer | Decoder benefits more from SLI |
| **Context Window** | 512-4096 tokens | 2K-128K tokens | Decoders scaling better |
| **Training Data** | Bi-directional | Left-to-right | Different pretraining |
| **Fine-tuning Cost** | Lower | Higher | Encoder = single forward pass |

### When to Choose Each

```
┌─────────────────────────────────────────────────────────────────┐
│                     Decision Tree                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Do you need to GENERATE text?                                  │
│                                                                  │
│     YES ─────────────────────────────────────▶ DECODER (GPT)    │
│     │                                                            │
│     NO                                                           │
│     │                                                            │
│     ▼                                                            │
│  Do you need to UNDERSTAND/CLASSIFY text?                       │
│                                                                  │
│     YES ─────────────────────────────────────▶ ENCODER (BERT)   │
│     │                                                            │
│     NO                                                           │
│     │                                                            │
│     ▼                                                            │
│  Do you need BOTH (e.g., translation, summarization)?           │
│                                                                  │
│     YES ─────────────────────────────────────▶ ENCODER-DECODER │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Hybrid Approaches

Some modern architectures combine both:

| Architecture | Type | Description |
|--------------|------|-------------|
| **T5** | Encoder-Decoder | Unified text-to-text framework |
| **BART** | Encoder-Decoder | Denoising seq2seq pretraining |
| **DeBERTa** | Encoder | Enhanced decoding in encoder |
| **ELECTRA** | Encoder | Discriminator-style pretraining |

---

## See Also

- [Architecture Registry](architecture_registry.py) - Full architecture support
- [SLI Universal Guide](SLI_UNIVERSAL_GUIDE.md) - Main SLI documentation
- [Quantization Documentation](QUANTIZATION.md) - Quantize encoder layers
- [Custom Layers](CUSTOM_LAYERS.md) - Extend with custom encoder layers

## References

- [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805)
- [RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/abs/1907.11692)
- [DeBERTa: Decoding-enhanced BERT with Disentangled Attention](https://arxiv.org/abs/2006.03654)
- [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers)
