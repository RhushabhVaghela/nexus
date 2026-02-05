# Test Suite Documentation

Complete reference for the Nexus Model test suite covering **~400 tests** across unit, integration, multimodal, reasoning, and E2E test categories.

---

## Quick Reference

| Category | Tests | Duration | Command |
|----------|-------|----------|---------|
| Unit | 180 | ~10s | `pytest tests/unit/ -v` |
| Integration | 68 | ~45s | `pytest tests/integration/ -v` |
| Multimodal | 20 | ~15s | `pytest tests/multimodal/ -v` |
| Nexus Final | 10 | ~30s | `pytest tests/nexus_final/ -v` |
| E2E | 21 | ~60s | `pytest tests/e2e/ -v` |
| Streaming | 10 | ~5s | `pytest tests/unit_streaming/ -v` |
| **Total** | **346** | **~120s** | `pytest tests/ --full-tests -v` |

---

## Test Categories

### 1. Unit Tests (`tests/unit/`)

Fast, isolated tests for individual modules.

#### `test_detect_modalities.py` (17 tests)

Tests the modality detection system that probes models for capabilities.

| Test | Description |
|------|-------------|
| `test_detect_modalities_import` | Verifies module can be imported |
| `test_detect_text_only_model` | Detects text-only model returns `text=True` only |
| `test_detect_returns_dict` | Output is properly structured dictionary |
| `test_modalities_keys_present` | All expected modality keys exist |
| `test_model_type_extracted` | Model type (qwen, llama, etc.) is extracted |
| `test_text_always_true` | Text modality is always True for LLMs |
| `test_vision_detection_logic` | Vision detection checks for vision_config |
| `test_audio_input_detection` | Audio input checks for audio_config |
| `test_audio_output_detection` | Audio output detection logic |
| `test_video_detection` | Video modality detection |
| `test_vision_output_detection` | Image generation capability detection |
| `test_video_output_detection` | Video generation capability detection |
| `test_omni_model_detection` | Full Omni model returns all modalities True |
| `test_config_analysis` | Internal `_analyze_config` function |
| `test_format_report_json` | JSON output format |
| `test_format_report_human` | Human-readable output format |
| `test_edge_cases` | Missing config files, invalid paths |
| **test_memory_trigger.py** | |
| `test_estimate_vram_7b` | Verify 7B model VRAM estimation (approx 3-5GB) |
| `test_estimate_vram_70b` | Verify 70B model VRAM estimation (approx 35-45GB) |
| `test_should_use_sli_trigger` | Verify SLI triggers only when VRAM is insufficient |

**Usage:**

```python
from src.detect_modalities import detect_modalities, format_report

result = detect_modalities("/path/to/model")
# Returns: {"modalities": {"text": True, "vision": False, ...}, "model_type": "qwen2"}

print(format_report(result, use_json=False))
```

---

#### `test_capability_registry.py` (14 tests)

Tests the capability definition and validation system.

| Test | Description |
|------|-------------|
| `test_registry_initialization` | CapabilityRegistry creates successfully |
| `test_registry_has_capabilities` | Registry contains capability definitions |
| `test_key_capabilities_registered` | Core capabilities (cot, reasoning, etc.) exist |
| `test_capability_has_required_modalities` | Each capability defines required modalities |
| `test_capability_has_training_script` | Each capability points to training script |
| `test_capability_has_vram_estimate` | VRAM requirements are specified |
| `test_validate_text_only_model` | Text-only validation with CoT |
| `test_validate_omni_model` | Full Omni model validates all capabilities |
| `test_get_nonexistent_capability` | Graceful handling of invalid capability names |
| `test_get_training_order` | Training order is returned as list |
| `test_text_capabilities_come_first` | Text capabilities ordered before multimodal |
| `test_generation_capabilities_come_last` | Generation capabilities are last |
| `test_capability_creation` | Manually creating Capability dataclass |
| `test_capability_required_fields` | Required vs optional fields |

**Usage:**

```python
from src.capability_registry import CapabilityRegistry

registry = CapabilityRegistry()
cot = registry.get("cot")

# Validate capability against model modalities
model_mods = {"text"}
valid, missing = cot.validate(model_mods)
# valid=True, missing=set()

# Get training order
order = registry.get_training_order(["cot", "reasoning", "podcast"])
# Returns: ["cot", "reasoning", "podcast"]
```

---

#### `test_training_controller.py` (17 tests)

Tests training safety features: pause/resume, cooldown, checkpointing.

| Test | Description |
|------|-------------|
| `test_setup_signal_handlers_runs` | Signal handlers register without error |
| `test_signal_handlers_registered` | SIGUSR1/SIGUSR2 handlers are set |
| `test_check_pause_state_not_paused` | Returns immediately when not paused |
| `test_check_pause_state_paused_then_resumed` | Blocks until resumed |
| `test_cooldown_interval_constant_defined` | COOLDOWN_INTERVAL_STEPS > 100 |
| `test_cooldown_duration_defined` | COOLDOWN_DURATION_SECONDS > 0 |
| `test_check_and_cooldown_not_at_interval` | No cooldown at non-interval steps |
| `test_check_and_cooldown_at_interval` | Cooldown triggers at intervals |
| `test_temperature_threshold_defined` | GPU_TEMP_THRESHOLD defined (75-90°C) |
| `test_get_gpu_temperature_success` | Successful nvidia-smi reading |
| `test_get_gpu_temperature_failure` | Graceful fallback when nvidia-smi fails |
| `test_extract_gzip` | .gz file extraction |
| `test_extract_tar_gz` | .tar.gz file extraction |
| `test_extract_zip` | .zip file extraction |
| `test_non_compressed_passthrough` | Regular files pass through unchanged |
| `test_hook_runs_without_error` | training_step_hook executes cleanly |
| `test_hook_calls_pause_check` | Hook invokes pause state check |
| `test_hook_calls_cooldown_check` | Hook invokes cooldown check |

**Pause/Resume Signal Usage:**

```bash
# Get training PID
ps aux | grep stage_cot.py

# Pause training
kill -USR1 <PID>

# Resume training
kill -USR1 <PID>

# Emergency checkpoint
kill -USR2 <PID>
```

---

### 2. Integration Tests (`tests/integration/`)

Tests with real model loading and component integration.

#### `test_real_model_loading.py` (17 tests)

Uses real **Qwen2.5-0.5B** model for realistic validation.

| Test | Description |
|------|-------------|
| `test_model_loads_successfully` | Model loads without errors |
| `test_tokenizer_loads_successfully` | Tokenizer loads correctly |
| `test_model_on_correct_device` | Model is on expected device (GPU/CPU) |
| `test_model_in_eval_mode` | Model is set to eval mode |
| `test_model_config_accessible` | Config attributes accessible |
| `test_encode_simple_text` | Tokenizer encodes text |
| `test_decode_tokens` | Tokenizer decodes back to text |
| `test_tokenizer_special_tokens` | Special tokens defined |
| `test_batch_encoding` | Batch encoding works |
| `test_simple_forward_pass` | Forward pass produces logits |
| `test_small_generation` | Model generates tokens |
| `test_math_inference` | Math reasoning test |
| `test_config_has_model_type` | model_type in config |
| `test_config_has_architectures` | architectures in config |
| `test_text_only_model_no_vision_config` | No vision_config for text model |
| `test_text_only_model_no_audio_config` | No audio_config for text model |
| `test_detect_on_real_text_model` | detect_modalities on real model |
| `test_model_type_is_qwen` | Detected as Qwen model |

#### `test_massive_model_sli.py` (2 tests, `tests/nexus_final/`)

Tests the Sequential Layer Ingestion (SLI) mechanism for handling models larger than VRAM.

| Test | Description |
|------|-------------|
| `test_layer_weight_loader_reassembly` | Verifies sharded weight files are reassembled correctly |
| `test_memory_integration_logic` | Verifies SLI activates when `should_use_sli` returns True |

---

#### `test_multimodal_encoders.py` (23 tests)

Tests encoder/decoder interfaces and configuration.

| Test Class | Description |
|------------|-------------|
| `TestVisionEncoderIntegration` | VisionEncoder import and default paths |
| `TestAudioEncoderIntegration` | AudioEncoder import and default paths |
| `TestProjectorShapes` | nn.Linear projector dimensions and forward pass |
| `TestPerceiverResampler` | Sequence reduction with Perceiver |
| `TestModularMultimodalWrapper` | Wrapper integration |
| `TestDecoderInterfaces` | Image/Audio/Video/Omni decoder methods |
| `TestEncodersConfig` | encoders.yaml validation |

---

### 3. E2E Tests (`tests/e2e/`)

Full pipeline validation with real orchestrator execution.

#### `test_orchestrator_pipeline.py` (12 tests)

| Test | Description |
|------|-------------|
| `test_script_exists` | nexus.sh CLI exists |
| `test_script_is_executable` | CLI has execute permission |
| `test_help_output` | --help shows usage |
| `test_help_lists_all_capabilities` | All 12 capabilities in help |
| `test_detect_modalities_for_text_model` | Detection returns correct flags |
| `test_capability_validation_rejects_invalid` | Invalid combos rejected |
| `test_capability_validation_accepts_valid` | Valid combos accepted |
| `test_enable_all_text_expands` | --enable-all-text flag expansion |
| `test_enable_full_omni_expands` | --enable-full-omni flag expansion |
| `test_training_order_from_registry` | Order determined correctly |
| `test_omni_comes_before_multimodal` | Omni stage first |
| `test_full_detection_to_validation_flow` | Complete flow test |

#### `test_pipeline_execution.py` (9 tests)

| Test | Description |
|------|-------------|
| `test_scenario_text_only_capabilities` | Run with CoT enabled |
| `test_scenario_help_runs` | Help runs without error |
| `test_scenario_validation_gates` | Modality gates block invalid |
| `test_detect_then_validate_flow` | Detection → Validation |
| `test_training_controller_integration` | Controller instantiation |
| `test_log_directory_creation` | Log dir created |
| `test_checkpoint_directory_creation` | Checkpoint dir created |
| `test_orchestrator_creates_log_dir` | Script creates logs |

---

## Test Fixtures (`tests/conftest.py`)

Shared fixtures for all tests.

| Fixture | Scope | Description |
|---------|-------|-------------|
| `device` | session | `"cuda"` or `"cpu"` |
| `has_gpu` | session | Boolean GPU availability |
| `text_model_path` | session | `/mnt/e/data/models/Qwen2.5-0.5B` |
| `omni_model_path` | session | Path to Qwen2.5-Omni-7B |
| `vision_encoder_path` | session | `/mnt/e/data/encoders/vision-encoders/...` |
| `audio_encoder_path` | session | `/mnt/e/data/encoders/audio-encoders/...` |
| `real_text_model` | session | Loaded Qwen2.5-0.5B model |
| `real_text_tokenizer` | session | Loaded tokenizer |
| `real_model_and_tokenizer` | session | Combined dict |
| `encoders_config` | session | Loaded encoders.yaml |
| `sample_text_prompt` | function | "What is 2 + 2?" |
| `sample_messages` | function | Chat message list |
| `temp_output_dir` | function | Temporary output directory |
| `temp_checkpoint_dir` | function | Temporary checkpoint directory |

---

## Custom Markers

```python
@pytest.mark.slow       # Long-running tests
@pytest.mark.gpu        # GPU required
@pytest.mark.real_model # Uses real model inference
```

**Run without slow tests:**

```bash
pytest tests/ -v -m "not slow"
```

**Run only GPU tests:**

```bash
pytest tests/ -v -m "gpu"
```

---

## Commands Reference

```bash
# Run all tests
pytest tests/unit/ tests/integration/ tests/e2e/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test file
pytest tests/unit/test_detect_modalities.py -v

# Run specific test
pytest tests/unit/test_detect_modalities.py::TestDetectModalities::test_detect_text_only_model -v

# Run tests matching pattern
pytest tests/ -k "modality" -v

# Run with detailed output
pytest tests/ -v --tb=long

# Parallel execution
pytest tests/ -v -n auto
```

---

## Adding New Tests

1. Create test file in appropriate directory
2. Import fixtures from conftest.py
3. Use appropriate markers
4. Follow naming convention `test_*.py`

Example:

```python
import pytest
from pathlib import Path

class TestNewFeature:
    @pytest.mark.real_model
    def test_feature_with_model(self, real_text_model):
        output = real_text_model.generate(...)
        assert output is not None
```

---

## 4. Optimization Tests (`tests/test_*.py`)

Comprehensive unit tests for the 8 optimization modules covering performance-critical components.

### 4.1 Async Decompression Tests (`tests/test_async_decompression.py`)

Tests for async I/O decompression with CUDA streams (nvCOMP-style).

| Test Class | Tests | Description |
|------------|-------|-------------|
| `TestAsyncDecompressionConfig` | 2 | Configuration validation |
| `TestCUDAStreamManager` | 12 | Stream management, round-robin, synchronization |
| `TestLayerBufferPool` | 10 | Buffer pooling, memory management |
| `TestAsyncDecompressor` | 20 | Async decompression, caching, stats |
| `TestStreamingLayerLoader` | 4 | Layer loading with prefetching |

**Total: 48 tests**

**Key Test Categories:**
- CUDA stream initialization and management
- Parallel decompression workflows
- Memory buffer pooling and reuse
- Cache hit/miss handling
- Error handling for missing dependencies
- Performance statistics tracking

**Usage Example:**
```python
from src.optimizations.async_decompression import (
    AsyncDecompressor,
    CUDAStreamManager,
    LayerBufferPool
)

# Test CUDA stream management
manager = CUDAStreamManager(num_streams=3)
stream = manager.get_next_stream()

# Test async decompression
decompressor = AsyncDecompressor()
tensor = decompressor.decompress_layer_async(
    "layer1", compressed_data, shape, dtype
)
```

---

### 4.2 Compression Optimization Tests (`tests/test_compression_optimized.py`)

Tests for ZSTD compression + quantization-aware compression.

| Test Class | Tests | Description |
|------------|-------|-------------|
| `TestCompressionConfig` | 2 | Configuration validation |
| `TestQuantizedTensor` | 6 | Quantization serialization/deserialization |
| `TestQuantizationCompressor` | 10 | Tensor quantization, sparsity pruning |
| `TestZSTDQuantizedCompressor` | 12 | Combined compression workflows |
| `TestOptimizedCompressor` | 5 | Tensor-level compression |

**Total: 35 tests**

**Key Test Categories:**
- ZSTD compression with/without quantization
- Group-wise quantization (4-bit, 8-bit)
- Compression ratio statistics
- Fallback behavior when ZSTD unavailable
- Model layer compression

**Usage Example:**
```python
from src.optimizations.compression_optimized import (
    QuantizationCompressor,
    ZSTDQuantizedCompressor,
    OptimizedCompressor
)

# Quantize tensor
compressor = QuantizationCompressor()
quantized = compressor.quantize(tensor, bits=8)

# Compress with ZSTD + quantization
zstd_compressor = ZSTDQuantizedCompressor()
compressed = zstd_compressor.compress(tensor)
```

---

### 4.3 Layer Fusion Tests (`tests/test_layer_fusion.py`)

Tests for attention + FFN fusion (Blackwell-style).

| Test Class | Tests | Description |
|------------|-------|-------------|
| `TestFusionConfig` | 2 | Configuration validation |
| `TestFusedQKVProjection` | 6 | QKV projection fusion |
| `TestFlashAttentionKernel` | 7 | FlashAttention implementation |
| `TestFusedFFN` | 6 | FFN fusion |
| `TestFusedAttentionFFN` | 7 | Complete fused block |
| `TestLayerFusionOptimizer` | 8 | Model fusion workflows |

**Total: 36 tests**

**Key Test Categories:**
- Fused QKV projections
- FlashAttention with/without CUDA
- Fused FFN (SwiGLU, GELU, ReLU)
- Residual connections
- Model-wide layer fusion

**Usage Example:**
```python
from src.optimizations.layer_fusion import (
    FusedAttentionFFN,
    LayerFusionOptimizer,
    FusionConfig
)

# Create fused block
block = FusedAttentionFFN(
    hidden_size=768,
    num_heads=12,
    intermediate_size=3072,
    head_dim=64
)

# Fuse entire model
optimizer = LayerFusionOptimizer(config)
fused_model = optimizer.fuse_model(model)
```

---

### 4.4 Early Exit Routing Tests (`tests/test_early_exit_routing.py`)

Tests for dynamic routing with early exits (LayerSkip, DASH).

| Test Class | Tests | Description |
|------------|-------|-------------|
| `TestDynamicRoutingConfig` | 2 | Configuration validation |
| `TestTokenRouter` | 8 | Per-token routing decisions |
| `TestDynamicLayerRouter` | 3 | Layer selection |
| `TestEarlyExitRouter` | 10 | Complete routing workflows |
| `TestAdaptiveExitLayer` | 4 | Layer wrapping |

**Total: 27 tests**

**Key Test Categories:**
- Token complexity estimation
- Exit layer prediction
- Confidence-based early exits
- Layer mask computation
- Training vs inference modes

**Usage Example:**
```python
from src.optimizations.early_exit_routing import (
    EarlyExitRouter,
    TokenRouter,
    DynamicRoutingConfig
)

router = EarlyExitRouter(
    model=model,
    num_layers=80,
    hidden_size=768
)

output, metrics = router.forward_with_routing(
    hidden_states, layers, attention_mask
)
```

---

### 4.5 Layer Pipelining Tests (`tests/test_layer_pipelining.py`)

Tests for layer pipelining with speculative execution (EasySpec, SpecPipe).

| Test Class | Tests | Description |
|------------|-------|-------------|
| `TestPipelineConfig` | 2 | Configuration validation |
| `TestStaleActivationPredictor` | 7 | Activation prediction |
| `TestSpeculativeLayerExecutor` | 10 | Speculative execution |
| `TestLayerPipeliningOptimizer` | 10 | Complete pipelining |

**Total: 29 tests**

**Key Test Categories:**
- Stale activation prediction
- Confidence-based speculation
- Hit/miss rate tracking
- Layer extraction from models
- Performance estimation

**Usage Example:**
```python
from src.optimizations.layer_pipelining import (
    LayerPipeliningOptimizer,
    StaleActivationPredictor,
    PipelineConfig
)

optimizer = LayerPipeliningOptimizer(
    model=model,
    num_layers=80,
    hidden_size=768
)

output, metrics = optimizer.forward(hidden_states)
```

---

### 4.6 Adaptive Layer Skipping Tests (`tests/test_adaptive_layer_skipping.py`)

Tests for adaptive layer skipping (SWIFT, LayerSkip, AdaSkip).

| Test Class | Tests | Description |
|------------|-------|-------------|
| `TestLayerSkipConfig` | 2 | Configuration validation |
| `TestLayerSkipRouter` | 6 | Skip decision logic |
| `TestSWIFTSkipper` | 5 | SWIFT-style skipping |
| `TestAdaptiveLayerSkipper` | 9 | Complete skipping workflows |
| `TestLayerSkipIntegration` | 4 | Model integration |

**Total: 26 tests**

**Key Test Categories:**
- Skip vs execute decisions
- Layer importance estimation
- Training vs inference modes
- Skip pattern strategies (adaptive, uniform, early)
- Speedup measurement

**Usage Example:**
```python
from src.optimizations.adaptive_layer_skipping import (
    AdaptiveLayerSkipper,
    LayerSkipRouter,
    LayerSkipConfig
)

skipper = AdaptiveLayerSkipper(
    model=model,
    num_layers=80,
    hidden_size=768
)

output, metrics = skipper.forward_with_skipping(
    hidden_states, layers, attention_mask
)
```

---

### 4.7 Semi-Autoregressive Decoding Tests (`tests/test_semi_autoregressive.py`)

Tests for parallel token generation (SPACE, Medusa).

| Test Class | Tests | Description |
|------------|-------|-------------|
| `TestSARConfig` | 2 | Configuration validation |
| `TestParallelTokenHead` | 8 | Parallel token heads |
| `TestSPACEDecoder` | 12 | Complete decoding workflows |
| `TestSemiAutoregressiveDecoder` | 4 | Decoder integration |

**Total: 26 tests**

**Key Test Categories:**
- Multi-token prediction
- Top-k and top-p filtering
- Temperature scaling
- Token verification
- Acceptance rate metrics

**Usage Example:**
```python
from src.optimizations.semi_autoregressive import (
    SemiAutoregressiveDecoder,
    SPACEDecoder,
    SARConfig
)

decoder = SemiAutoregressiveDecoder(base_model=model)
output = decoder.generate(input_ids, max_new_tokens=100)
```

---

## Optimization Test Summary

| Module | Tests | Coverage Target | Key Focus |
|--------|-------|----------------|-----------|
| async_decompression.py | 48 | 80%+ | CUDA streams, parallel I/O |
| compression_optimized.py | 35 | 80%+ | ZSTD + quantization |
| layer_fusion.py | 36 | 80%+ | Kernel fusion |
| early_exit_routing.py | 27 | 80%+ | Dynamic routing |
| layer_pipelining.py | 29 | 80%+ | Speculative execution |
| adaptive_layer_skipping.py | 26 | 80%+ | Layer skipping |
| semi_autoregressive.py | 26 | 80%+ | Parallel decoding |
| **Total** | **227** | **80%** | |

---

## Running Optimization Tests

```bash
# Run all optimization tests
pytest tests/test_async_decompression.py tests/test_compression_optimized.py \
       tests/test_layer_fusion.py tests/test_early_exit_routing.py \
       tests/test_layer_pipelining.py tests/test_adaptive_layer_skipping.py \
       tests/test_semi_autoregressive.py -v

# Run specific module tests
pytest tests/test_async_decompression.py -v

# Run with coverage
pytest tests/test_*.py --cov=src.optimizations --cov-report=html

# Run only fast tests (exclude integration-like tests)
pytest tests/test_*.py -v -m "unit"
```

---

## Test Patterns

All optimization tests follow consistent patterns:

### Configuration Tests
```python
class TestModuleConfig:
    def test_default_config_values(self):
        """Test default configuration."""
        config = ModuleConfig()
        assert config.key_param == expected_default

    def test_custom_config_values(self):
        """Test custom configuration."""
        config = ModuleConfig(key_param=custom_value)
        assert config.key_param == custom_value
```

### Component Tests
```python
class TestModuleComponent:
    def test_component_initialization(self):
        """Test component creation."""
        component = ModuleComponent(param=value)
        assert component.param == value

    def test_component_forward(self):
        """Test component functionality."""
        input_tensor = torch.randn(batch, seq, hidden)
        output = component(input_tensor)
        assert output.shape == input_tensor.shape
```

### Integration Tests
```python
class TestModuleIntegration:
    def test_end_to_end_workflow(self):
        """Test complete workflow."""
        optimizer = ModuleOptimizer(model=model)
        result = optimizer.process(input_data)
        assert result.success is True
```
