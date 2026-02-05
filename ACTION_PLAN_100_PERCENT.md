# ACTION PLAN: Completing Nexus to 100%
## Priority-Based Implementation Roadmap

---

## PHASE 1: CRITICAL FIXES (Week 1)

### 1.1 Fix Speculative Decoding Integration 🔴
**File**: `src/nexus/models/speculative_decoding.py:187-194`

**Current State**:
```python
# TODO: This requires UniversalSLIIntegrator to have a 'forward' or 'get_logits'
if not hasattr(self.target, 'forward_logits'):
     vocab_size = self.target.model_info['vocab_size']
     return torch.randn(1, input_ids.shape[1], vocab_size, device=self.device)
```

**Required Action**:
Add `forward_logits` method to `UniversalSLIIntegrator` or update SpeculativeDecoder to use existing methods.

**Implementation**:
```python
# In src/nexus/models/sli/universal_sli_integrator.py

def forward_logits(self, input_ids: torch.Tensor, **kwargs) -> torch.Tensor:
    """
    Run inference and return logits for the input sequence.
    Used by speculative decoding for verification.
    """
    # Run through SLI pipeline
    hidden_states = self.run_inference_pipeline(input_ids, **kwargs)
    
    # Apply final layer norm and LM head
    if hasattr(self, 'model'):
        hidden_states = self.model.model.norm(hidden_states)
        logits = self.model.lm_head(hidden_states)
    else:
        # Fallback: compute logits from hidden states
        logits = self._compute_logits(hidden_states)
    
    return logits
```

---

### 1.2 Remove README.md Duplication 🔴
**File**: `README.md` Lines 154-393

**Problem**: Entire sections duplicated
- "Capability Tier Declaration" appears 2x
- "New in v6.1" section duplicated
- "Key Features" section duplicated

**Action**: Delete duplicate sections (lines 154-393)

---

### 1.3 Fix Performance Claims 🔴
**File**: `🚀 ACHIEVING 100 TOKENS_SECOND_ Comprehensive Optim.md`

**Problem**: Filename and content claim 100 tok/s
**Reality**: 3-8 tok/s for 70B+ models with SLI

**Actions**:
1. Rename file to reflect reality: `OPTIMIZING_FOR_3-8_TOKENS_PER_SECOND.md`
2. Add disclaimer about TensorRT vs SLI differences
3. Update content with realistic benchmarks

---

### 1.4 Replace Critical Placeholder Tests 🔴

#### Test 1: `tests/unit/test_training_methods.py:206`
**Current**:
```python
assert True  # Placeholder for full integration test
```

**Required**: Real integration test

#### Test 2: `tests/unit/test_training_controller.py:70`
**Current**:
```python
assert True  # If we get here, it didn't block
```

**Required**: Actual controller validation

#### Test 3: `tests/integration/test_reasoning_pipeline.py`
**Lines**: 99, 118, 137
**Current**: Three `assert True` statements
**Required**: Validate reasoning pipeline outputs

---

## PHASE 2: HIGH PRIORITY (Weeks 2-3)

### 2.1 Implement Empty Methods in Resilience Module 🟠
**File**: `src/nexus/core/resilience.py`

**Lines with `pass`**: 66, 221, 234, 307, 395, 400, 472

**Actions**:
```python
# Line 66 - Add circuit breaker fallback
def fallback(self, *args, **kwargs):
    """Execute fallback logic when circuit is open."""
    logger.warning(f"Circuit {self.name} is open, executing fallback")
    if self.fallback_function:
        return self.fallback_function(*args, **kwargs)
    raise CircuitBreakerOpen(f"Circuit {self.name} is open")

# Line 221 - Add retry logic
def execute_with_retry(self, func, *args, **kwargs):
    """Execute function with retry logic."""
    for attempt in range(self.max_retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if attempt == self.max_retries - 1:
                raise
            time.sleep(self.backoff_factor ** attempt)
```

---

### 2.2 Implement Error Handlers 🟠

#### File: `src/nexus/models/data_loader.py:453`
```python
# Current:
except: pass

# Required:
except Exception as e:
    logger.error(f"Failed to load batch: {e}")
    raise DataLoaderError(f"Batch loading failed: {e}") from e
```

#### File: `src/nexus/monitoring/collectors.py`
**Lines**: 67, 72, 491, 498, 545, 552, 560

Add proper fallback metrics collection

---

### 2.3 Add Tests for Critical SLI Components 🟠

#### Component 1: `activation_cache.py`
**Test File**: `tests/unit/sli/test_activation_cache.py`
```python
def test_cache_store_and_retrieve():
    cache = ActivationCache(max_size=100)
    tensor = torch.randn(1, 10, 512)
    cache.store("layer_0", tensor)
    retrieved = cache.retrieve("layer_0")
    assert torch.equal(tensor, retrieved)

def test_cache_eviction():
    cache = ActivationCache(max_size=2)
    cache.store("layer_0", torch.randn(1, 10, 512))
    cache.store("layer_1", torch.randn(1, 10, 512))
    cache.store("layer_2", torch.randn(1, 10, 512))  # Should evict layer_0
    assert cache.retrieve("layer_0") is None
```

#### Component 2: `io_optimizer.py`
**Test File**: `tests/unit/sli/test_io_optimizer.py`
- Test prefetch logic
- Test async loading
- Test buffer management

#### Component 3: `nested_learning.py`
**Test File**: `tests/unit/sli/test_nested_learning.py`
- Test nested scheduler
- Test hierarchical cache
- Test level transitions

---

### 2.4 Fix Import Paths in Documentation 🟠

**Scan all docs for incorrect paths**:
- `nexus.core.sli` → `src.nexus.models.sli`
- `nexus_final` → `nexus`
- `src.nexus_final` → `src.nexus`

**Files to Update**:
1. `docs/NEXUS_V6_TECHNICAL_MANUAL.md`
2. `docs/NEXUS_USAGE_GUIDE.md`
3. `docs/OMNI_LOADER_GUIDE.md`
4. All code examples in README.md

---

## PHASE 3: MEDIUM PRIORITY (Weeks 4-6)

### 3.1 Add Functional Tests for Stage Classes 🟡

All stages need real functional tests, not just import tests.

#### Example: CoT Stage Test
**File**: `tests/unit/stages/test_cot_functional.py`
```python
def test_cot_stage_training_loop():
    """Test actual training loop execution."""
    config = StageConfig(
        capability_name="cot",
        base_model_path="gpt2",  # Small model for testing
        output_dir="/tmp/test_cot"
    )
    stage = CoTStage(config)
    
    # Prepare
    assert stage.prepare() == True
    
    # Train (1 step)
    results = stage.train()
    assert results["success"] == True
    assert "loss" in results
    
    # Save
    checkpoint_path = stage.save_checkpoint()
    assert Path(checkpoint_path).exists()
```

**Stages to Cover**:
- [ ] CoTStage
- [ ] ReasoningStage
- [ ] ToolsStage
- [ ] ThinkingStage
- [ ] StreamingStage
- [ ] VideoUnderstandingStage
- [ ] VisionQAStage
- [ ] ImageGenStage
- [ ] VideoGenStage
- [ ] PodcastStage
- [ ] RemotionGenStage
- [ ] TriStreamingStage
- [ ] OmniTrainingStage

---

### 3.2 Implement TensorRT Backend 🟡
**Files**:
- `src/nexus/models/tensorrt/inference_backend.py`
- `src/nexus/models/tensorrt/model_converter.py`
- `src/nexus/models/tensorrt/trt_engine.py`

**Actions**:
Replace all `pass` statements with actual TensorRT integration:

```python
# inference_backend.py
class TensorRTBackend:
    def __init__(self, engine_path: str):
        import tensorrt as trt
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        with open(engine_path, 'rb') as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
    
    def infer(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # Actual TensorRT inference
        pass  # Implement
```

---

### 3.3 Add "When NOT to Use SLI" Documentation 🟡
**New File**: `docs/WHEN_NOT_TO_USE_SLI.md`

**Content**:
```markdown
# When NOT to Use SLI

## I/O Bottleneck Limitations
SLI (Streaming Layer Inference) is NOT recommended when:
- Your storage read speed < 2GB/s
- You're using HDD instead of NVMe SSD
- Network storage without high-speed connection

## Hardware Requirements
Minimum for SLI:
- NVMe SSD with 3GB/s+ read speed
- 32GB+ RAM
- CUDA-capable GPU with 8GB+ VRAM

## Better Alternatives
- Use full model loading if you have sufficient VRAM
- Use quantization (GGUF, NF4) for memory reduction
- Use model parallelism across multiple GPUs

## Research Use Only
SLI is primarily for research and development.
Not recommended for production inference without extensive testing.
```

---

### 3.4 Create Architecture Decision Records 🟡
**New Directory**: `docs/architecture/`

**Files to Create**:
1. `ADR-001-universal-sli-architecture.md`
2. `ADR-002-multi-agent-orchestration.md`
3. `ADR-003-speculative-decoding-integration.md`
4. `ADR-004-omni-loader-design.md`

---

## PHASE 4: LOW PRIORITY (Ongoing)

### 4.1 Add Tests for Training Scripts 🟢
**Directory**: `src/nexus/training/scripts/`

All 27 training scripts need tests:
- [ ] `10_sft_training.py`
- [ ] `11_continued_pretraining.py`
- [ ] `12_grpo_training.py`
- [ ] ... (24 more)

### 4.2 Standardize Model Support Claims 🟢
**Update all docs to say**:
> "16 architecture families supported, covering 135+ potential model variants. Not all variants individually tested."

### 4.3 Add Property-Based Testing 🟢
For complex SLI components:
```python
from hypothesis import given, strategies as st

@given(st.integers(min_value=1, max_value=100))
def test_sli_with_various_layer_counts(num_layers):
    # Test SLI with different layer counts
    pass
```

---

## VERIFICATION SCRIPT

Create automated verification:

```python
#!/usr/bin/env python3
"""Verify codebase completeness."""

import subprocess
import re

def check_todos():
    """Check for remaining TODOs."""
    result = subprocess.run(
        ['grep', '-r', 'TODO', 'src/', '--include=*.py'],
        capture_output=True, text=True
    )
    todos = result.stdout.strip().split('\n')
    print(f"❌ Found {len(todos)} TODOs" if todos else "✅ No TODOs found")
    return len(todos) == 0

def check_pass_statements():
    """Check for empty pass statements."""
    result = subprocess.run(
        ['grep', '-r', '^\\s*pass\\s*$', 'src/', '--include=*.py'],
        capture_output=True, text=True
    )
    passes = [l for l in result.stdout.strip().split('\n') if l]
    print(f"❌ Found {len(passes)} empty pass statements" if passes else "✅ No empty pass statements")
    return len(passes) == 0

def check_placeholder_tests():
    """Check for placeholder tests."""
    result = subprocess.run(
        ['grep', '-r', 'assert True.*Placeholder', 'tests/', '--include=*.py'],
        capture_output=True, text=True
    )
    placeholders = result.stdout.strip().split('\n')
    print(f"❌ Found {len(placeholders)} placeholder tests" if placeholders else "✅ No placeholder tests")
    return len(placeholders) == 0

if __name__ == "__main__":
    all_good = all([
        check_todos(),
        check_pass_statements(),
        check_placeholder_tests()
    ])
    
    if all_good:
        print("\n✅ Codebase is 100% complete!")
    else:
        print("\n❌ Codebase needs work. See issues above.")
```

---

## SUMMARY

| Phase | Duration | Critical Items |
|-------|----------|----------------|
| Phase 1 | Week 1 | 4 items (speculative decoding, README, performance claims, placeholder tests) |
| Phase 2 | Weeks 2-3 | 4 items (resilience, error handlers, SLI tests, import paths) |
| Phase 3 | Weeks 4-6 | 4 items (stage tests, TensorRT, documentation, ADRs) |
| Phase 4 | Ongoing | 3 items (training script tests, standardization, property tests) |

**Total Critical Items**: 15
**Estimated Time to 100%**: 4-6 weeks
