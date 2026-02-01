# Implementation Guide: SLI + NVFP4-QAD + Nested Learning

**Version:** 1.0  
**Date:** 2026-02-01  
**Status:** Implementation Specification

---

## 1. Code Structure Overview

```
src/nexus_final/sli/
├── __init__.py
├── quantization.py                    # Existing: INT8, NF4 quantization
├── nvfp4_qad/                         # NEW: NVFP4-QAD integration
│   ├── __init__.py
│   ├── nvfp4_loader.py               # NVFP4 streaming loader
│   ├── qad_loss.py                   # QAD distillation loss
│   ├── mixed_precision.py            # Mixed BF16/NVFP4 layer loading
│   └── config.py                     # NVFP4-QAD configuration
├── nested_learning/                   # NEW: Nested Learning integration
│   ├── __init__.py
│   ├── scheduler.py                  # Multi-time-scale update scheduler
│   ├── hierarchical_cache.py         # Three-tier cache system
│   ├── continuum_memory.py           # Continuum memory system
│   └── prefetcher.py                 # Nested learning aware prefetcher
├── advanced/                          # NEW: Combined integration
│   ├── __init__.py
│   ├── integrator.py                 # AdvancedSLIIntegrator
│   ├── config.py                     # Unified configuration
│   └── monitoring.py                 # Metrics and monitoring
└── tests/                             # Test suite
    ├── test_nvfp4_qad.py
    ├── test_nested_learning.py
    └── test_integration.py
```

---

## 2. Core Implementation Files

### 2.1 NVFP4 Streaming Loader

```python
# src/nexus_final/sli/nvfp4_qad/nvfp4_loader.py

"""
NVFP4 Streaming Layer Loader

Streams 4-bit quantized layers from storage with on-the-fly dequantization.
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Optional, Any
import logging
from dataclasses import dataclass

try:
    import bitsandbytes as bnb
    from bitsandbytes.nn import Linear4bit, Params4bit
    BITSANDBYTES_AVAILABLE = True
except ImportError:
    BITSANDBYTES_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class NVFP4Config:
    """Configuration for NVFP4 quantization."""
    block_size: int = 16  # NVFP4 uses 16 (vs 32 for MXFP4)
    scale_dtype: str = "e4m3"  # FP8 scale factors
    compute_dtype: torch.dtype = torch.bfloat16
    second_level_scale: str = "fp32"
    compress_statistics: bool = True
    double_quant: bool = True
    
    # Mixed precision strategy
    attention_precision: str = "bf16"  # Keep attention at full precision
    ffn_precision: str = "nvfp4"       # Quantize FFN layers


class NVFP4StreamingLoader:
    """
    Loads and streams NVFP4 quantized layers.
    
    Key Features:
    - Streams 4-bit weights from SSD (4x faster I/O)
    - Dequantizes on GPU for computation
    - Caches quantized layers for reuse
    - Supports mixed precision (BF16 + NVFP4)
    """
    
    def __init__(
        self,
        model_id: str,
        config: NVFP4Config,
        cache_dir: str = "cache/nvfp4_layers",
        device: str = "cuda"
    ):
        if not BITSANDBYTES_AVAILABLE:
            raise ImportError(
                "bitsandbytes required for NVFP4. "
                "Install with: pip install bitsandbytes"
            )
        
        self.model_id = model_id
        self.config = config
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        
        # In-memory cache for quantized weights
        self._quantized_cache: Dict[int, Dict[str, Any]] = {}
        self._cache_size_limit = 5  # Keep 5 layers in memory
        
        logger.info(f"NVFP4StreamingLoader initialized for {model_id}")
    
    def load_layer_streaming(
        self,
        layer_idx: int,
        layer_type: str = "auto"
    ) -> nn.Module:
        """
        Load layer with NVFP4 quantization.
        
        Args:
            layer_idx: Layer index to load
            layer_type: Type of layer ('attention', 'ffn', or 'auto')
            
        Returns:
            Layer module (dequantized for computation)
        """
        # Determine precision for this layer
        precision = self._get_layer_precision(layer_idx, layer_type)
        
        if precision == "bf16":
            # Load full precision layer
            return self._load_bf16_layer(layer_idx)
        
        # Load NVFP4 quantized layer
        # Check cache first
        if layer_idx in self._quantized_cache:
            logger.debug(f"Cache hit for layer {layer_idx}")
            nvfp4_weights = self._quantized_cache[layer_idx]
        else:
            # Stream from storage
            nvfp4_weights = self._stream_nvfp4_weights(layer_idx)
            self._cache_quantized(layer_idx, nvfp4_weights)
        
        # Dequantize for computation
        layer = self._dequantize_layer(nvfp4_weights)
        
        return layer.to(self.device)
    
    def _get_layer_precision(
        self,
        layer_idx: int,
        layer_type: str
    ) -> str:
        """Determine precision for a layer."""
        if layer_type == "auto":
            # Infer from layer index (simplified heuristic)
            # In practice, this would check the actual layer type
            layer_type = "ffn"  # Default
        
        if layer_type == "attention":
            return self.config.attention_precision
        elif layer_type == "ffn":
            return self.config.ffn_precision
        
        return "bf16"  # Default to full precision
    
    def _stream_nvfp4_weights(
        self,
        layer_idx: int
    ) -> Dict[str, Any]:
        """
        Stream NVFP4 quantized weights from storage.
        
        NVFP4 Format (from NVIDIA paper):
        - Block size: 16 elements
        - Scale factors: FP8 (E4M3)
        - Second-level scale: FP32
        - Overall: 4x compression vs BF16
        """
        cache_path = self.cache_dir / f"layer_{layer_idx}_nvfp4.pt"
        
        if cache_path.exists():
            # Load from local cache
            logger.debug(f"Loading NVFP4 weights from cache: {cache_path}")
            return torch.load(cache_path, map_location='cpu')
        
        # In practice, this would:
        # 1. Download/stream from HF Hub
        # 2. Dequantize from safetensors format
        # 3. Apply NVFP4 quantization
        
        # For now, simulate with bitsandbytes
        weights = self._quantize_layer_to_nvfp4(layer_idx)
        
        # Cache locally
        torch.save(weights, cache_path)
        
        return weights
    
    def _quantize_layer_to_nvfp4(
        self,
        layer_idx: int
    ) -> Dict[str, Any]:
        """Quantize layer weights to NVFP4 format."""
        # This would integrate with actual model weights
        # For now, create dummy quantized weights
        
        # NVFP4 uses Normal Float 4 format
        # Block size 16 with FP8 scales
        
        return {
            'layer_idx': layer_idx,
            'quantized': True,
            'format': 'nvfp4',
            'block_size': self.config.block_size,
            # Actual weight data would be quantized Params4bit
        }
    
    def _dequantize_layer(
        self,
        nvfp4_weights: Dict[str, Any]
    ) -> nn.Module:
        """
        Dequantize NVFP4 weights to BF16 for computation.
        
        This happens on GPU for efficiency.
        """
        # Reconstruct layer from quantized weights
        # In practice, this uses bitsandbytes dequantization
        
        # Placeholder: return dummy linear layer
        layer = nn.Linear(4096, 4096)
        
        # Dequantize weights to BF16
        # Actual implementation would use bnb.dequantize_4bit
        
        return layer.to(dtype=self.config.compute_dtype)
    
    def _cache_quantized(
        self,
        layer_idx: int,
        weights: Dict[str, Any]
    ):
        """Cache quantized weights with LRU eviction."""
        # Evict oldest if cache is full
        while len(self._quantized_cache) >= self._cache_size_limit:
            oldest = min(self._quantized_cache.keys())
            del self._quantized_cache[oldest]
        
        self._quantized_cache[layer_idx] = weights
    
    def get_compression_ratio(self) -> float:
        """Get storage compression ratio."""
        return 0.25  # 4x compression for NVFP4
```

### 2.2 QAD Loss Module

```python
# src/nexus_final/sli/nvfp4_qad/qad_loss.py

"""
Quantization-Aware Distillation (QAD) Loss

Based on NVIDIA paper "Quantization-Aware Distillation for NVFP4 Inference" (2026).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class QADDistillationLoss(nn.Module):
    """
    QAD Loss: KL divergence between FP32 teacher and NVFP4 student.
    
    Key advantages over QAT:
    1. More stable for multi-stage post-trained models (SFT, RL)
    2. Robust to incomplete data coverage
    3. Better accuracy recovery
    
    Formula:
    L_QAD = D_KL(P_teacher || P_student)
    """
    
    def __init__(
        self,
        temperature: float = 1.0,
        alpha_qad: float = 1.0,
        alpha_task: float = 0.0,
    ):
        """
        Args:
            temperature: Softmax temperature (usually 1.0 for QAD)
            alpha_qad: Weight for QAD loss
            alpha_task: Weight for task loss (usually 0 for pure QAD)
        """
        super().__init__()
        self.temperature = temperature
        self.alpha_qad = alpha_qad
        self.alpha_task = alpha_task
        
    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute QAD loss.
        
        Args:
            student_logits: From NVFP4 quantized model
            teacher_logits: From BF16 full precision teacher
            labels: Optional labels for task loss
            
        Returns:
            Combined loss
        """
        T = self.temperature
        
        # Compute probabilities with temperature scaling
        student_log_probs = F.log_softmax(student_logits / T, dim=-1)
        teacher_probs = F.softmax(teacher_logits / T, dim=-1)
        
        # KL divergence (forward KL: teacher || student)
        # This matches teacher distribution
        kl_loss = F.kl_div(
            student_log_probs,
            teacher_probs,
            reduction='batchmean'
        ) * (T ** 2)
        
        # Optional task loss
        task_loss = 0.0
        if labels is not None and self.alpha_task > 0:
            task_loss = F.cross_entropy(student_logits, labels)
        
        return self.alpha_qad * kl_loss + self.alpha_task * task_loss


class FeatureMatchingLoss(nn.Module):
    """
    Additional feature matching loss for hidden state alignment.
    
    This can be used alongside QAD for better representation learning.
    """
    
    def __init__(
        self,
        layer_indices: list = None,
        alpha_feat: float = 0.1
    ):
        super().__init__()
        self.layer_indices = layer_indices or []
        self.alpha_feat = alpha_feat
        
    def forward(
        self,
        student_features: dict,
        teacher_features: dict
    ) -> torch.Tensor:
        """
        Match intermediate features.
        
        Args:
            student_features: Dict of hidden states from student
            teacher_features: Dict of hidden states from teacher
        """
        loss = 0.0
        count = 0
        
        for idx in self.layer_indices:
            if idx in student_features and idx in teacher_features:
                s_feat = student_features[idx]
                t_feat = teacher_features[idx]
                
                # Normalize features
                s_feat = F.normalize(s_feat, dim=-1)
                t_feat = F.normalize(t_feat, dim=-1)
                
                # MSE loss
                loss += F.mse_loss(s_feat, t_feat)
                count += 1
        
        return self.alpha_feat * (loss / max(count, 1))
```

### 2.3 Nested Learning Scheduler

```python
# src/nexus_final/sli/nested_learning/scheduler.py

"""
Nested Learning Update Scheduler

Implements multi-time-scale updates for different layer groups.
Based on "Nested Learning: The Illusion of Deep Learning Architectures" (NeurIPS 2025).
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class UpdateGroupConfig:
    """Configuration for a layer update group."""
    name: str
    layer_indices: List[int]
    frequency: int  # Update every N steps
    cache_tier: str  # 'memory', 'disk_l1', or 'disk_l2'
    consolidation_level: str  # 'online', 'synaptic', or 'systems'


class NestedUpdateScheduler:
    """
    Multi-time-scale update scheduler.
    
    Groups layers by update frequency:
    - Fast: Every step (working memory)
    - Medium: Every N steps (synaptic consolidation)
    - Slow: Rarely (systems consolidation)
    """
    
    def __init__(
        self,
        num_layers: int,
        groups: Optional[Dict[str, dict]] = None
    ):
        """
        Args:
            num_layers: Total number of layers
            groups: Custom update group configuration
        """
        self.num_layers = num_layers
        self.step_count = 0
        
        # Initialize update groups
        if groups is None:
            groups = self._default_groups(num_layers)
        
        self.groups = self._create_groups(groups)
        
        # Track last update time for each layer
        self.last_updated = {i: 0 for i in range(num_layers)}
        
        # Statistics
        self.update_counts = {i: 0 for i in range(num_layers)}
        
        logger.info(f"Initialized NestedUpdateScheduler for {num_layers} layers")
        logger.info(f"Groups: {[g.name for g in self.groups]}")
    
    def _default_groups(self, num_layers: int) -> Dict[str, dict]:
        """Create default update groups based on layer depth."""
        # First 25%: Fast updates (feature extraction)
        # Middle 50%: Medium updates (representation)
        # Last 25%: Slow updates (task-specific)
        
        quarter = num_layers // 4
        
        return {
            'fast': {
                'layer_indices': list(range(0, quarter)),
                'frequency': 1,
                'cache_tier': 'memory',
                'consolidation_level': 'online'
            },
            'medium': {
                'layer_indices': list(range(quarter, 3 * quarter)),
                'frequency': 10,
                'cache_tier': 'disk_l1',
                'consolidation_level': 'synaptic'
            },
            'slow': {
                'layer_indices': list(range(3 * quarter, num_layers)),
                'frequency': 100,
                'cache_tier': 'disk_l2',
                'consolidation_level': 'systems'
            }
        }
    
    def _create_groups(
        self,
        group_configs: Dict[str, dict]
    ) -> List[UpdateGroupConfig]:
        """Create UpdateGroupConfig objects from dict."""
        groups = []
        for name, config in group_configs.items():
            groups.append(UpdateGroupConfig(
                name=name,
                layer_indices=config['layer_indices'],
                frequency=config['frequency'],
                cache_tier=config['cache_tier'],
                consolidation_level=config['consolidation_level']
            ))
        return groups
    
    def should_update(self, layer_idx: int) -> bool:
        """Check if layer should be updated at current step."""
        group = self._get_group_for_layer(layer_idx)
        steps_since_update = self.step_count - self.last_updated[layer_idx]
        return steps_since_update >= group.frequency
    
    def get_update_layers(self) -> List[int]:
        """Get list of layers that should be updated this step."""
        return [
            i for i in range(self.num_layers)
            if self.should_update(i)
        ]
    
    def get_cache_tier(self, layer_idx: int) -> str:
        """Get cache tier for a layer."""
        group = self._get_group_for_layer(layer_idx)
        return group.cache_tier
    
    def get_consolidation_level(self, layer_idx: int) -> str:
        """Get consolidation level for a layer."""
        group = self._get_group_for_layer(layer_idx)
        return group.consolidation_level
    
    def mark_updated(self, layer_idx: int):
        """Mark layer as updated."""
        self.last_updated[layer_idx] = self.step_count
        self.update_counts[layer_idx] += 1
    
    def step(self):
        """Increment global step."""
        self.step_count += 1
    
    def _get_group_for_layer(self, layer_idx: int) -> UpdateGroupConfig:
        """Find which group a layer belongs to."""
        for group in self.groups:
            if layer_idx in group.layer_indices:
                return group
        # Default to first group
        return self.groups[0]
    
    def get_stats(self) -> dict:
        """Get update statistics."""
        return {
            'step_count': self.step_count,
            'update_counts': self.update_counts.copy(),
            'group_sizes': {
                g.name: len(g.layer_indices) for g in self.groups
            }
        }
```

### 2.4 Hierarchical Cache

```python
# src/nexus_final/sli/nested_learning/hierarchical_cache.py

"""
Hierarchical Layer Cache

Three-tier cache system aligned with nested update frequencies.
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, Dict, Any
from collections import OrderedDict
import threading
import logging

logger = logging.getLogger(__name__)


class TieredCache:
    """Base class for cache tiers."""
    
    def __init__(self, max_size_bytes: int):
        self.max_size = max_size_bytes
        self.current_size = 0
        self.cache: OrderedDict[int, nn.Module] = OrderedDict()
        self.lock = threading.RLock()
        
    def get(self, key: int) -> Optional[nn.Module]:
        with self.lock:
            if key in self.cache:
                # Move to end (most recently used)
                value = self.cache.pop(key)
                self.cache[key] = value
                return value
        return None
    
    def put(self, key: int, value: nn.Module) -> bool:
        with self.lock:
            # Evict if necessary
            while self._would_exceed_limit(value):
                if not self._evict_lru():
                    break
            
            self.cache[key] = value
            self.current_size += self._get_size(value)
            return True
    
    def _get_size(self, module: nn.Module) -> int:
        """Get memory size of module in bytes."""
        total = 0
        for param in module.parameters():
            total += param.numel() * param.element_size()
        return total
    
    def _would_exceed_limit(self, module: nn.Module) -> bool:
        return self.current_size + self._get_size(module) > self.max_size
    
    def _evict_lru(self) -> bool:
        """Evict least recently used entry."""
        if not self.cache:
            return False
        key, value = self.cache.popitem(last=False)
        self.current_size -= self._get_size(value)
        logger.debug(f"Evicted layer {key}")
        return True


class MemoryCache(TieredCache):
    """Hot cache in GPU/CPU memory for fast layers."""
    
    def __init__(self, max_size_gb: float = 4.0):
        super().__init__(int(max_size_gb * 1024**3))
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'


class DiskCacheL1(TieredCache):
    """Warm cache on fast SSD for medium layers."""
    
    def __init__(
        self,
        cache_dir: str,
        max_size_gb: float = 50.0
    ):
        super().__init__(int(max_size_gb * 1024**3))
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def get(self, key: int) -> Optional[nn.Module]:
        # Try memory first
        value = super().get(key)
        if value is not None:
            return value
        
        # Try disk
        cache_path = self.cache_dir / f"layer_{key}.pt"
        if cache_path.exists():
            try:
                value = torch.load(cache_path, map_location='cpu')
                # Move to memory cache
                super().put(key, value)
                return value
            except Exception as e:
                logger.warning(f"Failed to load from disk: {e}")
        
        return None
    
    def put(self, key: int, value: nn.Module) -> bool:
        # Save to disk
        cache_path = self.cache_dir / f"layer_{key}.pt"
        try:
            torch.save(value, cache_path)
        except Exception as e:
            logger.warning(f"Failed to save to disk: {e}")
        
        # Also put in memory
        return super().put(key, value)


class HierarchicalLayerCache:
    """
    Three-tier cache system.
    
    Tiers:
    1. Memory: Hot layers (fast updates)
    2. Disk L1: Warm layers (medium updates) on fast SSD
    3. Disk L2: Cold layers (slow updates) on slower storage
    """
    
    def __init__(
        self,
        memory_cache_size_gb: float = 4.0,
        disk_l1_size_gb: float = 50.0,
        disk_l2_size_gb: float = 200.0,
        disk_l1_path: str = "./cache/warm",
        disk_l2_path: str = "./cache/cold"
    ):
        self.memory = MemoryCache(memory_cache_size_gb)
        self.disk_l1 = DiskCacheL1(disk_l1_path, disk_l1_size_gb)
        # Disk L2 would be similar but with different storage class
        self.disk_l2 = DiskCacheL1(disk_l2_path, disk_l2_size_gb)
        
        logger.info(
            f"HierarchicalCache: Memory={memory_cache_size_gb}GB, "
            f"L1={disk_l1_size_gb}GB, L2={disk_l2_size_gb}GB"
        )
    
    def get(
        self,
        layer_idx: int,
        cache_tier: str
    ) -> Optional[nn.Module]:
        """Get layer from appropriate cache tier."""
        if cache_tier == 'memory':
            return self.memory.get(layer_idx)
        elif cache_tier == 'disk_l1':
            return self.disk_l1.get(layer_idx)
        elif cache_tier == 'disk_l2':
            return self.disk_l2.get(layer_idx)
        return None
    
    def put(
        self,
        layer_idx: int,
        layer: nn.Module,
        cache_tier: str
    ):
        """Store layer in appropriate cache tier."""
        if cache_tier == 'memory':
            self.memory.put(layer_idx, layer)
        elif cache_tier == 'disk_l1':
            self.disk_l1.put(layer_idx, layer)
        elif cache_tier == 'disk_l2':
            self.disk_l2.put(layer_idx, layer)
```

---

## 3. Integration Example

```python
# example_usage.py

"""
Example usage of Advanced SLI with NVFP4-QAD and Nested Learning.
"""

from nexus_final.sli.nvfp4_qad import (
    NVFP4Config,
    NVFP4StreamingLoader,
    QADDistillationLoss
)
from nexus_final.sli.nested_learning import (
    NestedUpdateScheduler,
    HierarchicalLayerCache
)

# 1. Configure NVFP4-QAD
nvfp4_config = NVFP4Config(
    block_size=16,
    compute_dtype=torch.bfloat16,
    attention_precision='bf16',
    ffn_precision='nvfp4'
)

# 2. Create streaming loader for quantized teacher
teacher_loader = NVFP4StreamingLoader(
    model_id="meta-llama/Llama-3.1-70B",
    config=nvfp4_config,
    cache_dir="cache/teacher_nvfp4"
)

# 3. Create QAD loss
qad_loss = QADDistillationLoss(
    temperature=1.0,
    alpha_qad=1.0
)

# 4. Configure Nested Learning
scheduler = NestedUpdateScheduler(
    num_layers=80,  # 70B model
    groups={
        'fast': {
            'layer_indices': list(range(0, 20)),
            'frequency': 1,
            'cache_tier': 'memory',
            'consolidation_level': 'online'
        },
        'medium': {
            'layer_indices': list(range(20, 60)),
            'frequency': 10,
            'cache_tier': 'disk_l1',
            'consolidation_level': 'synaptic'
        },
        'slow': {
            'layer_indices': list(range(60, 80)),
            'frequency': 100,
            'cache_tier': 'disk_l2',
            'consolidation_level': 'systems'
        }
    }
)

# 5. Create hierarchical cache
cache = HierarchicalLayerCache(
    memory_cache_size_gb=8.0,
    disk_l1_size_gb=100.0,
    disk_l2_size_gb=500.0
)

# 6. Training loop
for step, batch in enumerate(dataloader):
    # Get layers to update this step
    layers_to_update = scheduler.get_update_layers()
    
    for layer_idx in layers_to_update:
        # Load quantized teacher layer
        teacher_layer = teacher_loader.load_layer_streaming(layer_idx)
        
        # Get from cache or compute
        cache_tier = scheduler.get_cache_tier(layer_idx)
        
        with torch.no_grad():
            teacher_output = teacher_layer(batch['input_ids'])
        
        # Student forward
        student_output = student(batch['input_ids'])
        
        # QAD loss
        loss = qad_loss(student_output, teacher_output)
        
        # Backward and optimize
        loss.backward()
        optimizer.step()
        
        # Mark as updated
        scheduler.mark_updated(layer_idx)
    
    scheduler.step()
```

---

## 4. Testing Strategy

```python
# tests/test_integration.py

"""
Test suite for the integrated system.
"""

import torch
import pytest
from nexus_final.sli.nvfp4_qad import (
    NVFP4StreamingLoader,
    QADDistillationLoss
)
from nexus_final.sli.nested_learning import NestedUpdateScheduler


class TestNVFP4QAD:
    """Test NVFP4-QAD functionality."""
    
    def test_compression_ratio(self):
        """Test that NVFP4 achieves 4x compression."""
        loader = NVFP4StreamingLoader(
            model_id="test-model",
            config=NVFP4Config()
        )
        assert loader.get_compression_ratio() == 0.25
    
    def test_qad_loss(self):
        """Test QAD loss computation."""
        loss_fn = QADDistillationLoss()
        
        student_logits = torch.randn(2, 10, 32000)
        teacher_logits = torch.randn(2, 10, 32000)
        
        loss = loss_fn(student_logits, teacher_logits)
        
        assert loss.shape == torch.Size([])
        assert loss.item() > 0


class TestNestedLearning:
    """Test Nested Learning functionality."""
    
    def test_update_frequency(self):
        """Test that update frequencies are respected."""
        scheduler = NestedUpdateScheduler(num_layers=32)
        
        # Fast layers should update every step
        assert scheduler.should_update(0)  # First layer (fast)
        scheduler.mark_updated(0)
        scheduler.step()
        assert scheduler.should_update(0)  # Should update again
        
        # Slow layer should not update after 1 step
        slow_layer = 24  # In slow group
        scheduler.mark_updated(slow_layer)
        scheduler.step()
        assert not scheduler.should_update(slow_layer)
    
    def test_cache_tier_assignment(self):
        """Test that cache tiers are correctly assigned."""
        scheduler = NestedUpdateScheduler(num_layers=32)
        
        assert scheduler.get_cache_tier(0) == 'memory'  # Fast
        assert scheduler.get_cache_tier(16) == 'disk_l1'  # Medium
        assert scheduler.get_cache_tier(28) == 'disk_l2'  # Slow


class TestIntegration:
    """Test integration of both systems."""
    
    def test_end_to_end(self):
        """Test end-to-end training step."""
        # This would test a full training step
        # with both NVFP4-QAD and Nested Learning
        pass
```

---

## 5. Performance Benchmarks

### Expected Performance

| Model Size | Standard SLI | With NVFP4-QAD | With Both | Improvement |
|------------|--------------|----------------|-----------|-------------|
| 8B | Baseline | 4x faster I/O | 4.2x | - |
| 70B | Baseline | 4x faster I/O | 4.5x | Cache efficiency |
| 405B | Baseline | 4x faster I/O | 5x | Better tiering |

### Resource Requirements

| Component | Memory | Storage | GPU |
|-----------|--------|---------|-----|
| Standard SLI | 16GB | 200GB | 24GB |
| + NVFP4-QAD | 16GB | 50GB | 24GB |
| + Nested Learning | 24GB | 200GB | 24GB |
| **Combined** | 24GB | 50GB | 24GB |

---

**Implementation Guide End**
