"""
Universal SLI (Sequential Layer Ingestion) Module

This module provides universal support for 130+ model architectures
in the Sequential Layer Ingestion pipeline.

Supported Architecture Families:
- Llama-based (Llama, Mistral, Mixtral, Qwen2, etc.)
- GPT-based (GPT-2, GPT-J, GPT-NeoX, etc.)
- ChatGLM-based (ChatGLM, GLM-4, etc.)
- T5-based (T5, FLAN-T5, UL2, etc.)
- BLOOM-based
- OPT-based
- Mamba/State Space Models
- MoE Architectures (Mixtral, DeepSeek-MoE, etc.)
- Phi-based
- Gemma-based
- Encoder-only (BERT, RoBERTa, etc.)

Usage:
    from src.nexus_final.sli import UniversalSLIIntegrator
    
    integrator = UniversalSLIIntegrator("mistralai/Mistral-7B-v0.1")
    integrator.run_sli(dataset)
"""

from .architecture_registry import (
    ArchitectureRegistry,
    ArchitectureFamily,
    LlamaFamilyHandler,
    GPTFamilyHandler,
    ChatGLMFamilyHandler,
    T5FamilyHandler,
    BLOOMFamilyHandler,
    OPTFamilyHandler,
    MambaFamilyHandler,
    MoEFamilyHandler,
    PhiFamilyHandler,
    GemmaFamilyHandler,
    QwenFamilyHandler,
)
from .layer_factory import UniversalLayerFactory
from .weight_loader import UniversalWeightLoader
from .moe_handler import MoEHandler, MoEConfig
from .universal_sli_integrator import UniversalSLIIntegrator
from .exceptions import (
    UnsupportedArchitectureError,
    WeightLoadingError,
    LayerCreationError,
    MoEConfigurationError,
)

__all__ = [
    # Core classes
    "UniversalSLIIntegrator",
    "ArchitectureRegistry",
    "ArchitectureFamily",
    "UniversalLayerFactory",
    "UniversalWeightLoader",
    "MoEHandler",
    "MoEConfig",
    
    # Family handlers
    "LlamaFamilyHandler",
    "GPTFamilyHandler",
    "ChatGLMFamilyHandler",
    "T5FamilyHandler",
    "BLOOMFamilyHandler",
    "OPTFamilyHandler",
    "MambaFamilyHandler",
    "MoEFamilyHandler",
    "PhiFamilyHandler",
    "GemmaFamilyHandler",
    "QwenFamilyHandler",
    
    # Exceptions
    "UnsupportedArchitectureError",
    "WeightLoadingError",
    "LayerCreationError",
    "MoEConfigurationError",
]

__version__ = "1.0.0"
