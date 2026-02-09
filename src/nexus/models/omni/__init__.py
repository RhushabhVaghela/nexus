# src/omni/__init__.py
"""Omni model support for Qwen2.5-Omni models."""

from .loader import OmniModelLoader, OmniModelConfig, load_omni_model

__all__ = ['OmniModelLoader', 'OmniModelConfig', 'load_omni_model']
