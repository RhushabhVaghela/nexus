# src/data/__init__.py
"""Data loading utilities."""

from .universal_loader import UniversalDataLoader, LoadResult, load_dataset_universal

__all__ = ['UniversalDataLoader', 'LoadResult', 'load_dataset_universal']
