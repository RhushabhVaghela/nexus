"""
Tower implementations for the Nexus specialist tower architecture.

All tower types inherit from BaseTower and implement layer-wise
orchestration with adapter management, gradient checkpointing,
and teacher model integration.

Tower Types:
    - StaticTower: Fixed layer configuration
    - DynamicTower: Runtime-adjustable depth/width
    - MoETower: Mixture-of-Experts routing
    - RouterTower: Multi-strategy routing across towers
    - ReasoningTower: Specialized for chain-of-thought reasoning
    - VisionTower: Vision encoder integration

Registry:
    - TEACHER_REGISTRY, DATASET_REGISTRY: Model/dataset metadata
    - detect_architecture, get_model_info: Auto-detection helpers
    - TowerLoader: Load tower weights from disk
"""

import importlib as _importlib

# --- Lightweight registry (no torch dependency) ---
from .registry import (
    TEACHER_REGISTRY,
    DATASET_REGISTRY,
    ARCHITECTURE_MAP,
    detect_architecture,
    get_model_info,
    register_unknown_model,
    list_models_by_type,
    list_models_by_tag,
)

# --- Lazy imports for torch-dependent tower classes ---
_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    # base_tower.py
    "TowerMode": (".base_tower", "TowerMode"),
    "TowerConfig": (".base_tower", "TowerConfig"),
    "LayerState": (".base_tower", "LayerState"),
    "BaseTower": (".base_tower", "BaseTower"),
    # static_tower.py
    "StaticTowerConfig": (".static_tower", "StaticTowerConfig"),
    "StaticTower": (".static_tower", "StaticTower"),
    # dynamic_tower.py
    "DynamicTowerConfig": (".dynamic_tower", "DynamicTowerConfig"),
    "DynamicTower": (".dynamic_tower", "DynamicTower"),
    # moe_tower.py
    "MoETowerConfig": (".moe_tower", "MoETowerConfig"),
    "Expert": (".moe_tower", "Expert"),
    "Router": (".moe_tower", "Router"),
    "MoETower": (".moe_tower", "MoETower"),
    # router_tower.py
    "RoutingStrategy": (".router_tower", "RoutingStrategy"),
    "RouterTowerConfig": (".router_tower", "RouterTowerConfig"),
    "TowerRouter": (".router_tower", "TowerRouter"),
    "RouterTower": (".router_tower", "RouterTower"),
    # reasoning_tower.py
    "ReasoningTower": (".reasoning_tower", "ReasoningTower"),
    # vision_tower.py
    "VisionTower": (".vision_tower", "VisionTower"),
    # loader.py
    "TowerLoader": (".loader", "TowerLoader"),
}


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name]
        module = _importlib.import_module(module_path, __name__)
        return getattr(module, attr_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Registry (direct imports)
    "TEACHER_REGISTRY",
    "DATASET_REGISTRY",
    "ARCHITECTURE_MAP",
    "detect_architecture",
    "get_model_info",
    "register_unknown_model",
    "list_models_by_type",
    "list_models_by_tag",
    # Base tower (lazy)
    "TowerMode",
    "TowerConfig",
    "LayerState",
    "BaseTower",
    # Tower implementations (lazy)
    "StaticTowerConfig",
    "StaticTower",
    "DynamicTowerConfig",
    "DynamicTower",
    "MoETowerConfig",
    "Expert",
    "Router",
    "MoETower",
    "RoutingStrategy",
    "RouterTowerConfig",
    "TowerRouter",
    "RouterTower",
    "ReasoningTower",
    "VisionTower",
    # Loader (lazy)
    "TowerLoader",
]
