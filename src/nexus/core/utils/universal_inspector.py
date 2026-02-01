import torch
import torch.nn as nn
from typing import List, Optional, Any

class UniversalInspector:
    """
    Intelligently inspects any PyTorch module to find its 'Backbone' layers.
    Uses heuristics to identify the main stack of Transformer blocks.
    """

    @staticmethod
    def find_backbone_layers(model: nn.Module) -> nn.ModuleList:
        """
        Recursively searches for the largest ModuleList that looks like a layer stack.
        """
        # 1. Direct heuristics (Common HF names)
        # We try these first for speed, but fallback to deep inspection.
        # Note: We return the *actual object* if it's a ModuleList/Sequential
        candidates = []
        
        # Candidate 1: HF standard locations
        if hasattr(model, "model") and hasattr(model.model, "layers"): candidates.append(model.model.layers)
        if hasattr(model, "layers"): candidates.append(model.layers)
        if hasattr(model, "h"): candidates.append(model.h)
        if hasattr(model, "transformer") and hasattr(model.transformer, "h"): candidates.append(model.transformer.h)
        if hasattr(model, "encoder") and hasattr(model.encoder, "layers"): candidates.append(model.encoder.layers)
        if hasattr(model, "decoder") and hasattr(model.decoder, "layers"): candidates.append(model.decoder.layers)

        # Filter for valid containers
        valid_candidates = [c for c in candidates if isinstance(c, (nn.ModuleList, nn.Sequential))]
        if valid_candidates:
            # Return the largest one found
            return max(valid_candidates, key=len)

        # 2. Deep Inspection (Recursion)
        # If no obvious candidate, we walk the tree.
        found_containers = []
        
        def _walk(module: nn.Module, depth=0):
            if depth > 4: return # Don't go too deep
            
            for name, child in module.named_children():
                if isinstance(child, (nn.ModuleList, nn.Sequential)):
                    # Heuristic: A backbone usually has > 2 layers
                    if len(child) > 2:
                        found_containers.append(child)
                else:
                    _walk(child, depth + 1)
        
        _walk(model)
        
        if found_containers:
            # Return the longest container found deep in the tree
            return max(found_containers, key=len)

        # 3. Last Resort: Top-level children
        # If the model itself IS the container (rare but possible in custom implementations)
        # We wrap root children in a ModuleList if they seem uniform? 
        # For now, if we fail to find a container, we raise an error but with a helpful message.
        
        # Try to print structure for debug
        structure = [n for n, _ in model.named_children()]
        raise ValueError(f"UniversalInspector could not identify backbone layers. Top-level modules: {structure}. Please ensure model has a ModuleList of layers.")

    @staticmethod
    def find_custom_script(model_path: str, filename: str = "generate.py") -> Optional[str]:
        """
        Recursively searches for a specific script within the model directory.
        Returns the absolute path to the directory containing the file if found.
        """
        import os
        if not os.path.exists(model_path):
            return None
            
        if os.path.isfile(model_path):
            model_path = os.path.dirname(model_path)

        for root, dirs, files in os.walk(model_path):
            if filename in files:
                return os.path.abspath(root)
        return None
