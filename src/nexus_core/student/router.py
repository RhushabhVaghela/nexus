import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class SparseIntentRouter(nn.Module):
    """
    The 'Traffic Cop' of Nexus.
    A specialized, Pruned Mixture-of-Experts (MoE) gate.
    
    Research Basis:
    - "Sparse Capability Routing": Enables strict modality isolation.
    - Deterministic routing to avoid 'Feature Dilution'.
    """
    def __init__(self, input_dim: int, num_towers: int, top_k: int = 1):
        super().__init__()
        self.top_k = top_k
        self.gate = nn.Linear(input_dim, num_towers)
        
        # Capability Tags for Heuristic Fallback
        # 0: Reasoning, 1: Vision, 2: Audio, 3: Generation, 4: Agentic
        self.tower_tags = ["reasoning", "vision", "audio", "generation", "agentic"]

    def forward(self, x: torch.Tensor, heuristic_override: str = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input embeddings (Batch, Dim)
            heuristic_override: Optional string to force a tower (e.g. "vision")
            
        Returns:
            router_logits: Scaled output for each tower
            indices: Which towers were selected
        """
        # 1. Neural Routing
        logits = self.gate(x) # (B, num_towers)
        
        # 2. Heuristic Override (Critical for mitigating Routing Instability)
        if heuristic_override and heuristic_override in self.tower_tags:
            # Force high logit for target, low for others
            target_idx = self.tower_tags.index(heuristic_override)
            mask = torch.ones_like(logits) * -1e9
            mask[:, target_idx] = 1e9
            logits = logits + mask

        # 3. Top-K Gating (Standard MoE)
        # We use Softmax over top-k to allow gradients to flow
        top_logits, indices = torch.topk(logits, self.top_k, dim=1)
        scores = F.softmax(top_logits, dim=1)
        
        return scores, indices

    def get_routing_map(self, prompt: str) -> str:
        """
        Text-based analysis to hint the router (Keyword Heuristic).
        """
        prompt = prompt.lower()
        if any(w in prompt for w in ["draw", "generate", "create", "make a picture"]):
            return "generation"
        if any(w in prompt for w in ["listen", "sound", "voice", "speak"]):
            return "audio"
        if any(w in prompt for w in ["look", "see", "describe", "image", "photo"]):
            return "vision"
        if any(w in prompt for w in ["solve", "calculate", "prove"]):
            return "reasoning"
        return None
