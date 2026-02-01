import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

class SparseIntentRouter(nn.Module):
    def __init__(self, input_dim, num_towers=4):
        """
        Determines which Specialist Tower to activate based on the input.
        This is critical for latency reduction (Step 2.D in Mitigation Strategy).
        """
        super().__init__()
        # A simple MLP Router
        self.router = nn.Sequential(
            nn.Linear(input_dim, input_dim // 4),
            nn.ReLU(),
            nn.Linear(input_dim // 4, num_towers) 
        )
        self.tower_names = ["reasoning", "vision", "audio", "generation"]

    def forward(self, hidden_states, threshold=0.5):
        """
        Args:
            hidden_states: (Batch, Seq, Dim) - Usually the first token or pooled embedding of the student input.
        Returns:
            active_towers: List[str] of towers to spin up.
        """
        # Pool the sequence to valid classificaiton token (e.g., mean pool or first token)
        # Assuming hidden_states is (B, D) or (B, S, D) -> we take mean
        if len(hidden_states.shape) == 3:
            pooled = hidden_states.mean(dim=1)
        else:
            pooled = hidden_states
            
        logits = self.router(pooled)
        probs = torch.sigmoid(logits)
        
        # In inference, we threshold. In training, we might use Gumbel-Softmax for differentiability.
        active_mask = (probs > threshold).int()
        
        return probs, active_mask

class HardModalityRouter:
    """
    Rule-based fallback if the Neural Router is uncertain.
    """
    @staticmethod
    def route_by_keywords(text: str) -> List[str]:
        active = ["reasoning"] # Always active
        text = text.lower()
        
        if any(w in text for w in ["look", "image", "photo", "describe", "picture"]):
            active.append("vision")
        
        if any(w in text for w in ["listen", "audio", "sound", "voice", "hear"]):
            active.append("audio")
            
        if any(w in text for w in ["draw", "generate", "paint", "create a video", "make a video"]):
            active.append("generation")
            
        return active
