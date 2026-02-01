"""
Real-Time Vision Buffer
Sliding Window for Continuous Video Analysis
"""
import torch
from collections import deque

class VisionStreamBuffer:
    """
    Manages a sliding window of recent video frames.
    Sampled at 1-2 FPS for LLM consumption.
    """
    def __init__(self, max_frames: int = 16):
        self.max_frames = max_frames
        self.buffer = deque(maxlen=max_frames)
    
    def add_frame(self, frame_tensor: torch.Tensor):
        """
        Add a new frame (e.g. from webcam/stream).
        frame_tensor: [3, 512, 512] (SigLIP 2 format)
        """
        self.buffer.append(frame_tensor)
        
    def get_context(self) -> torch.Tensor:
        """
        Returns stacked tensor of current context window.
        Shape: [num_frames, 3, 512, 512]
        """
        if not self.buffer:
            return None
        return torch.stack(list(self.buffer))

    def clear(self):
        self.buffer.clear()
