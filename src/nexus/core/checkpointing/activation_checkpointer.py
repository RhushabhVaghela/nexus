"""Simple activation checkpointer for streaming SLI workloads.

The ActivationCheckpointer stores activations at a configurable
frequency (every Nth layer). It exposes a lightweight API for saving
and retrieving checkpoint activations. The module is intentionally
small so tests can validate the checkpointing semantics.
"""

from __future__ import annotations

from typing import Any, Dict


class ActivationCheckpointer:
    """Store activations at a specified checkpoint frequency.

    The checkpointer stores activations in memory in this implementation
    for simplicity. A production version would write to disk or an
    efficient key-value store to avoid large RAM usage.
    """

    def __init__(self, checkpoint_frequency: int = 4):
        if checkpoint_frequency < 1:
            raise ValueError("checkpoint_frequency must be >= 1")
        self.checkpoint_frequency = int(checkpoint_frequency)
        self.checkpoints: Dict[int, Any] = {}

    def maybe_save(self, layer_idx: int, activations: Any) -> None:
        """Save activations if layer_idx aligns with the checkpoint frequency."""
        if layer_idx % self.checkpoint_frequency == 0:
            # Store a shallow copy reference; caller controls memory
            self.checkpoints[layer_idx] = activations

    def get(self, layer_idx: int):
        """Return the stored activation for the layer or None."""
        return self.checkpoints.get(layer_idx)

    def clear(self) -> None:
        """Clear all stored checkpoints."""
        self.checkpoints.clear()
