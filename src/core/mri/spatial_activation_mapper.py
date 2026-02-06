"""Streaming-compatible spatial activation mapper for Nexus.

This module implements the SpatialActivationMapper described in the
`perplexity-conversation.md`. It supports building per-sample sparse
activation maps while streaming layers (load, run, discard).

The implementation is intentionally small and testable: it relies on a
`layer_loader` callable (injected) that, given a layer_idx, returns a
callable `layer_fn(activations) -> next_activations`. That keeps this
module independent of model/serialization details and easy to unit test.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable, Dict, Iterable, List

import numpy as np


class SpatialActivationMapper:
    """Build per-sample spatial activation maps while streaming layers.

    Args:
        layer_count: number of layers to stream
        layer_loader: callable layer_idx -> layer_fn
        threshold: activation threshold for considering a neuron "active"
    """

    def __init__(
        self,
        layer_count: int,
        layer_loader: Callable[[int], Callable[[Any], Any]],
        threshold: float = 1e-3,
    ):
        self.layer_count = int(layer_count)
        self.layer_loader = layer_loader
        self.threshold = float(threshold)

        # activation_maps[layer_idx][sample_idx] = {'indices': np.array, 'values': np.array}
        self.activation_maps: Dict[int, Dict[int, Dict[str, np.ndarray]]] = {
            i: {} for i in range(self.layer_count)
        }

    def build_spatial_map(
        self, samples: Iterable[Any], batch_mode: bool = False, max_topk: int = 512
    ) -> Dict[int, Dict[int, Dict[str, np.ndarray]]]:
        """Build activation maps for given samples.

        samples: iterable of sample inputs. Each sample is passed through
                 the layer functions sequentially.
        batch_mode: not implemented in this minimal version; kept for
                    API compatibility.
        max_topk: number of top neurons to record per sample/layer.
        """
        for sample_idx, sample in enumerate(samples):
            # initial activation is the sample representation expected by layers
            current_activation = self._encode_sample(sample)

            for layer_idx in range(self.layer_count):
                layer_fn = self.layer_loader(layer_idx)
                next_activation = layer_fn(current_activation)

                indices, values = self._identify_activated_neurons(
                    next_activation, self.threshold, max_topk
                )

                self.activation_maps[layer_idx][sample_idx] = {
                    "indices": indices,
                    "values": values,
                }

                # move on and discard references
                current_activation = next_activation

        return self.activation_maps

    def _identify_activated_neurons(
        self, activation_tensor: Any, threshold: float, max_topk: int
    ):
        """Return (indices, values) arrays of neurons above threshold.

        activation_tensor is assumed to be array-like (numpy or torch).
        We convert to numpy for deterministic behavior in tests.
        """
        # Accept numpy arrays or objects with .cpu().numpy() attr
        if hasattr(activation_tensor, "cpu"):
            try:
                arr = activation_tensor.cpu().numpy()
            except Exception:
                arr = np.asarray(activation_tensor)
        else:
            arr = np.asarray(activation_tensor)

        # Support shapes like (seq, hidden) or (hidden,) - reduce over seq
        if arr.ndim == 2:
            magnitudes = np.mean(np.abs(arr), axis=0)
        elif arr.ndim == 1:
            magnitudes = np.abs(arr)
        else:
            # collapse to last dimension
            magnitudes = np.mean(np.abs(arr.reshape(-1, arr.shape[-1])), axis=0)

        active_mask = magnitudes > threshold
        indices = np.nonzero(active_mask)[0]

        if indices.size == 0:
            return np.array([], dtype=int), np.array([], dtype=float)

        # pick top-k by magnitude
        topk = min(max_topk, indices.size)
        top_indices = indices[np.argsort(-magnitudes[indices])][:topk]
        values = magnitudes[top_indices]

        return top_indices.astype(int), values.astype(float)

    def _encode_sample(self, sample: Any) -> Any:
        """Identity mapping in this minimal implementation. Real code
        would run tokenization/embedding. Kept simple so unit tests can
        provide pre-encoded activations or small lambdas.
        """
        return sample
