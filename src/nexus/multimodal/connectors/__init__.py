"""
Nexus Multimodal Connectors — cross-modal alignment bridges.

Includes Deep Flow Matching (DFM) connector using optimal transport
for aligning embeddings across vision, audio, and text modalities.
"""

from .dfm import DFMConnector, OptimalTransport, FlowMatchingBlock

__all__ = [
    "DFMConnector",
    "OptimalTransport",
    "FlowMatchingBlock",
]
