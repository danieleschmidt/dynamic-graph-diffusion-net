"""Dynamic Graph Diffusion Network (DGDN).

A neural network for learning on graphs that evolve over time.
Core idea: represent temporal graph evolution as a sequence of snapshots,
apply heat-diffusion-based message passing within each snapshot, then
attend over the temporal sequence to capture how node roles change.
"""

from .temporal_graph import TemporalGraph, SyntheticTemporalGraph
from .diffusion import GraphDiffusionLayer
from .attention import TemporalAttention
from .model import DynamicGraphDiffusionNet

__all__ = [
    "TemporalGraph",
    "SyntheticTemporalGraph",
    "GraphDiffusionLayer",
    "TemporalAttention",
    "DynamicGraphDiffusionNet",
]
