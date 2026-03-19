"""DynamicGraphDiffusionNet — full model.

Architecture
------------
For a sequence of T graph snapshots:

  1. Per-snapshot diffusion
     Each snapshot is processed independently by a stack of
     GraphDiffusionLayer modules.  After L layers the node embeddings
     encode local + neighbourhood structure at that time step.

  2. Temporal attention
     The per-snapshot node embeddings [T, N, D] are passed through
     TemporalAttention to produce a single [N, D] representation that
     integrates information across all time steps.

  3. Graph-level readout (optional)
     Sum/mean pooling over the N nodes → [D] graph embedding.

  4. Task head
     A linear head maps [D] → num_classes for graph-level prediction,
     or [N, D] → num_classes for node-level prediction.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from typing import List, Optional, Tuple

from .temporal_graph import TemporalGraph
from .diffusion import GraphDiffusionLayer
from .attention import TemporalAttention


class DynamicGraphDiffusionNet(nn.Module):
    """Dynamic Graph Diffusion Network.

    Args:
        in_features:    Input node feature dimension.
        hidden_dim:     Hidden / output dimension of diffusion layers.
        num_diff_layers: Number of stacked GraphDiffusionLayer modules.
        diffusion_t:    Diffusion time parameter.
        num_diff_steps: Euler steps per diffusion layer.
        num_attn_heads: Attention heads in TemporalAttention.
        time_dim:       Dimension of the Time2Vec encoding.
        attn_pool:      Temporal pooling method ("last" | "mean" | "attn").
        dropout:        Dropout probability.
        num_classes:    Number of output classes.
        task:           ``"graph"`` or ``"node"`` classification.
    """

    def __init__(
        self,
        in_features: int,
        hidden_dim: int = 64,
        num_diff_layers: int = 2,
        diffusion_t: float = 1.0,
        num_diff_steps: int = 5,
        num_attn_heads: int = 4,
        time_dim: int = 16,
        attn_pool: str = "attn",
        dropout: float = 0.1,
        num_classes: int = 2,
        task: str = "graph",
    ):
        super().__init__()
        if task not in {"graph", "node"}:
            raise ValueError(f"task must be 'graph' or 'node', got {task!r}")

        self.task = task
        self.hidden_dim = hidden_dim

        # Input projection (in case feature_dim ≠ hidden_dim)
        self.input_proj = nn.Linear(in_features, hidden_dim)

        # Stack of diffusion layers
        self.diff_layers = nn.ModuleList(
            [
                GraphDiffusionLayer(
                    in_features=hidden_dim,
                    out_features=hidden_dim,
                    diffusion_t=diffusion_t,
                    num_steps=num_diff_steps,
                    dropout=dropout,
                )
                for _ in range(num_diff_layers)
            ]
        )

        # Temporal attention over snapshots
        self.temporal_attn = TemporalAttention(
            dim=hidden_dim,
            num_heads=num_attn_heads,
            time_dim=time_dim,
            dropout=dropout,
            pool=attn_pool,
        )

        # Classification head
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def _encode_snapshot(self, graph: TemporalGraph) -> Tensor:
        """Encode a single snapshot.  Returns [N, hidden_dim]."""
        # Project input features
        x = self.input_proj(graph.node_features)

        # Apply diffusion layers (each layer replaces node_features in a copy)
        for layer in self.diff_layers:
            # Build a lightweight wrapper that the layer can consume
            g = TemporalGraph(
                num_nodes=graph.num_nodes,
                edge_index=graph.edge_index,
                timestamps=graph.timestamps,
                node_features=x,
                time=graph.time,
            )
            x = layer(g)   # [N, hidden_dim]
        return x

    def forward(
        self,
        snapshots: List[TemporalGraph],
    ) -> Tensor:
        """Forward pass over a temporal sequence.

        Args:
            snapshots: Ordered list of TemporalGraph objects (same num_nodes).

        Returns:
            Logits tensor.
            - Graph task: [num_classes]
            - Node task:  [N, num_classes]
        """
        assert len(snapshots) > 0, "Need at least one snapshot"

        # Move everything to the same device as model parameters
        device = next(self.parameters()).device
        snapshots = [s.to(device) for s in snapshots]

        # Encode each snapshot independently
        embeddings = [self._encode_snapshot(s) for s in snapshots]  # T × [N, D]
        timestamps = [s.time for s in snapshots]

        # Attend over time
        node_emb = self.temporal_attn(embeddings, timestamps)  # [N, D]

        if self.task == "node":
            return self.head(node_emb)                         # [N, num_classes]
        else:
            graph_emb = node_emb.mean(dim=0)                   # [D]
            return self.head(graph_emb)                        # [num_classes]

    def embed(self, snapshots: List[TemporalGraph]) -> Tensor:
        """Return [N, D] node embeddings without the classification head."""
        device = next(self.parameters()).device
        snapshots = [s.to(device) for s in snapshots]
        embeddings = [self._encode_snapshot(s) for s in snapshots]
        timestamps = [s.time for s in snapshots]
        return self.temporal_attn(embeddings, timestamps)  # [N, D]
