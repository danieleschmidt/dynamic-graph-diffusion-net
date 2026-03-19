"""GraphDiffusionLayer — heat diffusion message passing.

The core idea is borrowed from spectral graph theory:
  The heat equation on a graph is  dX/dt = -L X
  Its solution at time t is        X(t) = exp(-t * L) X(0)

where L is the normalised graph Laplacian.

Computing the full matrix exponential is O(N³), so we use the
truncated Taylor / Chebyshev approximation:

  K ≈ Σ_{k=0}^{K} (-t)^k / k!  ·  L^k  X

which can be unrolled as K sequential diffusion steps:

  H_0   = X
  H_k   = H_{k-1} - (t/K) · L · H_{k-1}   for k = 1 … K

This is equivalent to K steps of normalised graph convolution.

The layer then passes the resulting diffused features through a
learnable linear + nonlinear transform.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional

from .temporal_graph import TemporalGraph


def _compute_laplacian_scatter(
    edge_index: Tensor,
    num_nodes: int,
    timestamps: Optional[Tensor] = None,
    time_decay: float = 0.1,
) -> Tensor:
    """Return D^{-1/2} A D^{-1/2} (symmetric normalised adjacency).

    If *timestamps* are provided the edge weights are decayed by
    ``exp(-time_decay * delta_t)`` where ``delta_t`` is the age of the
    edge relative to the maximum timestamp in this snapshot.

    Returns a [E] tensor of edge weights w_ij.
    """
    src, dst = edge_index[0], edge_index[1]
    E = edge_index.shape[1]

    if timestamps is not None:
        t_max = timestamps.max()
        age = t_max - timestamps          # [E] — older edges get larger age
        w = torch.exp(-time_decay * age)  # [E] — decay by age
    else:
        w = torch.ones(E, device=edge_index.device)

    # Degree per node (weighted)
    deg = torch.zeros(num_nodes, device=edge_index.device)
    deg.scatter_add_(0, dst, w)

    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0.0

    # Normalise: w_ij ← d_i^{-1/2} · w_ij · d_j^{-1/2}
    norm_w = deg_inv_sqrt[src] * w * deg_inv_sqrt[dst]
    return norm_w  # [E]


def _graph_conv_scatter(
    x: Tensor,          # [N, F]
    edge_index: Tensor, # [2, E]
    edge_w: Tensor,     # [E]
) -> Tensor:
    """One step of  H ← A_norm · H  via scatter operations (no sparse mm)."""
    src, dst = edge_index[0], edge_index[1]
    N, F = x.shape

    # Message: m_ij = w_ij * x_j
    msg = edge_w.unsqueeze(-1) * x[src]   # [E, F]

    # Aggregate at destination
    out = torch.zeros(N, F, device=x.device)
    out.scatter_add_(0, dst.unsqueeze(-1).expand_as(msg), msg)
    return out  # [N, F]


class GraphDiffusionLayer(nn.Module):
    """Diffusion-based message-passing layer.

    Implements K steps of heat diffusion  H ← (I - t/K · L) H
    followed by a learnable feature transform.

    Args:
        in_features:   Input feature dimension.
        out_features:  Output feature dimension.
        diffusion_t:   Diffusion time parameter (controls spread radius).
        num_steps:     Number of Euler steps to approximate exp(-t*L).
        time_decay:    Decay rate for edge-age weighting.
        dropout:       Dropout on the output features.
        activation:    Activation after the linear transform.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        diffusion_t: float = 1.0,
        num_steps: int = 5,
        time_decay: float = 0.1,
        dropout: float = 0.1,
        activation: str = "relu",
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.diffusion_t = diffusion_t
        self.num_steps = num_steps
        self.time_decay = time_decay

        self.transform = nn.Linear(in_features, out_features)
        self.residual = nn.Linear(in_features, out_features, bias=False)
        self.norm = nn.LayerNorm(out_features)
        self.dropout = nn.Dropout(dropout)

        _act = {"relu": nn.ReLU(), "gelu": nn.GELU(), "silu": nn.SiLU()}
        self.activation = _act.get(activation, nn.ReLU())

        nn.init.xavier_uniform_(self.transform.weight)
        nn.init.xavier_uniform_(self.residual.weight)
        nn.init.zeros_(self.transform.bias)

    def forward(self, graph: TemporalGraph) -> Tensor:
        """Diffuse node features and apply learnable transform.

        Args:
            graph: TemporalGraph snapshot.

        Returns:
            [N, out_features] diffused + transformed node embeddings.
        """
        x = graph.node_features          # [N, in_features]
        edge_index = graph.edge_index    # [2, E]
        N = graph.num_nodes

        # Add self-loops so every node receives its own signal
        loop = torch.arange(N, device=x.device)
        loop_idx = torch.stack([loop, loop], dim=0)
        loop_ts = torch.full((N,), graph.time, device=x.device)
        ei = torch.cat([edge_index, loop_idx], dim=1)
        ts = torch.cat([graph.timestamps.to(x.device), loop_ts])

        edge_w = _compute_laplacian_scatter(
            ei, N, timestamps=ts, time_decay=self.time_decay
        )

        # Euler integration of heat equation
        dt = self.diffusion_t / self.num_steps
        h = x
        for _ in range(self.num_steps):
            diffused = _graph_conv_scatter(h, ei, edge_w)  # A_norm · h
            h = h + dt * (diffused - h)                    # h += dt*(A·h - h) = -dt*L*h

        # Learnable transform + residual + norm
        out = self.activation(self.transform(h))
        out = out + self.residual(x)
        out = self.norm(out)
        out = self.dropout(out)
        return out  # [N, out_features]
