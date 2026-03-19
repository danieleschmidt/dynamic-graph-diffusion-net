"""TemporalGraph and SyntheticTemporalGraph.

A TemporalGraph is a snapshot of a graph at a specific time step.
It holds:
  - node_features: [N, F] float tensor — one feature vector per node
  - edge_index:    [2, E] long tensor  — (src, dst) pairs
  - timestamps:    [E]    float tensor — when each edge was observed
  - time:          scalar float        — the snapshot timestamp
  - num_nodes:     int

SyntheticTemporalGraph generates a sequence of TemporalGraph snapshots
that model realistic temporal dynamics (edges appear/disappear, node
features drift).
"""

from __future__ import annotations

import math
import torch
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class TemporalGraph:
    """One snapshot of a graph at time *t*.

    Args:
        num_nodes:     Number of nodes (fixed across all snapshots).
        edge_index:    [2, E] long tensor of (src, dst) pairs.
        timestamps:    [E] float tensor, one per edge.
        node_features: [N, F] float tensor.  If None, identity-like features
                       are created automatically.
        time:          Scalar timestamp of this snapshot.
    """

    num_nodes: int
    edge_index: torch.Tensor           # [2, E]
    timestamps: torch.Tensor           # [E]
    node_features: Optional[torch.Tensor] = None  # [N, F]
    time: float = 0.0

    def __post_init__(self):
        assert self.edge_index.shape[0] == 2, "edge_index must be [2, E]"
        assert self.edge_index.shape[1] == self.timestamps.shape[0], (
            "edge_index and timestamps must have the same number of edges"
        )
        if self.node_features is None:
            # Use a simple learnable-friendly default: one-hot-like identity
            self.node_features = torch.eye(self.num_nodes)
        assert self.node_features.shape[0] == self.num_nodes

    @property
    def num_edges(self) -> int:
        return self.edge_index.shape[1]

    @property
    def feature_dim(self) -> int:
        return self.node_features.shape[1]

    def to(self, device) -> "TemporalGraph":
        return TemporalGraph(
            num_nodes=self.num_nodes,
            edge_index=self.edge_index.to(device),
            timestamps=self.timestamps.to(device),
            node_features=self.node_features.to(device),
            time=self.time,
        )

    def add_self_loops(self) -> "TemporalGraph":
        """Return a copy with self-loops added (required for some diffusion ops)."""
        loop_src = torch.arange(self.num_nodes, dtype=torch.long)
        loop_idx = torch.stack([loop_src, loop_src], dim=0)
        loop_ts = torch.full((self.num_nodes,), self.time)
        new_edge_index = torch.cat([self.edge_index, loop_idx], dim=1)
        new_timestamps = torch.cat([self.timestamps, loop_ts])
        return TemporalGraph(
            num_nodes=self.num_nodes,
            edge_index=new_edge_index,
            timestamps=new_timestamps,
            node_features=self.node_features,
            time=self.time,
        )


# ---------------------------------------------------------------------------
# Synthetic data generator
# ---------------------------------------------------------------------------

class SyntheticTemporalGraph:
    """Generate a sequence of TemporalGraph snapshots with controllable dynamics.

    Three kinds of evolution are supported (mix them via *pattern*):

    ``"random"``
        Edges form/dissolve uniformly at random each step.

    ``"community"``
        Nodes are split into communities.  Intra-community edges are denser
        and more stable; inter-community edges are sparse and transient.

    ``"growing"``
        A growing connected component — new edges only connect to already-
        connected nodes, creating a preferential-attachment-like dynamic.

    Args:
        num_nodes:   Number of nodes (fixed).
        num_steps:   Number of time steps (snapshots).
        feature_dim: Dimension of node features.
        edge_prob:   Base probability for edge existence.
        pattern:     One of ``"random"``, ``"community"``, ``"growing"``.
        noise:       Gaussian noise added to features at each step.
        seed:        RNG seed for reproducibility.
    """

    def __init__(
        self,
        num_nodes: int = 20,
        num_steps: int = 8,
        feature_dim: int = 16,
        edge_prob: float = 0.3,
        pattern: str = "community",
        noise: float = 0.05,
        seed: int = 42,
    ):
        self.num_nodes = num_nodes
        self.num_steps = num_steps
        self.feature_dim = feature_dim
        self.edge_prob = edge_prob
        self.pattern = pattern
        self.noise = noise
        self.rng = torch.Generator()
        self.rng.manual_seed(seed)

        supported = {"random", "community", "growing"}
        if pattern not in supported:
            raise ValueError(f"pattern must be one of {supported}, got {pattern!r}")

        # Stable base features for each node — drift slightly each step
        self._base_features = torch.randn(num_nodes, feature_dim, generator=self.rng)

    # ------------------------------------------------------------------
    def generate(self) -> List[TemporalGraph]:
        """Return a list of TemporalGraph snapshots."""
        snapshots: List[TemporalGraph] = []
        for t in range(self.num_steps):
            g = self._snapshot(t)
            snapshots.append(g)
        return snapshots

    # ------------------------------------------------------------------
    def _snapshot(self, t: int) -> TemporalGraph:
        features = self._base_features + self.noise * torch.randn(
            self.num_nodes, self.feature_dim, generator=self.rng
        )

        if self.pattern == "random":
            edge_index, timestamps = self._random_edges(t)
        elif self.pattern == "community":
            edge_index, timestamps = self._community_edges(t)
        else:
            edge_index, timestamps = self._growing_edges(t)

        return TemporalGraph(
            num_nodes=self.num_nodes,
            edge_index=edge_index,
            timestamps=timestamps,
            node_features=features,
            time=float(t),
        )

    def _random_edges(self, t: int) -> Tuple[torch.Tensor, torch.Tensor]:
        srcs, dsts = [], []
        for i in range(self.num_nodes):
            for j in range(i + 1, self.num_nodes):
                if torch.rand(1, generator=self.rng).item() < self.edge_prob:
                    srcs.extend([i, j])
                    dsts.extend([j, i])
        if not srcs:
            # Guarantee at least one edge
            srcs, dsts = [0, 1], [1, 0]
        edge_index = torch.tensor([srcs, dsts], dtype=torch.long)
        timestamps = torch.full((edge_index.shape[1],), float(t))
        return edge_index, timestamps

    def _community_edges(self, t: int) -> Tuple[torch.Tensor, torch.Tensor]:
        n_communities = max(2, self.num_nodes // 5)
        community = torch.arange(self.num_nodes) % n_communities
        srcs, dsts = [], []
        for i in range(self.num_nodes):
            for j in range(i + 1, self.num_nodes):
                same = community[i] == community[j]
                p = self.edge_prob * 2.0 if same else self.edge_prob * 0.3
                if torch.rand(1, generator=self.rng).item() < p:
                    srcs.extend([i, j])
                    dsts.extend([j, i])
        if not srcs:
            srcs, dsts = [0, 1], [1, 0]
        edge_index = torch.tensor([srcs, dsts], dtype=torch.long)
        timestamps = torch.full((edge_index.shape[1],), float(t))
        return edge_index, timestamps

    def _growing_edges(self, t: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # At each step a fraction of nodes is "active"
        active = max(2, int(self.num_nodes * (0.3 + 0.7 * (t + 1) / self.num_steps)))
        srcs, dsts = [], []
        for i in range(1, active):
            # Connect to a random earlier node
            j = int(torch.randint(0, i, (1,), generator=self.rng).item())
            srcs.extend([i, j])
            dsts.extend([j, i])
            # Maybe add a second connection
            if active > 2 and torch.rand(1, generator=self.rng).item() < 0.4:
                k = int(torch.randint(0, i, (1,), generator=self.rng).item())
                if k != j:
                    srcs.extend([i, k])
                    dsts.extend([k, i])
        edge_index = torch.tensor([srcs, dsts], dtype=torch.long)
        timestamps = torch.full((edge_index.shape[1],), float(t))
        return edge_index, timestamps
