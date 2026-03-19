"""TemporalAttention — attend over a sequence of graph snapshots.

Given a sequence of per-snapshot node embeddings  [T, N, D],
we want each node to produce a summary that weighs all past snapshots.

We use standard multi-head self-attention over the time axis, with a
learned sinusoidal positional encoding that encodes the snapshot
timestamp (not the index, so irregular time intervals are handled).
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
from torch import Tensor
from typing import List, Optional


# ---------------------------------------------------------------------------
# Temporal positional encoding
# ---------------------------------------------------------------------------

class TimeEncoding(nn.Module):
    """Learnable sinusoidal encoding of a scalar timestamp.

    Outputs a [D] vector for a given time value using the formula from
    "Time2Vec" (Kazemi et al. 2019):

        v_0   = w_0 * t + b_0             (linear)
        v_k   = sin(w_k * t + b_k)        (periodic, k >= 1)

    Args:
        dim: Output dimension.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.w = nn.Parameter(torch.randn(dim))
        self.b = nn.Parameter(torch.zeros(dim))

    def forward(self, t: Tensor) -> Tensor:
        """Encode timestamps.

        Args:
            t: [T] or [B, T] scalar timestamps.

        Returns:
            [..., T, D] encoding.
        """
        # t: [...] → [..., 1] → broadcast over D
        t_exp = t.unsqueeze(-1)              # [..., 1]
        arg = self.w * t_exp + self.b        # [..., D]
        # First dimension is linear; rest are sinusoidal
        enc = torch.cat([arg[..., :1], torch.sin(arg[..., 1:])], dim=-1)
        return enc                           # [..., D]


# ---------------------------------------------------------------------------
# Temporal Attention
# ---------------------------------------------------------------------------

class TemporalAttention(nn.Module):
    """Multi-head self-attention over temporal snapshots, per node.

    Given embeddings  X ∈ [T, N, D]  (T snapshots, N nodes, D features)
    and timestamps   t ∈ [T],
    returns  Y ∈ [N, D]  — a weighted summary over time for each node.

    Architecture:
        1. Add Time2Vec positional encoding to X.
        2. Apply standard multi-head self-attention over the T axis.
        3. Pool the T outputs via (a) last-step, (b) mean, or (c) attention.
        4. Feed-forward + residual + LayerNorm.

    Args:
        dim:        Feature dimension D.
        num_heads:  Number of attention heads.
        time_dim:   Dimension of the temporal encoding (added to D).
        dropout:    Dropout probability.
        pool:       Temporal pooling method: ``"last"`` | ``"mean"`` | ``"attn"``.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        time_dim: int = 16,
        dropout: float = 0.1,
        pool: str = "attn",
    ):
        super().__init__()
        if pool not in {"last", "mean", "attn"}:
            raise ValueError(f"pool must be 'last', 'mean', or 'attn', got {pool!r}")

        self.dim = dim
        self.num_heads = num_heads
        self.pool = pool

        self.time_enc = TimeEncoding(time_dim)
        self.input_proj = nn.Linear(dim + time_dim, dim)

        # Standard PyTorch MHSA — operates on (seq, batch, dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=False,   # we'll use (T, N, D) format
        )

        self.ff = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(dim)

        if pool == "attn":
            self.pool_query = nn.Parameter(torch.randn(1, dim))

    def forward(
        self,
        embeddings: List[Tensor],   # list of T tensors, each [N, D]
        timestamps: List[float],    # list of T scalar times
    ) -> Tensor:
        """Aggregate node embeddings across time.

        Args:
            embeddings: Per-snapshot node embeddings, length T.
            timestamps: Scalar timestamp for each snapshot, length T.

        Returns:
            [N, D] aggregated node representations.
        """
        T = len(embeddings)
        assert T > 0, "Need at least one snapshot"
        N, D = embeddings[0].shape
        device = embeddings[0].device

        # Stack: [T, N, D]
        X = torch.stack(embeddings, dim=0)

        # Time encoding: [T, D_time]
        t_tensor = torch.tensor(timestamps, dtype=torch.float32, device=device)
        t_enc = self.time_enc(t_tensor)          # [T, time_dim]
        t_enc_exp = t_enc.unsqueeze(1).expand(T, N, -1)  # [T, N, time_dim]

        # Concatenate and project back to D
        X_aug = torch.cat([X, t_enc_exp], dim=-1)   # [T, N, D+time_dim]
        X_proj = self.input_proj(X_aug)               # [T, N, D]

        # Multi-head attention over time axis
        # PyTorch MHA: (seq_len, batch, embed)  →  here seq=T, batch=N
        X_t = X_proj  # [T, N, D]
        attn_out, _ = self.attn(X_t, X_t, X_t)       # [T, N, D]

        # Residual
        X_res = X_proj + attn_out                      # [T, N, D]

        # Feed-forward per node (apply over last two dims)
        X_flat = X_res.view(T * N, D)
        X_ff = X_flat + self.ff(X_flat)
        X_out = self.norm(X_ff).view(T, N, D)          # [T, N, D]

        # Temporal pooling
        if self.pool == "last":
            return X_out[-1]                            # [N, D]
        elif self.pool == "mean":
            return X_out.mean(dim=0)                    # [N, D]
        else:  # "attn"
            # Learnable query per node: [N, D] × [T, N, D] → [N, D]
            q = self.pool_query.expand(N, D)            # [N, D]
            # Dot-product with each time step
            scores = torch.einsum("nd,tnd->tn", q, X_out)  # [T, N]
            weights = torch.softmax(scores, dim=0)          # [T, N]
            pooled = (weights.unsqueeze(-1) * X_out).sum(dim=0)  # [N, D]
            return pooled
