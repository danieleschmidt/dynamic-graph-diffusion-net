"""Core neural network layers for DGDN."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree, softmax


class MultiHeadTemporalAttention(nn.Module):
    """Multi-head attention mechanism with temporal awareness.
    
    Enables selective information aggregation from temporal neighbors
    with position-aware attention scoring.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        bias: bool = True,
        temporal_dim: int = 32
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.temporal_dim = temporal_dim
        self.dropout = dropout
        
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        # Query, Key, Value projections
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        
        # Temporal projections
        self.temporal_q = nn.Linear(temporal_dim, hidden_dim, bias=bias)
        self.temporal_k = nn.Linear(temporal_dim, hidden_dim, bias=bias)
        
        # Output projection
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        
        # Dropout
        self.dropout_layer = nn.Dropout(dropout)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters using Xavier uniform."""
        for module in [self.q_proj, self.k_proj, self.v_proj, 
                      self.temporal_q, self.temporal_k, self.out_proj]:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        temporal_encoding: torch.Tensor,
        edge_index: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of temporal attention.
        
        Args:
            query: Query vectors [num_nodes, hidden_dim]
            key: Key vectors [num_nodes, hidden_dim] 
            value: Value vectors [num_nodes, hidden_dim]
            temporal_encoding: Time encodings [num_edges, temporal_dim]
            edge_index: Edge connectivity [2, num_edges]
            attention_mask: Optional attention mask
            
        Returns:
            Tuple of (attention_output, attention_weights)
        """
        batch_size = query.shape[0]
        
        # Project to multi-head space
        Q = self.q_proj(query).view(batch_size, self.num_heads, self.head_dim)
        K = self.k_proj(key).view(batch_size, self.num_heads, self.head_dim)
        V = self.v_proj(value).view(batch_size, self.num_heads, self.head_dim)
        
        # Add temporal information to queries and keys
        temporal_q = self.temporal_q(temporal_encoding).view(-1, self.num_heads, self.head_dim)
        temporal_k = self.temporal_k(temporal_encoding).view(-1, self.num_heads, self.head_dim)
        
        # Get source and target nodes for each edge
        src_nodes, tgt_nodes = edge_index[0], edge_index[1]
        
        # Compute attention scores with temporal awareness
        # Q_i * K_j + temporal_q_ij * temporal_k_ij
        Q_src = Q[src_nodes]  # [num_edges, num_heads, head_dim]
        K_tgt = K[tgt_nodes]  # [num_edges, num_heads, head_dim]
        
        # Scaled dot-product attention with temporal component
        attn_scores = torch.sum(Q_src * K_tgt, dim=-1) / math.sqrt(self.head_dim)
        temporal_scores = torch.sum(temporal_q * temporal_k, dim=-1)
        
        combined_scores = attn_scores + temporal_scores
        
        # Apply attention mask if provided
        if attention_mask is not None:
            combined_scores.masked_fill_(attention_mask, float('-inf'))
        
        # Softmax over edges for each target node
        attn_weights = softmax(combined_scores, tgt_nodes, num_nodes=batch_size)
        
        # Apply dropout
        attn_weights = self.dropout_layer(attn_weights)
        
        # Apply attention to values
        V_src = V[src_nodes]  # [num_edges, num_heads, head_dim]
        attn_output = attn_weights.unsqueeze(-1) * V_src  # [num_edges, num_heads, head_dim]
        
        # Aggregate messages for each target node
        attn_output = torch.zeros_like(V).scatter_add_(
            0, tgt_nodes.unsqueeze(-1).unsqueeze(-1).expand(-1, self.num_heads, self.head_dim),
            attn_output
        )
        
        # Reshape and project
        attn_output = attn_output.view(batch_size, self.hidden_dim)
        attn_output = self.out_proj(attn_output)
        
        return attn_output, attn_weights


class DGDNLayer(MessagePassing):
    """Single DGDN layer combining temporal encoding, diffusion, and attention."""
    
    def __init__(
        self,
        hidden_dim: int,
        time_dim: int = 32,
        num_heads: int = 8,
        num_diffusion_steps: int = 3,
        dropout: float = 0.1,
        activation: str = "relu",
        layer_norm: bool = True
    ):
        super().__init__(aggr='add', node_dim=-2)
        
        self.hidden_dim = hidden_dim
        self.time_dim = time_dim
        self.num_heads = num_heads
        self.num_diffusion_steps = num_diffusion_steps
        self.dropout = dropout
        self.layer_norm = layer_norm
        
        # Temporal attention
        self.temporal_attention = MultiHeadTemporalAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            temporal_dim=time_dim
        )
        
        # Diffusion components
        self.diffusion_layers = nn.ModuleList([
            self._create_diffusion_block(hidden_dim, activation)
            for _ in range(num_diffusion_steps)
        ])
        
        # Layer normalization
        if layer_norm:
            self.norm1 = nn.LayerNorm(hidden_dim)
            self.norm2 = nn.LayerNorm(hidden_dim)
        
        # Residual projections
        self.residual_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Dropout
        self.dropout_layer = nn.Dropout(dropout)
        
    def _create_diffusion_block(self, hidden_dim: int, activation: str) -> nn.Module:
        """Create a diffusion processing block."""
        activation_fn = self._get_activation(activation)
        
        return nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            activation_fn,
            nn.Dropout(self.dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(self.dropout)
        )
    
    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function by name."""
        activations = {
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "swish": nn.SiLU(),
            "leaky_relu": nn.LeakyReLU(0.1)
        }
        return activations.get(activation, nn.ReLU())
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        temporal_encoding: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Forward pass of DGDN layer.
        
        Args:
            x: Node features [num_nodes, hidden_dim]  
            edge_index: Edge connectivity [2, num_edges]
            temporal_encoding: Temporal encodings [num_edges, time_dim]
            edge_attr: Edge attributes [num_edges, edge_dim] (optional)
            
        Returns:
            Dictionary containing processed features and attention weights
        """
        # Store input for residual connection
        residual = x
        
        # 1. Temporal attention mechanism
        attn_output, attn_weights = self.temporal_attention(
            query=x,
            key=x, 
            value=x,
            temporal_encoding=temporal_encoding,
            edge_index=edge_index
        )
        
        # Apply layer norm and residual connection
        if self.layer_norm:
            x = self.norm1(x + self.dropout_layer(attn_output))
        else:
            x = x + self.dropout_layer(attn_output)
        
        # 2. Multi-step diffusion process
        diffusion_output = x
        
        for diffusion_layer in self.diffusion_layers:
            # Apply diffusion transformation
            step_output = diffusion_layer(diffusion_output)
            
            # Residual connection within diffusion
            diffusion_output = diffusion_output + self.dropout_layer(step_output)
        
        # 3. Final processing and residual connection
        output = self.residual_proj(diffusion_output)
        
        if self.layer_norm:
            output = self.norm2(residual + self.dropout_layer(output))
        else:
            output = residual + self.dropout_layer(output)
        
        return {
            "node_features": output,
            "attention_weights": attn_weights,
            "diffusion_features": diffusion_output
        }


class GraphNorm(nn.Module):
    """Graph normalization layer for stable training."""
    
    def __init__(self, hidden_dim: int, eps: float = 1e-5):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.eps = eps
        
        # Learnable parameters
        self.weight = nn.Parameter(torch.ones(hidden_dim))
        self.bias = nn.Parameter(torch.zeros(hidden_dim))
    
    def forward(self, x: torch.Tensor, batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Apply graph normalization.
        
        Args:
            x: Node features [num_nodes, hidden_dim]
            batch: Batch assignment for each node (optional)
            
        Returns:
            Normalized node features
        """
        if batch is None:
            # Global normalization
            mean = x.mean(dim=0, keepdim=True)
            var = x.var(dim=0, keepdim=True, unbiased=False)
        else:
            # Per-graph normalization
            mean = torch.zeros_like(x[0:1])
            var = torch.ones_like(x[0:1])
            
            for b in batch.unique():
                mask = batch == b
                if mask.sum() > 1:
                    x_batch = x[mask]
                    mean_batch = x_batch.mean(dim=0, keepdim=True)
                    var_batch = x_batch.var(dim=0, keepdim=True, unbiased=False)
                    
                    mean = torch.where(mask.unsqueeze(-1), mean_batch, mean)
                    var = torch.where(mask.unsqueeze(-1), var_batch, var)
        
        # Normalize
        normalized = (x - mean) / torch.sqrt(var + self.eps)
        
        # Scale and shift
        return self.weight * normalized + self.bias


class PositionalEncoding(nn.Module):
    """Positional encoding for sequence-like temporal patterns."""
    
    def __init__(self, hidden_dim: int, max_len: int = 10000):
        super().__init__()
        
        pe = torch.zeros(max_len, hidden_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, hidden_dim, 2).float() * 
                           (-math.log(10000.0) / hidden_dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input features.
        
        Args:
            x: Input features [batch_size, seq_len, hidden_dim]
            positions: Position indices [batch_size, seq_len]
            
        Returns:
            Features with positional encoding added
        """
        # Clamp positions to valid range
        positions = torch.clamp(positions, 0, self.pe.size(0) - 1).long()
        
        # Add positional encoding
        pos_encoding = self.pe[positions]
        return x + pos_encoding