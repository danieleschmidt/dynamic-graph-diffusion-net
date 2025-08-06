"""Edge-time encoding module for DGDN.

This module implements the sophisticated temporal encoding for evolving graph structures
using Fourier features for continuous time representation.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class EdgeTimeEncoder(nn.Module):
    """Edge-Time Encoder for continuous temporal modeling.
    
    Transforms continuous timestamps into rich temporal embeddings using 
    learnable Fourier features for scale-invariant representations.
    
    Args:
        time_dim: Dimension of the output time embeddings
        num_bases: Number of Fourier basis functions (default: 64)
        max_time: Maximum expected time value for normalization (default: 1000.0)
        learnable_bases: Whether to make Fourier bases learnable (default: True)
    """
    
    def __init__(
        self,
        time_dim: int,
        num_bases: int = 64,
        max_time: float = 1000.0,
        learnable_bases: bool = True,
    ):
        super().__init__()
        self.time_dim = time_dim
        self.num_bases = num_bases
        self.max_time = max_time
        
        # Initialize Fourier basis frequencies
        if learnable_bases:
            # Learnable frequencies and phases
            self.w = nn.Parameter(torch.randn(num_bases) * 0.1)
            self.b = nn.Parameter(torch.randn(num_bases) * 2 * math.pi)
        else:
            # Fixed logarithmic frequencies
            frequencies = torch.logspace(0, math.log10(max_time), num_bases)
            self.register_buffer('w', frequencies)
            self.register_buffer('b', torch.zeros(num_bases))
        
        # Projection layer to desired dimension
        self.projection = nn.Linear(num_bases, time_dim)
        
        # Layer normalization for stability
        self.layer_norm = nn.LayerNorm(time_dim)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters using Xavier uniform initialization."""
        nn.init.xavier_uniform_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)
        
        if hasattr(self, 'w') and self.w.requires_grad:
            # Initialize learnable frequencies uniformly in log space
            with torch.no_grad():
                self.w.uniform_(0.1, 10.0)
                self.b.uniform_(0, 2 * math.pi)
    
    def forward(self, timestamps: torch.Tensor) -> torch.Tensor:
        """Encode timestamps into temporal embeddings.
        
        Args:
            timestamps: Tensor of shape [num_edges] or [batch_size, num_edges]
                       containing continuous timestamp values
                       
        Returns:
            time_encoding: Tensor of shape [..., time_dim] containing temporal embeddings
        """
        # Handle empty input gracefully
        if timestamps.numel() == 0:
            return torch.empty(0, self.time_dim, device=timestamps.device, dtype=torch.float32)
        
        # Ensure timestamps are float
        timestamps = timestamps.float()
        
        # Normalize timestamps to [0, 1] range
        normalized_time = timestamps / self.max_time
        
        # Add dimension for broadcasting with basis functions
        if normalized_time.dim() == 1:
            normalized_time = normalized_time.unsqueeze(-1)  # [num_edges, 1]
        else:
            normalized_time = normalized_time.unsqueeze(-1)  # [batch_size, num_edges, 1]
        
        # Compute Fourier features: sin(w * t + b)
        fourier_features = torch.sin(normalized_time * self.w + self.b)
        
        # Project to desired dimension
        time_encoding = self.projection(fourier_features)
        
        # Apply layer normalization
        time_encoding = self.layer_norm(time_encoding)
        
        return time_encoding
    
    def get_time_range_encoding(
        self, 
        start_time: float, 
        end_time: float, 
        num_steps: int = 100
    ) -> torch.Tensor:
        """Generate time encodings for a range of timestamps.
        
        Useful for visualization and analysis of temporal patterns.
        
        Args:
            start_time: Starting timestamp
            end_time: Ending timestamp  
            num_steps: Number of time steps to generate
            
        Returns:
            encodings: Tensor of shape [num_steps, time_dim]
        """
        timestamps = torch.linspace(start_time, end_time, num_steps)
        return self.forward(timestamps)
    
    def compute_temporal_similarity(
        self, 
        time1: torch.Tensor, 
        time2: torch.Tensor
    ) -> torch.Tensor:
        """Compute cosine similarity between temporal encodings.
        
        Args:
            time1: First set of timestamps
            time2: Second set of timestamps
            
        Returns:
            similarity: Cosine similarities between encodings
        """
        enc1 = self.forward(time1)
        enc2 = self.forward(time2)
        
        return F.cosine_similarity(enc1, enc2, dim=-1)
    
    def get_frequency_response(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get the frequency response of the encoder.
        
        Returns:
            frequencies: The learned/fixed frequencies
            phases: The learned/fixed phase shifts
        """
        return self.w.clone(), self.b.clone()


class PositionalTimeEncoder(nn.Module):
    """Alternative time encoder using positional encoding similar to Transformers.
    
    Uses sine and cosine functions at different frequencies for time encoding.
    Can be more stable than learnable Fourier features in some cases.
    """
    
    def __init__(self, time_dim: int, max_time: float = 10000.0):
        super().__init__()
        self.time_dim = time_dim
        self.max_time = max_time
        
        # Create position encoding matrix
        pe = torch.zeros(int(max_time), time_dim)
        position = torch.arange(0, max_time, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, time_dim, 2).float() * 
                           (-math.log(10000.0) / time_dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
    
    def forward(self, timestamps: torch.Tensor) -> torch.Tensor:
        """Encode timestamps using positional encoding.
        
        Args:
            timestamps: Continuous timestamp values
            
        Returns:
            Positional encodings for the timestamps
        """
        # Clamp timestamps to valid range
        timestamps = torch.clamp(timestamps, 0, self.max_time - 1).long()
        
        return self.pe[timestamps]


class MultiScaleTimeEncoder(nn.Module):
    """Multi-scale temporal encoder that combines encodings at different time scales.
    
    Useful for capturing both short-term and long-term temporal patterns.
    """
    
    def __init__(
        self,
        time_dim: int,
        scales: Optional[list] = None,
        aggregation: str = "concat"
    ):
        super().__init__()
        
        if scales is None:
            scales = [1.0, 10.0, 100.0, 1000.0]
        
        self.scales = scales
        self.aggregation = aggregation
        
        # Create encoders for different scales
        if aggregation == "concat":
            scale_dim = time_dim // len(scales)
            self.encoders = nn.ModuleList([
                EdgeTimeEncoder(scale_dim, max_time=scale) 
                for scale in scales
            ])
        elif aggregation == "sum":
            self.encoders = nn.ModuleList([
                EdgeTimeEncoder(time_dim, max_time=scale) 
                for scale in scales
            ])
        else:
            raise ValueError(f"Unknown aggregation: {aggregation}")
    
    def forward(self, timestamps: torch.Tensor) -> torch.Tensor:
        """Encode timestamps at multiple scales."""
        encodings = [encoder(timestamps) for encoder in self.encoders]
        
        if self.aggregation == "concat":
            return torch.cat(encodings, dim=-1)
        elif self.aggregation == "sum":
            return torch.stack(encodings, dim=0).sum(dim=0)
        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")