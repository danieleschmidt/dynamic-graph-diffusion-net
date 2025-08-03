"""Variational diffusion sampling module for DGDN.

This module implements the core innovation of DGDN: probabilistic message passing
with uncertainty quantification through multi-step diffusion processes.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree


class VariationalDiffusion(nn.Module):
    """Variational Diffusion Sampler for probabilistic node embeddings.
    
    Implements multi-step diffusion process with variational inference framework
    for uncertainty quantification in dynamic graph learning.
    
    Args:
        hidden_dim: Dimension of node embeddings
        num_diffusion_steps: Number of diffusion steps (default: 5)
        num_heads: Number of attention heads (default: 8) 
        dropout: Dropout probability (default: 0.1)
        activation: Activation function (default: "relu")
        noise_schedule: Type of noise schedule ("linear", "cosine") (default: "linear")
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_diffusion_steps: int = 5,
        num_heads: int = 8,
        dropout: float = 0.1,
        activation: str = "relu",
        noise_schedule: str = "linear"
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_diffusion_steps = num_diffusion_steps
        self.num_heads = num_heads
        self.dropout = dropout
        
        # Create diffusion layers
        self.diffusion_layers = nn.ModuleList([
            DiffusionLayer(hidden_dim, num_heads, dropout, activation)
            for _ in range(num_diffusion_steps)
        ])
        
        # Noise schedule parameters
        self.noise_schedule = noise_schedule
        self.register_buffer("betas", self._get_noise_schedule(num_diffusion_steps))
        self.register_buffer("alphas", 1.0 - self.betas)
        self.register_buffer("alpha_cumprod", torch.cumprod(self.alphas, dim=0))
        
        # Prior distribution parameters (standard normal)
        self.register_buffer("prior_mean", torch.zeros(hidden_dim))
        self.register_buffer("prior_logvar", torch.zeros(hidden_dim))
    
    def _get_noise_schedule(self, num_steps: int) -> torch.Tensor:
        """Generate noise schedule for diffusion process."""
        if self.noise_schedule == "linear":
            return torch.linspace(0.0001, 0.02, num_steps)
        elif self.noise_schedule == "cosine":
            # Cosine schedule from "Improved Denoising Diffusion Probabilistic Models"
            s = 0.008
            steps = torch.arange(num_steps + 1, dtype=torch.float32) / num_steps
            alphas_cumprod = torch.cos((steps + s) / (1 + s) * math.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            return torch.clamp(betas, 0.0001, 0.9999)
        else:
            raise ValueError(f"Unknown noise schedule: {self.noise_schedule}")
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        return_all_steps: bool = False
    ) -> Dict[str, torch.Tensor]:
        """Forward diffusion process.
        
        Args:
            x: Node features [num_nodes, hidden_dim]
            edge_index: Edge indices [2, num_edges]
            edge_attr: Edge attributes [num_edges, edge_dim] (optional)
            return_all_steps: Whether to return intermediate diffusion steps
            
        Returns:
            Dictionary containing:
            - z: Final node embeddings [num_nodes, hidden_dim]
            - mean: Mean of final distribution [num_nodes, hidden_dim]
            - logvar: Log variance of final distribution [num_nodes, hidden_dim]
            - kl_loss: KL divergence loss
            - all_steps: All intermediate steps (if return_all_steps=True)
        """
        batch_size, num_nodes = x.shape[0], x.shape[1] if x.dim() == 3 else x.shape[0]
        
        # Initialize with input features
        current_mean = x
        current_logvar = torch.zeros_like(x)
        
        all_steps = [] if return_all_steps else None
        total_kl_loss = 0.0
        
        # Forward diffusion process
        for step, layer in enumerate(self.diffusion_layers):
            # Apply diffusion layer
            step_output = layer(
                current_mean, 
                current_logvar,
                edge_index, 
                edge_attr,
                step
            )
            
            current_mean = step_output["mean"]
            current_logvar = step_output["logvar"]
            z_step = step_output["z"]
            kl_step = step_output["kl_loss"]
            
            total_kl_loss += kl_step
            
            if return_all_steps:
                all_steps.append({
                    "step": step,
                    "z": z_step.clone(),
                    "mean": current_mean.clone(),
                    "logvar": current_logvar.clone(),
                    "kl_loss": kl_step
                })
        
        # Sample from final distribution
        if self.training:
            std = torch.exp(0.5 * current_logvar)
            eps = torch.randn_like(std)
            final_z = current_mean + eps * std
        else:
            final_z = current_mean
        
        result = {
            "z": final_z,
            "mean": current_mean,
            "logvar": current_logvar,
            "kl_loss": total_kl_loss,
        }
        
        if return_all_steps:
            result["all_steps"] = all_steps
            
        return result
    
    def sample(
        self,
        num_nodes: int,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """Sample from the learned distribution.
        
        Args:
            num_nodes: Number of nodes to sample
            edge_index: Edge connectivity
            edge_attr: Edge attributes (optional)
            device: Device to create tensors on
            
        Returns:
            Sampled node embeddings
        """
        if device is None:
            device = next(self.parameters()).device
        
        # Start from prior (standard normal)
        z = torch.randn(num_nodes, self.hidden_dim, device=device)
        
        # Reverse diffusion process (denoising)
        with torch.no_grad():
            for step in reversed(range(self.num_diffusion_steps)):
                layer = self.diffusion_layers[step]
                # In sampling, we reverse the process
                z = layer.reverse_step(z, edge_index, edge_attr, step)
        
        return z
    
    def get_uncertainty(self, logvar: torch.Tensor) -> torch.Tensor:
        """Convert log variance to uncertainty measure.
        
        Args:
            logvar: Log variance tensor
            
        Returns:
            Uncertainty measure (standard deviation)
        """
        return torch.exp(0.5 * logvar)
    
    def compute_mutual_information(
        self,
        z_mean: torch.Tensor,
        z_logvar: torch.Tensor
    ) -> torch.Tensor:
        """Compute mutual information between embeddings and inputs."""
        # Approximate MI using variational bound
        # MI(Z; X) ≈ E[log q(z|x)] - E[log q(z)]
        entropy_conditional = 0.5 * (1 + z_logvar).sum(dim=-1)
        entropy_marginal = 0.5 * self.hidden_dim * (1 + math.log(2 * math.pi))
        
        mi = entropy_conditional - entropy_marginal
        return mi.mean()


class DiffusionLayer(MessagePassing):
    """Single diffusion layer with attention-based message passing."""
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        activation: str = "relu"
    ):
        super().__init__(aggr='add', node_dim=-2)
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.dropout = dropout
        
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        # Multi-head attention components
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Diffusion network for mean and variance
        self.diffusion_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 4),
            self._get_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim * 2)  # Output: [mean, logvar]
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        self.reset_parameters()
    
    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function by name."""
        if activation == "relu":
            return nn.ReLU()
        elif activation == "gelu":
            return nn.GELU()
        elif activation == "swish":
            return nn.SiLU()
        else:
            raise ValueError(f"Unknown activation: {activation}")
    
    def reset_parameters(self):
        """Initialize parameters."""
        for module in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)
        
        # Initialize diffusion network
        for module in self.diffusion_net:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(
        self,
        x_mean: torch.Tensor,
        x_logvar: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        diffusion_step: int = 0
    ) -> Dict[str, torch.Tensor]:
        """Forward pass of diffusion layer."""
        
        # Sample from current distribution
        if self.training:
            std = torch.exp(0.5 * x_logvar)
            eps = torch.randn_like(std)
            x = x_mean + eps * std
        else:
            x = x_mean
        
        # Message passing with attention
        messages = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        
        # Combine with current embedding
        combined = torch.cat([x, messages], dim=-1)
        
        # Apply diffusion network
        diffusion_params = self.diffusion_net(combined)
        new_mean, new_logvar = diffusion_params.chunk(2, dim=-1)
        
        # Residual connection and layer norm
        new_mean = self.layer_norm(new_mean + x_mean)
        
        # Compute KL divergence with previous step
        kl_loss = self._compute_kl_loss(x_mean, x_logvar, new_mean, new_logvar)
        
        # Sample from new distribution
        if self.training:
            std = torch.exp(0.5 * new_logvar)
            eps = torch.randn_like(std)
            z = new_mean + eps * std
        else:
            z = new_mean
        
        return {
            "z": z,
            "mean": new_mean,
            "logvar": new_logvar,
            "kl_loss": kl_loss
        }
    
    def message(self, x_j: torch.Tensor, edge_attr: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute messages using multi-head attention."""
        batch_size, num_edges = x_j.shape[0], x_j.shape[1] if x_j.dim() == 3 else x_j.shape[0]
        
        # Project to query, key, value
        q = self.q_proj(x_j)
        k = self.k_proj(x_j)
        v = self.v_proj(x_j)
        
        # Reshape for multi-head attention
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_heads, self.head_dim)
        v = v.view(-1, self.num_heads, self.head_dim)
        
        # Compute attention scores
        attn_scores = torch.sum(q * k, dim=-1) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        # Apply dropout
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)
        
        # Apply attention to values
        attn_output = attn_weights.unsqueeze(-1) * v
        attn_output = attn_output.view(-1, self.hidden_dim)
        
        # Final projection
        messages = self.out_proj(attn_output)
        
        return messages
    
    def _compute_kl_loss(
        self,
        mean1: torch.Tensor,
        logvar1: torch.Tensor,
        mean2: torch.Tensor,
        logvar2: torch.Tensor
    ) -> torch.Tensor:
        """Compute KL divergence between two Gaussian distributions."""
        # KL(q1 || q2) = 0.5 * [log(σ2²/σ1²) + (σ1² + (μ1-μ2)²)/σ2² - 1]
        var1 = torch.exp(logvar1)
        var2 = torch.exp(logvar2)
        
        kl = 0.5 * (
            logvar2 - logvar1 +
            (var1 + (mean1 - mean2) ** 2) / var2 - 1
        )
        
        return kl.sum(dim=-1).mean()
    
    def reverse_step(
        self,
        z: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        step: int = 0
    ) -> torch.Tensor:
        """Reverse diffusion step for sampling."""
        # This is a simplified reverse step
        # In practice, this would need the learned reverse process
        messages = self.propagate(edge_index, x=z, edge_attr=edge_attr)
        combined = torch.cat([z, messages], dim=-1)
        
        # Apply reverse transformation (simplified)
        diffusion_params = self.diffusion_net(combined)
        new_mean, _ = diffusion_params.chunk(2, dim=-1)
        
        return self.layer_norm(new_mean + z)