"""Neuromorphic computing extensions for DGDN."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, List, Tuple
import math

from ..models.dgdn import DynamicGraphDiffusionNet


class SpikingTimeEncoder(nn.Module):
    """Spiking neural network-based temporal encoder."""
    
    def __init__(
        self,
        time_dim: int,
        num_neurons: int = 100,
        threshold: float = 1.0,
        decay: float = 0.95,
        spike_fn: str = "heaviside"
    ):
        super().__init__()
        
        self.time_dim = time_dim
        self.num_neurons = num_neurons
        self.threshold = threshold
        self.decay = decay
        
        # Learnable parameters
        self.weight = nn.Parameter(torch.randn(num_neurons, time_dim))
        self.bias = nn.Parameter(torch.zeros(num_neurons))
        
        # Spike function
        if spike_fn == "heaviside":
            self.spike_fn = self._heaviside_spike
        elif spike_fn == "sigmoid":
            self.spike_fn = self._sigmoid_spike
        else:
            raise ValueError(f"Unknown spike function: {spike_fn}")
            
        # State variables
        self.register_buffer('membrane_potential', torch.zeros(1, num_neurons))
        self.register_buffer('spike_history', torch.zeros(1, num_neurons))
        
    def _heaviside_spike(self, membrane_potential):
        """Heaviside step function for spiking."""
        return (membrane_potential > self.threshold).float()
    
    def _sigmoid_spike(self, membrane_potential):
        """Smooth approximation of spike function."""
        return torch.sigmoid(10 * (membrane_potential - self.threshold))
        
    def forward(self, timestamps: torch.Tensor):
        """Encode timestamps using spiking neurons."""
        batch_size = timestamps.size(0)
        
        # Expand buffers if needed
        if self.membrane_potential.size(0) != batch_size:
            self.membrane_potential = self.membrane_potential.expand(batch_size, -1).contiguous()
            self.spike_history = self.spike_history.expand(batch_size, -1).contiguous()
        
        # Input current based on timestamp
        input_current = torch.sin(timestamps.unsqueeze(-1) * self.weight.t()) + self.bias
        
        # Update membrane potential
        self.membrane_potential = self.decay * self.membrane_potential + input_current
        
        # Generate spikes
        spikes = self.spike_fn(self.membrane_potential)
        
        # Reset membrane potential for spiked neurons
        self.membrane_potential = self.membrane_potential * (1 - spikes)
        
        # Update spike history
        self.spike_history = 0.9 * self.spike_history + spikes
        
        # Project to time dimension
        time_encoding = F.linear(self.spike_history, self.weight)
        
        return time_encoding


class NeuromorphicDGDN(DynamicGraphDiffusionNet):
    """DGDN with neuromorphic computing elements."""
    
    def __init__(
        self,
        *args,
        spiking_layers: List[int] = [0, 2],
        spike_threshold: float = 1.0,
        membrane_decay: float = 0.95,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        
        self.spiking_layers = spiking_layers
        self.spike_threshold = spike_threshold
        self.membrane_decay = membrane_decay
        
        # Replace time encoder with spiking version
        self.time_encoder = SpikingTimeEncoder(
            time_dim=self.time_encoder.time_dim,
            threshold=spike_threshold,
            decay=membrane_decay
        )
        
        # Add spiking neurons to specified layers
        self.spiking_neurons = nn.ModuleDict()
        for layer_idx in spiking_layers:
            if layer_idx < len(self.diffusion_layers):
                self.spiking_neurons[str(layer_idx)] = SpikingNeuronLayer(
                    self.hidden_dim,
                    threshold=spike_threshold,
                    decay=membrane_decay
                )
    
    def forward(self, data):
        """Forward pass with neuromorphic components."""
        # Standard DGDN forward with spiking modifications
        x = data.x
        edge_index = data.edge_index
        edge_attr = getattr(data, 'edge_attr', None)
        timestamps = data.timestamps
        
        # Project to hidden dimension
        h = self.node_projection(x)
        
        # Time encoding with spiking neurons
        time_emb = self.time_encoder(timestamps)
        
        # Diffusion layers with selective spiking
        for i, layer in enumerate(self.diffusion_layers):
            h_prev = h
            h, uncertainty = layer(h, edge_index, edge_attr, time_emb)
            
            # Apply spiking neurons if specified
            if str(i) in self.spiking_neurons:
                h = self.spiking_neurons[str(i)](h)
                
        return {
            'node_embeddings': h,
            'uncertainty': uncertainty,
            'spike_activity': self._compute_spike_activity()
        }
    
    def _compute_spike_activity(self):
        """Compute overall spike activity metrics."""
        total_activity = 0
        num_layers = 0
        
        # Activity from time encoder
        if hasattr(self.time_encoder, 'spike_history'):
            total_activity += self.time_encoder.spike_history.mean()
            num_layers += 1
            
        # Activity from spiking layers
        for layer in self.spiking_neurons.values():
            if hasattr(layer, 'spike_rate'):
                total_activity += layer.spike_rate
                num_layers += 1
                
        return total_activity / max(num_layers, 1)


class SpikingNeuronLayer(nn.Module):
    """Layer of spiking neurons for neuromorphic processing."""
    
    def __init__(
        self,
        dim: int,
        threshold: float = 1.0,
        decay: float = 0.95,
        refractory_period: int = 1
    ):
        super().__init__()
        
        self.dim = dim
        self.threshold = threshold
        self.decay = decay
        self.refractory_period = refractory_period
        
        # State variables
        self.register_buffer('membrane_potential', torch.zeros(1, dim))
        self.register_buffer('refractory_timer', torch.zeros(1, dim))
        self.register_buffer('spike_rate', torch.tensor(0.0))
        
    def forward(self, x: torch.Tensor):
        """Forward pass through spiking neurons."""
        batch_size = x.size(0)
        
        # Expand state buffers
        if self.membrane_potential.size(0) != batch_size:
            self.membrane_potential = self.membrane_potential.expand(batch_size, -1).contiguous()
            self.refractory_timer = self.refractory_timer.expand(batch_size, -1).contiguous()
        
        # Update membrane potential
        self.membrane_potential = self.decay * self.membrane_potential + x
        
        # Apply refractory period
        active_mask = self.refractory_timer <= 0
        self.membrane_potential = self.membrane_potential * active_mask.float()
        
        # Generate spikes
        spike_mask = (self.membrane_potential > self.threshold) & active_mask
        spikes = spike_mask.float()
        
        # Reset spiked neurons
        self.membrane_potential = self.membrane_potential * (1 - spikes)
        
        # Set refractory period for spiked neurons
        self.refractory_timer = torch.where(
            spike_mask,
            torch.tensor(self.refractory_period, device=x.device),
            torch.clamp(self.refractory_timer - 1, min=0)
        )
        
        # Update spike rate (exponential moving average)
        current_spike_rate = spikes.mean()
        self.spike_rate = 0.9 * self.spike_rate + 0.1 * current_spike_rate
        
        # Output is combination of membrane potential and spikes
        output = self.membrane_potential + spikes * self.threshold
        
        return output


class AdaptiveSpikingLayer(nn.Module):
    """Adaptive spiking layer with learnable thresholds."""
    
    def __init__(
        self,
        dim: int,
        min_threshold: float = 0.5,
        max_threshold: float = 2.0
    ):
        super().__init__()
        
        self.dim = dim
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        
        # Learnable adaptive thresholds
        self.threshold_logits = nn.Parameter(torch.zeros(dim))
        
        # State variables
        self.register_buffer('membrane_potential', torch.zeros(1, dim))
        self.register_buffer('adaptation_state', torch.ones(1, dim))
        
    def forward(self, x: torch.Tensor):
        """Adaptive spiking forward pass."""
        batch_size = x.size(0)
        
        # Expand state buffers
        if self.membrane_potential.size(0) != batch_size:
            self.membrane_potential = self.membrane_potential.expand(batch_size, -1).contiguous()
            self.adaptation_state = self.adaptation_state.expand(batch_size, -1).contiguous()
        
        # Compute adaptive thresholds
        thresholds = self.min_threshold + (self.max_threshold - self.min_threshold) * torch.sigmoid(self.threshold_logits)
        
        # Update membrane potential
        self.membrane_potential = 0.9 * self.membrane_potential + x
        
        # Generate spikes with adaptive thresholds
        spike_mask = self.membrane_potential > thresholds.unsqueeze(0)
        spikes = spike_mask.float()
        
        # Reset and adapt
        self.membrane_potential = self.membrane_potential * (1 - spikes)
        self.adaptation_state = 0.95 * self.adaptation_state + 0.05 * spikes
        
        return self.membrane_potential + spikes


def demonstrate_neuromorphic_dgdn():
    """Demonstrate neuromorphic DGDN capabilities."""
    print("🧠 Neuromorphic DGDN Demo")
    print("=" * 40)
    
    # Create neuromorphic model
    model = NeuromorphicDGDN(
        node_dim=32,
        edge_dim=16,
        hidden_dim=64,
        num_layers=3,
        spiking_layers=[0, 2],
        spike_threshold=1.0
    )
    
    # Generate test data
    num_nodes = 50
    num_edges = 100
    
    data = type('Data', (), {
        'x': torch.randn(num_nodes, 32),
        'edge_index': torch.randint(0, num_nodes, (2, num_edges)),
        'edge_attr': torch.randn(num_edges, 16),
        'timestamps': torch.rand(num_edges) * 10
    })()
    
    # Forward pass
    output = model(data)
    
    print(f"✅ Node embeddings: {output['node_embeddings'].shape}")
    print(f"✅ Spike activity: {output['spike_activity']:.4f}")
    print(f"✅ Uncertainty: {output['uncertainty'].mean():.4f}")
    
    return model


if __name__ == "__main__":
    demonstrate_neuromorphic_dgdn()