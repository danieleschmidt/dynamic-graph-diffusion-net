"""Dynamic Graph Diffusion Net (DGDN) - PyTorch Implementation.

A PyTorch library implementing the Dynamic Graph Diffusion Network architecture
for temporal graph learning, as proposed in the ICLR 2025 paper.
"""

__version__ = "0.1.0"
__author__ = "Daniel Schmidt"
__email__ = "author@example.com"
__license__ = "MIT"

# Core model imports
from .models import DynamicGraphDiffusionNet, DGDNLayer, MultiHeadTemporalAttention

# Data handling imports
from .data import TemporalData, TemporalDataset, TemporalGraphDataset, TemporalDataLoader

# Temporal processing imports
from .temporal import EdgeTimeEncoder, VariationalDiffusion

# Training imports
from .training import DGDNTrainer, DGDNLoss, DGDNMetrics

__all__ = [
    "__version__",
    "__author__",
    "__email__",
    "__license__",
    # Core models
    "DynamicGraphDiffusionNet",
    "DGDNLayer", 
    "MultiHeadTemporalAttention",
    # Data structures
    "TemporalData",
    "TemporalDataset",
    "TemporalGraphDataset", 
    "TemporalDataLoader",
    # Temporal processing
    "EdgeTimeEncoder",
    "VariationalDiffusion",
    # Training
    "DGDNTrainer",
    "DGDNLoss",
    "DGDNMetrics",
]