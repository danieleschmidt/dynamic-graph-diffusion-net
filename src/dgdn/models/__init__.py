"""Core DGDN models and architectures."""

from .dgdn import DynamicGraphDiffusionNet
from .layers import DGDNLayer, MultiHeadTemporalAttention

# Advanced models (optional - will be loaded if available)
try:
    from .advanced import (
        FoundationDGDN,
        ContinuousDGDN,
        FederatedDGDN,
        ExplainableDGDN,
        MultiScaleDGDN
    )
    _ADVANCED_MODELS_AVAILABLE = True
except ImportError:
    _ADVANCED_MODELS_AVAILABLE = False
    FoundationDGDN = None
    ContinuousDGDN = None
    FederatedDGDN = None
    ExplainableDGDN = None
    MultiScaleDGDN = None

__all__ = [
    "DynamicGraphDiffusionNet", 
    "DGDNLayer", 
    "MultiHeadTemporalAttention",
    # Advanced research models
    "FoundationDGDN",
    "ContinuousDGDN",
    "FederatedDGDN", 
    "ExplainableDGDN",
    "MultiScaleDGDN"
]