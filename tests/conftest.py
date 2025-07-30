"""Pytest configuration and shared fixtures."""

import pytest
import torch
import numpy as np


@pytest.fixture
def device():
    """Return the appropriate device for testing."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def sample_graph_data():
    """Generate sample temporal graph data for testing."""
    num_nodes = 100
    num_edges = 500
    
    # Random edge indices
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    
    # Random edge features
    edge_attr = torch.randn(num_edges, 32)
    
    # Random timestamps
    timestamps = torch.sort(torch.rand(num_edges) * 1000)[0]
    
    # Random node features
    node_features = torch.randn(num_nodes, 64)
    
    return {
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "timestamps": timestamps,
        "node_features": node_features,
    }


@pytest.fixture
def small_graph_data():
    """Generate small graph data for quick tests."""
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
    edge_attr = torch.randn(3, 16)
    timestamps = torch.tensor([1.0, 2.0, 3.0])
    node_features = torch.randn(3, 32)
    
    return {
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "timestamps": timestamps,
        "node_features": node_features,
    }


@pytest.fixture(autouse=True)
def set_random_seeds():
    """Set random seeds for reproducible tests."""
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)