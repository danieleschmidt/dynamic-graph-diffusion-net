"""Graph data fixtures for testing."""

import numpy as np
import torch
from typing import Tuple, Dict, Any
import pytest


@pytest.fixture
def small_temporal_graph() -> Dict[str, Any]:
    """Create a small temporal graph for testing.
    
    Returns:
        Dict containing edge_index, edge_attr, timestamps, and node_features
    """
    # 10 nodes, 20 edges over 5 time steps
    num_nodes = 10
    num_edges = 20
    num_timesteps = 5
    
    # Random edge connections
    np.random.seed(42)
    source_nodes = np.random.randint(0, num_nodes, num_edges)
    target_nodes = np.random.randint(0, num_nodes, num_edges)
    
    # Ensure no self-loops
    mask = source_nodes != target_nodes
    source_nodes = source_nodes[mask][:num_edges//2]
    target_nodes = target_nodes[mask][:num_edges//2]
    
    edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)
    
    # Random timestamps
    timestamps = torch.rand(len(source_nodes)) * num_timesteps
    timestamps = torch.sort(timestamps)[0]  # Sort for temporal consistency
    
    # Random edge features
    edge_attr = torch.randn(len(source_nodes), 16)
    
    # Random node features
    node_features = torch.randn(num_nodes, 32)
    
    return {
        'edge_index': edge_index,
        'edge_attr': edge_attr,
        'timestamps': timestamps,
        'node_features': node_features,
        'num_nodes': num_nodes
    }


@pytest.fixture
def medium_temporal_graph() -> Dict[str, Any]:
    """Create a medium-sized temporal graph for testing.
    
    Returns:
        Dict containing edge_index, edge_attr, timestamps, and node_features
    """
    # 100 nodes, 500 edges over 50 time steps
    num_nodes = 100
    num_edges = 500
    num_timesteps = 50
    
    np.random.seed(123)
    
    # Create more realistic graph structure
    edges = []
    for t in range(num_timesteps):
        # Add some edges at each timestep
        n_edges_t = np.random.poisson(10)
        for _ in range(n_edges_t):
            source = np.random.randint(0, num_nodes)
            target = np.random.randint(0, num_nodes)
            if source != target:
                edges.append((source, target, t + np.random.random()))
    
    # Convert to tensors
    if edges:
        edge_array = np.array(edges)
        edge_index = torch.tensor(edge_array[:, :2].T, dtype=torch.long)
        timestamps = torch.tensor(edge_array[:, 2], dtype=torch.float)
        
        # Sort by timestamp
        sort_idx = torch.argsort(timestamps)
        edge_index = edge_index[:, sort_idx]
        timestamps = timestamps[sort_idx]
        
        edge_attr = torch.randn(len(timestamps), 32)
    else:
        edge_index = torch.tensor([[], []], dtype=torch.long)
        timestamps = torch.tensor([], dtype=torch.float)
        edge_attr = torch.tensor([], dtype=torch.float).reshape(0, 32)
    
    node_features = torch.randn(num_nodes, 64)
    
    return {
        'edge_index': edge_index,
        'edge_attr': edge_attr,
        'timestamps': timestamps,
        'node_features': node_features,
        'num_nodes': num_nodes
    }


@pytest.fixture
def batch_temporal_graphs() -> list:
    """Create a batch of temporal graphs for testing.
    
    Returns:
        List of temporal graph dictionaries
    """
    graphs = []
    
    for i in range(3):
        np.random.seed(i * 100)
        num_nodes = np.random.randint(5, 20)
        num_edges = np.random.randint(10, 50)
        
        source_nodes = np.random.randint(0, num_nodes, num_edges)
        target_nodes = np.random.randint(0, num_nodes, num_edges)
        
        # Remove self-loops
        mask = source_nodes != target_nodes
        source_nodes = source_nodes[mask]
        target_nodes = target_nodes[mask]
        
        if len(source_nodes) == 0:
            continue
            
        edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)
        timestamps = torch.rand(len(source_nodes)) * 10
        timestamps = torch.sort(timestamps)[0]
        
        edge_attr = torch.randn(len(source_nodes), 8)
        node_features = torch.randn(num_nodes, 16)
        
        graphs.append({
            'edge_index': edge_index,
            'edge_attr': edge_attr,
            'timestamps': timestamps,
            'node_features': node_features,
            'num_nodes': num_nodes
        })
    
    return graphs


@pytest.fixture
def temporal_graph_sequence() -> Dict[str, Any]:
    """Create a sequence of temporal snapshots for testing.
    
    Returns:
        Dict containing a sequence of graph snapshots
    """
    num_nodes = 20
    num_timesteps = 10
    snapshots = []
    
    np.random.seed(456)
    
    for t in range(num_timesteps):
        # Gradually evolving graph
        n_edges = max(5, int(20 * (1 + 0.1 * t)))
        
        source_nodes = np.random.randint(0, num_nodes, n_edges)
        target_nodes = np.random.randint(0, num_nodes, n_edges)
        
        # Remove self-loops
        mask = source_nodes != target_nodes
        source_nodes = source_nodes[mask]
        target_nodes = target_nodes[mask]
        
        if len(source_nodes) == 0:
            continue
            
        edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)
        timestamps = torch.ones(len(source_nodes)) * t
        edge_attr = torch.randn(len(source_nodes), 12)
        
        snapshots.append({
            'edge_index': edge_index,
            'edge_attr': edge_attr,
            'timestamps': timestamps,
            'time': t
        })
    
    node_features = torch.randn(num_nodes, 24)
    
    return {
        'snapshots': snapshots,
        'node_features': node_features,
        'num_nodes': num_nodes,
        'num_timesteps': num_timesteps
    }


@pytest.fixture
def edge_prediction_data() -> Dict[str, Any]:
    """Create data for edge prediction testing.
    
    Returns:
        Dict containing train/val/test splits with labels
    """
    # Create base temporal graph
    num_nodes = 50
    num_edges = 200
    
    np.random.seed(789)
    
    # Generate edges with timestamps
    source_nodes = np.random.randint(0, num_nodes, num_edges)
    target_nodes = np.random.randint(0, num_nodes, num_edges)
    
    # Remove self-loops
    mask = source_nodes != target_nodes
    source_nodes = source_nodes[mask]
    target_nodes = target_nodes[mask]
    
    timestamps = np.sort(np.random.random(len(source_nodes)) * 100)
    
    # Split by time
    train_mask = timestamps < 60
    val_mask = (timestamps >= 60) & (timestamps < 80)
    test_mask = timestamps >= 80
    
    def create_split(mask):
        if not np.any(mask):
            return {
                'edge_index': torch.tensor([[], []], dtype=torch.long),
                'timestamps': torch.tensor([], dtype=torch.float),
                'labels': torch.tensor([], dtype=torch.float)
            }
            
        split_sources = source_nodes[mask]
        split_targets = target_nodes[mask]
        split_times = timestamps[mask]
        
        # Positive edges (observed)
        pos_edges = torch.tensor([split_sources, split_targets], dtype=torch.long)
        pos_times = torch.tensor(split_times, dtype=torch.float)
        pos_labels = torch.ones(len(split_sources))
        
        # Negative edges (unobserved)
        neg_sources = np.random.randint(0, num_nodes, len(split_sources))
        neg_targets = np.random.randint(0, num_nodes, len(split_sources))
        neg_mask = neg_sources != neg_targets
        neg_sources = neg_sources[neg_mask]
        neg_targets = neg_targets[neg_mask]
        
        if len(neg_sources) > 0:
            neg_edges = torch.tensor([neg_sources, neg_targets], dtype=torch.long)
            neg_times = torch.tensor(split_times[:len(neg_sources)], dtype=torch.float)
            neg_labels = torch.zeros(len(neg_sources))
            
            # Combine positive and negative
            all_edges = torch.cat([pos_edges, neg_edges], dim=1)
            all_times = torch.cat([pos_times, neg_times])
            all_labels = torch.cat([pos_labels, neg_labels])
        else:
            all_edges = pos_edges
            all_times = pos_times
            all_labels = pos_labels
        
        return {
            'edge_index': all_edges,
            'timestamps': all_times,
            'labels': all_labels
        }
    
    node_features = torch.randn(num_nodes, 32)
    
    return {
        'train': create_split(train_mask),
        'val': create_split(val_mask),
        'test': create_split(test_mask),
        'node_features': node_features,
        'num_nodes': num_nodes
    }


@pytest.fixture
def benchmark_data() -> Dict[str, Any]:
    """Create benchmark data for performance testing.
    
    Returns:
        Dict containing different sized graphs for benchmarking
    """
    datasets = {}
    
    sizes = [
        ('tiny', 10, 20),
        ('small', 100, 500),
        ('medium', 1000, 5000),
        ('large', 5000, 25000)
    ]
    
    for name, num_nodes, num_edges in sizes:
        np.random.seed(hash(name) % 2**32)
        
        source_nodes = np.random.randint(0, num_nodes, num_edges)
        target_nodes = np.random.randint(0, num_nodes, num_edges)
        
        # Remove self-loops
        mask = source_nodes != target_nodes
        source_nodes = source_nodes[mask]
        target_nodes = target_nodes[mask]
        
        edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)
        timestamps = torch.sort(torch.rand(len(source_nodes)) * 1000)[0]
        edge_attr = torch.randn(len(source_nodes), 64)
        node_features = torch.randn(num_nodes, 128)
        
        datasets[name] = {
            'edge_index': edge_index,
            'edge_attr': edge_attr,
            'timestamps': timestamps,
            'node_features': node_features,
            'num_nodes': num_nodes
        }
    
    return datasets