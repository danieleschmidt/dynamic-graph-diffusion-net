"""Test helper functions for DGDN testing."""

import torch
import numpy as np
import time
from typing import Dict, Any, List, Optional, Callable, Union
from contextlib import contextmanager
import tempfile
import os
import shutil


def set_random_seeds(seed: int = 42) -> None:
    """Set random seeds for reproducible testing.
    
    Args:
        seed: Random seed value
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Get the best available device for testing.
    
    Returns:
        torch.device: Best available device
    """
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """Count model parameters.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dict with parameter counts
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'non_trainable': total_params - trainable_params
    }


def get_model_size_mb(model: torch.nn.Module) -> float:
    """Get model size in megabytes.
    
    Args:
        model: PyTorch model
        
    Returns:
        Model size in MB
    """
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    return (param_size + buffer_size) / (1024 ** 2)


@contextmanager
def temporary_directory():
    """Context manager for temporary directory.
    
    Yields:
        str: Path to temporary directory
    """
    temp_dir = tempfile.mkdtemp()
    try:
        yield temp_dir
    finally:
        shutil.rmtree(temp_dir)


@contextmanager
def timer():
    """Context manager for timing code execution.
    
    Yields:
        dict: Dictionary that will contain 'elapsed' time
    """
    result = {}
    start = time.time()
    try:
        yield result
    finally:
        result['elapsed'] = time.time() - start


def create_mock_data(
    num_nodes: int,
    num_edges: int,
    node_dim: int,
    edge_dim: int,
    num_timesteps: int = 10,
    seed: int = 42
) -> Dict[str, torch.Tensor]:
    """Create mock temporal graph data.
    
    Args:
        num_nodes: Number of nodes
        num_edges: Number of edges
        node_dim: Node feature dimension
        edge_dim: Edge feature dimension
        num_timesteps: Number of time steps
        seed: Random seed
        
    Returns:
        Dict containing mock data tensors
    """
    set_random_seeds(seed)
    
    # Create edges
    source_nodes = torch.randint(0, num_nodes, (num_edges,))
    target_nodes = torch.randint(0, num_nodes, (num_edges,))
    
    # Remove self-loops
    mask = source_nodes != target_nodes
    source_nodes = source_nodes[mask]
    target_nodes = target_nodes[mask]
    
    edge_index = torch.stack([source_nodes, target_nodes])
    
    # Create timestamps
    timestamps = torch.rand(edge_index.size(1)) * num_timesteps
    timestamps = torch.sort(timestamps)[0]
    
    # Create features
    node_features = torch.randn(num_nodes, node_dim)
    edge_features = torch.randn(edge_index.size(1), edge_dim)
    
    return {
        'edge_index': edge_index,
        'edge_attr': edge_features,
        'timestamps': timestamps,
        'node_features': node_features,
        'num_nodes': num_nodes
    }


def compare_model_outputs(
    output1: Union[torch.Tensor, Dict[str, torch.Tensor]],
    output2: Union[torch.Tensor, Dict[str, torch.Tensor]],
    rtol: float = 1e-5,
    atol: float = 1e-8
) -> bool:
    """Compare two model outputs for equality.
    
    Args:
        output1: First output
        output2: Second output
        rtol: Relative tolerance
        atol: Absolute tolerance
        
    Returns:
        True if outputs are close
    """
    if isinstance(output1, torch.Tensor) and isinstance(output2, torch.Tensor):
        return torch.allclose(output1, output2, rtol=rtol, atol=atol)
    
    elif isinstance(output1, dict) and isinstance(output2, dict):
        if set(output1.keys()) != set(output2.keys()):
            return False
        
        for key in output1.keys():
            if not torch.allclose(output1[key], output2[key], rtol=rtol, atol=atol):
                return False
        
        return True
    
    else:
        return False


def check_gradient_flow(model: torch.nn.Module) -> Dict[str, bool]:
    """Check if gradients are flowing through model parameters.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dict mapping parameter names to gradient flow status
    """
    gradient_flow = {}
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            gradient_flow[name] = param.grad is not None and param.grad.abs().sum() > 0
        else:
            gradient_flow[name] = False
    
    return gradient_flow


def memory_usage() -> Dict[str, float]:
    """Get current memory usage.
    
    Returns:
        Dict with memory usage info
    """
    import psutil
    
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    
    result = {
        'rss_mb': memory_info.rss / (1024 ** 2),
        'vms_mb': memory_info.vms / (1024 ** 2)
    }
    
    if torch.cuda.is_available():
        result['cuda_allocated_mb'] = torch.cuda.memory_allocated() / (1024 ** 2)
        result['cuda_cached_mb'] = torch.cuda.memory_reserved() / (1024 ** 2)
    
    return result


def benchmark_function(
    func: Callable,
    *args,
    num_runs: int = 10,
    warmup_runs: int = 2,
    **kwargs
) -> Dict[str, float]:
    """Benchmark a function's execution time.
    
    Args:
        func: Function to benchmark
        *args: Function arguments
        num_runs: Number of benchmark runs
        warmup_runs: Number of warmup runs
        **kwargs: Function keyword arguments
        
    Returns:
        Dict with timing statistics
    """
    # Warmup
    for _ in range(warmup_runs):
        func(*args, **kwargs)
    
    # Synchronize if using CUDA
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # Benchmark
    times = []
    for _ in range(num_runs):
        start_time = time.time()
        result = func(*args, **kwargs)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        end_time = time.time()
        times.append(end_time - start_time)
    
    times = np.array(times)
    
    return {
        'mean': float(np.mean(times)),
        'std': float(np.std(times)),
        'min': float(np.min(times)),
        'max': float(np.max(times)),
        'median': float(np.median(times))
    }


def create_batch_from_graphs(graphs: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Create a batch from a list of graph dictionaries.
    
    Args:
        graphs: List of graph dictionaries
        
    Returns:
        Batched graph dictionary
    """
    if not graphs:
        return {}
    
    # Get all keys
    keys = set()
    for graph in graphs:
        keys.update(graph.keys())
    
    batched = {}
    node_offset = 0
    
    for key in keys:
        if key == 'num_nodes':
            batched[key] = sum(graph.get(key, 0) for graph in graphs)
        elif key == 'edge_index':
            edge_indices = []
            for graph in graphs:
                if key in graph:
                    edge_index = graph[key].clone()
                    edge_index += node_offset
                    edge_indices.append(edge_index)
                    node_offset += graph.get('num_nodes', edge_index.max().item() + 1)
            
            if edge_indices:
                batched[key] = torch.cat(edge_indices, dim=1)
        else:
            # For other tensors, concatenate along first dimension
            tensors = [graph[key] for graph in graphs if key in graph]
            if tensors:
                batched[key] = torch.cat(tensors, dim=0)
    
    return batched


def save_test_results(results: Dict[str, Any], filepath: str) -> None:
    """Save test results to file.
    
    Args:
        results: Test results dictionary
        filepath: Output file path
    """
    import json
    
    # Convert tensors to lists for JSON serialization
    serializable_results = {}
    for key, value in results.items():
        if isinstance(value, torch.Tensor):
            serializable_results[key] = value.tolist()
        elif isinstance(value, np.ndarray):
            serializable_results[key] = value.tolist()
        else:
            serializable_results[key] = value
    
    with open(filepath, 'w') as f:
        json.dump(serializable_results, f, indent=2)


def load_test_results(filepath: str) -> Dict[str, Any]:
    """Load test results from file.
    
    Args:
        filepath: Input file path
        
    Returns:
        Test results dictionary
    """
    import json
    
    with open(filepath, 'r') as f:
        return json.load(f)


def assert_equal_graphs(
    graph1: Dict[str, torch.Tensor],
    graph2: Dict[str, torch.Tensor],
    rtol: float = 1e-5,
    atol: float = 1e-8
) -> None:
    """Assert that two graphs are equal.
    
    Args:
        graph1: First graph
        graph2: Second graph
        rtol: Relative tolerance
        atol: Absolute tolerance
    """
    assert set(graph1.keys()) == set(graph2.keys()), "Graphs have different keys"
    
    for key in graph1.keys():
        if isinstance(graph1[key], torch.Tensor):
            assert torch.allclose(graph1[key], graph2[key], rtol=rtol, atol=atol), \
                f"Graphs differ in key '{key}'"
        else:
            assert graph1[key] == graph2[key], f"Graphs differ in key '{key}'"


class MockDataLoader:
    """Mock data loader for testing."""
    
    def __init__(self, data: List[Dict[str, torch.Tensor]], batch_size: int = 1):
        self.data = data
        self.batch_size = batch_size
    
    def __iter__(self):
        for i in range(0, len(self.data), self.batch_size):
            batch_data = self.data[i:i + self.batch_size]
            if len(batch_data) == 1:
                yield batch_data[0]
            else:
                yield create_batch_from_graphs(batch_data)
    
    def __len__(self):
        return (len(self.data) + self.batch_size - 1) // self.batch_size