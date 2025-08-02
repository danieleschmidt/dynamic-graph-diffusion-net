"""Custom assertions for DGDN testing."""

import torch
import numpy as np
from typing import Union, Optional, Tuple
import pytest


def assert_tensor_shape(
    tensor: torch.Tensor, 
    expected_shape: Tuple[int, ...],
    msg: Optional[str] = None
) -> None:
    """Assert that a tensor has the expected shape.
    
    Args:
        tensor: Tensor to check
        expected_shape: Expected shape tuple
        msg: Optional custom error message
    """
    actual_shape = tuple(tensor.shape)
    if actual_shape != expected_shape:
        error_msg = f"Expected shape {expected_shape}, got {actual_shape}"
        if msg:
            error_msg = f"{msg}: {error_msg}"
        raise AssertionError(error_msg)


def assert_tensor_dtype(
    tensor: torch.Tensor, 
    expected_dtype: torch.dtype,
    msg: Optional[str] = None
) -> None:
    """Assert that a tensor has the expected dtype.
    
    Args:
        tensor: Tensor to check
        expected_dtype: Expected dtype
        msg: Optional custom error message
    """
    if tensor.dtype != expected_dtype:
        error_msg = f"Expected dtype {expected_dtype}, got {tensor.dtype}"
        if msg:
            error_msg = f"{msg}: {error_msg}"
        raise AssertionError(error_msg)


def assert_tensor_finite(
    tensor: torch.Tensor,
    msg: Optional[str] = None
) -> None:
    """Assert that all values in a tensor are finite.
    
    Args:
        tensor: Tensor to check
        msg: Optional custom error message
    """
    if not torch.isfinite(tensor).all():
        error_msg = "Tensor contains non-finite values (NaN or Inf)"
        if msg:
            error_msg = f"{msg}: {error_msg}"
        raise AssertionError(error_msg)


def assert_tensor_range(
    tensor: torch.Tensor,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
    msg: Optional[str] = None
) -> None:
    """Assert that tensor values are within specified range.
    
    Args:
        tensor: Tensor to check
        min_val: Minimum allowed value (inclusive)
        max_val: Maximum allowed value (inclusive)
        msg: Optional custom error message
    """
    if min_val is not None and tensor.min() < min_val:
        error_msg = f"Tensor contains values below {min_val}: min={tensor.min()}"
        if msg:
            error_msg = f"{msg}: {error_msg}"
        raise AssertionError(error_msg)
    
    if max_val is not None and tensor.max() > max_val:
        error_msg = f"Tensor contains values above {max_val}: max={tensor.max()}"
        if msg:
            error_msg = f"{msg}: {error_msg}"
        raise AssertionError(error_msg)


def assert_tensor_close(
    tensor1: torch.Tensor,
    tensor2: torch.Tensor,
    rtol: float = 1e-5,
    atol: float = 1e-8,
    msg: Optional[str] = None
) -> None:
    """Assert that two tensors are element-wise close.
    
    Args:
        tensor1: First tensor
        tensor2: Second tensor
        rtol: Relative tolerance
        atol: Absolute tolerance
        msg: Optional custom error message
    """
    if not torch.allclose(tensor1, tensor2, rtol=rtol, atol=atol):
        max_diff = torch.max(torch.abs(tensor1 - tensor2))
        error_msg = f"Tensors are not close: max_diff={max_diff}, rtol={rtol}, atol={atol}"
        if msg:
            error_msg = f"{msg}: {error_msg}"
        raise AssertionError(error_msg)


def assert_edge_index_valid(
    edge_index: torch.Tensor,
    num_nodes: int,
    msg: Optional[str] = None
) -> None:
    """Assert that edge_index tensor is valid.
    
    Args:
        edge_index: Edge index tensor of shape [2, num_edges]
        num_nodes: Number of nodes in the graph
        msg: Optional custom error message
    """
    # Check shape
    if edge_index.dim() != 2 or edge_index.size(0) != 2:
        error_msg = f"edge_index must have shape [2, num_edges], got {edge_index.shape}"
        if msg:
            error_msg = f"{msg}: {error_msg}"
        raise AssertionError(error_msg)
    
    # Check node indices are valid
    if edge_index.min() < 0 or edge_index.max() >= num_nodes:
        error_msg = f"edge_index contains invalid node indices: min={edge_index.min()}, max={edge_index.max()}, num_nodes={num_nodes}"
        if msg:
            error_msg = f"{msg}: {error_msg}"
        raise AssertionError(error_msg)
    
    # Check dtype
    if edge_index.dtype != torch.long:
        error_msg = f"edge_index must have dtype torch.long, got {edge_index.dtype}"
        if msg:
            error_msg = f"{msg}: {error_msg}"
        raise AssertionError(error_msg)


def assert_timestamps_sorted(
    timestamps: torch.Tensor,
    msg: Optional[str] = None
) -> None:
    """Assert that timestamps are sorted in ascending order.
    
    Args:
        timestamps: Timestamp tensor
        msg: Optional custom error message
    """
    if len(timestamps) > 1:
        diffs = timestamps[1:] - timestamps[:-1]
        if (diffs < 0).any():
            error_msg = "Timestamps are not sorted in ascending order"
            if msg:
                error_msg = f"{msg}: {error_msg}"
            raise AssertionError(error_msg)


def assert_probabilities(
    probs: torch.Tensor,
    msg: Optional[str] = None
) -> None:
    """Assert that values are valid probabilities (between 0 and 1).
    
    Args:
        probs: Probability tensor
        msg: Optional custom error message
    """
    assert_tensor_range(probs, 0.0, 1.0, msg)


def assert_log_probabilities(
    log_probs: torch.Tensor,
    msg: Optional[str] = None
) -> None:
    """Assert that values are valid log probabilities (≤ 0).
    
    Args:
        log_probs: Log probability tensor
        msg: Optional custom error message
    """
    assert_tensor_range(log_probs, max_val=0.0, msg=msg)


def assert_model_output_shape(
    output: Union[torch.Tensor, dict],
    expected_batch_size: int,
    expected_output_dim: Optional[int] = None,
    msg: Optional[str] = None
) -> None:
    """Assert that model output has correct shape.
    
    Args:
        output: Model output (tensor or dict)
        expected_batch_size: Expected batch size
        expected_output_dim: Expected output dimension (for tensor outputs)
        msg: Optional custom error message
    """
    if isinstance(output, torch.Tensor):
        if output.size(0) != expected_batch_size:
            error_msg = f"Expected batch size {expected_batch_size}, got {output.size(0)}"
            if msg:
                error_msg = f"{msg}: {error_msg}"
            raise AssertionError(error_msg)
        
        if expected_output_dim is not None and output.size(-1) != expected_output_dim:
            error_msg = f"Expected output dim {expected_output_dim}, got {output.size(-1)}"
            if msg:
                error_msg = f"{msg}: {error_msg}"
            raise AssertionError(error_msg)
    
    elif isinstance(output, dict):
        # Check that all outputs have consistent batch size
        batch_sizes = set()
        for key, value in output.items():
            if isinstance(value, torch.Tensor):
                batch_sizes.add(value.size(0))
        
        if len(batch_sizes) > 1:
            error_msg = f"Inconsistent batch sizes in output dict: {batch_sizes}"
            if msg:
                error_msg = f"{msg}: {error_msg}"
            raise AssertionError(error_msg)
        
        if batch_sizes and list(batch_sizes)[0] != expected_batch_size:
            error_msg = f"Expected batch size {expected_batch_size}, got {list(batch_sizes)[0]}"
            if msg:
                error_msg = f"{msg}: {error_msg}"
            raise AssertionError(error_msg)


def assert_gradient_finite(
    model: torch.nn.Module,
    msg: Optional[str] = None
) -> None:
    """Assert that all model gradients are finite.
    
    Args:
        model: PyTorch model
        msg: Optional custom error message
    """
    for name, param in model.named_parameters():
        if param.grad is not None:
            if not torch.isfinite(param.grad).all():
                error_msg = f"Parameter {name} has non-finite gradients"
                if msg:
                    error_msg = f"{msg}: {error_msg}"
                raise AssertionError(error_msg)


def assert_no_gradient_explosion(
    model: torch.nn.Module,
    max_grad_norm: float = 10.0,
    msg: Optional[str] = None
) -> None:
    """Assert that gradients are not exploding.
    
    Args:
        model: PyTorch model
        max_grad_norm: Maximum allowed gradient norm
        msg: Optional custom error message
    """
    total_norm = 0.0
    for param in model.parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    
    total_norm = total_norm ** (1. / 2)
    
    if total_norm > max_grad_norm:
        error_msg = f"Gradient norm {total_norm} exceeds maximum {max_grad_norm}"
        if msg:
            error_msg = f"{msg}: {error_msg}"
        raise AssertionError(error_msg)


def assert_loss_decreasing(
    losses: list,
    min_decrease: float = 0.0,
    msg: Optional[str] = None
) -> None:
    """Assert that loss values are generally decreasing.
    
    Args:
        losses: List of loss values
        min_decrease: Minimum required decrease
        msg: Optional custom error message
    """
    if len(losses) < 2:
        return
    
    initial_loss = losses[0]
    final_loss = losses[-1]
    
    if final_loss > initial_loss - min_decrease:
        error_msg = f"Loss did not decrease sufficiently: {initial_loss} -> {final_loss} (min_decrease={min_decrease})"
        if msg:
            error_msg = f"{msg}: {error_msg}"
        raise AssertionError(error_msg)


def assert_reproducible(
    func,
    *args,
    seed: int = 42,
    msg: Optional[str] = None,
    **kwargs
) -> None:
    """Assert that a function produces reproducible results.
    
    Args:
        func: Function to test
        *args: Function arguments
        seed: Random seed
        msg: Optional custom error message
        **kwargs: Function keyword arguments
    """
    # First run
    torch.manual_seed(seed)
    np.random.seed(seed)
    result1 = func(*args, **kwargs)
    
    # Second run with same seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    result2 = func(*args, **kwargs)
    
    # Compare results
    if isinstance(result1, torch.Tensor) and isinstance(result2, torch.Tensor):
        if not torch.equal(result1, result2):
            error_msg = "Function is not reproducible with same seed"
            if msg:
                error_msg = f"{msg}: {error_msg}"
            raise AssertionError(error_msg)
    elif isinstance(result1, (list, tuple)) and isinstance(result2, (list, tuple)):
        if len(result1) != len(result2):
            error_msg = "Function returns different lengths on repeated calls"
            if msg:
                error_msg = f"{msg}: {error_msg}"
            raise AssertionError(error_msg)
        
        for r1, r2 in zip(result1, result2):
            if isinstance(r1, torch.Tensor) and isinstance(r2, torch.Tensor):
                if not torch.equal(r1, r2):
                    error_msg = "Function is not reproducible with same seed"
                    if msg:
                        error_msg = f"{msg}: {error_msg}"
                    raise AssertionError(error_msg)
    else:
        if result1 != result2:
            error_msg = "Function is not reproducible with same seed"
            if msg:
                error_msg = f"{msg}: {error_msg}"
            raise AssertionError(error_msg)