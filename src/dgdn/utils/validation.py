"""Input validation and data integrity checks for DGDN."""

import torch
import numpy as np
from typing import Optional, Dict, Any, List, Union, Tuple
import warnings
from pathlib import Path

from .config import ModelConfig, DGDNConfig
from .logging import get_logger


def validate_tensor_properties(
    tensor: torch.Tensor,
    name: str,
    expected_shape: Optional[Tuple[int, ...]] = None,
    expected_dtype: Optional[torch.dtype] = None,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
    allow_nan: bool = False,
    allow_inf: bool = False
) -> bool:
    """Validate tensor properties.
    
    Args:
        tensor: Tensor to validate
        name: Name of tensor for error messages
        expected_shape: Expected shape (None to skip check)
        expected_dtype: Expected dtype (None to skip check)
        min_value: Minimum allowed value (None to skip check)
        max_value: Maximum allowed value (None to skip check)
        allow_nan: Whether to allow NaN values
        allow_inf: Whether to allow infinite values
        
    Returns:
        True if validation passes
        
    Raises:
        ValueError: If validation fails
    """
    logger = get_logger("dgdn.validation")
    
    # Check if it's actually a tensor
    if not isinstance(tensor, torch.Tensor):
        raise ValueError(f"{name} must be a torch.Tensor, got {type(tensor)}")
    
    # Check shape
    if expected_shape is not None:
        if tensor.shape != expected_shape:
            # Allow flexible dimensions (marked with -1)
            shape_matches = True
            if len(expected_shape) == len(tensor.shape):
                for expected_dim, actual_dim in zip(expected_shape, tensor.shape):
                    if expected_dim != -1 and expected_dim != actual_dim:
                        shape_matches = False
                        break
            else:
                shape_matches = False
            
            if not shape_matches:
                raise ValueError(f"{name} shape mismatch: expected {expected_shape}, got {tensor.shape}")
    
    # Check dtype
    if expected_dtype is not None and tensor.dtype != expected_dtype:
        warnings.warn(f"{name} dtype mismatch: expected {expected_dtype}, got {tensor.dtype}")
    
    # Check for NaN values
    if not allow_nan and torch.isnan(tensor).any():
        raise ValueError(f"{name} contains NaN values")
    
    # Check for infinite values
    if not allow_inf and torch.isinf(tensor).any():
        raise ValueError(f"{name} contains infinite values")
    
    # Check value range
    if min_value is not None or max_value is not None:
        if min_value is not None and tensor.min() < min_value:
            raise ValueError(f"{name} contains values below minimum {min_value}: {tensor.min()}")
        if max_value is not None and tensor.max() > max_value:
            raise ValueError(f"{name} contains values above maximum {max_value}: {tensor.max()}")
    
    logger.debug(f"Tensor {name} validation passed: shape={tensor.shape}, dtype={tensor.dtype}")
    return True


def validate_temporal_data(data) -> bool:
    """Validate TemporalData object.
    
    Args:
        data: TemporalData object to validate
        
    Returns:
        True if validation passes
        
    Raises:
        ValueError: If validation fails
    """
    logger = get_logger("dgdn.validation")
    
    # Check required attributes
    required_attrs = ['edge_index', 'timestamps', 'num_nodes']
    for attr in required_attrs:
        if not hasattr(data, attr):
            raise ValueError(f"TemporalData missing required attribute: {attr}")
    
    # Validate edge_index
    edge_index = data.edge_index
    validate_tensor_properties(
        edge_index, 
        "edge_index",
        expected_shape=(2, -1),
        expected_dtype=torch.long,
        min_value=0
    )
    
    # Check edge indices are within node range
    max_node_idx = edge_index.max().item()
    if max_node_idx >= data.num_nodes:
        raise ValueError(f"Edge index {max_node_idx} exceeds num_nodes {data.num_nodes}")
    
    # Validate timestamps
    timestamps = data.timestamps
    validate_tensor_properties(
        timestamps,
        "timestamps", 
        expected_shape=(edge_index.shape[1],),
        min_value=0.0
    )
    
    # Check timestamps are sorted (recommended but not required)
    if not torch.all(timestamps[1:] >= timestamps[:-1]):
        warnings.warn("Timestamps are not sorted - this may affect model performance")
    
    # Validate optional attributes
    if hasattr(data, 'node_features') and data.node_features is not None:
        node_features = data.node_features
        validate_tensor_properties(
            node_features,
            "node_features",
            expected_shape=(data.num_nodes, -1)
        )
    
    if hasattr(data, 'edge_attr') and data.edge_attr is not None:
        edge_attr = data.edge_attr
        validate_tensor_properties(
            edge_attr,
            "edge_attr",
            expected_shape=(edge_index.shape[1], -1)
        )
    
    # Check device consistency
    device = edge_index.device
    if timestamps.device != device:
        raise ValueError(f"Device mismatch: edge_index on {device}, timestamps on {timestamps.device}")
    
    if hasattr(data, 'node_features') and data.node_features is not None:
        if data.node_features.device != device:
            raise ValueError(f"Device mismatch: edge_index on {device}, node_features on {data.node_features.device}")
    
    if hasattr(data, 'edge_attr') and data.edge_attr is not None:
        if data.edge_attr.device != device:
            raise ValueError(f"Device mismatch: edge_index on {device}, edge_attr on {data.edge_attr.device}")
    
    logger.debug(f"TemporalData validation passed: {data.num_nodes} nodes, {edge_index.shape[1]} edges")
    return True


def validate_model_config(config: ModelConfig) -> bool:
    """Validate model configuration.
    
    Args:
        config: Model configuration to validate
        
    Returns:
        True if validation passes
        
    Raises:
        ValueError: If validation fails
    """
    logger = get_logger("dgdn.validation")
    
    # Dimension validations
    if config.node_dim <= 0:
        raise ValueError(f"node_dim must be positive, got {config.node_dim}")
    if config.edge_dim < 0:
        raise ValueError(f"edge_dim must be non-negative, got {config.edge_dim}")
    if config.time_dim <= 0:
        raise ValueError(f"time_dim must be positive, got {config.time_dim}")
    if config.hidden_dim <= 0:
        raise ValueError(f"hidden_dim must be positive, got {config.hidden_dim}")
    
    # Architecture validations
    if config.num_layers <= 0:
        raise ValueError(f"num_layers must be positive, got {config.num_layers}")
    if config.num_heads <= 0:
        raise ValueError(f"num_heads must be positive, got {config.num_heads}")
    if config.hidden_dim % config.num_heads != 0:
        raise ValueError(f"hidden_dim ({config.hidden_dim}) must be divisible by num_heads ({config.num_heads})")
    
    # Diffusion validations
    if config.diffusion_steps <= 0:
        raise ValueError(f"diffusion_steps must be positive, got {config.diffusion_steps}")
    
    # Hyperparameter validations
    if not (0.0 <= config.dropout < 1.0):
        raise ValueError(f"dropout must be in [0.0, 1.0), got {config.dropout}")
    if config.max_time <= 0:
        raise ValueError(f"max_time must be positive, got {config.max_time}")
    
    # String parameter validations
    valid_aggregations = {"attention", "mean", "sum", "max"}
    if config.aggregation not in valid_aggregations:
        raise ValueError(f"aggregation must be one of {valid_aggregations}, got {config.aggregation}")
    
    valid_activations = {"relu", "gelu", "swish", "leaky_relu", "elu", "selu"}
    if config.activation not in valid_activations:
        raise ValueError(f"activation must be one of {valid_activations}, got {config.activation}")
    
    valid_time_encodings = {"fourier", "positional", "multiscale", "learned"}
    if config.time_encoding not in valid_time_encodings:
        raise ValueError(f"time_encoding must be one of {valid_time_encodings}, got {config.time_encoding}")
    
    logger.debug("Model configuration validation passed")
    return True


def validate_data(data, strict: bool = True) -> bool:
    """General data validation function.
    
    Args:
        data: Data to validate (can be various types)
        strict: Whether to use strict validation
        
    Returns:
        True if validation passes
    """
    logger = get_logger("dgdn.validation")
    
    if data is None:
        if strict:
            raise ValueError("Data cannot be None")
        else:
            logger.warning("Data is None")
            return False
    
    # Handle different data types
    if hasattr(data, 'edge_index'):  # Likely TemporalData
        return validate_temporal_data(data)
    elif isinstance(data, torch.Tensor):
        return validate_tensor_properties(data, "input_tensor")
    elif isinstance(data, (list, tuple)):
        # Validate each item in sequence
        for i, item in enumerate(data):
            validate_data(item, strict=strict)
        return True
    else:
        logger.warning(f"Unknown data type for validation: {type(data)}")
        return True


def validate_tensors(tensors: Dict[str, torch.Tensor], specs: Dict[str, Dict[str, Any]]) -> bool:
    """Validate multiple tensors according to specifications.
    
    Args:
        tensors: Dictionary of tensors to validate
        specs: Dictionary of validation specifications for each tensor
        
    Returns:
        True if all validations pass
    """
    for name, tensor in tensors.items():
        if name in specs:
            spec = specs[name]
            validate_tensor_properties(tensor, name, **spec)
    
    return True


def check_memory_requirements(
    num_nodes: int,
    num_edges: int,
    hidden_dim: int,
    batch_size: int = 1,
    max_memory_gb: float = 16.0
) -> bool:
    """Check if model can fit in available memory.
    
    Args:
        num_nodes: Number of nodes in graph
        num_edges: Number of edges in graph
        hidden_dim: Hidden dimension size
        batch_size: Batch size
        max_memory_gb: Maximum available memory in GB
        
    Returns:
        True if memory requirements are acceptable
        
    Raises:
        ValueError: If memory requirements exceed limit
    """
    logger = get_logger("dgdn.validation")
    
    # Estimate memory requirements (rough calculation)
    # Node embeddings: num_nodes * hidden_dim * 4 bytes (float32)
    # Edge features: num_edges * hidden_dim * 4 bytes
    # Attention matrices: num_nodes * num_nodes * 4 bytes (worst case)
    # Gradients: 2x the model parameters
    
    node_memory = num_nodes * hidden_dim * 4 * batch_size
    edge_memory = num_edges * hidden_dim * 4 * batch_size
    attention_memory = num_nodes * num_nodes * 4 * batch_size  # Conservative estimate
    
    total_memory_bytes = node_memory + edge_memory + attention_memory
    total_memory_gb = total_memory_bytes / (1024 ** 3)
    
    logger.info(f"Estimated memory requirements: {total_memory_gb:.2f} GB")
    
    if total_memory_gb > max_memory_gb:
        raise ValueError(
            f"Memory requirements ({total_memory_gb:.2f} GB) exceed limit ({max_memory_gb} GB). "
            f"Consider reducing hidden_dim, batch_size, or graph size."
        )
    
    return True


def validate_file_path(file_path: Union[str, Path], must_exist: bool = True) -> Path:
    """Validate file path.
    
    Args:
        file_path: Path to validate
        must_exist: Whether file must already exist
        
    Returns:
        Validated Path object
        
    Raises:
        ValueError: If validation fails
    """
    path = Path(file_path)
    
    if must_exist and not path.exists():
        raise ValueError(f"File does not exist: {path}")
    
    if must_exist and not path.is_file():
        raise ValueError(f"Path is not a file: {path}")
    
    # Check file extension for known types
    if path.suffix.lower() not in ['.pt', '.pth', '.pkl', '.json', '.yaml', '.yml', '.csv']:
        warnings.warn(f"Unknown file extension: {path.suffix}")
    
    return path


class DataValidator:
    """Stateful data validator that can accumulate validation statistics."""
    
    def __init__(self, strict: bool = True):
        self.strict = strict
        self.validation_stats = {
            'total_validations': 0,
            'failed_validations': 0,
            'warnings_issued': 0
        }
        self.logger = get_logger("dgdn.validation")
    
    def validate(self, data, validation_name: str = "data") -> bool:
        """Validate data and update statistics."""
        self.validation_stats['total_validations'] += 1
        
        try:
            result = validate_data(data, strict=self.strict)
            if not result:
                self.validation_stats['failed_validations'] += 1
            return result
        except Exception as e:
            self.validation_stats['failed_validations'] += 1
            if self.strict:
                raise
            else:
                self.logger.warning(f"Validation failed for {validation_name}: {e}")
                return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get validation statistics."""
        return self.validation_stats.copy()
    
    def reset_stats(self) -> None:
        """Reset validation statistics."""
        self.validation_stats = {
            'total_validations': 0,
            'failed_validations': 0,
            'warnings_issued': 0
        }