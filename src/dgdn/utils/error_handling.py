"""
Robust error handling and validation utilities for DGDN.
"""

import torch
import traceback
import logging
from typing import Any, Dict, List, Optional, Union, Callable
from functools import wraps
import time
import warnings

class DGDNError(Exception):
    """Base exception class for DGDN-related errors."""
    pass

class ValidationError(DGDNError):
    """Raised when input validation fails."""
    pass

class ModelError(DGDNError):
    """Raised when model operations fail."""
    pass

class DataError(DGDNError):
    """Raised when data processing fails."""
    pass

class MemoryError(DGDNError):
    """Raised when memory constraints are exceeded."""
    pass

def validate_tensor_properties(tensor: torch.Tensor, 
                             name: str, 
                             expected_shape: Optional[tuple] = None,
                             expected_dtype: Optional[torch.dtype] = None,
                             min_value: Optional[float] = None,
                             max_value: Optional[float] = None,
                             allow_nan: bool = False,
                             allow_inf: bool = False) -> None:
    """
    Comprehensive tensor validation with detailed error messages.
    
    Args:
        tensor: Tensor to validate
        name: Name of the tensor for error messages
        expected_shape: Expected tensor shape (optional)
        expected_dtype: Expected tensor dtype (optional)
        min_value: Minimum allowed value (optional)
        max_value: Maximum allowed value (optional)
        allow_nan: Whether to allow NaN values
        allow_inf: Whether to allow infinite values
        
    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(tensor, torch.Tensor):
        raise ValidationError(f"{name} must be a torch.Tensor, got {type(tensor)}")
    
    # Shape validation
    if expected_shape is not None:
        if tensor.shape != expected_shape:
            raise ValidationError(
                f"{name} shape mismatch: expected {expected_shape}, got {tensor.shape}"
            )
    
    # Dtype validation
    if expected_dtype is not None:
        if tensor.dtype != expected_dtype:
            raise ValidationError(
                f"{name} dtype mismatch: expected {expected_dtype}, got {tensor.dtype}"
            )
    
    # NaN/Inf validation
    if not allow_nan and torch.isnan(tensor).any():
        raise ValidationError(f"{name} contains NaN values")
    
    if not allow_inf and torch.isinf(tensor).any():
        raise ValidationError(f"{name} contains infinite values")
    
    # Value range validation
    if min_value is not None and tensor.min().item() < min_value:
        raise ValidationError(
            f"{name} contains values below minimum {min_value}: min={tensor.min().item()}"
        )
    
    if max_value is not None and tensor.max().item() > max_value:
        raise ValidationError(
            f"{name} contains values above maximum {max_value}: max={tensor.max().item()}"
        )

def validate_graph_data(edge_index: torch.Tensor, 
                       timestamps: torch.Tensor,
                       num_nodes: int,
                       node_features: Optional[torch.Tensor] = None,
                       edge_attr: Optional[torch.Tensor] = None) -> None:
    """
    Validate temporal graph data structure.
    
    Args:
        edge_index: Edge connectivity tensor [2, num_edges]
        timestamps: Edge timestamps [num_edges]
        num_nodes: Number of nodes in the graph
        node_features: Node features [num_nodes, node_dim] (optional)
        edge_attr: Edge attributes [num_edges, edge_dim] (optional)
        
    Raises:
        ValidationError: If validation fails
    """
    # Validate edge_index
    validate_tensor_properties(
        edge_index, "edge_index", 
        expected_dtype=torch.long,
        min_value=0
    )
    
    if edge_index.dim() != 2 or edge_index.size(0) != 2:
        raise ValidationError(f"edge_index must have shape [2, num_edges], got {edge_index.shape}")
    
    num_edges = edge_index.size(1)
    max_node_idx = edge_index.max().item()
    
    if max_node_idx >= num_nodes:
        raise ValidationError(
            f"edge_index contains node indices >= num_nodes: max_idx={max_node_idx}, num_nodes={num_nodes}"
        )
    
    # Validate timestamps
    validate_tensor_properties(
        timestamps, "timestamps",
        expected_shape=(num_edges,),
        expected_dtype=torch.float32,
        min_value=0.0,
        allow_nan=False,
        allow_inf=False
    )
    
    # Validate optional node features
    if node_features is not None:
        validate_tensor_properties(
            node_features, "node_features",
            allow_nan=False,
            allow_inf=False
        )
        
        if node_features.size(0) != num_nodes:
            raise ValidationError(
                f"node_features size mismatch: expected {num_nodes} nodes, got {node_features.size(0)}"
            )
    
    # Validate optional edge attributes
    if edge_attr is not None:
        validate_tensor_properties(
            edge_attr, "edge_attr",
            allow_nan=False,
            allow_inf=False
        )
        
        if edge_attr.size(0) != num_edges:
            raise ValidationError(
                f"edge_attr size mismatch: expected {num_edges} edges, got {edge_attr.size(0)}"
            )

def robust_forward_pass(func: Callable) -> Callable:
    """
    Decorator for robust forward pass execution with error handling.
    
    Args:
        func: Forward pass function to wrap
        
    Returns:
        Wrapped function with error handling
    """
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        try:
            # Pre-execution validation
            if hasattr(self, '_validate_forward_input'):
                if args:
                    self._validate_forward_input(args[0])
            
            # Memory check
            if torch.cuda.is_available():
                memory_before = torch.cuda.memory_allocated()
                max_memory = torch.cuda.max_memory_allocated()
                
                # Warn if memory usage is high
                if memory_before > 0.8 * max_memory:
                    warnings.warn(
                        f"High GPU memory usage detected: {memory_before/1e9:.2f}GB used",
                        RuntimeWarning
                    )
            
            # Execute forward pass
            start_time = time.time()
            result = func(self, *args, **kwargs)
            execution_time = time.time() - start_time
            
            # Log performance metrics
            logging.debug(f"Forward pass completed in {execution_time:.4f}s")
            
            # Post-execution validation
            if isinstance(result, dict):
                for key, value in result.items():
                    if isinstance(value, torch.Tensor):
                        validate_tensor_properties(
                            value, f"output[{key}]",
                            allow_nan=False,
                            allow_inf=False
                        )
            
            return result
            
        except ValidationError as e:
            logging.error(f"Validation error in forward pass: {e}")
            raise
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logging.error(f"GPU out of memory error: {e}")
                raise MemoryError(f"GPU out of memory during forward pass: {e}")
            else:
                logging.error(f"Runtime error in forward pass: {e}")
                raise ModelError(f"Forward pass failed: {e}")
        except Exception as e:
            logging.error(f"Unexpected error in forward pass: {e}")
            logging.error(traceback.format_exc())
            raise ModelError(f"Unexpected error in forward pass: {e}")
    
    return wrapper

def check_memory_usage(device: torch.device) -> Dict[str, float]:
    """
    Check current memory usage on the specified device.
    
    Args:
        device: Device to check memory for
        
    Returns:
        Dictionary with memory statistics
    """
    stats = {}
    
    if device.type == 'cuda':
        stats['allocated'] = torch.cuda.memory_allocated(device) / 1e9  # GB
        stats['reserved'] = torch.cuda.memory_reserved(device) / 1e9   # GB
        stats['max_allocated'] = torch.cuda.max_memory_allocated(device) / 1e9  # GB
        
        # Get total GPU memory
        props = torch.cuda.get_device_properties(device)
        stats['total'] = props.total_memory / 1e9  # GB
        stats['utilization'] = stats['allocated'] / stats['total'] * 100  # %
        
    else:
        # For CPU, we can use psutil if available
        try:
            import psutil
            memory = psutil.virtual_memory()
            stats['allocated'] = memory.used / 1e9  # GB
            stats['total'] = memory.total / 1e9  # GB
            stats['utilization'] = memory.percent  # %
        except ImportError:
            stats['allocated'] = 0.0
            stats['total'] = 0.0
            stats['utilization'] = 0.0
    
    return stats

def safe_tensor_operation(operation: Callable, 
                         tensor: torch.Tensor, 
                         *args, 
                         **kwargs) -> torch.Tensor:
    """
    Safely execute tensor operations with error handling.
    
    Args:
        operation: Tensor operation to execute
        tensor: Input tensor
        *args: Additional arguments
        **kwargs: Additional keyword arguments
        
    Returns:
        Result of the operation
        
    Raises:
        ModelError: If operation fails
    """
    try:
        # Validate input tensor
        validate_tensor_properties(
            tensor, "input_tensor",
            allow_nan=False,
            allow_inf=False
        )
        
        # Execute operation
        result = operation(tensor, *args, **kwargs)
        
        # Validate output
        if isinstance(result, torch.Tensor):
            validate_tensor_properties(
                result, "operation_result",
                allow_nan=False,
                allow_inf=False
            )
        
        return result
        
    except Exception as e:
        logging.error(f"Tensor operation failed: {operation.__name__} - {e}")
        raise ModelError(f"Tensor operation failed: {e}")

class ErrorRecovery:
    """Error recovery strategies for robust training and inference."""
    
    @staticmethod
    def gradient_clipping_recovery(model: torch.nn.Module, 
                                 max_norm: float = 1.0) -> bool:
        """
        Apply gradient clipping to prevent gradient explosion.
        
        Args:
            model: Model to apply gradient clipping to
            max_norm: Maximum norm for gradient clipping
            
        Returns:
            True if gradients were clipped, False otherwise
        """
        try:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            if grad_norm > max_norm:
                logging.warning(f"Gradients clipped: norm={grad_norm:.4f} > max_norm={max_norm}")
                return True
            return False
        except Exception as e:
            logging.error(f"Gradient clipping failed: {e}")
            return False
    
    @staticmethod
    def nan_recovery(tensor: torch.Tensor, 
                    fill_value: float = 0.0) -> torch.Tensor:
        """
        Replace NaN values in tensor with fill_value.
        
        Args:
            tensor: Input tensor
            fill_value: Value to replace NaN with
            
        Returns:
            Tensor with NaN values replaced
        """
        if torch.isnan(tensor).any():
            logging.warning(f"NaN values detected and replaced with {fill_value}")
            return torch.where(torch.isnan(tensor), 
                             torch.tensor(fill_value, dtype=tensor.dtype, device=tensor.device),
                             tensor)
        return tensor
    
    @staticmethod
    def inf_recovery(tensor: torch.Tensor, 
                    max_value: float = 1e6) -> torch.Tensor:
        """
        Clamp infinite values in tensor.
        
        Args:
            tensor: Input tensor
            max_value: Maximum allowed absolute value
            
        Returns:
            Tensor with infinite values clamped
        """
        if torch.isinf(tensor).any():
            logging.warning(f"Infinite values detected and clamped to ±{max_value}")
            return torch.clamp(tensor, -max_value, max_value)
        return tensor

def setup_error_logging(log_level: str = "INFO", 
                       log_file: Optional[str] = None) -> None:
    """
    Setup comprehensive error logging for DGDN.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional log file path
    """
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=[
            logging.StreamHandler(),
            *([] if log_file is None else [logging.FileHandler(log_file)])
        ]
    )
    
    # Set up specific logger for DGDN
    logger = logging.getLogger('dgdn')
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Log system information
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA devices: {torch.cuda.device_count()}")
        logger.info(f"Current device: {torch.cuda.current_device()}")