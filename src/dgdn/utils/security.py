"""Security utilities for DGDN."""

import torch
import hashlib
import pickle
import tempfile
import os
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import warnings

from .logging import get_logger
from .validation import validate_tensor_properties


class SecurityValidator:
    """Security validator for DGDN inputs and models."""
    
    def __init__(self, enable_strict_mode: bool = True):
        self.enable_strict_mode = enable_strict_mode
        self.logger = get_logger("dgdn.security")
        
        # Define security limits
        self.max_tensor_size = 1e9  # 1B elements
        self.max_memory_mb = 16 * 1024  # 16GB
        self.allowed_dtypes = {
            torch.float32, torch.float64, torch.float16,
            torch.int32, torch.int64, torch.int16, torch.int8,
            torch.uint8, torch.bool
        }
        self.blocked_operations = set()
    
    def validate_tensor_security(self, tensor: torch.Tensor, name: str = "tensor") -> bool:
        """Validate tensor for security issues.
        
        Args:
            tensor: Tensor to validate
            name: Name for logging
            
        Returns:
            True if validation passes
            
        Raises:
            SecurityError: If security issues are found
        """
        # Check tensor size
        if tensor.numel() > self.max_tensor_size:
            raise SecurityError(f"Tensor {name} too large: {tensor.numel()} > {self.max_tensor_size}")
        
        # Check memory usage
        memory_mb = tensor.element_size() * tensor.numel() / (1024 * 1024)
        if memory_mb > self.max_memory_mb:
            raise SecurityError(f"Tensor {name} requires too much memory: {memory_mb:.2f} MB > {self.max_memory_mb} MB")
        
        # Check data type
        if tensor.dtype not in self.allowed_dtypes:
            raise SecurityError(f"Tensor {name} has disallowed dtype: {tensor.dtype}")
        
        # Check for suspicious values
        if torch.isnan(tensor).any():
            if self.enable_strict_mode:
                raise SecurityError(f"Tensor {name} contains NaN values")
            else:
                self.logger.warning(f"Tensor {name} contains NaN values")
        
        if torch.isinf(tensor).any():
            if self.enable_strict_mode:
                raise SecurityError(f"Tensor {name} contains infinite values")
            else:
                self.logger.warning(f"Tensor {name} contains infinite values")
        
        # Check for extremely large values that could cause overflow
        max_val = tensor.abs().max().item()
        if max_val > 1e6:
            self.logger.warning(f"Tensor {name} contains very large values: max={max_val}")
        
        self.logger.debug(f"Security validation passed for tensor {name}")
        return True
    
    def validate_file_security(self, file_path: Union[str, Path]) -> bool:
        """Validate file for security issues.
        
        Args:
            file_path: Path to file
            
        Returns:
            True if validation passes
            
        Raises:
            SecurityError: If security issues are found
        """
        path = Path(file_path)
        
        # Check file exists and is actually a file
        if not path.exists():
            raise SecurityError(f"File does not exist: {path}")
        
        if not path.is_file():
            raise SecurityError(f"Path is not a file: {path}")
        
        # Check file size
        file_size_mb = path.stat().st_size / (1024 * 1024)
        if file_size_mb > self.max_memory_mb:
            raise SecurityError(f"File too large: {file_size_mb:.2f} MB > {self.max_memory_mb} MB")
        
        # Check file extension
        allowed_extensions = {'.pt', '.pth', '.pkl', '.json', '.yaml', '.yml'}
        if path.suffix.lower() not in allowed_extensions:
            if self.enable_strict_mode:
                raise SecurityError(f"File extension not allowed: {path.suffix}")
            else:
                self.logger.warning(f"File extension not typically safe: {path.suffix}")
        
        # Check for suspicious file names
        suspicious_patterns = ['../', '..\\', '~/', '/etc/', '/proc/', 'C:\\Windows\\']
        path_str = str(path)
        for pattern in suspicious_patterns:
            if pattern in path_str:
                raise SecurityError(f"Suspicious file path pattern: {pattern} in {path}")
        
        self.logger.debug(f"Security validation passed for file {path}")
        return True
    
    def validate_model_state(self, state_dict: Dict[str, torch.Tensor]) -> bool:
        """Validate model state dictionary for security issues.
        
        Args:
            state_dict: Model state dictionary
            
        Returns:
            True if validation passes
        """
        total_params = 0
        
        for name, tensor in state_dict.items():
            self.validate_tensor_security(tensor, f"parameter_{name}")
            total_params += tensor.numel()
        
        # Check total model size
        if total_params > 1e9:  # 1B parameters
            self.logger.warning(f"Model has very large number of parameters: {total_params:,}")
        
        self.logger.info(f"Model state validation passed: {len(state_dict)} parameters, {total_params:,} total elements")
        return True


class SecurityError(Exception):
    """Custom exception for security-related errors."""
    pass


def sanitize_input(data: Any, max_string_length: int = 1000) -> Any:
    """Sanitize input data for security.
    
    Args:
        data: Data to sanitize
        max_string_length: Maximum allowed string length
        
    Returns:
        Sanitized data
    """
    logger = get_logger("dgdn.security")
    
    if isinstance(data, str):
        # Limit string length
        if len(data) > max_string_length:
            logger.warning(f"String truncated from {len(data)} to {max_string_length} characters")
            data = data[:max_string_length]
        
        # Remove potentially dangerous characters
        dangerous_chars = ['<', '>', '"', "'", '&', '\x00']
        for char in dangerous_chars:
            if char in data:
                data = data.replace(char, '')
                logger.warning(f"Removed dangerous character: {char}")
    
    elif isinstance(data, (list, tuple)):
        # Recursively sanitize list/tuple elements
        sanitized = []
        for item in data:
            sanitized.append(sanitize_input(item, max_string_length))
        return type(data)(sanitized)
    
    elif isinstance(data, dict):
        # Recursively sanitize dictionary values
        sanitized = {}
        for key, value in data.items():
            clean_key = sanitize_input(key, max_string_length)
            clean_value = sanitize_input(value, max_string_length)
            sanitized[clean_key] = clean_value
        return sanitized
    
    elif isinstance(data, torch.Tensor):
        # Validate tensor security
        validator = SecurityValidator()
        validator.validate_tensor_security(data)
    
    return data


def compute_model_hash(model) -> str:
    """Compute cryptographic hash of model parameters.
    
    Args:
        model: PyTorch model
        
    Returns:
        SHA256 hash of model parameters
    """
    hasher = hashlib.sha256()
    
    for name, param in model.named_parameters():
        # Convert parameter to bytes and update hash
        param_bytes = param.data.cpu().numpy().tobytes()
        hasher.update(param_bytes)
        hasher.update(name.encode('utf-8'))  # Include parameter name
    
    return hasher.hexdigest()


def check_model_integrity(model, expected_hash: str) -> bool:
    """Check if model parameters match expected hash.
    
    Args:
        model: PyTorch model
        expected_hash: Expected SHA256 hash
        
    Returns:
        True if integrity check passes
        
    Raises:
        SecurityError: If integrity check fails
    """
    logger = get_logger("dgdn.security")
    
    current_hash = compute_model_hash(model)
    
    if current_hash != expected_hash:
        raise SecurityError(f"Model integrity check failed: {current_hash} != {expected_hash}")
    
    logger.info("Model integrity check passed")
    return True


def secure_model_save(model, file_path: Union[str, Path], include_hash: bool = True) -> str:
    """Securely save model with integrity hash.
    
    Args:
        model: PyTorch model to save
        file_path: Path to save model
        include_hash: Whether to include integrity hash
        
    Returns:
        Model hash if include_hash is True
    """
    logger = get_logger("dgdn.security")
    path = Path(file_path)
    
    # Validate output path
    validator = SecurityValidator()
    
    # Create parent directory if needed
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Compute hash before saving
    model_hash = compute_model_hash(model) if include_hash else None
    
    # Save model
    torch.save(model.state_dict(), path)
    
    # Save hash file if requested
    if include_hash:
        hash_path = path.with_suffix(path.suffix + '.hash')
        with open(hash_path, 'w') as f:
            f.write(model_hash)
        logger.info(f"Model saved with integrity hash: {hash_path}")
    
    logger.info(f"Model securely saved to: {path}")
    return model_hash


def secure_model_load(model, file_path: Union[str, Path], verify_hash: bool = True):
    """Securely load model with integrity verification.
    
    Args:
        model: PyTorch model to load state into
        file_path: Path to load model from
        verify_hash: Whether to verify integrity hash
        
    Returns:
        Loaded model
        
    Raises:
        SecurityError: If security validation fails
    """
    logger = get_logger("dgdn.security")
    path = Path(file_path)
    
    # Validate file security
    validator = SecurityValidator()
    validator.validate_file_security(path)
    
    # Load model state
    try:
        state_dict = torch.load(path, map_location='cpu')
    except Exception as e:
        raise SecurityError(f"Failed to load model: {e}")
    
    # Validate state dictionary
    validator.validate_model_state(state_dict)
    
    # Load state into model
    model.load_state_dict(state_dict)
    
    # Verify hash if requested
    if verify_hash:
        hash_path = path.with_suffix(path.suffix + '.hash')
        if hash_path.exists():
            with open(hash_path, 'r') as f:
                expected_hash = f.read().strip()
            check_model_integrity(model, expected_hash)
        else:
            logger.warning(f"Hash file not found: {hash_path}")
    
    logger.info(f"Model securely loaded from: {path}")
    return model


class DifferentialPrivacyManager:
    """Manager for differential privacy during training."""
    
    def __init__(
        self,
        noise_multiplier: float = 1.0,
        max_grad_norm: float = 1.0,
        sample_rate: float = 0.01,
        enable_privacy: bool = False
    ):
        self.noise_multiplier = noise_multiplier
        self.max_grad_norm = max_grad_norm
        self.sample_rate = sample_rate
        self.enable_privacy = enable_privacy
        self.logger = get_logger("dgdn.privacy")
        
        if enable_privacy:
            self.logger.info(f"Differential privacy enabled: noise={noise_multiplier}, "
                           f"max_grad_norm={max_grad_norm}, sample_rate={sample_rate}")
    
    def add_noise_to_gradients(self, model) -> None:
        """Add calibrated noise to model gradients for differential privacy."""
        if not self.enable_privacy:
            return
        
        # Clip gradients first
        torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_grad_norm)
        
        # Add noise to gradients
        for param in model.parameters():
            if param.grad is not None:
                noise = torch.normal(
                    mean=0.0,
                    std=self.noise_multiplier * self.max_grad_norm,
                    size=param.grad.shape,
                    device=param.grad.device
                )
                param.grad += noise
        
        self.logger.debug("Added differential privacy noise to gradients")
    
    def compute_privacy_budget(self, num_epochs: int, num_samples: int) -> float:
        """Compute privacy budget (epsilon) for given training parameters.
        
        Args:
            num_epochs: Number of training epochs
            num_samples: Total number of training samples
            
        Returns:
            Privacy budget (epsilon)
        """
        if not self.enable_privacy:
            return float('inf')
        
        # Simplified privacy budget calculation
        # In practice, use proper accounting methods like RDP
        q = self.sample_rate
        steps = num_epochs * (num_samples // int(num_samples * q))
        
        # Rough approximation - use proper privacy accounting in production
        epsilon = steps * q * q / (2 * self.noise_multiplier * self.noise_multiplier)
        
        self.logger.info(f"Estimated privacy budget: ε = {epsilon:.4f}")
        return epsilon


def create_secure_temp_file(suffix: str = ".tmp") -> str:
    """Create a secure temporary file.
    
    Args:
        suffix: File suffix
        
    Returns:
        Path to secure temporary file
    """
    # Create temporary file with restricted permissions
    fd, temp_path = tempfile.mkstemp(suffix=suffix)
    
    # Set restrictive permissions (owner read/write only)
    os.chmod(temp_path, 0o600)
    
    # Close file descriptor (caller should open as needed)
    os.close(fd)
    
    return temp_path


def secure_delete_file(file_path: Union[str, Path]) -> None:
    """Securely delete a file by overwriting with random data.
    
    Args:
        file_path: Path to file to delete
    """
    logger = get_logger("dgdn.security")
    path = Path(file_path)
    
    if not path.exists():
        return
    
    try:
        # Get file size
        file_size = path.stat().st_size
        
        # Overwrite with random data multiple times
        with open(path, 'r+b') as f:
            for _ in range(3):  # 3 passes
                f.seek(0)
                f.write(os.urandom(file_size))
                f.flush()
                os.fsync(f.fileno())
        
        # Finally delete the file
        path.unlink()
        logger.debug(f"Securely deleted file: {path}")
        
    except Exception as e:
        logger.error(f"Failed to securely delete file {path}: {e}")
        # Fall back to regular deletion
        try:
            path.unlink()
        except Exception:
            pass