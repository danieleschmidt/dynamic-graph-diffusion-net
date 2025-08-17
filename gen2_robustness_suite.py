#!/usr/bin/env python3
"""
Generation 2: Robustness Enhancement Suite
Comprehensive error handling, validation, logging, monitoring, and security.
"""

import torch
import torch.nn.functional as F
import numpy as np
import logging
import time
import psutil
import json
import hashlib
import warnings
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass, asdict
from contextlib import contextmanager
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('dgdn_robust.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class ValidationResult:
    """Structured validation result."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    metrics: Dict[str, float]
    timestamp: float

@dataclass
class PerformanceMetrics:
    """Performance monitoring metrics."""
    memory_usage_mb: float
    cpu_usage_percent: float
    inference_time_ms: float
    throughput_samples_per_sec: float
    gpu_memory_mb: Optional[float] = None

class InputValidator:
    """Comprehensive input validation for DGDN models."""
    
    @staticmethod
    def validate_temporal_data(data) -> ValidationResult:
        """Validate temporal graph data structure."""
        errors = []
        warnings = []
        metrics = {}
        
        # Basic attribute checks
        required_attrs = ['edge_index', 'timestamps', 'num_nodes']
        for attr in required_attrs:
            if not hasattr(data, attr):
                errors.append(f"Missing required attribute: {attr}")
        
        if errors:
            return ValidationResult(False, errors, warnings, metrics, time.time())
        
        # Edge index validation
        edge_index = data.edge_index
        timestamps = data.timestamps
        num_nodes = data.num_nodes
        
        if not isinstance(edge_index, torch.Tensor):
            errors.append("edge_index must be a torch.Tensor")
        elif edge_index.dim() != 2 or edge_index.size(0) != 2:
            errors.append(f"edge_index must have shape [2, num_edges], got {edge_index.shape}")
        elif edge_index.dtype not in [torch.long, torch.int64, torch.int32]:
            errors.append(f"edge_index must be integer type, got {edge_index.dtype}")
        
        # Timestamps validation
        if not isinstance(timestamps, torch.Tensor):
            errors.append("timestamps must be a torch.Tensor")
        elif timestamps.dim() != 1:
            errors.append(f"timestamps must be 1-dimensional, got {timestamps.shape}")
        elif edge_index.size(1) != timestamps.size(0):
            errors.append(f"Number of edges ({edge_index.size(1)}) must match timestamps ({timestamps.size(0)})")
        
        # Node index bounds checking
        if not errors and edge_index.numel() > 0:
            max_node = edge_index.max().item()
            min_node = edge_index.min().item()
            
            if min_node < 0:
                errors.append(f"Node indices must be non-negative, found {min_node}")
            if max_node >= num_nodes:
                errors.append(f"Maximum node index ({max_node}) exceeds num_nodes ({num_nodes})")
        
        # Temporal ordering checks
        if not errors and timestamps.numel() > 1:
            if not torch.all(timestamps[1:] >= timestamps[:-1]):
                warnings.append("Timestamps are not sorted in ascending order")
            
            # Check for negative timestamps
            if torch.any(timestamps < 0):
                warnings.append("Found negative timestamps")
            
            # Check for timestamp gaps
            time_diffs = timestamps[1:] - timestamps[:-1]
            max_gap = time_diffs.max().item()
            metrics['max_time_gap'] = max_gap
            metrics['mean_time_gap'] = time_diffs.mean().item()
            
            if max_gap > 1000:  # Arbitrary threshold
                warnings.append(f"Large timestamp gap detected: {max_gap}")
        
        # Optional attributes validation
        if hasattr(data, 'node_features') and data.node_features is not None:
            node_features = data.node_features
            if not isinstance(node_features, torch.Tensor):
                errors.append("node_features must be a torch.Tensor")
            elif node_features.size(0) != num_nodes:
                errors.append(f"node_features size ({node_features.size(0)}) must match num_nodes ({num_nodes})")
            elif node_features.dim() != 2:
                errors.append(f"node_features must be 2-dimensional, got {node_features.shape}")
        
        if hasattr(data, 'edge_attr') and data.edge_attr is not None:
            edge_attr = data.edge_attr
            if not isinstance(edge_attr, torch.Tensor):
                errors.append("edge_attr must be a torch.Tensor")
            elif edge_attr.size(0) != edge_index.size(1):
                errors.append(f"edge_attr size ({edge_attr.size(0)}) must match number of edges ({edge_index.size(1)})")
        
        # Graph connectivity metrics
        metrics['num_nodes'] = num_nodes
        metrics['num_edges'] = edge_index.size(1)
        metrics['density'] = (2 * edge_index.size(1)) / (num_nodes * (num_nodes - 1)) if num_nodes > 1 else 0
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            metrics=metrics,
            timestamp=time.time()
        )
    
    @staticmethod
    def validate_model_config(config: Dict[str, Any]) -> ValidationResult:
        """Validate model configuration parameters."""
        errors = []
        warnings = []
        metrics = {}
        
        # Required parameters
        required_params = ['node_dim', 'hidden_dim', 'time_dim', 'num_layers']
        for param in required_params:
            if param not in config:
                errors.append(f"Missing required parameter: {param}")
            elif not isinstance(config[param], int) or config[param] <= 0:
                errors.append(f"{param} must be a positive integer, got {config[param]}")
        
        # Optional parameters with validation
        optional_params = {
            'num_heads': (int, lambda x: x > 0 and x <= 32),
            'diffusion_steps': (int, lambda x: 1 <= x <= 20),
            'dropout': (float, lambda x: 0.0 <= x < 1.0),
            'activation': (str, lambda x: x in ['relu', 'gelu', 'swish', 'leaky_relu']),
            'aggregation': (str, lambda x: x in ['attention', 'mean', 'sum'])
        }
        
        for param, (param_type, validator) in optional_params.items():
            if param in config:
                value = config[param]
                if not isinstance(value, param_type):
                    errors.append(f"{param} must be {param_type.__name__}, got {type(value).__name__}")
                elif not validator(value):
                    errors.append(f"Invalid value for {param}: {value}")
        
        # Check dimensional consistency
        if 'hidden_dim' in config and 'num_heads' in config:
            if config['hidden_dim'] % config['num_heads'] != 0:
                errors.append(f"hidden_dim ({config['hidden_dim']}) must be divisible by num_heads ({config['num_heads']})")
        
        # Performance warnings
        if 'hidden_dim' in config and config['hidden_dim'] > 1024:
            warnings.append(f"Large hidden_dim ({config['hidden_dim']}) may impact performance")
        
        if 'num_layers' in config and config['num_layers'] > 10:
            warnings.append(f"Many layers ({config['num_layers']}) may cause gradient issues")
        
        metrics['total_params_estimate'] = InputValidator._estimate_parameters(config)
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            metrics=metrics,
            timestamp=time.time()
        )
    
    @staticmethod
    def _estimate_parameters(config: Dict[str, Any]) -> int:
        """Estimate number of model parameters."""
        node_dim = config.get('node_dim', 64)
        hidden_dim = config.get('hidden_dim', 256)
        time_dim = config.get('time_dim', 32)
        num_layers = config.get('num_layers', 3)
        num_heads = config.get('num_heads', 8)
        
        # Rough parameter estimation
        input_proj = node_dim * hidden_dim
        time_encoder = time_dim * 64 + 64 * time_dim  # EdgeTimeEncoder
        layer_params = num_layers * (
            hidden_dim * hidden_dim * 4 +  # Attention projections
            hidden_dim * 2 * hidden_dim * 4  # Diffusion network
        )
        diffusion = hidden_dim * hidden_dim * 8  # VariationalDiffusion
        output_proj = hidden_dim * hidden_dim
        
        return input_proj + time_encoder + layer_params + diffusion + output_proj

class PerformanceMonitor:
    """Real-time performance monitoring for DGDN models."""
    
    def __init__(self):
        self.metrics_history = []
        self.start_time = None
        
    @contextmanager
    def monitor_inference(self):
        """Context manager for monitoring inference performance."""
        # Get initial metrics
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        initial_cpu = process.cpu_percent()
        
        # GPU memory if available
        gpu_memory = None
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024  # MB
        
        start_time = time.time()
        
        try:
            yield
        finally:
            end_time = time.time()
            
            # Calculate metrics
            final_memory = process.memory_info().rss / 1024 / 1024
            final_cpu = process.cpu_percent()
            inference_time = (end_time - start_time) * 1000  # ms
            
            final_gpu_memory = None
            if gpu_memory is not None:
                final_gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024
            
            metrics = PerformanceMetrics(
                memory_usage_mb=final_memory - initial_memory,
                cpu_usage_percent=(initial_cpu + final_cpu) / 2,
                inference_time_ms=inference_time,
                throughput_samples_per_sec=1000 / inference_time if inference_time > 0 else 0,
                gpu_memory_mb=final_gpu_memory - gpu_memory if final_gpu_memory else None
            )
            
            self.metrics_history.append(metrics)
            logger.info(f"Performance metrics: {asdict(metrics)}")
    
    def get_average_metrics(self, last_n: int = 10) -> PerformanceMetrics:
        """Get average performance metrics over last N inferences."""
        if not self.metrics_history:
            return PerformanceMetrics(0, 0, 0, 0)
        
        recent_metrics = self.metrics_history[-last_n:]
        
        return PerformanceMetrics(
            memory_usage_mb=np.mean([m.memory_usage_mb for m in recent_metrics]),
            cpu_usage_percent=np.mean([m.cpu_usage_percent for m in recent_metrics]),
            inference_time_ms=np.mean([m.inference_time_ms for m in recent_metrics]),
            throughput_samples_per_sec=np.mean([m.throughput_samples_per_sec for m in recent_metrics]),
            gpu_memory_mb=np.mean([m.gpu_memory_mb for m in recent_metrics if m.gpu_memory_mb is not None]) if any(m.gpu_memory_mb for m in recent_metrics) else None
        )

class SecurityManager:
    """Security measures for DGDN models."""
    
    @staticmethod
    def sanitize_input_data(data) -> bool:
        """Sanitize input data to prevent attacks."""
        # Check for NaN or infinite values
        if hasattr(data, 'edge_index') and torch.any(torch.isnan(data.edge_index.float())):
            logger.warning("Found NaN values in edge_index")
            return False
        
        if hasattr(data, 'timestamps') and torch.any(torch.isnan(data.timestamps)):
            logger.warning("Found NaN values in timestamps")
            return False
        
        if hasattr(data, 'timestamps') and torch.any(torch.isinf(data.timestamps)):
            logger.warning("Found infinite values in timestamps")
            return False
        
        # Check for suspiciously large values (potential memory attack)
        if hasattr(data, 'num_nodes') and data.num_nodes > 1000000:
            logger.warning(f"Suspiciously large number of nodes: {data.num_nodes}")
            return False
        
        if hasattr(data, 'edge_index') and data.edge_index.size(1) > 10000000:
            logger.warning(f"Suspiciously large number of edges: {data.edge_index.size(1)}")
            return False
        
        # Check for negative node indices (potential index attack)
        if hasattr(data, 'edge_index') and torch.any(data.edge_index < 0):
            logger.warning("Found negative node indices")
            return False
        
        return True
    
    @staticmethod
    def hash_model_state(model) -> str:
        """Create hash of model state for integrity checking."""
        state_dict = model.state_dict()
        
        # Concatenate all parameters
        params_bytes = b''
        for param in state_dict.values():
            params_bytes += param.cpu().numpy().tobytes()
        
        return hashlib.sha256(params_bytes).hexdigest()
    
    @staticmethod
    def validate_model_integrity(model, expected_hash: str) -> bool:
        """Validate model integrity against expected hash."""
        current_hash = SecurityManager.hash_model_state(model)
        return current_hash == expected_hash

class RobustDGDNWrapper:
    """Robust wrapper around DGDN model with comprehensive error handling."""
    
    def __init__(self, model_config: Dict[str, Any]):
        self.validator = InputValidator()
        self.monitor = PerformanceMonitor()
        self.security = SecurityManager()
        
        # Validate configuration
        config_validation = self.validator.validate_model_config(model_config)
        if not config_validation.is_valid:
            raise ValueError(f"Invalid model configuration: {config_validation.errors}")
        
        if config_validation.warnings:
            for warning in config_validation.warnings:
                logger.warning(warning)
        
        # Initialize model
        try:
            from dgdn.models.dgdn import DynamicGraphDiffusionNet
            self.model = DynamicGraphDiffusionNet(**model_config)
            self.model_hash = self.security.hash_model_state(self.model)
            logger.info(f"Model initialized successfully. Hash: {self.model_hash[:16]}...")
        except Exception as e:
            logger.error(f"Failed to initialize model: {str(e)}")
            raise
        
        self.config = model_config
        self.inference_count = 0
        
    def forward(self, data, **kwargs) -> Dict[str, torch.Tensor]:
        """Robust forward pass with comprehensive validation and monitoring."""
        self.inference_count += 1
        
        # Input validation
        validation_result = self.validator.validate_temporal_data(data)
        if not validation_result.is_valid:
            error_msg = f"Input validation failed: {validation_result.errors}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        if validation_result.warnings:
            for warning in validation_result.warnings:
                logger.warning(f"Input warning: {warning}")
        
        # Security checks
        if not self.security.sanitize_input_data(data):
            raise ValueError("Input data failed security checks")
        
        # Model integrity check (every 100 inferences)
        if self.inference_count % 100 == 0:
            if not self.security.validate_model_integrity(self.model, self.model_hash):
                logger.error("Model integrity check failed!")
                raise RuntimeError("Model integrity compromised")
        
        # Performance monitoring
        with self.monitor.monitor_inference():
            try:
                # Set model to appropriate mode
                training_mode = self.model.training
                
                # Forward pass with error handling
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=UserWarning)
                    output = self.model(data, **kwargs)
                
                # Validate output
                if not isinstance(output, dict):
                    raise ValueError("Model output must be a dictionary")
                
                required_keys = ['node_embeddings', 'mean', 'logvar', 'kl_loss']
                for key in required_keys:
                    if key not in output:
                        raise ValueError(f"Missing required output key: {key}")
                
                # Check for NaN or infinite values in output
                for key, tensor in output.items():
                    if isinstance(tensor, torch.Tensor):
                        if torch.any(torch.isnan(tensor)):
                            logger.error(f"NaN values detected in output '{key}'")
                            raise ValueError(f"NaN values in output '{key}'")
                        if torch.any(torch.isinf(tensor)):
                            logger.error(f"Infinite values detected in output '{key}'")
                            raise ValueError(f"Infinite values in output '{key}'")
                
                logger.debug(f"Inference {self.inference_count} completed successfully")
                return output
                
            except torch.cuda.OutOfMemoryError:
                logger.error("GPU out of memory during inference")
                torch.cuda.empty_cache()
                raise
            except Exception as e:
                logger.error(f"Error during model forward pass: {str(e)}")
                raise
    
    def predict_edges(self, source_nodes: torch.Tensor, target_nodes: torch.Tensor, 
                     time: float, data, **kwargs) -> torch.Tensor:
        """Robust edge prediction with validation."""
        # Validate input tensors
        if not isinstance(source_nodes, torch.Tensor) or not isinstance(target_nodes, torch.Tensor):
            raise TypeError("source_nodes and target_nodes must be torch.Tensor")
        
        if source_nodes.shape != target_nodes.shape:
            raise ValueError("source_nodes and target_nodes must have the same shape")
        
        if len(source_nodes.shape) != 1:
            raise ValueError("source_nodes and target_nodes must be 1-dimensional")
        
        # Check node indices bounds
        max_node = max(source_nodes.max().item(), target_nodes.max().item())
        if max_node >= data.num_nodes:
            raise ValueError(f"Node index {max_node} exceeds num_nodes {data.num_nodes}")
        
        # Validate time
        if not isinstance(time, (int, float)) or time < 0:
            raise ValueError("time must be a non-negative number")
        
        try:
            with self.monitor.monitor_inference():
                predictions = self.model.predict_edges(source_nodes, target_nodes, time, data, **kwargs)
            
            # Validate predictions
            if not isinstance(predictions, torch.Tensor):
                raise ValueError("Edge predictions must be a torch.Tensor")
            
            if torch.any(torch.isnan(predictions)) or torch.any(torch.isinf(predictions)):
                raise ValueError("Invalid predictions (NaN or infinite values)")
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error during edge prediction: {str(e)}")
            raise
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status of the model."""
        avg_metrics = self.monitor.get_average_metrics()
        
        status = {
            'model_status': 'healthy',
            'inference_count': self.inference_count,
            'model_hash': self.model_hash[:16],
            'performance_metrics': asdict(avg_metrics),
            'warnings': [],
            'timestamp': time.time()
        }
        
        # Performance-based health checks
        if avg_metrics.memory_usage_mb > 1000:  # 1GB threshold
            status['warnings'].append(f"High memory usage: {avg_metrics.memory_usage_mb:.1f}MB")
        
        if avg_metrics.inference_time_ms > 1000:  # 1 second threshold
            status['warnings'].append(f"Slow inference: {avg_metrics.inference_time_ms:.1f}ms")
        
        if avg_metrics.cpu_usage_percent > 80:
            status['warnings'].append(f"High CPU usage: {avg_metrics.cpu_usage_percent:.1f}%")
        
        # Model integrity
        try:
            if not self.security.validate_model_integrity(self.model, self.model_hash):
                status['model_status'] = 'compromised'
                status['warnings'].append("Model integrity check failed")
        except Exception as e:
            status['warnings'].append(f"Integrity check error: {str(e)}")
        
        return status

def test_robustness_suite():
    """Test the robustness enhancements."""
    print("🛡️ Testing Generation 2: Robustness Suite")
    print("=" * 60)
    
    # Test configuration
    config = {
        'node_dim': 64,
        'hidden_dim': 128,
        'time_dim': 32,
        'num_layers': 2,
        'num_heads': 4,
        'diffusion_steps': 3,
        'dropout': 0.1
    }
    
    try:
        # Initialize robust model wrapper
        print("🔧 Initializing robust DGDN wrapper...")
        robust_model = RobustDGDNWrapper(config)
        print("✅ Robust model initialized successfully")
        
        # Create test data
        print("\n📊 Creating test data...")
        class TemporalData:
            def __init__(self):
                self.edge_index = torch.randint(0, 50, (2, 100))
                self.timestamps = torch.sort(torch.rand(100) * 100.0)[0]
                self.node_features = torch.randn(50, 64)
                self.num_nodes = 50
                
            def time_window(self, start_time, end_time):
                mask = (self.timestamps >= start_time) & (self.timestamps <= end_time)
                new_data = TemporalData()
                new_data.edge_index = self.edge_index[:, mask]
                new_data.timestamps = self.timestamps[mask]
                new_data.node_features = self.node_features
                new_data.num_nodes = self.num_nodes
                return new_data
        
        data = TemporalData()
        print("✅ Test data created")
        
        # Test robust forward pass
        print("\n🚀 Testing robust forward pass...")
        output = robust_model.forward(data)
        print(f"✅ Forward pass successful: {list(output.keys())}")
        
        # Test edge prediction
        print("\n🔗 Testing robust edge prediction...")
        src_nodes = torch.randint(0, 50, (10,))
        tgt_nodes = torch.randint(0, 50, (10,))
        predictions = robust_model.predict_edges(src_nodes, tgt_nodes, 50.0, data)
        print(f"✅ Edge prediction successful: {predictions.shape}")
        
        # Test health monitoring
        print("\n🏥 Testing health monitoring...")
        health = robust_model.get_health_status()
        print(f"✅ Health status: {health['model_status']}")
        print(f"   Inference count: {health['inference_count']}")
        print(f"   Memory usage: {health['performance_metrics']['memory_usage_mb']:.2f}MB")
        print(f"   Inference time: {health['performance_metrics']['inference_time_ms']:.2f}ms")
        
        # Test error handling
        print("\n⚠️ Testing error handling...")
        
        # Test invalid data
        try:
            class InvalidData:
                pass
            robust_model.forward(InvalidData())
            print("❌ Should have failed with invalid data")
        except ValueError as e:
            print(f"✅ Correctly caught invalid data: {str(e)[:50]}...")
        
        # Test invalid edge prediction
        try:
            invalid_nodes = torch.tensor([100])  # Out of bounds
            robust_model.predict_edges(invalid_nodes, invalid_nodes, 50.0, data)
            print("❌ Should have failed with invalid nodes")
        except ValueError as e:
            print(f"✅ Correctly caught invalid nodes: {str(e)[:50]}...")
        
        print("\n🎉 Generation 2 Robustness Suite: ALL TESTS PASSED!")
        return True
        
    except Exception as e:
        print(f"\n❌ Error in robustness suite: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_robustness_suite()
    sys.exit(0 if success else 1)