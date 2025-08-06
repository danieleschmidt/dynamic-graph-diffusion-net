"""Memory optimization techniques for DGDN."""

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from typing import Dict, Any, Optional, Tuple, Callable
import gc
import psutil
import logging


class MemoryOptimizer:
    """Memory usage optimization for DGDN training."""
    
    def __init__(self, threshold_gb: float = 0.8):
        """Initialize memory optimizer.
        
        Args:
            threshold_gb: Memory usage threshold as fraction of available RAM
        """
        self.threshold_gb = threshold_gb
        self.logger = logging.getLogger(__name__)
        
    def optimize_batch_size(self, initial_batch_size: int, model: nn.Module, 
                           sample_data, device: torch.device) -> int:
        """Dynamically optimize batch size based on memory usage."""
        if not torch.cuda.is_available():
            return initial_batch_size
            
        batch_size = initial_batch_size
        max_batch_size = initial_batch_size * 4
        
        # Binary search for optimal batch size
        low, high = 1, max_batch_size
        optimal_batch_size = initial_batch_size
        
        while low <= high:
            mid_batch_size = (low + high) // 2
            
            try:
                # Test memory usage with this batch size
                if self._test_memory_usage(mid_batch_size, model, sample_data, device):
                    optimal_batch_size = mid_batch_size
                    low = mid_batch_size + 1
                else:
                    high = mid_batch_size - 1
                    
            except RuntimeError as e:
                if "out of memory" in str(e):
                    high = mid_batch_size - 1
                    torch.cuda.empty_cache()
                else:
                    raise e
        
        self.logger.info(f"Optimized batch size: {initial_batch_size} -> {optimal_batch_size}")
        return optimal_batch_size
    
    def _test_memory_usage(self, batch_size: int, model: nn.Module, 
                          sample_data, device: torch.device) -> bool:
        """Test if batch size fits in memory."""
        model.train()
        
        # Create larger batch
        test_data = self._expand_batch(sample_data, batch_size)
        test_data = test_data.to(device)
        
        try:
            with torch.no_grad():
                output = model(test_data)
                
                # Check GPU memory usage
                if torch.cuda.is_available():
                    memory_allocated = torch.cuda.memory_allocated(device)
                    memory_cached = torch.cuda.memory_reserved(device)
                    total_memory = torch.cuda.get_device_properties(device).total_memory
                    
                    usage_fraction = (memory_allocated + memory_cached) / total_memory
                    return usage_fraction < self.threshold_gb
                else:
                    return True
                    
        except RuntimeError as e:
            if "out of memory" in str(e):
                return False
            raise e
        finally:
            del test_data
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def _expand_batch(self, sample_data, target_batch_size: int):
        """Expand sample data to target batch size."""
        current_edges = sample_data.edge_index.size(1)
        if current_edges == 0:
            return sample_data
            
        # Calculate repetition factor
        repeat_factor = max(1, target_batch_size // current_edges)
        
        # Repeat edge data
        edge_index = sample_data.edge_index.repeat(1, repeat_factor)
        timestamps = sample_data.timestamps.repeat(repeat_factor)
        
        # Create new data object
        expanded_data = type(sample_data)(
            edge_index=edge_index,
            timestamps=timestamps,
            num_nodes=sample_data.num_nodes
        )
        
        if hasattr(sample_data, 'edge_attr') and sample_data.edge_attr is not None:
            expanded_data.edge_attr = sample_data.edge_attr.repeat(repeat_factor, 1)
        if hasattr(sample_data, 'node_features') and sample_data.node_features is not None:
            expanded_data.node_features = sample_data.node_features
            
        return expanded_data
    
    def monitor_memory_usage(self) -> Dict[str, float]:
        """Monitor current memory usage."""
        memory_info = {}
        
        # System memory
        process = psutil.Process()
        memory_info['ram_used_gb'] = process.memory_info().rss / 1e9
        memory_info['ram_percent'] = psutil.virtual_memory().percent
        
        # GPU memory
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                device = torch.device(f'cuda:{i}')
                allocated = torch.cuda.memory_allocated(device) / 1e9
                cached = torch.cuda.memory_reserved(device) / 1e9
                total = torch.cuda.get_device_properties(device).total_memory / 1e9
                
                memory_info[f'gpu_{i}_allocated_gb'] = allocated
                memory_info[f'gpu_{i}_cached_gb'] = cached
                memory_info[f'gpu_{i}_total_gb'] = total
                memory_info[f'gpu_{i}_usage_percent'] = (allocated + cached) / total * 100
        
        return memory_info
    
    def cleanup_memory(self):
        """Aggressive memory cleanup."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()


class GradientCheckpointing:
    """Gradient checkpointing to reduce memory usage."""
    
    def __init__(self, model: nn.Module, checkpoint_every: int = 1):
        """Initialize gradient checkpointing.
        
        Args:
            model: Model to apply checkpointing to
            checkpoint_every: Checkpoint every N layers
        """
        self.model = model
        self.checkpoint_every = checkpoint_every
        self.logger = logging.getLogger(__name__)
        
    def enable_checkpointing(self):
        """Enable gradient checkpointing for DGDN layers."""
        if hasattr(self.model, 'dgdn_layers'):
            for i, layer in enumerate(self.model.dgdn_layers):
                if i % self.checkpoint_every == 0:
                    layer.forward = self._checkpoint_wrapper(layer.forward)
            
            self.logger.info(f"Enabled gradient checkpointing for {len(self.model.dgdn_layers)} layers")
    
    def _checkpoint_wrapper(self, forward_fn: Callable) -> Callable:
        """Wrap forward function with gradient checkpointing."""
        def checkpointed_forward(*args, **kwargs):
            # Only use checkpointing during training
            if self.model.training:
                return checkpoint(forward_fn, *args, **kwargs)
            else:
                return forward_fn(*args, **kwargs)
        
        return checkpointed_forward
    
    def estimate_memory_savings(self, model_size_mb: float) -> Dict[str, float]:
        """Estimate memory savings from checkpointing."""
        # Rough estimates based on typical savings
        base_memory = model_size_mb
        
        # Gradient checkpointing typically saves 50-80% of activation memory
        activation_memory = base_memory * 2.0  # Typical ratio
        savings = activation_memory * 0.65  # Average savings
        
        return {
            'base_model_mb': base_memory,
            'estimated_activation_memory_mb': activation_memory,
            'estimated_savings_mb': savings,
            'total_memory_with_checkpointing_mb': base_memory + activation_memory - savings
        }


class DataParallelOptimizer:
    """Optimize data parallel training."""
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.logger = logging.getLogger(__name__)
    
    def setup_data_parallel(self, device_ids: Optional[list] = None) -> nn.Module:
        """Setup data parallel training."""
        if not torch.cuda.is_available():
            self.logger.warning("CUDA not available, skipping data parallel setup")
            return self.model
        
        if device_ids is None:
            device_ids = list(range(torch.cuda.device_count()))
        
        if len(device_ids) > 1:
            self.model = nn.DataParallel(self.model, device_ids=device_ids)
            self.logger.info(f"Setup data parallel training on devices: {device_ids}")
        
        return self.model
    
    def optimize_for_multi_gpu(self) -> Dict[str, Any]:
        """Optimize model for multi-GPU training."""
        recommendations = {}
        
        if torch.cuda.device_count() > 1:
            recommendations['use_data_parallel'] = True
            recommendations['optimal_batch_size'] = self._calculate_optimal_batch_size()
            recommendations['gradient_accumulation_steps'] = self._calculate_gradient_accumulation()
        else:
            recommendations['use_data_parallel'] = False
            recommendations['single_gpu_optimizations'] = [
                'Use gradient checkpointing',
                'Use mixed precision training',
                'Optimize batch size for memory'
            ]
        
        return recommendations
    
    def _calculate_optimal_batch_size(self) -> int:
        """Calculate optimal batch size for multi-GPU setup."""
        num_gpus = torch.cuda.device_count()
        base_batch_size = 32  # Conservative base
        
        # Scale batch size with number of GPUs
        return base_batch_size * num_gpus
    
    def _calculate_gradient_accumulation(self) -> int:
        """Calculate gradient accumulation steps."""
        num_gpus = torch.cuda.device_count()
        
        # Use gradient accumulation to simulate larger batches
        if num_gpus >= 4:
            return 2  # Accumulate gradients over 2 steps
        return 1