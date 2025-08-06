"""Computational optimization techniques for DGDN."""

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from typing import Dict, Any, Optional, Tuple
import logging
import time
from contextlib import contextmanager


class MixedPrecisionTrainer:
    """Mixed precision training for DGDN."""
    
    def __init__(self, enabled: bool = True, init_scale: float = 65536.0):
        """Initialize mixed precision trainer.
        
        Args:
            enabled: Whether to enable mixed precision
            init_scale: Initial gradient scaling factor
        """
        self.enabled = enabled and torch.cuda.is_available()
        self.scaler = GradScaler(enabled=self.enabled, init_scale=init_scale)
        self.logger = logging.getLogger(__name__)
        
        if self.enabled:
            self.logger.info(f"Mixed precision training enabled with init_scale={init_scale}")
        else:
            self.logger.info("Mixed precision training disabled")
    
    @contextmanager
    def autocast_context(self):
        """Context manager for automatic mixed precision."""
        with autocast(enabled=self.enabled):
            yield
    
    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """Scale loss for gradient computation."""
        if self.enabled:
            return self.scaler.scale(loss)
        return loss
    
    def step_optimizer(self, optimizer: torch.optim.Optimizer) -> bool:
        """Step optimizer with gradient scaling."""
        if self.enabled:
            self.scaler.step(optimizer)
            self.scaler.update()
            return True
        else:
            optimizer.step()
            return True
    
    def get_scale(self) -> float:
        """Get current gradient scale."""
        if self.enabled:
            return self.scaler.get_scale()
        return 1.0
    
    def state_dict(self) -> Dict[str, Any]:
        """Get scaler state dict for checkpointing."""
        if self.enabled:
            return self.scaler.state_dict()
        return {}
    
    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load scaler state dict."""
        if self.enabled and state_dict:
            self.scaler.load_state_dict(state_dict)


class ParallelismManager:
    """Manage different types of parallelism for DGDN."""
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.logger = logging.getLogger(__name__)
        self.parallelism_type = None
    
    def setup_parallelism(self, strategy: str = "auto") -> nn.Module:
        """Setup parallelism strategy.
        
        Args:
            strategy: Parallelism strategy ("auto", "dp", "ddp", "none")
        """
        num_gpus = torch.cuda.device_count()
        
        if strategy == "auto":
            if num_gpus > 1:
                strategy = "ddp" if num_gpus <= 8 else "dp"
            else:
                strategy = "none"
        
        if strategy == "dp" and num_gpus > 1:
            return self._setup_data_parallel()
        elif strategy == "ddp" and num_gpus > 1:
            return self._setup_distributed_data_parallel()
        else:
            self.logger.info("Using single GPU/CPU training")
            return self.model
    
    def _setup_data_parallel(self) -> nn.Module:
        """Setup DataParallel training."""
        self.model = nn.DataParallel(self.model)
        self.parallelism_type = "dp"
        self.logger.info(f"Setup DataParallel on {torch.cuda.device_count()} GPUs")
        return self.model
    
    def _setup_distributed_data_parallel(self) -> nn.Module:
        """Setup DistributedDataParallel training."""
        try:
            from torch.nn.parallel import DistributedDataParallel as DDP
            
            if not torch.distributed.is_initialized():
                self.logger.warning("torch.distributed not initialized, falling back to DataParallel")
                return self._setup_data_parallel()
            
            local_rank = torch.distributed.get_rank()
            device = torch.device(f'cuda:{local_rank}')
            self.model = self.model.to(device)
            self.model = DDP(self.model, device_ids=[local_rank])
            self.parallelism_type = "ddp"
            self.logger.info(f"Setup DistributedDataParallel on rank {local_rank}")
            return self.model
            
        except ImportError:
            self.logger.warning("DistributedDataParallel not available, using DataParallel")
            return self._setup_data_parallel()
    
    def get_effective_batch_size(self, base_batch_size: int) -> int:
        """Calculate effective batch size with parallelism."""
        if self.parallelism_type in ["dp", "ddp"]:
            num_gpus = torch.cuda.device_count()
            return base_batch_size * num_gpus
        return base_batch_size
    
    def optimize_communication(self) -> Dict[str, Any]:
        """Optimize communication for distributed training."""
        recommendations = {}
        
        if self.parallelism_type == "ddp":
            recommendations.update({
                'bucket_cap_mb': 25,  # Default DDP bucket size
                'find_unused_parameters': False,  # Set to True if needed
                'gradient_compression': True,
                'broadcast_buffers': False
            })
        elif self.parallelism_type == "dp":
            recommendations.update({
                'scatter_gather_optimization': True,
                'pin_memory': True,
                'non_blocking_transfer': True
            })
        
        return recommendations


class ComputationalProfiler:
    """Profile computational performance of DGDN."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.profiles = {}
    
    @contextmanager
    def profile_section(self, name: str):
        """Profile a section of code."""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        start_time = time.perf_counter()
        start_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        
        yield
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        end_time = time.perf_counter()
        end_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        
        profile_data = {
            'duration_ms': (end_time - start_time) * 1000,
            'memory_delta_mb': (end_memory - start_memory) / 1e6
        }
        
        if name not in self.profiles:
            self.profiles[name] = []
        self.profiles[name].append(profile_data)
    
    def profile_forward_pass(self, model: nn.Module, data, iterations: int = 10) -> Dict[str, float]:
        """Profile forward pass performance."""
        model.eval()
        
        # Warmup
        with torch.no_grad():
            for _ in range(3):
                _ = model(data)
        
        # Profile
        forward_times = []
        
        for _ in range(iterations):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            start_time = time.perf_counter()
            
            with torch.no_grad():
                output = model(data)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            end_time = time.perf_counter()
            forward_times.append((end_time - start_time) * 1000)
        
        return {
            'mean_forward_time_ms': sum(forward_times) / len(forward_times),
            'min_forward_time_ms': min(forward_times),
            'max_forward_time_ms': max(forward_times),
            'std_forward_time_ms': (sum((x - sum(forward_times)/len(forward_times))**2 for x in forward_times) / len(forward_times))**0.5
        }
    
    def profile_training_step(self, model: nn.Module, data, optimizer: torch.optim.Optimizer, 
                            loss_fn, iterations: int = 10) -> Dict[str, float]:
        """Profile complete training step."""
        model.train()
        
        step_times = []
        
        for _ in range(iterations):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            start_time = time.perf_counter()
            
            # Forward pass
            optimizer.zero_grad()
            output = model(data)
            targets = torch.ones(data.edge_index.shape[1])  # Dummy targets
            loss_dict = loss_fn(output, targets, data.edge_index)
            loss = loss_dict['total']
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            end_time = time.perf_counter()
            step_times.append((end_time - start_time) * 1000)
        
        return {
            'mean_training_step_ms': sum(step_times) / len(step_times),
            'min_training_step_ms': min(step_times),
            'max_training_step_ms': max(step_times),
            'std_training_step_ms': (sum((x - sum(step_times)/len(step_times))**2 for x in step_times) / len(step_times))**0.5
        }
    
    def get_profile_summary(self) -> Dict[str, Dict[str, float]]:
        """Get summary of all profiled sections."""
        summary = {}
        
        for name, profiles in self.profiles.items():
            durations = [p['duration_ms'] for p in profiles]
            memory_deltas = [p['memory_delta_mb'] for p in profiles]
            
            summary[name] = {
                'mean_duration_ms': sum(durations) / len(durations),
                'total_duration_ms': sum(durations),
                'mean_memory_delta_mb': sum(memory_deltas) / len(memory_deltas),
                'total_memory_delta_mb': sum(memory_deltas),
                'call_count': len(profiles)
            }
        
        return summary
    
    def recommend_optimizations(self, profile_data: Dict[str, Dict[str, float]]) -> Dict[str, list]:
        """Recommend optimizations based on profiling data."""
        recommendations = {
            'high_priority': [],
            'medium_priority': [],
            'low_priority': []
        }
        
        for section, metrics in profile_data.items():
            mean_duration = metrics['mean_duration_ms']
            memory_usage = metrics['mean_memory_delta_mb']
            
            if mean_duration > 1000:  # > 1 second
                recommendations['high_priority'].append(
                    f"Optimize {section}: {mean_duration:.1f}ms avg duration"
                )
            elif mean_duration > 100:  # > 100ms
                recommendations['medium_priority'].append(
                    f"Consider optimizing {section}: {mean_duration:.1f}ms avg duration"
                )
            
            if memory_usage > 1000:  # > 1GB
                recommendations['high_priority'].append(
                    f"Reduce memory usage in {section}: {memory_usage:.1f}MB avg usage"
                )
            elif memory_usage > 100:  # > 100MB
                recommendations['medium_priority'].append(
                    f"Consider memory optimization in {section}: {memory_usage:.1f}MB avg usage"
                )
        
        return recommendations