"""
Performance optimization and scaling utilities for DGDN.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import gc
import threading
from typing import Dict, Any, List, Optional, Tuple, Callable, Union
from functools import wraps, lru_cache
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
from dataclasses import dataclass
import warnings

@dataclass
class PerformanceMetrics:
    """Performance metrics tracking."""
    inference_time: float
    memory_usage: float
    throughput: float  # samples per second
    cache_hit_rate: float
    optimization_speedup: float

class PerformanceOptimizer:
    """Comprehensive performance optimization for DGDN models."""
    
    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model
        self.device = device
        self.cache = {}
        self.cache_stats = {"hits": 0, "misses": 0}
        self.performance_history = []
        
        # Optimization flags
        self.mixed_precision_enabled = False
        self.gradient_checkpointing_enabled = False
        self.model_compiled = False
        
    def enable_mixed_precision(self) -> None:
        """Enable mixed precision training/inference for performance."""
        if self.device.type == 'cuda':
            self.mixed_precision_enabled = True
            # Enable autocast for forward passes
            self.original_forward = self.model.forward
            
            def mixed_precision_forward(*args, **kwargs):
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    return self.original_forward(*args, **kwargs)
            
            self.model.forward = mixed_precision_forward
            print("Mixed precision enabled")
        else:
            warnings.warn("Mixed precision is only available on CUDA devices")
    
    def enable_gradient_checkpointing(self) -> None:
        """Enable gradient checkpointing to save memory."""
        if hasattr(self.model, 'dgdn_layers'):
            for layer in self.model.dgdn_layers:
                if hasattr(layer, 'use_checkpoint'):
                    layer.use_checkpoint = True
        
        self.gradient_checkpointing_enabled = True
        print("Gradient checkpointing enabled")
    
    def compile_model(self) -> None:
        """Compile model for optimized execution (PyTorch 2.0+)."""
        try:
            if hasattr(torch, 'compile'):
                self.model = torch.compile(self.model)
                self.model_compiled = True
                print("Model compiled successfully")
            else:
                warnings.warn("torch.compile not available in this PyTorch version")
        except Exception as e:
            warnings.warn(f"Model compilation failed: {e}")
    
    def optimize_for_inference(self) -> None:
        """Apply all inference optimizations."""
        self.model.eval()
        
        # Disable gradients for inference
        for param in self.model.parameters():
            param.requires_grad_(False)
        
        # Enable optimizations
        if self.device.type == 'cuda':
            self.enable_mixed_precision()
        
        self.compile_model()
        
        print("Model optimized for inference")
    
    def benchmark_performance(self, data_loader, num_batches: int = 10) -> PerformanceMetrics:
        """Benchmark model performance."""
        self.model.eval()
        
        inference_times = []
        memory_usage = []
        total_samples = 0
        
        with torch.no_grad():
            for i, data in enumerate(data_loader):
                if i >= num_batches:
                    break
                
                # Memory before
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                    memory_before = torch.cuda.memory_allocated()
                else:
                    memory_before = 0
                
                # Time inference
                start_time = time.time()
                output = self.model(data)
                
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                
                inference_time = time.time() - start_time
                
                # Memory after
                if self.device.type == 'cuda':
                    memory_after = torch.cuda.memory_allocated()
                    memory_used = (memory_after - memory_before) / 1e6  # MB
                else:
                    memory_used = 0
                
                inference_times.append(inference_time)
                memory_usage.append(memory_used)
                
                # Count samples (approximate)
                if hasattr(data, 'num_nodes'):
                    total_samples += data.num_nodes
                else:
                    total_samples += 1
        
        # Calculate metrics
        avg_inference_time = np.mean(inference_times)
        avg_memory_usage = np.mean(memory_usage)
        throughput = total_samples / sum(inference_times) if inference_times else 0
        cache_hit_rate = self.get_cache_hit_rate()
        
        metrics = PerformanceMetrics(
            inference_time=avg_inference_time,
            memory_usage=avg_memory_usage,
            throughput=throughput,
            cache_hit_rate=cache_hit_rate,
            optimization_speedup=1.0  # Base case
        )
        
        self.performance_history.append(metrics)
        return metrics
    
    def get_cache_hit_rate(self) -> float:
        """Get cache hit rate."""
        total = self.cache_stats["hits"] + self.cache_stats["misses"]
        return self.cache_stats["hits"] / max(total, 1)

class AdaptiveCaching:
    """Adaptive caching system for temporal graph embeddings."""
    
    def __init__(self, max_cache_size: int = 1000, ttl_seconds: float = 300.0):
        self.max_cache_size = max_cache_size
        self.ttl_seconds = ttl_seconds
        self.cache = {}
        self.access_times = {}
        self.access_counts = {}
        self.lock = threading.Lock()
    
    def _generate_key(self, data: Any) -> str:
        """Generate cache key for temporal data."""
        if hasattr(data, 'edge_index') and hasattr(data, 'timestamps'):
            # Create hash based on graph structure and time window
            edge_hash = hash(data.edge_index.detach().cpu().numpy().tobytes())
            time_hash = hash(data.timestamps.detach().cpu().numpy().tobytes())
            return f"{edge_hash}_{time_hash}"
        else:
            return str(hash(str(data)))
    
    def get(self, key: str) -> Optional[torch.Tensor]:
        """Get item from cache if valid."""
        with self.lock:
            if key in self.cache:
                current_time = time.time()
                access_time = self.access_times.get(key, 0)
                
                # Check TTL
                if current_time - access_time < self.ttl_seconds:
                    self.access_times[key] = current_time
                    self.access_counts[key] = self.access_counts.get(key, 0) + 1
                    return self.cache[key]
                else:
                    # Expired, remove from cache
                    self._remove_key(key)
            
            return None
    
    def put(self, key: str, value: torch.Tensor) -> None:
        """Add item to cache."""
        with self.lock:
            current_time = time.time()
            
            # Check if cache is full
            if len(self.cache) >= self.max_cache_size:
                self._evict_lru()
            
            self.cache[key] = value.clone().detach()
            self.access_times[key] = current_time
            self.access_counts[key] = 1
    
    def _remove_key(self, key: str) -> None:
        """Remove key from all cache structures."""
        self.cache.pop(key, None)
        self.access_times.pop(key, None)
        self.access_counts.pop(key, None)
    
    def _evict_lru(self) -> None:
        """Evict least recently used item."""
        if not self.access_times:
            return
        
        # Find LRU key
        lru_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
        self._remove_key(lru_key)
    
    def clear(self) -> None:
        """Clear all cache."""
        with self.lock:
            self.cache.clear()
            self.access_times.clear()
            self.access_counts.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            return {
                "cache_size": len(self.cache),
                "max_size": self.max_cache_size,
                "utilization": len(self.cache) / self.max_cache_size,
                "total_accesses": sum(self.access_counts.values()),
                "unique_keys": len(self.access_counts),
            }

class BatchProcessor:
    """Efficient batch processing for temporal graphs."""
    
    def __init__(self, batch_size: int = 32, num_workers: int = 4):
        self.batch_size = batch_size
        self.num_workers = min(num_workers, mp.cpu_count())
        self.executor = None
    
    def process_batch(self, model: nn.Module, batch_data: List[Any]) -> List[torch.Tensor]:
        """Process a batch of temporal graphs efficiently."""
        model.eval()
        results = []
        
        with torch.no_grad():
            for data in batch_data:
                output = model(data)
                if isinstance(output, dict):
                    results.append(output.get('node_embeddings', output))
                else:
                    results.append(output)
        
        return results
    
    def parallel_process(self, model: nn.Module, data_list: List[Any]) -> List[torch.Tensor]:
        """Process data in parallel batches."""
        # Split data into batches
        batches = [data_list[i:i+self.batch_size] for i in range(0, len(data_list), self.batch_size)]
        
        results = []
        
        # Process batches sequentially (for now, to avoid memory issues)
        for batch in batches:
            batch_results = self.process_batch(model, batch)
            results.extend(batch_results)
        
        return results

class MemoryManager:
    """Advanced memory management for large-scale processing."""
    
    def __init__(self, device: torch.device):
        self.device = device
        self.memory_threshold = 0.9  # 90% memory usage threshold
    
    def check_memory_pressure(self) -> bool:
        """Check if memory pressure is high."""
        if self.device.type == 'cuda':
            allocated = torch.cuda.memory_allocated(self.device)
            props = torch.cuda.get_device_properties(self.device)
            total = props.total_memory
            return (allocated / total) > self.memory_threshold
        return False
    
    def emergency_cleanup(self) -> None:
        """Perform emergency memory cleanup."""
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        
        # Force garbage collection
        gc.collect()
    
    def with_memory_management(self, func: Callable) -> Callable:
        """Decorator for automatic memory management."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Check memory before execution
            if self.check_memory_pressure():
                self.emergency_cleanup()
            
            try:
                result = func(*args, **kwargs)
                return result
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    # Try emergency cleanup and retry once
                    self.emergency_cleanup()
                    try:
                        result = func(*args, **kwargs)
                        return result
                    except RuntimeError:
                        raise MemoryError(f"Out of memory even after cleanup: {e}")
                else:
                    raise
        
        return wrapper

class ModelSharding:
    """Model sharding for large models that don't fit in memory."""
    
    def __init__(self, model: nn.Module, num_shards: int = 2):
        self.model = model
        self.num_shards = num_shards
        self.shards = []
        self._create_shards()
    
    def _create_shards(self) -> None:
        """Create model shards."""
        # Simple sharding by splitting layers
        if hasattr(self.model, 'dgdn_layers'):
            layers = self.model.dgdn_layers
            layers_per_shard = len(layers) // self.num_shards
            
            for i in range(self.num_shards):
                start_idx = i * layers_per_shard
                end_idx = start_idx + layers_per_shard if i < self.num_shards - 1 else len(layers)
                
                shard_layers = layers[start_idx:end_idx]
                self.shards.append(shard_layers)
    
    def forward_sharded(self, x: torch.Tensor, edge_index: torch.Tensor, 
                       temporal_encoding: torch.Tensor) -> torch.Tensor:
        """Forward pass through sharded model."""
        current_x = x
        
        for shard in self.shards:
            # Process through current shard
            for layer in shard:
                layer_output = layer(
                    x=current_x,
                    edge_index=edge_index,
                    temporal_encoding=temporal_encoding
                )
                current_x = layer_output.get("node_features", current_x)
            
            # Optional: move to CPU between shards to save GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return current_x

def profile_model_performance(model: nn.Module, data_loader, 
                            device: torch.device) -> Dict[str, Any]:
    """Comprehensive model performance profiling."""
    model.eval()
    
    # Initialize profiling
    profiling_results = {
        "total_time": 0.0,
        "forward_time": 0.0,
        "memory_peak": 0.0,
        "throughput": 0.0,
        "layer_times": {},
        "bottlenecks": [],
    }
    
    # Hook for layer profiling
    layer_times = {}
    
    def create_hook(name):
        def hook(module, input, output):
            if name not in layer_times:
                layer_times[name] = []
            # Simple timing (not perfectly accurate but gives insight)
            layer_times[name].append(time.time())
        return hook
    
    # Register hooks
    hooks = []
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf modules only
            hook = module.register_forward_hook(create_hook(name))
            hooks.append(hook)
    
    total_samples = 0
    start_time = time.time()
    
    try:
        with torch.no_grad():
            for i, data in enumerate(data_loader):
                if i >= 10:  # Limit profiling to 10 batches
                    break
                
                # Memory before
                if device.type == 'cuda':
                    memory_before = torch.cuda.memory_allocated()
                
                # Forward pass
                forward_start = time.time()
                output = model(data)
                forward_time = time.time() - forward_start
                
                profiling_results["forward_time"] += forward_time
                
                # Memory tracking
                if device.type == 'cuda':
                    memory_after = torch.cuda.memory_allocated()
                    memory_used = memory_after - memory_before
                    profiling_results["memory_peak"] = max(
                        profiling_results["memory_peak"], memory_used
                    )
                
                # Count samples
                if hasattr(data, 'num_nodes'):
                    total_samples += data.num_nodes
                else:
                    total_samples += 1
    
    finally:
        # Remove hooks
        for hook in hooks:
            hook.remove()
    
    total_time = time.time() - start_time
    profiling_results["total_time"] = total_time
    profiling_results["throughput"] = total_samples / total_time if total_time > 0 else 0
    
    # Identify bottlenecks (simplified)
    if profiling_results["forward_time"] > total_time * 0.8:
        profiling_results["bottlenecks"].append("Forward pass is the main bottleneck")
    
    if device.type == 'cuda' and profiling_results["memory_peak"] > 1e9:  # > 1GB
        profiling_results["bottlenecks"].append("High memory usage detected")
    
    return profiling_results

def optimize_graph_operations(edge_index: torch.Tensor, 
                            num_nodes: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Optimize graph operations for better performance."""
    # Sort edges for better memory access patterns
    sorted_indices = torch.argsort(edge_index[0] * num_nodes + edge_index[1])
    optimized_edge_index = edge_index[:, sorted_indices]
    
    return optimized_edge_index, sorted_indices

def create_efficient_data_loader(dataset, batch_size: int = 32, 
                               num_workers: int = 4, pin_memory: bool = True):
    """Create optimized data loader for temporal graphs."""
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # Don't shuffle for temporal data
        num_workers=min(num_workers, mp.cpu_count()),
        pin_memory=pin_memory and torch.cuda.is_available(),
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else 2,
    )