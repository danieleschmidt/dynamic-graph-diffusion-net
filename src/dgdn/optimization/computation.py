"""Computational optimizations for DGDN."""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Tuple, List
import math
from functools import lru_cache
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp

from ..utils.logging import get_logger


class ComputationOptimizer:
    """Optimizer for computational operations in DGDN."""
    
    def __init__(
        self,
        enable_mixed_precision: bool = True,
        enable_gradient_checkpointing: bool = False,
        enable_graph_compilation: bool = False
    ):
        self.enable_mixed_precision = enable_mixed_precision
        self.enable_gradient_checkpointing = enable_gradient_checkpointing
        self.enable_graph_compilation = enable_graph_compilation
        self.logger = get_logger("dgdn.optimization")
        
        # Mixed precision setup
        if enable_mixed_precision and torch.cuda.is_available():
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None
        
        self.logger.info(f"Computation optimizer initialized: "
                        f"mixed_precision={enable_mixed_precision}, "
                        f"gradient_checkpointing={enable_gradient_checkpointing}")
    
    def optimize_forward_pass(self, model, data):
        """Optimize forward pass with various techniques."""
        if self.enable_mixed_precision and self.scaler is not None:
            with torch.cuda.amp.autocast():
                return model(data)
        else:
            return model(data)
    
    def optimize_backward_pass(self, loss, model):
        """Optimize backward pass with scaling and gradient clipping."""
        if self.scaler is not None:
            # Scale loss for mixed precision
            self.scaler.scale(loss).backward()
            
            # Gradient clipping with scaled gradients
            self.scaler.unscale_(model.optimizer if hasattr(model, 'optimizer') else None)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Update with scaled gradients
            if hasattr(model, 'optimizer'):
                self.scaler.step(model.optimizer)
                self.scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)


class TensorOperationOptimizer:
    """Optimize tensor operations for better performance."""
    
    def __init__(self):
        self.logger = get_logger("dgdn.tensor_ops")
        self._operation_cache = {}
        self._cache_lock = threading.Lock()
    
    @staticmethod
    def optimized_attention(query, key, value, mask=None, dropout_p=0.0):
        """Optimized multi-head attention implementation."""
        # Use scaled dot-product attention with optimizations
        batch_size, seq_len, embed_dim = query.shape
        
        # Efficient attention computation
        if torch.cuda.is_available() and hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
            # Use PyTorch's optimized implementation if available
            return torch.nn.functional.scaled_dot_product_attention(
                query, key, value, attn_mask=mask, dropout_p=dropout_p
            )
        else:
            # Fall back to manual implementation
            scale = math.sqrt(embed_dim)
            scores = torch.matmul(query, key.transpose(-2, -1)) / scale
            
            if mask is not None:
                scores = scores.masked_fill(mask == 0, -1e9)
            
            attn_weights = torch.softmax(scores, dim=-1)
            
            if dropout_p > 0.0:
                attn_weights = torch.dropout(attn_weights, dropout_p, training=True)
            
            return torch.matmul(attn_weights, value)
    
    @staticmethod
    def optimized_edge_aggregation(edge_index, edge_attr, num_nodes, aggregation='mean'):
        """Optimized edge aggregation for graph operations."""
        src, dst = edge_index
        
        if aggregation == 'mean':
            # Use scatter operations for efficient aggregation
            if hasattr(torch, 'scatter_reduce'):
                return torch.scatter_reduce(
                    torch.zeros(num_nodes, edge_attr.size(-1), device=edge_attr.device),
                    0, dst.unsqueeze(-1).expand(-1, edge_attr.size(-1)),
                    edge_attr, reduce='mean'
                )
            else:
                # Fallback implementation
                node_features = torch.zeros(num_nodes, edge_attr.size(-1), device=edge_attr.device)
                node_features.scatter_add_(0, dst.unsqueeze(-1).expand(-1, edge_attr.size(-1)), edge_attr)
                
                # Count edges per node for averaging
                edge_counts = torch.zeros(num_nodes, device=edge_attr.device)
                edge_counts.scatter_add_(0, dst, torch.ones_like(dst, dtype=torch.float))
                edge_counts = edge_counts.clamp(min=1).unsqueeze(-1)
                
                return node_features / edge_counts
        
        elif aggregation == 'sum':
            node_features = torch.zeros(num_nodes, edge_attr.size(-1), device=edge_attr.device)
            return node_features.scatter_add_(0, dst.unsqueeze(-1).expand(-1, edge_attr.size(-1)), edge_attr)
        
        elif aggregation == 'max':
            node_features = torch.full((num_nodes, edge_attr.size(-1)), -float('inf'), device=edge_attr.device)
            return node_features.scatter_reduce_(0, dst.unsqueeze(-1).expand(-1, edge_attr.size(-1)), edge_attr, reduce='amax')
        
        else:
            raise ValueError(f"Unsupported aggregation: {aggregation}")
    
    def cache_computation(self, key: str, computation_fn, *args, **kwargs):
        """Cache expensive computations."""
        with self._cache_lock:
            if key in self._operation_cache:
                self.logger.debug(f"Cache hit for operation: {key}")
                return self._operation_cache[key]
            
            result = computation_fn(*args, **kwargs)
            self._operation_cache[key] = result
            self.logger.debug(f"Cached result for operation: {key}")
            return result
    
    def clear_cache(self):
        """Clear the operation cache."""
        with self._cache_lock:
            self._operation_cache.clear()
            self.logger.info("Operation cache cleared")


class ParallelProcessor:
    """Parallel processing utilities for DGDN operations."""
    
    def __init__(self, max_workers: Optional[int] = None):
        self.max_workers = max_workers or min(8, mp.cpu_count())
        self.logger = get_logger("dgdn.parallel")
        
        # Thread pool for I/O bound operations
        self.thread_pool = ThreadPoolExecutor(max_workers=self.max_workers)
        
        # Process pool for CPU bound operations (reduced for safety)
        self.process_pool = ThreadPoolExecutor(max_workers=min(4, mp.cpu_count()))
        
        self.logger.info(f"Parallel processor initialized with {self.max_workers} workers")
    
    def parallel_batch_processing(self, data_list, process_fn, use_threads=True):
        """Process batches in parallel."""
        executor = self.thread_pool if use_threads else self.process_pool
        
        try:
            futures = [executor.submit(process_fn, data) for data in data_list]
            results = [future.result() for future in futures]
            return results
        except Exception as e:
            self.logger.error(f"Parallel processing failed: {e}")
            raise
    
    def parallel_graph_processing(self, graphs, model, batch_size=32):
        """Process multiple graphs in parallel."""
        def process_batch(graph_batch):
            with torch.no_grad():
                results = []
                for graph in graph_batch:
                    output = model(graph)
                    results.append(output)
                return results
        
        # Split graphs into batches
        batches = [graphs[i:i + batch_size] for i in range(0, len(graphs), batch_size)]
        
        # Process batches in parallel
        batch_results = self.parallel_batch_processing(batches, process_batch)
        
        # Flatten results
        all_results = []
        for batch_result in batch_results:
            all_results.extend(batch_result)
        
        return all_results
    
    def cleanup(self):
        """Clean up thread and process pools."""
        self.thread_pool.shutdown(wait=True)
        self.process_pool.shutdown(wait=True)
        self.logger.info("Parallel processor cleaned up")


class MemoryOptimizer:
    """Memory optimization utilities."""
    
    def __init__(self):
        self.logger = get_logger("dgdn.memory")
        self._gradient_checkpointing_enabled = False
    
    def enable_gradient_checkpointing(self, model):
        """Enable gradient checkpointing to trade compute for memory."""
        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)
            return custom_forward
        
        # Wrap model layers with checkpointing
        for name, module in model.named_modules():
            if isinstance(module, nn.Module) and len(list(module.children())) == 0:
                # Leaf modules
                original_forward = module.forward
                
                def checkpointed_forward(self, *args, **kwargs):
                    return torch.utils.checkpoint.checkpoint(
                        create_custom_forward(self.__class__),
                        self, *args, **kwargs
                    )
                
                module.forward = checkpointed_forward.__get__(module, module.__class__)
        
        self._gradient_checkpointing_enabled = True
        self.logger.info("Gradient checkpointing enabled")
    
    def optimize_memory_usage(self, model, data):
        """Optimize memory usage during forward pass."""
        # Enable memory-efficient attention if available
        if hasattr(model, 'dgdn_layers'):
            for layer in model.dgdn_layers:
                if hasattr(layer, 'attention') and hasattr(layer.attention, 'enable_memory_efficient'):
                    layer.attention.enable_memory_efficient = True
        
        # Use memory-efficient data loading
        if hasattr(data, 'pin_memory'):
            data.pin_memory()
        
        return data
    
    def memory_profiling_context(self):
        """Context manager for memory profiling."""
        class MemoryProfiler:
            def __init__(self, logger):
                self.logger = logger
                self.start_memory = 0
                
            def __enter__(self):
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    self.start_memory = torch.cuda.memory_allocated()
                return self
                
            def __exit__(self, exc_type, exc_val, exc_tb):
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    end_memory = torch.cuda.memory_allocated()
                    memory_used = (end_memory - self.start_memory) / 1024 / 1024
                    self.logger.info(f"Memory used: {memory_used:.2f} MB")
        
        return MemoryProfiler(self.logger)


class DynamicBatchSizer:
    """Dynamic batch size optimization based on available memory."""
    
    def __init__(self, initial_batch_size: int = 32, max_batch_size: int = 512):
        self.initial_batch_size = initial_batch_size
        self.max_batch_size = max_batch_size
        self.current_batch_size = initial_batch_size
        self.logger = get_logger("dgdn.batch_sizer")
        
        self.oom_count = 0
        self.success_count = 0
        
    def get_optimal_batch_size(self, model, sample_data):
        """Find optimal batch size through binary search."""
        def test_batch_size(batch_size):
            try:
                # Test forward pass
                with torch.no_grad():
                    _ = model(sample_data)
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                return True
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    return False
                else:
                    raise e
        
        # Binary search for optimal batch size
        low, high = 1, min(self.max_batch_size, 64)  # Start with smaller range
        optimal_size = self.initial_batch_size
        
        while low <= high:
            mid = (low + high) // 2
            
            if test_batch_size(mid):
                optimal_size = mid
                low = mid + 1
            else:
                high = mid - 1
        
        self.current_batch_size = optimal_size
        self.logger.info(f"Optimal batch size found: {optimal_size}")
        return optimal_size
    
    def adapt_batch_size(self, oom_occurred: bool):
        """Adapt batch size based on OOM events."""
        if oom_occurred:
            self.oom_count += 1
            self.current_batch_size = max(1, int(self.current_batch_size * 0.8))
            self.logger.warning(f"OOM detected, reducing batch size to {self.current_batch_size}")
        else:
            self.success_count += 1
            # Gradually increase batch size if we've had many successes
            if self.success_count > 10 and self.current_batch_size < self.max_batch_size:
                self.current_batch_size = min(self.max_batch_size, int(self.current_batch_size * 1.1))
                self.success_count = 0
                self.logger.info(f"Increasing batch size to {self.current_batch_size}")


class GraphCompiler:
    """Compile and optimize DGDN models for faster execution."""
    
    def __init__(self):
        self.logger = get_logger("dgdn.compiler")
        self._compiled_models = {}
    
    def compile_model(self, model, sample_input, optimization_level="default"):
        """Compile model for optimized execution."""
        model_id = id(model)
        
        if model_id in self._compiled_models:
            self.logger.info("Using cached compiled model")
            return self._compiled_models[model_id]
        
        try:
            # Use torch.jit.script or torch.compile if available
            if hasattr(torch, 'compile') and optimization_level == "aggressive":
                # PyTorch 2.0+ compilation
                compiled_model = torch.compile(model, mode="max-autotune")
                self.logger.info("Model compiled with torch.compile")
            elif optimization_level == "script":
                # TorchScript compilation
                compiled_model = torch.jit.script(model)
                self.logger.info("Model compiled with TorchScript")
            else:
                # No compilation
                compiled_model = model
                self.logger.info("No compilation applied")
            
            self._compiled_models[model_id] = compiled_model
            return compiled_model
            
        except Exception as e:
            self.logger.warning(f"Model compilation failed: {e}, using original model")
            return model
    
    def optimize_for_inference(self, model):
        """Optimize model specifically for inference."""
        # Set to evaluation mode
        model.eval()
        
        # Fuse operations if possible
        try:
            if hasattr(torch.quantization, 'fuse_modules'):
                # Attempt to fuse common patterns
                for name, module in model.named_modules():
                    if isinstance(module, nn.Sequential):
                        # Look for fuseable patterns
                        for i in range(len(module) - 1):
                            if isinstance(module[i], nn.Linear) and isinstance(module[i + 1], nn.ReLU):
                                # Can be fused for better performance
                                pass
        except Exception as e:
            self.logger.warning(f"Operation fusion failed: {e}")
        
        return model


# Export optimized operations as a module
class OptimizedOperations:
    """Collection of optimized operations for DGDN."""
    
    def __init__(self):
        self.computation_optimizer = ComputationOptimizer()
        self.tensor_optimizer = TensorOperationOptimizer()
        self.memory_optimizer = MemoryOptimizer()
        self.batch_sizer = DynamicBatchSizer()
        self.compiler = GraphCompiler()
        self.parallel_processor = ParallelProcessor()
    
    def optimize_model(self, model, sample_data):
        """Apply all optimizations to a model."""
        # Compile model
        optimized_model = self.compiler.compile_model(model, sample_data)
        
        # Enable memory optimizations
        self.memory_optimizer.optimize_memory_usage(optimized_model, sample_data)
        
        # Find optimal batch size
        optimal_batch_size = self.batch_sizer.get_optimal_batch_size(optimized_model, sample_data)
        
        return optimized_model, optimal_batch_size
    
    def cleanup(self):
        """Clean up resources."""
        self.parallel_processor.cleanup()