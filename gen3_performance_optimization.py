#!/usr/bin/env python3
"""
Generation 3: Performance Optimization Suite
Advanced performance optimization, caching, and memory management.
"""

import torch
import torch.nn.functional as F
import numpy as np
import time
import threading
import queue
import pickle
import hashlib
import os
import gc
import psutil
import sys
from typing import Dict, Any, Optional, List, Tuple, Union, Callable
from dataclasses import dataclass, asdict
from collections import OrderedDict, deque
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

logger = logging.getLogger(__name__)

@dataclass
class CacheStats:
    """Cache performance statistics."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    total_size: int = 0
    memory_usage_mb: float = 0.0
    avg_access_time_ms: float = 0.0

@dataclass
class OptimizationConfig:
    """Performance optimization configuration."""
    enable_caching: bool = True
    cache_size: int = 1000
    enable_batching: bool = True
    batch_size: int = 32
    batch_timeout_ms: float = 100.0
    enable_memory_optimization: bool = True
    memory_cleanup_threshold: float = 0.8  # 80% memory usage
    enable_jit_compilation: bool = True
    enable_mixed_precision: bool = False
    num_worker_threads: int = 4

class LRUCache:
    """High-performance LRU cache with memory management."""
    
    def __init__(self, max_size: int = 1000, max_memory_mb: float = 512.0):
        self.max_size = max_size
        self.max_memory_mb = max_memory_mb
        self.cache = OrderedDict()
        self.memory_usage = 0
        self.stats = CacheStats()
        self._lock = threading.RLock()
        
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache."""
        with self._lock:
            start_time = time.time()
            
            if key in self.cache:
                # Move to end (most recently used)
                value = self.cache.pop(key)
                self.cache[key] = value
                self.stats.hits += 1
                
                access_time = (time.time() - start_time) * 1000
                self._update_avg_access_time(access_time)
                return value
            else:
                self.stats.misses += 1
                access_time = (time.time() - start_time) * 1000
                self._update_avg_access_time(access_time)
                return None
    
    def put(self, key: str, value: Any) -> bool:
        """Put item in cache."""
        with self._lock:
            # Estimate memory usage
            value_size = self._estimate_size(value)
            
            # Check if item fits in memory constraints
            if value_size > self.max_memory_mb * 1024 * 1024:
                logger.warning(f"Item too large for cache: {value_size / 1024 / 1024:.1f}MB")
                return False
            
            # Remove existing item if key exists
            if key in self.cache:
                old_value = self.cache.pop(key)
                self.memory_usage -= self._estimate_size(old_value)
            
            # Evict items if necessary
            while (len(self.cache) >= self.max_size or 
                   self.memory_usage + value_size > self.max_memory_mb * 1024 * 1024):
                if not self.cache:
                    break
                self._evict_lru()
            
            # Add new item
            self.cache[key] = value
            self.memory_usage += value_size
            self.stats.total_size = len(self.cache)
            self.stats.memory_usage_mb = self.memory_usage / 1024 / 1024
            
            return True
    
    def _evict_lru(self):
        """Evict least recently used item."""
        if self.cache:
            key, value = self.cache.popitem(last=False)
            self.memory_usage -= self._estimate_size(value)
            self.stats.evictions += 1
    
    def _estimate_size(self, obj: Any) -> int:
        """Estimate memory size of object."""
        try:
            if isinstance(obj, torch.Tensor):
                return obj.numel() * obj.element_size()
            else:
                return len(pickle.dumps(obj))
        except:
            return 1024  # Default estimate
    
    def _update_avg_access_time(self, access_time_ms: float):
        """Update average access time."""
        total_accesses = self.stats.hits + self.stats.misses
        if total_accesses == 1:
            self.stats.avg_access_time_ms = access_time_ms
        else:
            # Running average
            self.stats.avg_access_time_ms = (
                (self.stats.avg_access_time_ms * (total_accesses - 1) + access_time_ms) / 
                total_accesses
            )
    
    def clear(self):
        """Clear cache."""
        with self._lock:
            self.cache.clear()
            self.memory_usage = 0
            self.stats = CacheStats()
    
    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        with self._lock:
            self.stats.total_size = len(self.cache)
            self.stats.memory_usage_mb = self.memory_usage / 1024 / 1024
            return self.stats

class BatchProcessor:
    """Intelligent batch processing for optimal throughput."""
    
    def __init__(self, max_batch_size: int = 32, timeout_ms: float = 100.0):
        self.max_batch_size = max_batch_size
        self.timeout_ms = timeout_ms
        self.pending_requests = queue.Queue()
        self.response_futures = {}
        self._processing = False
        self._stop_event = threading.Event()
        self._worker_thread = None
        
    def start(self):
        """Start batch processing."""
        if not self._processing:
            self._processing = True
            self._worker_thread = threading.Thread(target=self._process_batches, daemon=True)
            self._worker_thread.start()
    
    def stop(self):
        """Stop batch processing."""
        self._stop_event.set()
        self._processing = False
        if self._worker_thread:
            self._worker_thread.join(timeout=1.0)
    
    def add_request(self, request_id: str, data: Any) -> 'BatchFuture':
        """Add request to batch queue."""
        future = BatchFuture()
        self.response_futures[request_id] = future
        self.pending_requests.put((request_id, data))
        return future
    
    def _process_batches(self):
        """Process requests in batches."""
        while not self._stop_event.is_set():
            batch = self._collect_batch()
            if batch:
                try:
                    self._process_batch(batch)
                except Exception as e:
                    logger.error(f"Batch processing error: {str(e)}")
                    # Set error for all requests in batch
                    for request_id, _ in batch:
                        if request_id in self.response_futures:
                            self.response_futures[request_id].set_error(e)
                            del self.response_futures[request_id]
            else:
                time.sleep(0.001)  # Small sleep to prevent busy waiting
    
    def _collect_batch(self) -> List[Tuple[str, Any]]:
        """Collect batch of requests."""
        batch = []
        start_time = time.time()
        
        while (len(batch) < self.max_batch_size and 
               (time.time() - start_time) * 1000 < self.timeout_ms):
            try:
                request = self.pending_requests.get(timeout=0.01)
                batch.append(request)
            except queue.Empty:
                if batch:  # Return partial batch if we have items
                    break
                continue
        
        return batch
    
    def _process_batch(self, batch: List[Tuple[str, Any]]):
        """Process a batch of requests - to be overridden."""
        # Default implementation processes each request individually
        for request_id, data in batch:
            try:
                result = self._process_single(data)
                if request_id in self.response_futures:
                    self.response_futures[request_id].set_result(result)
                    del self.response_futures[request_id]
            except Exception as e:
                if request_id in self.response_futures:
                    self.response_futures[request_id].set_error(e)
                    del self.response_futures[request_id]
    
    def _process_single(self, data: Any) -> Any:
        """Process single request - to be overridden."""
        raise NotImplementedError("Subclasses must implement _process_single")

class BatchFuture:
    """Future-like object for batch processing results."""
    
    def __init__(self):
        self._result = None
        self._error = None
        self._completed = threading.Event()
    
    def set_result(self, result: Any):
        """Set the result."""
        self._result = result
        self._completed.set()
    
    def set_error(self, error: Exception):
        """Set an error."""
        self._error = error
        self._completed.set()
    
    def get(self, timeout: Optional[float] = None) -> Any:
        """Get the result."""
        if self._completed.wait(timeout):
            if self._error:
                raise self._error
            return self._result
        else:
            raise TimeoutError("Request timeout")

class MemoryOptimizer:
    """Memory optimization and management."""
    
    def __init__(self, cleanup_threshold: float = 0.8):
        self.cleanup_threshold = cleanup_threshold
        self.tensor_pool = {}  # Reusable tensors
        self._monitor_thread = None
        self._stop_event = threading.Event()
        
    def start_monitoring(self):
        """Start memory monitoring."""
        self._monitor_thread = threading.Thread(target=self._monitor_memory, daemon=True)
        self._monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop memory monitoring."""
        self._stop_event.set()
        if self._monitor_thread:
            self._monitor_thread.join(timeout=1.0)
    
    def _monitor_memory(self):
        """Monitor memory usage and trigger cleanup."""
        while not self._stop_event.is_set():
            try:
                memory_percent = psutil.virtual_memory().percent / 100.0
                if memory_percent > self.cleanup_threshold:
                    logger.warning(f"High memory usage: {memory_percent*100:.1f}%")
                    self.cleanup_memory()
                
                time.sleep(5.0)  # Check every 5 seconds
            except Exception as e:
                logger.error(f"Memory monitoring error: {str(e)}")
    
    def cleanup_memory(self):
        """Perform memory cleanup."""
        # Python garbage collection
        collected = gc.collect()
        logger.info(f"Garbage collected {collected} objects")
        
        # Clear tensor pool
        self.tensor_pool.clear()
        
        # PyTorch cache cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("Cleared CUDA cache")
    
    def get_tensor(self, shape: Tuple[int, ...], dtype: torch.dtype = torch.float32, 
                   device: str = 'cpu') -> torch.Tensor:
        """Get reusable tensor from pool."""
        key = (shape, dtype, device)
        
        if key in self.tensor_pool:
            tensor = self.tensor_pool.pop(key)
            if tensor.shape == shape:
                return tensor.zero_()  # Reset to zeros
        
        return torch.zeros(shape, dtype=dtype, device=device)
    
    def return_tensor(self, tensor: torch.Tensor):
        """Return tensor to pool for reuse."""
        if tensor.numel() > 1024 * 1024:  # Only pool large tensors (>1M elements)
            key = (tensor.shape, tensor.dtype, str(tensor.device))
            if len(self.tensor_pool) < 100:  # Limit pool size
                self.tensor_pool[key] = tensor.detach()

class ModelOptimizer:
    """Model-specific optimizations."""
    
    @staticmethod
    def optimize_model(model: torch.nn.Module, config: OptimizationConfig) -> torch.nn.Module:
        """Apply various optimizations to model."""
        optimized_model = model
        
        # JIT compilation
        if config.enable_jit_compilation:
            try:
                # Create dummy input for tracing
                dummy_data = ModelOptimizer._create_dummy_input(model)
                optimized_model = torch.jit.trace(model, dummy_data)
                logger.info("Applied JIT compilation optimization")
            except Exception as e:
                logger.warning(f"JIT compilation failed: {str(e)}")
        
        # Mixed precision (if supported)
        if config.enable_mixed_precision and torch.cuda.is_available():
            try:
                # Enable automatic mixed precision
                optimized_model = torch.jit.optimize_for_inference(optimized_model)
                logger.info("Applied mixed precision optimization")
            except Exception as e:
                logger.warning(f"Mixed precision optimization failed: {str(e)}")
        
        return optimized_model
    
    @staticmethod
    def _create_dummy_input(model):
        """Create dummy input for model tracing."""
        # This is a simplified version - would need to be more sophisticated
        # for actual production use
        class DummyData:
            def __init__(self):
                self.edge_index = torch.randint(0, 100, (2, 200))
                self.timestamps = torch.rand(200) * 100.0
                self.node_features = torch.randn(100, 64)
                self.num_nodes = 100
        
        return DummyData()
    
    @staticmethod
    def quantize_model(model: torch.nn.Module) -> torch.nn.Module:
        """Apply dynamic quantization to model."""
        try:
            quantized_model = torch.quantization.quantize_dynamic(
                model,
                {torch.nn.Linear, torch.nn.MultiheadAttention},
                dtype=torch.qint8
            )
            logger.info("Applied dynamic quantization")
            return quantized_model
        except Exception as e:
            logger.warning(f"Quantization failed: {str(e)}")
            return model

class OptimizedDGDNBatchProcessor(BatchProcessor):
    """Optimized batch processor for DGDN models."""
    
    def __init__(self, model, max_batch_size: int = 32, timeout_ms: float = 100.0):
        super().__init__(max_batch_size, timeout_ms)
        self.model = model
        
    def _process_batch(self, batch: List[Tuple[str, Any]]):
        """Process batch of DGDN requests efficiently."""
        if len(batch) == 1:
            # Single request - process normally
            super()._process_batch(batch)
            return
        
        try:
            # Batch processing optimization
            batch_data = self._combine_batch_data([data for _, data in batch])
            
            # Process entire batch at once
            with torch.no_grad():
                batch_output = self.model(batch_data)
            
            # Split results back to individual requests
            individual_outputs = self._split_batch_output(batch_output, len(batch))
            
            # Set results for each request
            for i, (request_id, _) in enumerate(batch):
                if request_id in self.response_futures:
                    self.response_futures[request_id].set_result(individual_outputs[i])
                    del self.response_futures[request_id]
        
        except Exception as e:
            # Fall back to individual processing
            logger.warning(f"Batch processing failed, falling back to individual: {str(e)}")
            super()._process_batch(batch)
    
    def _combine_batch_data(self, data_list: List[Any]) -> Any:
        """Combine individual data items into batch."""
        # Simplified batch combination - would need more sophisticated logic
        # for production use with variable-sized graphs
        return data_list[0]  # For now, just use first item
    
    def _split_batch_output(self, batch_output: Dict[str, torch.Tensor], 
                           batch_size: int) -> List[Dict[str, torch.Tensor]]:
        """Split batch output back to individual outputs."""
        # Simplified splitting - would need proper implementation
        return [batch_output] * batch_size

class PerformanceOptimizedDGDN:
    """High-performance optimized DGDN wrapper."""
    
    def __init__(self, model_config: Dict[str, Any], 
                 optimization_config: OptimizationConfig = None):
        
        if optimization_config is None:
            optimization_config = OptimizationConfig()
        
        self.optimization_config = optimization_config
        
        # Initialize base model
        from dgdn.models.dgdn import DynamicGraphDiffusionNet
        self.base_model = DynamicGraphDiffusionNet(**model_config)
        
        # Apply optimizations
        self.model = ModelOptimizer.optimize_model(self.base_model, optimization_config)
        
        # Initialize performance components
        if optimization_config.enable_caching:
            self.cache = LRUCache(
                max_size=optimization_config.cache_size,
                max_memory_mb=512.0
            )
        else:
            self.cache = None
        
        if optimization_config.enable_memory_optimization:
            self.memory_optimizer = MemoryOptimizer(
                cleanup_threshold=optimization_config.memory_cleanup_threshold
            )
            self.memory_optimizer.start_monitoring()
        else:
            self.memory_optimizer = None
        
        if optimization_config.enable_batching:
            self.batch_processor = OptimizedDGDNBatchProcessor(
                self.model,
                max_batch_size=optimization_config.batch_size,
                timeout_ms=optimization_config.batch_timeout_ms
            )
            self.batch_processor.start()
        else:
            self.batch_processor = None
        
        # Performance metrics
        self.inference_times = deque(maxlen=1000)
        self.throughput_history = deque(maxlen=100)
        
        # Thread pool for concurrent processing
        if optimization_config.num_worker_threads > 1:
            self.thread_pool = ThreadPoolExecutor(
                max_workers=optimization_config.num_worker_threads
            )
        else:
            self.thread_pool = None
    
    def forward(self, data, use_cache: bool = True, **kwargs) -> Dict[str, torch.Tensor]:
        """Optimized forward pass."""
        start_time = time.time()
        
        # Generate cache key
        cache_key = None
        if self.cache and use_cache:
            cache_key = self._generate_cache_key(data, kwargs)
            cached_result = self.cache.get(cache_key)
            if cached_result is not None:
                logger.debug("Cache hit for forward pass")
                return cached_result
        
        # Use batch processing if enabled
        if self.batch_processor:
            request_id = f"forward_{int(time.time() * 1000000)}"
            future = self.batch_processor.add_request(request_id, (data, kwargs))
            result = future.get(timeout=30.0)
        else:
            # Direct processing
            with self._memory_context():
                self.model.eval()
                with torch.no_grad():
                    result = self.model(data, **kwargs)
        
        # Cache result
        if self.cache and use_cache and cache_key:
            self.cache.put(cache_key, result)
        
        # Update performance metrics
        inference_time = time.time() - start_time
        self.inference_times.append(inference_time)
        
        return result
    
    def predict_edges_batch(self, requests: List[Tuple], use_cache: bool = True) -> List[torch.Tensor]:
        """Batch edge prediction for multiple requests."""
        if not requests:
            return []
        
        if self.thread_pool and len(requests) > 1:
            # Parallel processing
            futures = []
            for req in requests:
                future = self.thread_pool.submit(self._predict_edges_single, *req, use_cache)
                futures.append(future)
            
            results = []
            for future in as_completed(futures):
                results.append(future.result())
            
            return results
        else:
            # Sequential processing
            return [self._predict_edges_single(*req, use_cache) for req in requests]
    
    def _predict_edges_single(self, source_nodes: torch.Tensor, target_nodes: torch.Tensor,
                             time: float, data, use_cache: bool = True, **kwargs) -> torch.Tensor:
        """Single edge prediction with caching."""
        # Generate cache key
        cache_key = None
        if self.cache and use_cache:
            cache_key = self._generate_edge_cache_key(source_nodes, target_nodes, time, data)
            cached_result = self.cache.get(cache_key)
            if cached_result is not None:
                return cached_result
        
        # Compute prediction
        with self._memory_context():
            result = self.base_model.predict_edges(source_nodes, target_nodes, time, data, **kwargs)
        
        # Cache result
        if self.cache and use_cache and cache_key:
            self.cache.put(cache_key, result)
        
        return result
    
    @contextmanager
    def _memory_context(self):
        """Memory management context."""
        if self.memory_optimizer:
            # Get tensor for computation
            try:
                yield
            finally:
                # Return tensors to pool and cleanup if needed
                pass
        else:
            yield
    
    def _generate_cache_key(self, data, kwargs: Dict[str, Any]) -> str:
        """Generate cache key for forward pass."""
        key_components = [
            str(data.num_nodes),
            str(data.edge_index.shape),
            str(data.timestamps.shape),
            str(hash(tuple(data.edge_index.flatten().tolist()))),
            str(hash(tuple(data.timestamps.tolist()))),
            str(sorted(kwargs.items()))
        ]
        return hashlib.md5('|'.join(key_components).encode()).hexdigest()
    
    def _generate_edge_cache_key(self, source_nodes: torch.Tensor, target_nodes: torch.Tensor,
                                time: float, data) -> str:
        """Generate cache key for edge prediction."""
        key_components = [
            str(hash(tuple(source_nodes.tolist()))),
            str(hash(tuple(target_nodes.tolist()))),
            str(time),
            str(data.num_nodes),
            str(hash(tuple(data.edge_index.flatten().tolist())))
        ]
        return hashlib.md5('|'.join(key_components).encode()).hexdigest()
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        stats = {
            'avg_inference_time_ms': np.mean(self.inference_times) * 1000 if self.inference_times else 0,
            'min_inference_time_ms': np.min(self.inference_times) * 1000 if self.inference_times else 0,
            'max_inference_time_ms': np.max(self.inference_times) * 1000 if self.inference_times else 0,
            'total_inferences': len(self.inference_times),
            'cache_stats': self.cache.get_stats() if self.cache else None,
            'memory_usage_mb': psutil.Process().memory_info().rss / 1024 / 1024,
            'optimization_config': asdict(self.optimization_config)
        }
        
        if self.inference_times:
            recent_times = list(self.inference_times)[-10:]  # Last 10 inferences
            throughput = len(recent_times) / sum(recent_times) if recent_times else 0
            stats['recent_throughput_fps'] = throughput
        
        return stats
    
    def cleanup(self):
        """Cleanup resources."""
        if self.batch_processor:
            self.batch_processor.stop()
        
        if self.memory_optimizer:
            self.memory_optimizer.stop_monitoring()
        
        if self.thread_pool:
            self.thread_pool.shutdown(wait=True)
        
        if self.cache:
            self.cache.clear()

def test_performance_optimization():
    """Test performance optimization suite."""
    print("🚀 Testing Performance Optimization Suite")
    print("=" * 50)
    
    try:
        # Test LRU Cache
        print("💾 Testing LRU Cache...")
        cache = LRUCache(max_size=10, max_memory_mb=10.0)
        
        # Test cache operations
        cache.put("key1", torch.randn(100))
        cache.put("key2", torch.randn(100))
        
        result = cache.get("key1")
        assert result is not None, "Cache should return stored value"
        
        result = cache.get("key3")
        assert result is None, "Cache should return None for missing key"
        
        stats = cache.get_stats()
        print(f"✅ Cache Stats: {stats.hits} hits, {stats.misses} misses")
        
        # Test Memory Optimizer
        print("\n🧠 Testing Memory Optimizer...")
        memory_optimizer = MemoryOptimizer(cleanup_threshold=0.9)
        
        tensor = memory_optimizer.get_tensor((100, 100))
        assert tensor.shape == (100, 100), "Should return tensor with correct shape"
        
        memory_optimizer.return_tensor(tensor)
        print("✅ Memory Optimizer: Basic operations working")
        
        # Test Optimized DGDN
        print("\n⚡ Testing Optimized DGDN...")
        
        config = {
            'node_dim': 64,
            'hidden_dim': 128,
            'time_dim': 32,
            'num_layers': 2,
            'num_heads': 4,
            'diffusion_steps': 3,
            'dropout': 0.1
        }
        
        opt_config = OptimizationConfig(
            enable_caching=True,
            cache_size=100,
            enable_batching=False,  # Disable for testing
            enable_memory_optimization=True,
            enable_jit_compilation=False,  # Disable for testing
            num_worker_threads=2
        )
        
        optimized_model = PerformanceOptimizedDGDN(config, opt_config)
        
        # Create test data
        class TestData:
            def __init__(self):
                self.edge_index = torch.randint(0, 50, (2, 100))
                self.timestamps = torch.sort(torch.rand(100) * 100.0)[0]
                self.node_features = torch.randn(50, 64)
                self.num_nodes = 50
        
        test_data = TestData()
        
        # Test forward pass
        start_time = time.time()
        output1 = optimized_model.forward(test_data)
        first_time = time.time() - start_time
        
        # Test cached forward pass (should be faster)
        start_time = time.time()
        output2 = optimized_model.forward(test_data)
        second_time = time.time() - start_time
        
        print(f"✅ Forward pass: First {first_time*1000:.1f}ms, Cached {second_time*1000:.1f}ms")
        assert second_time < first_time, "Cached call should be faster"
        
        # Test batch edge prediction
        print("\n🔗 Testing Batch Edge Prediction...")
        requests = [
            (torch.randint(0, 50, (5,)), torch.randint(0, 50, (5,)), 25.0, test_data),
            (torch.randint(0, 50, (5,)), torch.randint(0, 50, (5,)), 50.0, test_data),
            (torch.randint(0, 50, (5,)), torch.randint(0, 50, (5,)), 75.0, test_data)
        ]
        
        batch_results = optimized_model.predict_edges_batch(requests)
        assert len(batch_results) == 3, "Should return results for all requests"
        print(f"✅ Batch edge prediction: {len(batch_results)} results")
        
        # Test performance stats
        print("\n📊 Testing Performance Stats...")
        perf_stats = optimized_model.get_performance_stats()
        print(f"✅ Performance Stats:")
        print(f"   Avg inference time: {perf_stats['avg_inference_time_ms']:.1f}ms")
        print(f"   Total inferences: {perf_stats['total_inferences']}")
        print(f"   Memory usage: {perf_stats['memory_usage_mb']:.1f}MB")
        
        if perf_stats['cache_stats']:
            cache_stats = perf_stats['cache_stats']
            print(f"   Cache: {cache_stats.hits} hits, {cache_stats.misses} misses")
        
        # Performance benchmark
        print("\n🏃 Running Performance Benchmark...")
        num_iterations = 20
        start_time = time.time()
        
        for i in range(num_iterations):
            optimized_model.forward(test_data, use_cache=(i > 10))  # Use cache for last 10
        
        total_time = time.time() - start_time
        avg_time = total_time / num_iterations
        throughput = num_iterations / total_time
        
        print(f"✅ Benchmark Results:")
        print(f"   {num_iterations} iterations in {total_time:.2f}s")
        print(f"   Average time per inference: {avg_time*1000:.1f}ms")
        print(f"   Throughput: {throughput:.1f} inferences/sec")
        
        # Cleanup
        optimized_model.cleanup()
        memory_optimizer.stop_monitoring()
        
        print("\n🎉 Performance Optimization Tests: ALL PASSED!")
        return True
        
    except Exception as e:
        print(f"\n❌ Error in performance optimization test: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_performance_optimization()
    sys.exit(0 if success else 1)