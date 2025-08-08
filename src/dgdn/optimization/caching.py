"""Caching mechanisms for DGDN performance optimization."""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional, Any, Union
from collections import OrderedDict
import logging
import hashlib
import pickle
import time


class EmbeddingCache:
    """Cache for node embeddings to avoid recomputation."""
    
    def __init__(self, max_size: int = 10000, ttl_seconds: float = 300):
        """Initialize embedding cache.
        
        Args:
            max_size: Maximum number of cached embeddings
            ttl_seconds: Time-to-live for cached entries
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache = OrderedDict()
        self.access_times = {}
        self.hit_count = 0
        self.miss_count = 0
        self.logger = logging.getLogger(__name__)
    
    def _generate_key(self, node_ids: torch.Tensor, time: float, 
                     model_state: Optional[str] = None) -> str:
        """Generate cache key for embeddings."""
        # Convert tensors to hashable format
        node_ids_str = str(sorted(node_ids.cpu().tolist()))
        time_str = f"{time:.6f}"
        
        # Include model state if provided (for cache invalidation)
        if model_state:
            key_data = f"{node_ids_str}_{time_str}_{model_state}"
        else:
            key_data = f"{node_ids_str}_{time_str}"
        
        return hashlib.sha256(key_data.encode()).hexdigest()
    
    def get(self, node_ids: torch.Tensor, time: float, 
            model_state: Optional[str] = None) -> Optional[torch.Tensor]:
        """Get cached embeddings."""
        key = self._generate_key(node_ids, time, model_state)
        
        if key in self.cache:
            # Check TTL
            if time - self.access_times[key] > self.ttl_seconds:
                self._evict_key(key)
                self.miss_count += 1
                return None
            
            # Move to end (most recently used)
            embeddings = self.cache[key]
            self.cache.move_to_end(key)
            self.access_times[key] = time.time()
            self.hit_count += 1
            return embeddings.clone()  # Return copy to avoid modifications
        
        self.miss_count += 1
        return None
    
    def put(self, node_ids: torch.Tensor, time: float, embeddings: torch.Tensor,
            model_state: Optional[str] = None):
        """Store embeddings in cache."""
        key = self._generate_key(node_ids, time, model_state)
        
        # Evict oldest entries if at capacity
        while len(self.cache) >= self.max_size:
            oldest_key, _ = self.cache.popitem(last=False)
            if oldest_key in self.access_times:
                del self.access_times[oldest_key]
        
        # Store embedding (detached from computation graph)
        self.cache[key] = embeddings.detach().clone()
        self.access_times[key] = time.time()
    
    def _evict_key(self, key: str):
        """Evict specific key from cache."""
        if key in self.cache:
            del self.cache[key]
        if key in self.access_times:
            del self.access_times[key]
    
    def clear(self):
        """Clear all cached embeddings."""
        self.cache.clear()
        self.access_times.clear()
        self.hit_count = 0
        self.miss_count = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.hit_count + self.miss_count
        hit_rate = self.hit_count / total_requests if total_requests > 0 else 0
        
        return {
            'hit_count': self.hit_count,
            'miss_count': self.miss_count,
            'hit_rate': hit_rate,
            'cache_size': len(self.cache),
            'max_size': self.max_size,
            'memory_usage_mb': self._estimate_memory_usage() / 1e6
        }
    
    def _estimate_memory_usage(self) -> int:
        """Estimate memory usage of cache in bytes."""
        total_bytes = 0
        for embedding in self.cache.values():
            total_bytes += embedding.numel() * embedding.element_size()
        return total_bytes
    
    def cleanup_expired(self):
        """Remove expired cache entries."""
        current_time = time.time()
        expired_keys = []
        
        for key, access_time in self.access_times.items():
            if current_time - access_time > self.ttl_seconds:
                expired_keys.append(key)
        
        for key in expired_keys:
            self._evict_key(key)
        
        if expired_keys:
            self.logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")


class AttentionCache:
    """Cache for attention weights to speed up inference."""
    
    def __init__(self, max_size: int = 5000):
        """Initialize attention cache.
        
        Args:
            max_size: Maximum number of cached attention patterns
        """
        self.max_size = max_size
        self.cache = OrderedDict()
        self.logger = logging.getLogger(__name__)
    
    def _generate_attention_key(self, edge_index: torch.Tensor, 
                               temporal_encoding: torch.Tensor) -> str:
        """Generate key for attention pattern."""
        # Hash edge structure and temporal pattern
        edge_hash = hashlib.sha256(edge_index.cpu().numpy().tobytes()).hexdigest()
        temporal_hash = hashlib.sha256(temporal_encoding.cpu().numpy().tobytes()).hexdigest()
        
        return f"{edge_hash}_{temporal_hash}"
    
    def get_attention(self, edge_index: torch.Tensor, 
                     temporal_encoding: torch.Tensor) -> Optional[torch.Tensor]:
        """Get cached attention weights."""
        key = self._generate_attention_key(edge_index, temporal_encoding)
        
        if key in self.cache:
            attention_weights = self.cache[key]
            self.cache.move_to_end(key)  # Mark as recently used
            return attention_weights.clone()
        
        return None
    
    def store_attention(self, edge_index: torch.Tensor, 
                       temporal_encoding: torch.Tensor, 
                       attention_weights: torch.Tensor):
        """Store attention weights in cache."""
        key = self._generate_attention_key(edge_index, temporal_encoding)
        
        # Evict oldest if at capacity
        if len(self.cache) >= self.max_size:
            self.cache.popitem(last=False)
        
        self.cache[key] = attention_weights.detach().clone()


class ComputationCache:
    """General purpose computation cache for expensive operations."""
    
    def __init__(self, max_memory_mb: float = 1024):
        """Initialize computation cache.
        
        Args:
            max_memory_mb: Maximum memory usage in MB
        """
        self.max_memory_bytes = max_memory_mb * 1e6
        self.cache = {}
        self.memory_usage = 0
        self.access_order = OrderedDict()
        self.logger = logging.getLogger(__name__)
    
    def _estimate_size(self, obj: Any) -> int:
        """Estimate object size in bytes."""
        if isinstance(obj, torch.Tensor):
            return obj.numel() * obj.element_size()
        else:
            try:
                return len(pickle.dumps(obj))
            except:
                return 1000  # Conservative estimate
    
    def _make_key(self, func_name: str, args: Tuple, kwargs: Dict) -> str:
        """Create cache key from function name and arguments."""
        # Convert tensors to hashable representation
        def tensorize_args(obj):
            if isinstance(obj, torch.Tensor):
                return f"tensor_{obj.shape}_{obj.dtype}_{torch.sum(obj).item():.6f}"
            elif isinstance(obj, (list, tuple)):
                return tuple(tensorize_args(x) for x in obj)
            elif isinstance(obj, dict):
                return tuple(sorted((k, tensorize_args(v)) for k, v in obj.items()))
            else:
                return obj
        
        hashable_args = tensorize_args(args)
        hashable_kwargs = tensorize_args(kwargs)
        
        key_data = f"{func_name}_{hashable_args}_{hashable_kwargs}"
        return hashlib.sha256(str(key_data).encode()).hexdigest()
    
    def get(self, func_name: str, args: Tuple, kwargs: Dict) -> Tuple[bool, Any]:
        """Get cached result."""
        key = self._make_key(func_name, args, kwargs)
        
        if key in self.cache:
            # Update access order
            self.access_order.move_to_end(key)
            return True, self.cache[key]
        
        return False, None
    
    def put(self, func_name: str, args: Tuple, kwargs: Dict, result: Any):
        """Store computation result."""
        key = self._make_key(func_name, args, kwargs)
        result_size = self._estimate_size(result)
        
        # Evict entries if memory limit would be exceeded
        while (self.memory_usage + result_size > self.max_memory_bytes and 
               len(self.cache) > 0):
            self._evict_oldest()
        
        # Store result
        self.cache[key] = result
        self.access_order[key] = True
        self.memory_usage += result_size
    
    def _evict_oldest(self):
        """Evict oldest cache entry."""
        if self.access_order:
            oldest_key, _ = self.access_order.popitem(last=False)
            if oldest_key in self.cache:
                evicted_size = self._estimate_size(self.cache[oldest_key])
                del self.cache[oldest_key]
                self.memory_usage -= evicted_size


def cached_computation(cache: ComputationCache, func_name: str):
    """Decorator for caching expensive computations."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Check cache
            hit, result = cache.get(func_name, args, kwargs)
            if hit:
                return result
            
            # Compute and cache result
            result = func(*args, **kwargs)
            cache.put(func_name, args, kwargs, result)
            
            return result
        return wrapper
    return decorator


class CacheManager:
    """Unified cache management for DGDN."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize cache manager with configuration."""
        config = config or {}
        
        # Initialize caches
        self.embedding_cache = EmbeddingCache(
            max_size=config.get('embedding_cache_size', 10000),
            ttl_seconds=config.get('embedding_ttl', 300)
        )
        
        self.attention_cache = AttentionCache(
            max_size=config.get('attention_cache_size', 5000)
        )
        
        self.computation_cache = ComputationCache(
            max_memory_mb=config.get('computation_cache_mb', 1024)
        )
        
        self.logger = logging.getLogger(__name__)
        
        # Performance tracking
        self.enabled = config.get('enabled', True)
        self.stats = {
            'cache_hits': 0,
            'cache_misses': 0,
            'time_saved_ms': 0.0
        }
    
    def clear_all(self):
        """Clear all caches."""
        self.embedding_cache.clear()
        self.attention_cache.cache.clear()
        self.computation_cache.cache.clear()
        self.computation_cache.access_order.clear()
        self.computation_cache.memory_usage = 0
        
        self.logger.info("Cleared all caches")
    
    def get_global_stats(self) -> Dict[str, Any]:
        """Get statistics for all caches."""
        return {
            'embedding_cache': self.embedding_cache.get_stats(),
            'attention_cache': {
                'size': len(self.attention_cache.cache),
                'max_size': self.attention_cache.max_size
            },
            'computation_cache': {
                'size': len(self.computation_cache.cache),
                'memory_usage_mb': self.computation_cache.memory_usage / 1e6,
                'max_memory_mb': self.computation_cache.max_memory_bytes / 1e6
            },
            'global_stats': self.stats
        }
    
    def optimize_cache_sizes(self, memory_budget_mb: float):
        """Optimize cache sizes based on memory budget."""
        # Allocate memory budget across caches
        embedding_ratio = 0.6
        attention_ratio = 0.2
        computation_ratio = 0.2
        
        embedding_mb = memory_budget_mb * embedding_ratio
        attention_mb = memory_budget_mb * attention_ratio
        computation_mb = memory_budget_mb * computation_ratio
        
        # Estimate sizes and adjust cache limits
        # (This is a simplified heuristic - could be more sophisticated)
        avg_embedding_size_mb = 0.01  # Rough estimate
        new_embedding_size = int(embedding_mb / avg_embedding_size_mb)
        
        self.embedding_cache.max_size = min(new_embedding_size, 50000)
        self.computation_cache.max_memory_bytes = computation_mb * 1e6
        
        self.logger.info(f"Optimized cache sizes for {memory_budget_mb}MB budget")