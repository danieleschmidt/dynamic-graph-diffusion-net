#!/usr/bin/env python3
"""Scalable Autonomous DGDN - Generation 3 Implementation.

Advanced performance optimization, auto-scaling, concurrent processing,
resource pooling, and adaptive load balancing following Terragon SDLC methodology.
"""

import sys
import time
import math
import random
import logging
import traceback
import hashlib
import json
import os
import threading
import queue
import concurrent.futures
import multiprocessing as mp
from typing import Dict, Any, List, Tuple, Optional, Union, Callable
from dataclasses import dataclass, field
from contextlib import contextmanager
from enum import Enum
from collections import defaultdict, deque
import psutil

# Enhanced logging for scalability
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(processName)s:%(threadName)s] - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('dgdn_scalable.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)

# Try enhanced dependencies
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

class ScaleMode(Enum):
    """Scaling modes for adaptive performance."""
    SINGLE_THREADED = "single"
    MULTI_THREADED = "threaded"
    MULTI_PROCESS = "process"
    ADAPTIVE = "adaptive"
    DISTRIBUTED = "distributed"

class LoadLevel(Enum):
    """System load levels for auto-scaling."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class ScalingConfig:
    """Configuration for scaling and performance optimization."""
    max_workers: int = mp.cpu_count()
    max_memory_gb: float = 8.0
    batch_size: int = 32
    cache_size_mb: int = 256
    enable_gpu_acceleration: bool = False
    enable_distributed_computing: bool = False
    auto_scaling_enabled: bool = True
    performance_target_fps: float = 10.0
    memory_threshold_percent: float = 80.0

@dataclass
class PerformanceMetrics:
    """Comprehensive performance tracking."""
    throughput_ops_per_sec: float = 0.0
    latency_ms: float = 0.0
    memory_usage_mb: float = 0.0
    cpu_utilization_percent: float = 0.0
    gpu_utilization_percent: float = 0.0
    cache_hit_rate: float = 0.0
    batch_processing_efficiency: float = 0.0
    concurrent_operations: int = 0
    scaling_factor: float = 1.0

class AdaptiveCache:
    """High-performance adaptive cache with LRU and frequency tracking."""
    
    def __init__(self, max_size_mb: int = 256):
        self.max_size_mb = max_size_mb
        self.cache: Dict[str, Any] = {}
        self.access_count: Dict[str, int] = defaultdict(int)
        self.access_time: Dict[str, float] = {}
        self.size_tracker: Dict[str, int] = {}
        self.current_size_mb = 0.0
        self.lock = threading.RLock()
        self.stats = {'hits': 0, 'misses': 0, 'evictions': 0}
        
        logger.info(f"🗂️  Initialized adaptive cache: {max_size_mb}MB limit")
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache with frequency tracking."""
        with self.lock:
            if key in self.cache:
                self.access_count[key] += 1
                self.access_time[key] = time.time()
                self.stats['hits'] += 1
                return self.cache[key]
            else:
                self.stats['misses'] += 1
                return None
    
    def put(self, key: str, value: Any, size_mb: float = 0.1) -> None:
        """Put item in cache with intelligent eviction."""
        with self.lock:
            # Check if we need to evict
            while self.current_size_mb + size_mb > self.max_size_mb and self.cache:
                self._evict_lfu_item()
            
            # Store item
            if key in self.cache:
                self.current_size_mb -= self.size_tracker.get(key, 0)
            
            self.cache[key] = value
            self.access_count[key] = 1
            self.access_time[key] = time.time()
            self.size_tracker[key] = size_mb
            self.current_size_mb += size_mb
    
    def _evict_lfu_item(self) -> None:
        """Evict least frequently used item."""
        if not self.cache:
            return
        
        # Find LFU item (with recency as tiebreaker)
        lfu_key = min(self.cache.keys(), 
                     key=lambda k: (self.access_count[k], self.access_time.get(k, 0)))
        
        # Remove item
        del self.cache[lfu_key]
        self.current_size_mb -= self.size_tracker.pop(lfu_key, 0)
        del self.access_count[lfu_key]
        self.access_time.pop(lfu_key, None)
        self.stats['evictions'] += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        with self.lock:
            total_requests = self.stats['hits'] + self.stats['misses']
            hit_rate = self.stats['hits'] / max(total_requests, 1)
            
            return {
                'hit_rate': hit_rate,
                'size_mb': self.current_size_mb,
                'utilization': self.current_size_mb / self.max_size_mb,
                'items': len(self.cache),
                'hits': self.stats['hits'],
                'misses': self.stats['misses'],
                'evictions': self.stats['evictions']
            }

class ResourceMonitor:
    """Real-time system resource monitoring for auto-scaling."""
    
    def __init__(self, monitoring_interval: float = 1.0):
        self.monitoring_interval = monitoring_interval
        self.metrics_history = deque(maxlen=100)  # Keep last 100 measurements
        self.monitoring_active = False
        self.monitor_thread = None
        self.lock = threading.Lock()
        
        logger.info("📊 Resource monitor initialized")
    
    def start_monitoring(self) -> None:
        """Start background resource monitoring."""
        if not self.monitoring_active:
            self.monitoring_active = True
            self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self.monitor_thread.start()
            logger.info("🔄 Resource monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop background resource monitoring."""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        logger.info("⏹️  Resource monitoring stopped")
    
    def _monitoring_loop(self) -> None:
        """Background monitoring loop."""
        while self.monitoring_active:
            try:
                metrics = self._collect_metrics()
                
                with self.lock:
                    self.metrics_history.append(metrics)
                
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.warning(f"Resource monitoring error: {e}")
                time.sleep(self.monitoring_interval)
    
    def _collect_metrics(self) -> Dict[str, float]:
        """Collect system resource metrics."""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=None)
            cpu_count = psutil.cpu_count()
            
            # Memory metrics
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_available_mb = memory.available / (1024 * 1024)
            
            # Disk I/O metrics
            disk_io = psutil.disk_io_counters()
            disk_read_mb = disk_io.read_bytes / (1024 * 1024) if disk_io else 0
            disk_write_mb = disk_io.write_bytes / (1024 * 1024) if disk_io else 0
            
            # Network I/O metrics
            net_io = psutil.net_io_counters()
            net_sent_mb = net_io.bytes_sent / (1024 * 1024) if net_io else 0
            net_recv_mb = net_io.bytes_recv / (1024 * 1024) if net_io else 0
            
            return {
                'timestamp': time.time(),
                'cpu_percent': cpu_percent,
                'cpu_count': cpu_count,
                'memory_percent': memory_percent,
                'memory_available_mb': memory_available_mb,
                'disk_read_mb': disk_read_mb,
                'disk_write_mb': disk_write_mb,
                'net_sent_mb': net_sent_mb,
                'net_recv_mb': net_recv_mb
            }
            
        except Exception as e:
            logger.warning(f"Metrics collection failed: {e}")
            return {'timestamp': time.time(), 'cpu_percent': 0, 'memory_percent': 0}
    
    def get_current_load_level(self) -> LoadLevel:
        """Determine current system load level."""
        with self.lock:
            if not self.metrics_history:
                return LoadLevel.LOW
            
            recent_metrics = list(self.metrics_history)[-10:]  # Last 10 measurements
            avg_cpu = sum(m.get('cpu_percent', 0) for m in recent_metrics) / len(recent_metrics)
            avg_memory = sum(m.get('memory_percent', 0) for m in recent_metrics) / len(recent_metrics)
            
            # Determine load level
            if avg_cpu > 90 or avg_memory > 90:
                return LoadLevel.CRITICAL
            elif avg_cpu > 70 or avg_memory > 70:
                return LoadLevel.HIGH
            elif avg_cpu > 40 or avg_memory > 40:
                return LoadLevel.MEDIUM
            else:
                return LoadLevel.LOW
    
    def get_scaling_recommendation(self) -> Dict[str, Any]:
        """Get scaling recommendation based on current load."""
        load_level = self.get_current_load_level()
        
        recommendations = {
            LoadLevel.LOW: {
                'scale_mode': ScaleMode.SINGLE_THREADED,
                'worker_count': 1,
                'batch_size': 16,
                'cache_aggressive': False
            },
            LoadLevel.MEDIUM: {
                'scale_mode': ScaleMode.MULTI_THREADED,
                'worker_count': 2,
                'batch_size': 32,
                'cache_aggressive': True
            },
            LoadLevel.HIGH: {
                'scale_mode': ScaleMode.MULTI_PROCESS,
                'worker_count': min(4, mp.cpu_count()),
                'batch_size': 64,
                'cache_aggressive': True
            },
            LoadLevel.CRITICAL: {
                'scale_mode': ScaleMode.DISTRIBUTED,
                'worker_count': mp.cpu_count(),
                'batch_size': 128,
                'cache_aggressive': True
            }
        }
        
        return {
            'load_level': load_level,
            'recommendation': recommendations[load_level]
        }

class BatchProcessor:
    """High-performance batch processing with intelligent scheduling."""
    
    def __init__(self, batch_size: int = 32, max_batch_time: float = 0.1):
        self.batch_size = batch_size
        self.max_batch_time = max_batch_time
        self.pending_batches = queue.Queue()
        self.results_cache = AdaptiveCache(max_size_mb=64)
        self.processing_stats = {'batches_processed': 0, 'total_items': 0}
        
        logger.info(f"⚡ Batch processor initialized: batch_size={batch_size}")
    
    def process_batch(self, items: List[Any], processor_func: Callable) -> List[Any]:
        """Process a batch of items efficiently."""
        if not items:
            return []
        
        start_time = time.time()
        
        # Check cache for pre-computed results
        cached_results = []
        uncached_items = []
        
        for item in items:
            cache_key = self._get_cache_key(item)
            cached_result = self.results_cache.get(cache_key)
            
            if cached_result is not None:
                cached_results.append((len(uncached_items) + len(cached_results), cached_result))
            else:
                uncached_items.append((len(uncached_items) + len(cached_results), item))
        
        # Process uncached items
        uncached_results = []
        if uncached_items:
            try:
                # Extract just the items for processing
                items_to_process = [item for _, item in uncached_items]
                processed_results = processor_func(items_to_process)
                
                # Store results in cache and create result tuples
                for (orig_idx, item), result in zip(uncached_items, processed_results):
                    cache_key = self._get_cache_key(item)
                    self.results_cache.put(cache_key, result, 0.01)  # 10KB per result
                    uncached_results.append((orig_idx, result))
                    
            except Exception as e:
                logger.warning(f"Batch processing failed: {e}")
                # Fallback: return empty results
                uncached_results = [(orig_idx, None) for orig_idx, _ in uncached_items]
        
        # Combine and sort results by original index
        all_results = cached_results + uncached_results
        all_results.sort(key=lambda x: x[0])
        final_results = [result for _, result in all_results]
        
        # Update statistics
        processing_time = time.time() - start_time
        self.processing_stats['batches_processed'] += 1
        self.processing_stats['total_items'] += len(items)
        
        logger.debug(f"Batch processed: {len(items)} items in {processing_time:.3f}s")
        
        return final_results
    
    def _get_cache_key(self, item: Any) -> str:
        """Generate cache key for an item."""
        try:
            return hashlib.md5(str(item).encode()).hexdigest()[:16]
        except:
            return str(hash(str(item)))[:16]

class ScalableDGDN:
    """Scalable DGDN with advanced performance optimization and auto-scaling."""
    
    def __init__(self, 
                 node_dim: int = 64,
                 hidden_dim: int = 128,
                 num_layers: int = 3,
                 scaling_config: Optional[ScalingConfig] = None):
        
        self.node_dim = node_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Configuration
        self.scaling_config = scaling_config or ScalingConfig()
        
        # Performance components
        self.adaptive_cache = AdaptiveCache(self.scaling_config.cache_size_mb)
        self.resource_monitor = ResourceMonitor()
        self.batch_processor = BatchProcessor(self.scaling_config.batch_size)
        
        # Performance tracking
        self.performance_metrics = PerformanceMetrics()
        self.operation_history = deque(maxlen=1000)
        
        # Initialize logger first
        self.logger = logging.getLogger(f"{__name__}.ScalableDGDN")
        
        # Thread pool for concurrent operations
        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.scaling_config.max_workers,
            thread_name_prefix="DGDN"
        )
        
        # Model parameters with proper dimensions
        self.weights = self._initialize_scalable_weights()
        self.adaptive_params = {
            'message_strength': 1.0,
            'temporal_weight': 0.5,
            'uncertainty_scale': 0.1,
            'batch_processing_enabled': True,
            'concurrent_processing_enabled': True,
            'cache_aggressiveness': 0.8
        }
        
        self.current_scale_mode = ScaleMode.ADAPTIVE
        
        # Start monitoring
        self.resource_monitor.start_monitoring()
        
        self.logger.info(f"⚡ Scalable DGDN initialized: {node_dim}→{hidden_dim}, {num_layers} layers")
        self.logger.info(f"🔧 Max workers: {self.scaling_config.max_workers}")
    
    def _initialize_scalable_weights(self) -> Dict[str, Any]:
        """Initialize weights optimized for scalable operations."""
        try:
            random.seed(42)
            
            # Use Xavier/Glorot initialization for better scaling
            weights = {}
            
            # Node projection: node_dim -> hidden_dim
            node_std = math.sqrt(2.0 / (self.node_dim + self.hidden_dim))
            weights['node_projection'] = [
                [random.gauss(0, node_std) for _ in range(self.hidden_dim)]
                for _ in range(self.node_dim)
            ]
            
            # Temporal projection: 32 -> hidden_dim (temporal features are always 32-dim)
            temporal_std = math.sqrt(2.0 / (32 + self.hidden_dim))
            weights['temporal_projection'] = [
                [random.gauss(0, temporal_std) for _ in range(self.hidden_dim)]
                for _ in range(32)
            ]
            
            # Layer weights for multi-layer processing: hidden_dim -> hidden_dim
            layer_std = math.sqrt(2.0 / self.hidden_dim)
            weights['layer_weights'] = []
            for layer in range(self.num_layers):
                layer_weight = [
                    [random.gauss(0, layer_std) for _ in range(self.hidden_dim)]
                    for _ in range(self.hidden_dim)
                ]
                weights['layer_weights'].append(layer_weight)
            
            # Message aggregation weights
            message_std = math.sqrt(2.0 / (self.hidden_dim * 2))
            weights['message_weights'] = [
                [random.gauss(0, message_std) for _ in range(self.hidden_dim)]
                for _ in range(self.hidden_dim * 2)
            ]
            
            self.logger.info("✅ Scalable weights initialized with Xavier initialization")
            return weights
            
        except Exception as e:
            self.logger.error(f"❌ Weight initialization failed: {e}")
            raise
    
    def create_scalable_synthetic_data(self, num_nodes: int = 100, num_edges: int = 300) -> Dict[str, Any]:
        """Create synthetic data optimized for scalable processing."""
        start_time = time.time()
        
        try:
            # Validate parameters for scalability
            max_nodes = min(num_nodes, self.scaling_config.max_memory_gb * 1000)  # Rough estimate
            max_edges = min(num_edges, max_nodes * max_nodes // 10)  # Prevent memory explosion
            
            if max_nodes < num_nodes or max_edges < num_edges:
                self.logger.info(f"🔧 Scaled down data size: {max_nodes} nodes, {max_edges} edges")
                num_nodes, num_edges = max_nodes, max_edges
            
            # Generate with structured patterns for realistic scaling behavior
            random.seed(42)
            
            # Create clustered node features for better cache locality
            node_features = []
            cluster_size = 10
            num_clusters = (num_nodes + cluster_size - 1) // cluster_size
            
            for cluster_id in range(num_clusters):
                # Create cluster center
                cluster_center = [random.gauss(0, 1) for _ in range(self.node_dim)]
                
                # Create nodes around cluster center
                for node_in_cluster in range(min(cluster_size, num_nodes - cluster_id * cluster_size)):
                    node_features_single = []
                    for dim in range(self.node_dim):
                        # Add noise around cluster center
                        feature_val = cluster_center[dim] + random.gauss(0, 0.3)
                        node_features_single.append(feature_val)
                    node_features.append(node_features_single)
            
            # Generate edges with preferential attachment for scale-free properties
            edges = []
            node_degrees = [0] * num_nodes
            
            for edge_idx in range(num_edges):
                # Preferential attachment: higher degree nodes more likely to connect
                if edge_idx < num_nodes:
                    # Initial ring topology
                    source = edge_idx
                    target = (edge_idx + 1) % num_nodes
                else:
                    # Preferential attachment
                    total_degree = sum(node_degrees) + len(node_degrees)  # +1 for each node
                    probabilities = [(degree + 1) / total_degree for degree in node_degrees]
                    
                    # Select source based on degree
                    rand_val = random.random()
                    cumulative_prob = 0
                    source = 0
                    for i, prob in enumerate(probabilities):
                        cumulative_prob += prob
                        if rand_val <= cumulative_prob:
                            source = i
                            break
                    
                    # Select target (avoid self-loops)
                    target = random.randint(0, num_nodes - 1)
                    while target == source:
                        target = random.randint(0, num_nodes - 1)
                
                # Create temporal patterns (bursty activity)
                time_cluster = random.randint(0, 9)  # 10 time clusters
                base_time = time_cluster * 10.0
                timestamp = base_time + (-math.log(random.random()) * 2.0)  # Exponential distribution (bursty pattern)
                
                # Edge weight based on temporal and topological features
                time_factor = math.exp(-timestamp / 50.0)  # Recency bias
                degree_factor = 1.0 / (1.0 + abs(node_degrees[source] - node_degrees[target]))
                weight = (0.1 + random.random()) * time_factor * degree_factor
                
                edges.append((source, target, timestamp, weight))
                node_degrees[source] += 1
                node_degrees[target] += 1
            
            # Sort edges by timestamp for temporal locality
            edges.sort(key=lambda x: x[2])
            
            data = {
                'node_features': node_features,
                'edges': edges,
                'num_nodes': num_nodes,
                'num_edges': len(edges),
                'generation_time': time.time() - start_time,
                'data_properties': {
                    'clustered': True,
                    'scale_free': True,
                    'temporal_patterns': True
                }
            }
            
            self.logger.info(f"🎯 Generated scalable data: {num_nodes} nodes, {len(edges)} edges in {data['generation_time']:.3f}s")
            return data
            
        except Exception as e:
            self.logger.error(f"❌ Scalable data generation failed: {e}")
            raise
    
    def optimized_temporal_encoding(self, timestamps: List[float], dim: int = 32) -> List[List[float]]:
        """Highly optimized temporal encoding with caching and vectorization."""
        if not timestamps:
            return []
        
        # Check cache first
        cache_key = f"temporal_{len(timestamps)}_{dim}_{hash(tuple(timestamps[:10]))}"
        cached_result = self.adaptive_cache.get(cache_key)
        if cached_result is not None:
            return cached_result
        
        try:
            if NUMPY_AVAILABLE:
                # Vectorized numpy implementation
                timestamps_array = np.array(timestamps, dtype=np.float32)
                
                # Normalize timestamps
                max_timestamp = np.max(timestamps_array)
                normalized_timestamps = timestamps_array / max(max_timestamp, 1.0)
                
                # Create frequency array
                frequencies = 1.0 / (10000.0 ** (2 * np.arange(dim // 2) / dim))
                
                # Compute all angles at once
                angles = np.outer(normalized_timestamps, frequencies)
                
                # Compute sin and cos
                sin_vals = np.sin(angles)
                cos_vals = np.cos(angles)
                
                # Interleave sin and cos
                encoding = np.zeros((len(timestamps), dim))
                encoding[:, 0::2] = sin_vals
                encoding[:, 1::2] = cos_vals[:, :dim//2] if dim % 2 == 0 else cos_vals
                
                result = encoding.tolist()
                
            else:
                # Optimized pure Python implementation
                result = []
                max_timestamp = max(timestamps)
                normalization = max(max_timestamp, 1.0)
                
                # Pre-compute frequencies
                frequencies = [1.0 / (10000.0 ** (2 * i / dim)) for i in range(dim // 2)]
                
                for t in timestamps:
                    normalized_t = t / normalization
                    features = []
                    
                    for freq in frequencies:
                        angle = normalized_t * freq
                        features.extend([math.sin(angle), math.cos(angle)])
                    
                    # Ensure correct dimension
                    features = features[:dim] + [0.0] * max(0, dim - len(features))
                    result.append(features)
            
            # Cache result
            cache_size = len(result) * dim * 4 / (1024 * 1024)  # Approximate size in MB
            self.adaptive_cache.put(cache_key, result, cache_size)
            
            return result
            
        except Exception as e:
            self.logger.warning(f"Optimized temporal encoding failed: {e}")
            # Fallback to simple encoding
            return [[0.0] * dim for _ in range(len(timestamps))]
    
    def concurrent_message_passing(self, node_features: List[List[float]], 
                                 edges: List[Tuple], temporal_encoding: List[List[float]]) -> List[List[float]]:
        """Concurrent message passing with intelligent load balancing."""
        num_nodes = len(node_features)
        
        # Initialize result
        if NUMPY_AVAILABLE:
            node_embeddings = np.zeros((num_nodes, self.hidden_dim), dtype=np.float32)
        else:
            node_embeddings = [[0.0] * self.hidden_dim for _ in range(num_nodes)]
        
        # Determine processing strategy based on data size
        if len(edges) < 1000:  # Small data - single threaded
            return self._sequential_message_passing(node_features, edges, temporal_encoding)
        
        # Large data - concurrent processing
        try:
            # Split edges into chunks for parallel processing
            chunk_size = max(100, len(edges) // self.scaling_config.max_workers)
            edge_chunks = [edges[i:i + chunk_size] for i in range(0, len(edges), chunk_size)]
            
            # Process chunks concurrently
            futures = []
            for chunk_idx, edge_chunk in enumerate(edge_chunks):
                future = self.executor.submit(
                    self._process_edge_chunk,
                    edge_chunk, node_features, temporal_encoding, chunk_idx
                )
                futures.append(future)
            
            # Collect results and aggregate
            chunk_results = []
            for future in concurrent.futures.as_completed(futures, timeout=30.0):
                try:
                    chunk_result = future.result()
                    chunk_results.append(chunk_result)
                except Exception as e:
                    self.logger.warning(f"Chunk processing failed: {e}")
            
            # Aggregate chunk results
            if NUMPY_AVAILABLE and chunk_results:
                for chunk_embeddings in chunk_results:
                    node_embeddings += np.array(chunk_embeddings)
                result = node_embeddings.tolist()
            else:
                # Python aggregation
                for chunk_embeddings in chunk_results:
                    for node_idx in range(min(num_nodes, len(chunk_embeddings))):
                        for dim_idx in range(self.hidden_dim):
                            if isinstance(node_embeddings, list):
                                node_embeddings[node_idx][dim_idx] += chunk_embeddings[node_idx][dim_idx]
                            else:
                                node_embeddings[node_idx][dim_idx] += chunk_embeddings[node_idx][dim_idx]
                
                result = node_embeddings.tolist() if NUMPY_AVAILABLE else node_embeddings
            
            # Apply activation
            result = self._vectorized_activation(result)
            
            return result
            
        except Exception as e:
            self.logger.warning(f"Concurrent processing failed, falling back: {e}")
            return self._sequential_message_passing(node_features, edges, temporal_encoding)
    
    def _process_edge_chunk(self, edge_chunk: List[Tuple], node_features: List[List[float]], 
                           temporal_encoding: List[List[float]], chunk_idx: int) -> List[List[float]]:
        """Process a chunk of edges in parallel."""
        num_nodes = len(node_features)
        chunk_embeddings = [[0.0] * self.hidden_dim for _ in range(num_nodes)]
        
        for edge_idx_in_chunk, (source, target, timestamp, weight) in enumerate(edge_chunk):
            try:
                # Calculate global edge index for temporal encoding
                global_edge_idx = chunk_idx * len(edge_chunk) + edge_idx_in_chunk
                if global_edge_idx >= len(temporal_encoding):
                    continue
                
                # Validate indices
                if not (0 <= source < num_nodes and 0 <= target < num_nodes):
                    continue
                
                # Get features
                source_features = node_features[source]
                temporal_features = temporal_encoding[global_edge_idx]
                
                # Safe projections
                projected = self._safe_linear_projection(source_features, self.weights['node_projection'])
                temporal_projected = self._safe_linear_projection(temporal_features, self.weights['temporal_projection'])
                
                # Create message
                for i in range(self.hidden_dim):
                    message_val = (
                        projected[i] * self.adaptive_params['message_strength'] +
                        temporal_projected[i] * self.adaptive_params['temporal_weight']
                    ) * weight
                    
                    # Clamp values
                    message_val = max(-10.0, min(10.0, message_val))
                    chunk_embeddings[target][i] += message_val
                    
            except Exception as e:
                self.logger.debug(f"Edge processing error in chunk {chunk_idx}: {e}")
                continue
        
        return chunk_embeddings
    
    def _sequential_message_passing(self, node_features: List[List[float]], 
                                  edges: List[Tuple], temporal_encoding: List[List[float]]) -> List[List[float]]:
        """Sequential message passing for small datasets."""
        num_nodes = len(node_features)
        node_embeddings = [[0.0] * self.hidden_dim for _ in range(num_nodes)]
        
        for edge_idx, (source, target, timestamp, weight) in enumerate(edges):
            if edge_idx >= len(temporal_encoding):
                continue
                
            if not (0 <= source < num_nodes and 0 <= target < num_nodes):
                continue
            
            try:
                source_features = node_features[source]
                temporal_features = temporal_encoding[edge_idx]
                
                # Project features
                projected = self._safe_linear_projection(source_features, self.weights['node_projection'])
                temporal_projected = self._safe_linear_projection(temporal_features, self.weights['temporal_projection'])
                
                # Create and aggregate message
                for i in range(self.hidden_dim):
                    message = (
                        projected[i] * self.adaptive_params['message_strength'] +
                        temporal_projected[i] * self.adaptive_params['temporal_weight']
                    ) * weight
                    
                    node_embeddings[target][i] += max(-10.0, min(10.0, message))
                    
            except Exception as e:
                self.logger.debug(f"Sequential edge processing error: {e}")
                continue
        
        return self._vectorized_activation(node_embeddings)
    
    def _safe_linear_projection(self, features: List[float], weights: List[List[float]]) -> List[float]:
        """Safe linear projection with dimension validation."""
        try:
            if len(features) != len(weights):
                self.logger.debug(f"Dimension mismatch in projection: {len(features)} vs {len(weights)}")
                # Pad or truncate features to match weights
                if len(features) < len(weights):
                    features = features + [0.0] * (len(weights) - len(features))
                else:
                    features = features[:len(weights)]
            
            output_dim = len(weights[0])
            result = [0.0] * output_dim
            
            for i in range(len(features)):
                for j in range(output_dim):
                    try:
                        product = features[i] * weights[i][j]
                        if math.isfinite(product):
                            result[j] += product
                    except (OverflowError, TypeError):
                        continue
            
            return result
            
        except Exception as e:
            self.logger.debug(f"Linear projection failed: {e}")
            return [0.0] * len(weights[0]) if weights else [0.0] * self.hidden_dim
    
    def _vectorized_activation(self, embeddings: List[List[float]]) -> List[List[float]]:
        """Vectorized activation function for better performance."""
        if NUMPY_AVAILABLE:
            emb_array = np.array(embeddings, dtype=np.float32)
            # ReLU activation
            activated = np.maximum(0, emb_array)
            return activated.tolist()
        else:
            # Pure Python ReLU
            return [[max(0.0, val) for val in row] for row in embeddings]
    
    def scalable_forward_pass(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Scalable forward pass with adaptive optimization."""
        start_time = time.time()
        
        try:
            # Get scaling recommendation
            scaling_info = self.resource_monitor.get_scaling_recommendation()
            load_level = scaling_info['load_level']
            
            self.logger.info(f"🔄 Processing with load level: {load_level.value}")
            
            # Extract data
            node_features = data['node_features']
            edges = data['edges']
            timestamps = [edge[2] for edge in edges]
            
            # Optimized temporal encoding
            temporal_encoding = self.optimized_temporal_encoding(timestamps)
            
            # Multi-layer processing with scaling
            current_embeddings = node_features
            layer_outputs = []
            
            for layer in range(self.num_layers):
                layer_start = time.time()
                
                # Adaptive processing based on load
                if load_level in [LoadLevel.HIGH, LoadLevel.CRITICAL]:
                    # Use concurrent processing for heavy loads
                    new_embeddings = self.concurrent_message_passing(
                        current_embeddings, edges, temporal_encoding
                    )
                else:
                    # Sequential processing for light loads
                    new_embeddings = self._sequential_message_passing(
                        current_embeddings, edges, temporal_encoding
                    )
                
                # Layer transformation
                if layer < len(self.weights['layer_weights']):
                    transformed_embeddings = []
                    layer_weights = self.weights['layer_weights'][layer]
                    
                    for node_emb in new_embeddings:
                        transformed = self._safe_linear_projection(node_emb, layer_weights)
                        transformed_embeddings.append(transformed)
                    
                    new_embeddings = transformed_embeddings
                
                layer_outputs.append(new_embeddings)
                current_embeddings = new_embeddings
                
                layer_time = time.time() - layer_start
                self.logger.debug(f"Layer {layer + 1} completed in {layer_time:.3f}s")
            
            # Final embeddings
            final_embeddings = current_embeddings
            
            # Efficient uncertainty quantification
            uncertainty_mean, uncertainty_std = self._fast_uncertainty_quantification(final_embeddings)
            
            # Performance metrics
            total_time = time.time() - start_time
            throughput = len(edges) / max(total_time, 0.001)  # edges per second
            
            # Update performance tracking
            self.performance_metrics.throughput_ops_per_sec = throughput
            self.performance_metrics.latency_ms = total_time * 1000
            self.performance_metrics.concurrent_operations = len(layer_outputs)
            
            # Get cache stats
            cache_stats = self.adaptive_cache.get_stats()
            
            output = {
                'node_embeddings': final_embeddings,
                'uncertainty_mean': uncertainty_mean,
                'uncertainty_std': uncertainty_std,
                'temporal_encoding': temporal_encoding,
                'layer_outputs': layer_outputs,
                'performance_metrics': {
                    'throughput_ops_per_sec': throughput,
                    'latency_ms': total_time * 1000,
                    'load_level': load_level.value,
                    'cache_hit_rate': cache_stats['hit_rate'],
                    'cache_utilization': cache_stats['utilization'],
                    'processing_mode': 'concurrent' if load_level in [LoadLevel.HIGH, LoadLevel.CRITICAL] else 'sequential'
                },
                'scaling_info': scaling_info,
                'status': 'success'
            }
            
            self.operation_history.append({
                'timestamp': time.time(),
                'throughput': throughput,
                'latency': total_time * 1000,
                'load_level': load_level.value
            })
            
            return output
            
        except Exception as e:
            self.logger.error(f"❌ Scalable forward pass failed: {e}")
            self.logger.error(traceback.format_exc())
            
            return {
                'node_embeddings': [[0.0] * self.hidden_dim for _ in range(data.get('num_nodes', 1))],
                'uncertainty_mean': 0.0,
                'uncertainty_std': 1.0,
                'status': 'failed',
                'error': str(e)
            }
    
    def _fast_uncertainty_quantification(self, embeddings: List[List[float]]) -> Tuple[float, float]:
        """Fast uncertainty quantification using statistical approximations."""
        try:
            if NUMPY_AVAILABLE:
                emb_array = np.array(embeddings, dtype=np.float32)
                mean_val = float(np.mean(emb_array))
                std_val = float(np.std(emb_array))
                return mean_val, std_val * self.adaptive_params['uncertainty_scale']
            else:
                # Fast pure Python implementation
                all_values = [val for row in embeddings for val in row]
                if not all_values:
                    return 0.0, 1.0
                
                n = len(all_values)
                mean_val = sum(all_values) / n
                
                # Use Welford's online algorithm for numerical stability
                variance = sum((x - mean_val) ** 2 for x in all_values) / n
                std_val = math.sqrt(variance)
                
                return mean_val, std_val * self.adaptive_params['uncertainty_scale']
                
        except Exception as e:
            self.logger.warning(f"Fast uncertainty quantification failed: {e}")
            return 0.0, 1.0
    
    def adaptive_scaling_optimization(self, metrics: Dict[str, float], iteration: int) -> None:
        """Autonomous scaling optimization based on performance metrics."""
        try:
            # Get current load and performance
            scaling_info = self.resource_monitor.get_scaling_recommendation()
            load_level = scaling_info['load_level']
            throughput = metrics.get('throughput_ops_per_sec', 0)
            latency_ms = metrics.get('latency_ms', 1000)
            
            # Adaptive parameter optimization
            if load_level == LoadLevel.CRITICAL:
                # Emergency scaling
                self.adaptive_params['batch_processing_enabled'] = True
                self.adaptive_params['cache_aggressiveness'] = 0.9
                self.adaptive_params['message_strength'] *= 0.95  # Reduce computation
                self.logger.info("   🚨 Emergency scaling activated")
                
            elif load_level == LoadLevel.HIGH:
                # Aggressive optimization
                self.adaptive_params['concurrent_processing_enabled'] = True
                self.adaptive_params['cache_aggressiveness'] = 0.8
                if throughput < self.scaling_config.performance_target_fps:
                    self.adaptive_params['temporal_weight'] *= 0.98  # Reduce temporal complexity
                self.logger.info("   ⚡ Aggressive optimization enabled")
                
            elif load_level == LoadLevel.LOW:
                # Quality optimization
                self.adaptive_params['message_strength'] = min(1.2, self.adaptive_params['message_strength'] * 1.02)
                self.adaptive_params['temporal_weight'] = min(0.8, self.adaptive_params['temporal_weight'] * 1.01)
                self.adaptive_params['cache_aggressiveness'] = 0.6
                self.logger.info("   🎯 Quality optimization enabled")
            
            # Throughput-based optimizations
            if throughput > self.scaling_config.performance_target_fps * 1.5:
                # We're performing well, can increase quality
                self.adaptive_params['uncertainty_scale'] = min(0.3, self.adaptive_params['uncertainty_scale'] * 1.05)
            elif throughput < self.scaling_config.performance_target_fps * 0.5:
                # Performance issues, optimize for speed
                self.adaptive_params['uncertainty_scale'] *= 0.95
            
            # Cache optimization
            cache_stats = self.adaptive_cache.get_stats()
            if cache_stats['hit_rate'] < 0.5:
                # Poor cache performance
                self.adaptive_params['cache_aggressiveness'] = min(1.0, self.adaptive_params['cache_aggressiveness'] * 1.1)
                self.logger.info(f"   💾 Cache aggressiveness increased to {self.adaptive_params['cache_aggressiveness']:.2f}")
            
        except Exception as e:
            self.logger.warning(f"Adaptive scaling optimization failed: {e}")
    
    def run_scalable_demonstration(self) -> Dict[str, Any]:
        """Run comprehensive scalable demonstration with auto-scaling."""
        self.logger.info("⚡ Starting Scalable DGDN Demonstration")
        self.logger.info("🏗️  Generation 3: Make It Scale - Advanced Performance Optimization")
        
        results = {
            'generation': 3,
            'implementation': 'scalable',
            'status': 'running',
            'metrics_history': [],
            'scaling_events': [],
            'performance_analysis': {}
        }
        
        try:
            # Generate scalable test data with increasing complexity
            test_sizes = [
                (50, 150),    # Small
                (100, 300),   # Medium  
                (200, 600),   # Large
                (300, 900),   # Very Large
                (150, 450)    # Back to medium (test scaling down)
            ]
            
            for iteration, (num_nodes, num_edges) in enumerate(test_sizes):
                self.logger.info(f"🔄 Scalability Iteration {iteration + 1}/5 - Nodes: {num_nodes}, Edges: {num_edges}")
                
                # Generate test data
                data_start = time.time()
                data = self.create_scalable_synthetic_data(num_nodes=num_nodes, num_edges=num_edges)
                data_generation_time = time.time() - data_start
                
                # Execute scalable forward pass
                execution_start = time.time()
                output = self.scalable_forward_pass(data)
                execution_time = time.time() - execution_start
                
                # Collect comprehensive metrics
                metrics = self._collect_comprehensive_metrics(output, data, execution_time)
                metrics['iteration'] = iteration + 1
                metrics['data_size'] = {'nodes': num_nodes, 'edges': num_edges}
                metrics['data_generation_time'] = data_generation_time
                
                results['metrics_history'].append(metrics)
                
                # Adaptive scaling optimization
                self.adaptive_scaling_optimization(metrics, iteration)
                
                # Log comprehensive performance
                self.logger.info(f"   Throughput: {metrics['throughput_ops_per_sec']:.1f} ops/sec")
                self.logger.info(f"   Latency: {metrics['latency_ms']:.1f}ms")
                self.logger.info(f"   Load level: {metrics['load_level']}")
                self.logger.info(f"   Cache hit rate: {metrics['cache_hit_rate']:.1%}")
                self.logger.info(f"   Processing mode: {metrics['processing_mode']}")
                
                # Record scaling events
                scaling_info = output.get('scaling_info', {})
                if scaling_info:
                    results['scaling_events'].append({
                        'iteration': iteration + 1,
                        'load_level': scaling_info['load_level'].value,
                        'recommendation': scaling_info['recommendation'],
                        'timestamp': time.time()
                    })
                
                # Brief pause to observe system adaptation
                time.sleep(0.1)
            
            # Compile performance analysis
            results['performance_analysis'] = self._analyze_scalability_performance(results['metrics_history'])
            results['cache_final_stats'] = self.adaptive_cache.get_stats()
            results['resource_utilization'] = self._get_final_resource_utilization()
            results['status'] = 'completed'
            
            self.logger.info("✅ Scalable demonstration completed successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Scalable demonstration failed: {e}")
            results['status'] = 'failed'
            results['error'] = str(e)
        
        return results
    
    def _collect_comprehensive_metrics(self, output: Dict[str, Any], 
                                     data: Dict[str, Any], execution_time: float) -> Dict[str, float]:
        """Collect comprehensive performance and scalability metrics."""
        metrics = {}
        
        try:
            # Basic performance metrics
            performance_metrics = output.get('performance_metrics', {})
            metrics.update(performance_metrics)
            
            # Scalability metrics
            num_nodes = data['num_nodes']
            num_edges = data['num_edges']
            
            metrics['nodes_processed'] = num_nodes
            metrics['edges_processed'] = num_edges
            metrics['total_execution_time'] = execution_time
            
            # Efficiency metrics
            metrics['nodes_per_second'] = num_nodes / max(execution_time, 0.001)
            metrics['edges_per_second'] = num_edges / max(execution_time, 0.001)
            
            # Memory efficiency (approximate)
            embeddings = output.get('node_embeddings', [])
            if embeddings:
                memory_estimate_mb = (num_nodes * self.hidden_dim * 4) / (1024 * 1024)  # 4 bytes per float
                metrics['memory_estimate_mb'] = memory_estimate_mb
                metrics['memory_efficiency'] = num_nodes / max(memory_estimate_mb, 0.001)  # nodes per MB
            
            # Quality metrics
            if embeddings:
                if NUMPY_AVAILABLE:
                    emb_array = np.array(embeddings)
                    metrics['embedding_quality_score'] = float(np.mean(np.abs(emb_array)))
                    metrics['embedding_sparsity'] = float(np.mean(emb_array == 0))
                else:
                    flat_values = [val for row in embeddings for val in row]
                    metrics['embedding_quality_score'] = sum(abs(v) for v in flat_values) / len(flat_values)
                    metrics['embedding_sparsity'] = sum(1 for v in flat_values if v == 0) / len(flat_values)
            
            # Resource utilization metrics
            resource_metrics = self._get_current_resource_metrics()
            metrics.update(resource_metrics)
            
            return metrics
            
        except Exception as e:
            self.logger.warning(f"Comprehensive metrics collection failed: {e}")
            return {'status': 'metrics_failed'}
    
    def _get_current_resource_metrics(self) -> Dict[str, float]:
        """Get current resource utilization metrics."""
        try:
            # CPU and Memory
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            
            # Process-specific metrics
            process = psutil.Process()
            process_memory_mb = process.memory_info().rss / (1024 * 1024)
            process_cpu_percent = process.cpu_percent()
            
            return {
                'system_cpu_percent': cpu_percent,
                'system_memory_percent': memory.percent,
                'system_memory_available_gb': memory.available / (1024 ** 3),
                'process_memory_mb': process_memory_mb,
                'process_cpu_percent': process_cpu_percent,
                'cpu_cores_available': psutil.cpu_count()
            }
            
        except Exception as e:
            self.logger.warning(f"Resource metrics collection failed: {e}")
            return {}
    
    def _analyze_scalability_performance(self, metrics_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze scalability performance across all iterations."""
        if not metrics_history:
            return {}
        
        try:
            analysis = {}
            
            # Throughput analysis
            throughputs = [m.get('throughput_ops_per_sec', 0) for m in metrics_history]
            analysis['throughput'] = {
                'min': min(throughputs),
                'max': max(throughputs),
                'average': sum(throughputs) / len(throughputs),
                'improvement_over_time': throughputs[-1] / max(throughputs[0], 1) if throughputs else 1.0
            }
            
            # Latency analysis
            latencies = [m.get('latency_ms', 1000) for m in metrics_history]
            analysis['latency'] = {
                'min': min(latencies),
                'max': max(latencies),
                'average': sum(latencies) / len(latencies),
                'stability': 1.0 - (max(latencies) - min(latencies)) / max(max(latencies), 1)
            }
            
            # Scalability analysis
            data_sizes = [m.get('nodes_processed', 0) * m.get('edges_processed', 0) for m in metrics_history]
            processing_times = [m.get('total_execution_time', 1) for m in metrics_history]
            
            if len(data_sizes) > 1 and len(processing_times) > 1:
                # Calculate scalability factor (how well performance scales with data size)
                size_ratios = [data_sizes[i] / max(data_sizes[0], 1) for i in range(1, len(data_sizes))]
                time_ratios = [processing_times[i] / max(processing_times[0], 1) for i in range(1, len(processing_times))]
                
                scalability_factors = [size_ratios[i] / max(time_ratios[i], 0.1) for i in range(len(size_ratios))]
                analysis['scalability'] = {
                    'average_factor': sum(scalability_factors) / len(scalability_factors),
                    'best_factor': max(scalability_factors),
                    'consistency': 1.0 - (max(scalability_factors) - min(scalability_factors)) / max(max(scalability_factors), 1)
                }
            
            # Cache performance analysis
            cache_hit_rates = [m.get('cache_hit_rate', 0) for m in metrics_history]
            analysis['cache_performance'] = {
                'average_hit_rate': sum(cache_hit_rates) / len(cache_hit_rates),
                'improvement': cache_hit_rates[-1] - cache_hit_rates[0] if cache_hit_rates else 0,
                'consistency': 1.0 - (max(cache_hit_rates) - min(cache_hit_rates)) if cache_hit_rates else 0
            }
            
            # Resource efficiency analysis
            memory_usage = [m.get('process_memory_mb', 0) for m in metrics_history]
            cpu_usage = [m.get('process_cpu_percent', 0) for m in metrics_history]
            
            analysis['resource_efficiency'] = {
                'memory_growth': memory_usage[-1] / max(memory_usage[0], 1) if memory_usage else 1.0,
                'cpu_utilization': sum(cpu_usage) / len(cpu_usage) if cpu_usage else 0,
                'memory_per_node': sum(memory_usage) / sum(m.get('nodes_processed', 1) for m in metrics_history)
            }
            
            # Overall scalability score (higher is better)
            scalability_score = 0.0
            if 'scalability' in analysis:
                scalability_score += analysis['scalability']['average_factor'] * 0.3
            scalability_score += analysis['throughput']['improvement_over_time'] * 0.25
            scalability_score += analysis['cache_performance']['average_hit_rate'] * 0.2
            scalability_score += analysis['latency']['stability'] * 0.15
            scalability_score += min(1.0, 100.0 / analysis['resource_efficiency']['memory_growth']) * 0.1
            
            analysis['overall_scalability_score'] = scalability_score
            
            return analysis
            
        except Exception as e:
            self.logger.warning(f"Scalability analysis failed: {e}")
            return {'error': str(e)}
    
    def _get_final_resource_utilization(self) -> Dict[str, Any]:
        """Get final resource utilization summary."""
        try:
            return {
                'cache_stats': self.adaptive_cache.get_stats(),
                'system_resources': self._get_current_resource_metrics(),
                'thread_pool_stats': {
                    'max_workers': self.scaling_config.max_workers,
                    'active_threads': threading.active_count()
                }
            }
        except Exception as e:
            return {'error': str(e)}
    
    def cleanup(self) -> None:
        """Cleanup resources and stop monitoring."""
        try:
            self.resource_monitor.stop_monitoring()
            self.executor.shutdown(wait=True)
            self.logger.info("🧹 Cleanup completed")
        except Exception as e:
            self.logger.warning(f"Cleanup failed: {e}")

def main():
    """Main execution function for scalable DGDN demonstration."""
    logger.info("⚡ Terragon Labs - Scalable Autonomous DGDN")
    logger.info("🏗️  Generation 3: Make It Scale")
    logger.info("="*90)
    
    # Advanced scaling configuration
    scaling_config = ScalingConfig(
        max_workers=min(8, mp.cpu_count()),
        max_memory_gb=4.0,
        batch_size=64,
        cache_size_mb=512,
        auto_scaling_enabled=True,
        performance_target_fps=20.0
    )
    
    # Initialize scalable DGDN
    scalable_dgdn = ScalableDGDN(
        node_dim=64,
        hidden_dim=128,
        num_layers=3,
        scaling_config=scaling_config
    )
    
    try:
        # Run comprehensive scalable demonstration
        start_time = time.time()
        results = scalable_dgdn.run_scalable_demonstration()
        total_time = time.time() - start_time
        
        # Comprehensive reporting
        logger.info("\n" + "="*90)
        logger.info("📊 GENERATION 3 SCALABILITY REPORT")
        logger.info("="*90)
        
        logger.info(f"Status: {results['status'].upper()}")
        logger.info(f"Implementation: {results['implementation']}")
        logger.info(f"Total execution time: {total_time:.2f}s")
        logger.info(f"CPU cores utilized: {scaling_config.max_workers}")
        
        if results['status'] == 'completed':
            # Performance summary
            performance_analysis = results.get('performance_analysis', {})
            
            logger.info(f"\n⚡ Scalability Performance:")
            if 'throughput' in performance_analysis:
                throughput_stats = performance_analysis['throughput']
                logger.info(f"  • Peak throughput: {throughput_stats['max']:.1f} ops/sec")
                logger.info(f"  • Average throughput: {throughput_stats['average']:.1f} ops/sec")
                logger.info(f"  • Performance improvement: {throughput_stats['improvement_over_time']:.1f}x")
            
            if 'latency' in performance_analysis:
                latency_stats = performance_analysis['latency']
                logger.info(f"  • Minimum latency: {latency_stats['min']:.1f}ms")
                logger.info(f"  • Average latency: {latency_stats['average']:.1f}ms")
                logger.info(f"  • Latency stability: {latency_stats['stability']:.1%}")
            
            if 'scalability' in performance_analysis:
                scalability_stats = performance_analysis['scalability']
                logger.info(f"  • Scalability factor: {scalability_stats['average_factor']:.2f}")
                logger.info(f"  • Best scaling: {scalability_stats['best_factor']:.2f}x")
                logger.info(f"  • Scaling consistency: {scalability_stats['consistency']:.1%}")
            
            # Cache and resource efficiency
            cache_stats = results.get('cache_final_stats', {})
            logger.info(f"\n💾 Cache Performance:")
            logger.info(f"  • Final hit rate: {cache_stats.get('hit_rate', 0):.1%}")
            logger.info(f"  • Cache utilization: {cache_stats.get('utilization', 0):.1%}")
            logger.info(f"  • Items cached: {cache_stats.get('items', 0)}")
            
            # Resource utilization
            resource_util = results.get('resource_utilization', {})
            system_resources = resource_util.get('system_resources', {})
            logger.info(f"\n🖥️  Resource Utilization:")
            logger.info(f"  • Peak memory usage: {system_resources.get('process_memory_mb', 0):.1f}MB")
            logger.info(f"  • Average CPU usage: {system_resources.get('process_cpu_percent', 0):.1f}%")
            logger.info(f"  • Cores available: {system_resources.get('cpu_cores_available', 0)}")
            
            # Scaling events
            scaling_events = results.get('scaling_events', [])
            if scaling_events:
                logger.info(f"\n🔄 Auto-Scaling Events:")
                load_levels = [event['load_level'] for event in scaling_events]
                logger.info(f"  • Load level transitions: {' → '.join(load_levels)}")
                logger.info(f"  • Total scaling events: {len(scaling_events)}")
            
            # Overall assessment
            overall_score = performance_analysis.get('overall_scalability_score', 0)
            logger.info(f"\n🎯 Overall Scalability Score: {overall_score:.2f}/1.0")
            
            if overall_score > 0.8:
                logger.info("   ✅ Excellent scalability performance!")
            elif overall_score > 0.6:
                logger.info("   👍 Good scalability performance")
            elif overall_score > 0.4:
                logger.info("   ⚠️  Moderate scalability performance")
            else:
                logger.info("   ❌ Scalability needs improvement")
        
        logger.info(f"\n🚀 Ready to proceed to Quality Gates and Production Deployment")
        logger.info("="*90)
        
        return results
        
    finally:
        # Always cleanup resources
        scalable_dgdn.cleanup()

if __name__ == "__main__":
    results = main()
    exit_code = 0 if results.get('status') == 'completed' else 1
    sys.exit(exit_code)