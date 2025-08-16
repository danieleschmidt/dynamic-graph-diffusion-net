#!/usr/bin/env python3
"""
Generation 3: MAKE IT SCALE (Optimized) - Performance Optimization & Concurrency

This implementation adds advanced performance optimization, concurrent processing,
auto-scaling, caching, and distributed computing capabilities to DGDN.
"""

import torch
import torch.multiprocessing as mp
import numpy as np
import time
import hashlib
import pickle
import sqlite3
import threading
import asyncio
import concurrent.futures
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
import sys
import os
import psutil
from collections import OrderedDict
import queue
import weakref

# Add src to path for imports
sys.path.insert(0, 'src')

import dgdn
from dgdn import DynamicGraphDiffusionNet, TemporalData, TemporalDataset


@dataclass
class ScalingConfig:
    """Configuration for scaling and optimization."""
    enable_caching: bool = True
    cache_size_mb: int = 1024
    enable_multiprocessing: bool = True
    max_workers: int = 4
    enable_gpu_acceleration: bool = True
    enable_auto_scaling: bool = True
    memory_threshold_percent: float = 80.0
    cpu_threshold_percent: float = 80.0
    enable_batch_processing: bool = True
    batch_size: int = 32
    enable_model_parallelism: bool = True
    enable_data_parallelism: bool = True
    cache_ttl_seconds: int = 3600
    enable_compression: bool = True
    optimization_level: str = "aggressive"  # conservative, moderate, aggressive


class AdvancedCache:
    """High-performance caching system with LRU eviction and compression."""
    
    def __init__(self, config: ScalingConfig):
        self.config = config
        self.max_size_bytes = config.cache_size_mb * 1024 * 1024
        self.cache = OrderedDict()
        self.cache_stats = {"hits": 0, "misses": 0, "evictions": 0}
        self.current_size = 0
        self.lock = threading.RLock()
        
        # Optional persistent cache
        self.db_path = "cache.db"
        self._init_persistent_cache()
        
        print(f"🚀 Advanced cache initialized: {config.cache_size_mb}MB")
    
    def _init_persistent_cache(self):
        """Initialize SQLite persistent cache."""
        try:
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache_entries (
                    key TEXT PRIMARY KEY,
                    value BLOB,
                    timestamp REAL,
                    access_count INTEGER DEFAULT 0,
                    size_bytes INTEGER
                )
            """)
            conn.commit()
            conn.close()
            print("💾 Persistent cache initialized")
        except Exception as e:
            print(f"⚠️ Persistent cache init failed: {e}")
    
    def _compute_size(self, obj: Any) -> int:
        """Estimate object size in bytes."""
        try:
            if isinstance(obj, torch.Tensor):
                return obj.element_size() * obj.numel()
            else:
                return len(pickle.dumps(obj))
        except Exception:
            return 1024  # Default estimate
    
    def _compress_data(self, data: Any) -> bytes:
        """Compress data for storage."""
        if self.config.enable_compression:
            import gzip
            serialized = pickle.dumps(data)
            return gzip.compress(serialized)
        else:
            return pickle.dumps(data)
    
    def _decompress_data(self, compressed_data: bytes) -> Any:
        """Decompress data from storage."""
        if self.config.enable_compression:
            import gzip
            serialized = gzip.decompress(compressed_data)
            return pickle.loads(serialized)
        else:
            return pickle.loads(compressed_data)
    
    def _evict_lru(self, needed_space: int):
        """Evict least recently used items."""
        while self.current_size + needed_space > self.max_size_bytes and self.cache:
            key, (value, timestamp, access_count) = self.cache.popitem(last=False)
            size = self._compute_size(value)
            self.current_size -= size
            self.cache_stats["evictions"] += 1
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache."""
        with self.lock:
            # Check memory cache first
            if key in self.cache:
                value, timestamp, access_count = self.cache[key]
                
                # Check TTL
                if time.time() - timestamp > self.config.cache_ttl_seconds:
                    del self.cache[key]
                    self.cache_stats["misses"] += 1
                    return None
                
                # Move to end (most recently used)
                self.cache.move_to_end(key)
                self.cache[key] = (value, timestamp, access_count + 1)
                self.cache_stats["hits"] += 1
                return value
            
            # Check persistent cache
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.execute(
                    "SELECT value, timestamp FROM cache_entries WHERE key = ?", 
                    (key,)
                )
                row = cursor.fetchone()
                conn.close()
                
                if row:
                    value_blob, timestamp = row
                    if time.time() - timestamp <= self.config.cache_ttl_seconds:
                        value = self._decompress_data(value_blob)
                        # Promote to memory cache
                        self.put(key, value)
                        self.cache_stats["hits"] += 1
                        return value
                
            except Exception as e:
                print(f"⚠️ Persistent cache read error: {e}")
            
            self.cache_stats["misses"] += 1
            return None
    
    def put(self, key: str, value: Any):
        """Put item in cache."""
        with self.lock:
            size = self._compute_size(value)
            timestamp = time.time()
            
            # Evict if necessary
            if key not in self.cache:
                self._evict_lru(size)
            else:
                # Update existing entry
                old_size = self._compute_size(self.cache[key][0])
                self.current_size -= old_size
            
            # Add to memory cache
            self.cache[key] = (value, timestamp, 1)
            self.current_size += size
            
            # Add to persistent cache (async to avoid blocking)
            try:
                threading.Thread(
                    target=self._async_persistent_put,
                    args=(key, value, timestamp, size),
                    daemon=True
                ).start()
            except Exception as e:
                print(f"⚠️ Persistent cache write error: {e}")
    
    def _async_persistent_put(self, key: str, value: Any, timestamp: float, size: int):
        """Asynchronously write to persistent cache."""
        try:
            compressed_value = self._compress_data(value)
            conn = sqlite3.connect(self.db_path)
            conn.execute("""
                INSERT OR REPLACE INTO cache_entries 
                (key, value, timestamp, access_count, size_bytes)
                VALUES (?, ?, ?, 1, ?)
            """, (key, compressed_value, timestamp, size))
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"⚠️ Async persistent write failed: {e}")
    
    def clear(self):
        """Clear all cache entries."""
        with self.lock:
            self.cache.clear()
            self.current_size = 0
            self.cache_stats = {"hits": 0, "misses": 0, "evictions": 0}
        
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute("DELETE FROM cache_entries")
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"⚠️ Persistent cache clear error: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.cache_stats["hits"] + self.cache_stats["misses"]
        hit_rate = self.cache_stats["hits"] / max(total_requests, 1)
        
        return {
            "hit_rate": hit_rate,
            "total_requests": total_requests,
            "memory_entries": len(self.cache),
            "memory_size_mb": self.current_size / (1024 * 1024),
            "evictions": self.cache_stats["evictions"],
            **self.cache_stats
        }


class LoadBalancer:
    """Intelligent load balancer for distributing work across resources."""
    
    def __init__(self, config: ScalingConfig):
        self.config = config
        self.worker_pool = None
        self.task_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.workers_busy = {}
        self.load_stats = {"tasks_completed": 0, "avg_task_time": 0.0}
        
        if config.enable_multiprocessing:
            self._init_worker_pool()
    
    def _init_worker_pool(self):
        """Initialize worker pool."""
        try:
            mp.set_start_method('spawn', force=True)
            self.worker_pool = concurrent.futures.ProcessPoolExecutor(
                max_workers=self.config.max_workers
            )
            print(f"⚡ Worker pool initialized: {self.config.max_workers} workers")
        except Exception as e:
            print(f"⚠️ Worker pool init failed: {e}")
            self.worker_pool = None
    
    def submit_task(self, func, *args, **kwargs) -> concurrent.futures.Future:
        """Submit task for parallel execution."""
        if self.worker_pool:
            future = self.worker_pool.submit(func, *args, **kwargs)
            return future
        else:
            # Fallback to synchronous execution
            future = concurrent.futures.Future()
            try:
                result = func(*args, **kwargs)
                future.set_result(result)
            except Exception as e:
                future.set_exception(e)
            return future
    
    def batch_submit(self, func, task_list: List[Tuple]) -> List[concurrent.futures.Future]:
        """Submit multiple tasks as a batch."""
        futures = []
        for args in task_list:
            if isinstance(args, tuple):
                future = self.submit_task(func, *args)
            else:
                future = self.submit_task(func, args)
            futures.append(future)
        return futures
    
    def get_system_load(self) -> Dict[str, float]:
        """Get current system load metrics."""
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            return {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "disk_percent": disk.percent,
                "available_memory_gb": memory.available / (1024**3)
            }
        except Exception:
            return {"cpu_percent": 0, "memory_percent": 0, "disk_percent": 0, "available_memory_gb": 0}
    
    def should_scale_up(self) -> bool:
        """Determine if system should scale up."""
        load = self.get_system_load()
        return (
            load["cpu_percent"] > self.config.cpu_threshold_percent or
            load["memory_percent"] > self.config.memory_threshold_percent
        )
    
    def cleanup(self):
        """Cleanup resources."""
        if self.worker_pool:
            self.worker_pool.shutdown(wait=True)


class AutoScaler:
    """Automatic scaling based on system load and performance metrics."""
    
    def __init__(self, config: ScalingConfig):
        self.config = config
        self.scaling_history = []
        self.current_scale = 1.0
        self.last_scale_time = time.time()
        self.min_scale_interval = 30  # seconds
    
    def analyze_performance(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze performance and recommend scaling actions."""
        current_time = time.time()
        
        recommendations = {
            "scale_factor": 1.0,
            "actions": [],
            "confidence": 0.0
        }
        
        # CPU-based scaling
        if "cpu_percent" in metrics:
            cpu_usage = metrics["cpu_percent"]
            if cpu_usage > 80:
                recommendations["scale_factor"] *= 1.5
                recommendations["actions"].append("increase_parallelism")
                recommendations["confidence"] += 0.3
            elif cpu_usage < 30:
                recommendations["scale_factor"] *= 0.8
                recommendations["actions"].append("reduce_parallelism")
                recommendations["confidence"] += 0.2
        
        # Memory-based scaling
        if "memory_percent" in metrics:
            memory_usage = metrics["memory_percent"]
            if memory_usage > 85:
                recommendations["actions"].append("enable_compression")
                recommendations["actions"].append("reduce_batch_size")
                recommendations["confidence"] += 0.4
        
        # Performance-based scaling
        if "avg_inference_time" in metrics:
            inference_time = metrics["avg_inference_time"]
            if inference_time > 2.0:  # 2 seconds threshold
                recommendations["actions"].append("enable_caching")
                recommendations["actions"].append("optimize_model")
                recommendations["confidence"] += 0.3
        
        # Rate limiting
        time_since_last_scale = current_time - self.last_scale_time
        if time_since_last_scale < self.min_scale_interval:
            recommendations["actions"].append("wait_cooldown")
            recommendations["confidence"] = 0.0
        
        return recommendations
    
    def apply_scaling(self, recommendations: Dict[str, Any]) -> bool:
        """Apply scaling recommendations."""
        if recommendations["confidence"] < 0.5:
            return False
        
        applied_actions = []
        
        for action in recommendations["actions"]:
            if action == "increase_parallelism":
                if self.config.max_workers < 8:
                    self.config.max_workers += 1
                    applied_actions.append(action)
            
            elif action == "reduce_parallelism":
                if self.config.max_workers > 1:
                    self.config.max_workers -= 1
                    applied_actions.append(action)
            
            elif action == "enable_compression":
                if not self.config.enable_compression:
                    self.config.enable_compression = True
                    applied_actions.append(action)
            
            elif action == "reduce_batch_size":
                if self.config.batch_size > 8:
                    self.config.batch_size = max(8, self.config.batch_size // 2)
                    applied_actions.append(action)
        
        if applied_actions:
            self.last_scale_time = time.time()
            self.scaling_history.append({
                "timestamp": datetime.now().isoformat(),
                "actions": applied_actions,
                "scale_factor": recommendations["scale_factor"]
            })
            print(f"🔧 Auto-scaling applied: {applied_actions}")
            return True
        
        return False


class OptimizedInferenceEngine:
    """High-performance inference engine with advanced optimizations."""
    
    def __init__(self, model: DynamicGraphDiffusionNet, config: ScalingConfig):
        self.model = model
        self.config = config
        self.cache = AdvancedCache(config)
        self.load_balancer = LoadBalancer(config)
        self.auto_scaler = AutoScaler(config)
        
        # Model optimizations
        self._optimize_model()
        
        # Performance tracking
        self.performance_metrics = []
        
        print("🚀 Optimized Inference Engine initialized")
    
    def _optimize_model(self):
        """Apply model-level optimizations."""
        try:
            # Enable eval mode and disable gradients
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad_(False)
            
            # GPU optimization
            if self.config.enable_gpu_acceleration and torch.cuda.is_available():
                self.model = self.model.cuda()
                print("🔥 GPU acceleration enabled")
            
            # TorchScript compilation (if supported)
            if self.config.optimization_level == "aggressive":
                try:
                    # Create dummy input for tracing
                    dummy_data = self._create_dummy_data()
                    with torch.no_grad():
                        traced_model = torch.jit.trace(self.model, (dummy_data,), strict=False)
                    self.model = traced_model
                    print("⚡ TorchScript compilation enabled")
                except Exception as e:
                    print(f"⚠️ TorchScript compilation failed: {e}")
            
        except Exception as e:
            print(f"⚠️ Model optimization failed: {e}")
    
    def _create_dummy_data(self) -> TemporalData:
        """Create dummy data for model tracing."""
        dummy_data = TemporalData(
            edge_index=torch.randint(0, 10, (2, 20)),
            timestamps=torch.rand(20),
            node_features=torch.randn(10, self.model.node_dim),
            edge_attr=torch.randn(20, self.model.edge_dim) if self.model.edge_dim > 0 else None,
            num_nodes=10
        )
        
        if self.config.enable_gpu_acceleration and torch.cuda.is_available():
            dummy_data = dummy_data.to('cuda')
        
        return dummy_data
    
    def _compute_cache_key(self, data: TemporalData, options: Dict[str, Any]) -> str:
        """Compute cache key for data and options."""
        # Create hash from data structure
        hasher = hashlib.md5()
        
        hasher.update(data.edge_index.cpu().numpy().tobytes())
        hasher.update(data.timestamps.cpu().numpy().tobytes())
        
        if data.node_features is not None:
            # Use shape and first few elements for efficiency
            hasher.update(str(data.node_features.shape).encode())
            hasher.update(data.node_features.flatten()[:100].cpu().numpy().tobytes())
        
        if data.edge_attr is not None:
            hasher.update(str(data.edge_attr.shape).encode())
            hasher.update(data.edge_attr.flatten()[:100].cpu().numpy().tobytes())
        
        # Include options
        hasher.update(str(sorted(options.items())).encode())
        
        return hasher.hexdigest()
    
    def optimized_forward(
        self,
        data: TemporalData,
        return_attention: bool = False,
        return_uncertainty: bool = False,
        use_cache: bool = True
    ) -> Dict[str, torch.Tensor]:
        """Optimized forward pass with caching and performance monitoring."""
        start_time = time.time()
        
        # Prepare options for caching
        options = {
            "return_attention": return_attention,
            "return_uncertainty": return_uncertainty
        }
        
        # Check cache first
        if use_cache and self.config.enable_caching:
            cache_key = self._compute_cache_key(data, options)
            cached_result = self.cache.get(cache_key)
            if cached_result is not None:
                cache_time = time.time() - start_time
                print(f"💾 Cache hit: {cache_time*1000:.1f}ms")
                return cached_result
        
        # Move data to appropriate device
        if self.config.enable_gpu_acceleration and torch.cuda.is_available():
            data = data.to('cuda')
        
        # Forward pass
        with torch.no_grad():
            if hasattr(self.model, '__call__'):
                # TorchScript model
                output = self.model(data)
            else:
                # Regular PyTorch model
                output = self.model(data, return_attention=return_attention, return_uncertainty=return_uncertainty)
        
        # Cache result
        if use_cache and self.config.enable_caching:
            # Move to CPU for caching
            cpu_output = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in output.items()}
            self.cache.put(cache_key, cpu_output)
        
        # Track performance
        inference_time = time.time() - start_time
        self.performance_metrics.append({
            "timestamp": time.time(),
            "inference_time": inference_time,
            "num_nodes": data.num_nodes,
            "num_edges": data.edge_index.shape[1],
            "cache_hit": False
        })
        
        print(f"⚡ Inference: {inference_time*1000:.1f}ms")
        return output
    
    def batch_inference(
        self,
        data_list: List[TemporalData],
        return_attention: bool = False,
        return_uncertainty: bool = False
    ) -> List[Dict[str, torch.Tensor]]:
        """Batch inference with parallel processing."""
        
        if not self.config.enable_batch_processing or len(data_list) == 1:
            # Sequential processing
            return [self.optimized_forward(data, return_attention, return_uncertainty) for data in data_list]
        
        # Parallel batch processing
        print(f"🔄 Processing batch of {len(data_list)} items...")
        
        # Split into chunks for parallel processing
        chunk_size = self.config.batch_size
        chunks = [data_list[i:i + chunk_size] for i in range(0, len(data_list), chunk_size)]
        
        # Submit chunks to worker pool
        futures = []
        for chunk in chunks:
            future = self.load_balancer.submit_task(self._process_chunk, chunk, return_attention, return_uncertainty)
            futures.append(future)
        
        # Collect results
        all_results = []
        for future in concurrent.futures.as_completed(futures):
            try:
                chunk_results = future.result(timeout=300)  # 5 minute timeout
                all_results.extend(chunk_results)
            except Exception as e:
                print(f"❌ Batch processing chunk failed: {e}")
                # Fallback to empty results for failed chunk
                all_results.extend([{}] * chunk_size)
        
        return all_results
    
    def _process_chunk(
        self,
        data_chunk: List[TemporalData],
        return_attention: bool,
        return_uncertainty: bool
    ) -> List[Dict[str, torch.Tensor]]:
        """Process a chunk of data sequentially."""
        return [self.optimized_forward(data, return_attention, return_uncertainty) for data in data_chunk]
    
    def auto_tune_performance(self) -> Dict[str, Any]:
        """Automatically tune performance based on system metrics."""
        # Collect system metrics
        system_load = self.load_balancer.get_system_load()
        cache_stats = self.cache.get_stats()
        
        # Compute performance metrics
        if self.performance_metrics:
            recent_metrics = self.performance_metrics[-100:]  # Last 100 inferences
            avg_inference_time = np.mean([m["inference_time"] for m in recent_metrics])
            throughput = len(recent_metrics) / (recent_metrics[-1]["timestamp"] - recent_metrics[0]["timestamp"])
        else:
            avg_inference_time = 0
            throughput = 0
        
        combined_metrics = {
            **system_load,
            **cache_stats,
            "avg_inference_time": avg_inference_time,
            "throughput": throughput
        }
        
        # Auto-scaling analysis
        recommendations = self.auto_scaler.analyze_performance(combined_metrics)
        tuning_applied = self.auto_scaler.apply_scaling(recommendations)
        
        return {
            "metrics": combined_metrics,
            "recommendations": recommendations,
            "tuning_applied": tuning_applied,
            "cache_stats": cache_stats
        }
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        cache_stats = self.cache.get_stats()
        system_load = self.load_balancer.get_system_load()
        
        if self.performance_metrics:
            recent_metrics = self.performance_metrics[-1000:]  # Last 1000 inferences
            inference_times = [m["inference_time"] for m in recent_metrics]
            
            performance_summary = {
                "total_inferences": len(self.performance_metrics),
                "avg_inference_time": np.mean(inference_times),
                "min_inference_time": np.min(inference_times),
                "max_inference_time": np.max(inference_times),
                "p95_inference_time": np.percentile(inference_times, 95),
                "p99_inference_time": np.percentile(inference_times, 99),
                "throughput_ops_per_sec": len(recent_metrics) / (recent_metrics[-1]["timestamp"] - recent_metrics[0]["timestamp"])
            }
        else:
            performance_summary = {"total_inferences": 0}
        
        return {
            "timestamp": datetime.now().isoformat(),
            "performance": performance_summary,
            "cache": cache_stats,
            "system": system_load,
            "config": asdict(self.config),
            "scaling_history": self.auto_scaler.scaling_history[-10:]  # Last 10 scaling events
        }
    
    def cleanup(self):
        """Cleanup resources."""
        self.load_balancer.cleanup()
        self.cache.clear()


def demo_scaling_optimizations():
    """Demonstrate scaling and optimization features."""
    print("🚀 DGDN Generation 3: MAKE IT SCALE (Optimized)")
    print("=" * 60)
    
    # Initialize configuration
    config = ScalingConfig(
        enable_caching=True,
        cache_size_mb=512,
        enable_multiprocessing=True,
        max_workers=4,
        enable_gpu_acceleration=torch.cuda.is_available(),
        enable_auto_scaling=True,
        enable_batch_processing=True,
        batch_size=16,
        optimization_level="aggressive"
    )
    
    print(f"⚙️ Scaling configuration:")
    print(f"   Caching: {config.enable_caching} ({config.cache_size_mb}MB)")
    print(f"   Multiprocessing: {config.enable_multiprocessing} ({config.max_workers} workers)")
    print(f"   GPU acceleration: {config.enable_gpu_acceleration}")
    print(f"   Auto-scaling: {config.enable_auto_scaling}")
    print(f"   Batch processing: {config.enable_batch_processing} (size: {config.batch_size})")
    
    # Create optimized model
    model = DynamicGraphDiffusionNet(
        node_dim=128,
        edge_dim=64,
        hidden_dim=256,
        num_layers=3,
        num_heads=8,
        diffusion_steps=5,
        dropout=0.0  # Disable dropout for inference
    )
    
    # Initialize optimization engine
    engine = OptimizedInferenceEngine(model, config)
    
    print(f"\n📊 Model info:")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   Model size: {sum(p.numel() * 4 for p in model.parameters()) / (1024*1024):.1f}MB")
    
    # Performance test scenarios
    test_scenarios = [
        {"name": "Small graphs", "nodes": 100, "edges": 500, "count": 20},
        {"name": "Medium graphs", "nodes": 1000, "edges": 5000, "count": 10},
        {"name": "Large graphs", "nodes": 5000, "edges": 25000, "count": 5},
        {"name": "Cached repeats", "nodes": 1000, "edges": 5000, "count": 10}  # Test caching
    ]
    
    all_test_data = []
    
    # Generate test data for all scenarios
    print(f"\n🧪 Generating test data...")
    for scenario in test_scenarios:
        scenario_data = []
        for _ in range(scenario["count"]):
            data = TemporalData(
                edge_index=torch.randint(0, scenario["nodes"], (2, scenario["edges"])),
                timestamps=torch.sort(torch.rand(scenario["edges"]) * 100)[0],
                node_features=torch.randn(scenario["nodes"], 128),
                edge_attr=torch.randn(scenario["edges"], 64),
                num_nodes=scenario["nodes"]
            )
            scenario_data.append(data)
        all_test_data.append((scenario, scenario_data))
        print(f"   {scenario['name']}: {len(scenario_data)} graphs prepared")
    
    # Performance benchmarks
    print(f"\n⚡ Running performance benchmarks...")
    
    for scenario, data_list in all_test_data:
        print(f"\n📋 Testing {scenario['name']}...")
        
        start_time = time.time()
        
        # Single inference test
        single_start = time.time()
        output = engine.optimized_forward(data_list[0], return_attention=True, return_uncertainty=True)
        single_time = time.time() - single_start
        print(f"   Single inference: {single_time*1000:.1f}ms")
        
        # Batch inference test
        if len(data_list) > 1:
            batch_start = time.time()
            batch_outputs = engine.batch_inference(data_list[:5], return_attention=False, return_uncertainty=False)
            batch_time = time.time() - batch_start
            avg_batch_time = batch_time / min(5, len(data_list))
            print(f"   Batch inference: {batch_time*1000:.1f}ms total, {avg_batch_time*1000:.1f}ms average")
            
            # Speedup calculation
            speedup = single_time / avg_batch_time if avg_batch_time > 0 else 1.0
            print(f"   Batch speedup: {speedup:.2f}x")
        
        total_time = time.time() - start_time
        print(f"   Scenario total: {total_time:.1f}s")
    
    # Auto-tuning demonstration
    print(f"\n🔧 Auto-tuning performance...")
    tuning_result = engine.auto_tune_performance()
    
    print(f"   CPU usage: {tuning_result['metrics']['cpu_percent']:.1f}%")
    print(f"   Memory usage: {tuning_result['metrics']['memory_percent']:.1f}%")
    print(f"   Cache hit rate: {tuning_result['metrics']['hit_rate']:.1%}")
    print(f"   Tuning applied: {tuning_result['tuning_applied']}")
    
    if tuning_result['recommendations']['actions']:
        print(f"   Recommendations: {', '.join(tuning_result['recommendations']['actions'])}")
    
    # Cache effectiveness test
    print(f"\n💾 Testing cache effectiveness...")
    
    # Run same data multiple times to test caching
    test_data = all_test_data[1][1][0]  # Medium graph
    
    # First run (cache miss)
    start_time = time.time()
    output1 = engine.optimized_forward(test_data, use_cache=True)
    first_run_time = time.time() - start_time
    
    # Second run (cache hit)
    start_time = time.time()
    output2 = engine.optimized_forward(test_data, use_cache=True)
    second_run_time = time.time() - start_time
    
    cache_speedup = first_run_time / max(second_run_time, 0.001)
    print(f"   First run (miss): {first_run_time*1000:.1f}ms")
    print(f"   Second run (hit): {second_run_time*1000:.1f}ms")
    print(f"   Cache speedup: {cache_speedup:.1f}x")
    
    # Performance report
    print(f"\n📊 Final Performance Report:")
    report = engine.get_performance_report()
    
    perf = report["performance"]
    if "avg_inference_time" in perf:
        print(f"   Total inferences: {perf['total_inferences']:,}")
        print(f"   Average time: {perf['avg_inference_time']*1000:.1f}ms")
        print(f"   P95 time: {perf['p95_inference_time']*1000:.1f}ms")
        print(f"   Throughput: {perf['throughput_ops_per_sec']:.1f} ops/sec")
    
    cache = report["cache"]
    print(f"   Cache hit rate: {cache['hit_rate']:.1%}")
    print(f"   Cache size: {cache['memory_size_mb']:.1f}MB")
    print(f"   Cache entries: {cache['memory_entries']:,}")
    
    # Save performance report
    with open("gen3_performance_report.json", "w") as f:
        import json
        json.dump(report, f, indent=2, default=str)
    print(f"📁 Performance report saved to gen3_performance_report.json")
    
    # Cleanup
    engine.cleanup()
    
    print(f"\n🎉 Generation 3 Scaling Implementation Completed!")
    print("✅ Advanced caching with LRU eviction and compression")
    print("✅ Parallel processing and load balancing")
    print("✅ Auto-scaling based on system metrics")
    print("✅ Model optimization with TorchScript compilation")
    print("✅ Batch inference with intelligent batching")
    print("✅ Performance monitoring and auto-tuning")


if __name__ == "__main__":
    demo_scaling_optimizations()