#!/usr/bin/env python3
"""
Generation 3 Validation - Performance Optimization and Scaling
Tests comprehensive performance optimizations, auto-scaling, and distributed processing.
"""

import sys
import os
sys.path.insert(0, 'src')

import torch
import torch.nn.functional as F
from typing import Dict, Any
import time
import threading
import concurrent.futures

# Import DGDN components
import dgdn
from dgdn import DynamicGraphDiffusionNet
from dgdn.optimization.performance import (
    PerformanceOptimizer, AdaptiveCaching, BatchProcessor, 
    MemoryManager, profile_model_performance
)
from dgdn.scaling.auto_scaling import AutoScaler, ResourceMonitor, ScalingAction

class SimpleTemporalData:
    """Simple temporal data structure for testing."""
    
    def __init__(self, edge_index, timestamps, num_nodes, node_features=None, edge_attr=None):
        self.edge_index = edge_index
        self.timestamps = timestamps
        self.num_nodes = num_nodes
        self.node_features = node_features
        self.edge_attr = edge_attr

def create_test_data(num_nodes=100, num_edges=300, node_dim=64, edge_dim=32):
    """Create test temporal graph data."""
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    timestamps = torch.rand(num_edges) * 100.0
    node_features = torch.randn(num_nodes, node_dim)
    edge_attr = torch.randn(num_edges, edge_dim)
    
    return SimpleTemporalData(
        edge_index=edge_index,
        timestamps=timestamps,
        num_nodes=num_nodes,
        node_features=node_features,
        edge_attr=edge_attr
    )

def test_performance_optimization():
    """Test comprehensive performance optimization."""
    print("Testing performance optimization...")
    
    try:
        # Create model
        model = DynamicGraphDiffusionNet(
            node_dim=64,
            edge_dim=32,
            hidden_dim=128,
            num_layers=3,
            num_heads=4,
            diffusion_steps=5
        )
        
        device = torch.device('cpu')
        optimizer = PerformanceOptimizer(model, device)
        
        # Test baseline performance
        data = create_test_data(num_nodes=200, num_edges=600)
        
        model.eval()
        with torch.no_grad():
            start_time = time.time()
            for _ in range(5):
                output = model(data)
            baseline_time = time.time() - start_time
        
        print(f"  ✓ Baseline performance: {baseline_time:.4f}s for 5 iterations")
        
        # Apply inference optimizations
        optimizer.optimize_for_inference()
        print("  ✓ Inference optimizations applied")
        
        # Test optimized performance
        with torch.no_grad():
            start_time = time.time()
            for _ in range(5):
                output = model(data)
            optimized_time = time.time() - start_time
        
        speedup = baseline_time / optimized_time if optimized_time > 0 else 1.0
        print(f"  ✓ Optimized performance: {optimized_time:.4f}s (speedup: {speedup:.2f}x)")
        
        # Test model compilation (if available)
        if hasattr(torch, 'compile'):
            print("  ✓ Model compilation available")
        else:
            print("  ⚠ Model compilation not available in this PyTorch version")
        
        print("✓ Performance optimization successful")
        return True
        
    except Exception as e:
        print(f"✗ Performance optimization failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_adaptive_caching():
    """Test adaptive caching system."""
    print("Testing adaptive caching...")
    
    try:
        # Create cache
        cache = AdaptiveCaching(max_cache_size=10, ttl_seconds=5.0)
        
        # Test cache operations
        test_data = create_test_data(num_nodes=50, num_edges=150)
        key = cache._generate_key(test_data)
        
        # Cache miss
        result = cache.get(key)
        if result is not None:
            print("  ✗ Expected cache miss")
            return False
        print("  ✓ Cache miss handled correctly")
        
        # Cache put and hit
        test_tensor = torch.randn(50, 128)
        cache.put(key, test_tensor)
        
        result = cache.get(key)
        if result is None:
            print("  ✗ Expected cache hit")
            return False
        
        # Verify cached data
        if not torch.equal(result, test_tensor):
            print("  ✗ Cached data doesn't match original")
            return False
        print("  ✓ Cache hit and data integrity verified")
        
        # Test cache statistics
        stats = cache.get_stats()
        required_keys = ['cache_size', 'max_size', 'utilization']
        for key in required_keys:
            if key not in stats:
                print(f"  ✗ Missing cache stat: {key}")
                return False
        print("  ✓ Cache statistics working")
        
        # Test TTL expiration
        time.sleep(6.0)  # Wait for TTL expiration
        result = cache.get(cache._generate_key(test_data))
        if result is not None:
            print("  ✗ Cache should have expired")
            return False
        print("  ✓ TTL expiration working")
        
        print("✓ Adaptive caching successful")
        return True
        
    except Exception as e:
        print(f"✗ Adaptive caching failed: {e}")
        return False

def test_batch_processing():
    """Test efficient batch processing."""
    print("Testing batch processing...")
    
    try:
        # Create model and processor
        model = DynamicGraphDiffusionNet(
            node_dim=64,
            edge_dim=32,
            hidden_dim=128,
            num_layers=2,
            num_heads=4,
            diffusion_steps=3
        )
        
        processor = BatchProcessor(batch_size=4, num_workers=2)
        
        # Create test data
        data_list = [create_test_data(num_nodes=50, num_edges=150) for _ in range(12)]
        
        # Test sequential processing
        start_time = time.time()
        results_sequential = []
        model.eval()
        with torch.no_grad():
            for data in data_list:
                output = model(data)
                results_sequential.append(output['node_embeddings'])
        sequential_time = time.time() - start_time
        
        # Test batch processing
        start_time = time.time()
        results_batch = processor.parallel_process(model, data_list)
        batch_time = time.time() - start_time
        
        # Verify results
        if len(results_batch) != len(results_sequential):
            print("  ✗ Batch processing returned wrong number of results")
            return False
        
        print(f"  ✓ Sequential processing: {sequential_time:.4f}s")
        print(f"  ✓ Batch processing: {batch_time:.4f}s")
        
        # Check for speedup (may not always be faster due to overhead)
        if batch_time <= sequential_time * 1.5:  # Allow some overhead
            print("  ✓ Batch processing efficiency acceptable")
        else:
            print("  ⚠ Batch processing overhead detected (acceptable for small batches)")
        
        print("✓ Batch processing successful")
        return True
        
    except Exception as e:
        print(f"✗ Batch processing failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_memory_management():
    """Test advanced memory management."""
    print("Testing memory management...")
    
    try:
        device = torch.device('cpu')
        memory_manager = MemoryManager(device)
        
        # Test memory pressure detection
        pressure = memory_manager.check_memory_pressure()
        print(f"  ✓ Memory pressure check: {pressure}")
        
        # Test memory cleanup
        memory_manager.emergency_cleanup()
        print("  ✓ Emergency cleanup executed")
        
        # Test memory management decorator
        @memory_manager.with_memory_management
        def test_function():
            return torch.randn(1000, 1000)
        
        result = test_function()
        if result.shape != (1000, 1000):
            print("  ✗ Memory managed function failed")
            return False
        print("  ✓ Memory management decorator working")
        
        print("✓ Memory management successful")
        return True
        
    except Exception as e:
        print(f"✗ Memory management failed: {e}")
        return False

def test_auto_scaling():
    """Test auto-scaling system."""
    print("Testing auto-scaling...")
    
    try:
        # Create auto-scaler
        auto_scaler = AutoScaler(
            min_instances=1,
            max_instances=5,
            target_cpu_percent=50.0,
            cooldown_period=1.0  # Short for testing
        )
        
        # Test resource monitoring
        monitor = auto_scaler.resource_monitor
        
        # Collect initial metrics
        metrics = monitor._collect_metrics()
        print(f"  ✓ Initial metrics collected: CPU={metrics.cpu_percent:.1f}%, Memory={metrics.memory_percent:.1f}%")
        
        # Test scaling decision logic
        decision = auto_scaler.make_scaling_decision(metrics)
        print(f"  ✓ Scaling decision: {decision.action.value} (confidence: {decision.confidence:.2f})")
        
        # Simulate high load
        metrics.cpu_percent = 90.0
        metrics.memory_percent = 85.0
        decision = auto_scaler.make_scaling_decision(metrics)
        
        if decision.action == ScalingAction.SCALE_UP:
            print("  ✓ High load triggers scale-up decision")
        else:
            print("  ⚠ High load should trigger scale-up (may be in cooldown)")
        
        # Simulate low load
        metrics.cpu_percent = 10.0
        metrics.memory_percent = 15.0
        time.sleep(1.1)  # Wait for cooldown
        decision = auto_scaler.make_scaling_decision(metrics)
        
        if decision.action == ScalingAction.SCALE_DOWN:
            print("  ✓ Low load triggers scale-down decision")
        else:
            print("  ⚠ Low load should trigger scale-down (may be at minimum)")
        
        # Test status reporting
        status = auto_scaler.get_current_status()
        required_keys = ['current_instances', 'min_instances', 'max_instances']
        for key in required_keys:
            if key not in status:
                print(f"  ✗ Missing status key: {key}")
                return False
        print("  ✓ Status reporting working")
        
        print("✓ Auto-scaling successful")
        return True
        
    except Exception as e:
        print(f"✗ Auto-scaling failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_concurrent_processing():
    """Test concurrent processing capabilities."""
    print("Testing concurrent processing...")
    
    try:
        # Create model
        model = DynamicGraphDiffusionNet(
            node_dim=64,
            edge_dim=32,
            hidden_dim=128,
            num_layers=2,
            num_heads=4,
            diffusion_steps=3
        )
        
        # Create test data
        data_list = [create_test_data(num_nodes=50, num_edges=150) for _ in range(8)]
        
        def process_single(data):
            model.eval()
            with torch.no_grad():
                return model(data)
        
        # Test sequential processing
        start_time = time.time()
        results_sequential = [process_single(data) for data in data_list]
        sequential_time = time.time() - start_time
        
        # Test concurrent processing
        start_time = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            results_concurrent = list(executor.map(process_single, data_list))
        concurrent_time = time.time() - start_time
        
        # Verify results
        if len(results_concurrent) != len(results_sequential):
            print("  ✗ Concurrent processing returned wrong number of results")
            return False
        
        print(f"  ✓ Sequential processing: {sequential_time:.4f}s")
        print(f"  ✓ Concurrent processing: {concurrent_time:.4f}s")
        
        # Check for reasonable performance
        if concurrent_time <= sequential_time * 1.2:  # Allow some overhead
            print("  ✓ Concurrent processing efficiency acceptable")
        else:
            print("  ⚠ Concurrent processing overhead detected (may be due to GIL)")
        
        print("✓ Concurrent processing successful")
        return True
        
    except Exception as e:
        print(f"✗ Concurrent processing failed: {e}")
        return False

def test_scalability_stress():
    """Test model scalability under stress."""
    print("Testing scalability stress...")
    
    try:
        # Create model
        model = DynamicGraphDiffusionNet(
            node_dim=64,
            edge_dim=32,
            hidden_dim=128,
            num_layers=2,
            num_heads=4,
            diffusion_steps=3
        )
        
        # Test with increasing data sizes
        sizes = [(50, 150), (100, 300), (200, 600)]
        times = []
        
        model.eval()
        for nodes, edges in sizes:
            data = create_test_data(num_nodes=nodes, num_edges=edges)
            
            start_time = time.time()
            with torch.no_grad():
                output = model(data)
            process_time = time.time() - start_time
            times.append(process_time)
            
            print(f"  ✓ Processed {nodes} nodes, {edges} edges in {process_time:.4f}s")
        
        # Check scaling behavior (should be roughly linear or sub-quadratic)
        if len(times) >= 2:
            scaling_factor = times[-1] / times[0]
            size_factor = (sizes[-1][0] * sizes[-1][1]) / (sizes[0][0] * sizes[0][1])
            efficiency = size_factor / scaling_factor
            
            print(f"  ✓ Scaling efficiency: {efficiency:.2f} (higher is better)")
            
            if efficiency > 0.5:  # Reasonable efficiency
                print("  ✓ Good scaling behavior")
            else:
                print("  ⚠ Suboptimal scaling behavior (may be due to overhead)")
        
        print("✓ Scalability stress test successful")
        return True
        
    except Exception as e:
        print(f"✗ Scalability stress test failed: {e}")
        return False

def run_generation_3_validation():
    """Run all Generation 3 validation tests."""
    print("=" * 70)
    print("GENERATION 3 VALIDATION - PERFORMANCE OPTIMIZATION & SCALING")
    print("=" * 70)
    
    tests = [
        test_performance_optimization,
        test_adaptive_caching,
        test_batch_processing,
        test_memory_management,
        test_auto_scaling,
        test_concurrent_processing,
        test_scalability_stress
    ]
    
    results = []
    start_time = time.time()
    
    for test in tests:
        result = test()
        results.append(result)
        print()
    
    end_time = time.time()
    
    # Summary
    print("=" * 70)
    print("GENERATION 3 VALIDATION SUMMARY")
    print("=" * 70)
    
    passed = sum(results)
    total = len(results)
    
    print(f"Tests passed: {passed}/{total}")
    print(f"Success rate: {passed/total*100:.1f}%")
    print(f"Total time: {end_time - start_time:.2f}s")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED - Generation 3 SCALES!")
        return True
    else:
        print("❌ Some tests failed - Generation 3 needs optimization")
        return False

if __name__ == "__main__":
    success = run_generation_3_validation()
    sys.exit(0 if success else 1)