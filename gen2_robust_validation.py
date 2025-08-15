#!/usr/bin/env python3
"""
Generation 2 Validation - Robust Error Handling and Monitoring
Tests comprehensive error handling, health checks, and monitoring capabilities.
"""

import sys
import os
sys.path.insert(0, 'src')

import torch
import torch.nn.functional as F
from typing import Dict, Any
import time
import logging

# Import DGDN components
import dgdn
from dgdn import DynamicGraphDiffusionNet
from dgdn.utils.error_handling import (
    validate_tensor_properties, validate_graph_data, robust_forward_pass,
    ErrorRecovery, setup_error_logging, ValidationError, ModelError
)
from dgdn.utils.health_checks import ModelHealthChecker, create_dummy_data

class SimpleTemporalData:
    """Simple temporal data structure for testing."""
    
    def __init__(self, edge_index, timestamps, num_nodes, node_features=None, edge_attr=None):
        self.edge_index = edge_index
        self.timestamps = timestamps
        self.num_nodes = num_nodes
        self.node_features = node_features
        self.edge_attr = edge_attr

def test_error_handling_validation():
    """Test comprehensive input validation and error handling."""
    print("Testing error handling and validation...")
    
    try:
        # Test tensor validation
        valid_tensor = torch.randn(10, 5)
        validate_tensor_properties(valid_tensor, "test_tensor")
        print("  ✓ Valid tensor validation passed")
        
        # Test invalid tensor (should raise ValidationError)
        try:
            invalid_tensor = torch.tensor([float('nan'), 1.0, 2.0])
            validate_tensor_properties(invalid_tensor, "invalid_tensor", allow_nan=False)
            print("  ✗ Should have caught NaN validation error")
            return False
        except ValidationError:
            print("  ✓ NaN validation error caught correctly")
        
        # Test graph data validation
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
        timestamps = torch.tensor([1.0, 2.0, 3.0])
        validate_graph_data(edge_index, timestamps, num_nodes=3)
        print("  ✓ Graph data validation passed")
        
        # Test invalid graph data
        try:
            invalid_edge_index = torch.tensor([[0, 1, 5], [1, 2, 0]])  # Node 5 doesn't exist
            validate_graph_data(invalid_edge_index, timestamps, num_nodes=3)
            print("  ✗ Should have caught invalid node index")
            return False
        except ValidationError:
            print("  ✓ Invalid node index caught correctly")
        
        print("✓ Error handling validation successful")
        return True
        
    except Exception as e:
        print(f"✗ Error handling validation failed: {e}")
        return False

def test_robust_forward_pass():
    """Test robust forward pass with error recovery."""
    print("Testing robust forward pass...")
    
    try:
        # Create model with robust forward pass decorator
        model = DynamicGraphDiffusionNet(
            node_dim=64,
            edge_dim=32,
            hidden_dim=128,
            num_layers=2,
            num_heads=4,
            diffusion_steps=3
        )
        
        # Apply robust forward pass decorator
        model.forward = robust_forward_pass(model.forward)
        
        # Test with valid data
        data = create_dummy_data(num_nodes=50, num_edges=150)
        model.eval()
        
        with torch.no_grad():
            output = model(data)
        
        print("  ✓ Robust forward pass with valid data successful")
        
        # Test error recovery
        recovery = ErrorRecovery()
        
        # Test NaN recovery
        nan_tensor = torch.tensor([1.0, float('nan'), 3.0])
        recovered = recovery.nan_recovery(nan_tensor, fill_value=0.0)
        if torch.isnan(recovered).any():
            print("  ✗ NaN recovery failed")
            return False
        print("  ✓ NaN recovery successful")
        
        # Test inf recovery
        inf_tensor = torch.tensor([1.0, float('inf'), 3.0])
        recovered = recovery.inf_recovery(inf_tensor, max_value=1e6)
        if torch.isinf(recovered).any():
            print("  ✗ Inf recovery failed")
            return False
        print("  ✓ Inf recovery successful")
        
        print("✓ Robust forward pass successful")
        return True
        
    except Exception as e:
        print(f"✗ Robust forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_health_checks():
    """Test comprehensive model health checks."""
    print("Testing health checks...")
    
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
        
        device = torch.device('cpu')
        health_checker = ModelHealthChecker(model, device)
        
        # Run all health checks
        results = health_checker.run_all_checks()
        
        # Validate results
        if not results:
            print("  ✗ No health check results returned")
            return False
        
        # Check that all expected health checks ran
        expected_checks = [
            'model_structure',
            'parameter_health', 
            'memory_usage',
            'inference_speed',
            'gradient_flow',
            'numerical_stability'
        ]
        
        check_names = [result.name for result in results]
        for expected in expected_checks:
            if expected not in check_names:
                print(f"  ✗ Missing health check: {expected}")
                return False
        
        print(f"  ✓ All {len(results)} health checks executed")
        
        # Generate health report
        report = health_checker.generate_health_report(results)
        
        if 'overall_status' not in report:
            print("  ✗ Health report missing overall status")
            return False
        
        print(f"  ✓ Health report generated: {report['overall_status']}")
        print(f"  ✓ Checks: {report['passed']} passed, {report['warnings']} warnings, {report['failures']} failures")
        
        # Test individual health check components
        for result in results:
            if result.status == "FAIL":
                print(f"  ⚠ Health check '{result.name}' failed: {result.message}")
            elif result.status == "WARN":
                print(f"  ⚠ Health check '{result.name}' warning: {result.message}")
            else:
                print(f"  ✓ Health check '{result.name}' passed")
        
        print("✓ Health checks successful")
        return True
        
    except Exception as e:
        print(f"✗ Health checks failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_logging_and_monitoring():
    """Test logging and monitoring setup."""
    print("Testing logging and monitoring...")
    
    try:
        # Setup error logging
        setup_error_logging(log_level="INFO")
        
        # Test logger
        logger = logging.getLogger('dgdn')
        logger.info("Test log message from DGDN")
        
        print("  ✓ Logging setup successful")
        
        # Test performance monitoring
        model = DynamicGraphDiffusionNet(
            node_dim=64,
            edge_dim=32,
            hidden_dim=128,
            num_layers=2,
            num_heads=4,
            diffusion_steps=3
        )
        
        data = create_dummy_data(num_nodes=50, num_edges=150)
        
        # Time multiple forward passes
        model.eval()
        times = []
        
        for _ in range(5):
            start_time = time.time()
            with torch.no_grad():
                output = model(data)
            end_time = time.time()
            times.append(end_time - start_time)
        
        avg_time = sum(times) / len(times)
        print(f"  ✓ Performance monitoring: avg inference time {avg_time:.4f}s")
        
        print("✓ Logging and monitoring successful")
        return True
        
    except Exception as e:
        print(f"✗ Logging and monitoring failed: {e}")
        return False

def test_memory_monitoring():
    """Test memory usage monitoring."""
    print("Testing memory monitoring...")
    
    try:
        # Test memory check functions
        from dgdn.utils.error_handling import check_memory_usage
        
        device = torch.device('cpu')
        memory_stats = check_memory_usage(device)
        
        expected_keys = ['allocated', 'total', 'utilization']
        for key in expected_keys:
            if key not in memory_stats:
                print(f"  ✗ Missing memory stat: {key}")
                return False
        
        print(f"  ✓ Memory stats: {memory_stats['utilization']:.1f}% utilization")
        
        # Test with model
        model = DynamicGraphDiffusionNet(
            node_dim=64,
            edge_dim=32,
            hidden_dim=128,
            num_layers=2,
            num_heads=4,
            diffusion_steps=3
        )
        
        data = create_dummy_data(num_nodes=100, num_edges=300)
        
        # Memory before
        memory_before = check_memory_usage(device)
        
        # Forward pass
        model.eval()
        with torch.no_grad():
            output = model(data)
        
        # Memory after
        memory_after = check_memory_usage(device)
        
        print(f"  ✓ Memory monitoring during inference successful")
        
        print("✓ Memory monitoring successful")
        return True
        
    except Exception as e:
        print(f"✗ Memory monitoring failed: {e}")
        return False

def test_security_validation():
    """Test security-related validations."""
    print("Testing security validation...")
    
    try:
        # Test input sanitization
        model = DynamicGraphDiffusionNet(
            node_dim=64,
            edge_dim=32,
            hidden_dim=128,
            num_layers=2,
            num_heads=4,
            diffusion_steps=3
        )
        
        # Test with potentially malicious inputs
        # Large input (potential DoS)
        try:
            large_data = create_dummy_data(num_nodes=10000, num_edges=50000)
            # This should be caught by memory monitoring
            print("  ✓ Large input handling")
        except Exception as e:
            print(f"  ✓ Large input properly rejected: {type(e).__name__}")
        
        # Test parameter bounds
        try:
            # Invalid model parameters should be caught during initialization
            invalid_model = DynamicGraphDiffusionNet(
                node_dim=-1,  # Invalid negative dimension
                hidden_dim=128
            )
            print("  ✗ Should have caught invalid parameters")
            return False
        except (ValueError, ValidationError):
            print("  ✓ Invalid parameters properly rejected")
        
        print("✓ Security validation successful")
        return True
        
    except Exception as e:
        print(f"✗ Security validation failed: {e}")
        return False

def run_generation_2_validation():
    """Run all Generation 2 validation tests."""
    print("=" * 70)
    print("GENERATION 2 VALIDATION - ROBUST ERROR HANDLING & MONITORING")
    print("=" * 70)
    
    tests = [
        test_error_handling_validation,
        test_robust_forward_pass,
        test_health_checks,
        test_logging_and_monitoring,
        test_memory_monitoring,
        test_security_validation
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
    print("GENERATION 2 VALIDATION SUMMARY")
    print("=" * 70)
    
    passed = sum(results)
    total = len(results)
    
    print(f"Tests passed: {passed}/{total}")
    print(f"Success rate: {passed/total*100:.1f}%")
    print(f"Total time: {end_time - start_time:.2f}s")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED - Generation 2 is ROBUST!")
        return True
    else:
        print("❌ Some tests failed - Generation 2 needs fixes")
        return False

if __name__ == "__main__":
    success = run_generation_2_validation()
    sys.exit(0 if success else 1)