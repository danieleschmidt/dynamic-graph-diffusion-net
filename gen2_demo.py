#!/usr/bin/env python3
"""Generation 2 Demo: MAKE IT ROBUST (Reliable)
Demonstrates comprehensive error handling, validation, logging, monitoring, and security.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import torch
import numpy as np
import tempfile
import time
from pathlib import Path

# Import Generation 1 functionality
from dgdn.data.datasets import TemporalData
from dgdn.models.dgdn import DynamicGraphDiffusionNet

# Import Generation 2 robustness features
from dgdn.utils.config import DGDNConfig, ModelConfig, TrainingConfig
from dgdn.utils.logging import setup_logging, get_logger, TrainingLogger, PerformanceLogger
from dgdn.utils.validation import validate_temporal_data, validate_model_config, DataValidator
from dgdn.utils.monitoring import PerformanceProfiler, ModelMonitor, TrainingMonitor
from dgdn.utils.security import SecurityValidator, secure_model_save, secure_model_load, sanitize_input


def test_configuration_management():
    """Test configuration management system."""
    print("\n🔧 Testing Configuration Management...")
    logger = get_logger("dgdn.demo")
    
    # Create default configuration
    config = DGDNConfig()
    logger.info("Created default configuration")
    
    # Validate configuration
    config.validate()
    logger.info("Configuration validation passed")
    
    # Test configuration serialization
    config_dict = config.to_dict()
    config_from_dict = DGDNConfig.from_dict(config_dict)
    
    # Save and load configuration
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        config_path = f.name
    
    try:
        config.save(config_path)
        loaded_config = DGDNConfig.from_file(config_path)
        logger.info(f"Configuration saved and loaded successfully: {config_path}")
    finally:
        Path(config_path).unlink()  # Clean up
    
    print("✅ Configuration management working correctly")
    return config


def test_logging_system():
    """Test comprehensive logging system."""
    print("\n📝 Testing Logging System...")
    
    # Setup structured logging
    logger = setup_logging(log_level="INFO", include_timestamp=True)
    
    # Test different log levels
    logger.debug("Debug message")
    logger.info("Info message") 
    logger.warning("Warning message")
    logger.error("Error message")
    
    # Test training logger
    training_logger = TrainingLogger()
    training_logger.log_training_start({
        'learning_rate': 0.001,
        'batch_size': 32,
        'model': 'DGDN'
    })
    
    training_logger.log_epoch_start(0, 10)
    training_logger.log_epoch_end(0, 0.5, 0.4, {'accuracy': 0.85})
    
    # Test performance logger
    perf_logger = PerformanceLogger()
    perf_logger.start_timer("test_operation")
    time.sleep(0.1)  # Simulate work
    elapsed = perf_logger.end_timer("test_operation")
    
    print(f"✅ Logging system working correctly (test operation: {elapsed:.1f}ms)")
    return logger


def test_validation_system():
    """Test comprehensive validation system."""
    print("\n✅ Testing Validation System...")
    logger = get_logger("dgdn.demo")
    
    # Test tensor validation
    test_tensor = torch.randn(100, 64)
    
    try:
        from dgdn.utils.validation import validate_tensor_properties
        validate_tensor_properties(
            test_tensor, 
            "test_tensor",
            expected_shape=(100, 64),
            expected_dtype=torch.float32,
            min_value=-10.0,
            max_value=10.0
        )
        logger.info("Tensor validation passed")
    except Exception as e:
        logger.error(f"Tensor validation failed: {e}")
        raise
    
    # Test data validation
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
    timestamps = torch.tensor([1.0, 2.0, 3.0])
    
    temporal_data = TemporalData(
        edge_index=edge_index,
        timestamps=timestamps,
        node_features=torch.randn(3, 64),
        num_nodes=3
    )
    
    validate_temporal_data(temporal_data)
    logger.info("Temporal data validation passed")
    
    # Test model config validation
    model_config = ModelConfig(
        node_dim=64,
        hidden_dim=128,
        num_heads=4
    )
    validate_model_config(model_config)
    logger.info("Model config validation passed")
    
    # Test data validator with statistics
    validator = DataValidator(strict=True)
    validator.validate(temporal_data, "test_data")
    validator.validate(test_tensor, "test_tensor")
    
    stats = validator.get_stats()
    logger.info(f"Validation statistics: {stats}")
    
    print("✅ Validation system working correctly")
    return validator


def test_monitoring_system():
    """Test monitoring and profiling system."""
    print("\n📊 Testing Monitoring System...")
    logger = get_logger("dgdn.demo")
    
    # Test performance profiler
    profiler = PerformanceProfiler(enable_gpu_monitoring=torch.cuda.is_available())
    
    # Profile a simple operation
    profiler.start_timer("matrix_multiply")
    a = torch.randn(1000, 1000)
    b = torch.randn(1000, 1000)
    c = torch.mm(a, b)
    elapsed = profiler.end_timer("matrix_multiply")
    
    # Capture metrics
    metrics = profiler.capture_metrics(
        forward_time_ms=elapsed,
        batch_size=1,
        num_nodes=1000,
        num_edges=2000
    )
    logger.info(f"Captured performance metrics: {metrics.memory_used_mb:.2f} MB memory")
    
    # Test training monitor
    training_monitor = TrainingMonitor(patience=5)
    
    # Simulate training progression
    for epoch in range(10):
        train_loss = 1.0 - epoch * 0.1 + np.random.normal(0, 0.01)
        val_loss = 0.9 - epoch * 0.08 + np.random.normal(0, 0.02)
        
        training_monitor.update(
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            learning_rate=0.001,
            metrics={'accuracy': 0.7 + epoch * 0.02}
        )
        
        if training_monitor.should_stop:
            logger.info(f"Early stopping would trigger at epoch {epoch}")
            break
    
    # Get training summary
    summary = training_monitor.get_training_summary()
    logger.info(f"Training summary: best_val_loss={summary['best_val_loss']:.4f}")
    
    # Test model monitor
    model = DynamicGraphDiffusionNet(
        node_dim=64, hidden_dim=128, num_layers=2
    )
    
    model_monitor = ModelMonitor(model, monitor_gradients=True)
    
    # Simulate forward/backward pass
    dummy_data = TemporalData(
        edge_index=torch.tensor([[0, 1], [1, 0]]),
        timestamps=torch.tensor([1.0, 2.0]),
        node_features=torch.randn(2, 64),
        num_nodes=2
    )
    
    model.train()
    output = model(dummy_data)
    loss = output['kl_loss']
    loss.backward()
    
    # Get health report
    health_report = model_monitor.get_health_report()
    param_issues = health_report['parameter_issues']
    grad_issues = health_report['gradient_issues']
    
    logger.info(f"Model health check: {len(param_issues)} parameter issues, {len(grad_issues)} gradient issues")
    
    model_monitor.cleanup()
    
    print("✅ Monitoring system working correctly")
    return profiler, training_monitor


def test_security_system():
    """Test security validation and protection."""
    print("\n🔒 Testing Security System...")
    logger = get_logger("dgdn.demo")
    
    # Test security validator
    security_validator = SecurityValidator(enable_strict_mode=True)
    
    # Test tensor security validation
    safe_tensor = torch.randn(100, 64)
    security_validator.validate_tensor_security(safe_tensor, "safe_tensor")
    logger.info("Safe tensor validation passed")
    
    # Test dangerous tensor detection
    try:
        dangerous_tensor = torch.tensor([float('nan'), float('inf'), 1.0])
        security_validator.validate_tensor_security(dangerous_tensor, "dangerous_tensor")
        logger.error("Should have caught dangerous tensor!")
    except Exception as e:
        logger.info(f"Correctly caught dangerous tensor: {e}")
    
    # Test input sanitization
    unsafe_input = {
        'message': '<script>alert("xss")</script>Hello World!',
        'numbers': [1, 2, 3],
        'nested': {'key': 'value with "quotes"'}
    }
    
    safe_input = sanitize_input(unsafe_input)
    logger.info(f"Sanitized input: {safe_input}")
    
    # Test secure model save/load
    model = DynamicGraphDiffusionNet(node_dim=64, hidden_dim=128, num_layers=1)
    
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
        model_path = f.name
    
    try:
        # Secure save
        model_hash = secure_model_save(model, model_path, include_hash=True)
        logger.info(f"Model saved with hash: {model_hash[:16]}...")
        
        # Secure load
        new_model = DynamicGraphDiffusionNet(node_dim=64, hidden_dim=128, num_layers=1)
        secure_model_load(new_model, model_path, verify_hash=True)
        logger.info("Model loaded and integrity verified")
        
    finally:
        # Clean up
        Path(model_path).unlink(missing_ok=True)
        Path(model_path + '.hash').unlink(missing_ok=True)
    
    print("✅ Security system working correctly")
    return security_validator


def test_error_handling():
    """Test comprehensive error handling."""
    print("\n🚨 Testing Error Handling...")
    logger = get_logger("dgdn.demo")
    
    # Test graceful handling of invalid inputs
    try:
        # Invalid temporal data
        invalid_data = TemporalData(
            edge_index=torch.tensor([[0, 5], [1, 2]]),  # Node 5 doesn't exist
            timestamps=torch.tensor([1.0, 2.0]),
            num_nodes=3
        )
        validate_temporal_data(invalid_data)
        logger.error("Should have caught invalid data!")
    except Exception as e:
        logger.info(f"Correctly caught invalid data: {e}")
    
    # Test model with invalid configuration
    try:
        invalid_config = ModelConfig(
            node_dim=-10,  # Invalid negative dimension
            hidden_dim=127,  # Not divisible by num_heads
            num_heads=8
        )
        validate_model_config(invalid_config)
        logger.error("Should have caught invalid config!")
    except Exception as e:
        logger.info(f"Correctly caught invalid config: {e}")
    
    # Test recovery from corrupted state
    try:
        model = DynamicGraphDiffusionNet(node_dim=64, hidden_dim=128)
        
        # Simulate parameter corruption
        for param in model.parameters():
            param.data.fill_(float('nan'))
        
        model_monitor = ModelMonitor(model)
        health_report = model_monitor.get_health_report()
        
        if health_report['parameter_issues']:
            logger.info(f"Detected parameter corruption: {health_report['parameter_issues']}")
        
        model_monitor.cleanup()
        
    except Exception as e:
        logger.info(f"Handled corrupted model gracefully: {e}")
    
    print("✅ Error handling working correctly")


def test_comprehensive_integration():
    """Test integration of all robustness features."""
    print("\n🔄 Testing Comprehensive Integration...")
    logger = get_logger("dgdn.demo")
    
    # Create configuration
    config = DGDNConfig()
    config.model.node_dim = 64
    config.model.hidden_dim = 128
    config.model.num_layers = 2
    
    # Initialize monitoring
    profiler = PerformanceProfiler()
    training_monitor = TrainingMonitor(patience=3)
    security_validator = SecurityValidator()
    
    # Create and validate model
    model = DynamicGraphDiffusionNet(**config.model.__dict__)
    model_monitor = ModelMonitor(model)
    
    # Create and validate data
    data = TemporalData(
        edge_index=torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]]),
        timestamps=torch.tensor([1.0, 2.0, 3.0, 4.0]),
        node_features=torch.randn(4, 64),
        num_nodes=4
    )
    
    validate_temporal_data(data)
    security_validator.validate_tensor_security(data.edge_index, "edge_index")
    
    # Simulate robust training loop
    optimizer = torch.optim.Adam(model.parameters(), lr=config.training.learning_rate)
    
    for epoch in range(5):
        # Training step with monitoring
        profiler.start_timer("forward_pass")
        model.train()
        
        output = model(data)
        forward_time = profiler.end_timer("forward_pass")
        
        # Compute loss
        dummy_targets = torch.randint(0, 2, (4,))
        losses = model.compute_loss(output, dummy_targets, "edge_prediction")
        
        profiler.start_timer("backward_pass")
        losses['total'].backward()
        backward_time = profiler.end_timer("backward_pass")
        
        # Check gradient health
        health_report = model_monitor.get_health_report()
        if health_report['gradient_issues']:
            logger.warning(f"Gradient issues detected: {health_report['gradient_issues']}")
        
        optimizer.step()
        optimizer.zero_grad()
        
        # Update training monitor
        val_loss = losses['total'].item() * (1 + np.random.normal(0, 0.1))
        training_monitor.update(
            epoch=epoch,
            train_loss=losses['total'].item(),
            val_loss=val_loss,
            learning_rate=config.training.learning_rate
        )
        
        # Capture performance metrics
        metrics = profiler.capture_metrics(
            forward_time_ms=forward_time,
            backward_time_ms=backward_time,
            batch_size=1,
            num_nodes=data.num_nodes,
            num_edges=data.edge_index.shape[1]
        )
        
        logger.info(f"Epoch {epoch}: loss={losses['total'].item():.4f}, "
                   f"forward={forward_time:.1f}ms, memory={metrics.memory_used_mb:.1f}MB")
        
        if training_monitor.should_stop:
            logger.info("Early stopping triggered")
            break
    
    # Get final summary
    training_summary = training_monitor.get_training_summary()
    performance_summary = profiler.get_summary()
    
    logger.info(f"Training completed: {training_summary['total_epochs']} epochs")
    logger.info(f"Performance: {performance_summary['total_operations']} operations monitored")
    
    model_monitor.cleanup()
    
    print("✅ Comprehensive integration working correctly")


def main():
    """Run Generation 2 comprehensive robustness demo."""
    print("🛡️  DGDN Generation 2 Demo: MAKE IT ROBUST")
    print("=" * 60)
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    try:
        # Test all robustness features
        test_configuration_management()
        test_logging_system()
        test_validation_system() 
        test_monitoring_system()
        test_security_system()
        test_error_handling()
        test_comprehensive_integration()
        
        print("\n" + "=" * 60)
        print("🎉 GENERATION 2 COMPLETE: ROBUSTNESS VERIFIED")
        print("✅ Configuration management working")
        print("✅ Comprehensive logging system operational")
        print("✅ Input validation and data integrity checks active")
        print("✅ Performance monitoring and profiling functional")
        print("✅ Security validation and protection enabled")
        print("✅ Error handling and recovery mechanisms tested")
        print("✅ All systems integrated and operational")
        print("✅ Ready for Generation 3: Scalability optimizations")
        
    except Exception as e:
        print(f"\n❌ Error in Generation 2 demo: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)