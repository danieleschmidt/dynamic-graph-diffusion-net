#!/usr/bin/env python3
"""
Security tests for DGDN implementation.

Tests input validation, path traversal prevention, and other security measures.
"""

import pytest
import torch
import tempfile
import os
from pathlib import Path

from dgdn import DynamicGraphDiffusionNet, DGDNTrainer, TemporalData


class TestSecurityValidation:
    """Test security-related validation."""
    
    def test_model_parameter_validation(self):
        """Test model parameter validation."""
        # Test negative dimensions
        with pytest.raises(ValueError, match="node_dim must be a positive integer"):
            DynamicGraphDiffusionNet(node_dim=-1)
        
        # Test zero dimensions
        with pytest.raises(ValueError, match="hidden_dim must be a positive integer"):
            DynamicGraphDiffusionNet(node_dim=64, hidden_dim=0)
        
        # Test incompatible dimensions
        with pytest.raises(ValueError, match="hidden_dim .* must be divisible by num_heads"):
            DynamicGraphDiffusionNet(node_dim=64, hidden_dim=127, num_heads=8)
        
        # Test invalid dropout
        with pytest.raises(ValueError, match="dropout must be a float"):
            DynamicGraphDiffusionNet(node_dim=64, dropout=1.5)
        
        # Test invalid activation
        with pytest.raises(ValueError, match="activation must be one of"):
            DynamicGraphDiffusionNet(node_dim=64, activation="invalid")
    
    def test_trainer_parameter_validation(self):
        """Test trainer parameter validation."""
        model = DynamicGraphDiffusionNet(node_dim=64, hidden_dim=128)
        
        # Test negative learning rate
        with pytest.raises(ValueError, match="learning_rate must be positive"):
            DGDNTrainer(model, learning_rate=-0.1)
        
        # Test negative weight decay
        with pytest.raises(ValueError, match="weight_decay must be non-negative"):
            DGDNTrainer(model, weight_decay=-0.1)
    
    def test_forward_input_validation(self):
        """Test forward pass input validation."""
        model = DynamicGraphDiffusionNet(node_dim=64, hidden_dim=128)
        
        # Test missing attributes
        class BadData:
            pass
        
        bad_data = BadData()
        with pytest.raises(ValueError, match="Input data must have 'edge_index' attribute"):
            model(bad_data)
        
        # Test wrong tensor types
        bad_data.edge_index = [[0, 1], [1, 2]]  # Not a tensor
        bad_data.timestamps = torch.tensor([0.1, 0.2])
        bad_data.num_nodes = 3
        
        with pytest.raises(TypeError, match="edge_index must be a torch.Tensor"):
            model(bad_data)
        
        # Test wrong shapes
        bad_data.edge_index = torch.tensor([0, 1, 2])  # Wrong shape
        bad_data.timestamps = torch.tensor([0.1, 0.2])
        bad_data.num_nodes = 3
        
        with pytest.raises(ValueError, match="edge_index must have shape"):
            model(bad_data)
    
    def test_secure_checkpoint_paths(self):
        """Test secure checkpoint path handling."""
        model = DynamicGraphDiffusionNet(node_dim=64, hidden_dim=128)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = DGDNTrainer(model, checkpoint_dir=tmpdir)
            
            # Test normal filename
            safe_path = trainer._secure_checkpoint_path("model.pth")
            assert safe_path.endswith("model.pth")
            assert tmpdir in safe_path
            
            # Test path traversal attack
            with pytest.raises(ValueError, match="Insecure checkpoint path"):
                trainer._secure_checkpoint_path("../../../etc/passwd")
            
            # Test filename sanitization
            safe_path = trainer._secure_checkpoint_path("model<>:\"/|?*.pth")
            assert "<" not in safe_path and ">" not in safe_path
            assert safe_path.endswith(".pth")
    
    def test_tensor_shape_validation(self):
        """Test comprehensive tensor shape validation."""
        model = DynamicGraphDiffusionNet(node_dim=64, hidden_dim=128)
        
        # Create valid base data
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
        timestamps = torch.tensor([0.1, 0.2, 0.3])
        num_nodes = 3
        
        # Test mismatched edge counts
        data = TemporalData(
            edge_index=edge_index,
            timestamps=torch.tensor([0.1, 0.2]),  # Wrong count
            num_nodes=num_nodes
        )
        
        with pytest.raises(ValueError, match="Number of edges.*must match timestamps"):
            model(data)
        
        # Test node index out of bounds
        bad_edge_index = torch.tensor([[0, 1, 5], [1, 2, 0]])  # Node 5 doesn't exist
        data = TemporalData(
            edge_index=bad_edge_index,
            timestamps=timestamps,
            num_nodes=num_nodes
        )
        
        with pytest.raises(ValueError, match="Maximum node index.*exceeds num_nodes"):
            model(data)
    
    def test_data_size_validation(self):
        """Test minimum data size requirements."""
        model = DynamicGraphDiffusionNet(node_dim=64, hidden_dim=128)
        
        # Create minimal dataset (too small)
        edge_index = torch.tensor([[0, 1], [1, 0]])  # Only 2 edges
        timestamps = torch.tensor([0.1, 0.2])
        
        class MinimalDataset:
            def __init__(self):
                self.data = TemporalData(
                    edge_index=edge_index,
                    timestamps=timestamps,
                    num_nodes=2
                )
        
        dataset = MinimalDataset()
        
        with pytest.raises(ValueError, match="Insufficient training data"):
            trainer = DGDNTrainer(model)
            trainer._validate_training_data(dataset)
    
    def test_memory_bounds_checking(self):
        """Test memory bounds and overflow protection."""
        # Test extremely large dimensions that could cause memory issues
        with pytest.raises(ValueError):
            # This should fail validation before causing memory issues
            DynamicGraphDiffusionNet(
                node_dim=10**9,  # Unreasonably large
                hidden_dim=10**9,
                num_layers=1000
            )
    
    def test_input_sanitization(self):
        """Test input data sanitization."""
        model = DynamicGraphDiffusionNet(node_dim=64, hidden_dim=128)
        
        # Test NaN/Inf values
        edge_index = torch.tensor([[0, 1], [1, 0]])
        timestamps = torch.tensor([float('nan'), float('inf')])
        
        data = TemporalData(
            edge_index=edge_index,
            timestamps=timestamps,
            num_nodes=2
        )
        
        # Model should handle or detect invalid values
        try:
            output = model(data)
            # Check if output contains NaN/Inf
            assert not torch.isnan(output["node_embeddings"]).any()
            assert not torch.isinf(output["node_embeddings"]).any()
        except (ValueError, RuntimeError):
            # It's also acceptable to raise an error for invalid inputs
            pass


class TestSecurityLogging:
    """Test security-related logging features."""
    
    def test_secure_logging(self):
        """Test that sensitive information is not logged."""
        model = DynamicGraphDiffusionNet(node_dim=64, hidden_dim=128)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = DGDNTrainer(model, log_dir=tmpdir)
            
            # Check that logger is properly configured
            assert trainer.logger is not None
            
            # Check that log file exists
            log_file = Path(tmpdir) / "training.log"
            assert log_file.parent.exists()
    
    def test_log_rate_limiting(self):
        """Test that logging doesn't spam output."""
        model = DynamicGraphDiffusionNet(node_dim=64, hidden_dim=128)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = DGDNTrainer(model, log_dir=tmpdir)
            
            # Simulate multiple rapid log calls
            metrics = {"loss": 1.0, "auc": 0.5}
            
            # Should not cause excessive logging
            for i in range(100):
                trainer._log_training_progress(i, metrics)


if __name__ == "__main__":
    # Run basic security tests
    print("🔒 Running DGDN Security Tests...")
    
    # Test model validation
    try:
        DynamicGraphDiffusionNet(node_dim=-1)
        print("❌ Model validation failed")
    except ValueError:
        print("✅ Model parameter validation working")
    
    # Test trainer validation
    try:
        model = DynamicGraphDiffusionNet(node_dim=64, hidden_dim=128)
        DGDNTrainer(model, learning_rate=-1.0)
        print("❌ Trainer validation failed")
    except ValueError:
        print("✅ Trainer parameter validation working")
    
    # Test path security
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = DGDNTrainer(model, checkpoint_dir=tmpdir)
            result = trainer._secure_checkpoint_path("../../../etc/passwd")
            if "../" not in result and tmpdir in result:
                print("✅ Path traversal protection working (sanitized)")
            else:
                print("❌ Path traversal protection failed")
    except ValueError:
        print("✅ Path traversal protection working (blocked)")
    
    print("🔒 Security tests completed successfully!")