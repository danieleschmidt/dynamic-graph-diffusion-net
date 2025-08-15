#!/usr/bin/env python3
"""
Generation 1 Validation - Simple DGDN Functionality Test
Tests basic import and model instantiation capabilities.
"""

import sys
import os
sys.path.insert(0, 'src')

import torch
import torch.nn.functional as F
from typing import Dict, Any
import time

# Import DGDN components
import dgdn
from dgdn import DynamicGraphDiffusionNet

class SimpleTemporalData:
    """Simple temporal data structure for testing."""
    
    def __init__(self, edge_index, timestamps, num_nodes, node_features=None, edge_attr=None):
        self.edge_index = edge_index
        self.timestamps = timestamps
        self.num_nodes = num_nodes
        self.node_features = node_features
        self.edge_attr = edge_attr

def create_synthetic_data(num_nodes=100, num_edges=300, node_dim=64, edge_dim=32):
    """Create synthetic temporal graph data for testing."""
    
    # Random edge indices
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    
    # Random timestamps
    timestamps = torch.rand(num_edges) * 100.0
    
    # Random node features
    node_features = torch.randn(num_nodes, node_dim)
    
    # Random edge attributes
    edge_attr = torch.randn(num_edges, edge_dim)
    
    return SimpleTemporalData(
        edge_index=edge_index,
        timestamps=timestamps,
        num_nodes=num_nodes,
        node_features=node_features,
        edge_attr=edge_attr
    )

def test_basic_model_creation():
    """Test basic DGDN model creation."""
    print("Testing basic model creation...")
    
    try:
        model = DynamicGraphDiffusionNet(
            node_dim=64,
            edge_dim=32,
            hidden_dim=128,
            num_layers=2,
            num_heads=4,
            diffusion_steps=3
        )
        print("✓ Model created successfully")
        return True
    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        return False

def test_model_forward_pass():
    """Test model forward pass with synthetic data."""
    print("Testing model forward pass...")
    
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
        
        # Create synthetic data
        data = create_synthetic_data(num_nodes=50, num_edges=150)
        
        # Set model to evaluation mode
        model.eval()
        
        # Forward pass
        with torch.no_grad():
            output = model(data)
        
        # Validate output structure
        required_keys = ['node_embeddings', 'mean', 'logvar', 'kl_loss', 'temporal_encoding']
        for key in required_keys:
            if key not in output:
                print(f"✗ Missing output key: {key}")
                return False
        
        # Validate tensor shapes
        node_embeddings = output['node_embeddings']
        if node_embeddings.shape != (50, 128):
            print(f"✗ Wrong node_embeddings shape: {node_embeddings.shape}")
            return False
        
        print("✓ Forward pass successful")
        return True
        
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_edge_prediction():
    """Test edge prediction functionality."""
    print("Testing edge prediction...")
    
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
        
        # Create synthetic data
        data = create_synthetic_data(num_nodes=50, num_edges=150)
        
        # Mock edge prediction (simplified)
        model.eval()
        with torch.no_grad():
            output = model(data)
            node_embeddings = output['node_embeddings']
            
            # Simple edge prediction test
            src_nodes = torch.tensor([0, 1, 2])
            tgt_nodes = torch.tensor([10, 11, 12])
            
            edge_predictions = model.edge_predictor(
                node_embeddings[src_nodes],
                node_embeddings[tgt_nodes]
            )
            
            if edge_predictions.shape != (3, 2):
                print(f"✗ Wrong edge prediction shape: {edge_predictions.shape}")
                return False
        
        print("✓ Edge prediction successful")
        return True
        
    except Exception as e:
        print(f"✗ Edge prediction failed: {e}")
        return False

def test_loss_computation():
    """Test loss computation."""
    print("Testing loss computation...")
    
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
        
        # Create synthetic data
        data = create_synthetic_data(num_nodes=50, num_edges=150)
        
        # Forward pass
        model.eval()
        with torch.no_grad():
            output = model(data)
            
            # Create dummy targets
            targets = torch.randint(0, 2, (150,))  # Binary edge targets
            
            # Compute loss
            losses = model.compute_loss(
                output=output,
                targets=targets,
                task="edge_prediction"
            )
            
            # Validate loss components
            required_loss_keys = ['reconstruction', 'kl_divergence', 'temporal_regularization', 'total']
            for key in required_loss_keys:
                if key not in losses:
                    print(f"✗ Missing loss component: {key}")
                    return False
                
                if not torch.is_tensor(losses[key]):
                    print(f"✗ Loss component {key} is not a tensor")
                    return False
        
        print("✓ Loss computation successful")
        return True
        
    except Exception as e:
        print(f"✗ Loss computation failed: {e}")
        return False

def run_generation_1_validation():
    """Run all Generation 1 validation tests."""
    print("=" * 60)
    print("GENERATION 1 VALIDATION - SIMPLE FUNCTIONALITY")
    print("=" * 60)
    
    tests = [
        test_basic_model_creation,
        test_model_forward_pass,
        test_edge_prediction,
        test_loss_computation
    ]
    
    results = []
    start_time = time.time()
    
    for test in tests:
        result = test()
        results.append(result)
        print()
    
    end_time = time.time()
    
    # Summary
    print("=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    
    passed = sum(results)
    total = len(results)
    
    print(f"Tests passed: {passed}/{total}")
    print(f"Success rate: {passed/total*100:.1f}%")
    print(f"Total time: {end_time - start_time:.2f}s")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED - Generation 1 is WORKING!")
        return True
    else:
        print("❌ Some tests failed - Generation 1 needs fixes")
        return False

if __name__ == "__main__":
    success = run_generation_1_validation()
    sys.exit(0 if success else 1)