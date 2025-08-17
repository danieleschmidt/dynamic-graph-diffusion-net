#!/usr/bin/env python3
"""
Generation 1: Core DGDN Functionality Validation
Test basic DGDN model functionality with minimal viable implementation.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def create_test_data(num_nodes: int = 100, num_edges: int = 200) -> Dict[str, Any]:
    """Create synthetic temporal graph data for testing."""
    # Generate random edge connections
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    
    # Generate random timestamps (sorted for realism)
    timestamps = torch.sort(torch.rand(num_edges) * 100.0)[0]
    
    # Generate node features
    node_features = torch.randn(num_nodes, 64)
    
    # Create simple data object
    class TemporalData:
        def __init__(self):
            self.edge_index = edge_index
            self.timestamps = timestamps
            self.node_features = node_features
            self.num_nodes = num_nodes
            
        def time_window(self, start_time, end_time):
            """Get edges within time window."""
            mask = (self.timestamps >= start_time) & (self.timestamps <= end_time)
            new_data = TemporalData()
            new_data.edge_index = self.edge_index[:, mask]
            new_data.timestamps = self.timestamps[mask]
            new_data.node_features = self.node_features
            new_data.num_nodes = self.num_nodes
            return new_data
    
    return TemporalData()

def test_core_dgdn():
    """Test core DGDN functionality."""
    print("🧪 Testing Core DGDN Functionality...")
    
    try:
        # Import the model
        from dgdn.models.dgdn import DynamicGraphDiffusionNet
        print("✅ Successfully imported DynamicGraphDiffusionNet")
        
        # Create test data
        data = create_test_data(num_nodes=50, num_edges=100)
        print(f"✅ Created test data: {data.num_nodes} nodes, {data.edge_index.shape[1]} edges")
        
        # Initialize model
        model = DynamicGraphDiffusionNet(
            node_dim=64,
            hidden_dim=128,
            time_dim=32,
            num_layers=2,
            num_heads=4,
            diffusion_steps=3,
            dropout=0.1
        )
        print("✅ Successfully initialized DGDN model")
        
        # Test forward pass
        model.eval()
        with torch.no_grad():
            output = model(data)
            
        print(f"✅ Forward pass successful!")
        print(f"   - Node embeddings shape: {output['node_embeddings'].shape}")
        print(f"   - Mean shape: {output['mean'].shape}")
        print(f"   - Logvar shape: {output['logvar'].shape}")
        print(f"   - KL loss: {output['kl_loss'].item():.4f}")
        
        # Test edge prediction
        src_nodes = torch.randint(0, data.num_nodes, (10,))
        tgt_nodes = torch.randint(0, data.num_nodes, (10,))
        edge_probs = model.predict_edges(src_nodes, tgt_nodes, time=50.0, data=data)
        print(f"✅ Edge prediction successful! Shape: {edge_probs.shape}")
        
        # Test training mode
        model.train()
        output_train = model(data)
        
        # Simple loss computation
        dummy_targets = torch.randint(0, 2, (data.num_nodes,))
        losses = model.compute_loss(output_train, dummy_targets, task="node_classification")
        
        print(f"✅ Loss computation successful!")
        print(f"   - Total loss: {losses['total'].item():.4f}")
        print(f"   - Reconstruction: {losses['reconstruction'].item():.4f}")
        print(f"   - KL divergence: {losses['kl_divergence'].item():.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in core DGDN test: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_temporal_encoding():
    """Test temporal encoding functionality."""
    print("\n🕐 Testing Temporal Encoding...")
    
    try:
        from dgdn.temporal.encoding import EdgeTimeEncoder
        
        # Test EdgeTimeEncoder
        encoder = EdgeTimeEncoder(time_dim=32, num_bases=64, max_time=1000.0)
        timestamps = torch.rand(100) * 500.0
        
        encoding = encoder(timestamps)
        print(f"✅ EdgeTimeEncoder successful! Shape: {encoding.shape}")
        
        # Test time range encoding
        range_encoding = encoder.get_time_range_encoding(0, 100, num_steps=50)
        print(f"✅ Time range encoding successful! Shape: {range_encoding.shape}")
        
        # Test temporal similarity
        time1 = torch.tensor([10.0, 20.0, 30.0])
        time2 = torch.tensor([11.0, 22.0, 35.0])
        similarity = encoder.compute_temporal_similarity(time1, time2)
        print(f"✅ Temporal similarity computation successful! Shape: {similarity.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in temporal encoding test: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_variational_diffusion():
    """Test variational diffusion functionality."""
    print("\n🌀 Testing Variational Diffusion...")
    
    try:
        from dgdn.temporal.diffusion import VariationalDiffusion
        
        # Create diffusion module
        diffusion = VariationalDiffusion(
            hidden_dim=128,
            num_diffusion_steps=3,
            num_heads=4,
            dropout=0.1
        )
        
        # Create test data
        num_nodes = 50
        x = torch.randn(num_nodes, 128)
        edge_index = torch.randint(0, num_nodes, (2, 100))
        
        # Test forward pass
        output = diffusion(x, edge_index)
        print(f"✅ Variational diffusion forward pass successful!")
        print(f"   - Output z shape: {output['z'].shape}")
        print(f"   - Mean shape: {output['mean'].shape}")
        print(f"   - Logvar shape: {output['logvar'].shape}")
        print(f"   - KL loss: {output['kl_loss'].item():.4f}")
        
        # Test uncertainty computation
        uncertainty = diffusion.get_uncertainty(output['logvar'])
        print(f"✅ Uncertainty computation successful! Shape: {uncertainty.shape}")
        
        # Test sampling
        sampled = diffusion.sample(num_nodes, edge_index)
        print(f"✅ Sampling successful! Shape: {sampled.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in variational diffusion test: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_model_layers():
    """Test model layer functionality."""
    print("\n🏗️ Testing Model Layers...")
    
    try:
        from dgdn.models.layers import DGDNLayer, GraphNorm, MultiHeadTemporalAttention
        
        # Test DGDNLayer
        layer = DGDNLayer(
            hidden_dim=128,
            time_dim=32,
            num_heads=4,
            num_diffusion_steps=2,
            dropout=0.1
        )
        
        num_nodes = 50
        x = torch.randn(num_nodes, 128)
        edge_index = torch.randint(0, num_nodes, (2, 100))
        temporal_encoding = torch.randn(100, 32)
        
        output = layer(x, edge_index, temporal_encoding)
        print(f"✅ DGDNLayer successful!")
        print(f"   - Node features shape: {output['node_features'].shape}")
        print(f"   - Attention weights shape: {output['attention_weights'].shape}")
        
        # Test GraphNorm
        norm = GraphNorm(128)
        normalized = norm(x)
        print(f"✅ GraphNorm successful! Shape: {normalized.shape}")
        
        # Test MultiHeadTemporalAttention
        attention = MultiHeadTemporalAttention(hidden_dim=128, num_heads=4, temporal_dim=32)
        attn_out, attn_weights = attention(x, x, x, temporal_encoding, edge_index)
        print(f"✅ MultiHeadTemporalAttention successful!")
        print(f"   - Attention output shape: {attn_out.shape}")
        print(f"   - Attention weights shape: {attn_weights.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in model layers test: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def run_generation1_validation():
    """Run complete Generation 1 validation."""
    print("=" * 60)
    print("🚀 DGDN Generation 1: Core Functionality Validation")
    print("=" * 60)
    
    tests = [
        ("Temporal Encoding", test_temporal_encoding),
        ("Variational Diffusion", test_variational_diffusion),
        ("Model Layers", test_model_layers),
        ("Core DGDN Model", test_core_dgdn),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*40}")
        print(f"Running: {test_name}")
        print(f"{'='*40}")
        
        success = test_func()
        results.append((test_name, success))
    
    # Summary
    print(f"\n{'='*60}")
    print("🎯 Generation 1 Validation Summary")
    print(f"{'='*60}")
    
    total_tests = len(results)
    passed_tests = sum(1 for _, success in results if success)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")
    
    success_rate = (passed_tests / total_tests) * 100
    print(f"\nSuccess Rate: {passed_tests}/{total_tests} ({success_rate:.1f}%)")
    
    if success_rate >= 75:
        print("🎉 Generation 1 VALIDATION SUCCESSFUL!")
        print("Ready to proceed to Generation 2 (Robustness)")
        return True
    else:
        print("⚠️  Generation 1 needs fixes before proceeding")
        return False

if __name__ == "__main__":
    success = run_generation1_validation()
    sys.exit(0 if success else 1)