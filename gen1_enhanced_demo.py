#!/usr/bin/env python3
"""
Generation 1: MAKE IT WORK (Enhanced) - DGDN Basic Functionality Demo

This demo showcases the core DGDN functionality with proper error handling
and comprehensive testing of all major components.
"""

import torch
import numpy as np
from typing import Dict, Any
import sys
import os

# Add src to path for imports
sys.path.insert(0, 'src')

import dgdn
from dgdn import DynamicGraphDiffusionNet, TemporalData, TemporalDataset
from dgdn.models.layers import DGDNLayer, MultiHeadTemporalAttention, GraphNorm
from dgdn.temporal import EdgeTimeEncoder, VariationalDiffusion


def create_synthetic_temporal_graph(num_nodes=100, num_edges=500, time_span=100.0, node_dim=64, edge_dim=32):
    """Create synthetic temporal graph for testing."""
    print(f"🔧 Creating synthetic temporal graph: {num_nodes} nodes, {num_edges} edges")
    
    # Generate temporal edges
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    timestamps = torch.sort(torch.rand(num_edges) * time_span)[0]
    
    # Node and edge features (use parameters from function args)
    node_features = torch.randn(num_nodes, node_dim)
    edge_attr = torch.randn(num_edges, edge_dim)
    
    # Create labels for edge prediction
    y = torch.randint(0, 2, (num_edges,)).float()
    
    data = TemporalData(
        edge_index=edge_index,
        timestamps=timestamps,
        edge_attr=edge_attr,
        node_features=node_features,
        y=y,
        num_nodes=num_nodes
    )
    
    print(f"✅ Temporal graph created successfully")
    stats = data.get_temporal_statistics()
    print(f"   Time span: {stats['time_span']:.2f}")
    print(f"   Temporal density: {stats['temporal_density']:.3f}")
    
    return data


def test_temporal_encoding():
    """Test edge-time encoding functionality."""
    print("\n🧪 Testing Edge-Time Encoding...")
    
    try:
        # Test basic encoder
        encoder = EdgeTimeEncoder(time_dim=32, num_bases=64, max_time=1000.0)
        
        # Test with various timestamp patterns
        timestamps = torch.tensor([0.0, 10.0, 25.5, 50.0, 100.0])
        encoding = encoder(timestamps)
        
        assert encoding.shape == (5, 32), f"Expected shape (5, 32), got {encoding.shape}"
        print(f"✅ Basic encoding test passed: {encoding.shape}")
        
        # Test empty input
        empty_encoding = encoder(torch.tensor([]))
        assert empty_encoding.shape == (0, 32), f"Empty input test failed"
        print(f"✅ Empty input test passed")
        
        # Test frequency response
        frequencies, phases = encoder.get_frequency_response()
        print(f"✅ Learned frequencies range: {frequencies.min():.3f} - {frequencies.max():.3f}")
        
        # Test temporal similarity
        time1 = torch.tensor([10.0, 20.0])
        time2 = torch.tensor([10.5, 19.5]) 
        similarity = encoder.compute_temporal_similarity(time1, time2)
        print(f"✅ Temporal similarity computed: {similarity}")
        
    except Exception as e:
        print(f"❌ Temporal encoding test failed: {e}")
        raise


def test_variational_diffusion():
    """Test variational diffusion functionality."""
    print("\n🧪 Testing Variational Diffusion...")
    
    try:
        # Create test data
        num_nodes, hidden_dim = 50, 128
        x = torch.randn(num_nodes, hidden_dim)
        edge_index = torch.randint(0, num_nodes, (2, 100))
        
        # Initialize diffusion model
        diffusion = VariationalDiffusion(
            hidden_dim=hidden_dim,
            num_diffusion_steps=3,
            num_heads=8,
            dropout=0.1
        )
        
        # Test forward pass
        output = diffusion(x, edge_index, return_all_steps=True)
        
        required_keys = ['z', 'mean', 'logvar', 'kl_loss', 'all_steps']
        for key in required_keys:
            assert key in output, f"Missing key: {key}"
        
        assert output['z'].shape == x.shape, f"Output shape mismatch"
        assert len(output['all_steps']) == 3, f"Wrong number of diffusion steps"
        
        print(f"✅ Diffusion forward pass successful")
        print(f"   KL loss: {output['kl_loss']:.4f}")
        
        # Test uncertainty quantification
        uncertainty = diffusion.get_uncertainty(output['logvar'])
        print(f"✅ Uncertainty computed: mean={uncertainty.mean():.4f}")
        
        # Test sampling
        sampled = diffusion.sample(num_nodes, edge_index)
        assert sampled.shape == x.shape, f"Sampling shape mismatch"
        print(f"✅ Sampling test passed")
        
    except Exception as e:
        print(f"❌ Variational diffusion test failed: {e}")
        raise


def test_dgdn_layer():
    """Test individual DGDN layer functionality."""
    print("\n🧪 Testing DGDN Layer...")
    
    try:
        num_nodes, hidden_dim, time_dim = 50, 128, 32
        x = torch.randn(num_nodes, hidden_dim)
        edge_index = torch.randint(0, num_nodes, (2, 100))
        temporal_encoding = torch.randn(100, time_dim)
        
        # Create DGDN layer
        layer = DGDNLayer(
            hidden_dim=hidden_dim,
            time_dim=time_dim,
            num_heads=8,
            num_diffusion_steps=3,
            dropout=0.1
        )
        
        # Forward pass
        output = layer(x, edge_index, temporal_encoding)
        
        required_keys = ['node_features', 'attention_weights', 'diffusion_features']
        for key in required_keys:
            assert key in output, f"Missing key: {key}"
        
        assert output['node_features'].shape == x.shape, f"Node features shape mismatch"
        print(f"✅ DGDN layer forward pass successful")
        print(f"   Output shape: {output['node_features'].shape}")
        print(f"   Attention weights: {output['attention_weights'].shape}")
        
    except Exception as e:
        print(f"❌ DGDN layer test failed: {e}")
        raise


def test_full_dgdn_model():
    """Test complete DGDN model."""
    print("\n🧪 Testing Full DGDN Model...")
    
    try:
        # Create model
        model = DynamicGraphDiffusionNet(
            node_dim=64,
            edge_dim=32,
            time_dim=32,
            hidden_dim=128,
            num_layers=2,
            num_heads=8,
            diffusion_steps=3,
            dropout=0.1
        )
        
        print(f"✅ Model created with {sum(p.numel() for p in model.parameters())} parameters")
        
        # Create test data
        data = create_synthetic_temporal_graph(num_nodes=50, num_edges=200, time_span=50.0)
        
        # Forward pass
        model.eval()
        with torch.no_grad():
            output = model(data, return_attention=True, return_uncertainty=True)
        
        # Verify output structure
        required_keys = ['node_embeddings', 'mean', 'logvar', 'kl_loss', 'temporal_encoding', 'attention_weights', 'uncertainty']
        for key in required_keys:
            assert key in output, f"Missing output key: {key}"
        
        print(f"✅ Forward pass successful")
        print(f"   Node embeddings: {output['node_embeddings'].shape}")
        print(f"   KL loss: {output['kl_loss']:.4f}")
        print(f"   Mean uncertainty: {output['uncertainty'].mean():.4f}")
        
        # Test edge prediction
        src_nodes = torch.tensor([0, 1, 2])
        tgt_nodes = torch.tensor([10, 11, 12])
        edge_probs = model.predict_edges(src_nodes, tgt_nodes, time=25.0, data=data)
        print(f"✅ Edge prediction: {edge_probs.shape}")
        
        # Test node classification
        node_ids = torch.tensor([0, 1, 2, 3, 4])
        node_preds = model.predict_nodes(node_ids, time=25.0, data=data)
        print(f"✅ Node classification: {node_preds.shape}")
        
        # Test loss computation
        targets = torch.randint(0, 2, (data.edge_index.shape[1],))
        losses = model.compute_loss(output, targets, task="edge_prediction")
        print(f"✅ Loss computation: total={losses['total']:.4f}")
        
    except Exception as e:
        print(f"❌ Full DGDN model test failed: {e}")
        raise


def test_dataset_functionality():
    """Test dataset creation and manipulation."""
    print("\n🧪 Testing Dataset Functionality...")
    
    try:
        # Create synthetic dataset
        dataset = TemporalDataset.load("synthetic")
        print(f"✅ Synthetic dataset loaded: {dataset.name}")
        
        # Get statistics
        stats = dataset.get_statistics()
        print(f"   Nodes: {stats['num_nodes']}, Edges: {stats['num_edges']}")
        print(f"   Time span: {stats['time_span']:.2f}")
        
        # Test split
        train_data, val_data, test_data = dataset.split(method="temporal")
        print(f"✅ Dataset split completed:")
        print(f"   Train: {train_data.data.timestamps.shape[0]} edges")
        print(f"   Val: {val_data.data.timestamps.shape[0]} edges") 
        print(f"   Test: {test_data.data.timestamps.shape[0]} edges")
        
        # Test time window extraction
        window_data = dataset.data.time_window(0, 500)
        print(f"✅ Time window extraction: {window_data.timestamps.shape[0]} edges")
        
        # Test snapshot creation
        snapshot = dataset.data.get_snapshot(250.0)
        print(f"✅ Snapshot creation: {snapshot.edge_index.shape[1]} edges")
        
    except Exception as e:
        print(f"❌ Dataset functionality test failed: {e}")
        raise


def performance_benchmark():
    """Run basic performance benchmarks."""
    print("\n📊 Running Performance Benchmarks...")
    
    try:
        import time
        
        # Create larger model for benchmarking
        model = DynamicGraphDiffusionNet(
            node_dim=128,
            edge_dim=64,
            hidden_dim=256,
            num_layers=3,
            num_heads=8,
            diffusion_steps=5
        )
        
        # Create larger test data (match model's node_dim)
        data = create_synthetic_temporal_graph(num_nodes=500, num_edges=2000, time_span=200.0, node_dim=128, edge_dim=64)
        
        model.eval()
        
        # Warm-up
        with torch.no_grad():
            _ = model(data)
        
        # Benchmark inference
        start_time = time.time()
        num_runs = 10
        
        with torch.no_grad():
            for _ in range(num_runs):
                output = model(data)
        
        avg_time = (time.time() - start_time) / num_runs
        
        print(f"✅ Performance benchmark completed:")
        print(f"   Average inference time: {avg_time*1000:.1f}ms")
        print(f"   Throughput: {data.num_nodes/avg_time:.0f} nodes/second")
        print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Memory usage estimate
        total_params = sum(p.numel() for p in model.parameters())
        param_size_mb = total_params * 4 / (1024 * 1024)  # 4 bytes per float32
        print(f"   Model size: {param_size_mb:.1f} MB")
        
    except Exception as e:
        print(f"❌ Performance benchmark failed: {e}")
        raise


def main():
    """Run Generation 1 enhanced demo."""
    print("🚀 DGDN Generation 1: MAKE IT WORK (Enhanced)")
    print("=" * 60)
    
    try:
        # Set random seed for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)
        
        # Check DGDN installation
        print(f"📦 DGDN version: {dgdn.__version__}")
        print(f"🔧 PyTorch version: {torch.__version__}")
        
        # Run all tests
        test_temporal_encoding()
        test_variational_diffusion()
        test_dgdn_layer()
        test_full_dgdn_model()
        test_dataset_functionality()
        performance_benchmark()
        
        print("\n🎉 Generation 1 Demo Completed Successfully!")
        print("=" * 60)
        print("✅ All core functionality is working correctly")
        print("✅ Temporal encoding operational")
        print("✅ Variational diffusion functional")
        print("✅ DGDN layers processing correctly")
        print("✅ Full model inference working")
        print("✅ Dataset handling operational")
        print("✅ Performance benchmarks completed")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Generation 1 Demo Failed: {e}")
        print("=" * 60)
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)