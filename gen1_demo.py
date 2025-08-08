#!/usr/bin/env python3
"""Generation 1 Demo: MAKE IT WORK (Simple)
Demonstrates basic DGDN functionality with edge prediction task.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import torch
import numpy as np
from dgdn.data.datasets import TemporalData
from dgdn.models.dgdn import DynamicGraphDiffusionNet

def create_synthetic_temporal_graph(num_nodes=50, num_edges=200, time_span=100.0):
    """Create a synthetic temporal graph for testing."""
    print(f"🏗️  Creating synthetic temporal graph...")
    print(f"   Nodes: {num_nodes}, Edges: {num_edges}, Time span: {time_span}")
    
    # Generate random edges
    source_nodes = torch.randint(0, num_nodes, (num_edges,))
    target_nodes = torch.randint(0, num_nodes, (num_edges,))
    edge_index = torch.stack([source_nodes, target_nodes], dim=0)
    
    # Generate random timestamps sorted
    timestamps = torch.sort(torch.rand(num_edges) * time_span)[0]
    
    # Generate node features
    node_features = torch.randn(num_nodes, 64)  # 64-dim node features
    
    # Create temporal data
    data = TemporalData(
        edge_index=edge_index,
        timestamps=timestamps,
        node_features=node_features,
        num_nodes=num_nodes
    )
    
    print(f"✅ Created temporal graph: {num_nodes} nodes, {num_edges} edges")
    return data

def test_dgdn_model():
    """Test the complete DGDN model."""
    print("\n🧠 Testing DynamicGraphDiffusionNet...")
    
    # Create synthetic data
    data = create_synthetic_temporal_graph(num_nodes=50, num_edges=200)
    
    # Initialize DGDN model
    model = DynamicGraphDiffusionNet(
        node_dim=64,          # Input node feature dimension
        edge_dim=0,           # No edge features for simplicity
        time_dim=32,          # Temporal embedding dimension
        hidden_dim=128,       # Hidden layer dimension
        num_layers=2,         # Number of DGDN layers
        num_heads=4,          # Number of attention heads
        diffusion_steps=3,    # Number of diffusion steps
        dropout=0.1
    )
    
    print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Forward pass
    print("   Running forward pass...")
    with torch.no_grad():
        output = model(data, return_attention=True, return_uncertainty=True)
    
    print(f"✅ Forward pass successful!")
    print(f"   Node embeddings: {output['node_embeddings'].shape}")
    print(f"   KL loss: {output['kl_loss'].item():.4f}")
    print(f"   Uncertainty available: {'uncertainty' in output}")
    print(f"   Attention weights: {len(output['attention_weights'])} layers")
    
    return model, data, output

def test_edge_prediction():
    """Test edge prediction functionality."""
    print("\n🔗 Testing Edge Prediction...")
    
    model, data, _ = test_dgdn_model()
    
    # Test edge prediction between random node pairs
    num_predictions = 10
    source_nodes = torch.randint(0, data.num_nodes, (num_predictions,))
    target_nodes = torch.randint(0, data.num_nodes, (num_predictions,))
    prediction_time = 50.0
    
    print(f"   Predicting {num_predictions} edges at time {prediction_time}")
    
    with torch.no_grad():
        edge_probs = model.predict_edges(
            source_nodes=source_nodes,
            target_nodes=target_nodes,
            time=prediction_time,
            data=data,
            return_probs=True
        )
    
    print(f"✅ Edge prediction successful!")
    print(f"   Predictions shape: {edge_probs.shape}")
    print(f"   Sample probabilities: {edge_probs[:3, 1].tolist()}")  # Positive class probs
    
    return edge_probs

def test_training_step():
    """Test a basic training step."""
    print("\n🎯 Testing Training Step...")
    
    # Create fresh model and data for training
    data = create_synthetic_temporal_graph(num_nodes=50, num_edges=200)
    
    model = DynamicGraphDiffusionNet(
        node_dim=64, edge_dim=0, time_dim=32, hidden_dim=128,
        num_layers=2, num_heads=4, diffusion_steps=3, dropout=0.1
    )
    
    # Enable training mode and gradients
    model.train()
    
    # Forward pass WITH gradients
    output = model(data, return_attention=False, return_uncertainty=False)
    
    # Create dummy targets for edge prediction
    num_edges = data.edge_index.shape[1]
    edge_targets = torch.randint(0, 2, (num_edges,))  # Binary targets
    
    # Compute loss
    losses = model.compute_loss(
        output=output,
        targets=edge_targets,
        task="edge_prediction",
        beta_kl=0.1,
        beta_temporal=0.05
    )
    
    print(f"✅ Loss computation successful!")
    print(f"   Total loss: {losses['total'].item():.4f}")
    print(f"   Reconstruction: {losses['reconstruction'].item():.4f}")
    print(f"   KL divergence: {losses['kl_divergence'].item():.4f}")
    print(f"   Temporal reg: {losses['temporal_regularization'].item():.4f}")
    
    # Test backward pass
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    losses['total'].backward()
    optimizer.step()
    optimizer.zero_grad()
    
    print(f"✅ Backward pass and optimization successful!")
    
    return losses

def main():
    """Run Generation 1 comprehensive demo."""
    print("🚀 DGDN Generation 1 Demo: MAKE IT WORK")
    print("=" * 50)
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    try:
        # Test core model functionality
        test_dgdn_model()
        
        # Test edge prediction
        test_edge_prediction()
        
        # Test training step
        test_training_step()
        
        print("\n" + "=" * 50)
        print("🎉 GENERATION 1 COMPLETE: BASIC FUNCTIONALITY VERIFIED")
        print("✅ Model imports and initializes correctly")
        print("✅ Forward pass works with uncertainty quantification")
        print("✅ Edge prediction functionality works")
        print("✅ Loss computation and backpropagation work")
        print("✅ Ready for Generation 2: Robustness enhancements")
        
    except Exception as e:
        print(f"\n❌ Error in Generation 1 demo: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)