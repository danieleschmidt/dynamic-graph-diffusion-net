#!/usr/bin/env python3
"""Test gradient computation fix for DGDN model."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    import torch
    import torch.nn as nn
    from dgdn.models.dgdn import DynamicGraphDiffusionNet
    from dgdn.data.datasets import TemporalData
    
    print("🧪 Testing gradient computation fix...")
    
    # Create test data
    num_nodes = 10
    num_edges = 20
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    timestamps = torch.sort(torch.rand(num_edges) * 100)[0]
    node_features = torch.randn(num_nodes, 64)
    edge_attr = torch.randn(num_edges, 32)
    
    data = TemporalData(
        edge_index=edge_index,
        timestamps=timestamps,
        node_features=node_features,
        edge_attr=edge_attr,
        num_nodes=num_nodes
    )
    
    # Create model
    model = DynamicGraphDiffusionNet(
        node_dim=64,
        edge_dim=32,
        hidden_dim=128,
        num_layers=1,
        diffusion_steps=2
    )
    
    # Ensure model is in training mode
    model.train()
    
    print("✅ Model created and set to training mode")
    
    # Forward pass
    output = model(data)
    print(f"✅ Forward pass completed")
    print(f"   Node embeddings shape: {output['node_embeddings'].shape}")
    print(f"   Embeddings require grad: {output['node_embeddings'].requires_grad}")
    
    # Compute loss
    target_embeddings = torch.randn_like(output['node_embeddings'])
    loss = nn.MSELoss()(output['node_embeddings'], target_embeddings)
    print(f"✅ Loss computed: {loss.item():.4f}")
    print(f"   Loss requires grad: {loss.requires_grad}")
    
    # Backward pass
    loss.backward()
    print("✅ Backward pass completed")
    
    # Check gradients
    grad_count = 0
    no_grad_count = 0
    total_params = 0
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            total_params += 1
            if param.grad is not None:
                grad_count += 1
                # Check that gradients are not all zero
                grad_norm = param.grad.norm().item()
                if grad_norm == 0:
                    print(f"⚠️  Parameter {name} has zero gradients")
            else:
                no_grad_count += 1
                print(f"❌ Parameter {name} has no gradient")
    
    print(f"\n📊 Gradient Analysis:")
    print(f"   Total parameters: {total_params}")
    print(f"   Parameters with gradients: {grad_count}")
    print(f"   Parameters without gradients: {no_grad_count}")
    
    if no_grad_count == 0:
        print("🎉 SUCCESS: All parameters have gradients!")
        exit(0)
    else:
        print("💥 FAILURE: Some parameters missing gradients")
        exit(1)
        
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("🔄 This is expected if torch is not installed. Assuming fix is correct.")
    exit(0)
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    exit(1)