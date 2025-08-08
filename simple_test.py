#!/usr/bin/env python3
"""Simple test script to validate DGDN core functionality."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import torch
import torch.nn as nn
import numpy as np

# Test core temporal encoding module
print("🧪 Testing EdgeTimeEncoder...")
from dgdn.temporal.encoding import EdgeTimeEncoder

time_encoder = EdgeTimeEncoder(time_dim=32, max_time=1000.0)
timestamps = torch.tensor([10.0, 25.5, 100.0, 500.0])
temporal_embeddings = time_encoder(timestamps)
print(f"✅ EdgeTimeEncoder: {temporal_embeddings.shape} -> {temporal_embeddings.dtype}")

# Test variational diffusion module
print("🧪 Testing VariationalDiffusion...")
from dgdn.temporal.diffusion import VariationalDiffusion

diffusion = VariationalDiffusion(hidden_dim=128, num_diffusion_steps=3, num_heads=4)
dummy_x = torch.randn(10, 128)  # 10 nodes, 128 features
dummy_edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]])  # Simple cycle
diffusion_output = diffusion(dummy_x, dummy_edge_index)
print(f"✅ VariationalDiffusion: z={diffusion_output['z'].shape}, kl_loss={diffusion_output['kl_loss'].item():.4f}")

# Test core model layers
print("🧪 Testing DGDNLayer...")
from dgdn.models.layers import DGDNLayer

layer = DGDNLayer(hidden_dim=128, time_dim=32, num_heads=4, num_diffusion_steps=3)
layer_output = layer(
    x=dummy_x,
    edge_index=dummy_edge_index, 
    temporal_encoding=temporal_embeddings[:4]  # Match edge count
)
print(f"✅ DGDNLayer: {layer_output['node_features'].shape}")

# Test basic TemporalData structure
print("🧪 Testing TemporalData...")
from dgdn.data.datasets import TemporalData

temporal_data = TemporalData(
    edge_index=dummy_edge_index,
    timestamps=timestamps[:4],  # Match edge count
    num_nodes=10
)
print(f"✅ TemporalData: {temporal_data.num_nodes} nodes, {temporal_data.edge_index.shape[1]} edges")

print("\n🎉 Generation 1 Basic Functionality Test: PASSED")
print("✅ Core modules import and execute successfully")
print("✅ Temporal encoding works")
print("✅ Variational diffusion works") 
print("✅ DGDN layers work")
print("✅ Data structures work")