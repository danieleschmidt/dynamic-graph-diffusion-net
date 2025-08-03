#!/usr/bin/env python3
"""
Basic usage example for Dynamic Graph Diffusion Net (DGDN).

This script demonstrates the core functionality of DGDN including:
- Creating synthetic temporal graph data
- Initializing and training a DGDN model
- Making predictions with uncertainty quantification
- Evaluating model performance
"""

import torch
import numpy as np
from dgdn import (
    DynamicGraphDiffusionNet, 
    TemporalDataset, 
    DGDNTrainer, 
    TemporalData
)


def create_synthetic_dataset(num_nodes=500, num_edges=2000, time_span=100):
    """Create a synthetic temporal graph dataset for demonstration."""
    print("Creating synthetic temporal graph dataset...")
    
    # Generate random edges with temporal information
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    timestamps = torch.sort(torch.rand(num_edges) * time_span)[0]
    
    # Generate node and edge features
    node_features = torch.randn(num_nodes, 64)
    edge_attr = torch.randn(num_edges, 32)
    
    # Create binary labels for edge prediction
    y = torch.randint(0, 2, (num_edges,)).float()
    
    # Create TemporalData object
    data = TemporalData(
        edge_index=edge_index,
        timestamps=timestamps,
        node_features=node_features,
        edge_attr=edge_attr,
        y=y,
        num_nodes=num_nodes
    )
    
    # Create dataset and split
    dataset = TemporalDataset(data, name="synthetic")
    train_data, val_data, test_data = dataset.split(
        ratios=[0.7, 0.15, 0.15],
        method="temporal"
    )
    
    print(f"Dataset created: {num_nodes} nodes, {num_edges} edges")
    print(f"Train: {len(train_data.data.timestamps)} edges")
    print(f"Val: {len(val_data.data.timestamps)} edges") 
    print(f"Test: {len(test_data.data.timestamps)} edges")
    
    return train_data, val_data, test_data


def initialize_model():
    """Initialize DGDN model with appropriate parameters."""
    print("\nInitializing DGDN model...")
    
    model = DynamicGraphDiffusionNet(
        node_dim=64,           # Input node feature dimension
        edge_dim=32,           # Input edge feature dimension
        time_dim=32,           # Temporal embedding dimension
        hidden_dim=128,        # Hidden representation dimension
        num_layers=3,          # Number of DGDN layers
        num_heads=4,           # Number of attention heads
        diffusion_steps=5,     # Number of diffusion steps
        aggregation="attention",
        dropout=0.1,
        activation="relu",
        layer_norm=True,
        max_time=100.0
    )
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Model initialized with {total_params:,} total parameters")
    print(f"Trainable parameters: {trainable_params:,}")
    
    return model


def train_model(model, train_data, val_data):
    """Train the DGDN model."""
    print("\nStarting model training...")
    
    # Initialize trainer
    trainer = DGDNTrainer(
        model=model,
        learning_rate=1e-3,
        weight_decay=1e-4,
        optimizer_type="adam",
        scheduler_type="cosine",
        diffusion_loss_weight=0.1,
        temporal_reg_weight=0.05,
        task="edge_prediction",
        log_dir="logs/basic_example",
        checkpoint_dir="checkpoints/basic_example"
    )
    
    # Train the model
    history = trainer.fit(
        train_data=train_data,
        val_data=val_data,
        epochs=50,
        batch_size=16,
        early_stopping_patience=10,
        save_best=True,
        verbose=True
    )
    
    print("Training completed!")
    return trainer, history


def evaluate_model(trainer, test_data):
    """Evaluate the trained model."""
    print("\nEvaluating model on test data...")
    
    # Evaluate on test set
    test_metrics = trainer.evaluate(test_data, batch_size=32)
    
    print("Test Results:")
    for metric, value in test_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    return test_metrics


def demonstrate_predictions(model, test_data):
    """Demonstrate model predictions with uncertainty quantification."""
    print("\nDemonstrating predictions with uncertainty...")
    
    # Get a sample from test data
    sample_data = test_data.data.time_window(0, 50)  # First 50 time units
    
    model.eval()
    with torch.no_grad():
        # Make predictions
        output = model(sample_data, return_uncertainty=True)
        
        node_embeddings = output["node_embeddings"]
        uncertainty = output["uncertainty"]
        
        print(f"Generated embeddings for {node_embeddings.shape[0]} nodes")
        print(f"Embedding dimension: {node_embeddings.shape[1]}")
        print(f"Mean uncertainty: {uncertainty.mean().item():.4f}")
        print(f"Std uncertainty: {uncertainty.std().item():.4f}")
        
        # Demonstrate edge prediction
        if sample_data.edge_index.shape[1] > 0:
            src_nodes = sample_data.edge_index[0][:5]  # First 5 edges
            tgt_nodes = sample_data.edge_index[1][:5]
            
            edge_predictions = model.predict_edges(
                src_nodes, tgt_nodes, time=25.0, data=sample_data
            )
            
            print(f"\nEdge prediction probabilities (first 5 edges):")
            for i, prob in enumerate(edge_predictions[:, 1]):  # Positive class probability
                print(f"  Edge {src_nodes[i].item()} -> {tgt_nodes[i].item()}: {prob.item():.4f}")
    
    return output


def main():
    """Main demonstration function."""
    print("="*60)
    print("DGDN (Dynamic Graph Diffusion Network) - Basic Usage Demo")
    print("="*60)
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    try:
        # 1. Create synthetic dataset
        train_data, val_data, test_data = create_synthetic_dataset()
        
        # 2. Initialize model
        model = initialize_model()
        
        # 3. Train model
        trainer, history = train_model(model, train_data, val_data)
        
        # 4. Evaluate model
        test_metrics = evaluate_model(trainer, test_data)
        
        # 5. Demonstrate predictions
        predictions = demonstrate_predictions(model, test_data)
        
        print("\n" + "="*60)
        print("Demo completed successfully!")
        print("Key features demonstrated:")
        print("✓ Temporal graph data creation and handling")
        print("✓ DGDN model initialization and training")
        print("✓ Variational diffusion with uncertainty quantification")
        print("✓ Edge prediction with temporal encoding")
        print("✓ Comprehensive evaluation metrics")
        print("="*60)
        
    except Exception as e:
        print(f"\nError during demo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()