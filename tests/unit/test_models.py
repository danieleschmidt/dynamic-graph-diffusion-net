"""Unit tests for DGDN models."""

import pytest
import torch
import torch.nn as nn
from dgdn.models import DynamicGraphDiffusionNet, DGDNLayer, MultiHeadTemporalAttention
from dgdn.data import TemporalData


class TestDynamicGraphDiffusionNet:
    """Test cases for the main DGDN model."""
    
    def test_model_initialization(self):
        """Test that the model initializes correctly."""
        model = DynamicGraphDiffusionNet(
            node_dim=64,
            edge_dim=32,
            hidden_dim=128,
            num_layers=2,
            time_dim=16,
            num_heads=4
        )
        
        assert model.node_dim == 64
        assert model.edge_dim == 32
        assert model.hidden_dim == 128
        assert model.num_layers == 2
        assert model.time_dim == 16
        assert model.num_heads == 4
        
        # Check that submodules are initialized
        assert hasattr(model, 'time_encoder')
        assert hasattr(model, 'dgdn_layers')
        assert hasattr(model, 'variational_diffusion')
        assert len(model.dgdn_layers) == 2
    
    def test_forward_pass(self):
        """Test forward pass with synthetic data."""
        model = DynamicGraphDiffusionNet(
            node_dim=32,
            edge_dim=16,
            hidden_dim=64,
            num_layers=2,
            time_dim=8,
            diffusion_steps=3
        )
        
        # Create synthetic temporal data
        num_nodes, num_edges = 10, 20
        edge_index = torch.randint(0, num_nodes, (2, num_edges))
        timestamps = torch.sort(torch.rand(num_edges) * 100)[0]
        node_features = torch.randn(num_nodes, 32)
        edge_attr = torch.randn(num_edges, 16)
        
        data = TemporalData(
            edge_index=edge_index,
            timestamps=timestamps,
            node_features=node_features,
            edge_attr=edge_attr,
            num_nodes=num_nodes
        )
        
        # Forward pass
        model.eval()
        with torch.no_grad():
            output = model(data, return_uncertainty=True)
        
        # Check output structure
        assert "node_embeddings" in output
        assert "mean" in output
        assert "logvar" in output
        assert "kl_loss" in output
        assert "uncertainty" in output
        
        # Check tensor shapes
        assert output["node_embeddings"].shape == (num_nodes, 64)
        assert output["mean"].shape == (num_nodes, 64)
        assert output["logvar"].shape == (num_nodes, 64)
        assert output["uncertainty"].shape == (num_nodes, 64)
        
        # Check that KL loss is a scalar
        assert output["kl_loss"].dim() == 0
    
    def test_edge_prediction(self):
        """Test edge prediction functionality."""
        model = DynamicGraphDiffusionNet(
            node_dim=32,
            hidden_dim=64,
            num_layers=1
        )
        
        # Create test data
        num_nodes = 20
        data = self._create_test_data(num_nodes)
        
        # Test edge prediction
        src_nodes = torch.tensor([0, 1, 2])
        tgt_nodes = torch.tensor([5, 6, 7])
        
        predictions = model.predict_edges(
            src_nodes, tgt_nodes, time=50.0, data=data
        )
        
        assert predictions.shape == (3, 2)  # 3 edges, 2 classes (binary)
        assert torch.all(predictions >= 0) and torch.all(predictions <= 1)  # Probabilities
    
    def test_node_embeddings(self):
        """Test node embedding extraction."""
        model = DynamicGraphDiffusionNet(
            node_dim=32,
            hidden_dim=64,
            num_layers=1
        )
        
        data = self._create_test_data(15)
        node_ids = [0, 5, 10]
        
        embeddings = model.get_node_embeddings(node_ids, time=30.0, data=data)
        
        assert embeddings.shape == (3, 64)
    
    def _create_test_data(self, num_nodes=10, num_edges=20):
        """Helper to create test temporal data."""
        edge_index = torch.randint(0, num_nodes, (2, num_edges))
        timestamps = torch.sort(torch.rand(num_edges) * 100)[0]
        node_features = torch.randn(num_nodes, 32)
        
        return TemporalData(
            edge_index=edge_index,
            timestamps=timestamps,
            node_features=node_features,
            num_nodes=num_nodes
        )


class TestDGDNLayer:
    """Test cases for DGDN layer."""
    
    def test_layer_initialization(self):
        """Test layer initialization."""
        layer = DGDNLayer(
            hidden_dim=64,
            time_dim=16,
            num_heads=4,
            num_diffusion_steps=3
        )
        
        assert layer.hidden_dim == 64
        assert layer.time_dim == 16
        assert layer.num_heads == 4
        assert layer.num_diffusion_steps == 3
        
        # Check submodules
        assert hasattr(layer, 'temporal_attention')
        assert hasattr(layer, 'diffusion_layers')
        assert len(layer.diffusion_layers) == 3
    
    def test_layer_forward(self):
        """Test layer forward pass."""
        layer = DGDNLayer(
            hidden_dim=32,
            time_dim=8,
            num_heads=2,
            num_diffusion_steps=2
        )
        
        # Create test inputs
        num_nodes, num_edges = 10, 15
        x = torch.randn(num_nodes, 32)
        edge_index = torch.randint(0, num_nodes, (2, num_edges))
        temporal_encoding = torch.randn(num_edges, 8)
        
        output = layer(x, edge_index, temporal_encoding)
        
        # Check output structure
        assert "node_features" in output
        assert "attention_weights" in output
        assert "diffusion_features" in output
        
        # Check shapes
        assert output["node_features"].shape == (num_nodes, 32)
        assert output["diffusion_features"].shape == (num_nodes, 32)


class TestMultiHeadTemporalAttention:
    """Test cases for temporal attention mechanism."""
    
    def test_attention_initialization(self):
        """Test attention initialization."""
        attention = MultiHeadTemporalAttention(
            hidden_dim=64,
            num_heads=8,
            temporal_dim=16
        )
        
        assert attention.hidden_dim == 64
        assert attention.num_heads == 8
        assert attention.head_dim == 8  # 64 / 8
        assert attention.temporal_dim == 16
    
    def test_attention_forward(self):
        """Test attention forward pass."""
        attention = MultiHeadTemporalAttention(
            hidden_dim=32,
            num_heads=4,
            temporal_dim=8
        )
        
        # Create test inputs
        num_nodes, num_edges = 10, 15
        q = k = v = torch.randn(num_nodes, 32)
        temporal_encoding = torch.randn(num_edges, 8)
        edge_index = torch.randint(0, num_nodes, (2, num_edges))
        
        attn_output, attn_weights = attention(
            query=q, key=k, value=v,
            temporal_encoding=temporal_encoding,
            edge_index=edge_index
        )
        
        # Check output shapes
        assert attn_output.shape == (num_nodes, 32)
        assert attn_weights.shape == (num_edges, 4)  # num_edges x num_heads


@pytest.fixture
def sample_temporal_data():
    """Fixture providing sample temporal data for testing."""
    num_nodes, num_edges = 20, 40
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    timestamps = torch.sort(torch.rand(num_edges) * 100)[0]
    node_features = torch.randn(num_nodes, 64)
    edge_attr = torch.randn(num_edges, 32)
    
    return TemporalData(
        edge_index=edge_index,
        timestamps=timestamps,
        node_features=node_features,
        edge_attr=edge_attr,
        num_nodes=num_nodes
    )


class TestModelIntegration:
    """Integration tests for complete model functionality."""
    
    def test_end_to_end_training_step(self, sample_temporal_data):
        """Test a complete training step."""
        model = DynamicGraphDiffusionNet(
            node_dim=64,
            edge_dim=32,
            hidden_dim=128,
            num_layers=1,
            diffusion_steps=2
        )
        
        # Forward pass
        output = model(sample_temporal_data)
        
        # Compute simple loss (MSE for demonstration)
        target_embeddings = torch.randn_like(output["node_embeddings"])
        loss = nn.MSELoss()(output["node_embeddings"], target_embeddings)
        
        # Backward pass
        loss.backward()
        
        # Check that gradients were computed
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None
    
    def test_model_device_handling(self, sample_temporal_data):
        """Test model behavior with different devices."""
        model = DynamicGraphDiffusionNet(
            node_dim=64,
            edge_dim=32,
            hidden_dim=64,
            num_layers=1
        )
        
        # Test CPU
        output_cpu = model(sample_temporal_data)
        assert output_cpu["node_embeddings"].device.type == "cpu"
        
        # Test CUDA if available
        if torch.cuda.is_available():
            model_cuda = model.cuda()
            data_cuda = sample_temporal_data.to(torch.device("cuda"))
            
            output_cuda = model_cuda(data_cuda)
            assert output_cuda["node_embeddings"].device.type == "cuda"