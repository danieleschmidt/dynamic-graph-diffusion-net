"""Main DGDN model implementation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, List, Tuple, Union
import math

from ..temporal import EdgeTimeEncoder, VariationalDiffusion
from .layers import DGDNLayer, GraphNorm


class DynamicGraphDiffusionNet(nn.Module):
    """Dynamic Graph Diffusion Network for temporal graph learning.
    
    This is the main model that combines edge-time encoding, variational diffusion,
    and multi-head attention for learning on temporal graphs with uncertainty quantification.
    
    Args:
        node_dim: Dimension of input node features
        edge_dim: Dimension of input edge features (optional)
        time_dim: Dimension of temporal embeddings
        hidden_dim: Dimension of hidden representations
        num_layers: Number of DGDN layers
        num_heads: Number of attention heads per layer
        diffusion_steps: Number of diffusion steps per layer
        aggregation: Aggregation method ("attention", "mean", "sum")
        dropout: Dropout probability
        activation: Activation function ("relu", "gelu", "swish")
        layer_norm: Whether to use layer normalization
        graph_norm: Whether to use graph normalization
        time_encoding: Type of time encoding ("fourier", "positional", "multiscale")
    """
    
    def __init__(
        self,
        node_dim: int,
        edge_dim: int = 0,
        time_dim: int = 32,
        hidden_dim: int = 256,
        num_layers: int = 3,
        num_heads: int = 8,
        diffusion_steps: int = 5,
        aggregation: str = "attention",
        dropout: float = 0.1,
        activation: str = "relu",
        layer_norm: bool = True,
        graph_norm: bool = False,
        time_encoding: str = "fourier",
        max_time: float = 1000.0,
        **kwargs
    ):
        super().__init__()
        
        # Input validation
        self._validate_init_parameters(
            node_dim, edge_dim, time_dim, hidden_dim, num_layers, 
            num_heads, diffusion_steps, aggregation, dropout, 
            activation, time_encoding, max_time
        )
        
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.time_dim = time_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.diffusion_steps = diffusion_steps
        self.aggregation = aggregation
        self.dropout = dropout
        self.max_time = max_time
        
        # Time encoder
        if time_encoding == "fourier":
            self.time_encoder = EdgeTimeEncoder(
                time_dim=time_dim,
                max_time=max_time,
                **kwargs
            )
        else:
            raise NotImplementedError(f"Time encoding '{time_encoding}' not implemented")
        
        # Input projections
        self.node_projection = nn.Linear(node_dim, hidden_dim)
        
        if edge_dim > 0:
            self.edge_projection = nn.Linear(edge_dim, hidden_dim)
        else:
            self.edge_projection = None
        
        # DGDN layers
        self.dgdn_layers = nn.ModuleList([
            DGDNLayer(
                hidden_dim=hidden_dim,
                time_dim=time_dim,
                num_heads=num_heads,
                num_diffusion_steps=diffusion_steps,
                dropout=dropout,
                activation=activation,
                layer_norm=layer_norm
            )
            for _ in range(num_layers)
        ])
        
        # Variational diffusion for uncertainty quantification
        self.variational_diffusion = VariationalDiffusion(
            hidden_dim=hidden_dim,
            num_diffusion_steps=diffusion_steps,
            num_heads=num_heads,
            dropout=dropout,
            activation=activation
        )
        
        # Normalization layers
        if graph_norm:
            self.graph_norms = nn.ModuleList([
                GraphNorm(hidden_dim) for _ in range(num_layers)
            ])
        else:
            self.graph_norms = None
        
        # Final processing layers
        self.final_norm = nn.LayerNorm(hidden_dim) if layer_norm else nn.Identity()
        self.output_projection = nn.Linear(hidden_dim, hidden_dim)
        
        # Prediction heads (can be extended for different tasks)
        self.edge_predictor = EdgePredictor(hidden_dim)
        self.node_classifier = NodeClassifier(hidden_dim, num_classes=2)  # Binary by default
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize all parameters."""
        # Initialize linear layers
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(
        self,
        data,
        return_attention: bool = False,
        return_uncertainty: bool = False
    ) -> Dict[str, torch.Tensor]:
        """Forward pass of DGDN.
        
        Args:
            data: TemporalData object containing:
                - edge_index: Edge connectivity [2, num_edges]
                - timestamps: Edge timestamps [num_edges]
                - node_features: Node features [num_nodes, node_dim] (optional)
                - edge_attr: Edge attributes [num_edges, edge_dim] (optional)
            return_attention: Whether to return attention weights
            return_uncertainty: Whether to return uncertainty estimates
            
        Returns:
            Dictionary containing model outputs
        """
        # Validate input data
        self._validate_forward_input(data)
        edge_index = data.edge_index
        timestamps = data.timestamps
        node_features = getattr(data, 'node_features', None)
        edge_attr = getattr(data, 'edge_attr', None)
        num_nodes = data.num_nodes
        
        # Handle missing node features
        if node_features is None:
            # Create learnable embeddings for nodes
            node_features = torch.randn(num_nodes, self.node_dim, device=edge_index.device)
        
        # Encode temporal information
        temporal_encoding = self.time_encoder(timestamps)
        
        # Project inputs to hidden dimension
        x = self.node_projection(node_features)
        
        # Process edge attributes if available
        if edge_attr is not None and self.edge_projection is not None:
            edge_features = self.edge_projection(edge_attr)
        else:
            edge_features = None
        
        # Store attention weights if requested
        attention_weights = [] if return_attention else None
        
        # Apply DGDN layers
        for i, layer in enumerate(self.dgdn_layers):
            layer_output = layer(
                x=x,
                edge_index=edge_index,
                temporal_encoding=temporal_encoding,
                edge_attr=edge_features
            )
            
            x = layer_output["node_features"]
            
            if return_attention:
                attention_weights.append(layer_output["attention_weights"])
            
            # Apply graph normalization if enabled
            if self.graph_norms is not None:
                x = self.graph_norms[i](x)
        
        # Apply variational diffusion for uncertainty quantification
        diffusion_output = self.variational_diffusion(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_features,
            return_all_steps=return_uncertainty
        )
        
        # Final processing
        final_embeddings = diffusion_output["z"]
        final_embeddings = self.final_norm(final_embeddings)
        final_embeddings = self.output_projection(final_embeddings)
        
        # Prepare output dictionary
        output = {
            "node_embeddings": final_embeddings,
            "mean": diffusion_output["mean"],
            "logvar": diffusion_output["logvar"],
            "kl_loss": diffusion_output["kl_loss"],
            "temporal_encoding": temporal_encoding
        }
        
        if return_attention:
            output["attention_weights"] = attention_weights
        
        if return_uncertainty:
            output["uncertainty"] = self.variational_diffusion.get_uncertainty(diffusion_output["logvar"])
            if "all_steps" in diffusion_output:
                output["diffusion_steps"] = diffusion_output["all_steps"]
        
        return output
    
    def get_node_embeddings(
        self,
        node_ids: List[int],
        time: float,
        data
    ) -> torch.Tensor:
        """Get node embeddings at a specific time.
        
        Args:
            node_ids: List of node IDs
            time: Target timestamp
            data: TemporalData object
            
        Returns:
            Node embeddings at the specified time
        """
        # Get subgraph up to the specified time
        subgraph_data = data.time_window(0, time)
        
        # Forward pass
        output = self.forward(subgraph_data)
        node_embeddings = output["node_embeddings"]
        
        # Extract embeddings for specified nodes
        return node_embeddings[node_ids]
    
    def predict_edges(
        self,
        source_nodes: torch.Tensor,
        target_nodes: torch.Tensor,
        time: float,
        data,
        return_probs: bool = True
    ) -> torch.Tensor:
        """Predict edge probabilities between node pairs.
        
        Args:
            source_nodes: Source node IDs [num_pairs]
            target_nodes: Target node IDs [num_pairs]
            time: Prediction timestamp
            data: TemporalData object
            return_probs: Whether to return probabilities (vs logits)
            
        Returns:
            Edge predictions [num_pairs] or [num_pairs, 2]
        """
        # Get node embeddings at the specified time
        subgraph_data = data.time_window(0, time)
        output = self.forward(subgraph_data)
        node_embeddings = output["node_embeddings"]
        
        # Get embeddings for source and target nodes
        src_embeddings = node_embeddings[source_nodes]
        tgt_embeddings = node_embeddings[target_nodes]
        
        # Predict edges
        edge_predictions = self.edge_predictor(src_embeddings, tgt_embeddings)
        
        if return_probs:
            return torch.softmax(edge_predictions, dim=-1)
        return edge_predictions
    
    def predict_nodes(
        self,
        node_ids: torch.Tensor,
        time: float,
        data,
        return_probs: bool = True
    ) -> torch.Tensor:
        """Predict node classifications.
        
        Args:
            node_ids: Node IDs to classify
            time: Prediction timestamp  
            data: TemporalData object
            return_probs: Whether to return probabilities
            
        Returns:
            Node predictions
        """
        # Get node embeddings
        node_embeddings = self.get_node_embeddings(node_ids, time, data)
        
        # Classify nodes
        node_predictions = self.node_classifier(node_embeddings)
        
        if return_probs:
            return torch.softmax(node_predictions, dim=-1)
        return node_predictions
    
    def compute_loss(
        self,
        output: Dict[str, torch.Tensor],
        targets: torch.Tensor,
        task: str = "edge_prediction",
        beta_kl: float = 0.1,
        beta_temporal: float = 0.05
    ) -> Dict[str, torch.Tensor]:
        """Compute total loss with multiple components.
        
        Args:
            output: Model output dictionary
            targets: Ground truth targets
            task: Type of task ("edge_prediction", "node_classification")
            beta_kl: Weight for KL divergence loss
            beta_temporal: Weight for temporal regularization
            
        Returns:
            Dictionary of loss components
        """
        losses = {}
        
        # Task-specific reconstruction loss
        if task == "edge_prediction":
            node_embeddings = output["node_embeddings"]
            # For edge prediction, we need to implement edge-specific logic
            # This is simplified - in practice, you'd use the edge predictor
            recon_loss = F.mse_loss(node_embeddings, targets)
        elif task == "node_classification":
            node_embeddings = output["node_embeddings"]
            recon_loss = F.cross_entropy(self.node_classifier(node_embeddings), targets)
        else:
            raise ValueError(f"Unknown task: {task}")
        
        losses["reconstruction"] = recon_loss
        
        # KL divergence loss from variational diffusion
        losses["kl_divergence"] = output["kl_loss"]
        
        # Temporal regularization (encourage smooth temporal transitions)
        if "temporal_encoding" in output:
            temporal_encoding = output["temporal_encoding"]
            if temporal_encoding.shape[0] > 1:
                temporal_diff = temporal_encoding[1:] - temporal_encoding[:-1]
                temporal_reg = torch.mean(torch.sum(temporal_diff ** 2, dim=-1))
                losses["temporal_regularization"] = temporal_reg
            else:
                losses["temporal_regularization"] = torch.tensor(0.0, device=recon_loss.device)
        
        # Total loss
        total_loss = (
            losses["reconstruction"] +
            beta_kl * losses["kl_divergence"] +
            beta_temporal * losses["temporal_regularization"]
        )
        
        losses["total"] = total_loss
        
        return losses
    
    def _validate_init_parameters(self, node_dim, edge_dim, time_dim, hidden_dim, 
                                 num_layers, num_heads, diffusion_steps, aggregation,
                                 dropout, activation, time_encoding, max_time):
        """Validate initialization parameters."""
        # Dimension validations
        if not isinstance(node_dim, int) or node_dim <= 0:
            raise ValueError(f"node_dim must be a positive integer, got {node_dim}")
        if not isinstance(edge_dim, int) or edge_dim < 0:
            raise ValueError(f"edge_dim must be a non-negative integer, got {edge_dim}")
        if not isinstance(time_dim, int) or time_dim <= 0:
            raise ValueError(f"time_dim must be a positive integer, got {time_dim}")
        if not isinstance(hidden_dim, int) or hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be a positive integer, got {hidden_dim}")
        
        # Architecture validations
        if not isinstance(num_layers, int) or num_layers <= 0:
            raise ValueError(f"num_layers must be a positive integer, got {num_layers}")
        if not isinstance(num_heads, int) or num_heads <= 0:
            raise ValueError(f"num_heads must be a positive integer, got {num_heads}")
        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})")
        
        # Diffusion validations
        if not isinstance(diffusion_steps, int) or diffusion_steps <= 0:
            raise ValueError(f"diffusion_steps must be a positive integer, got {diffusion_steps}")
        
        # String parameter validations
        valid_aggregations = {"attention", "mean", "sum"}
        if aggregation not in valid_aggregations:
            raise ValueError(f"aggregation must be one of {valid_aggregations}, got {aggregation}")
        
        valid_activations = {"relu", "gelu", "swish", "leaky_relu"}
        if activation not in valid_activations:
            raise ValueError(f"activation must be one of {valid_activations}, got {activation}")
        
        valid_time_encodings = {"fourier", "positional", "multiscale"}
        if time_encoding not in valid_time_encodings:
            raise ValueError(f"time_encoding must be one of {valid_time_encodings}, got {time_encoding}")
        
        # Numerical parameter validations
        if not isinstance(dropout, (int, float)) or not (0.0 <= dropout < 1.0):
            raise ValueError(f"dropout must be a float in [0.0, 1.0), got {dropout}")
        if not isinstance(max_time, (int, float)) or max_time <= 0:
            raise ValueError(f"max_time must be a positive number, got {max_time}")
    
    def _validate_forward_input(self, data):
        """Validate forward pass input data."""
        if not hasattr(data, 'edge_index'):
            raise ValueError("Input data must have 'edge_index' attribute")
        if not hasattr(data, 'timestamps'):
            raise ValueError("Input data must have 'timestamps' attribute")
        if not hasattr(data, 'num_nodes'):
            raise ValueError("Input data must have 'num_nodes' attribute")
        
        # Validate tensor shapes and types
        edge_index = data.edge_index
        timestamps = data.timestamps
        
        if not isinstance(edge_index, torch.Tensor):
            raise TypeError("edge_index must be a torch.Tensor")
        if not isinstance(timestamps, torch.Tensor):
            raise TypeError("timestamps must be a torch.Tensor")
        
        if edge_index.dim() != 2 or edge_index.size(0) != 2:
            raise ValueError(f"edge_index must have shape [2, num_edges], got {edge_index.shape}")
        if timestamps.dim() != 1:
            raise ValueError(f"timestamps must be 1-dimensional, got {timestamps.shape}")
        if edge_index.size(1) != timestamps.size(0):
            raise ValueError(f"Number of edges in edge_index ({edge_index.size(1)}) must match timestamps ({timestamps.size(0)})")
        
        # Check for valid node indices
        max_node_idx = edge_index.max().item()
        if max_node_idx >= data.num_nodes:
            raise ValueError(f"Maximum node index ({max_node_idx}) exceeds num_nodes ({data.num_nodes})")
        if edge_index.min().item() < 0:
            raise ValueError("Node indices in edge_index must be non-negative")
        
        # Validate optional attributes
        if hasattr(data, 'node_features') and data.node_features is not None:
            node_features = data.node_features
            if not isinstance(node_features, torch.Tensor):
                raise TypeError("node_features must be a torch.Tensor")
            if node_features.size(0) != data.num_nodes:
                raise ValueError(f"node_features size ({node_features.size(0)}) must match num_nodes ({data.num_nodes})")
            if node_features.size(1) != self.node_dim:
                raise ValueError(f"node_features dimension ({node_features.size(1)}) must match node_dim ({self.node_dim})")
        
        if hasattr(data, 'edge_attr') and data.edge_attr is not None:
            edge_attr = data.edge_attr
            if not isinstance(edge_attr, torch.Tensor):
                raise TypeError("edge_attr must be a torch.Tensor")
            if edge_attr.size(0) != timestamps.size(0):
                raise ValueError(f"edge_attr size ({edge_attr.size(0)}) must match number of edges ({timestamps.size(0)})")
            if self.edge_dim > 0 and edge_attr.size(1) != self.edge_dim:
                raise ValueError(f"edge_attr dimension ({edge_attr.size(1)}) must match edge_dim ({self.edge_dim})")


class EdgePredictor(nn.Module):
    """Edge prediction head for link prediction tasks."""
    
    def __init__(self, hidden_dim: int, num_classes: int = 2):
        super().__init__()
        
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_classes)
        )
    
    def forward(self, src_embeddings: torch.Tensor, tgt_embeddings: torch.Tensor) -> torch.Tensor:
        """Predict edge existence between source and target nodes."""
        # Concatenate source and target embeddings
        edge_features = torch.cat([src_embeddings, tgt_embeddings], dim=-1)
        return self.predictor(edge_features)


class NodeClassifier(nn.Module):
    """Node classification head."""
    
    def __init__(self, hidden_dim: int, num_classes: int = 2):
        super().__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_classes)
        )
    
    def forward(self, node_embeddings: torch.Tensor) -> torch.Tensor:
        """Classify nodes based on embeddings."""
        return self.classifier(node_embeddings)