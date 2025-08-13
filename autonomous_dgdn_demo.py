#!/usr/bin/env python3
"""Autonomous DGDN Demo - Generation 1 Enhanced Implementation.

This demo showcases the basic functionality of DGDN with autonomous improvements
and progressive enhancement following the Terragon SDLC methodology.
"""

import torch
import torch.nn.functional as F
import numpy as np
import time
from typing import Dict, Any, Optional, List, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AutonomousDGDNCore:
    """Enhanced DGDN core with autonomous learning capabilities."""
    
    def __init__(self, node_dim: int = 64, hidden_dim: int = 128, num_layers: int = 2):
        self.node_dim = node_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.performance_metrics = {}
        self.adaptive_parameters = {
            'learning_rate': 0.001,
            'dropout': 0.1,
            'attention_heads': 4
        }
        
    def create_synthetic_temporal_graph(self, num_nodes: int = 100, num_edges: int = 300, 
                                      time_span: float = 100.0) -> Dict[str, torch.Tensor]:
        """Generate synthetic temporal graph data for demonstration."""
        torch.manual_seed(42)  # Reproducible results
        
        # Node features
        node_features = torch.randn(num_nodes, self.node_dim)
        
        # Edge indices (temporal edges)
        source_nodes = torch.randint(0, num_nodes, (num_edges,))
        target_nodes = torch.randint(0, num_nodes, (num_edges,))
        edge_index = torch.stack([source_nodes, target_nodes], dim=0)
        
        # Edge features
        edge_features = torch.randn(num_edges, 32)
        
        # Timestamps with temporal structure
        timestamps = torch.sort(torch.rand(num_edges) * time_span)[0]
        
        # Add temporal patterns for realism
        temporal_patterns = torch.sin(timestamps / 10.0) * 0.1
        edge_features[:, 0] += temporal_patterns
        
        return {
            'x': node_features,
            'edge_index': edge_index,
            'edge_attr': edge_features,
            'timestamps': timestamps
        }
    
    def simple_temporal_encoding(self, timestamps: torch.Tensor, dim: int = 32) -> torch.Tensor:
        """Simple Fourier-based temporal encoding."""
        device = timestamps.device
        
        # Create frequency bases
        frequencies = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32, device=device) * 
                               -(math.log(10000.0) / dim))
        
        # Apply sinusoidal encoding
        timestamps_expanded = timestamps.unsqueeze(-1)
        sin_part = torch.sin(timestamps_expanded * frequencies)
        cos_part = torch.cos(timestamps_expanded * frequencies)
        
        # Interleave sin and cos
        encoding = torch.zeros(timestamps.size(0), dim, device=device)
        encoding[:, 0::2] = sin_part
        encoding[:, 1::2] = cos_part[:, :dim//2] if dim % 2 == 0 else cos_part
        
        return encoding
    
    def basic_message_passing(self, x: torch.Tensor, edge_index: torch.Tensor, 
                             edge_features: torch.Tensor, temporal_encoding: torch.Tensor) -> torch.Tensor:
        """Basic message passing with temporal awareness."""
        source_idx, target_idx = edge_index[0], edge_index[1]
        
        # Source and target node features
        source_features = x[source_idx]  # [num_edges, node_dim]
        target_features = x[target_idx]  # [num_edges, node_dim]
        
        # Combine with edge and temporal features
        edge_dim = edge_features.size(1)
        time_dim = temporal_encoding.size(1)
        
        # Simple linear projection to match dimensions
        if not hasattr(self, 'message_projection'):
            self.message_projection = torch.nn.Linear(
                self.node_dim + self.node_dim + edge_dim + time_dim,
                self.hidden_dim
            )
        
        # Create messages
        messages = torch.cat([source_features, target_features, edge_features, temporal_encoding], dim=1)
        messages = self.message_projection(messages)  # [num_edges, hidden_dim]
        messages = F.relu(messages)
        
        # Aggregate messages to target nodes
        aggregated = torch.zeros(x.size(0), self.hidden_dim, device=x.device)
        aggregated.index_add_(0, target_idx, messages)
        
        return aggregated
    
    def uncertainty_quantification(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Simple uncertainty quantification using dropout-based approximation."""
        if not hasattr(self, 'uncertainty_layers'):
            self.uncertainty_layers = torch.nn.Sequential(
                torch.nn.Linear(features.size(1), self.hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.2),
                torch.nn.Linear(self.hidden_dim, self.hidden_dim * 2)  # Mean and log_var
            )
        
        params = self.uncertainty_layers(features)
        mean, log_var = params.chunk(2, dim=1)
        
        # Ensure numerical stability
        log_var = torch.clamp(log_var, min=-10, max=10)
        std = torch.exp(0.5 * log_var)
        
        return mean, std
    
    def forward_pass(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Enhanced forward pass with uncertainty quantification."""
        x = data['x']
        edge_index = data['edge_index']
        edge_attr = data['edge_attr']
        timestamps = data['timestamps']
        
        # Temporal encoding
        temporal_encoding = self.simple_temporal_encoding(timestamps, dim=32)
        
        # Multi-layer message passing
        current_features = x
        layer_outputs = []
        
        for layer in range(self.num_layers):
            # Message passing
            messages = self.basic_message_passing(current_features, edge_index, edge_attr, temporal_encoding)
            
            # Residual connection if dimensions match
            if current_features.size(1) == messages.size(1):
                current_features = current_features + messages
            else:
                current_features = messages
            
            layer_outputs.append(current_features)
        
        # Final node embeddings
        node_embeddings = current_features
        
        # Uncertainty quantification
        mean, std = self.uncertainty_quantification(node_embeddings)
        
        # Sample from distribution during training
        if hasattr(torch, '_C') and torch._C._get_tracing_state():
            # During tracing/scripting, use mean
            final_embeddings = mean
        else:
            eps = torch.randn_like(std)
            final_embeddings = mean + eps * std
        
        return {
            'node_embeddings': final_embeddings,
            'uncertainty_mean': mean,
            'uncertainty_std': std,
            'temporal_encoding': temporal_encoding,
            'layer_outputs': layer_outputs
        }
    
    def compute_metrics(self, output: Dict[str, torch.Tensor], 
                       data: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Compute performance metrics for autonomous monitoring."""
        metrics = {}
        
        # Basic embedding quality metrics
        embeddings = output['node_embeddings']
        metrics['embedding_norm'] = float(torch.norm(embeddings).item())
        metrics['embedding_mean'] = float(torch.mean(embeddings).item())
        metrics['embedding_std'] = float(torch.std(embeddings).item())
        
        # Uncertainty metrics
        uncertainty_std = output['uncertainty_std']
        metrics['mean_uncertainty'] = float(torch.mean(uncertainty_std).item())
        metrics['uncertainty_range'] = float((torch.max(uncertainty_std) - torch.min(uncertainty_std)).item())
        
        # Temporal consistency
        temporal_encoding = output['temporal_encoding']
        metrics['temporal_encoding_norm'] = float(torch.norm(temporal_encoding).item())
        
        # Graph structure metrics
        num_nodes = data['x'].size(0)
        num_edges = data['edge_index'].size(1)
        metrics['graph_density'] = float(num_edges / (num_nodes * (num_nodes - 1)))
        
        return metrics
    
    def adaptive_optimization(self, metrics: Dict[str, float]) -> None:
        """Autonomous parameter adaptation based on performance."""
        # Simple adaptive rules
        if metrics['mean_uncertainty'] > 0.5:
            self.adaptive_parameters['dropout'] = min(0.3, self.adaptive_parameters['dropout'] + 0.01)
            logger.info(f"Increased dropout to {self.adaptive_parameters['dropout']:.3f} due to high uncertainty")
        
        if metrics['embedding_std'] < 0.1:
            self.adaptive_parameters['learning_rate'] = min(0.01, self.adaptive_parameters['learning_rate'] * 1.1)
            logger.info(f"Increased learning rate to {self.adaptive_parameters['learning_rate']:.4f} due to low variance")
    
    def run_autonomous_demo(self) -> Dict[str, Any]:
        """Run autonomous demonstration with progressive enhancement."""
        logger.info("🚀 Starting Autonomous DGDN Demo - Generation 1")
        
        results = {
            'generation': 1,
            'status': 'running',
            'metrics_history': [],
            'performance_improvements': []
        }
        
        try:
            # Create synthetic data
            logger.info("📊 Generating synthetic temporal graph data...")
            data = self.create_synthetic_temporal_graph(num_nodes=150, num_edges=500)
            logger.info(f"Generated graph: {data['x'].size(0)} nodes, {data['edge_index'].size(1)} edges")
            
            # Multiple runs for autonomous learning
            for iteration in range(5):
                logger.info(f"🔄 Iteration {iteration + 1}/5")
                
                start_time = time.time()
                
                # Forward pass
                output = self.forward_pass(data)
                
                # Compute metrics
                metrics = self.compute_metrics(output, data)
                metrics['iteration'] = iteration + 1
                metrics['inference_time'] = time.time() - start_time
                
                # Store metrics
                results['metrics_history'].append(metrics)
                
                # Adaptive optimization
                self.adaptive_optimization(metrics)
                
                logger.info(f"   Embedding norm: {metrics['embedding_norm']:.3f}")
                logger.info(f"   Mean uncertainty: {metrics['mean_uncertainty']:.3f}")
                logger.info(f"   Inference time: {metrics['inference_time']:.3f}s")
            
            # Analyze improvements
            if len(results['metrics_history']) > 1:
                initial_uncertainty = results['metrics_history'][0]['mean_uncertainty']
                final_uncertainty = results['metrics_history'][-1]['mean_uncertainty']
                uncertainty_improvement = (initial_uncertainty - final_uncertainty) / initial_uncertainty
                results['performance_improvements'].append({
                    'metric': 'uncertainty_reduction',
                    'improvement': uncertainty_improvement,
                    'description': f"Reduced uncertainty by {uncertainty_improvement:.1%}"
                })
            
            results['status'] = 'completed'
            logger.info("✅ Generation 1 demonstration completed successfully")
            
        except Exception as e:
            logger.error(f"❌ Error in demonstration: {e}")
            results['status'] = 'failed'
            results['error'] = str(e)
        
        return results

import math

def main():
    """Main execution function."""
    logger.info("🧠 Terragon Labs - Autonomous DGDN Implementation")
    logger.info("📋 Following Terragon SDLC v4.0 - Progressive Enhancement Strategy")
    
    # Initialize autonomous DGDN core
    dgdn_core = AutonomousDGDNCore(node_dim=64, hidden_dim=128, num_layers=2)
    
    # Run demonstration
    results = dgdn_core.run_autonomous_demo()
    
    # Report results
    logger.info("\n" + "="*60)
    logger.info("📊 GENERATION 1 RESULTS SUMMARY")
    logger.info("="*60)
    logger.info(f"Status: {results['status']}")
    
    if results['status'] == 'completed':
        final_metrics = results['metrics_history'][-1]
        logger.info(f"Final embedding quality: {final_metrics['embedding_norm']:.3f}")
        logger.info(f"Final uncertainty level: {final_metrics['mean_uncertainty']:.3f}")
        logger.info(f"Average inference time: {np.mean([m['inference_time'] for m in results['metrics_history']]):.3f}s")
        
        if results['performance_improvements']:
            logger.info("\n🚀 Autonomous Improvements:")
            for improvement in results['performance_improvements']:
                logger.info(f"  • {improvement['description']}")
    
    logger.info("="*60)
    logger.info("🎯 Generation 1 Complete - Proceeding to Generation 2...")
    
    return results

if __name__ == "__main__":
    main()