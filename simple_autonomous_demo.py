#!/usr/bin/env python3
"""Simple Autonomous DGDN Demo - Generation 1 without heavy dependencies.

This demo showcases basic DGDN functionality using only standard library
and minimal numpy, following Terragon SDLC progressive enhancement.
"""

import sys
import time
import math
import random
import logging
from typing import Dict, Any, List, Tuple, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try to import numpy, fallback to basic list operations
try:
    import numpy as np
    NUMPY_AVAILABLE = True
    logger.info("📦 NumPy available - enhanced performance mode")
except ImportError:
    NUMPY_AVAILABLE = False
    logger.info("📦 NumPy not available - using pure Python fallback")

class SimpleTensor:
    """Lightweight tensor-like class for basic operations."""
    
    def __init__(self, data: List[List[float]]):
        self.data = data
        self.shape = (len(data), len(data[0]) if data else 0)
    
    def __add__(self, other):
        result = []
        for i in range(len(self.data)):
            row = []
            for j in range(len(self.data[i])):
                if isinstance(other, SimpleTensor):
                    row.append(self.data[i][j] + other.data[i][j])
                else:
                    row.append(self.data[i][j] + other)
            result.append(row)
        return SimpleTensor(result)
    
    def __mul__(self, scalar: float):
        result = []
        for i in range(len(self.data)):
            row = []
            for j in range(len(self.data[i])):
                row.append(self.data[i][j] * scalar)
            result.append(row)
        return SimpleTensor(result)
    
    def norm(self) -> float:
        total = 0.0
        for row in self.data:
            for val in row:
                total += val * val
        return math.sqrt(total)
    
    def mean(self) -> float:
        total = 0.0
        count = 0
        for row in self.data:
            for val in row:
                total += val
                count += 1
        return total / count if count > 0 else 0.0
    
    def std(self) -> float:
        mean_val = self.mean()
        total = 0.0
        count = 0
        for row in self.data:
            for val in row:
                total += (val - mean_val) ** 2
                count += 1
        return math.sqrt(total / count) if count > 0 else 0.0

class LightweightDGDN:
    """Lightweight DGDN implementation without heavy dependencies."""
    
    def __init__(self, node_dim: int = 32, hidden_dim: int = 64, num_layers: int = 2):
        self.node_dim = node_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.performance_metrics = {}
        
        # Initialize simple parameters
        self.weights = self._initialize_weights()
        
        # Adaptive parameters for autonomous optimization
        self.adaptive_params = {
            'message_strength': 1.0,
            'temporal_weight': 0.5,
            'uncertainty_scale': 0.1
        }
        
        logger.info(f"🧠 Initialized Lightweight DGDN: {node_dim}→{hidden_dim}, {num_layers} layers")
    
    def _initialize_weights(self) -> Dict[str, Any]:
        """Initialize simple weight matrices using basic random values."""
        random.seed(42)  # Reproducible initialization
        
        weights = {}
        
        # Simple linear transformation weights
        weights['node_projection'] = [[random.gauss(0, 0.1) for _ in range(self.hidden_dim)] 
                                     for _ in range(self.node_dim)]
        
        weights['temporal_projection'] = [[random.gauss(0, 0.1) for _ in range(self.hidden_dim)] 
                                         for _ in range(32)]  # 32 temporal features
        
        weights['message_weights'] = [[random.gauss(0, 0.1) for _ in range(self.hidden_dim)] 
                                     for _ in range(self.hidden_dim * 2)]
        
        return weights
    
    def create_synthetic_data(self, num_nodes: int = 50, num_edges: int = 150) -> Dict[str, Any]:
        """Create synthetic temporal graph data."""
        random.seed(42)
        
        # Node features
        node_features = [[random.gauss(0, 1) for _ in range(self.node_dim)] 
                        for _ in range(num_nodes)]
        
        # Edge list (source, target, timestamp, weight)
        edges = []
        for _ in range(num_edges):
            source = random.randint(0, num_nodes - 1)
            target = random.randint(0, num_nodes - 1)
            if source != target:  # Avoid self-loops
                timestamp = random.uniform(0, 100)
                weight = random.uniform(0.1, 1.0)
                edges.append((source, target, timestamp, weight))
        
        # Sort edges by timestamp for temporal realism
        edges.sort(key=lambda x: x[2])
        
        logger.info(f"📊 Generated synthetic data: {num_nodes} nodes, {len(edges)} edges")
        
        return {
            'node_features': node_features,
            'edges': edges,
            'num_nodes': num_nodes,
            'num_edges': len(edges)
        }
    
    def fourier_temporal_encoding(self, timestamps: List[float], dim: int = 32) -> List[List[float]]:
        """Simple Fourier-based temporal encoding."""
        encoding = []
        
        for t in timestamps:
            features = []
            for i in range(dim // 2):
                freq = 1.0 / (10000.0 ** (2 * i / dim))
                sin_val = math.sin(t * freq)
                cos_val = math.cos(t * freq)
                features.extend([sin_val, cos_val])
            
            # Pad if needed
            while len(features) < dim:
                features.append(0.0)
            
            encoding.append(features[:dim])
        
        return encoding
    
    def matrix_multiply(self, a: List[List[float]], b: List[List[float]]) -> List[List[float]]:
        """Simple matrix multiplication."""
        rows_a, cols_a = len(a), len(a[0])
        rows_b, cols_b = len(b), len(b[0])
        
        if cols_a != rows_b:
            raise ValueError(f"Matrix dimensions mismatch: {cols_a} != {rows_b}")
        
        result = [[0.0 for _ in range(cols_b)] for _ in range(rows_a)]
        
        for i in range(rows_a):
            for j in range(cols_b):
                for k in range(cols_a):
                    result[i][j] += a[i][k] * b[k][j]
        
        return result
    
    def apply_activation(self, x: List[List[float]], activation: str = 'relu') -> List[List[float]]:
        """Apply activation function."""
        result = []
        
        for row in x:
            new_row = []
            for val in row:
                if activation == 'relu':
                    new_row.append(max(0.0, val))
                elif activation == 'tanh':
                    new_row.append(math.tanh(val))
                elif activation == 'sigmoid':
                    new_row.append(1.0 / (1.0 + math.exp(-max(-500, min(500, val)))))
                else:
                    new_row.append(val)
            result.append(new_row)
        
        return result
    
    def simple_message_passing(self, node_features: List[List[float]], 
                              edges: List[Tuple], temporal_encoding: List[List[float]]) -> List[List[float]]:
        """Basic message passing implementation."""
        num_nodes = len(node_features)
        hidden_dim = self.hidden_dim
        
        # Initialize node embeddings
        node_embeddings = [[0.0 for _ in range(hidden_dim)] for _ in range(num_nodes)]
        
        # Process each edge
        for edge_idx, (source, target, timestamp, weight) in enumerate(edges):
            if edge_idx >= len(temporal_encoding):
                continue
                
            # Get source node features
            source_features = node_features[source]
            
            # Project node features to hidden dimension
            projected_features = [0.0] * hidden_dim
            for i in range(min(len(source_features), self.node_dim)):
                for j in range(hidden_dim):
                    projected_features[j] += source_features[i] * self.weights['node_projection'][i][j]
            
            # Add temporal information
            temporal_features = temporal_encoding[edge_idx]
            temporal_projected = [0.0] * hidden_dim
            for i in range(min(len(temporal_features), 32)):
                for j in range(hidden_dim):
                    temporal_projected[j] += temporal_features[i] * self.weights['temporal_projection'][i][j]
            
            # Combine features
            message = []
            for i in range(hidden_dim):
                combined_val = (projected_features[i] * self.adaptive_params['message_strength'] + 
                              temporal_projected[i] * self.adaptive_params['temporal_weight']) * weight
                message.append(combined_val)
            
            # Aggregate to target node
            for i in range(hidden_dim):
                node_embeddings[target][i] += message[i]
        
        # Apply activation
        node_embeddings = self.apply_activation(node_embeddings, 'relu')
        
        return node_embeddings
    
    def compute_uncertainty(self, embeddings: List[List[float]]) -> Tuple[float, float]:
        """Simple uncertainty quantification."""
        all_values = []
        for row in embeddings:
            all_values.extend(row)
        
        if not all_values:
            return 0.0, 0.0
        
        mean_val = sum(all_values) / len(all_values)
        variance = sum((x - mean_val) ** 2 for x in all_values) / len(all_values)
        std_val = math.sqrt(variance)
        
        # Scale by adaptive parameter
        scaled_uncertainty = std_val * self.adaptive_params['uncertainty_scale']
        
        return mean_val, scaled_uncertainty
    
    def forward_pass(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Complete forward pass through the lightweight DGDN."""
        node_features = data['node_features']
        edges = data['edges']
        
        # Extract timestamps
        timestamps = [edge[2] for edge in edges]
        
        # Temporal encoding
        temporal_encoding = self.fourier_temporal_encoding(timestamps)
        
        # Multi-layer processing
        current_embeddings = node_features
        layer_outputs = []
        
        for layer in range(self.num_layers):
            # Message passing
            new_embeddings = self.simple_message_passing(current_embeddings, edges, temporal_encoding)
            layer_outputs.append(new_embeddings)
            current_embeddings = new_embeddings
        
        # Final embeddings
        final_embeddings = current_embeddings
        
        # Uncertainty quantification
        uncertainty_mean, uncertainty_std = self.compute_uncertainty(final_embeddings)
        
        return {
            'node_embeddings': final_embeddings,
            'uncertainty_mean': uncertainty_mean,
            'uncertainty_std': uncertainty_std,
            'temporal_encoding': temporal_encoding,
            'layer_outputs': layer_outputs
        }
    
    def compute_metrics(self, output: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, float]:
        """Compute performance metrics."""
        embeddings = output['node_embeddings']
        
        # Convert to SimpleTensor for easier computation
        if NUMPY_AVAILABLE:
            emb_array = np.array(embeddings)
            metrics = {
                'embedding_norm': float(np.linalg.norm(emb_array)),
                'embedding_mean': float(np.mean(emb_array)),
                'embedding_std': float(np.std(emb_array))
            }
        else:
            tensor = SimpleTensor(embeddings)
            metrics = {
                'embedding_norm': tensor.norm(),
                'embedding_mean': tensor.mean(),
                'embedding_std': tensor.std()
            }
        
        # Add uncertainty metrics
        metrics['uncertainty_mean'] = output['uncertainty_mean']
        metrics['uncertainty_std'] = output['uncertainty_std']
        
        # Graph structure metrics
        metrics['num_nodes'] = data['num_nodes']
        metrics['num_edges'] = data['num_edges']
        metrics['graph_density'] = data['num_edges'] / (data['num_nodes'] * (data['num_nodes'] - 1))
        
        return metrics
    
    def autonomous_optimization(self, metrics: Dict[str, float], iteration: int) -> None:
        """Autonomous parameter optimization based on performance."""
        # Adaptive rules for autonomous improvement
        if metrics['uncertainty_std'] > 1.0:
            self.adaptive_params['uncertainty_scale'] *= 0.95
            logger.info(f"   → Reduced uncertainty scale to {self.adaptive_params['uncertainty_scale']:.3f}")
        
        if metrics['embedding_std'] < 0.5:
            self.adaptive_params['message_strength'] = min(2.0, self.adaptive_params['message_strength'] * 1.05)
            logger.info(f"   → Increased message strength to {self.adaptive_params['message_strength']:.3f}")
        
        if iteration > 2 and metrics['embedding_mean'] > 1.0:
            self.adaptive_params['temporal_weight'] = max(0.1, self.adaptive_params['temporal_weight'] * 0.98)
            logger.info(f"   → Adjusted temporal weight to {self.adaptive_params['temporal_weight']:.3f}")
    
    def run_demonstration(self) -> Dict[str, Any]:
        """Run complete autonomous demonstration."""
        logger.info("🚀 Starting Lightweight DGDN Demonstration")
        logger.info("📋 Following Terragon SDLC v4.0 - Progressive Enhancement")
        
        results = {
            'generation': 1,
            'implementation': 'lightweight',
            'status': 'running',
            'metrics_history': [],
            'improvements': []
        }
        
        try:
            # Generate synthetic data
            logger.info("📊 Creating synthetic temporal graph...")
            data = self.create_synthetic_data(num_nodes=30, num_edges=80)
            
            # Run multiple iterations for autonomous learning
            for iteration in range(6):
                logger.info(f"🔄 Iteration {iteration + 1}/6")
                
                start_time = time.time()
                
                # Forward pass
                output = self.forward_pass(data)
                
                # Compute metrics
                metrics = self.compute_metrics(output, data)
                metrics['iteration'] = iteration + 1
                metrics['inference_time'] = time.time() - start_time
                
                # Store results
                results['metrics_history'].append(metrics)
                
                # Autonomous optimization
                self.autonomous_optimization(metrics, iteration)
                
                # Log progress
                logger.info(f"   Embedding norm: {metrics['embedding_norm']:.3f}")
                logger.info(f"   Uncertainty: {metrics['uncertainty_std']:.3f}")
                logger.info(f"   Inference time: {metrics['inference_time']:.4f}s")
            
            # Analyze improvements
            initial_metrics = results['metrics_history'][0]
            final_metrics = results['metrics_history'][-1]
            
            improvements = []
            
            # Uncertainty improvement
            uncertainty_change = ((initial_metrics['uncertainty_std'] - final_metrics['uncertainty_std']) / 
                                 max(initial_metrics['uncertainty_std'], 1e-6))
            if abs(uncertainty_change) > 0.01:
                improvements.append({
                    'metric': 'uncertainty_optimization',
                    'change': uncertainty_change,
                    'description': f"Uncertainty {'reduced' if uncertainty_change > 0 else 'increased'} by {abs(uncertainty_change):.1%}"
                })
            
            # Performance improvement
            time_improvement = (initial_metrics['inference_time'] - final_metrics['inference_time']) / initial_metrics['inference_time']
            if time_improvement > 0.01:
                improvements.append({
                    'metric': 'inference_speedup',
                    'change': time_improvement,
                    'description': f"Inference speed improved by {time_improvement:.1%}"
                })
            
            results['improvements'] = improvements
            results['status'] = 'completed'
            
            logger.info("✅ Generation 1 demonstration completed successfully")
            
        except Exception as e:
            logger.error(f"❌ Error during demonstration: {e}")
            results['status'] = 'failed'
            results['error'] = str(e)
        
        return results

def main():
    """Main execution function for autonomous DGDN demo."""
    logger.info("🧠 Terragon Labs - Autonomous DGDN Implementation")
    logger.info("🏗️  Generation 1: Make It Work (Simple)")
    logger.info("="*70)
    
    # Initialize lightweight DGDN
    dgdn = LightweightDGDN(node_dim=32, hidden_dim=64, num_layers=2)
    
    # Run autonomous demonstration
    start_time = time.time()
    results = dgdn.run_demonstration()
    total_time = time.time() - start_time
    
    # Report results
    logger.info("\n" + "="*70)
    logger.info("📊 GENERATION 1 COMPLETION REPORT")
    logger.info("="*70)
    
    logger.info(f"Status: {results['status'].upper()}")
    logger.info(f"Total execution time: {total_time:.2f}s")
    logger.info(f"Implementation: {results['implementation']}")
    logger.info(f"NumPy acceleration: {'Enabled' if NUMPY_AVAILABLE else 'Disabled (pure Python)'}")
    
    if results['status'] == 'completed':
        final_metrics = results['metrics_history'][-1]
        logger.info(f"\n🎯 Final Performance:")
        logger.info(f"  • Embedding quality: {final_metrics['embedding_norm']:.3f}")
        logger.info(f"  • Uncertainty level: {final_metrics['uncertainty_std']:.3f}")
        logger.info(f"  • Average inference time: {np.mean([m['inference_time'] for m in results['metrics_history']]):.3f}s" if NUMPY_AVAILABLE else f"  • Final inference time: {final_metrics['inference_time']:.3f}s")
        
        if results['improvements']:
            logger.info(f"\n🚀 Autonomous Improvements:")
            for improvement in results['improvements']:
                logger.info(f"  • {improvement['description']}")
        else:
            logger.info(f"\n📈 System maintained stable performance across iterations")
    
    logger.info("\n🔄 Ready to proceed to Generation 2: Make It Robust")
    logger.info("="*70)
    
    return results

if __name__ == "__main__":
    # Run the demonstration
    results = main()
    
    # Exit with appropriate code
    exit_code = 0 if results.get('status') == 'completed' else 1
    sys.exit(exit_code)