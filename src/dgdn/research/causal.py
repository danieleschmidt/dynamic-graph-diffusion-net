"""Causal discovery and inference for temporal graphs."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import numpy as np

from ..models.dgdn import DynamicGraphDiffusionNet


class CausalDGDN(DynamicGraphDiffusionNet):
    """DGDN with causal discovery and intervention capabilities."""
    
    def __init__(self, *args, causal_discovery: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.causal_discovery = causal_discovery
        
        # Causal mechanism networks
        self.causal_encoder = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2)
        )
        
        # Intervention network
        self.intervention_network = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim * 2),
            nn.ReLU(), 
            nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        )
        
        # Causal graph adjacency matrix (learnable)
        max_nodes = kwargs.get('max_nodes', 1000)
        self.causal_adjacency = nn.Parameter(
            torch.zeros(max_nodes, max_nodes),
            requires_grad=True
        )
        
        # Temporal causal delays
        self.temporal_delays = nn.Parameter(
            torch.zeros(max_nodes, max_nodes),
            requires_grad=True
        )
        
    def discover_causal_structure(self, data, lambda_sparsity: float = 0.1):
        """Discover causal structure in temporal graph."""
        # Forward pass to get embeddings
        output = self.forward(data)
        node_embeddings = output['node_embeddings']
        
        # Encode causal mechanisms
        causal_features = self.causal_encoder(node_embeddings)
        
        # Learn causal adjacency matrix
        num_nodes = node_embeddings.size(0)
        causal_adj = self.causal_adjacency[:num_nodes, :num_nodes]
        
        # Apply sparsity regularization
        sparsity_loss = lambda_sparsity * torch.norm(causal_adj, p=1)
        
        # Acyclicity constraint (DAG constraint)
        # h(A) = tr(e^(A ⊙ A)) - d = 0 for DAG
        eye = torch.eye(num_nodes, device=causal_adj.device)
        expm_A = torch.matrix_exp(causal_adj * causal_adj)
        h_A = torch.trace(expm_A) - num_nodes
        
        # Causal discovery loss
        causal_loss = self.compute_causal_loss(
            causal_features, causal_adj, data
        )
        
        return {
            'causal_adjacency': causal_adj,
            'causal_features': causal_features,
            'losses': {
                'causal': causal_loss,
                'sparsity': sparsity_loss,
                'acyclicity': h_A ** 2  # Penalty for non-DAG
            }
        }
        
    def compute_causal_loss(self, causal_features, causal_adj, data):
        """Compute causal discovery loss."""
        num_nodes = causal_features.size(0)
        
        # Structural equation model loss
        total_loss = 0
        for i in range(num_nodes):
            # Parents of node i
            parents = causal_adj[:, i]
            
            # Predicted value from parents
            parent_contributions = torch.sum(
                parents.unsqueeze(0) * causal_features, dim=1
            )
            
            # Observed value
            observed = causal_features[i]
            
            # Reconstruction loss
            recon_loss = F.mse_loss(parent_contributions, observed)
            total_loss += recon_loss
            
        return total_loss / num_nodes
        
    def perform_intervention(self, data, intervention_nodes: List[int], 
                           intervention_values: torch.Tensor):
        """Perform causal intervention on specified nodes."""
        # Get original embeddings
        output = self.forward(data)
        original_embeddings = output['node_embeddings']
        
        # Apply interventions
        intervened_embeddings = original_embeddings.clone()
        for node_id, value in zip(intervention_nodes, intervention_values):
            # Intervene on node embedding
            intervened_embeddings[node_id] = self.intervention_network(value)
            
        # Propagate intervention effects through causal graph
        num_nodes = original_embeddings.size(0)
        causal_adj = self.causal_adjacency[:num_nodes, :num_nodes]
        
        # Iterative propagation
        for _ in range(5):  # Multiple propagation steps
            new_embeddings = intervened_embeddings.clone()
            for i in range(num_nodes):
                if i not in intervention_nodes:  # Don't change intervened nodes
                    parents = causal_adj[:, i]
                    parent_effects = torch.sum(
                        parents.unsqueeze(-1) * intervened_embeddings, dim=0
                    )
                    new_embeddings[i] = parent_effects
                    
            intervened_embeddings = new_embeddings
            
        return {
            'original_embeddings': original_embeddings,
            'intervened_embeddings': intervened_embeddings,
            'intervention_effect': intervened_embeddings - original_embeddings
        }
        
    def estimate_causal_effect(self, data, treatment_nodes: List[int], 
                              outcome_nodes: List[int]):
        """Estimate average treatment effect (ATE)."""
        # Control (no intervention)
        control_output = self.forward(data)
        
        # Treatment (intervention)
        treatment_values = torch.randn(
            len(treatment_nodes), self.hidden_dim,
            device=data.x.device
        )
        
        treatment_output = self.perform_intervention(
            data, treatment_nodes, treatment_values
        )
        
        # Compute ATE
        control_outcomes = control_output['node_embeddings'][outcome_nodes]
        treatment_outcomes = treatment_output['intervened_embeddings'][outcome_nodes]
        
        ate = (treatment_outcomes - control_outcomes).mean()
        
        return {
            'ate': ate,
            'control_outcomes': control_outcomes,
            'treatment_outcomes': treatment_outcomes
        }


class CausalDiscovery:
    """Standalone causal discovery module for temporal graphs."""
    
    def __init__(self, max_nodes: int = 1000):
        self.max_nodes = max_nodes
        
    def notears_discovery(self, data, lambda_1: float = 0.01, lambda_2: float = 0.01):
        """NO-TEARS algorithm for causal discovery."""
        # Extract node features across time
        X = self._extract_temporal_features(data)
        
        # Initialize adjacency matrix
        W = torch.randn(X.size(1), X.size(1), requires_grad=True)
        
        # Optimization
        optimizer = torch.optim.Adam([W], lr=0.01)
        
        for epoch in range(1000):
            optimizer.zero_grad()
            
            # Least squares loss
            ls_loss = 0.5 * torch.norm(X - X @ W) ** 2
            
            # L1 regularization (sparsity)
            l1_reg = lambda_1 * torch.norm(W, p=1)
            
            # L2 regularization  
            l2_reg = lambda_2 * torch.norm(W, p='fro') ** 2
            
            # DAG constraint
            h = self._h_func(W)
            
            # Total loss
            loss = ls_loss + l1_reg + l2_reg + 0.5 * h ** 2
            
            loss.backward()
            optimizer.step()
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item():.4f}, h: {h.item():.4f}")
                
        return W.detach()
        
    def _extract_temporal_features(self, data):
        """Extract temporal node features for causal discovery."""
        timestamps = data.timestamps.unique().sort()[0]
        num_nodes = data.x.size(0)
        
        # Create feature matrix with temporal snapshots
        features = []
        for t in timestamps:
            mask = data.timestamps == t
            if mask.sum() > 0:
                # Get node features at time t
                node_feats = torch.zeros(num_nodes, data.x.size(1))
                active_nodes = data.edge_index[:, mask].unique()
                node_feats[active_nodes] = data.x[active_nodes]
                features.append(node_feats.mean(dim=1))  # Aggregate features
                
        return torch.stack(features, dim=0)  # [time_steps, nodes]
        
    def _h_func(self, W):
        """DAG constraint function h(W) = tr(e^(W ⊙ W)) - d"""
        d = W.size(0)
        A = W * W
        expm_A = torch.matrix_exp(A)
        return torch.trace(expm_A) - d
        
    def granger_causality(self, data, max_lag: int = 5):
        """Granger causality test for temporal graphs."""
        X = self._extract_temporal_features(data)
        T, n = X.shape
        
        causality_matrix = torch.zeros(n, n)
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    # Test if j Granger-causes i
                    granger_stat = self._granger_test(X[:, i], X[:, j], max_lag)
                    causality_matrix[j, i] = granger_stat
                    
        return causality_matrix
        
    def _granger_test(self, y, x, max_lag: int):
        """Perform Granger causality test between two time series."""
        T = y.size(0)
        
        # Create lagged variables
        Y = []
        X = []
        
        for t in range(max_lag, T):
            y_lags = [y[t-l] for l in range(1, max_lag + 1)]
            x_lags = [x[t-l] for l in range(1, max_lag + 1)]
            
            Y.append(y[t])
            X.append(torch.cat([torch.stack(y_lags), torch.stack(x_lags)]))
            
        if len(Y) == 0:
            return 0.0
            
        Y = torch.stack(Y)
        X = torch.stack(X)
        
        # Restricted model (without x)
        X_restricted = X[:, :max_lag]  # Only y lags
        
        # Fit models
        beta_full = torch.linalg.lstsq(X, Y)[0]
        beta_restricted = torch.linalg.lstsq(X_restricted, Y)[0]
        
        # Compute F-statistic
        rss_full = torch.sum((Y - X @ beta_full) ** 2)
        rss_restricted = torch.sum((Y - X_restricted @ beta_restricted) ** 2)
        
        f_stat = ((rss_restricted - rss_full) / max_lag) / (rss_full / (X.size(0) - X.size(1)))
        
        return f_stat.item()
        
    def pc_algorithm(self, data, alpha: float = 0.05):
        """PC algorithm for causal discovery."""
        # Simplified PC algorithm implementation
        X = self._extract_temporal_features(data)
        n = X.size(1)
        
        # Initialize complete graph
        adjacency = torch.ones(n, n) - torch.eye(n)
        
        # Skeleton discovery phase
        for i in range(n):
            for j in range(i + 1, n):
                # Test conditional independence
                for k in range(n):
                    if k != i and k != j:
                        # Test independence of i and j given k
                        indep_stat = self._conditional_independence_test(X[:, i], X[:, j], X[:, k])
                        if indep_stat > alpha:  # Independent
                            adjacency[i, j] = 0
                            adjacency[j, i] = 0
                            break
                            
        return adjacency
        
    def _conditional_independence_test(self, x, y, z):
        """Test conditional independence of x and y given z."""
        # Partial correlation test
        # Regress x on z
        beta_xz = torch.linalg.lstsq(z.unsqueeze(1), x)[0]
        residual_x = x - z * beta_xz
        
        # Regress y on z  
        beta_yz = torch.linalg.lstsq(z.unsqueeze(1), y)[0]
        residual_y = y - z * beta_yz
        
        # Correlation of residuals
        corr = torch.corrcoef(torch.stack([residual_x, residual_y]))[0, 1]
        
        # Fisher's z-transformation
        n = x.size(0)
        z_score = 0.5 * torch.log((1 + corr) / (1 - corr)) * torch.sqrt(n - 3)
        
        # P-value (approximate)
        p_value = 2 * (1 - torch.erf(torch.abs(z_score) / torch.sqrt(2)))
        
        return p_value.item()