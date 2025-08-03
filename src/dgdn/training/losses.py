"""Loss functions for DGDN training."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Any
import math


class DGDNLoss(nn.Module):
    """Comprehensive loss function for DGDN training.
    
    Combines multiple loss components:
    - Reconstruction loss (task-specific)
    - Variational loss (KL divergence)
    - Temporal regularization loss
    - Diffusion loss (for proper denoising)
    """
    
    def __init__(
        self,
        reconstruction_weight: float = 1.0,
        kl_weight: float = 0.1,
        temporal_weight: float = 0.05,
        diffusion_weight: float = 0.01,
        task: str = "edge_prediction"
    ):
        super().__init__()
        
        self.reconstruction_weight = reconstruction_weight
        self.kl_weight = kl_weight
        self.temporal_weight = temporal_weight
        self.diffusion_weight = diffusion_weight
        self.task = task
        
        # Initialize component losses
        self.variational_loss = VariationalLoss()
        self.temporal_loss = TemporalRegularizationLoss()
        self.diffusion_loss = DiffusionLoss()
        
    def forward(
        self,
        model_output: Dict[str, torch.Tensor],
        targets: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Compute total loss and components.
        
        Args:
            model_output: Dictionary from DGDN forward pass
            targets: Ground truth targets
            edge_index: Edge connectivity for edge prediction tasks
            
        Returns:
            Dictionary of loss components
        """
        losses = {}
        
        # 1. Reconstruction loss (task-specific)
        recon_loss = self._compute_reconstruction_loss(
            model_output, targets, edge_index
        )
        losses["reconstruction"] = recon_loss
        
        # 2. Variational loss (KL divergence)
        if "mean" in model_output and "logvar" in model_output:
            var_loss = self.variational_loss(
                model_output["mean"], 
                model_output["logvar"]
            )
            losses["variational"] = var_loss
        else:
            losses["variational"] = torch.tensor(0.0, device=recon_loss.device)
        
        # 3. Temporal regularization
        if "temporal_encoding" in model_output:
            temp_loss = self.temporal_loss(model_output["temporal_encoding"])
            losses["temporal"] = temp_loss
        else:
            losses["temporal"] = torch.tensor(0.0, device=recon_loss.device)
        
        # 4. Diffusion loss (if diffusion steps available)
        if "diffusion_steps" in model_output:
            diff_loss = self.diffusion_loss(model_output["diffusion_steps"])
            losses["diffusion"] = diff_loss
        else:
            losses["diffusion"] = torch.tensor(0.0, device=recon_loss.device)
        
        # 5. Total weighted loss
        total_loss = (
            self.reconstruction_weight * losses["reconstruction"] +
            self.kl_weight * losses["variational"] +
            self.temporal_weight * losses["temporal"] +
            self.diffusion_weight * losses["diffusion"]
        )
        
        losses["total"] = total_loss
        
        return losses
    
    def _compute_reconstruction_loss(
        self,
        model_output: Dict[str, torch.Tensor],
        targets: torch.Tensor,
        edge_index: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Compute task-specific reconstruction loss."""
        
        if self.task == "edge_prediction":
            return self._edge_prediction_loss(model_output, targets, edge_index)
        elif self.task == "node_classification":
            return self._node_classification_loss(model_output, targets)
        elif self.task == "link_prediction":
            return self._link_prediction_loss(model_output, targets, edge_index)
        else:
            raise ValueError(f"Unknown task: {self.task}")
    
    def _edge_prediction_loss(
        self,
        model_output: Dict[str, torch.Tensor],
        targets: torch.Tensor,
        edge_index: torch.Tensor
    ) -> torch.Tensor:
        """Binary cross-entropy loss for edge prediction."""
        node_embeddings = model_output["node_embeddings"]
        
        # Get source and target node embeddings
        src_embeddings = node_embeddings[edge_index[0]]
        tgt_embeddings = node_embeddings[edge_index[1]]
        
        # Compute edge scores (dot product)
        edge_scores = torch.sum(src_embeddings * tgt_embeddings, dim=-1)
        edge_probs = torch.sigmoid(edge_scores)
        
        # Binary cross-entropy loss
        return F.binary_cross_entropy(edge_probs, targets.float())
    
    def _node_classification_loss(
        self,
        model_output: Dict[str, torch.Tensor],
        targets: torch.Tensor
    ) -> torch.Tensor:
        """Cross-entropy loss for node classification.""" 
        node_embeddings = model_output["node_embeddings"]
        
        # Simple linear classifier for demonstration
        # In practice, this would use the model's node classifier
        num_classes = targets.max().item() + 1
        classifier = nn.Linear(
            node_embeddings.shape[-1], 
            num_classes,
            device=node_embeddings.device
        )
        
        logits = classifier(node_embeddings)
        return F.cross_entropy(logits, targets.long())
    
    def _link_prediction_loss(
        self,
        model_output: Dict[str, torch.Tensor],
        targets: torch.Tensor,
        edge_index: torch.Tensor
    ) -> torch.Tensor:
        """Loss for link prediction with positive and negative samples."""
        node_embeddings = model_output["node_embeddings"]
        
        # Positive edges
        pos_src = node_embeddings[edge_index[0]]
        pos_tgt = node_embeddings[edge_index[1]]
        pos_scores = torch.sum(pos_src * pos_tgt, dim=-1)
        
        # Negative sampling (simplified)
        num_neg = edge_index.shape[1]
        neg_src_idx = torch.randint(0, node_embeddings.shape[0], (num_neg,))
        neg_tgt_idx = torch.randint(0, node_embeddings.shape[0], (num_neg,))
        
        neg_src = node_embeddings[neg_src_idx]
        neg_tgt = node_embeddings[neg_tgt_idx]
        neg_scores = torch.sum(neg_src * neg_tgt, dim=-1)
        
        # BPR loss (Bayesian Personalized Ranking)
        loss = -torch.log(torch.sigmoid(pos_scores - neg_scores)).mean()
        
        return loss


class VariationalLoss(nn.Module):
    """KL divergence loss for variational inference."""
    
    def __init__(self, beta: float = 1.0):
        super().__init__()
        self.beta = beta
    
    def forward(
        self, 
        mean: torch.Tensor, 
        logvar: torch.Tensor,
        prior_mean: Optional[torch.Tensor] = None,
        prior_logvar: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute KL divergence loss.
        
        Args:
            mean: Posterior mean [batch_size, hidden_dim]
            logvar: Posterior log variance [batch_size, hidden_dim]
            prior_mean: Prior mean (defaults to zero)
            prior_logvar: Prior log variance (defaults to zero)
            
        Returns:
            KL divergence loss
        """
        if prior_mean is None:
            prior_mean = torch.zeros_like(mean)
        if prior_logvar is None:
            prior_logvar = torch.zeros_like(logvar)
        
        # KL(q||p) = 0.5 * [log(σ_p²/σ_q²) + (σ_q² + (μ_q - μ_p)²)/σ_p² - 1]
        var = torch.exp(logvar)
        prior_var = torch.exp(prior_logvar)
        
        kl_div = 0.5 * (
            prior_logvar - logvar +
            (var + (mean - prior_mean) ** 2) / prior_var - 1
        )
        
        return self.beta * kl_div.sum(dim=-1).mean()


class TemporalRegularizationLoss(nn.Module):
    """Temporal smoothness regularization loss."""
    
    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha
    
    def forward(self, temporal_encoding: torch.Tensor) -> torch.Tensor:
        """Encourage smooth temporal transitions.
        
        Args:
            temporal_encoding: Temporal encodings [num_edges, time_dim]
            
        Returns:
            Temporal regularization loss
        """
        if temporal_encoding.shape[0] <= 1:
            return torch.tensor(0.0, device=temporal_encoding.device)
        
        # Compute differences between consecutive time encodings
        temporal_diff = temporal_encoding[1:] - temporal_encoding[:-1]
        
        # L2 regularization on differences
        smooth_loss = torch.mean(torch.sum(temporal_diff ** 2, dim=-1))
        
        return self.alpha * smooth_loss


class DiffusionLoss(nn.Module):
    """Loss for proper diffusion process training."""
    
    def __init__(self, noise_schedule: str = "linear"):
        super().__init__()
        self.noise_schedule = noise_schedule
    
    def forward(self, diffusion_steps: list) -> torch.Tensor:
        """Compute diffusion denoising loss.
        
        Args:
            diffusion_steps: List of diffusion step outputs
            
        Returns:
            Diffusion loss
        """
        if not diffusion_steps:
            return torch.tensor(0.0)
        
        total_loss = 0.0
        num_steps = len(diffusion_steps)
        
        for i, step_output in enumerate(diffusion_steps):
            # Loss for each diffusion step
            # Encourage proper denoising behavior
            z = step_output["z"]
            mean = step_output["mean"]
            
            # MSE between sampled z and mean (encourage low variance when appropriate)
            step_loss = F.mse_loss(z, mean)
            
            # Weight by step position (later steps should be more accurate)
            weight = (i + 1) / num_steps
            total_loss += weight * step_loss
        
        return total_loss / num_steps


class ContrastiveLoss(nn.Module):
    """Contrastive loss for temporal graph representation learning."""
    
    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature
    
    def forward(
        self,
        embeddings1: torch.Tensor,
        embeddings2: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """Compute contrastive loss between embedding pairs.
        
        Args:
            embeddings1: First set of embeddings
            embeddings2: Second set of embeddings  
            labels: Binary labels (1 for positive pairs, 0 for negative)
            
        Returns:
            Contrastive loss
        """
        # Normalize embeddings
        embeddings1 = F.normalize(embeddings1, dim=-1)
        embeddings2 = F.normalize(embeddings2, dim=-1)
        
        # Compute similarities
        similarities = torch.sum(embeddings1 * embeddings2, dim=-1) / self.temperature
        
        # Contrastive loss
        pos_loss = labels * torch.exp(similarities)
        neg_loss = (1 - labels) * torch.exp(-similarities)
        
        loss = -torch.log(pos_loss / (pos_loss + neg_loss + 1e-8))
        
        return loss.mean()


class AdversarialLoss(nn.Module):
    """Adversarial loss for robustness training."""
    
    def __init__(self, epsilon: float = 0.1):
        super().__init__()
        self.epsilon = epsilon
    
    def forward(
        self,
        model,
        data,
        targets: torch.Tensor,
        base_loss_fn: nn.Module
    ) -> torch.Tensor:
        """Compute adversarial loss using FGSM attack.
        
        Args:
            model: DGDN model
            data: Input data
            targets: Ground truth targets
            base_loss_fn: Base loss function
            
        Returns:
            Adversarial loss
        """
        # Require gradients for input features
        if hasattr(data, 'node_features') and data.node_features is not None:
            data.node_features.requires_grad_(True)
        
        # Forward pass
        output = model(data)
        
        # Compute loss
        loss = base_loss_fn(output, targets)
        
        # Compute gradients
        loss.backward(retain_graph=True)
        
        # Generate adversarial perturbation
        if data.node_features.grad is not None:
            perturbation = self.epsilon * torch.sign(data.node_features.grad)
            
            # Apply perturbation
            perturbed_features = data.node_features + perturbation
            
            # Create perturbed data
            perturbed_data = type(data)(
                edge_index=data.edge_index,
                timestamps=data.timestamps,
                node_features=perturbed_features,
                edge_attr=getattr(data, 'edge_attr', None),
                num_nodes=data.num_nodes
            )
            
            # Forward pass with perturbed data
            adv_output = model(perturbed_data)
            
            # Adversarial loss
            adv_loss = base_loss_fn(adv_output, targets)
            
            return adv_loss
        
        return loss