"""Evaluation metrics for DGDN training."""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score
import torch.nn.functional as F


class DGDNMetrics:
    """Comprehensive metrics for DGDN evaluation."""
    
    def __init__(self, task: str = "edge_prediction"):
        self.task = task
        self.reset()
    
    def reset(self):
        """Reset all metrics."""
        self.predictions = []
        self.targets = []
        self.uncertainties = []
        self.losses = []
    
    def update(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        uncertainties: Optional[torch.Tensor] = None,
        loss: Optional[float] = None
    ):
        """Update metrics with new batch."""
        self.predictions.append(predictions.detach().cpu())
        self.targets.append(targets.detach().cpu())
        
        if uncertainties is not None:
            self.uncertainties.append(uncertainties.detach().cpu())
        
        if loss is not None:
            self.losses.append(loss)
    
    def compute(self) -> Dict[str, float]:
        """Compute all metrics."""
        if not self.predictions:
            return {}
        
        # Concatenate all batches
        predictions = torch.cat(self.predictions, dim=0)
        targets = torch.cat(self.targets, dim=0)
        
        metrics = {}
        
        # Loss metrics
        if self.losses:
            metrics["loss"] = np.mean(self.losses)
        
        # Task-specific metrics
        if self.task == "edge_prediction":
            metrics.update(self._compute_edge_prediction_metrics(predictions, targets))
        elif self.task == "node_classification":
            metrics.update(self._compute_node_classification_metrics(predictions, targets))
        
        # Uncertainty metrics
        if self.uncertainties:
            uncertainties = torch.cat(self.uncertainties, dim=0)
            metrics.update(self._compute_uncertainty_metrics(predictions, targets, uncertainties))
        
        return metrics
    
    def _compute_edge_prediction_metrics(
        self, 
        predictions: torch.Tensor, 
        targets: torch.Tensor
    ) -> Dict[str, float]:
        """Compute edge prediction metrics."""
        # Convert to numpy for sklearn
        if predictions.dim() > 1:
            # Multi-class case
            pred_probs = F.softmax(predictions, dim=-1)[:, 1].numpy()
            pred_labels = torch.argmax(predictions, dim=-1).numpy()
        else:
            # Binary case
            pred_probs = torch.sigmoid(predictions).numpy()
            pred_labels = (pred_probs > 0.5).astype(int)
        
        targets_np = targets.numpy()
        
        metrics = {}
        
        # AUC-ROC
        if len(np.unique(targets_np)) > 1:
            metrics["auc"] = roc_auc_score(targets_np, pred_probs)
            metrics["ap"] = average_precision_score(targets_np, pred_probs)
        
        # Accuracy
        metrics["accuracy"] = accuracy_score(targets_np, pred_labels)
        
        # F1 Score
        metrics["f1"] = f1_score(targets_np, pred_labels, average='binary')
        
        # Mean Reciprocal Rank (MRR)
        if len(pred_probs) > 1:
            sorted_indices = np.argsort(-pred_probs)
            ranks = np.where(targets_np[sorted_indices] == 1)[0]
            if len(ranks) > 0:
                mrr = np.mean(1.0 / (ranks + 1))
                metrics["mrr"] = mrr
        
        return metrics
    
    def _compute_node_classification_metrics(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> Dict[str, float]:
        """Compute node classification metrics."""
        pred_labels = torch.argmax(predictions, dim=-1).numpy()
        targets_np = targets.numpy()
        
        metrics = {
            "accuracy": accuracy_score(targets_np, pred_labels),
            "f1_micro": f1_score(targets_np, pred_labels, average='micro'),
            "f1_macro": f1_score(targets_np, pred_labels, average='macro'),
        }
        
        # Multi-class AUC if applicable
        if predictions.shape[-1] > 2:
            pred_probs = F.softmax(predictions, dim=-1).numpy()
            try:
                metrics["auc_macro"] = roc_auc_score(
                    targets_np, pred_probs, multi_class='ovr', average='macro'
                )
            except ValueError:
                pass  # Skip if not enough classes
        
        return metrics
    
    def _compute_uncertainty_metrics(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        uncertainties: torch.Tensor
    ) -> Dict[str, float]:
        """Compute uncertainty calibration metrics."""
        uncertainties_np = uncertainties.numpy()
        
        metrics = {
            "mean_uncertainty": np.mean(uncertainties_np),
            "std_uncertainty": np.std(uncertainties_np)
        }
        
        # Calibration metrics
        if self.task == "edge_prediction":
            pred_probs = torch.sigmoid(predictions).numpy()
            targets_np = targets.numpy()
            
            # Expected Calibration Error (ECE)
            ece = self._compute_ece(pred_probs, targets_np, uncertainties_np)
            metrics["ece"] = ece
            
            # Reliability metrics
            reliability = self._compute_reliability(pred_probs, targets_np, uncertainties_np)
            metrics.update(reliability)
        
        return metrics
    
    def _compute_ece(
        self,
        confidences: np.ndarray,
        accuracies: np.ndarray,
        uncertainties: np.ndarray,
        n_bins: int = 10
    ) -> float:
        """Compute Expected Calibration Error."""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Samples in this bin
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = accuracies[in_bin].mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return ece
    
    def _compute_reliability(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        uncertainties: np.ndarray
    ) -> Dict[str, float]:
        """Compute reliability metrics."""
        # Prediction accuracy vs uncertainty correlation
        errors = np.abs(predictions - targets)
        correlation = np.corrcoef(errors, uncertainties)[0, 1]
        
        return {
            "uncertainty_error_correlation": correlation if not np.isnan(correlation) else 0.0,
            "mean_error": np.mean(errors),
            "error_std": np.std(errors)
        }


class EdgePredictionMetrics(DGDNMetrics):
    """Specialized metrics for edge prediction tasks."""
    
    def __init__(self):
        super().__init__(task="edge_prediction")
    
    def compute_ranking_metrics(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        k_values: List[int] = [1, 5, 10, 20]
    ) -> Dict[str, float]:
        """Compute ranking-based metrics."""
        pred_probs = torch.sigmoid(predictions).numpy()
        targets_np = targets.numpy()
        
        # Sort by prediction scores
        sorted_indices = np.argsort(-pred_probs)
        sorted_targets = targets_np[sorted_indices]
        
        metrics = {}
        
        # Hits@K
        for k in k_values:
            if k <= len(sorted_targets):
                hits_at_k = np.sum(sorted_targets[:k]) / min(k, np.sum(targets_np))
                metrics[f"hits@{k}"] = hits_at_k
        
        # NDCG@K
        for k in k_values:
            if k <= len(sorted_targets):
                ndcg_at_k = self._compute_ndcg(sorted_targets[:k])
                metrics[f"ndcg@{k}"] = ndcg_at_k
        
        return metrics
    
    def _compute_ndcg(self, relevance_scores: np.ndarray) -> float:
        """Compute Normalized Discounted Cumulative Gain."""
        if len(relevance_scores) == 0:
            return 0.0
        
        # DCG
        dcg = relevance_scores[0]
        for i in range(1, len(relevance_scores)):
            dcg += relevance_scores[i] / np.log2(i + 1)
        
        # IDCG (perfect ranking)
        ideal_relevance = np.sort(relevance_scores)[::-1]
        idcg = ideal_relevance[0]
        for i in range(1, len(ideal_relevance)):
            idcg += ideal_relevance[i] / np.log2(i + 1)
        
        return dcg / idcg if idcg > 0 else 0.0


class NodeClassificationMetrics(DGDNMetrics):
    """Specialized metrics for node classification tasks."""
    
    def __init__(self, num_classes: int = 2):
        super().__init__(task="node_classification")
        self.num_classes = num_classes
    
    def compute_per_class_metrics(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> Dict[str, float]:
        """Compute per-class metrics."""
        pred_labels = torch.argmax(predictions, dim=-1).numpy()
        targets_np = targets.numpy()
        
        metrics = {}
        
        for class_id in range(self.num_classes):
            class_mask = targets_np == class_id
            if np.sum(class_mask) > 0:
                class_pred = pred_labels[class_mask]
                class_target = targets_np[class_mask]
                
                accuracy = accuracy_score(class_target, class_pred)
                metrics[f"class_{class_id}_accuracy"] = accuracy
                
                # Precision and Recall
                if len(np.unique(class_pred)) > 1:
                    f1 = f1_score(class_target, class_pred, average='binary', pos_label=class_id)
                    metrics[f"class_{class_id}_f1"] = f1
        
        return metrics


class TemporalMetrics:
    """Metrics for evaluating temporal aspects of predictions."""
    
    def __init__(self):
        self.temporal_predictions = []
        self.temporal_targets = []
        self.timestamps = []
    
    def update(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        timestamps: torch.Tensor
    ):
        """Update with temporal predictions."""
        self.temporal_predictions.append(predictions.detach().cpu())
        self.temporal_targets.append(targets.detach().cpu())
        self.timestamps.append(timestamps.detach().cpu())
    
    def compute_temporal_stability(self) -> Dict[str, float]:
        """Compute temporal stability metrics."""
        if not self.temporal_predictions:
            return {}
        
        predictions = torch.cat(self.temporal_predictions, dim=0)
        timestamps = torch.cat(self.timestamps, dim=0)
        
        # Sort by timestamp
        sorted_indices = torch.argsort(timestamps)
        sorted_predictions = predictions[sorted_indices]
        
        # Compute prediction variance over time
        if len(sorted_predictions) > 1:
            pred_diffs = sorted_predictions[1:] - sorted_predictions[:-1]
            temporal_variance = torch.var(pred_diffs).item()
            temporal_smoothness = torch.mean(torch.abs(pred_diffs)).item()
            
            return {
                "temporal_variance": temporal_variance,
                "temporal_smoothness": temporal_smoothness
            }
        
        return {}
    
    def compute_prediction_drift(self, window_size: int = 100) -> Dict[str, float]:
        """Compute prediction drift over time windows."""
        if not self.temporal_predictions:
            return {}
        
        predictions = torch.cat(self.temporal_predictions, dim=0)
        timestamps = torch.cat(self.timestamps, dim=0)
        
        # Sort by timestamp
        sorted_indices = torch.argsort(timestamps)
        sorted_predictions = predictions[sorted_indices]
        
        if len(sorted_predictions) < window_size * 2:
            return {}
        
        # Compare early and late windows
        early_window = sorted_predictions[:window_size]
        late_window = sorted_predictions[-window_size:]
        
        # Compute drift metrics
        mean_drift = torch.abs(torch.mean(late_window) - torch.mean(early_window)).item()
        std_drift = torch.abs(torch.std(late_window) - torch.std(early_window)).item()
        
        return {
            "mean_drift": mean_drift,
            "std_drift": std_drift
        }


class UncertaintyMetrics:
    """Specialized metrics for uncertainty quantification."""
    
    def __init__(self):
        self.predictions = []
        self.uncertainties = []
        self.targets = []
    
    def update(
        self,
        predictions: torch.Tensor,
        uncertainties: torch.Tensor,
        targets: torch.Tensor
    ):
        """Update uncertainty metrics."""
        self.predictions.append(predictions.detach().cpu())
        self.uncertainties.append(uncertainties.detach().cpu())
        self.targets.append(targets.detach().cpu())
    
    def compute_calibration_metrics(self) -> Dict[str, float]:
        """Compute uncertainty calibration metrics."""
        if not self.predictions:
            return {}
        
        predictions = torch.cat(self.predictions, dim=0)
        uncertainties = torch.cat(self.uncertainties, dim=0)
        targets = torch.cat(self.targets, dim=0)
        
        # Convert to numpy
        pred_np = predictions.numpy()
        unc_np = uncertainties.numpy()
        targets_np = targets.numpy()
        
        # Compute errors
        errors = np.abs(pred_np - targets_np)
        
        # Uncertainty-error correlation
        correlation = np.corrcoef(errors, unc_np)[0, 1]
        
        # Calibration curve
        n_bins = 10
        bin_edges = np.percentile(unc_np, np.linspace(0, 100, n_bins + 1))
        
        calibration_error = 0.0
        for i in range(n_bins):
            mask = (unc_np >= bin_edges[i]) & (unc_np < bin_edges[i + 1])
            if np.sum(mask) > 0:
                bin_uncertainty = np.mean(unc_np[mask])
                bin_error = np.mean(errors[mask])
                calibration_error += np.abs(bin_uncertainty - bin_error)
        
        calibration_error /= n_bins
        
        return {
            "uncertainty_error_correlation": correlation if not np.isnan(correlation) else 0.0,
            "calibration_error": calibration_error,
            "mean_uncertainty": np.mean(unc_np),
            "mean_error": np.mean(errors)
        }