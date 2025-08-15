"""
Adaptive learning systems that continuously improve model performance.
"""

import torch
import torch.nn as nn
import numpy as np
import time
import logging
from typing import Dict, Any, List, Optional, Tuple, Callable
from dataclasses import dataclass
from collections import deque
import pickle
import threading

@dataclass
class LearningMetrics:
    """Metrics for adaptive learning."""
    epoch: int
    loss: float
    accuracy: float
    learning_rate: float
    convergence_rate: float
    adaptation_score: float
    timestamp: float

class AdaptiveLearningSystem:
    """
    Advanced adaptive learning system that automatically adjusts training strategies
    based on performance patterns and data characteristics.
    """
    
    def __init__(self, 
                 base_learning_rate: float = 1e-3,
                 adaptation_window: int = 10,
                 convergence_threshold: float = 1e-6,
                 patience: int = 20):
        
        self.base_learning_rate = base_learning_rate
        self.adaptation_window = adaptation_window
        self.convergence_threshold = convergence_threshold
        self.patience = patience
        
        # Learning history
        self.learning_history = deque(maxlen=1000)
        self.performance_metrics = deque(maxlen=adaptation_window)
        
        # Adaptive strategies
        self.strategies = {
            "learning_rate_schedule": self._adaptive_lr_schedule,
            "batch_size_adaptation": self._adaptive_batch_size,
            "architecture_adaptation": self._adaptive_architecture,
            "regularization_adaptation": self._adaptive_regularization
        }
        
        # Current strategy parameters
        self.current_lr = base_learning_rate
        self.current_batch_size = 32
        self.regularization_strength = 0.01
        
        self.logger = logging.getLogger(f'{__name__}.AdaptiveLearningSystem')
        
        # Meta-learning components
        self.meta_learner = MetaLearner()
        self.online_learner = OnlineLearner()
        
    def adapt_training(self, model: nn.Module, 
                      optimizer: torch.optim.Optimizer,
                      current_loss: float,
                      current_accuracy: float,
                      epoch: int) -> Dict[str, Any]:
        """
        Adapt training strategy based on current performance.
        
        Args:
            model: Current model
            optimizer: Current optimizer
            current_loss: Current training loss
            current_accuracy: Current accuracy
            epoch: Current epoch
            
        Returns:
            Adaptation recommendations
        """
        # Calculate convergence rate
        convergence_rate = self._calculate_convergence_rate()
        
        # Calculate adaptation score
        adaptation_score = self._calculate_adaptation_score(current_loss, current_accuracy)
        
        # Record metrics
        metrics = LearningMetrics(
            epoch=epoch,
            loss=current_loss,
            accuracy=current_accuracy,
            learning_rate=self.current_lr,
            convergence_rate=convergence_rate,
            adaptation_score=adaptation_score,
            timestamp=time.time()
        )
        
        self.learning_history.append(metrics)
        self.performance_metrics.append(metrics)
        
        # Apply adaptive strategies
        adaptations = {}
        
        # 1. Learning rate adaptation
        new_lr = self._adaptive_lr_schedule(metrics)
        if abs(new_lr - self.current_lr) > 1e-6:
            adaptations["learning_rate"] = new_lr
            self.current_lr = new_lr
            
            # Update optimizer learning rate
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lr
        
        # 2. Batch size adaptation
        new_batch_size = self._adaptive_batch_size(metrics)
        if new_batch_size != self.current_batch_size:
            adaptations["batch_size"] = new_batch_size
            self.current_batch_size = new_batch_size
        
        # 3. Regularization adaptation
        new_reg = self._adaptive_regularization(metrics)
        if abs(new_reg - self.regularization_strength) > 1e-6:
            adaptations["regularization"] = new_reg
            self.regularization_strength = new_reg
        
        # 4. Architecture adaptation (if applicable)
        arch_changes = self._adaptive_architecture(metrics, model)
        if arch_changes:
            adaptations["architecture"] = arch_changes
        
        # Log adaptations
        if adaptations:
            self.logger.info(f"Epoch {epoch} adaptations: {adaptations}")
        
        return adaptations
    
    def _calculate_convergence_rate(self) -> float:
        """Calculate the convergence rate based on recent loss trends."""
        if len(self.performance_metrics) < 3:
            return 0.0
        
        recent_losses = [m.loss for m in list(self.performance_metrics)[-3:]]
        
        # Calculate rate of change
        deltas = [recent_losses[i] - recent_losses[i-1] for i in range(1, len(recent_losses))]
        avg_delta = np.mean(deltas)
        
        # Normalize by current loss to get relative convergence rate
        current_loss = recent_losses[-1]
        if current_loss > 0:
            return abs(avg_delta) / current_loss
        return 0.0
    
    def _calculate_adaptation_score(self, current_loss: float, current_accuracy: float) -> float:
        """Calculate adaptation score based on performance trends."""
        if len(self.performance_metrics) < 2:
            return 0.5  # Neutral score
        
        # Compare with previous metrics
        prev_metrics = list(self.performance_metrics)[-2]
        
        # Loss improvement (lower is better)
        loss_improvement = (prev_metrics.loss - current_loss) / max(prev_metrics.loss, 1e-8)
        
        # Accuracy improvement (higher is better)
        acc_improvement = current_accuracy - prev_metrics.accuracy
        
        # Combine improvements (0-1 scale, higher is better)
        adaptation_score = 0.5 + 0.3 * np.tanh(loss_improvement) + 0.2 * np.tanh(acc_improvement * 10)
        
        return max(0.0, min(1.0, adaptation_score))
    
    def _adaptive_lr_schedule(self, metrics: LearningMetrics) -> float:
        """Adaptive learning rate scheduling."""
        base_lr = self.base_learning_rate
        
        # Reduce LR if convergence is slow
        if metrics.convergence_rate < self.convergence_threshold:
            if len(self.performance_metrics) >= self.adaptation_window:
                recent_improvements = [
                    m.adaptation_score for m in list(self.performance_metrics)[-5:]
                ]
                avg_improvement = np.mean(recent_improvements)
                
                if avg_improvement < 0.3:  # Poor performance
                    return self.current_lr * 0.8  # Reduce LR
                elif avg_improvement > 0.7:  # Good performance
                    return min(self.current_lr * 1.1, base_lr * 2)  # Increase LR slightly
        
        # Cyclical learning rate component
        cycle_length = 100
        cycle_position = metrics.epoch % cycle_length
        cycle_factor = 0.5 * (1 + np.cos(np.pi * cycle_position / cycle_length))
        
        # Combine adaptive and cyclical components
        adaptive_lr = self.current_lr
        cyclical_component = base_lr * 0.1 * cycle_factor
        
        return adaptive_lr + cyclical_component
    
    def _adaptive_batch_size(self, metrics: LearningMetrics) -> int:
        """Adaptive batch size adjustment."""
        current_batch_size = self.current_batch_size
        
        # Increase batch size if training is stable and fast
        if metrics.adaptation_score > 0.7 and metrics.convergence_rate > self.convergence_threshold:
            return min(current_batch_size * 2, 1024)
        
        # Decrease batch size if training is unstable
        elif metrics.adaptation_score < 0.3:
            return max(current_batch_size // 2, 8)
        
        return current_batch_size
    
    def _adaptive_regularization(self, metrics: LearningMetrics) -> float:
        """Adaptive regularization strength."""
        current_reg = self.regularization_strength
        
        # Increase regularization if overfitting (high training accuracy, poor adaptation)
        if len(self.performance_metrics) >= 3:
            recent_accuracies = [m.accuracy for m in list(self.performance_metrics)[-3:]]
            recent_adaptations = [m.adaptation_score for m in list(self.performance_metrics)[-3:]]
            
            avg_accuracy = np.mean(recent_accuracies)
            avg_adaptation = np.mean(recent_adaptations)
            
            # High accuracy but poor adaptation might indicate overfitting
            if avg_accuracy > 0.9 and avg_adaptation < 0.4:
                return min(current_reg * 1.5, 0.1)
            
            # Low accuracy might need less regularization
            elif avg_accuracy < 0.6:
                return max(current_reg * 0.8, 1e-6)
        
        return current_reg
    
    def _adaptive_architecture(self, metrics: LearningMetrics, model: nn.Module) -> Optional[Dict[str, Any]]:
        """Adaptive architecture modifications."""
        # This is a placeholder for more sophisticated architecture adaptation
        # In practice, this could involve adding/removing layers, changing widths, etc.
        
        arch_changes = {}
        
        # Example: Suggest adding dropout if overfitting
        if metrics.adaptation_score < 0.3 and metrics.accuracy > 0.8:
            arch_changes["add_dropout"] = True
        
        # Example: Suggest increasing capacity if underfitting
        elif metrics.adaptation_score < 0.4 and metrics.accuracy < 0.6:
            arch_changes["increase_capacity"] = True
        
        return arch_changes if arch_changes else None
    
    def get_learning_summary(self) -> Dict[str, Any]:
        """Get summary of learning progress and adaptations."""
        if not self.learning_history:
            return {"status": "no_data"}
        
        history = list(self.learning_history)
        
        return {
            "total_epochs": len(history),
            "current_loss": history[-1].loss,
            "current_accuracy": history[-1].accuracy,
            "current_lr": self.current_lr,
            "avg_convergence_rate": np.mean([m.convergence_rate for m in history[-10:]]),
            "avg_adaptation_score": np.mean([m.adaptation_score for m in history[-10:]]),
            "learning_trend": "improving" if len(history) > 1 and history[-1].loss < history[-2].loss else "degrading"
        }

class OnlineLearner:
    """Online learning component for continuous adaptation."""
    
    def __init__(self, learning_rate: float = 1e-4, momentum: float = 0.9):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.gradients_buffer = {}
        self.velocity = {}
        
    def update_online(self, model: nn.Module, loss: torch.Tensor) -> None:
        """Perform online learning update."""
        # Compute gradients
        loss.backward(retain_graph=True)
        
        # Apply momentum-based updates
        for name, param in model.named_parameters():
            if param.grad is not None:
                if name not in self.velocity:
                    self.velocity[name] = torch.zeros_like(param.grad)
                
                # Momentum update
                self.velocity[name] = self.momentum * self.velocity[name] + self.learning_rate * param.grad
                param.data -= self.velocity[name]
    
    def adapt_to_distribution_shift(self, model: nn.Module, new_data: torch.Tensor) -> float:
        """Adapt model to distribution shift in new data."""
        # Simple adaptation score based on prediction confidence
        model.eval()
        with torch.no_grad():
            predictions = model(new_data)
            
            # Calculate entropy as measure of uncertainty
            if isinstance(predictions, dict):
                predictions = predictions.get('node_embeddings', predictions)
            
            # Simple uncertainty measure
            pred_std = torch.std(predictions, dim=-1).mean().item()
            adaptation_score = 1.0 / (1.0 + pred_std)  # Higher std = lower adaptation
            
            return adaptation_score

class MetaLearner:
    """Meta-learning component for learning how to learn."""
    
    def __init__(self, meta_lr: float = 1e-3):
        self.meta_lr = meta_lr
        self.strategy_performance = {}
        self.strategy_history = deque(maxlen=100)
        
    def evaluate_strategy(self, strategy_name: str, performance_before: float, 
                         performance_after: float) -> float:
        """Evaluate the effectiveness of an adaptation strategy."""
        improvement = performance_after - performance_before
        
        # Update strategy performance tracking
        if strategy_name not in self.strategy_performance:
            self.strategy_performance[strategy_name] = deque(maxlen=50)
        
        self.strategy_performance[strategy_name].append(improvement)
        
        # Calculate strategy effectiveness
        avg_improvement = np.mean(self.strategy_performance[strategy_name])
        return avg_improvement
    
    def recommend_strategy(self, current_metrics: LearningMetrics) -> str:
        """Recommend the best adaptation strategy based on current context."""
        if not self.strategy_performance:
            return "learning_rate_schedule"  # Default
        
        # Find strategy with best historical performance
        best_strategy = None
        best_performance = float('-inf')
        
        for strategy, performances in self.strategy_performance.items():
            avg_performance = np.mean(performances)
            if avg_performance > best_performance:
                best_performance = avg_performance
                best_strategy = strategy
        
        return best_strategy or "learning_rate_schedule"
    
    def meta_update(self, strategies_used: List[str], outcomes: List[float]) -> None:
        """Update meta-learning parameters based on strategy outcomes."""
        # This is a simplified meta-update
        # In practice, this could involve more sophisticated meta-learning algorithms
        
        for strategy, outcome in zip(strategies_used, outcomes):
            self.strategy_history.append({
                "strategy": strategy,
                "outcome": outcome,
                "timestamp": time.time()
            })

class ContinualLearningSystem:
    """System for continual learning without forgetting."""
    
    def __init__(self, model: nn.Module, importance_lambda: float = 1000.0):
        self.model = model
        self.importance_lambda = importance_lambda
        self.fisher_information = {}
        self.optimal_params = {}
        self.task_id = 0
        
    def compute_fisher_information(self, data_loader) -> None:
        """Compute Fisher Information Matrix for current task."""
        self.model.eval()
        
        fisher = {}
        for name, param in self.model.named_parameters():
            fisher[name] = torch.zeros_like(param)
        
        num_samples = 0
        for batch in data_loader:
            self.model.zero_grad()
            
            # Forward pass
            output = self.model(batch)
            if isinstance(output, dict):
                loss = output.get('loss', output['node_embeddings'].sum())
            else:
                loss = output.sum()
            
            # Backward pass
            loss.backward()
            
            # Accumulate Fisher information
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    fisher[name] += param.grad.data.clone().pow(2)
            
            num_samples += 1
            if num_samples >= 100:  # Limit computation
                break
        
        # Average Fisher information
        for name in fisher:
            fisher[name] /= num_samples
        
        self.fisher_information[self.task_id] = fisher
        
        # Store optimal parameters for this task
        optimal = {}
        for name, param in self.model.named_parameters():
            optimal[name] = param.data.clone()
        self.optimal_params[self.task_id] = optimal
    
    def ewc_loss(self, current_loss: torch.Tensor) -> torch.Tensor:
        """Compute Elastic Weight Consolidation loss."""
        ewc_loss = torch.tensor(0.0, device=current_loss.device)
        
        for task_id in self.fisher_information:
            fisher = self.fisher_information[task_id]
            optimal = self.optimal_params[task_id]
            
            for name, param in self.model.named_parameters():
                if name in fisher:
                    penalty = fisher[name] * (param - optimal[name]).pow(2)
                    ewc_loss += penalty.sum()
        
        return current_loss + (self.importance_lambda / 2) * ewc_loss
    
    def switch_task(self, data_loader) -> None:
        """Switch to a new task and update continual learning parameters."""
        # Compute Fisher information for current task
        self.compute_fisher_information(data_loader)
        
        # Increment task ID
        self.task_id += 1