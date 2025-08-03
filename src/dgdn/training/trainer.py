"""DGDN training pipeline implementation."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import time
import os
from typing import Dict, Optional, List, Tuple, Any, Callable
from tqdm import tqdm
import warnings

from ..models import DynamicGraphDiffusionNet
from ..data import TemporalDataset, create_data_loaders
from .losses import DGDNLoss
from .metrics import DGDNMetrics, EdgePredictionMetrics, NodeClassificationMetrics


class DGDNTrainer:
    """Comprehensive trainer for DGDN models.
    
    Handles training, validation, testing, and model management with
    support for various tasks and advanced training techniques.
    
    Args:
        model: DGDN model to train
        learning_rate: Learning rate for optimizer
        weight_decay: L2 regularization weight
        optimizer_type: Type of optimizer ("adam", "adamw", "sgd")
        scheduler_type: Learning rate scheduler ("cosine", "step", "plateau")
        diffusion_loss_weight: Weight for diffusion loss component
        temporal_reg_weight: Weight for temporal regularization
        task: Training task type
        device: Device for training
        log_dir: Directory for logging
        checkpoint_dir: Directory for model checkpoints
    """
    
    def __init__(
        self,
        model: DynamicGraphDiffusionNet,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        optimizer_type: str = "adam",
        scheduler_type: Optional[str] = "cosine",
        diffusion_loss_weight: float = 0.1,
        temporal_reg_weight: float = 0.05,
        task: str = "edge_prediction",
        device: Optional[torch.device] = None,
        log_dir: str = "logs",
        checkpoint_dir: str = "checkpoints",
        **kwargs
    ):
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.task = task
        
        # Device setup
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        self.model.to(self.device)
        
        # Setup optimizer
        self.optimizer = self._setup_optimizer(optimizer_type)
        
        # Setup scheduler
        self.scheduler = self._setup_scheduler(scheduler_type) if scheduler_type else None
        
        # Setup loss function
        self.loss_fn = DGDNLoss(
            kl_weight=diffusion_loss_weight,
            temporal_weight=temporal_reg_weight,
            task=task
        )
        
        # Setup metrics
        if task == "edge_prediction":
            self.metrics = EdgePredictionMetrics()
        elif task == "node_classification":
            self.metrics = NodeClassificationMetrics()
        else:
            self.metrics = DGDNMetrics(task=task)
        
        # Logging and checkpointing
        self.log_dir = log_dir
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        self.writer = SummaryWriter(log_dir)
        
        # Training state
        self.current_epoch = 0
        self.best_val_metric = float('-inf') if task == "edge_prediction" else float('inf')
        self.training_history = {"train": [], "val": []}
        self.early_stopping_patience = None
        self.early_stopping_counter = 0
        
    def _setup_optimizer(self, optimizer_type: str) -> optim.Optimizer:
        """Setup optimizer."""
        if optimizer_type.lower() == "adam":
            return optim.Adam(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
        elif optimizer_type.lower() == "adamw":
            return optim.AdamW(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
        elif optimizer_type.lower() == "sgd":
            return optim.SGD(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                momentum=0.9
            )
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")
    
    def _setup_scheduler(self, scheduler_type: str) -> Optional[optim.lr_scheduler._LRScheduler]:
        """Setup learning rate scheduler."""
        if scheduler_type.lower() == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=100, eta_min=1e-6
            )
        elif scheduler_type.lower() == "step":
            return optim.lr_scheduler.StepLR(
                self.optimizer, step_size=30, gamma=0.1
            )
        elif scheduler_type.lower() == "plateau":
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='max' if self.task == "edge_prediction" else 'min',
                factor=0.5, patience=10, verbose=True
            )
        else:
            warnings.warn(f"Unknown scheduler type: {scheduler_type}")
            return None
    
    def fit(
        self,
        train_data: TemporalDataset,
        val_data: Optional[TemporalDataset] = None,
        epochs: int = 100,
        batch_size: int = 32,
        early_stopping_patience: Optional[int] = 15,
        save_best: bool = True,
        verbose: bool = True
    ) -> Dict[str, List[float]]:
        """Train the DGDN model.
        
        Args:
            train_data: Training dataset
            val_data: Validation dataset (optional)
            epochs: Number of training epochs
            batch_size: Batch size for training
            early_stopping_patience: Early stopping patience
            save_best: Whether to save best model
            verbose: Whether to print progress
            
        Returns:
            Training history dictionary
        """
        self.early_stopping_patience = early_stopping_patience
        
        # Create data loaders
        train_loader, val_loader, _ = create_data_loaders(
            train_data if hasattr(train_data, '_train_data') else train_data,
            batch_size=batch_size,
            num_workers=0,
            dynamic_batching=True
        )
        
        if val_data is not None:
            val_loader = create_data_loaders(
                val_data if hasattr(val_data, '_val_data') else val_data,
                batch_size=batch_size * 2,  # Larger batch for validation
                num_workers=0,
                dynamic_batching=False
            )[1]
        
        # Training loop
        for epoch in range(epochs):
            self.current_epoch = epoch
            
            # Training phase
            train_metrics = self._train_epoch(train_loader, verbose)
            self.training_history["train"].append(train_metrics)
            
            # Validation phase
            if val_data is not None:
                val_metrics = self._validate_epoch(val_loader, verbose)
                self.training_history["val"].append(val_metrics)
                
                # Early stopping and best model saving
                if self._should_stop_early(val_metrics, save_best):
                    if verbose:
                        print(f"Early stopping at epoch {epoch}")
                    break
            
            # Update learning rate scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    if val_data is not None:
                        metric_key = "auc" if self.task == "edge_prediction" else "accuracy"
                        self.scheduler.step(val_metrics.get(metric_key, 0))
                else:
                    self.scheduler.step()
            
            # Log metrics
            self._log_metrics(train_metrics, val_metrics if val_data else None, epoch)
            
            if verbose and epoch % 10 == 0:
                self._print_progress(epoch, train_metrics, val_metrics if val_data else None)
        
        self.writer.close()
        return self.training_history
    
    def _train_epoch(self, train_loader, verbose: bool = True) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        self.metrics.reset()
        
        epoch_losses = []
        progress_bar = tqdm(train_loader, desc=f"Epoch {self.current_epoch}") if verbose else train_loader
        
        for batch_idx, batch_data in enumerate(progress_bar):
            # Move data to device
            batch_data = batch_data.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            try:
                output = self.model(batch_data, return_uncertainty=True)
                
                # Compute loss
                if self.task == "edge_prediction":
                    targets = getattr(batch_data, 'y', torch.ones(batch_data.edge_index.shape[1]))
                    loss_dict = self.loss_fn(output, targets, batch_data.edge_index)
                else:
                    targets = getattr(batch_data, 'y', torch.zeros(batch_data.num_nodes, dtype=torch.long))
                    loss_dict = self.loss_fn(output, targets)
                
                loss = loss_dict["total"]
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                # Optimizer step
                self.optimizer.step()
                
                # Update metrics
                predictions = self._extract_predictions(output, batch_data)
                self.metrics.update(
                    predictions=predictions,
                    targets=targets,
                    uncertainties=output.get("uncertainty"),
                    loss=loss.item()
                )
                
                epoch_losses.append(loss.item())
                
                # Update progress bar
                if verbose:
                    progress_bar.set_postfix({
                        "loss": f"{loss.item():.4f}",
                        "kl": f"{loss_dict['variational'].item():.4f}"
                    })
                    
            except Exception as e:
                print(f"Error in batch {batch_idx}: {e}")
                continue
        
        return self.metrics.compute()
    
    def _validate_epoch(self, val_loader, verbose: bool = True) -> Dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()
        val_metrics = DGDNMetrics(task=self.task)
        
        with torch.no_grad():
            for batch_data in val_loader:
                batch_data = batch_data.to(self.device)
                
                try:
                    # Forward pass
                    output = self.model(batch_data, return_uncertainty=True)
                    
                    # Compute loss
                    if self.task == "edge_prediction":
                        targets = getattr(batch_data, 'y', torch.ones(batch_data.edge_index.shape[1]))
                        loss_dict = self.loss_fn(output, targets, batch_data.edge_index)
                    else:
                        targets = getattr(batch_data, 'y', torch.zeros(batch_data.num_nodes, dtype=torch.long))
                        loss_dict = self.loss_fn(output, targets)
                    
                    # Update metrics
                    predictions = self._extract_predictions(output, batch_data)
                    val_metrics.update(
                        predictions=predictions,
                        targets=targets,
                        uncertainties=output.get("uncertainty"),
                        loss=loss_dict["total"].item()
                    )
                    
                except Exception as e:
                    print(f"Validation error: {e}")
                    continue
        
        return val_metrics.compute()
    
    def _extract_predictions(self, model_output: Dict, batch_data) -> torch.Tensor:
        """Extract predictions from model output based on task."""
        if self.task == "edge_prediction":
            # Use node embeddings to compute edge predictions
            node_embeddings = model_output["node_embeddings"]
            edge_index = batch_data.edge_index
            
            src_embeddings = node_embeddings[edge_index[0]]
            tgt_embeddings = node_embeddings[edge_index[1]]
            
            # Simple dot product for edge prediction
            predictions = torch.sum(src_embeddings * tgt_embeddings, dim=-1)
            return predictions
        
        elif self.task == "node_classification":
            # Use model's node classifier
            node_embeddings = model_output["node_embeddings"]
            return self.model.node_classifier(node_embeddings)
        
        else:
            return model_output["node_embeddings"]
    
    def _should_stop_early(self, val_metrics: Dict[str, float], save_best: bool) -> bool:
        """Check if training should stop early."""
        if self.early_stopping_patience is None:
            return False
        
        # Get validation metric for comparison
        if self.task == "edge_prediction":
            current_metric = val_metrics.get("auc", 0)
            is_better = current_metric > self.best_val_metric
        else:
            current_metric = val_metrics.get("accuracy", 0)
            is_better = current_metric > self.best_val_metric
        
        if is_better:
            self.best_val_metric = current_metric
            self.early_stopping_counter = 0
            
            if save_best:
                self._save_checkpoint("best_model.pt")
        else:
            self.early_stopping_counter += 1
        
        return self.early_stopping_counter >= self.early_stopping_patience
    
    def _log_metrics(
        self,
        train_metrics: Dict[str, float],
        val_metrics: Optional[Dict[str, float]],
        epoch: int
    ):
        """Log metrics to tensorboard."""
        # Log training metrics
        for key, value in train_metrics.items():
            self.writer.add_scalar(f"train/{key}", value, epoch)
        
        # Log validation metrics
        if val_metrics:
            for key, value in val_metrics.items():
                self.writer.add_scalar(f"val/{key}", value, epoch)
        
        # Log learning rate
        current_lr = self.optimizer.param_groups[0]['lr']
        self.writer.add_scalar("learning_rate", current_lr, epoch)
    
    def _print_progress(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Optional[Dict[str, float]]
    ):
        """Print training progress."""
        train_loss = train_metrics.get("loss", 0)
        train_metric = train_metrics.get("auc" if self.task == "edge_prediction" else "accuracy", 0)
        
        print_str = f"Epoch {epoch:3d} | Train Loss: {train_loss:.4f} | Train Metric: {train_metric:.4f}"
        
        if val_metrics:
            val_loss = val_metrics.get("loss", 0)
            val_metric = val_metrics.get("auc" if self.task == "edge_prediction" else "accuracy", 0)
            print_str += f" | Val Loss: {val_loss:.4f} | Val Metric: {val_metric:.4f}"
        
        print(print_str)
    
    def evaluate(self, test_data: TemporalDataset, batch_size: int = 64) -> Dict[str, float]:
        """Evaluate model on test data."""
        self.model.eval()
        test_metrics = DGDNMetrics(task=self.task)
        
        # Create test loader
        _, _, test_loader = create_data_loaders(
            test_data,
            batch_size=batch_size,
            num_workers=0,
            dynamic_batching=False
        )
        
        with torch.no_grad():
            for batch_data in tqdm(test_loader, desc="Testing"):
                batch_data = batch_data.to(self.device)
                
                # Forward pass
                output = self.model(batch_data, return_uncertainty=True)
                
                # Compute predictions
                predictions = self._extract_predictions(output, batch_data)
                
                if self.task == "edge_prediction":
                    targets = getattr(batch_data, 'y', torch.ones(batch_data.edge_index.shape[1]))
                else:
                    targets = getattr(batch_data, 'y', torch.zeros(batch_data.num_nodes, dtype=torch.long))
                
                # Update metrics
                test_metrics.update(
                    predictions=predictions,
                    targets=targets,
                    uncertainties=output.get("uncertainty")
                )
        
        return test_metrics.compute()
    
    def _save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_metric': self.best_val_metric,
            'training_history': self.training_history
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        torch.save(checkpoint, os.path.join(self.checkpoint_dir, filename))
    
    def load_checkpoint(self, filename: str):
        """Load model checkpoint."""
        checkpoint_path = os.path.join(self.checkpoint_dir, filename)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_metric = checkpoint['best_val_metric']
        self.training_history = checkpoint['training_history']
        
        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    def predict(self, data, return_uncertainty: bool = False) -> Dict[str, torch.Tensor]:
        """Make predictions on new data."""
        self.model.eval()
        
        with torch.no_grad():
            data = data.to(self.device)
            output = self.model(data, return_uncertainty=return_uncertainty)
            
            predictions = self._extract_predictions(output, data)
            
            result = {"predictions": predictions}
            
            if return_uncertainty:
                result["uncertainty"] = output.get("uncertainty")
                result["mean"] = output.get("mean")
                result["logvar"] = output.get("logvar")
            
            return result