"""Monitoring and performance profiling for DGDN."""

import time
import torch
import psutil
import threading
from typing import Dict, Any, Optional, List, Callable
from collections import defaultdict, deque
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime

from .logging import get_logger


@dataclass
class PerformanceMetrics:
    """Performance metrics data structure."""
    timestamp: float
    memory_used_mb: float
    gpu_memory_mb: Optional[float]
    cpu_percent: float
    forward_time_ms: Optional[float] = None
    backward_time_ms: Optional[float] = None
    data_loading_time_ms: Optional[float] = None
    batch_size: Optional[int] = None
    num_nodes: Optional[int] = None
    num_edges: Optional[int] = None


class PerformanceProfiler:
    """Performance profiler for DGDN operations."""
    
    def __init__(self, enable_gpu_monitoring: bool = True):
        self.enable_gpu_monitoring = enable_gpu_monitoring and torch.cuda.is_available()
        self.logger = get_logger("dgdn.monitoring")
        self.metrics_history: List[PerformanceMetrics] = []
        self.active_timers: Dict[str, float] = {}
        self.operation_stats = defaultdict(list)
        
    def start_timer(self, operation_name: str) -> None:
        """Start timing an operation."""
        self.active_timers[operation_name] = time.time()
    
    def end_timer(self, operation_name: str) -> float:
        """End timing an operation and return elapsed time in ms."""
        if operation_name not in self.active_timers:
            self.logger.warning(f"Timer '{operation_name}' was not started")
            return 0.0
        
        elapsed_ms = (time.time() - self.active_timers[operation_name]) * 1000
        del self.active_timers[operation_name]
        
        self.operation_stats[operation_name].append(elapsed_ms)
        return elapsed_ms
    
    def profile_memory(self) -> Dict[str, float]:
        """Profile current memory usage."""
        metrics = {}
        
        # CPU memory
        process = psutil.Process()
        memory_info = process.memory_info()
        metrics['cpu_memory_mb'] = memory_info.rss / 1024 / 1024
        metrics['cpu_percent'] = process.cpu_percent()
        
        # GPU memory
        if self.enable_gpu_monitoring:
            gpu_memory = 0
            for i in range(torch.cuda.device_count()):
                gpu_memory += torch.cuda.memory_allocated(i)
            metrics['gpu_memory_mb'] = gpu_memory / 1024 / 1024
        else:
            metrics['gpu_memory_mb'] = None
        
        return metrics
    
    def capture_metrics(
        self,
        forward_time_ms: Optional[float] = None,
        backward_time_ms: Optional[float] = None,
        data_loading_time_ms: Optional[float] = None,
        batch_size: Optional[int] = None,
        num_nodes: Optional[int] = None,
        num_edges: Optional[int] = None
    ) -> PerformanceMetrics:
        """Capture current performance metrics."""
        memory_metrics = self.profile_memory()
        
        metrics = PerformanceMetrics(
            timestamp=time.time(),
            memory_used_mb=memory_metrics['cpu_memory_mb'],
            gpu_memory_mb=memory_metrics['gpu_memory_mb'],
            cpu_percent=memory_metrics['cpu_percent'],
            forward_time_ms=forward_time_ms,
            backward_time_ms=backward_time_ms,
            data_loading_time_ms=data_loading_time_ms,
            batch_size=batch_size,
            num_nodes=num_nodes,
            num_edges=num_edges
        )
        
        self.metrics_history.append(metrics)
        return metrics
    
    def get_operation_stats(self, operation_name: str) -> Dict[str, float]:
        """Get statistics for a specific operation."""
        if operation_name not in self.operation_stats:
            return {}
        
        times = self.operation_stats[operation_name]
        return {
            'count': len(times),
            'mean_ms': sum(times) / len(times),
            'min_ms': min(times),
            'max_ms': max(times),
            'total_ms': sum(times)
        }
    
    def get_summary(self) -> Dict[str, Any]:
        """Get profiling summary."""
        summary = {
            'total_operations': sum(len(times) for times in self.operation_stats.values()),
            'operations': {},
            'memory_stats': {}
        }
        
        # Operation statistics
        for op_name in self.operation_stats:
            summary['operations'][op_name] = self.get_operation_stats(op_name)
        
        # Memory statistics
        if self.metrics_history:
            cpu_memory = [m.memory_used_mb for m in self.metrics_history]
            summary['memory_stats']['cpu_memory_mb'] = {
                'mean': sum(cpu_memory) / len(cpu_memory),
                'min': min(cpu_memory),
                'max': max(cpu_memory)
            }
            
            if self.enable_gpu_monitoring:
                gpu_memory = [m.gpu_memory_mb for m in self.metrics_history if m.gpu_memory_mb is not None]
                if gpu_memory:
                    summary['memory_stats']['gpu_memory_mb'] = {
                        'mean': sum(gpu_memory) / len(gpu_memory),
                        'min': min(gpu_memory),
                        'max': max(gpu_memory)
                    }
        
        return summary
    
    def save_metrics(self, file_path: str) -> None:
        """Save metrics to file."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'summary': self.get_summary(),
            'metrics_history': [asdict(m) for m in self.metrics_history],
            'operation_stats': dict(self.operation_stats)
        }
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        
        self.logger.info(f"Performance metrics saved to {file_path}")


class ModelMonitor:
    """Monitor model behavior during training and inference."""
    
    def __init__(self, model, monitor_gradients: bool = True):
        self.model = model
        self.monitor_gradients = monitor_gradients
        self.logger = get_logger("dgdn.model_monitor")
        
        self.parameter_stats = {}
        self.gradient_stats = {}
        self.activation_stats = {}
        self.hooks = []
        
        if monitor_gradients:
            self._register_gradient_hooks()
    
    def _register_gradient_hooks(self) -> None:
        """Register hooks to monitor gradients."""
        def gradient_hook(name):
            def hook(grad):
                if grad is not None:
                    self.gradient_stats[name] = {
                        'mean': grad.mean().item(),
                        'std': grad.std().item(),
                        'max': grad.max().item(),
                        'min': grad.min().item(),
                        'norm': grad.norm().item()
                    }
            return hook
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                handle = param.register_hook(gradient_hook(name))
                self.hooks.append(handle)
    
    def monitor_parameters(self) -> None:
        """Monitor parameter statistics."""
        for name, param in self.model.named_parameters():
            self.parameter_stats[name] = {
                'mean': param.data.mean().item(),
                'std': param.data.std().item(),
                'max': param.data.max().item(),
                'min': param.data.min().item(),
                'norm': param.data.norm().item()
            }
    
    def check_parameter_health(self) -> Dict[str, List[str]]:
        """Check for parameter health issues."""
        issues = defaultdict(list)
        
        for name, stats in self.parameter_stats.items():
            # Check for NaN or infinite values
            if not all(torch.isfinite(torch.tensor(v)) for v in stats.values()):
                issues['nan_or_inf'].append(name)
            
            # Check for very large or small values
            if abs(stats['max']) > 1e6 or abs(stats['min']) > 1e6:
                issues['large_values'].append(name)
            
            if abs(stats['mean']) < 1e-8 and stats['std'] < 1e-8:
                issues['very_small_values'].append(name)
        
        return dict(issues)
    
    def check_gradient_health(self) -> Dict[str, List[str]]:
        """Check for gradient health issues."""
        issues = defaultdict(list)
        
        for name, stats in self.gradient_stats.items():
            # Check for vanishing gradients
            if stats['norm'] < 1e-8:
                issues['vanishing_gradients'].append(name)
            
            # Check for exploding gradients
            if stats['norm'] > 100:
                issues['exploding_gradients'].append(name)
            
            # Check for NaN or infinite gradients
            if not all(torch.isfinite(torch.tensor(v)) for v in stats.values()):
                issues['nan_or_inf_gradients'].append(name)
        
        return dict(issues)
    
    def get_health_report(self) -> Dict[str, Any]:
        """Get comprehensive health report."""
        self.monitor_parameters()
        
        param_issues = self.check_parameter_health()
        grad_issues = self.check_gradient_health()
        
        return {
            'parameter_issues': param_issues,
            'gradient_issues': grad_issues,
            'parameter_stats': self.parameter_stats,
            'gradient_stats': self.gradient_stats,
            'timestamp': datetime.utcnow().isoformat()
        }
    
    def cleanup(self) -> None:
        """Remove hooks and cleanup."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()


class TrainingMonitor:
    """Monitor training progress and detect issues."""
    
    def __init__(self, patience: int = 10, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.logger = get_logger("dgdn.training_monitor")
        
        self.train_losses = deque(maxlen=1000)
        self.val_losses = deque(maxlen=1000)
        self.learning_rates = deque(maxlen=1000)
        
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        self.should_stop = False
        
        self.loss_history = []
        self.metrics_history = []
    
    def update(
        self,
        epoch: int,
        train_loss: float,
        val_loss: Optional[float] = None,
        learning_rate: Optional[float] = None,
        metrics: Optional[Dict[str, float]] = None
    ) -> None:
        """Update monitoring with latest training metrics."""
        self.train_losses.append(train_loss)
        
        if val_loss is not None:
            self.val_losses.append(val_loss)
            
            # Early stopping logic
            if val_loss < self.best_val_loss - self.min_delta:
                self.best_val_loss = val_loss
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1
                
            if self.epochs_without_improvement >= self.patience:
                self.should_stop = True
                self.logger.info(f"Early stopping triggered after {self.patience} epochs without improvement")
        
        if learning_rate is not None:
            self.learning_rates.append(learning_rate)
        
        # Store history
        self.loss_history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'learning_rate': learning_rate
        })
        
        if metrics:
            self.metrics_history.append({
                'epoch': epoch,
                **metrics
            })
    
    def detect_training_issues(self) -> List[str]:
        """Detect potential training issues."""
        issues = []
        
        if len(self.train_losses) < 5:
            return issues
        
        recent_losses = list(self.train_losses)[-5:]
        
        # Check for non-decreasing loss
        if all(recent_losses[i] >= recent_losses[i-1] for i in range(1, len(recent_losses))):
            issues.append("Training loss not decreasing")
        
        # Check for loss explosion
        if recent_losses[-1] > recent_losses[0] * 10:
            issues.append("Training loss exploding")
        
        # Check for loss plateauing
        if len(recent_losses) >= 5:
            loss_variance = torch.tensor(recent_losses).var().item()
            if loss_variance < 1e-8:
                issues.append("Training loss plateaued")
        
        # Check for oscillating loss
        if len(recent_losses) >= 4:
            diffs = [recent_losses[i] - recent_losses[i-1] for i in range(1, len(recent_losses))]
            sign_changes = sum(1 for i in range(1, len(diffs)) if diffs[i] * diffs[i-1] < 0)
            if sign_changes >= len(diffs) - 1:
                issues.append("Training loss oscillating")
        
        return issues
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get training summary."""
        summary = {
            'total_epochs': len(self.loss_history),
            'best_val_loss': self.best_val_loss,
            'epochs_without_improvement': self.epochs_without_improvement,
            'should_stop': self.should_stop,
            'detected_issues': self.detect_training_issues()
        }
        
        if self.train_losses:
            recent_train_losses = list(self.train_losses)[-10:]
            summary['recent_train_loss_trend'] = {
                'mean': sum(recent_train_losses) / len(recent_train_losses),
                'latest': recent_train_losses[-1],
                'improvement': recent_train_losses[0] - recent_train_losses[-1] if len(recent_train_losses) > 1 else 0
            }
        
        if self.val_losses:
            recent_val_losses = list(self.val_losses)[-10:]
            summary['recent_val_loss_trend'] = {
                'mean': sum(recent_val_losses) / len(recent_val_losses),
                'latest': recent_val_losses[-1],
                'improvement': recent_val_losses[0] - recent_val_losses[-1] if len(recent_val_losses) > 1 else 0
            }
        
        return summary


class SystemMonitor:
    """Monitor system resources during training."""
    
    def __init__(self, monitoring_interval: float = 1.0):
        self.monitoring_interval = monitoring_interval
        self.logger = get_logger("dgdn.system_monitor")
        
        self.monitoring = False
        self.monitor_thread = None
        self.metrics = []
        
    def start_monitoring(self) -> None:
        """Start system monitoring in background thread."""
        if self.monitoring:
            self.logger.warning("Monitoring already started")
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        self.logger.info("System monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop system monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        self.logger.info("System monitoring stopped")
    
    def _monitor_loop(self) -> None:
        """Background monitoring loop."""
        while self.monitoring:
            try:
                metrics = {
                    'timestamp': time.time(),
                    'cpu_percent': psutil.cpu_percent(),
                    'memory_percent': psutil.virtual_memory().percent,
                    'disk_usage_percent': psutil.disk_usage('/').percent
                }
                
                # GPU metrics if available
                if torch.cuda.is_available():
                    gpu_metrics = {}
                    for i in range(torch.cuda.device_count()):
                        allocated = torch.cuda.memory_allocated(i)
                        reserved = torch.cuda.memory_reserved(i)
                        total = torch.cuda.get_device_properties(i).total_memory
                        
                        gpu_metrics[f'gpu_{i}_memory_allocated_percent'] = (allocated / total) * 100
                        gpu_metrics[f'gpu_{i}_memory_reserved_percent'] = (reserved / total) * 100
                    
                    metrics.update(gpu_metrics)
                
                self.metrics.append(metrics)
                
                # Keep only last 1000 measurements
                if len(self.metrics) > 1000:
                    self.metrics = self.metrics[-1000:]
                
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current system metrics."""
        if not self.metrics:
            return {}
        return self.metrics[-1].copy()
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of collected metrics."""
        if not self.metrics:
            return {}
        
        # Calculate statistics for each metric
        summary = {}
        metric_names = ['cpu_percent', 'memory_percent', 'disk_usage_percent']
        
        for metric_name in metric_names:
            values = [m[metric_name] for m in self.metrics if metric_name in m]
            if values:
                summary[metric_name] = {
                    'mean': sum(values) / len(values),
                    'min': min(values),
                    'max': max(values),
                    'current': values[-1]
                }
        
        return summary