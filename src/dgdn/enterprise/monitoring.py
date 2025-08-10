"""Advanced monitoring and observability for DGDN systems."""

import torch
import time
import psutil
import threading
from typing import Dict, List, Optional, Any, Callable
import json
import logging
from collections import defaultdict, deque
import numpy as np
from datetime import datetime, timedelta
import warnings


class AdvancedMonitoring:
    """Comprehensive monitoring system for DGDN models and infrastructure."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger('DGDN.Monitoring')
        
        # Monitoring components
        self.metrics_collector = MetricsCollector()
        self.alert_manager = AlertManager()
        self.performance_monitor = PerformanceMonitor()
        self.health_checker = HealthChecker()
        
        # Monitoring state
        self.is_monitoring = False
        self.monitoring_thread = None
        self.monitoring_interval = self.config.get('monitoring_interval', 30)  # seconds
        
        # Metric history
        self.metric_history = defaultdict(lambda: deque(maxlen=1000))
        
    def start_monitoring(self):
        """Start continuous monitoring."""
        if self.is_monitoring:
            self.logger.warning("Monitoring already started")
            return
            
        self.is_monitoring = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        self.logger.info("Started advanced monitoring")
        
    def stop_monitoring(self):
        """Stop continuous monitoring."""
        self.is_monitoring = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
            
        self.logger.info("Stopped advanced monitoring")
        
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.is_monitoring:
            try:
                # Collect all metrics
                metrics = self._collect_all_metrics()
                
                # Store in history
                timestamp = time.time()
                for metric_name, value in metrics.items():
                    self.metric_history[metric_name].append((timestamp, value))
                
                # Check for alerts
                self._check_alerts(metrics)
                
                # Sleep until next collection
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.monitoring_interval)
                
    def _collect_all_metrics(self) -> Dict[str, float]:
        """Collect all available metrics."""
        metrics = {}
        
        # System metrics
        metrics.update(self.metrics_collector.collect_system_metrics())
        
        # GPU metrics (if available)
        if torch.cuda.is_available():
            metrics.update(self.metrics_collector.collect_gpu_metrics())
            
        # Performance metrics
        metrics.update(self.performance_monitor.get_current_metrics())
        
        # Health check results
        health_status = self.health_checker.check_all_health()
        metrics['health_score'] = health_status['overall_score']
        
        return metrics
        
    def _check_alerts(self, metrics: Dict[str, float]):
        """Check metrics against alert thresholds."""
        for metric_name, value in metrics.items():
            if self.alert_manager.should_alert(metric_name, value):
                self.alert_manager.send_alert(metric_name, value)
                
    def get_metric_summary(self, time_window: int = 3600) -> Dict[str, Any]:
        """Get summary statistics for metrics in time window."""
        cutoff_time = time.time() - time_window
        summary = {}
        
        for metric_name, history in self.metric_history.items():
            # Filter to time window
            recent_values = [
                value for timestamp, value in history 
                if timestamp > cutoff_time
            ]
            
            if recent_values:
                summary[metric_name] = {
                    'count': len(recent_values),
                    'mean': np.mean(recent_values),
                    'std': np.std(recent_values),
                    'min': np.min(recent_values),
                    'max': np.max(recent_values),
                    'p95': np.percentile(recent_values, 95),
                    'p99': np.percentile(recent_values, 99)
                }
                
        return summary
        
    def create_dashboard_data(self) -> Dict[str, Any]:
        """Create data for monitoring dashboard."""
        current_time = time.time()
        
        # Recent metrics (last hour)
        recent_metrics = {}
        for metric_name, history in self.metric_history.items():
            recent_data = [
                (timestamp, value) for timestamp, value in history
                if timestamp > current_time - 3600
            ]
            recent_metrics[metric_name] = recent_data
            
        # System status
        health_status = self.health_checker.check_all_health()
        
        # Performance summary
        perf_summary = self.performance_monitor.get_performance_summary()
        
        return {
            'timestamp': current_time,
            'metrics': recent_metrics,
            'health': health_status,
            'performance': perf_summary,
            'alerts': self.alert_manager.get_recent_alerts()
        }


class MetricsCollector:
    """Collects various system and model metrics."""
    
    def __init__(self):
        self.logger = logging.getLogger('DGDN.MetricsCollector')
        
    def collect_system_metrics(self) -> Dict[str, float]:
        """Collect system-level metrics."""
        metrics = {}
        
        try:
            # CPU metrics
            metrics['cpu_percent'] = psutil.cpu_percent()
            metrics['cpu_count'] = psutil.cpu_count()
            
            # Memory metrics
            memory = psutil.virtual_memory()
            metrics['memory_percent'] = memory.percent
            metrics['memory_used_gb'] = memory.used / (1024**3)
            metrics['memory_available_gb'] = memory.available / (1024**3)
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            metrics['disk_percent'] = (disk.used / disk.total) * 100
            metrics['disk_used_gb'] = disk.used / (1024**3)
            metrics['disk_free_gb'] = disk.free / (1024**3)
            
            # Network metrics
            network = psutil.net_io_counters()
            metrics['network_bytes_sent'] = network.bytes_sent
            metrics['network_bytes_recv'] = network.bytes_recv
            
        except Exception as e:
            self.logger.error(f"Error collecting system metrics: {e}")
            
        return metrics
        
    def collect_gpu_metrics(self) -> Dict[str, float]:
        """Collect GPU metrics if available."""
        metrics = {}
        
        if not torch.cuda.is_available():
            return metrics
            
        try:
            # Basic GPU metrics
            device_count = torch.cuda.device_count()
            metrics['gpu_count'] = device_count
            
            for i in range(device_count):
                # Memory metrics
                memory_allocated = torch.cuda.memory_allocated(i) / (1024**3)
                memory_reserved = torch.cuda.memory_reserved(i) / (1024**3)
                
                metrics[f'gpu_{i}_memory_allocated_gb'] = memory_allocated
                metrics[f'gpu_{i}_memory_reserved_gb'] = memory_reserved
                
                # Utilization (requires nvidia-ml-py if available)
                try:
                    import pynvml
                    pynvml.nvmlInit()
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    
                    # GPU utilization
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    metrics[f'gpu_{i}_utilization_percent'] = util.gpu
                    
                    # Temperature
                    temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                    metrics[f'gpu_{i}_temperature_c'] = temp
                    
                    # Power
                    power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert to watts
                    metrics[f'gpu_{i}_power_watts'] = power
                    
                except ImportError:
                    self.logger.debug("pynvml not available for detailed GPU metrics")
                except Exception as e:
                    self.logger.warning(f"Error collecting detailed GPU metrics: {e}")
                    
        except Exception as e:
            self.logger.error(f"Error collecting GPU metrics: {e}")
            
        return metrics
        
    def collect_model_metrics(self, model: torch.nn.Module) -> Dict[str, float]:
        """Collect model-specific metrics."""
        metrics = {}
        
        try:
            # Parameter count
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            metrics['model_total_params'] = total_params
            metrics['model_trainable_params'] = trainable_params
            
            # Model memory usage
            param_size = sum(p.numel() * p.element_size() for p in model.parameters())
            buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
            
            metrics['model_memory_mb'] = (param_size + buffer_size) / (1024**2)
            
            # Gradient statistics (if available)
            grad_norms = []
            for param in model.parameters():
                if param.grad is not None:
                    grad_norms.append(param.grad.norm().item())
                    
            if grad_norms:
                metrics['gradient_norm_mean'] = np.mean(grad_norms)
                metrics['gradient_norm_max'] = np.max(grad_norms)
                metrics['gradient_norm_std'] = np.std(grad_norms)
                
        except Exception as e:
            self.logger.error(f"Error collecting model metrics: {e}")
            
        return metrics


class AlertManager:
    """Manages alerts and notifications for monitoring."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger('DGDN.AlertManager')
        
        # Alert thresholds
        self.thresholds = self.config.get('thresholds', {
            'cpu_percent': 90.0,
            'memory_percent': 85.0,
            'gpu_memory_percent': 90.0,
            'disk_percent': 90.0,
            'inference_latency_ms': 1000.0,
            'training_loss': 10.0,
            'health_score': 0.5
        })
        
        # Alert history
        self.alert_history = deque(maxlen=1000)
        self.last_alert_time = defaultdict(float)
        self.alert_cooldown = 300  # 5 minutes between same alerts
        
        # Notification handlers
        self.notification_handlers = []
        
    def add_notification_handler(self, handler: Callable[[str, str, float], None]):
        """Add notification handler (email, Slack, etc.)."""
        self.notification_handlers.append(handler)
        
    def should_alert(self, metric_name: str, value: float) -> bool:
        """Check if metric value should trigger alert."""
        if metric_name not in self.thresholds:
            return False
            
        threshold = self.thresholds[metric_name]
        current_time = time.time()
        
        # Check if value exceeds threshold
        exceeds_threshold = False
        if metric_name == 'health_score':
            exceeds_threshold = value < threshold  # Health score should be high
        else:
            exceeds_threshold = value > threshold
            
        if not exceeds_threshold:
            return False
            
        # Check cooldown
        last_alert = self.last_alert_time[metric_name]
        if current_time - last_alert < self.alert_cooldown:
            return False
            
        return True
        
    def send_alert(self, metric_name: str, value: float):
        """Send alert for metric."""
        current_time = time.time()
        
        # Create alert
        alert = {
            'timestamp': current_time,
            'metric': metric_name,
            'value': value,
            'threshold': self.thresholds.get(metric_name),
            'severity': self._get_alert_severity(metric_name, value),
            'message': self._create_alert_message(metric_name, value)
        }
        
        # Store in history
        self.alert_history.append(alert)
        self.last_alert_time[metric_name] = current_time
        
        # Send notifications
        for handler in self.notification_handlers:
            try:
                handler(alert['message'], alert['severity'], value)
            except Exception as e:
                self.logger.error(f"Error sending alert notification: {e}")
                
        self.logger.warning(f"ALERT: {alert['message']}")
        
    def _get_alert_severity(self, metric_name: str, value: float) -> str:
        """Determine alert severity."""
        threshold = self.thresholds.get(metric_name, 0)
        
        if metric_name == 'health_score':
            if value < threshold * 0.5:
                return 'critical'
            elif value < threshold * 0.75:
                return 'warning'
        else:
            if value > threshold * 1.5:
                return 'critical'
            elif value > threshold * 1.2:
                return 'warning'
                
        return 'info'
        
    def _create_alert_message(self, metric_name: str, value: float) -> str:
        """Create human-readable alert message."""
        threshold = self.thresholds.get(metric_name)
        
        if metric_name == 'cpu_percent':
            return f"High CPU usage: {value:.1f}% (threshold: {threshold}%)"
        elif metric_name == 'memory_percent':
            return f"High memory usage: {value:.1f}% (threshold: {threshold}%)"
        elif metric_name == 'gpu_memory_percent':
            return f"High GPU memory usage: {value:.1f}% (threshold: {threshold}%)"
        elif metric_name == 'health_score':
            return f"Low system health score: {value:.3f} (threshold: {threshold})"
        elif 'latency' in metric_name:
            return f"High latency detected: {value:.1f}ms (threshold: {threshold}ms)"
        else:
            return f"Alert for {metric_name}: {value} (threshold: {threshold})"
            
    def get_recent_alerts(self, time_window: int = 3600) -> List[Dict[str, Any]]:
        """Get alerts from recent time window."""
        cutoff_time = time.time() - time_window
        
        return [
            alert for alert in self.alert_history
            if alert['timestamp'] > cutoff_time
        ]


class PerformanceMonitor:
    """Monitors model and system performance."""
    
    def __init__(self):
        self.logger = logging.getLogger('DGDN.PerformanceMonitor')
        
        # Performance tracking
        self.inference_times = deque(maxlen=1000)
        self.training_times = deque(maxlen=100)
        self.throughput_history = deque(maxlen=1000)
        
        # Resource usage tracking
        self.resource_usage = defaultdict(lambda: deque(maxlen=1000))
        
    def record_inference_time(self, duration: float):
        """Record inference duration."""
        self.inference_times.append(duration)
        
    def record_training_time(self, duration: float):
        """Record training step duration."""
        self.training_times.append(duration)
        
    def record_throughput(self, requests_per_second: float):
        """Record throughput measurement."""
        self.throughput_history.append(requests_per_second)
        
    def get_current_metrics(self) -> Dict[str, float]:
        """Get current performance metrics."""
        metrics = {}
        
        # Inference metrics
        if self.inference_times:
            metrics['inference_latency_mean_ms'] = np.mean(self.inference_times) * 1000
            metrics['inference_latency_p95_ms'] = np.percentile(self.inference_times, 95) * 1000
            metrics['inference_latency_p99_ms'] = np.percentile(self.inference_times, 99) * 1000
            
        # Training metrics
        if self.training_times:
            metrics['training_step_mean_ms'] = np.mean(self.training_times) * 1000
            metrics['training_step_p95_ms'] = np.percentile(self.training_times, 95) * 1000
            
        # Throughput metrics
        if self.throughput_history:
            metrics['throughput_mean_rps'] = np.mean(self.throughput_history)
            metrics['throughput_max_rps'] = np.max(self.throughput_history)
            
        return metrics
        
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        summary = {
            'inference_performance': self._summarize_timings(self.inference_times, 'seconds'),
            'training_performance': self._summarize_timings(self.training_times, 'seconds'),
            'throughput_performance': self._summarize_values(self.throughput_history, 'RPS')
        }
        
        return summary
        
    def _summarize_timings(self, timings: deque, unit: str) -> Dict[str, Any]:
        """Summarize timing measurements."""
        if not timings:
            return {'count': 0}
            
        return {
            'count': len(timings),
            'mean': np.mean(timings),
            'median': np.median(timings),
            'std': np.std(timings),
            'min': np.min(timings),
            'max': np.max(timings),
            'p95': np.percentile(timings, 95),
            'p99': np.percentile(timings, 99),
            'unit': unit
        }
        
    def _summarize_values(self, values: deque, unit: str) -> Dict[str, Any]:
        """Summarize general value measurements."""
        if not values:
            return {'count': 0}
            
        return {
            'count': len(values),
            'mean': np.mean(values),
            'median': np.median(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'unit': unit
        }


class HealthChecker:
    """Comprehensive health checking for DGDN systems."""
    
    def __init__(self):
        self.logger = logging.getLogger('DGDN.HealthChecker')
        
        # Health check functions
        self.health_checks = {
            'system_resources': self._check_system_resources,
            'gpu_availability': self._check_gpu_availability,
            'model_integrity': self._check_model_integrity,
            'memory_leaks': self._check_memory_leaks,
            'disk_space': self._check_disk_space
        }
        
        # Health history
        self.health_history = defaultdict(lambda: deque(maxlen=100))
        
    def check_all_health(self) -> Dict[str, Any]:
        """Run all health checks."""
        results = {}
        scores = []
        
        for check_name, check_func in self.health_checks.items():
            try:
                result = check_func()
                results[check_name] = result
                scores.append(result['score'])
                
                # Store in history
                self.health_history[check_name].append({
                    'timestamp': time.time(),
                    'score': result['score'],
                    'status': result['status']
                })
                
            except Exception as e:
                self.logger.error(f"Health check {check_name} failed: {e}")
                results[check_name] = {
                    'status': 'error',
                    'score': 0.0,
                    'message': str(e)
                }
                scores.append(0.0)
                
        # Overall health score
        overall_score = np.mean(scores) if scores else 0.0
        
        return {
            'overall_score': overall_score,
            'overall_status': self._score_to_status(overall_score),
            'checks': results,
            'timestamp': time.time()
        }
        
    def _check_system_resources(self) -> Dict[str, Any]:
        """Check system resource health."""
        cpu_percent = psutil.cpu_percent()
        memory_percent = psutil.virtual_memory().percent
        
        # Score based on resource usage (lower is better)
        cpu_score = max(0, (100 - cpu_percent) / 100)
        memory_score = max(0, (100 - memory_percent) / 100)
        
        overall_score = (cpu_score + memory_score) / 2
        
        return {
            'score': overall_score,
            'status': self._score_to_status(overall_score),
            'details': {
                'cpu_percent': cpu_percent,
                'memory_percent': memory_percent,
                'cpu_score': cpu_score,
                'memory_score': memory_score
            }
        }
        
    def _check_gpu_availability(self) -> Dict[str, Any]:
        """Check GPU availability and health."""
        if not torch.cuda.is_available():
            return {
                'score': 1.0,  # OK if no GPU expected
                'status': 'healthy',
                'details': {'message': 'No GPU required'}
            }
            
        try:
            # Test GPU functionality
            device = torch.device('cuda:0')
            test_tensor = torch.randn(100, 100, device=device)
            _ = torch.matmul(test_tensor, test_tensor)
            
            # Check memory
            memory_allocated = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
            memory_score = max(0, 1 - memory_allocated)
            
            return {
                'score': memory_score,
                'status': self._score_to_status(memory_score),
                'details': {
                    'gpu_available': True,
                    'memory_utilization': memory_allocated,
                    'device_count': torch.cuda.device_count()
                }
            }
            
        except Exception as e:
            return {
                'score': 0.0,
                'status': 'unhealthy',
                'details': {'error': str(e)}
            }
            
    def _check_model_integrity(self) -> Dict[str, Any]:
        """Check model integrity (placeholder)."""
        # This would check loaded models for corruption, etc.
        return {
            'score': 1.0,
            'status': 'healthy',
            'details': {'message': 'No models to check'}
        }
        
    def _check_memory_leaks(self) -> Dict[str, Any]:
        """Check for memory leaks."""
        # Simple heuristic: check if memory usage is growing over time
        if len(self.health_history['system_resources']) < 10:
            return {
                'score': 1.0,
                'status': 'healthy',
                'details': {'message': 'Insufficient data for leak detection'}
            }
            
        # Get recent memory percentages
        recent_memory = [
            check['details']['memory_percent'] 
            for check in list(self.health_history['system_resources'])[-10:]
            if 'details' in check
        ]
        
        if len(recent_memory) < 5:
            return {'score': 1.0, 'status': 'healthy', 'details': {}}
            
        # Check for consistent growth
        growth_rate = np.polyfit(range(len(recent_memory)), recent_memory, 1)[0]
        
        # Score based on growth rate (negative growth is good)
        leak_score = max(0, 1 - max(0, growth_rate) / 10)
        
        return {
            'score': leak_score,
            'status': self._score_to_status(leak_score),
            'details': {
                'memory_growth_rate': growth_rate,
                'recent_memory_usage': recent_memory[-1] if recent_memory else 0
            }
        }
        
    def _check_disk_space(self) -> Dict[str, Any]:
        """Check available disk space."""
        disk = psutil.disk_usage('/')
        disk_percent = (disk.used / disk.total) * 100
        
        # Score based on free space
        disk_score = max(0, (100 - disk_percent) / 100)
        
        return {
            'score': disk_score,
            'status': self._score_to_status(disk_score),
            'details': {
                'disk_percent_used': disk_percent,
                'disk_free_gb': disk.free / (1024**3),
                'disk_total_gb': disk.total / (1024**3)
            }
        }
        
    def _score_to_status(self, score: float) -> str:
        """Convert score to status string."""
        if score >= 0.8:
            return 'healthy'
        elif score >= 0.6:
            return 'degraded'
        elif score >= 0.3:
            return 'unhealthy'
        else:
            return 'critical'