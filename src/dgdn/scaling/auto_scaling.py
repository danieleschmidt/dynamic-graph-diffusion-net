"""
Auto-scaling capabilities for DGDN deployments.
"""

import torch
import psutil
import time
import threading
import logging
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import queue
import json

class ScalingAction(Enum):
    """Scaling actions."""
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    NO_ACTION = "no_action"

@dataclass
class ResourceMetrics:
    """Resource utilization metrics."""
    cpu_percent: float
    memory_percent: float
    gpu_percent: float
    request_queue_size: int
    response_time_avg: float
    throughput: float
    timestamp: float

@dataclass
class ScalingDecision:
    """Scaling decision with reasoning."""
    action: ScalingAction
    target_instances: int
    reason: str
    confidence: float
    metrics: ResourceMetrics

class ResourceMonitor:
    """Monitor system resources for auto-scaling decisions."""
    
    def __init__(self, monitoring_interval: float = 10.0):
        self.monitoring_interval = monitoring_interval
        self.metrics_history = []
        self.max_history_size = 100
        self.is_monitoring = False
        self.monitor_thread = None
        self.callbacks = []
        self.logger = logging.getLogger(f'{__name__}.ResourceMonitor')
    
    def start_monitoring(self) -> None:
        """Start resource monitoring."""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        self.logger.info("Resource monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop resource monitoring."""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
        self.logger.info("Resource monitoring stopped")
    
    def add_callback(self, callback: Callable[[ResourceMetrics], None]) -> None:
        """Add callback for metric updates."""
        self.callbacks.append(callback)
    
    def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        while self.is_monitoring:
            try:
                metrics = self._collect_metrics()
                self._store_metrics(metrics)
                
                # Notify callbacks
                for callback in self.callbacks:
                    try:
                        callback(metrics)
                    except Exception as e:
                        self.logger.error(f"Callback error: {e}")
                
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                self.logger.error(f"Monitoring error: {e}")
                time.sleep(self.monitoring_interval)
    
    def _collect_metrics(self) -> ResourceMetrics:
        """Collect current resource metrics."""
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # Memory metrics
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        
        # GPU metrics (if available)
        gpu_percent = 0.0
        if torch.cuda.is_available():
            try:
                gpu_memory = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
                gpu_percent = gpu_memory * 100
            except:
                gpu_percent = 0.0
        
        # Placeholder for request/response metrics (would be set by application)
        request_queue_size = getattr(self, '_request_queue_size', 0)
        response_time_avg = getattr(self, '_response_time_avg', 0.0)
        throughput = getattr(self, '_throughput', 0.0)
        
        return ResourceMetrics(
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            gpu_percent=gpu_percent,
            request_queue_size=request_queue_size,
            response_time_avg=response_time_avg,
            throughput=throughput,
            timestamp=time.time()
        )
    
    def _store_metrics(self, metrics: ResourceMetrics) -> None:
        """Store metrics in history."""
        self.metrics_history.append(metrics)
        
        # Limit history size
        if len(self.metrics_history) > self.max_history_size:
            self.metrics_history.pop(0)
    
    def get_recent_metrics(self, window_seconds: float = 60.0) -> List[ResourceMetrics]:
        """Get metrics from recent time window."""
        current_time = time.time()
        cutoff_time = current_time - window_seconds
        
        return [m for m in self.metrics_history if m.timestamp >= cutoff_time]
    
    def get_average_metrics(self, window_seconds: float = 60.0) -> Optional[ResourceMetrics]:
        """Get average metrics over time window."""
        recent_metrics = self.get_recent_metrics(window_seconds)
        
        if not recent_metrics:
            return None
        
        # Calculate averages
        avg_cpu = sum(m.cpu_percent for m in recent_metrics) / len(recent_metrics)
        avg_memory = sum(m.memory_percent for m in recent_metrics) / len(recent_metrics)
        avg_gpu = sum(m.gpu_percent for m in recent_metrics) / len(recent_metrics)
        avg_queue = sum(m.request_queue_size for m in recent_metrics) / len(recent_metrics)
        avg_response_time = sum(m.response_time_avg for m in recent_metrics) / len(recent_metrics)
        avg_throughput = sum(m.throughput for m in recent_metrics) / len(recent_metrics)
        
        return ResourceMetrics(
            cpu_percent=avg_cpu,
            memory_percent=avg_memory,
            gpu_percent=avg_gpu,
            request_queue_size=int(avg_queue),
            response_time_avg=avg_response_time,
            throughput=avg_throughput,
            timestamp=time.time()
        )
    
    def set_application_metrics(self, queue_size: int, response_time: float, throughput: float) -> None:
        """Set application-specific metrics."""
        self._request_queue_size = queue_size
        self._response_time_avg = response_time
        self._throughput = throughput

class AutoScaler:
    """Intelligent auto-scaling system for DGDN deployments."""
    
    def __init__(self, 
                 min_instances: int = 1,
                 max_instances: int = 10,
                 target_cpu_percent: float = 70.0,
                 target_memory_percent: float = 80.0,
                 target_response_time: float = 1.0,
                 scale_up_threshold: float = 0.8,
                 scale_down_threshold: float = 0.3,
                 cooldown_period: float = 300.0):  # 5 minutes
        
        self.min_instances = min_instances
        self.max_instances = max_instances
        self.target_cpu_percent = target_cpu_percent
        self.target_memory_percent = target_memory_percent
        self.target_response_time = target_response_time
        self.scale_up_threshold = scale_up_threshold
        self.scale_down_threshold = scale_down_threshold
        self.cooldown_period = cooldown_period
        
        self.current_instances = min_instances
        self.last_scaling_time = 0.0
        self.scaling_history = []
        self.resource_monitor = ResourceMonitor()
        self.logger = logging.getLogger(f'{__name__}.AutoScaler')
        
        # Register for metric updates
        self.resource_monitor.add_callback(self._on_metrics_update)
    
    def start(self) -> None:
        """Start auto-scaling."""
        self.resource_monitor.start_monitoring()
        self.logger.info("Auto-scaler started")
    
    def stop(self) -> None:
        """Stop auto-scaling."""
        self.resource_monitor.stop_monitoring()
        self.logger.info("Auto-scaler stopped")
    
    def _on_metrics_update(self, metrics: ResourceMetrics) -> None:
        """Handle metric updates."""
        decision = self.make_scaling_decision(metrics)
        
        if decision.action != ScalingAction.NO_ACTION:
            self.logger.info(f"Scaling decision: {decision.action.value} to {decision.target_instances} instances")
            self.logger.info(f"Reason: {decision.reason}")
            
            # Execute scaling (in practice, this would call cloud APIs)
            self._execute_scaling(decision)
    
    def make_scaling_decision(self, current_metrics: ResourceMetrics) -> ScalingDecision:
        """Make intelligent scaling decision based on metrics."""
        # Check cooldown period
        current_time = time.time()
        if current_time - self.last_scaling_time < self.cooldown_period:
            return ScalingDecision(
                action=ScalingAction.NO_ACTION,
                target_instances=self.current_instances,
                reason="Cooldown period active",
                confidence=1.0,
                metrics=current_metrics
            )
        
        # Get historical context
        avg_metrics = self.resource_monitor.get_average_metrics(window_seconds=120.0)
        if avg_metrics is None:
            avg_metrics = current_metrics
        
        # Calculate scaling factors
        cpu_factor = avg_metrics.cpu_percent / self.target_cpu_percent
        memory_factor = avg_metrics.memory_percent / self.target_memory_percent
        
        # Response time factor
        response_factor = 1.0
        if self.target_response_time > 0 and avg_metrics.response_time_avg > 0:
            response_factor = avg_metrics.response_time_avg / self.target_response_time
        
        # Queue size factor
        queue_factor = 1.0
        if avg_metrics.request_queue_size > 10:  # Arbitrary threshold
            queue_factor = avg_metrics.request_queue_size / 10.0
        
        # Overall scaling factor (weighted combination)
        weights = [0.3, 0.3, 0.2, 0.2]  # CPU, Memory, Response, Queue
        factors = [cpu_factor, memory_factor, response_factor, queue_factor]
        overall_factor = sum(w * f for w, f in zip(weights, factors))
        
        # Determine action
        action = ScalingAction.NO_ACTION
        target_instances = self.current_instances
        reason = "Metrics within acceptable range"
        confidence = 0.5
        
        if overall_factor > self.scale_up_threshold:
            # Scale up
            if self.current_instances < self.max_instances:
                action = ScalingAction.SCALE_UP
                # Calculate target instances
                suggested_instances = int(self.current_instances * overall_factor)
                target_instances = min(suggested_instances, self.max_instances)
                reason = f"High resource utilization (factor: {overall_factor:.2f})"
                confidence = min(overall_factor - self.scale_up_threshold, 1.0)
        
        elif overall_factor < self.scale_down_threshold:
            # Scale down
            if self.current_instances > self.min_instances:
                action = ScalingAction.SCALE_DOWN
                # Calculate target instances
                suggested_instances = max(int(self.current_instances * overall_factor), 1)
                target_instances = max(suggested_instances, self.min_instances)
                reason = f"Low resource utilization (factor: {overall_factor:.2f})"
                confidence = min(self.scale_down_threshold - overall_factor, 1.0)
        
        return ScalingDecision(
            action=action,
            target_instances=target_instances,
            reason=reason,
            confidence=confidence,
            metrics=current_metrics
        )
    
    def _execute_scaling(self, decision: ScalingDecision) -> None:
        """Execute scaling decision."""
        if decision.action == ScalingAction.NO_ACTION:
            return
        
        # Update internal state
        old_instances = self.current_instances
        self.current_instances = decision.target_instances
        self.last_scaling_time = time.time()
        
        # Record scaling event
        scaling_event = {
            "timestamp": time.time(),
            "action": decision.action.value,
            "old_instances": old_instances,
            "new_instances": self.current_instances,
            "reason": decision.reason,
            "confidence": decision.confidence,
            "metrics": {
                "cpu_percent": decision.metrics.cpu_percent,
                "memory_percent": decision.metrics.memory_percent,
                "response_time": decision.metrics.response_time_avg,
                "queue_size": decision.metrics.request_queue_size
            }
        }
        
        self.scaling_history.append(scaling_event)
        
        # In practice, here you would call cloud APIs to actually scale
        # For now, just log the action
        self.logger.info(f"Scaling {decision.action.value}: {old_instances} -> {self.current_instances} instances")
    
    def get_scaling_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent scaling history."""
        return self.scaling_history[-limit:]
    
    def get_current_status(self) -> Dict[str, Any]:
        """Get current auto-scaler status."""
        current_metrics = self.resource_monitor.get_average_metrics(window_seconds=60.0)
        
        return {
            "current_instances": self.current_instances,
            "min_instances": self.min_instances,
            "max_instances": self.max_instances,
            "last_scaling_time": self.last_scaling_time,
            "time_since_last_scaling": time.time() - self.last_scaling_time,
            "cooldown_remaining": max(0, self.cooldown_period - (time.time() - self.last_scaling_time)),
            "current_metrics": current_metrics.__dict__ if current_metrics else None,
            "total_scaling_events": len(self.scaling_history)
        }

class PredictiveScaler:
    """Predictive auto-scaler using machine learning for forecast-based scaling."""
    
    def __init__(self, auto_scaler: AutoScaler):
        self.auto_scaler = auto_scaler
        self.prediction_window = 300.0  # 5 minutes ahead
        self.model = None  # Placeholder for ML model
        self.feature_history = []
        self.logger = logging.getLogger(f'{__name__}.PredictiveScaler')
    
    def predict_future_load(self, current_metrics: ResourceMetrics) -> ResourceMetrics:
        """Predict future resource requirements."""
        # Simple trend-based prediction (placeholder for ML model)
        recent_metrics = self.auto_scaler.resource_monitor.get_recent_metrics(window_seconds=300.0)
        
        if len(recent_metrics) < 3:
            return current_metrics
        
        # Calculate trends
        cpu_trend = self._calculate_trend([m.cpu_percent for m in recent_metrics])
        memory_trend = self._calculate_trend([m.memory_percent for m in recent_metrics])
        response_trend = self._calculate_trend([m.response_time_avg for m in recent_metrics])
        
        # Project forward
        prediction_steps = self.prediction_window / self.auto_scaler.resource_monitor.monitoring_interval
        
        predicted_cpu = current_metrics.cpu_percent + (cpu_trend * prediction_steps)
        predicted_memory = current_metrics.memory_percent + (memory_trend * prediction_steps)
        predicted_response = current_metrics.response_time_avg + (response_trend * prediction_steps)
        
        # Bound predictions
        predicted_cpu = max(0, min(100, predicted_cpu))
        predicted_memory = max(0, min(100, predicted_memory))
        predicted_response = max(0, predicted_response)
        
        return ResourceMetrics(
            cpu_percent=predicted_cpu,
            memory_percent=predicted_memory,
            gpu_percent=current_metrics.gpu_percent,  # No prediction for GPU yet
            request_queue_size=current_metrics.request_queue_size,
            response_time_avg=predicted_response,
            throughput=current_metrics.throughput,
            timestamp=current_metrics.timestamp + self.prediction_window
        )
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate simple linear trend."""
        if len(values) < 2:
            return 0.0
        
        # Simple linear regression slope
        n = len(values)
        x = list(range(n))
        y = values
        
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(x[i] * y[i] for i in range(n))
        sum_x2 = sum(x[i] ** 2 for i in range(n))
        
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
        return slope
    
    def make_predictive_decision(self, current_metrics: ResourceMetrics) -> ScalingDecision:
        """Make scaling decision based on predicted future load."""
        predicted_metrics = self.predict_future_load(current_metrics)
        
        # Use the regular auto-scaler logic with predicted metrics
        decision = self.auto_scaler.make_scaling_decision(predicted_metrics)
        
        # Adjust confidence and reason for predictive nature
        if decision.action != ScalingAction.NO_ACTION:
            decision.reason = f"Predictive: {decision.reason}"
            decision.confidence *= 0.8  # Lower confidence for predictions
        
        return decision