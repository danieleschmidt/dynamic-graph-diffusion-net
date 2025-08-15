"""
Adaptive monitoring and anomaly detection systems.
"""

import torch
import numpy as np
import time
import logging
from typing import Dict, Any, List, Optional, Tuple, Callable
from dataclasses import dataclass
from collections import deque
import threading

@dataclass
class MonitoringMetrics:
    """Metrics for adaptive monitoring."""
    timestamp: float
    cpu_usage: float
    memory_usage: float
    inference_time: float
    accuracy: float
    loss: float
    anomaly_score: float

class AdaptiveMonitoring:
    """Adaptive monitoring system that adjusts monitoring frequency and thresholds."""
    
    def __init__(self, base_frequency: float = 1.0):
        self.base_frequency = base_frequency
        self.current_frequency = base_frequency
        self.metrics_history = deque(maxlen=1000)
        self.thresholds = {}
        self.logger = logging.getLogger(f'{__name__}.AdaptiveMonitoring')
    
    def monitor(self, metrics: MonitoringMetrics) -> Dict[str, Any]:
        """Monitor and adapt monitoring parameters."""
        self.metrics_history.append(metrics)
        
        # Adaptive threshold adjustment
        self._adapt_thresholds()
        
        # Adaptive frequency adjustment
        self._adapt_frequency(metrics)
        
        return {"frequency": self.current_frequency, "thresholds": self.thresholds}
    
    def _adapt_thresholds(self):
        """Adapt monitoring thresholds based on historical data."""
        if len(self.metrics_history) < 10:
            return
        
        recent_metrics = list(self.metrics_history)[-50:]
        
        # Calculate dynamic thresholds
        inference_times = [m.inference_time for m in recent_metrics]
        memory_usages = [m.memory_usage for m in recent_metrics]
        
        self.thresholds["inference_time"] = np.mean(inference_times) + 2 * np.std(inference_times)
        self.thresholds["memory_usage"] = np.mean(memory_usages) + 2 * np.std(memory_usages)
    
    def _adapt_frequency(self, metrics: MonitoringMetrics):
        """Adapt monitoring frequency based on system state."""
        if metrics.anomaly_score > 0.8:
            self.current_frequency = min(self.base_frequency * 2, 10.0)
        elif metrics.anomaly_score < 0.2:
            self.current_frequency = max(self.base_frequency * 0.5, 0.1)
        else:
            self.current_frequency = self.base_frequency

class AnomalyDetector:
    """Anomaly detection for model behavior."""
    
    def __init__(self, window_size: int = 50, threshold: float = 2.0):
        self.window_size = window_size
        self.threshold = threshold
        self.baseline_metrics = deque(maxlen=window_size)
        self.logger = logging.getLogger(f'{__name__}.AnomalyDetector')
    
    def detect_anomaly(self, current_metrics: Dict[str, float]) -> float:
        """Detect anomalies in current metrics."""
        if len(self.baseline_metrics) < 10:
            self.baseline_metrics.append(current_metrics)
            return 0.0
        
        # Calculate anomaly score
        anomaly_scores = []
        
        for key, value in current_metrics.items():
            if key in ['inference_time', 'memory_usage', 'loss']:
                baseline_values = [m.get(key, 0) for m in self.baseline_metrics]
                if baseline_values:
                    mean_val = np.mean(baseline_values)
                    std_val = np.std(baseline_values) + 1e-6
                    z_score = abs(value - mean_val) / std_val
                    anomaly_scores.append(min(z_score / self.threshold, 1.0))
        
        # Update baseline
        self.baseline_metrics.append(current_metrics)
        
        return np.mean(anomaly_scores) if anomaly_scores else 0.0