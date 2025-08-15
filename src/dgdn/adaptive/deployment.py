"""
Self-healing deployment and circuit breaker patterns.
"""

import time
import threading
import logging
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
from enum import Enum
from collections import deque

class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

@dataclass
class HealthStatus:
    """Health status for deployment."""
    is_healthy: bool
    response_time: float
    error_rate: float
    last_check: float
    consecutive_failures: int

class SelfHealingDeployment:
    """Self-healing deployment system."""
    
    def __init__(self, health_check_interval: float = 30.0):
        self.health_check_interval = health_check_interval
        self.is_monitoring = False
        self.health_status = HealthStatus(
            is_healthy=True,
            response_time=0.0,
            error_rate=0.0,
            last_check=time.time(),
            consecutive_failures=0
        )
        self.logger = logging.getLogger(f'{__name__}.SelfHealingDeployment')
    
    def start_monitoring(self):
        """Start health monitoring."""
        self.is_monitoring = True
        self.logger.info("Self-healing monitoring started")
    
    def stop_monitoring(self):
        """Stop health monitoring."""
        self.is_monitoring = False
        self.logger.info("Self-healing monitoring stopped")
    
    def health_check(self) -> HealthStatus:
        """Perform health check."""
        # Simulate health check
        current_time = time.time()
        
        # Update health status
        self.health_status.last_check = current_time
        
        return self.health_status
    
    def heal(self) -> bool:
        """Attempt to heal the system."""
        self.logger.info("Attempting system healing")
        
        # Simulate healing actions
        self.health_status.consecutive_failures = 0
        self.health_status.is_healthy = True
        
        return True

class CircuitBreaker:
    """Circuit breaker for fault tolerance."""
    
    def __init__(self, 
                 failure_threshold: int = 5,
                 timeout: float = 60.0,
                 expected_exception: type = Exception):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.expected_exception = expected_exception
        
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time = None
        self.success_count = 0
        
        self.logger = logging.getLogger(f'{__name__}.CircuitBreaker')
    
    def call(self, func: Callable, *args, **kwargs):
        """Call function with circuit breaker protection."""
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
                self.logger.info("Circuit breaker moved to HALF_OPEN")
            else:
                raise Exception("Circuit breaker is OPEN")
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exception as e:
            self._on_failure()
            raise e
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit should attempt reset."""
        return (self.last_failure_time and 
                time.time() - self.last_failure_time >= self.timeout)
    
    def _on_success(self):
        """Handle successful call."""
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= 3:  # Reset after 3 successes
                self.state = CircuitState.CLOSED
                self.failure_count = 0
                self.success_count = 0
                self.logger.info("Circuit breaker CLOSED")
    
    def _on_failure(self):
        """Handle failed call."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
            self.logger.warning("Circuit breaker OPENED")
        
        if self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.OPEN
            self.logger.warning("Circuit breaker returned to OPEN")