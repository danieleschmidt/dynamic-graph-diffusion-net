"""Logging utilities for DGDN."""

import logging
import sys
from typing import Optional, Dict, Any
from pathlib import Path
import json
import time
from datetime import datetime


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    format_string: Optional[str] = None,
    include_timestamp: bool = True
) -> logging.Logger:
    """Setup logging configuration for DGDN.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path to write logs to
        format_string: Custom format string for log messages
        include_timestamp: Whether to include timestamp in logs
        
    Returns:
        Configured logger instance
    """
    # Clear any existing handlers
    logger = logging.getLogger("dgdn")
    logger.handlers.clear()
    
    # Set log level
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {log_level}')
    logger.setLevel(numeric_level)
    
    # Create formatter
    if format_string is None:
        if include_timestamp:
            format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        else:
            format_string = '%(name)s - %(levelname)s - %(message)s'
    
    formatter = logging.Formatter(format_string)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str = "dgdn") -> logging.Logger:
    """Get a logger instance.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


class StructuredLogger:
    """Structured logging for machine-readable logs."""
    
    def __init__(self, logger_name: str = "dgdn", log_file: Optional[str] = None):
        self.logger = get_logger(logger_name)
        self.log_file = log_file
        
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
    
    def log_event(
        self, 
        event_type: str, 
        data: Dict[str, Any], 
        level: str = "INFO"
    ) -> None:
        """Log a structured event.
        
        Args:
            event_type: Type of event (e.g., 'training_start', 'model_forward')
            data: Event data dictionary
            level: Log level
        """
        event = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": event_type,
            "data": data
        }
        
        # Log to standard logger
        log_func = getattr(self.logger, level.lower())
        log_func(f"{event_type}: {json.dumps(data)}")
        
        # Write to structured log file
        if self.log_file:
            with open(self.log_file, 'a') as f:
                f.write(json.dumps(event) + '\n')


def log_model_info(model, logger: Optional[logging.Logger] = None) -> None:
    """Log model information.
    
    Args:
        model: PyTorch model to log information about
        logger: Logger instance (will create if None)
    """
    if logger is None:
        logger = get_logger()
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"Model: {model.__class__.__name__}")
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    logger.info(f"Model size (MB): {total_params * 4 / 1024 / 1024:.2f}")  # Assuming float32
    
    # Log model configuration if available
    if hasattr(model, 'node_dim'):
        logger.info(f"Node dimension: {model.node_dim}")
    if hasattr(model, 'hidden_dim'):
        logger.info(f"Hidden dimension: {model.hidden_dim}")
    if hasattr(model, 'num_layers'):
        logger.info(f"Number of layers: {model.num_layers}")


class TrainingLogger:
    """Specialized logger for training progress."""
    
    def __init__(self, logger_name: str = "dgdn.training"):
        self.logger = get_logger(logger_name)
        self.start_time = None
        self.epoch_start_time = None
        
    def log_training_start(self, config: Dict[str, Any]) -> None:
        """Log training start."""
        self.start_time = time.time()
        self.logger.info("=" * 50)
        self.logger.info("TRAINING STARTED")
        self.logger.info("=" * 50)
        
        for key, value in config.items():
            self.logger.info(f"{key}: {value}")
    
    def log_epoch_start(self, epoch: int, total_epochs: int) -> None:
        """Log epoch start."""
        self.epoch_start_time = time.time()
        self.logger.info(f"Epoch {epoch + 1}/{total_epochs}")
    
    def log_epoch_end(
        self, 
        epoch: int, 
        train_loss: float, 
        val_loss: Optional[float] = None,
        metrics: Optional[Dict[str, float]] = None
    ) -> None:
        """Log epoch end with metrics."""
        epoch_time = time.time() - self.epoch_start_time if self.epoch_start_time else 0
        
        log_msg = f"Epoch {epoch + 1} completed in {epoch_time:.2f}s - "
        log_msg += f"Train Loss: {train_loss:.4f}"
        
        if val_loss is not None:
            log_msg += f", Val Loss: {val_loss:.4f}"
        
        if metrics:
            metric_strs = [f"{k}: {v:.4f}" for k, v in metrics.items()]
            log_msg += f", {', '.join(metric_strs)}"
        
        self.logger.info(log_msg)
    
    def log_training_end(self, best_metrics: Optional[Dict[str, float]] = None) -> None:
        """Log training completion."""
        total_time = time.time() - self.start_time if self.start_time else 0
        
        self.logger.info("=" * 50)
        self.logger.info("TRAINING COMPLETED")
        self.logger.info(f"Total training time: {total_time:.2f}s")
        
        if best_metrics:
            self.logger.info("Best metrics:")
            for key, value in best_metrics.items():
                self.logger.info(f"  {key}: {value:.4f}")
        
        self.logger.info("=" * 50)


class PerformanceLogger:
    """Logger for performance metrics."""
    
    def __init__(self, logger_name: str = "dgdn.performance"):
        self.logger = get_logger(logger_name)
        self.timers = {}
    
    def start_timer(self, name: str) -> None:
        """Start a named timer."""
        self.timers[name] = time.time()
    
    def end_timer(self, name: str, log_result: bool = True) -> float:
        """End a named timer and optionally log the result."""
        if name not in self.timers:
            self.logger.warning(f"Timer '{name}' was not started")
            return 0.0
        
        elapsed = time.time() - self.timers[name]
        del self.timers[name]
        
        if log_result:
            self.logger.info(f"Timer '{name}': {elapsed:.4f}s")
        
        return elapsed
    
    def log_memory_usage(self) -> None:
        """Log current memory usage."""
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            
            self.logger.info(f"Memory - RSS: {memory_info.rss / 1024 / 1024:.2f} MB, "
                           f"VMS: {memory_info.vms / 1024 / 1024:.2f} MB")
        except ImportError:
            self.logger.warning("psutil not available for memory monitoring")
    
    def log_gpu_usage(self) -> None:
        """Log GPU usage if available."""
        try:
            import torch
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    memory_used = torch.cuda.memory_allocated(i) / 1024 / 1024
                    memory_total = torch.cuda.get_device_properties(i).total_memory / 1024 / 1024
                    utilization = memory_used / memory_total * 100
                    
                    self.logger.info(f"GPU {i} - Memory: {memory_used:.2f}/{memory_total:.2f} MB "
                                   f"({utilization:.1f}%)")
        except ImportError:
            pass


# Global logger instance
_global_logger = None


def get_global_logger() -> logging.Logger:
    """Get the global DGDN logger."""
    global _global_logger
    if _global_logger is None:
        _global_logger = setup_logging()
    return _global_logger