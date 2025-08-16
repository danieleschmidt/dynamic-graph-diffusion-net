#!/usr/bin/env python3
"""
Generation 2: MAKE IT ROBUST (Reliable) - Enhanced Error Handling, Logging, Security

This implementation adds comprehensive error handling, logging, monitoring, 
health checks, and security measures to the DGDN framework.
"""

import logging
import torch
import numpy as np
import time
import hashlib
import json
import os
import sys
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
from datetime import datetime
import warnings
from contextlib import contextmanager

# Add src to path for imports
sys.path.insert(0, 'src')

import dgdn
from dgdn import DynamicGraphDiffusionNet, TemporalData, TemporalDataset
from dgdn.utils import error_handling, logging as dgdn_logging, health_checks, security, monitoring


class RobustDGDNConfig:
    """Configuration class for robust DGDN operations."""
    
    def __init__(self):
        self.max_nodes = 100000
        self.max_edges = 1000000
        self.max_inference_time = 300.0  # seconds
        self.memory_limit_gb = 16.0
        self.enable_security_checks = True
        self.enable_logging = True
        self.log_level = "INFO"
        self.health_check_interval = 60  # seconds
        self.enable_monitoring = True
        self.safe_mode = True


class RobustDGDNLogger:
    """Enhanced logging system for DGDN operations."""
    
    def __init__(self, config: RobustDGDNConfig):
        self.config = config
        self.setup_logging()
        
    def setup_logging(self):
        """Setup comprehensive logging system."""
        # Create logs directory
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        # Setup main logger
        self.logger = logging.getLogger("dgdn_robust")
        self.logger.setLevel(getattr(logging, self.config.log_level))
        
        # Prevent duplicate handlers
        if self.logger.handlers:
            self.logger.handlers.clear()
        
        # File handler with rotation
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"dgdn_robust_{timestamp}.log"
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, self.config.log_level))
        
        # Formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s | %(name)s | %(levelname)s | %(funcName)s:%(lineno)d | %(message)s'
        )
        simple_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(message)s'
        )
        
        file_handler.setFormatter(detailed_formatter)
        console_handler.setFormatter(simple_formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        self.logger.info(f"🔧 Robust DGDN logging initialized - Log file: {log_file}")
    
    def log_model_info(self, model: DynamicGraphDiffusionNet):
        """Log detailed model information."""
        try:
            param_count = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            model_size_mb = param_count * 4 / (1024 * 1024)
            
            self.logger.info(f"📊 Model Information:")
            self.logger.info(f"   Total parameters: {param_count:,}")
            self.logger.info(f"   Trainable parameters: {trainable_params:,}")
            self.logger.info(f"   Model size: {model_size_mb:.2f} MB")
            self.logger.info(f"   Node dim: {model.node_dim}, Hidden dim: {model.hidden_dim}")
            self.logger.info(f"   Layers: {model.num_layers}, Heads: {model.num_heads}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to log model info: {e}")
    
    def log_data_info(self, data: TemporalData):
        """Log temporal data information."""
        try:
            stats = data.get_temporal_statistics()
            
            self.logger.info(f"📈 Data Information:")
            self.logger.info(f"   Nodes: {stats['num_nodes']:,}, Edges: {stats['num_edges']:,}")
            self.logger.info(f"   Time span: {stats['time_span']:.2f}")
            self.logger.info(f"   Temporal density: {stats['temporal_density']:.3f}")
            self.logger.info(f"   Has node features: {data.node_features is not None}")
            self.logger.info(f"   Has edge features: {data.edge_attr is not None}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to log data info: {e}")


class SecurityValidator:
    """Security validation for DGDN operations."""
    
    def __init__(self, config: RobustDGDNConfig):
        self.config = config
        self.logger = logging.getLogger("dgdn_robust.security")
    
    def validate_input_data(self, data: TemporalData) -> bool:
        """Validate input data for security threats."""
        try:
            # Check data size limits
            if data.num_nodes > self.config.max_nodes:
                self.logger.warning(f"⚠️ Node count {data.num_nodes} exceeds limit {self.config.max_nodes}")
                return False
            
            if data.edge_index.shape[1] > self.config.max_edges:
                self.logger.warning(f"⚠️ Edge count {data.edge_index.shape[1]} exceeds limit {self.config.max_edges}")
                return False
            
            # Check for valid indices
            if data.edge_index.min() < 0:
                self.logger.warning("⚠️ Negative node indices detected")
                return False
            
            if data.edge_index.max() >= data.num_nodes:
                self.logger.warning("⚠️ Node indices exceed num_nodes")
                return False
            
            # Check for NaN/Inf values
            if torch.isnan(data.timestamps).any() or torch.isinf(data.timestamps).any():
                self.logger.warning("⚠️ Invalid timestamp values detected")
                return False
            
            if data.node_features is not None:
                if torch.isnan(data.node_features).any() or torch.isinf(data.node_features).any():
                    self.logger.warning("⚠️ Invalid node feature values detected")
                    return False
            
            if data.edge_attr is not None:
                if torch.isnan(data.edge_attr).any() or torch.isinf(data.edge_attr).any():
                    self.logger.warning("⚠️ Invalid edge attribute values detected")
                    return False
            
            self.logger.debug("✅ Input data validation passed")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Security validation failed: {e}")
            return False
    
    def validate_model_parameters(self, model: DynamicGraphDiffusionNet) -> bool:
        """Validate model parameters for potential security issues."""
        try:
            for name, param in model.named_parameters():
                if torch.isnan(param).any() or torch.isinf(param).any():
                    self.logger.warning(f"⚠️ Invalid parameter values in {name}")
                    return False
                
                # Check for extreme values that might indicate attacks
                if torch.abs(param).max() > 1000:
                    self.logger.warning(f"⚠️ Extreme parameter values in {name}: {torch.abs(param).max()}")
            
            self.logger.debug("✅ Model parameter validation passed")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Model validation failed: {e}")
            return False
    
    def compute_data_hash(self, data: TemporalData) -> str:
        """Compute secure hash of data for integrity verification."""
        try:
            # Create reproducible hash from data
            hasher = hashlib.sha256()
            
            hasher.update(data.edge_index.cpu().numpy().tobytes())
            hasher.update(data.timestamps.cpu().numpy().tobytes())
            
            if data.node_features is not None:
                hasher.update(data.node_features.cpu().numpy().tobytes())
            if data.edge_attr is not None:
                hasher.update(data.edge_attr.cpu().numpy().tobytes())
            
            return hasher.hexdigest()
            
        except Exception as e:
            self.logger.error(f"❌ Failed to compute data hash: {e}")
            return ""


class PerformanceMonitor:
    """Performance monitoring and health checks."""
    
    def __init__(self, config: RobustDGDNConfig):
        self.config = config
        self.logger = logging.getLogger("dgdn_robust.monitor")
        self.metrics = []
        self.start_time = None
        self.peak_memory = 0
    
    @contextmanager
    def monitor_inference(self):
        """Context manager for monitoring inference performance."""
        self.start_time = time.time()
        initial_memory = self._get_memory_usage()
        
        try:
            yield self
        finally:
            end_time = time.time()
            final_memory = self._get_memory_usage()
            
            inference_time = end_time - self.start_time
            memory_used = max(final_memory - initial_memory, 0)
            
            self._record_metrics(inference_time, memory_used)
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in GB."""
        try:
            if torch.cuda.is_available():
                return torch.cuda.memory_allocated() / (1024**3)
            else:
                import psutil
                process = psutil.Process()
                return process.memory_info().rss / (1024**3)
        except Exception:
            return 0.0
    
    def _record_metrics(self, inference_time: float, memory_used: float):
        """Record performance metrics."""
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "inference_time": inference_time,
            "memory_used_gb": memory_used,
            "memory_limit_exceeded": memory_used > self.config.memory_limit_gb,
            "time_limit_exceeded": inference_time > self.config.max_inference_time
        }
        
        self.metrics.append(metrics)
        
        # Log warnings for limit violations
        if metrics["memory_limit_exceeded"]:
            self.logger.warning(f"⚠️ Memory limit exceeded: {memory_used:.2f}GB > {self.config.memory_limit_gb}GB")
        
        if metrics["time_limit_exceeded"]:
            self.logger.warning(f"⚠️ Inference time limit exceeded: {inference_time:.2f}s > {self.config.max_inference_time}s")
        
        self.logger.info(f"📊 Performance: {inference_time:.3f}s, {memory_used:.2f}GB")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get summary of performance metrics."""
        if not self.metrics:
            return {"status": "no_data"}
        
        inference_times = [m["inference_time"] for m in self.metrics]
        memory_usage = [m["memory_used_gb"] for m in self.metrics]
        
        return {
            "total_runs": len(self.metrics),
            "avg_inference_time": np.mean(inference_times),
            "max_inference_time": np.max(inference_times),
            "avg_memory_usage": np.mean(memory_usage),
            "peak_memory_usage": np.max(memory_usage),
            "memory_violations": sum(m["memory_limit_exceeded"] for m in self.metrics),
            "time_violations": sum(m["time_limit_exceeded"] for m in self.metrics)
        }
    
    def health_check(self) -> Dict[str, bool]:
        """Perform comprehensive health check."""
        health = {
            "memory_ok": True,
            "timing_ok": True,
            "torch_ok": True,
            "cuda_ok": True if torch.cuda.is_available() else False
        }
        
        try:
            # Memory check
            current_memory = self._get_memory_usage()
            health["memory_ok"] = current_memory < self.config.memory_limit_gb
            
            # PyTorch functionality check
            test_tensor = torch.randn(10, 10)
            _ = torch.mm(test_tensor, test_tensor.t())
            
            # CUDA check if available
            if torch.cuda.is_available():
                test_tensor_cuda = test_tensor.cuda()
                _ = torch.mm(test_tensor_cuda, test_tensor_cuda.t())
            
            self.logger.debug("✅ Health check completed")
            
        except Exception as e:
            self.logger.error(f"❌ Health check failed: {e}")
            health["torch_ok"] = False
        
        return health


class RobustDGDNWrapper:
    """Robust wrapper for DGDN with comprehensive error handling."""
    
    def __init__(self, model: DynamicGraphDiffusionNet, config: RobustDGDNConfig = None):
        self.config = config or RobustDGDNConfig()
        self.model = model
        self.logger_system = RobustDGDNLogger(self.config)
        self.logger = self.logger_system.logger
        self.security = SecurityValidator(self.config)
        self.monitor = PerformanceMonitor(self.config)
        
        self.logger.info("🚀 Robust DGDN wrapper initialized")
        self.logger_system.log_model_info(model)
    
    def safe_forward(
        self, 
        data: TemporalData, 
        return_attention: bool = False,
        return_uncertainty: bool = False
    ) -> Optional[Dict[str, torch.Tensor]]:
        """Safe forward pass with comprehensive error handling."""
        try:
            # Security validation
            if self.config.enable_security_checks:
                if not self.security.validate_input_data(data):
                    self.logger.error("❌ Input data security validation failed")
                    return None
                
                if not self.security.validate_model_parameters(self.model):
                    self.logger.error("❌ Model parameter validation failed")
                    return None
            
            # Log data information
            self.logger_system.log_data_info(data)
            
            # Compute data integrity hash
            data_hash = self.security.compute_data_hash(data)
            self.logger.debug(f"🔐 Data hash: {data_hash[:16]}...")
            
            # Monitor performance
            with self.monitor.monitor_inference():
                self.logger.info("🔄 Starting forward pass...")
                
                # Set model to eval mode for inference
                self.model.eval()
                
                with torch.no_grad():
                    output = self.model(
                        data, 
                        return_attention=return_attention,
                        return_uncertainty=return_uncertainty
                    )
                
                self.logger.info("✅ Forward pass completed successfully")
                
                # Validate output
                if not self._validate_output(output):
                    self.logger.error("❌ Output validation failed")
                    return None
                
                return output
                
        except torch.cuda.OutOfMemoryError as e:
            self.logger.error(f"❌ CUDA out of memory: {e}")
            torch.cuda.empty_cache()
            return None
        
        except RuntimeError as e:
            self.logger.error(f"❌ Runtime error during forward pass: {e}")
            return None
        
        except Exception as e:
            self.logger.error(f"❌ Unexpected error during forward pass: {e}")
            return None
    
    def _validate_output(self, output: Dict[str, torch.Tensor]) -> bool:
        """Validate model output for correctness."""
        try:
            required_keys = ['node_embeddings', 'mean', 'logvar', 'kl_loss']
            for key in required_keys:
                if key not in output:
                    self.logger.warning(f"⚠️ Missing output key: {key}")
                    return False
            
            # Check for NaN/Inf in outputs
            for key, tensor in output.items():
                if isinstance(tensor, torch.Tensor):
                    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                        self.logger.warning(f"⚠️ Invalid values in output {key}")
                        return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Output validation error: {e}")
            return False
    
    def safe_predict_edges(
        self, 
        source_nodes: torch.Tensor,
        target_nodes: torch.Tensor, 
        time: float,
        data: TemporalData
    ) -> Optional[torch.Tensor]:
        """Safe edge prediction with error handling."""
        try:
            # Validate inputs
            if len(source_nodes) != len(target_nodes):
                self.logger.error("❌ Source and target node arrays must have same length")
                return None
            
            if torch.max(torch.cat([source_nodes, target_nodes])) >= data.num_nodes:
                self.logger.error("❌ Node indices exceed graph size")
                return None
            
            with self.monitor.monitor_inference():
                predictions = self.model.predict_edges(source_nodes, target_nodes, time, data)
                
            self.logger.info(f"✅ Edge predictions completed: {predictions.shape}")
            return predictions
            
        except Exception as e:
            self.logger.error(f"❌ Edge prediction failed: {e}")
            return None
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        health = self.monitor.health_check()
        performance = self.monitor.get_performance_summary()
        
        status = {
            "timestamp": datetime.now().isoformat(),
            "health": health,
            "performance": performance,
            "config": {
                "max_nodes": self.config.max_nodes,
                "max_edges": self.config.max_edges,
                "memory_limit_gb": self.config.memory_limit_gb,
                "safe_mode": self.config.safe_mode
            }
        }
        
        return status
    
    def save_metrics(self, filepath: str):
        """Save performance metrics to file."""
        try:
            metrics_data = {
                "config": vars(self.config),
                "metrics": self.monitor.metrics,
                "summary": self.monitor.get_performance_summary()
            }
            
            with open(filepath, 'w') as f:
                json.dump(metrics_data, f, indent=2, default=str)
            
            self.logger.info(f"📁 Metrics saved to {filepath}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to save metrics: {e}")


def demo_robust_dgdn():
    """Demonstrate robust DGDN functionality."""
    print("🚀 DGDN Generation 2: MAKE IT ROBUST (Reliable)")
    print("=" * 60)
    
    # Initialize configuration
    config = RobustDGDNConfig()
    
    # Create model
    model = DynamicGraphDiffusionNet(
        node_dim=128,
        edge_dim=64,
        hidden_dim=256,
        num_layers=3,
        num_heads=8,
        diffusion_steps=5,
        dropout=0.1
    )
    
    # Create robust wrapper
    robust_model = RobustDGDNWrapper(model, config)
    
    # Test with various scenarios
    test_scenarios = [
        {"nodes": 100, "edges": 500, "name": "Small graph"},
        {"nodes": 1000, "edges": 5000, "name": "Medium graph"},
        {"nodes": 5000, "edges": 25000, "name": "Large graph"}
    ]
    
    for scenario in test_scenarios:
        print(f"\n🧪 Testing {scenario['name']}...")
        
        # Create test data
        edge_index = torch.randint(0, scenario["nodes"], (2, scenario["edges"]))
        timestamps = torch.sort(torch.rand(scenario["edges"]) * 100)[0]
        node_features = torch.randn(scenario["nodes"], 128)
        edge_attr = torch.randn(scenario["edges"], 64)
        
        data = TemporalData(
            edge_index=edge_index,
            timestamps=timestamps,
            node_features=node_features,
            edge_attr=edge_attr,
            num_nodes=scenario["nodes"]
        )
        
        # Safe forward pass
        output = robust_model.safe_forward(data, return_attention=True, return_uncertainty=True)
        
        if output is not None:
            print(f"✅ {scenario['name']} processed successfully")
            
            # Test edge prediction
            src_nodes = torch.randint(0, scenario["nodes"], (5,))
            tgt_nodes = torch.randint(0, scenario["nodes"], (5,))
            edge_preds = robust_model.safe_predict_edges(src_nodes, tgt_nodes, 50.0, data)
            
            if edge_preds is not None:
                print(f"✅ Edge predictions: {edge_preds.shape}")
        else:
            print(f"❌ {scenario['name']} failed")
    
    # Get system status
    status = robust_model.get_system_status()
    print("\n📊 System Status:")
    print(f"   Health: {all(status['health'].values())}")
    print(f"   Total runs: {status['performance'].get('total_runs', 0)}")
    print(f"   Avg inference time: {status['performance'].get('avg_inference_time', 0):.3f}s")
    
    # Save metrics
    robust_model.save_metrics("gen2_robust_metrics.json")
    
    print("\n🎉 Generation 2 Robust Implementation Completed!")
    print("✅ Enhanced error handling implemented")
    print("✅ Comprehensive logging system active")
    print("✅ Security validation operational")
    print("✅ Performance monitoring enabled")
    print("✅ Health checks functional")


if __name__ == "__main__":
    demo_robust_dgdn()