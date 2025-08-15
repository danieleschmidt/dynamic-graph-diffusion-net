"""
Comprehensive health checks and monitoring for DGDN models.
"""

import torch
import torch.nn as nn
import time
import psutil
import json
import logging
from typing import Dict, Any, List, Optional, Tuple, Callable
from dataclasses import dataclass
from pathlib import Path
import warnings

@dataclass
class HealthCheckResult:
    """Result of a health check operation."""
    name: str
    status: str  # "PASS", "WARN", "FAIL"
    message: str
    details: Dict[str, Any]
    timestamp: float
    execution_time: float

class ModelHealthChecker:
    """Comprehensive health checker for DGDN models."""
    
    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model
        self.device = device
        self.logger = logging.getLogger(f'{__name__}.ModelHealthChecker')
        
        # Health check thresholds
        self.thresholds = {
            'memory_warning': 0.8,  # 80% memory usage warning
            'memory_critical': 0.95,  # 95% memory usage critical
            'inference_time_warning': 1.0,  # 1 second warning
            'inference_time_critical': 5.0,  # 5 seconds critical
            'gradient_norm_warning': 10.0,  # Gradient norm warning
            'gradient_norm_critical': 100.0,  # Gradient norm critical
            'parameter_change_warning': 0.1,  # 10% parameter change warning
        }
    
    def run_all_checks(self) -> List[HealthCheckResult]:
        """Run all health checks and return results."""
        checks = [
            self.check_model_structure,
            self.check_parameter_health,
            self.check_memory_usage,
            self.check_inference_speed,
            self.check_gradient_flow,
            self.check_numerical_stability,
        ]
        
        results = []
        for check in checks:
            try:
                result = check()
                results.append(result)
                self.logger.info(f"Health check '{result.name}': {result.status}")
            except Exception as e:
                error_result = HealthCheckResult(
                    name=check.__name__,
                    status="FAIL",
                    message=f"Health check failed: {e}",
                    details={"error": str(e)},
                    timestamp=time.time(),
                    execution_time=0.0
                )
                results.append(error_result)
                self.logger.error(f"Health check '{check.__name__}' failed: {e}")
        
        return results
    
    def check_model_structure(self) -> HealthCheckResult:
        """Check model structure and parameter counts."""
        start_time = time.time()
        
        try:
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            
            # Check for unusually large models
            param_size_mb = total_params * 4 / (1024 * 1024)  # Assuming float32
            
            status = "PASS"
            message = f"Model structure healthy: {total_params:,} total parameters"
            
            if param_size_mb > 1000:  # > 1GB
                status = "WARN"
                message = f"Large model detected: {param_size_mb:.1f}MB"
            
            details = {
                "total_parameters": total_params,
                "trainable_parameters": trainable_params,
                "model_size_mb": param_size_mb,
                "modules": len(list(self.model.modules())),
            }
            
            return HealthCheckResult(
                name="model_structure",
                status=status,
                message=message,
                details=details,
                timestamp=time.time(),
                execution_time=time.time() - start_time
            )
            
        except Exception as e:
            return HealthCheckResult(
                name="model_structure",
                status="FAIL",
                message=f"Structure check failed: {e}",
                details={"error": str(e)},
                timestamp=time.time(),
                execution_time=time.time() - start_time
            )
    
    def check_parameter_health(self) -> HealthCheckResult:
        """Check parameter values for NaN, inf, and unusual distributions."""
        start_time = time.time()
        
        try:
            param_stats = {
                "total_params": 0,
                "nan_params": 0,
                "inf_params": 0,
                "zero_params": 0,
                "min_value": float('inf'),
                "max_value": float('-inf'),
                "mean_abs": 0.0,
                "std": 0.0,
            }
            
            all_params = []
            
            for name, param in self.model.named_parameters():
                if param is None:
                    continue
                
                param_data = param.data.flatten()
                all_params.append(param_data)
                
                param_stats["total_params"] += param_data.numel()
                param_stats["nan_params"] += torch.isnan(param_data).sum().item()
                param_stats["inf_params"] += torch.isinf(param_data).sum().item()
                param_stats["zero_params"] += (param_data == 0).sum().item()
                
                param_min = param_data.min().item()
                param_max = param_data.max().item()
                
                param_stats["min_value"] = min(param_stats["min_value"], param_min)
                param_stats["max_value"] = max(param_stats["max_value"], param_max)
            
            # Compute global statistics
            if all_params:
                all_params_tensor = torch.cat(all_params)
                param_stats["mean_abs"] = torch.abs(all_params_tensor).mean().item()
                param_stats["std"] = all_params_tensor.std().item()
            
            # Determine status
            status = "PASS"
            issues = []
            
            if param_stats["nan_params"] > 0:
                status = "FAIL"
                issues.append(f"{param_stats['nan_params']} NaN parameters")
            
            if param_stats["inf_params"] > 0:
                status = "FAIL"
                issues.append(f"{param_stats['inf_params']} infinite parameters")
            
            if param_stats["zero_params"] / param_stats["total_params"] > 0.5:
                status = "WARN"
                issues.append("Over 50% of parameters are zero")
            
            if param_stats["std"] < 1e-6:
                status = "WARN"
                issues.append("Very low parameter variance")
            
            message = "Parameter health good" if status == "PASS" else "; ".join(issues)
            
            return HealthCheckResult(
                name="parameter_health",
                status=status,
                message=message,
                details=param_stats,
                timestamp=time.time(),
                execution_time=time.time() - start_time
            )
            
        except Exception as e:
            return HealthCheckResult(
                name="parameter_health",
                status="FAIL",
                message=f"Parameter check failed: {e}",
                details={"error": str(e)},
                timestamp=time.time(),
                execution_time=time.time() - start_time
            )
    
    def check_memory_usage(self) -> HealthCheckResult:
        """Check memory usage on CPU and GPU."""
        start_time = time.time()
        
        try:
            memory_stats = {}
            
            # CPU memory
            cpu_memory = psutil.virtual_memory()
            memory_stats["cpu"] = {
                "total_gb": cpu_memory.total / 1e9,
                "used_gb": cpu_memory.used / 1e9,
                "available_gb": cpu_memory.available / 1e9,
                "percent": cpu_memory.percent,
            }
            
            # GPU memory (if available)
            if torch.cuda.is_available() and self.device.type == 'cuda':
                gpu_memory = {
                    "allocated_gb": torch.cuda.memory_allocated(self.device) / 1e9,
                    "reserved_gb": torch.cuda.memory_reserved(self.device) / 1e9,
                    "max_allocated_gb": torch.cuda.max_memory_allocated(self.device) / 1e9,
                }
                
                # Get total GPU memory
                props = torch.cuda.get_device_properties(self.device)
                gpu_memory["total_gb"] = props.total_memory / 1e9
                gpu_memory["percent"] = (gpu_memory["allocated_gb"] / gpu_memory["total_gb"]) * 100
                
                memory_stats["gpu"] = gpu_memory
            
            # Determine status
            status = "PASS"
            issues = []
            
            if memory_stats["cpu"]["percent"] > self.thresholds["memory_critical"] * 100:
                status = "FAIL"
                issues.append(f"Critical CPU memory usage: {memory_stats['cpu']['percent']:.1f}%")
            elif memory_stats["cpu"]["percent"] > self.thresholds["memory_warning"] * 100:
                status = "WARN"
                issues.append(f"High CPU memory usage: {memory_stats['cpu']['percent']:.1f}%")
            
            if "gpu" in memory_stats:
                gpu_percent = memory_stats["gpu"]["percent"]
                if gpu_percent > self.thresholds["memory_critical"] * 100:
                    status = "FAIL"
                    issues.append(f"Critical GPU memory usage: {gpu_percent:.1f}%")
                elif gpu_percent > self.thresholds["memory_warning"] * 100:
                    status = "WARN"
                    issues.append(f"High GPU memory usage: {gpu_percent:.1f}%")
            
            message = "Memory usage normal" if status == "PASS" else "; ".join(issues)
            
            return HealthCheckResult(
                name="memory_usage",
                status=status,
                message=message,
                details=memory_stats,
                timestamp=time.time(),
                execution_time=time.time() - start_time
            )
            
        except Exception as e:
            return HealthCheckResult(
                name="memory_usage",
                status="FAIL",
                message=f"Memory check failed: {e}",
                details={"error": str(e)},
                timestamp=time.time(),
                execution_time=time.time() - start_time
            )
    
    def check_inference_speed(self) -> HealthCheckResult:
        """Check model inference speed with dummy data."""
        start_time = time.time()
        
        try:
            # Create dummy data for speed test
            from .error_handling import create_dummy_data
            dummy_data = create_dummy_data(num_nodes=100, num_edges=300)
            
            # Warm up
            self.model.eval()
            with torch.no_grad():
                _ = self.model(dummy_data)
            
            # Time multiple runs
            num_runs = 5
            inference_times = []
            
            for _ in range(num_runs):
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                run_start = time.time()
                
                with torch.no_grad():
                    _ = self.model(dummy_data)
                
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                run_time = time.time() - run_start
                inference_times.append(run_time)
            
            avg_inference_time = sum(inference_times) / len(inference_times)
            
            # Determine status
            status = "PASS"
            if avg_inference_time > self.thresholds["inference_time_critical"]:
                status = "FAIL"
                message = f"Critical: Inference time {avg_inference_time:.3f}s > {self.thresholds['inference_time_critical']}s"
            elif avg_inference_time > self.thresholds["inference_time_warning"]:
                status = "WARN"
                message = f"Warning: Inference time {avg_inference_time:.3f}s > {self.thresholds['inference_time_warning']}s"
            else:
                message = f"Inference speed good: {avg_inference_time:.3f}s average"
            
            details = {
                "avg_inference_time": avg_inference_time,
                "min_inference_time": min(inference_times),
                "max_inference_time": max(inference_times),
                "std_inference_time": torch.tensor(inference_times).std().item(),
                "num_runs": num_runs,
            }
            
            return HealthCheckResult(
                name="inference_speed",
                status=status,
                message=message,
                details=details,
                timestamp=time.time(),
                execution_time=time.time() - start_time
            )
            
        except Exception as e:
            return HealthCheckResult(
                name="inference_speed",
                status="FAIL",
                message=f"Inference speed check failed: {e}",
                details={"error": str(e)},
                timestamp=time.time(),
                execution_time=time.time() - start_time
            )
    
    def check_gradient_flow(self) -> HealthCheckResult:
        """Check gradient flow through the model."""
        start_time = time.time()
        
        try:
            # Enable gradient computation
            self.model.train()
            for param in self.model.parameters():
                param.requires_grad_(True)
            
            # Forward pass with dummy data
            from .error_handling import create_dummy_data
            dummy_data = create_dummy_data(num_nodes=50, num_edges=150)
            
            output = self.model(dummy_data)
            
            # Create dummy loss
            if isinstance(output, dict) and 'node_embeddings' in output:
                loss = output['node_embeddings'].sum()
            else:
                loss = output.sum() if hasattr(output, 'sum') else torch.tensor(1.0)
            
            # Backward pass
            loss.backward()
            
            # Analyze gradients
            grad_stats = {
                "total_params_with_grad": 0,
                "params_without_grad": 0,
                "zero_grad_params": 0,
                "nan_grad_params": 0,
                "inf_grad_params": 0,
                "max_grad_norm": 0.0,
                "mean_grad_norm": 0.0,
            }
            
            grad_norms = []
            
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    grad_stats["total_params_with_grad"] += 1
                    
                    grad_data = param.grad.data.flatten()
                    grad_norm = torch.norm(grad_data).item()
                    grad_norms.append(grad_norm)
                    
                    if torch.isnan(grad_data).any():
                        grad_stats["nan_grad_params"] += 1
                    
                    if torch.isinf(grad_data).any():
                        grad_stats["inf_grad_params"] += 1
                    
                    if grad_norm < 1e-8:
                        grad_stats["zero_grad_params"] += 1
                else:
                    grad_stats["params_without_grad"] += 1
            
            if grad_norms:
                grad_stats["max_grad_norm"] = max(grad_norms)
                grad_stats["mean_grad_norm"] = sum(grad_norms) / len(grad_norms)
            
            # Clear gradients
            self.model.zero_grad()
            
            # Determine status
            status = "PASS"
            issues = []
            
            if grad_stats["nan_grad_params"] > 0:
                status = "FAIL"
                issues.append(f"{grad_stats['nan_grad_params']} parameters with NaN gradients")
            
            if grad_stats["inf_grad_params"] > 0:
                status = "FAIL"
                issues.append(f"{grad_stats['inf_grad_params']} parameters with infinite gradients")
            
            if grad_stats["max_grad_norm"] > self.thresholds["gradient_norm_critical"]:
                status = "FAIL"
                issues.append(f"Critical gradient norm: {grad_stats['max_grad_norm']:.4f}")
            elif grad_stats["max_grad_norm"] > self.thresholds["gradient_norm_warning"]:
                status = "WARN"
                issues.append(f"High gradient norm: {grad_stats['max_grad_norm']:.4f}")
            
            if grad_stats["zero_grad_params"] > grad_stats["total_params_with_grad"] * 0.5:
                status = "WARN"
                issues.append("Over 50% of parameters have zero gradients")
            
            message = "Gradient flow healthy" if status == "PASS" else "; ".join(issues)
            
            return HealthCheckResult(
                name="gradient_flow",
                status=status,
                message=message,
                details=grad_stats,
                timestamp=time.time(),
                execution_time=time.time() - start_time
            )
            
        except Exception as e:
            return HealthCheckResult(
                name="gradient_flow",
                status="FAIL",
                message=f"Gradient flow check failed: {e}",
                details={"error": str(e)},
                timestamp=time.time(),
                execution_time=time.time() - start_time
            )
    
    def check_numerical_stability(self) -> HealthCheckResult:
        """Check numerical stability of model outputs."""
        start_time = time.time()
        
        try:
            self.model.eval()
            
            # Test with same input multiple times
            from .error_handling import create_dummy_data
            dummy_data = create_dummy_data(num_nodes=50, num_edges=150)
            
            outputs = []
            for _ in range(3):
                with torch.no_grad():
                    output = self.model(dummy_data)
                    if isinstance(output, dict) and 'node_embeddings' in output:
                        outputs.append(output['node_embeddings'])
                    else:
                        outputs.append(output)
            
            # Check consistency
            max_diff = 0.0
            for i in range(1, len(outputs)):
                diff = torch.abs(outputs[i] - outputs[0]).max().item()
                max_diff = max(max_diff, diff)
            
            # Check for NaN/inf in outputs
            has_nan = any(torch.isnan(out).any() for out in outputs)
            has_inf = any(torch.isinf(out).any() for out in outputs)
            
            # Determine status
            status = "PASS"
            issues = []
            
            if has_nan:
                status = "FAIL"
                issues.append("Model outputs contain NaN values")
            
            if has_inf:
                status = "FAIL"
                issues.append("Model outputs contain infinite values")
            
            if max_diff > 1e-4:
                status = "WARN"
                issues.append(f"Output inconsistency detected: max_diff={max_diff:.6f}")
            
            message = "Numerical stability good" if status == "PASS" else "; ".join(issues)
            
            details = {
                "max_output_diff": max_diff,
                "has_nan": has_nan,
                "has_inf": has_inf,
                "num_runs": len(outputs),
            }
            
            return HealthCheckResult(
                name="numerical_stability",
                status=status,
                message=message,
                details=details,
                timestamp=time.time(),
                execution_time=time.time() - start_time
            )
            
        except Exception as e:
            return HealthCheckResult(
                name="numerical_stability",
                status="FAIL",
                message=f"Numerical stability check failed: {e}",
                details={"error": str(e)},
                timestamp=time.time(),
                execution_time=time.time() - start_time
            )
    
    def generate_health_report(self, results: List[HealthCheckResult]) -> Dict[str, Any]:
        """Generate comprehensive health report."""
        report = {
            "timestamp": time.time(),
            "overall_status": "PASS",
            "total_checks": len(results),
            "passed": 0,
            "warnings": 0,
            "failures": 0,
            "checks": [],
            "summary": "",
        }
        
        for result in results:
            report["checks"].append({
                "name": result.name,
                "status": result.status,
                "message": result.message,
                "execution_time": result.execution_time,
                "details": result.details,
            })
            
            if result.status == "PASS":
                report["passed"] += 1
            elif result.status == "WARN":
                report["warnings"] += 1
                if report["overall_status"] == "PASS":
                    report["overall_status"] = "WARN"
            elif result.status == "FAIL":
                report["failures"] += 1
                report["overall_status"] = "FAIL"
        
        # Generate summary
        if report["overall_status"] == "PASS":
            report["summary"] = f"All {report['total_checks']} health checks passed successfully"
        elif report["overall_status"] == "WARN":
            report["summary"] = f"{report['warnings']} warnings detected in health checks"
        else:
            report["summary"] = f"{report['failures']} critical issues detected in health checks"
        
        return report

def create_dummy_data(num_nodes: int = 50, num_edges: int = 150):
    """Create dummy temporal data for health checks."""
    class DummyData:
        def __init__(self):
            self.edge_index = torch.randint(0, num_nodes, (2, num_edges))
            self.timestamps = torch.rand(num_edges) * 100.0
            self.num_nodes = num_nodes
            self.node_features = torch.randn(num_nodes, 64)
            self.edge_attr = torch.randn(num_edges, 32)
    
    return DummyData()