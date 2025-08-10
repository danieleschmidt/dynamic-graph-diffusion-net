"""Edge computing and mobile inference optimizations for DGDN."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import time
import logging
import json
from collections import deque
import threading
import queue

from ..models.dgdn import DynamicGraphDiffusionNet


class EdgeDGDN(nn.Module):
    """Optimized DGDN variant for edge computing deployment."""
    
    def __init__(
        self,
        full_model: DynamicGraphDiffusionNet,
        compression_config: Dict[str, Any] = None
    ):
        super().__init__()
        
        self.compression_config = compression_config or {}
        self.logger = logging.getLogger('DGDN.EdgeDGDN')
        
        # Compression parameters
        self.quantization_bits = self.compression_config.get('quantization_bits', 8)
        self.pruning_ratio = self.compression_config.get('pruning_ratio', 0.5)
        self.knowledge_distillation = self.compression_config.get('knowledge_distillation', True)
        self.layer_fusion = self.compression_config.get('layer_fusion', True)
        
        # Create compressed model
        self.compressed_model = self._compress_model(full_model)
        
        # Edge-specific optimizations
        self.edge_cache = EdgeCache()
        self.batch_processor = EdgeBatchProcessor()
        
        # Performance tracking
        self.inference_times = deque(maxlen=100)
        self.memory_usage = deque(maxlen=100)
        
    def _compress_model(self, full_model: DynamicGraphDiffusionNet):
        """Compress full model for edge deployment."""
        compressed = EdgeCompressedDGDN(
            node_dim=full_model.node_dim,
            edge_dim=full_model.edge_dim,
            hidden_dim=max(64, full_model.hidden_dim // 4),  # Reduce hidden dimension
            num_layers=max(1, full_model.num_layers // 2),   # Reduce layers
            quantization_bits=self.quantization_bits
        )
        
        # Transfer knowledge from full model
        if self.knowledge_distillation:
            self._distill_knowledge(full_model, compressed)
            
        # Apply pruning
        if self.pruning_ratio > 0:
            self._apply_pruning(compressed)
            
        return compressed
        
    def _distill_knowledge(self, teacher: DynamicGraphDiffusionNet, student: nn.Module):
        """Knowledge distillation from full model to compressed model."""
        self.logger.info("Performing knowledge distillation...")
        
        # Create synthetic training data for distillation
        num_samples = 1000
        synthetic_data = self._generate_synthetic_data(teacher, num_samples)
        
        # Distillation training
        teacher.eval()
        student.train()
        
        optimizer = torch.optim.Adam(student.parameters(), lr=1e-3)
        distillation_loss_fn = nn.KLDivLoss(reduction='batchmean')
        
        for epoch in range(10):  # Quick distillation
            epoch_loss = 0
            
            for batch_data in synthetic_data:
                optimizer.zero_grad()
                
                # Teacher predictions (no gradients)
                with torch.no_grad():
                    teacher_output = teacher(batch_data)
                    teacher_logits = teacher_output.get('predictions', teacher_output['node_embeddings'])
                    
                # Student predictions
                student_output = student(batch_data)
                student_logits = student_output.get('predictions', student_output['node_embeddings'])
                
                # Distillation loss
                loss = distillation_loss_fn(
                    F.log_softmax(student_logits / 3.0, dim=-1),  # Temperature scaling
                    F.softmax(teacher_logits / 3.0, dim=-1)
                )
                
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                
            self.logger.info(f"Distillation epoch {epoch}, loss: {epoch_loss:.4f}")
            
    def _generate_synthetic_data(self, model: DynamicGraphDiffusionNet, num_samples: int):
        """Generate synthetic data for knowledge distillation."""
        synthetic_batches = []
        
        # Create simple synthetic temporal graphs
        for _ in range(num_samples // 32):  # Batch size 32
            batch_size = 32
            num_nodes = np.random.randint(10, 50)
            num_edges = np.random.randint(20, num_nodes * 2)
            
            # Random node features
            node_features = torch.randn(num_nodes, model.node_dim)
            
            # Random edges
            edge_index = torch.randint(0, num_nodes, (2, num_edges))
            edge_features = torch.randn(num_edges, model.edge_dim) if model.edge_dim > 0 else None
            timestamps = torch.sort(torch.rand(num_edges) * 100)[0]
            
            # Create batch data structure (simplified)
            batch_data = type('Data', (), {
                'x': node_features,
                'edge_index': edge_index,
                'edge_attr': edge_features,
                'timestamps': timestamps
            })()
            
            synthetic_batches.append(batch_data)
            
        return synthetic_batches
        
    def _apply_pruning(self, model: nn.Module):
        """Apply structured pruning to reduce model size."""
        self.logger.info(f"Applying {self.pruning_ratio:.2f} pruning ratio...")
        
        # Magnitude-based pruning
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                # Calculate importance scores (L2 norm of weights)
                weight = module.weight.data
                importance = torch.norm(weight, dim=1)
                
                # Determine pruning threshold
                k = int(weight.size(0) * (1 - self.pruning_ratio))
                if k > 0:
                    threshold = torch.topk(importance, k).values[-1]
                    
                    # Create pruning mask
                    mask = importance >= threshold
                    
                    # Apply pruning
                    module.weight.data = weight * mask.unsqueeze(1)
                    if module.bias is not None:
                        module.bias.data = module.bias.data * mask
                        
    def forward(self, data, use_cache: bool = True):
        """Optimized forward pass for edge devices."""
        start_time = time.time()
        
        # Check cache first
        if use_cache:
            cache_key = self._compute_cache_key(data)
            cached_result = self.edge_cache.get(cache_key)
            if cached_result is not None:
                self.inference_times.append(time.time() - start_time)
                return cached_result
                
        # Batch processing for efficiency
        processed_data = self.batch_processor.process(data)
        
        # Forward pass through compressed model
        with torch.no_grad():  # Save memory on edge devices
            output = self.compressed_model(processed_data)
            
        # Cache result
        if use_cache:
            self.edge_cache.put(cache_key, output)
            
        # Track performance
        inference_time = time.time() - start_time
        self.inference_times.append(inference_time)
        self._track_memory_usage()
        
        return output
        
    def _compute_cache_key(self, data) -> str:
        """Compute cache key for input data."""
        # Simple hash of input data
        if hasattr(data, 'x') and hasattr(data, 'edge_index'):
            node_hash = hash(data.x.data_ptr())
            edge_hash = hash(data.edge_index.data_ptr())
            return f"{node_hash}_{edge_hash}"
        return "default"
        
    def _track_memory_usage(self):
        """Track current memory usage."""
        if torch.cuda.is_available():
            memory_mb = torch.cuda.memory_allocated() / (1024 ** 2)
        else:
            memory_mb = 0  # CPU memory tracking would require psutil
            
        self.memory_usage.append(memory_mb)
        
    def get_edge_stats(self) -> Dict[str, Any]:
        """Get edge computing performance statistics."""
        stats = {
            'model_size_mb': self._get_model_size_mb(),
            'cache_hit_rate': self.edge_cache.get_hit_rate(),
            'cache_size': self.edge_cache.get_size()
        }
        
        if self.inference_times:
            stats.update({
                'avg_inference_time_ms': np.mean(self.inference_times) * 1000,
                'p95_inference_time_ms': np.percentile(self.inference_times, 95) * 1000,
                'max_inference_time_ms': np.max(self.inference_times) * 1000
            })
            
        if self.memory_usage:
            stats.update({
                'avg_memory_usage_mb': np.mean(self.memory_usage),
                'peak_memory_usage_mb': np.max(self.memory_usage)
            })
            
        return stats
        
    def _get_model_size_mb(self) -> float:
        """Calculate model size in MB."""
        param_size = sum(p.numel() * p.element_size() for p in self.compressed_model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in self.compressed_model.buffers())
        return (param_size + buffer_size) / (1024 ** 2)


class EdgeCompressedDGDN(nn.Module):
    """Lightweight DGDN architecture for edge deployment."""
    
    def __init__(
        self,
        node_dim: int,
        edge_dim: int = 0,
        hidden_dim: int = 64,
        num_layers: int = 2,
        quantization_bits: int = 8
    ):
        super().__init__()
        
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.quantization_bits = quantization_bits
        
        # Lightweight components
        self.node_encoder = QuantizedLinear(node_dim, hidden_dim, quantization_bits)
        
        if edge_dim > 0:
            self.edge_encoder = QuantizedLinear(edge_dim, hidden_dim // 4, quantization_bits)
            
        # Simplified temporal encoding
        self.time_encoder = SimpleTimeEncoder(16)  # Reduced dimension
        
        # Lightweight message passing layers
        self.layers = nn.ModuleList([
            LightweightDGDNLayer(hidden_dim, quantization_bits)
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_proj = QuantizedLinear(hidden_dim, hidden_dim, quantization_bits)
        
    def forward(self, data):
        """Lightweight forward pass."""
        # Encode nodes
        x = self.node_encoder(data.x)
        
        # Encode edges (if available)
        edge_attr = None
        if self.edge_dim > 0 and hasattr(data, 'edge_attr') and data.edge_attr is not None:
            edge_attr = self.edge_encoder(data.edge_attr)
            
        # Encode time
        if hasattr(data, 'timestamps'):
            time_emb = self.time_encoder(data.timestamps)
            
            # Add time information to node features
            if time_emb.size(0) == x.size(0):
                x = x + time_emb[:x.size(0)]
                
        # Message passing
        for layer in self.layers:
            x = layer(x, data.edge_index, edge_attr)
            
        # Output projection
        x = self.output_proj(x)
        
        return {'node_embeddings': x}


class QuantizedLinear(nn.Module):
    """Quantized linear layer for edge deployment."""
    
    def __init__(self, in_features: int, out_features: int, quantization_bits: int = 8):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.quantization_bits = quantization_bits
        
        # Full precision weights for training
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        
        # Quantization parameters
        self.register_buffer('weight_scale', torch.tensor(1.0))
        self.register_buffer('weight_zero_point', torch.tensor(0))
        
        # Initialize weights
        nn.init.kaiming_uniform_(self.weight)
        
    def quantize_weights(self):
        """Quantize weights for inference."""
        if self.quantization_bits >= 32:
            return self.weight
            
        # Calculate quantization parameters
        w_min, w_max = self.weight.min().item(), self.weight.max().item()
        
        if self.quantization_bits == 8:
            qmin, qmax = -128, 127
        else:
            qmin, qmax = -(2**(self.quantization_bits-1)), 2**(self.quantization_bits-1) - 1
            
        scale = (w_max - w_min) / (qmax - qmin)
        zero_point = qmin - w_min / scale
        
        self.weight_scale.data = torch.tensor(scale)
        self.weight_zero_point.data = torch.tensor(zero_point)
        
        # Quantize and dequantize
        quantized = torch.clamp(
            torch.round(self.weight / scale + zero_point),
            qmin, qmax
        )
        
        return (quantized - zero_point) * scale
        
    def forward(self, x):
        """Forward pass with quantized weights."""
        if self.training:
            weight = self.weight
        else:
            weight = self.quantize_weights()
            
        return F.linear(x, weight, self.bias)


class SimpleTimeEncoder(nn.Module):
    """Simplified time encoder for edge devices."""
    
    def __init__(self, time_dim: int):
        super().__init__()
        self.time_dim = time_dim
        
        # Use fewer basis functions
        self.w = nn.Parameter(torch.randn(time_dim // 2))
        self.b = nn.Parameter(torch.randn(time_dim // 2))
        
    def forward(self, timestamps):
        """Encode timestamps with reduced computation."""
        if timestamps.dim() == 0:
            timestamps = timestamps.unsqueeze(0)
            
        # Simple trigonometric encoding
        scaled_time = timestamps.unsqueeze(-1) * self.w + self.b
        
        # Concatenate sin and cos
        sin_enc = torch.sin(scaled_time)
        cos_enc = torch.cos(scaled_time)
        
        time_encoding = torch.cat([sin_enc, cos_enc], dim=-1)
        
        return time_encoding


class LightweightDGDNLayer(nn.Module):
    """Lightweight DGDN layer for edge deployment."""
    
    def __init__(self, hidden_dim: int, quantization_bits: int = 8):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # Simplified message passing
        self.message_net = nn.Sequential(
            QuantizedLinear(hidden_dim * 2, hidden_dim, quantization_bits),
            nn.ReLU(),
            QuantizedLinear(hidden_dim, hidden_dim, quantization_bits)
        )
        
        # Simple update function
        self.update_net = QuantizedLinear(hidden_dim * 2, hidden_dim, quantization_bits)
        
        # Layer normalization (lightweight)
        self.norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, x, edge_index, edge_attr=None):
        """Lightweight message passing."""
        row, col = edge_index
        
        if row.max() >= x.size(0) or col.max() >= x.size(0):
            # Handle edge case where edge indices are out of bounds
            return x
            
        # Simple message computation
        messages = torch.cat([x[row], x[col]], dim=-1)
        messages = self.message_net(messages)
        
        # Aggregate messages (simple scatter_add)
        aggregated = torch.zeros_like(x)
        aggregated.scatter_add_(0, col.unsqueeze(-1).expand(-1, messages.size(-1)), messages)
        
        # Update node features
        updated = self.update_net(torch.cat([x, aggregated], dim=-1))
        
        # Residual connection and normalization
        x = self.norm(x + updated)
        
        return x


class EdgeCache:
    """Simple LRU cache for edge inference."""
    
    def __init__(self, max_size: int = 100):
        self.max_size = max_size
        self.cache = {}
        self.access_order = deque()
        
        # Statistics
        self.hits = 0
        self.misses = 0
        
    def get(self, key: str):
        """Get cached result."""
        if key in self.cache:
            # Move to end (most recently used)
            self.access_order.remove(key)
            self.access_order.append(key)
            self.hits += 1
            return self.cache[key]
        else:
            self.misses += 1
            return None
            
    def put(self, key: str, value):
        """Cache result."""
        if key in self.cache:
            # Update existing
            self.cache[key] = value
            self.access_order.remove(key)
            self.access_order.append(key)
        else:
            # Add new
            if len(self.cache) >= self.max_size:
                # Remove least recently used
                lru_key = self.access_order.popleft()
                del self.cache[lru_key]
                
            self.cache[key] = value
            self.access_order.append(key)
            
    def get_hit_rate(self) -> float:
        """Get cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0
        
    def get_size(self) -> int:
        """Get current cache size."""
        return len(self.cache)
        
    def clear(self):
        """Clear cache."""
        self.cache.clear()
        self.access_order.clear()
        self.hits = 0
        self.misses = 0


class EdgeBatchProcessor:
    """Batch processor optimized for edge devices."""
    
    def __init__(self, max_batch_size: int = 8):
        self.max_batch_size = max_batch_size
        self.pending_batch = []
        self.batch_queue = queue.Queue()
        
    def process(self, data):
        """Process data with edge-optimized batching."""
        # For now, just return the data as-is
        # In a real implementation, this would accumulate small requests
        # into batches for more efficient processing
        return data
        
    def add_to_batch(self, data):
        """Add data to current batch."""
        self.pending_batch.append(data)
        
        if len(self.pending_batch) >= self.max_batch_size:
            self._flush_batch()
            
    def _flush_batch(self):
        """Flush current batch to processing queue."""
        if self.pending_batch:
            self.batch_queue.put(self.pending_batch.copy())
            self.pending_batch.clear()
            
    def get_batch(self):
        """Get next batch for processing."""
        if not self.batch_queue.empty():
            return self.batch_queue.get()
        return None


class EdgeOptimizer:
    """Optimizer for edge deployment configurations."""
    
    def __init__(self):
        self.logger = logging.getLogger('DGDN.EdgeOptimizer')
        
    def optimize_for_device(self, model: DynamicGraphDiffusionNet, device_specs: Dict[str, Any]) -> EdgeDGDN:
        """Optimize model for specific edge device."""
        self.logger.info(f"Optimizing model for device: {device_specs}")
        
        # Determine optimal compression based on device constraints
        compression_config = self._calculate_compression_config(device_specs)
        
        # Create edge-optimized model
        edge_model = EdgeDGDN(model, compression_config)
        
        # Device-specific optimizations
        if device_specs.get('gpu_memory_mb', 0) < 1000:
            # Very limited GPU memory
            edge_model.compressed_model = self._apply_aggressive_compression(edge_model.compressed_model)
            
        if device_specs.get('cpu_cores', 1) < 4:
            # Limited CPU cores
            self._optimize_for_single_thread(edge_model)
            
        return edge_model
        
    def _calculate_compression_config(self, device_specs: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate optimal compression configuration."""
        memory_mb = device_specs.get('memory_mb', 2048)
        gpu_memory_mb = device_specs.get('gpu_memory_mb', 0)
        cpu_cores = device_specs.get('cpu_cores', 4)
        
        config = {}
        
        # Quantization based on memory constraints
        if memory_mb < 512:
            config['quantization_bits'] = 4
        elif memory_mb < 2048:
            config['quantization_bits'] = 8
        else:
            config['quantization_bits'] = 16
            
        # Pruning based on compute constraints
        if cpu_cores < 2:
            config['pruning_ratio'] = 0.75
        elif cpu_cores < 4:
            config['pruning_ratio'] = 0.5
        else:
            config['pruning_ratio'] = 0.25
            
        # Other optimizations
        config['knowledge_distillation'] = memory_mb > 1024
        config['layer_fusion'] = True
        
        return config
        
    def _apply_aggressive_compression(self, model: nn.Module) -> nn.Module:
        """Apply aggressive compression for very constrained devices."""
        # Further reduce dimensions
        for module in model.modules():
            if isinstance(module, nn.Linear):
                # Reduce output features by half
                if module.out_features > 16:
                    new_out_features = module.out_features // 2
                    new_weight = module.weight.data[:new_out_features]
                    new_bias = module.bias.data[:new_out_features] if module.bias is not None else None
                    
                    module.weight = nn.Parameter(new_weight)
                    if new_bias is not None:
                        module.bias = nn.Parameter(new_bias)
                        
        return model
        
    def _optimize_for_single_thread(self, edge_model: EdgeDGDN):
        """Optimize for single-threaded execution."""
        # Disable parallel processing in batch processor
        edge_model.batch_processor.max_batch_size = 1
        
        # Reduce cache size to save memory
        edge_model.edge_cache.max_size = 10


class MobileInference:
    """Mobile-specific inference optimizations."""
    
    def __init__(self, model: EdgeDGDN):
        self.model = model
        self.logger = logging.getLogger('DGDN.MobileInference')
        
        # Mobile-specific settings
        self.power_save_mode = False
        self.thermal_throttling = False
        self.battery_aware = True
        
        # Performance tracking
        self.power_consumption = deque(maxlen=100)
        self.thermal_state = deque(maxlen=100)
        
    def set_power_mode(self, power_save: bool):
        """Enable/disable power saving mode."""
        self.power_save_mode = power_save
        
        if power_save:
            # Reduce model precision further
            self._enable_power_optimizations()
        else:
            self._disable_power_optimizations()
            
        self.logger.info(f"Power save mode: {'enabled' if power_save else 'disabled'}")
        
    def _enable_power_optimizations(self):
        """Enable power saving optimizations."""
        # Increase cache size to avoid recomputation
        self.model.edge_cache.max_size = 200
        
        # Use lower precision quantization
        for module in self.model.compressed_model.modules():
            if isinstance(module, QuantizedLinear):
                module.quantization_bits = min(4, module.quantization_bits)
                
    def _disable_power_optimizations(self):
        """Disable power saving optimizations."""
        # Reset to normal cache size
        self.model.edge_cache.max_size = 100
        
        # Use higher precision
        for module in self.model.compressed_model.modules():
            if isinstance(module, QuantizedLinear):
                module.quantization_bits = 8
                
    def adaptive_inference(self, data, thermal_state: float = 0.5, battery_level: float = 1.0):
        """Perform adaptive inference based on device state."""
        # Adjust inference strategy based on device state
        if battery_level < 0.2:
            # Low battery: maximum power saving
            self.set_power_mode(True)
            use_cache = True
            
        elif thermal_state > 0.8:
            # High temperature: reduce computation
            self.thermal_throttling = True
            use_cache = True
            
        else:
            # Normal operation
            self.set_power_mode(False)
            self.thermal_throttling = False
            use_cache = True
            
        # Perform inference
        start_time = time.time()
        result = self.model.forward(data, use_cache=use_cache)
        inference_time = time.time() - start_time
        
        # Estimate power consumption (simplified)
        base_power = 1.0  # Base power in watts
        compute_power = 2.0 if not self.power_save_mode else 0.5
        estimated_power = base_power + (compute_power * inference_time)
        
        self.power_consumption.append(estimated_power)
        self.thermal_state.append(thermal_state)
        
        return {
            **result,
            'inference_time': inference_time,
            'estimated_power_consumption': estimated_power,
            'power_save_mode': self.power_save_mode
        }
        
    def get_mobile_stats(self) -> Dict[str, Any]:
        """Get mobile inference statistics."""
        stats = self.model.get_edge_stats()
        
        if self.power_consumption:
            stats.update({
                'avg_power_consumption_w': np.mean(self.power_consumption),
                'max_power_consumption_w': np.max(self.power_consumption)
            })
            
        stats.update({
            'power_save_mode': self.power_save_mode,
            'thermal_throttling': self.thermal_throttling,
            'battery_aware': self.battery_aware
        })
        
        return stats