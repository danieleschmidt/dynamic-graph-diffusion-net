# Development Guide

This guide provides detailed information for developers working on the Dynamic Graph Diffusion Net library.

## Quick Setup

```bash
# Clone and setup
git clone https://github.com/yourusername/dynamic-graph-diffusion-net.git
cd dynamic-graph-diffusion-net
make install-dev

# Verify installation
make test
```

## Development Workflow

### 1. Environment Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install development dependencies
make install-dev
```

### 2. Code Development

#### Project Structure
```
src/dgdn/
├── __init__.py          # Package initialization
├── models/              # Core model implementations
│   ├── __init__.py
│   ├── dgdn.py         # Main DGDN model
│   └── layers.py       # Individual layers
├── data/               # Data handling
│   ├── __init__.py
│   ├── temporal.py     # Temporal graph data structures
│   └── datasets.py     # Dataset loaders
├── modules/            # Reusable components
│   ├── __init__.py
│   ├── encoding.py     # Time encoding modules
│   └── diffusion.py    # Diffusion mechanisms
└── utils/              # Utility functions
    ├── __init__.py
    └── visualization.py
```

#### Coding Standards

**Type Hints**: All functions should include type annotations
```python
from typing import Optional, Tuple
import torch

def process_graph(
    edge_index: torch.Tensor,
    node_features: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Process graph data with type safety."""
    pass
```

**Documentation**: Use Google-style docstrings
```python
def compute_attention(query: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
    """Compute attention weights between query and key.
    
    Args:
        query: Query tensor of shape [batch_size, seq_len, dim]
        key: Key tensor of shape [batch_size, seq_len, dim]
        
    Returns:
        Attention weights of shape [batch_size, seq_len, seq_len]
        
    Raises:
        ValueError: If input dimensions don't match
        
    Example:
        >>> attention = compute_attention(query, key)
        >>> assert attention.shape == (batch_size, seq_len, seq_len)
    """
    pass
```

### 3. Testing Strategy

#### Test Categories

**Unit Tests**: Test individual components
```python
# tests/models/test_dgdn.py
def test_dgdn_forward_pass(sample_graph_data):
    model = DynamicGraphDiffusionNet(node_dim=64, edge_dim=32)
    output = model(sample_graph_data)
    assert output.shape[0] == sample_graph_data.num_nodes
```

**Integration Tests**: Test component interactions
```python
# tests/integration/test_training.py
def test_end_to_end_training():
    # Test complete training pipeline
    pass
```

**Performance Tests**: Benchmark critical functions
```python
# tests/performance/test_scalability.py
@pytest.mark.slow
def test_large_graph_performance():
    # Test with graphs of 100K+ nodes
    pass
```

#### Running Tests

```bash
# All tests
make test

# With coverage
make test-cov

# Specific test file
pytest tests/models/test_dgdn.py

# Performance tests (marked as slow)
pytest -m slow

# Parallel execution
pytest -n auto
```

### 4. Code Quality

#### Automated Checks

```bash
# Format code
make format

# Run linting
make lint

# All quality checks
pre-commit run --all-files
```

#### Manual Review Checklist

- [ ] Code follows project conventions
- [ ] All functions have type hints
- [ ] Comprehensive docstrings with examples
- [ ] Tests cover new functionality
- [ ] Performance impact assessed
- [ ] Memory usage optimized
- [ ] Error handling implemented

### 5. Performance Optimization

#### Memory Management

```python
# Use torch.no_grad() for inference
@torch.no_grad()
def inference(model, data):
    return model(data)

# Clear intermediate tensors
def memory_efficient_forward(x):
    intermediate = expensive_operation(x)
    result = final_operation(intermediate)
    del intermediate  # Explicit cleanup
    return result
```

#### GPU Optimization

```python
# Use appropriate data types
x = x.to(dtype=torch.float16)  # Mixed precision

# Efficient tensor operations
# Avoid: multiple operations creating intermediate tensors
result = x.transpose(0, 1).contiguous().view(-1, dim)

# Better: single operation
result = x.permute(1, 0).contiguous().view(-1, dim)
```

### 6. Debugging

#### Common Issues

**CUDA Memory Errors**:
```python
# Monitor memory usage
import torch
print(f"GPU memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

# Use smaller batch sizes
# Clear cache between runs
torch.cuda.empty_cache()
```

**Numerical Instabilities**:
```python
# Add numerical stability
eps = 1e-8
result = torch.sqrt(x + eps)

# Check for NaN/Inf
assert not torch.isnan(result).any()
assert not torch.isinf(result).any()
```

#### Debugging Tools

```python
# Tensor debugging
def debug_tensor(x, name="tensor"):
    print(f"{name}: shape={x.shape}, dtype={x.dtype}")
    print(f"  min={x.min():.4f}, max={x.max():.4f}")
    print(f"  mean={x.mean():.4f}, std={x.std():.4f}")
    print(f"  has_nan={torch.isnan(x).any()}")

# Model debugging
def debug_model(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")
```

### 7. Documentation

#### API Documentation

Generate docs:
```bash
make docs
make docs-serve  # Local preview
```

#### Code Examples

All public APIs should include working examples:
```python
def temporal_attention(queries, keys, timestamps):
    """Temporal attention mechanism.
    
    Example:
        >>> import torch
        >>> from dgdn.modules import temporal_attention
        >>> 
        >>> queries = torch.randn(10, 64)
        >>> keys = torch.randn(20, 64)
        >>> timestamps = torch.linspace(0, 100, 20)
        >>> 
        >>> attention_weights = temporal_attention(queries, keys, timestamps)
        >>> assert attention_weights.shape == (10, 20)
    """
    pass
```

### 8. Release Process

#### Version Management

Follow semantic versioning:
- **0.1.0 → 0.1.1**: Bug fixes
- **0.1.0 → 0.2.0**: New features
- **0.1.0 → 1.0.0**: Breaking changes

#### Release Checklist

1. [ ] All tests pass
2. [ ] Documentation updated
3. [ ] Changelog updated
4. [ ] Version bumped in `pyproject.toml`
5. [ ] Git tag created
6. [ ] Package built and tested
7. [ ] Published to PyPI

```bash
# Build and test
make clean
make build
make publish-test  # Test PyPI first
make publish      # Production PyPI
```

### 9. Advanced Development

#### Custom Extensions

```python
# Custom temporal encoders
class CustomTimeEncoder(nn.Module):
    def __init__(self, time_dim):
        super().__init__()
        # Implementation
        
    def forward(self, timestamps):
        # Encoding logic
        return encoded_time

# Register custom components
from dgdn.registry import register_encoder
register_encoder("custom", CustomTimeEncoder)
```

#### Profiling and Optimization

```python
# Profile GPU usage
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU, 
                torch.profiler.ProfilerActivity.CUDA]
) as prof:
    model(data)

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
```

### 10. Contributing Back

#### Pull Request Process

1. Create feature branch from `main`
2. Implement changes with tests
3. Update documentation
4. Run full test suite
5. Submit PR with clear description
6. Address review feedback
7. Merge after approval

#### Code Review Guidelines

**As Author**:
- Keep PRs focused and reasonably sized
- Include comprehensive tests
- Write clear commit messages
- Respond promptly to feedback

**As Reviewer**:
- Focus on correctness and maintainability
- Suggest specific improvements
- Test functionality when needed
- Approve when ready

---

## Getting Help

- **Documentation**: Check the full docs first
- **Issues**: Search existing issues before creating new ones
- **Discord**: Join our developer community
- **Maintainers**: Tag `@maintainers` for urgent issues

Happy coding! 🚀