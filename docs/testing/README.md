# Testing Guide for Dynamic Graph Diffusion Network (DGDN)

## Overview

This document provides comprehensive guidance for testing the DGDN library. Our testing strategy ensures reliability, performance, and maintainability across all components.

## Testing Philosophy

### Core Principles

1. **Comprehensive Coverage**: Every component should have corresponding tests
2. **Fast Feedback**: Unit tests should run quickly for rapid development
3. **Realistic Scenarios**: Integration tests should mirror real-world usage
4. **Performance Awareness**: Critical paths should have performance benchmarks
5. **Reproducibility**: All tests should be deterministic and reproducible

### Testing Pyramid

```
       /\
      /  \
     / E2E\     <- End-to-End Tests (slow, comprehensive)
    /______\
   /        \
  /Integration\ <- Integration Tests (medium speed, realistic)
 /____________\
/              \
/   Unit Tests  \ <- Unit Tests (fast, focused)
/________________\
```

## Test Structure

### Directory Organization

```
tests/
├── __init__.py
├── conftest.py              # Shared pytest configuration
├── fixtures/                # Test data and configurations
│   ├── __init__.py
│   ├── graph_data.py       # Graph data fixtures
│   └── model_configs.py    # Model configuration fixtures
├── utils/                   # Testing utilities
│   ├── __init__.py
│   ├── assertions.py       # Custom assertions
│   └── helpers.py          # Helper functions
├── unit/                    # Unit tests
│   ├── __init__.py
│   ├── test_models.py      # Model component tests
│   ├── test_layers.py      # Layer tests
│   ├── test_attention.py   # Attention mechanism tests
│   ├── test_diffusion.py   # Diffusion sampler tests
│   └── test_temporal.py    # Temporal encoding tests
├── integration/             # Integration tests
│   ├── __init__.py
│   ├── test_end_to_end.py  # Full pipeline tests
│   ├── test_training.py    # Training pipeline tests
│   └── test_inference.py   # Inference pipeline tests
├── performance/             # Performance tests
│   ├── __init__.py
│   ├── test_benchmarks.py  # Benchmark tests
│   ├── test_memory.py      # Memory usage tests
│   └── test_scalability.py # Scalability tests
└── e2e/                     # End-to-end tests
    ├── __init__.py
    └── test_training_pipeline.py # Complete workflows
```

## Test Categories

### 1. Unit Tests

**Purpose**: Test individual components in isolation
**Speed**: Very fast (< 1 second per test)
**Scope**: Single functions, classes, or small modules

**Examples**:
- Edge-time encoder functionality
- Attention mechanism calculations
- Individual layer forward/backward passes
- Utility functions

**Guidelines**:
- Mock external dependencies
- Test edge cases and error conditions
- Verify mathematical correctness
- Ensure gradient flow

### 2. Integration Tests

**Purpose**: Test component interactions
**Speed**: Fast to medium (1-10 seconds per test)
**Scope**: Multiple components working together

**Examples**:
- Model forward pass with real data
- Training loop with multiple epochs
- Data loading and preprocessing pipeline
- Loss computation and backpropagation

**Guidelines**:
- Use realistic data sizes
- Test common workflows
- Verify data flow between components
- Check for memory leaks

### 3. Performance Tests

**Purpose**: Ensure performance requirements are met
**Speed**: Medium to slow (10 seconds - 5 minutes)
**Scope**: Performance-critical operations

**Examples**:
- Training speed benchmarks
- Memory usage profiling
- Scalability tests with large graphs
- GPU utilization tests

**Guidelines**:
- Use consistent hardware for benchmarking
- Set performance thresholds
- Profile memory usage
- Test with different data sizes

### 4. End-to-End Tests

**Purpose**: Test complete user workflows
**Speed**: Slow (minutes to hours)
**Scope**: Full application scenarios

**Examples**:
- Complete research workflow
- Production deployment pipeline
- Model export and serving
- Hyperparameter optimization

**Guidelines**:
- Mirror real user scenarios
- Test with production-like data
- Verify final outputs
- Include error handling

## Test Configuration

### pytest Configuration (`conftest.py`)

```python
import pytest
import torch
import numpy as np
from typing import Generator

@pytest.fixture(scope="session", autouse=True)
def set_random_seeds():
    """Set random seeds for reproducible tests."""
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

@pytest.fixture(scope="session")
def device():
    """Get the best available device for testing."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

@pytest.fixture(autouse=True)
def cleanup_cuda():
    """Cleanup CUDA memory after each test."""
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
```

### Markers

We use pytest markers to categorize tests:

```python
# pytest.ini or pyproject.toml
[tool.pytest.ini_options]
markers = [
    "slow: marks tests as slow (deselect with '-m \"not slow\"')",
    "gpu: marks tests that require GPU",
    "memory_intensive: marks tests that use significant memory",
    "integration: marks integration tests",
    "unit: marks unit tests",
    "performance: marks performance tests",
    "e2e: marks end-to-end tests"
]
```

## Running Tests

### Basic Test Execution

```bash
# Run all tests
pytest

# Run only unit tests
pytest tests/unit/

# Run with coverage
pytest --cov=src/dgdn --cov-report=html

# Run tests in parallel
pytest -n auto

# Run only fast tests
pytest -m "not slow"

# Run specific test file
pytest tests/unit/test_models.py

# Run specific test
pytest tests/unit/test_models.py::TestDGDN::test_forward_pass
```

### Performance Testing

```bash
# Run performance benchmarks
pytest tests/performance/ -v

# Generate performance report
pytest tests/performance/ --benchmark-json=benchmark.json

# Compare with baseline
pytest tests/performance/ --benchmark-compare=baseline.json
```

### Continuous Integration

```bash
# CI test suite (fast tests only)
pytest tests/ -m "not slow" --cov=src/dgdn --cov-fail-under=90

# Nightly test suite (all tests)
pytest tests/ --cov=src/dgdn --cov-report=xml
```

## Writing Good Tests

### Test Naming Conventions

- Test files: `test_<module_name>.py`
- Test classes: `Test<ClassName>`
- Test methods: `test_<what_is_being_tested>`

**Examples**:
```python
# Good test names
def test_edge_time_encoder_forward_pass()
def test_attention_mechanism_with_masked_input()
def test_model_training_convergence_on_small_dataset()

# Bad test names
def test_model()
def test_function1()
def test_works()
```

### Test Structure (Arrange-Act-Assert)

```python
def test_diffusion_sampler_forward_step():
    # Arrange
    batch_size = 16
    hidden_dim = 64
    diffusion_steps = 5
    
    sampler = DiffusionSampler(hidden_dim, diffusion_steps)
    x = torch.randn(batch_size, hidden_dim)
    timestep = torch.randint(0, diffusion_steps, (batch_size,))
    
    # Act
    output = sampler.forward_step(x, timestep)
    
    # Assert
    assert output.shape == (batch_size, hidden_dim)
    assert torch.isfinite(output).all()
    assert output.dtype == torch.float32
```

### Custom Assertions

Use our custom assertions for common patterns:

```python
from tests.utils.assertions import (
    assert_tensor_shape,
    assert_tensor_finite,
    assert_edge_index_valid,
    assert_probabilities
)

def test_model_output():
    model = DynamicGraphDiffusionNet(config)
    output = model(data)
    
    # Use custom assertions
    assert_tensor_shape(output, (batch_size, output_dim))
    assert_tensor_finite(output)
    assert_probabilities(torch.sigmoid(output))
```

### Parameterized Tests

Use parameterization for testing multiple scenarios:

```python
@pytest.mark.parametrize("hidden_dim,num_heads", [
    (64, 4),
    (128, 8),
    (256, 16),
])
def test_attention_with_different_configs(hidden_dim, num_heads):
    attention = MultiHeadAttention(hidden_dim, num_heads)
    x = torch.randn(32, 10, hidden_dim)
    output = attention(x)
    assert output.shape == x.shape
```

### Fixtures for Test Data

Use fixtures for reusable test data:

```python
@pytest.fixture
def sample_temporal_graph():
    return {
        'edge_index': torch.tensor([[0, 1, 2], [1, 2, 0]]),
        'timestamps': torch.tensor([0.0, 1.0, 2.0]),
        'node_features': torch.randn(3, 16),
        'edge_attr': torch.randn(3, 8)
    }

def test_model_forward(sample_temporal_graph):
    model = DynamicGraphDiffusionNet(config)
    output = model(sample_temporal_graph)
    # ... assertions
```

## Testing Best Practices

### 1. Test Independence

Each test should be independent and not rely on the state from other tests:

```python
# Good: Each test sets up its own data
def test_function_a():
    data = create_test_data()
    result = function_a(data)
    assert result == expected

def test_function_b():
    data = create_test_data()
    result = function_b(data)
    assert result == expected

# Bad: Tests depend on shared state
data = create_test_data()

def test_function_a():
    result = function_a(data)
    assert result == expected
    data.modify()  # Modifies shared state

def test_function_b():
    result = function_b(data)  # Depends on modified state
    assert result == expected
```

### 2. Reproducible Tests

Always set random seeds for reproducible tests:

```python
def test_random_behavior():
    torch.manual_seed(42)
    np.random.seed(42)
    
    result1 = random_function()
    
    torch.manual_seed(42)
    np.random.seed(42)
    
    result2 = random_function()
    
    assert torch.equal(result1, result2)
```

### 3. Error Testing

Test error conditions and edge cases:

```python
def test_model_with_invalid_input():
    model = DynamicGraphDiffusionNet(config)
    
    # Test with negative node indices
    invalid_data = create_invalid_graph_data()
    
    with pytest.raises(ValueError, match="Node indices must be non-negative"):
        model(invalid_data)

def test_empty_graph():
    model = DynamicGraphDiffusionNet(config)
    empty_graph = create_empty_graph()
    
    # Should handle empty graphs gracefully
    output = model(empty_graph)
    assert output.numel() == 0
```

### 4. Memory Testing

Test for memory leaks in long-running operations:

```python
def test_training_memory_usage():
    initial_memory = get_memory_usage()
    
    model = DynamicGraphDiffusionNet(config)
    optimizer = torch.optim.Adam(model.parameters())
    
    # Train for many steps
    for _ in range(100):
        loss = train_step(model, optimizer, data)
        
        # Clear gradients
        optimizer.zero_grad()
        
        # Force garbage collection
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    final_memory = get_memory_usage()
    memory_increase = final_memory - initial_memory
    
    # Memory should not increase significantly
    assert memory_increase < 100  # MB
```

### 5. Performance Regression Testing

Set performance baselines and test for regressions:

```python
@pytest.mark.performance
def test_training_speed_regression():
    model = DynamicGraphDiffusionNet(config)
    data = create_benchmark_data()
    
    # Warmup
    for _ in range(5):
        _ = model(data)
    
    # Benchmark
    start_time = time.time()
    for _ in range(20):
        output = model(data)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    elapsed = time.time() - start_time
    avg_time_per_forward = elapsed / 20
    
    # Should be faster than baseline
    baseline_time = 0.1  # seconds
    assert avg_time_per_forward < baseline_time
```

## Debugging Failed Tests

### 1. Verbose Output

```bash
# Run with verbose output
pytest -v

# Show print statements
pytest -s

# Show local variables on failure
pytest --tb=long

# Drop into debugger on failure
pytest --pdb
```

### 2. Logging in Tests

```python
import logging

def test_with_logging():
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)
    
    logger.debug("Starting test")
    
    result = function_under_test()
    logger.debug(f"Result: {result}")
    
    assert result == expected
```

### 3. Test Data Inspection

```python
def test_with_data_inspection():
    data = create_test_data()
    
    # Save test data for inspection
    torch.save(data, 'debug_test_data.pt')
    
    result = function_under_test(data)
    
    # Save result for inspection
    torch.save(result, 'debug_test_result.pt')
    
    assert condition(result)
```

## Continuous Integration

### GitHub Actions Configuration

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, 3.10, 3.11]
    
    steps:
    - uses: actions/checkout@v3
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        pip install -e ".[dev,test]"
    
    - name: Run tests
      run: |
        pytest tests/ -m "not slow" --cov=src/dgdn --cov-report=xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
```

### Test Quality Metrics

Track these metrics in CI:

- **Coverage**: Aim for >90% line coverage
- **Test Count**: Monitor test count growth
- **Test Speed**: Track test execution time
- **Flaky Tests**: Identify and fix non-deterministic tests

## Conclusion

A comprehensive testing strategy ensures the reliability and maintainability of the DGDN library. By following these guidelines and best practices, we can catch bugs early, prevent regressions, and provide confidence in the library's correctness and performance.

Remember:
- Write tests as you develop features
- Keep tests simple and focused
- Use appropriate test categories
- Monitor test performance and coverage
- Regularly review and update tests

For questions or suggestions about testing practices, please open an issue or discuss in the development channels.