# Contributing to Dynamic Graph Diffusion Net

We welcome contributions! This document provides guidelines for contributing to the project.

## Quick Start

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Install development dependencies: `pip install -e ".[dev]"`
4. Install pre-commit hooks: `pre-commit install`
5. Make your changes and commit them: `git commit -m 'Add amazing feature'`
6. Push to the branch: `git push origin feature/amazing-feature`
7. Open a Pull Request

## Development Setup

### Environment Setup

```bash
# Clone your fork
git clone https://github.com/yourusername/dynamic-graph-diffusion-net.git
cd dynamic-graph-diffusion-net

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### Code Quality

We use several tools to maintain code quality:

- **Black**: Code formatting
- **Ruff**: Linting and import sorting
- **MyPy**: Type checking
- **Pre-commit**: Automated checks before commits

Run quality checks:
```bash
# Format code
black .

# Lint code
ruff check .

# Type check
mypy src/

# Run all pre-commit hooks
pre-commit run --all-files
```

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=dgdn --cov-report=html

# Run specific test file
pytest tests/test_models.py

# Run tests in parallel
pytest -n auto
```

### Writing Tests

- Place tests in the `tests/` directory
- Use descriptive test names: `test_should_handle_empty_graph`
- Follow the Arrange-Act-Assert pattern
- Include both unit and integration tests
- Aim for high code coverage (>90%)

Example test structure:
```python
import pytest
import torch
from dgdn import DynamicGraphDiffusionNet

class TestDGDN:
    def test_should_initialize_with_valid_params(self):
        # Arrange
        node_dim, edge_dim = 64, 32
        
        # Act
        model = DynamicGraphDiffusionNet(node_dim=node_dim, edge_dim=edge_dim)
        
        # Assert
        assert model.node_dim == node_dim
        assert model.edge_dim == edge_dim
```

## Contribution Types

### Bug Reports

Use the bug report template and include:
- Clear description of the issue
- Steps to reproduce
- Expected vs actual behavior
- Environment details (Python version, dependencies)
- Minimal code example

### Feature Requests

Use the feature request template and include:
- Clear description of the proposed feature
- Use case and motivation
- API design suggestions
- Implementation considerations

### Code Contributions

#### Priority Areas

- **Core Algorithms**: Edge-time encoding variations, diffusion mechanisms
- **Performance**: Memory optimization, GPU acceleration
- **Documentation**: API docs, tutorials, examples
- **Testing**: Edge cases, performance benchmarks
- **Visualization**: Interactive tools, plotting utilities

#### Implementation Guidelines

1. **API Design**: Follow existing patterns, maintain backward compatibility
2. **Documentation**: Include docstrings with examples
3. **Type Hints**: Use type annotations for all public APIs
4. **Error Handling**: Provide clear error messages with suggestions
5. **Performance**: Consider memory usage and computational efficiency

#### Code Style

- Follow PEP 8 (enforced by Black and Ruff)
- Use descriptive variable names
- Keep functions focused and small
- Add type hints for parameters and return values
- Write comprehensive docstrings

Example function:
```python
def compute_edge_embeddings(
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    timestamps: torch.Tensor,
    time_encoder: nn.Module,
) -> torch.Tensor:
    """Compute temporal edge embeddings.
    
    Args:
        edge_index: Graph connectivity [2, num_edges]
        edge_attr: Edge features [num_edges, edge_dim]
        timestamps: Edge timestamps [num_edges]
        time_encoder: Temporal encoding module
        
    Returns:
        Enhanced edge embeddings [num_edges, hidden_dim]
        
    Example:
        >>> edge_embeddings = compute_edge_embeddings(
        ...     edge_index, edge_attr, timestamps, time_encoder
        ... )
    """
    # Implementation here
    pass
```

## Documentation

### API Documentation

- Use Google-style docstrings
- Include parameter types and descriptions
- Provide usage examples
- Document return values and exceptions

### Tutorials and Examples

- Create Jupyter notebooks for complex workflows
- Include real-world use cases
- Explain both theory and implementation
- Test all code examples

## Review Process

### Pull Request Guidelines

1. **Description**: Clear summary of changes and motivation
2. **Testing**: All tests pass, new tests for new features
3. **Documentation**: Updated docs for API changes
4. **Performance**: No significant performance regressions
5. **Breaking Changes**: Clearly documented with migration guide

### Review Criteria

- Code quality and style compliance
- Test coverage and quality
- Documentation completeness
- Performance implications
- Backward compatibility
- Security considerations

## Community Guidelines

### Code of Conduct

We follow the [Contributor Covenant](https://www.contributor-covenant.org/) code of conduct. Be respectful, inclusive, and constructive in all interactions.

### Communication

- **Issues**: For bug reports and feature requests
- **Discussions**: For questions and general discussion
- **Discord**: Real-time chat (link in README)
- **Email**: Maintainer contact for sensitive issues

### Getting Help

- Check existing issues and documentation
- Search discussions for similar questions
- Join our Discord community
- Tag maintainers for urgent issues

## Release Process

Releases follow semantic versioning (SemVer):
- **Major**: Breaking changes
- **Minor**: New features, backward compatible
- **Patch**: Bug fixes, backward compatible

## Legal

By contributing, you agree that your contributions will be licensed under the MIT License.

## Acknowledgments

Contributors are recognized in:
- GitHub contributors list
- Release notes for significant contributions
- Annual acknowledgments in documentation

Thank you for contributing to Dynamic Graph Diffusion Net! 🚀