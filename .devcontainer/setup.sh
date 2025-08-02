#!/bin/bash
set -e

echo "🚀 Setting up Dynamic Graph Diffusion Net development environment..."

# Upgrade pip and install build tools
python -m pip install --upgrade pip setuptools wheel

# Install the package in development mode with all extras
echo "📦 Installing DGDN in development mode..."
pip install -e ".[dev,test,docs]"

# Install additional development tools
echo "🔧 Installing additional development tools..."
pip install jupyterlab tensorboard pre-commit

# Setup pre-commit hooks
echo "🪝 Setting up pre-commit hooks..."
pre-commit install

# Create necessary directories
echo "📁 Creating necessary directories..."
mkdir -p data logs models results .cache/torch

# Set up git configuration for container
echo "🔧 Configuring Git..."
git config --global --add safe.directory /workspace
git config --global init.defaultBranch main

# Install PyTorch with CUDA support if available
echo "🔥 Installing PyTorch with CUDA support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Verify installation
echo "✅ Verifying installation..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Test import of main package
echo "🧪 Testing package import..."
python -c "import sys; sys.path.append('src'); import dgdn; print('✅ DGDN import successful')"

echo "🎉 Development environment setup complete!"
echo ""
echo "🚀 Quick start commands:"
echo "  - Run tests: pytest"
echo "  - Start Jupyter: jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root"
echo "  - Format code: black src/ tests/"
echo "  - Lint code: ruff check src/ tests/"
echo "  - Type check: mypy src/"
echo ""