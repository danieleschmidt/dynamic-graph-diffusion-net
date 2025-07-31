# Multi-stage Docker build for Dynamic Graph Diffusion Net
# Optimized for both development and production use

# Base image with CUDA support (optional)
ARG CUDA_VERSION=11.8
ARG UBUNTU_VERSION=20.04
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION} as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.9 \
    python3.9-dev \
    python3-pip \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create symlink for python
RUN ln -sf /usr/bin/python3.9 /usr/bin/python

# Set working directory
WORKDIR /app

# Copy requirements first for better layer caching
COPY pyproject.toml README.md ./

# Install Python dependencies
RUN pip install --upgrade pip setuptools wheel

# Development stage
FROM base as development

# Install development dependencies
RUN pip install -e ".[dev]"

# Copy source code
COPY . .

# Install package in development mode
RUN pip install -e .

# Set up pre-commit hooks
RUN pre-commit install || true

# Default command for development
CMD ["python", "-c", "import dgdn; print(f'DGDN {dgdn.__version__} ready for development')"]

# Production stage
FROM base as production

# Install only production dependencies
RUN pip install torch torch-geometric --index-url https://download.pytorch.org/whl/cu118

# Copy and install package
COPY . .
RUN pip install .

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash dgdn
USER dgdn

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import dgdn; print('OK')" || exit 1

# Default command for production
CMD ["python", "-c", "import dgdn; print(f'DGDN {dgdn.__version__} ready')"]

# CPU-only version
FROM python:3.9-slim as cpu-only

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy and install
COPY . .
RUN pip install --upgrade pip && \
    pip install torch torch-geometric --index-url https://download.pytorch.org/whl/cpu && \
    pip install .

# Create non-root user
RUN useradd --create-home --shell /bin/bash dgdn
USER dgdn

CMD ["python", "-c", "import dgdn; print(f'DGDN {dgdn.__version__} (CPU) ready')"]