# Production Dockerfile for DGDN
FROM python:3.12-slim as builder

# Build arguments
ARG BUILD_ENV=production
ARG DGDN_VERSION=1.0.0

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r dgdn && useradd -r -g dgdn dgdn

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt requirements-dev.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY setup.py setup.cfg pyproject.toml ./

# Install DGDN package
RUN pip install -e .

# Production stage
FROM python:3.12-slim as production

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DGDN_ENV=production \
    DGDN_LOG_LEVEL=INFO

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r dgdn && useradd -r -g dgdn dgdn

# Set working directory
WORKDIR /app

# Copy from builder
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /app/src ./src

# Copy deployment files
COPY deployment/configs/ ./configs/
COPY deployment/scripts/ ./scripts/

# Create necessary directories
RUN mkdir -p /app/logs /app/data /app/models && \
    chown -R dgdn:dgdn /app

# Switch to non-root user
USER dgdn

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import dgdn; print('DGDN OK')" || exit 1

# Expose port
EXPOSE 8000

# Default command
CMD ["python", "-m", "dgdn.server", "--config", "configs/production.yaml"]