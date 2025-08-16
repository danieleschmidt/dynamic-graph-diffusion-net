#!/bin/bash
set -e

echo "🚀 DGDN Production Deployment Script"
echo "===================================="

# Check prerequisites
echo "📋 Checking prerequisites..."
command -v docker >/dev/null 2>&1 || { echo "❌ Docker is required but not installed. Aborting." >&2; exit 1; }
command -v docker-compose >/dev/null 2>&1 || { echo "❌ Docker Compose is required but not installed. Aborting." >&2; exit 1; }

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p logs data models ssl monitoring/grafana

# Build and deploy
echo "🏗️ Building DGDN production image..."
docker-compose -f docker-compose.production.yml build

echo "🚀 Starting production deployment..."
docker-compose -f docker-compose.production.yml up -d

# Wait for services to be ready
echo "⏳ Waiting for services to be ready..."
sleep 30

# Health check
echo "🔍 Performing health checks..."
if curl -f http://localhost/health > /dev/null 2>&1; then
    echo "✅ DGDN API is healthy"
else
    echo "❌ DGDN API health check failed"
    docker-compose -f docker-compose.production.yml logs dgdn-api
    exit 1
fi

if curl -f http://localhost:9090/-/healthy > /dev/null 2>&1; then
    echo "✅ Prometheus is healthy"
else
    echo "⚠️ Prometheus health check failed"
fi

if curl -f http://localhost:3000/api/health > /dev/null 2>&1; then
    echo "✅ Grafana is healthy"
else
    echo "⚠️ Grafana health check failed"
fi

echo ""
echo "🎉 DGDN Production Deployment Complete!"
echo "================================="
echo "📡 API:        http://localhost/api/"
echo "📊 Monitoring: http://localhost:3000 (admin/dgdn-admin-2025)"
echo "📈 Metrics:    http://localhost:9090"
echo "📚 Docs:       http://localhost/docs"
echo "❤️ Health:     http://localhost/health"
echo ""
echo "View logs: docker-compose -f docker-compose.production.yml logs -f"
echo "Stop:      docker-compose -f docker-compose.production.yml down"
