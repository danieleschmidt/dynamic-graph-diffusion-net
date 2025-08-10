# Production Deployment Guide

## Overview

This guide covers deploying DGDN in production environments with high availability, scalability, and security.

## Quick Deployment

### Docker Deployment

```bash
# Clone repository
git clone https://github.com/your-username/dgdn.git
cd dgdn

# Build and deploy with Docker Compose
cd deployment/docker
docker-compose up -d
```

### Kubernetes Deployment

```bash
# Deploy to Kubernetes cluster
cd deployment/kubernetes
kubectl apply -f dgdn-deployment.yaml
kubectl apply -f dgdn-service.yaml

# Or use automated deployment script
chmod +x ../scripts/deploy.sh
./deployment/scripts/deploy.sh deploy
```

## Infrastructure Components

### Core Services

| Service | Purpose | Replicas | Resources |
|---------|---------|----------|-----------|
| dgdn-api | REST API server | 3 | 2 CPU, 4GB RAM |
| dgdn-worker | ML inference worker | 2 | 4 CPU, 8GB RAM, 1 GPU |
| postgres | Database | 1 | 2 CPU, 4GB RAM, 100GB SSD |
| redis | Cache & message queue | 2 | 1 CPU, 2GB RAM |

### Monitoring Stack

| Service | Purpose | Port |
|---------|---------|------|
| Prometheus | Metrics collection | 9090 |
| Grafana | Visualization | 3000 |
| Jaeger | Distributed tracing | 16686 |

## Configuration

### Environment Variables

```bash
# Core configuration
DGDN_ENV=production
DGDN_LOG_LEVEL=INFO
DGDN_SECRET_KEY=<your-secret-key>

# Database
DGDN_POSTGRES_URL=postgresql://user:pass@host:5432/dgdn
DGDN_REDIS_URL=redis://host:6379

# Security
DGDN_ENCRYPTION_KEY=<encryption-key>
DGDN_JWT_SECRET=<jwt-secret>

# Monitoring
DGDN_METRICS_ENABLED=true
DGDN_TRACING_ENABLED=true
```

### Production Configuration File

```yaml
# /app/configs/production.yaml
server:
  host: "0.0.0.0"
  port: 8000
  workers: 4
  timeout: 300

models:
  default_model: "foundation_dgdn"
  model_cache_size: 10
  
  foundation_dgdn:
    type: "FoundationDGDN"
    node_dim: 64
    hidden_dim: 256
    num_layers: 4

security:
  secret_key: "${DGDN_SECRET_KEY}"
  encryption:
    algorithm: "AES-256-GCM"
  rate_limiting:
    requests_per_minute: 100

monitoring:
  metrics:
    enabled: true
    endpoint: "/metrics"
  health_checks:
    enabled: true
    endpoints:
      liveness: "/health/live"
      readiness: "/health/ready"
```

## Scaling & Performance

### Horizontal Scaling

```bash
# Scale API servers
kubectl scale deployment dgdn-api --replicas=5

# Scale workers based on load
kubectl scale deployment dgdn-worker --replicas=4

# Auto-scaling with HPA
kubectl autoscale deployment dgdn-api --cpu-percent=70 --min=3 --max=10
```

### Performance Optimization

```yaml
# Deployment optimization
resources:
  requests:
    memory: "2Gi"
    cpu: "1000m"
  limits:
    memory: "4Gi"
    cpu: "2000m"

# GPU workers
resources:
  requests:
    nvidia.com/gpu: "1"
    memory: "4Gi"
    cpu: "2000m"
```

## Security

### Network Security

```yaml
# Network policies
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: dgdn-network-policy
spec:
  podSelector:
    matchLabels:
      app: dgdn
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: ingress-nginx
```

### TLS Configuration

```yaml
# Ingress with TLS
spec:
  tls:
  - hosts:
    - dgdn-api.yourdomain.com
    secretName: dgdn-tls-secret
  rules:
  - host: dgdn-api.yourdomain.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: dgdn-api-service
            port:
              number: 80
```

## Monitoring

### Health Checks

```python
# Health check endpoints
@app.get("/health/live")
def liveness_check():
    return {"status": "alive", "timestamp": time.time()}

@app.get("/health/ready")
def readiness_check():
    # Check database connectivity
    # Check model loading status
    # Check external dependencies
    return {"status": "ready", "checks": {...}}
```

### Metrics Collection

```python
# Prometheus metrics
from prometheus_client import Counter, Histogram, Gauge

REQUEST_COUNT = Counter('dgdn_requests_total', 'Total requests', ['method', 'endpoint'])
REQUEST_LATENCY = Histogram('dgdn_request_duration_seconds', 'Request latency')
MODEL_PREDICTIONS = Counter('dgdn_predictions_total', 'Total predictions')
```

### Grafana Dashboards

Key dashboard panels:
- Request rate and latency
- Model inference metrics
- Resource utilization (CPU, memory, GPU)
- Error rates and alerts
- Database performance
- Cache hit rates

## Backup & Recovery

### Database Backups

```bash
# Automated PostgreSQL backups
kubectl create configmap backup-script --from-file=backup.sh

# CronJob for regular backups
apiVersion: batch/v1
kind: CronJob
metadata:
  name: postgres-backup
spec:
  schedule: "0 2 * * *"  # Daily at 2 AM
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: backup
            image: postgres:15-alpine
            command: ["/bin/sh", "/scripts/backup.sh"]
```

### Model Backups

```bash
# S3 model backup
aws s3 sync /app/models s3://dgdn-model-backup/$(date +%Y-%m-%d)/
```

## Troubleshooting

### Common Issues

**High Memory Usage**
```bash
# Check memory usage
kubectl top pods -n dgdn

# Scale down model cache
kubectl set env deployment/dgdn-api MODEL_CACHE_SIZE=5
```

**Database Connection Issues**
```bash
# Check database connectivity
kubectl exec -it dgdn-api-pod -- psql $DGDN_POSTGRES_URL

# Restart database connections
kubectl rollout restart deployment/dgdn-api
```

**Model Loading Failures**
```bash
# Check model files
kubectl exec -it dgdn-worker-pod -- ls -la /app/models/

# Check logs
kubectl logs -f deployment/dgdn-worker
```

### Performance Monitoring

```bash
# Real-time metrics
kubectl port-forward svc/prometheus 9090:9090
# Open http://localhost:9090

# Grafana dashboards
kubectl port-forward svc/grafana 3000:3000
# Open http://localhost:3000 (admin/admin)
```

## Deployment Checklist

- [ ] Infrastructure provisioned (AWS/GCP/Azure)
- [ ] Kubernetes cluster ready
- [ ] Database initialized and configured
- [ ] Redis cluster deployed
- [ ] SSL certificates configured
- [ ] Monitoring stack deployed
- [ ] Backup procedures tested
- [ ] Load testing completed
- [ ] Security scan passed
- [ ] Documentation updated

## Support

For production deployment support:
- Create GitHub issue with deployment logs
- Contact: dgdn-devops@yourorganization.com
- Slack: #dgdn-production