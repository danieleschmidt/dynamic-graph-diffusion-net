# DGDN Monitoring and Observability

This directory contains monitoring and observability configurations for the Dynamic Graph Diffusion Net project.

## Overview

The monitoring stack includes:

- **Prometheus**: Metrics collection and storage
- **Grafana**: Metrics visualization and dashboards
- **AlertManager**: Alert routing and management
- **Node Exporter**: System metrics
- **cAdvisor**: Container metrics
- **NVIDIA GPU Exporter**: GPU metrics (if available)
- **Jaeger**: Distributed tracing

## Quick Start

### 1. Start Monitoring Stack

```bash
# Start all monitoring services
docker-compose -f monitoring/docker-compose.monitoring.yml up -d

# Check services are running
docker-compose -f monitoring/docker-compose.monitoring.yml ps
```

### 2. Access Dashboards

- **Grafana**: http://localhost:3000 (admin/admin)
- **Prometheus**: http://localhost:9090
- **AlertManager**: http://localhost:9093
- **Jaeger**: http://localhost:16686

### 3. Import DGDN Dashboard

1. Open Grafana at http://localhost:3000
2. Login with admin/admin
3. Navigate to Dashboards > Import
4. Upload `grafana/dgdn-dashboard.json`
5. Configure Prometheus data source: http://prometheus:9090

## Metrics Overview

### Training Metrics

- `dgdn_training_loss`: Current training loss
- `dgdn_validation_loss`: Validation loss
- `dgdn_training_accuracy`: Training accuracy
- `dgdn_samples_processed_total`: Total samples processed
- `dgdn_epoch_duration_seconds`: Time per epoch
- `dgdn_batch_processing_time`: Time per batch

### Model Metrics

- `dgdn_model_parameters_total`: Total model parameters
- `dgdn_model_memory_usage_bytes`: Model memory usage
- `dgdn_inference_time_seconds`: Inference time per sample
- `dgdn_graph_nodes`: Number of nodes in current graph
- `dgdn_graph_edges`: Number of edges in current graph

### System Metrics

- CPU usage and load
- Memory usage and availability
- GPU utilization and memory
- Disk I/O and space
- Network statistics

## Custom Metrics

### Adding New Metrics

1. **In your Python code**:

```python
from prometheus_client import Counter, Histogram, Gauge

# Define metrics
training_loss = Gauge('dgdn_training_loss', 'Current training loss')
samples_processed = Counter('dgdn_samples_processed_total', 'Total samples processed')
batch_time = Histogram('dgdn_batch_processing_seconds', 'Time spent processing batches')

# Use metrics
training_loss.set(current_loss)
samples_processed.inc(batch_size)
with batch_time.time():
    # Your training code here
    pass
```

2. **Expose metrics endpoint**:

```python
from prometheus_client import start_http_server

# Start metrics server on port 8000
start_http_server(8000)
```

### Metric Guidelines

- Use appropriate metric types:
  - **Counter**: Monotonically increasing values (samples processed, errors)
  - **Gauge**: Values that can go up/down (memory usage, queue size)
  - **Histogram**: Measure distributions (response times, batch sizes)
  - **Summary**: Similar to histogram but with quantiles

- Follow naming conventions:
  - Use `dgdn_` prefix for application metrics
  - Use descriptive names with units (e.g., `_seconds`, `_bytes`)
  - Include help text for all metrics

## Alerting

### Alert Rules

Create alert rules in `prometheus/rules/dgdn-alerts.yml`:

```yaml
groups:
  - name: dgdn.rules
    rules:
      - alert: DGDNHighTrainingLoss
        expr: dgdn_training_loss > 10.0
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "DGDN training loss is high"
          description: "Training loss has been above 10.0 for more than 5 minutes"

      - alert: DGDNGPUMemoryHigh
        expr: nvidia_gpu_memory_used_bytes / nvidia_gpu_memory_total_bytes > 0.9
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "GPU memory usage is critical"
          description: "GPU memory usage is above 90%"
```

### Alert Configuration

Configure AlertManager in `alertmanager/alertmanager.yml`:

```yaml
global:
  smtp_smarthost: 'localhost:587'
  smtp_from: 'alerts@yourcompany.com'

route:
  group_by: ['alertname']
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 1h
  receiver: 'web.hook'

receivers:
  - name: 'web.hook'
    email_configs:
      - to: 'admin@yourcompany.com'
        subject: 'DGDN Alert: {{ .GroupLabels.alertname }}'
        body: |
          {{ range .Alerts }}
          Alert: {{ .Annotations.summary }}
          Description: {{ .Annotations.description }}
          {{ end }}
```

## Distributed Tracing

### Enable Tracing in Code

```python
from opentelemetry import trace
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

# Configure tracing
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)

jaeger_exporter = JaegerExporter(
    agent_host_name="localhost",
    agent_port=6831,
)

span_processor = BatchSpanProcessor(jaeger_exporter)
trace.get_tracer_provider().add_span_processor(span_processor)

# Use tracing
with tracer.start_as_current_span("training_step"):
    # Your training code
    with tracer.start_as_current_span("forward_pass"):
        # Forward pass code
        pass
    with tracer.start_as_current_span("backward_pass"):
        # Backward pass code
        pass
```

## Performance Monitoring

### Key Performance Indicators (KPIs)

Monitor these critical metrics:

1. **Training Performance**:
   - Samples per second
   - Time per epoch
   - Loss convergence rate
   - Memory efficiency

2. **Model Performance**:
   - Inference latency
   - Throughput (predictions/sec)
   - Model accuracy
   - Resource utilization

3. **System Health**:
   - CPU/GPU utilization
   - Memory usage trends
   - Disk I/O patterns
   - Network bandwidth

### Setting Up Alerts

Create alerts for:
- Training stagnation (loss not improving)
- Resource exhaustion (high memory/GPU usage)
- System errors and failures
- Performance degradation

## Troubleshooting

### Common Issues

1. **Metrics not appearing**:
   - Check if metrics endpoint is accessible
   - Verify Prometheus scrape configuration
   - Check firewall settings

2. **Grafana dashboard empty**:
   - Verify Prometheus data source configuration
   - Check metric names match dashboard queries
   - Ensure time range is appropriate

3. **Alerts not firing**:
   - Verify alert rule syntax
   - Check AlertManager configuration
   - Ensure notification channels are configured

### Debug Commands

```bash
# Check Prometheus targets
curl http://localhost:9090/api/v1/targets

# Test metric endpoint
curl http://localhost:8000/metrics

# Check AlertManager status
curl http://localhost:9093/api/v1/status

# View container logs
docker-compose -f monitoring/docker-compose.monitoring.yml logs -f prometheus
```

## Production Considerations

### Security

- Change default passwords
- Use TLS for external connections
- Implement authentication/authorization
- Secure metric endpoints

### Scalability

- Consider Prometheus federation for large deployments
- Use remote storage for long-term retention
- Implement metric aggregation for high-cardinality data
- Set up proper retention policies

### High Availability

- Deploy multiple Prometheus instances
- Use AlertManager clustering
- Implement backup strategies
- Monitor the monitoring system itself

## Integration with CI/CD

Add monitoring checks to your CI/CD pipeline:

```yaml
# Example GitHub Action step
- name: Check metrics endpoint
  run: |
    curl -f http://localhost:8000/metrics || exit 1

- name: Validate Prometheus config
  run: |
    docker run --rm -v $(pwd)/monitoring/prometheus:/prometheus prom/prometheus:latest \
      promtool check config /prometheus/prometheus.yml
```