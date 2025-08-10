# DGDN API Documentation

## Core Components

### Base Models
- [`DynamicGraphDiffusionNet`](core/dgdn.md) - Main DGDN implementation
- [`EdgeTimeEncoder`](core/temporal.md) - Temporal edge encoding
- [`VariationalDiffusion`](core/diffusion.md) - Uncertainty quantification

### Advanced Models
- [`FoundationDGDN`](advanced/foundation.md) - Self-supervised pretraining
- [`ContinuousDGDN`](advanced/continuous.md) - Neural ODE dynamics
- [`MultiScaleDGDN`](advanced/multiscale.md) - Multi-resolution modeling
- [`FederatedDGDN`](advanced/federated.md) - Privacy-preserving learning
- [`ExplainableDGDN`](advanced/explainable.md) - Interpretable predictions

### Research Extensions
- [`CausalDGDN`](research/causal.md) - Causal discovery and inference
- [`QuantumDGDN`](research/quantum.md) - Quantum-inspired computing
- [`NeuromorphicDGDN`](research/neuromorphic.md) - Spiking neural networks

### Enterprise Features
- [`SecurityManager`](enterprise/security.md) - Encryption and privacy
- [`AdvancedMonitoring`](enterprise/monitoring.md) - Metrics and health checks
- [`DistributedTrainer`](enterprise/distributed.md) - Multi-GPU training
- [`EdgeOptimizer`](enterprise/edge.md) - Mobile deployment

### Global Compliance
- [`GlobalComplianceFramework`](compliance/framework.md) - Multi-region compliance
- [`ComplianceValidator`](compliance/validator.md) - Deployment validation
- [`InternationalizationManager`](compliance/i18n.md) - Multi-language support

## Quick Reference

```python
# Core usage
from dgdn import DynamicGraphDiffusionNet
model = DynamicGraphDiffusionNet(node_dim=64, hidden_dim=256)

# Advanced models
from dgdn.models.advanced import FoundationDGDN
foundation = FoundationDGDN(node_dim=64, hidden_dim=256)

# Research features
from dgdn.research.causal import CausalDGDN
causal = CausalDGDN(node_dim=64, max_nodes=1000)

# Enterprise deployment
from dgdn.enterprise.security import EncryptedDGDN
secure = EncryptedDGDN(node_dim=64, security_config={'encryption': True})
```