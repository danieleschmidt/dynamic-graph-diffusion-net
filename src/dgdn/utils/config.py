"""Configuration management for DGDN."""

import os
import json
# import yaml  # Optional dependency - imported when needed
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, Union
from pathlib import Path


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    node_dim: int = 128
    edge_dim: int = 0
    time_dim: int = 32
    hidden_dim: int = 256
    num_layers: int = 3
    num_heads: int = 8
    diffusion_steps: int = 5
    aggregation: str = "attention"
    dropout: float = 0.1
    activation: str = "relu"
    layer_norm: bool = True
    graph_norm: bool = False
    time_encoding: str = "fourier"
    max_time: float = 1000.0


@dataclass
class TrainingConfig:
    """Training configuration."""
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 32
    epochs: int = 100
    patience: int = 10
    beta_kl: float = 0.1
    beta_temporal: float = 0.05
    gradient_clip_norm: float = 1.0
    warmup_steps: int = 1000
    scheduler: str = "cosine"
    mixed_precision: bool = True
    accumulate_grad_batches: int = 1


@dataclass
class SecurityConfig:
    """Security and privacy configuration."""
    enable_input_validation: bool = True
    enable_model_integrity_check: bool = True
    enable_differential_privacy: bool = False
    dp_noise_multiplier: float = 1.0
    dp_max_grad_norm: float = 1.0
    enable_secure_aggregation: bool = False
    max_input_size: int = 1000000
    allowed_file_types: list = None
    
    def __post_init__(self):
        if self.allowed_file_types is None:
            self.allowed_file_types = ['.pt', '.pth', '.json', '.yaml', '.yml']


@dataclass 
class MonitoringConfig:
    """Monitoring and logging configuration."""
    enable_tensorboard: bool = False
    enable_wandb: bool = False
    log_level: str = "INFO"
    log_interval: int = 100
    save_interval: int = 1000
    metrics_dir: str = "metrics"
    checkpoints_dir: str = "checkpoints"
    enable_profiling: bool = False
    profile_memory: bool = True
    profile_cpu: bool = True


@dataclass
class DGDNConfig:
    """Complete DGDN configuration."""
    model: ModelConfig
    training: TrainingConfig
    security: SecurityConfig
    monitoring: MonitoringConfig
    
    def __init__(
        self,
        model: Optional[ModelConfig] = None,
        training: Optional[TrainingConfig] = None,
        security: Optional[SecurityConfig] = None,
        monitoring: Optional[MonitoringConfig] = None
    ):
        self.model = model or ModelConfig()
        self.training = training or TrainingConfig()
        self.security = security or SecurityConfig()
        self.monitoring = monitoring or MonitoringConfig()
    
    @classmethod
    def from_file(cls, config_path: Union[str, Path]) -> 'DGDNConfig':
        """Load configuration from file."""
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                try:
                    # import yaml  # Optional dependency - imported when needed
                    config_dict = yaml.safe_load(f)
                except ImportError:
                    raise ImportError("PyYAML is required for YAML configuration files")
            elif config_path.suffix.lower() == '.json':
                config_dict = json.load(f)
            else:
                raise ValueError(f"Unsupported configuration file format: {config_path.suffix}")
        
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'DGDNConfig':
        """Create configuration from dictionary."""
        model_config = ModelConfig(**config_dict.get('model', {}))
        training_config = TrainingConfig(**config_dict.get('training', {}))
        security_config = SecurityConfig(**config_dict.get('security', {}))
        monitoring_config = MonitoringConfig(**config_dict.get('monitoring', {}))
        
        return cls(
            model=model_config,
            training=training_config,
            security=security_config,
            monitoring=monitoring_config
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'model': asdict(self.model),
            'training': asdict(self.training),
            'security': asdict(self.security),
            'monitoring': asdict(self.monitoring)
        }
    
    def save(self, config_path: Union[str, Path]) -> None:
        """Save configuration to file."""
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        config_dict = self.to_dict()
        
        with open(config_path, 'w') as f:
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                try:
                    # import yaml  # Optional dependency - imported when needed
                    yaml.dump(config_dict, f, default_flow_style=False, indent=2)
                except ImportError:
                    raise ImportError("PyYAML is required for YAML configuration files")
            elif config_path.suffix.lower() == '.json':
                json.dump(config_dict, f, indent=2)
            else:
                raise ValueError(f"Unsupported configuration file format: {config_path.suffix}")
    
    def validate(self) -> bool:
        """Validate configuration values."""
        errors = []
        
        # Model validation
        if self.model.node_dim <= 0:
            errors.append("node_dim must be positive")
        if self.model.hidden_dim <= 0:
            errors.append("hidden_dim must be positive")
        if self.model.hidden_dim % self.model.num_heads != 0:
            errors.append("hidden_dim must be divisible by num_heads")
        if not 0 <= self.model.dropout < 1:
            errors.append("dropout must be in [0, 1)")
        
        # Training validation
        if self.training.learning_rate <= 0:
            errors.append("learning_rate must be positive")
        if self.training.batch_size <= 0:
            errors.append("batch_size must be positive")
        if self.training.epochs <= 0:
            errors.append("epochs must be positive")
        
        # Security validation
        if self.security.max_input_size <= 0:
            errors.append("max_input_size must be positive")
        
        if errors:
            raise ValueError(f"Configuration validation failed: {'; '.join(errors)}")
        
        return True


def load_config_from_env() -> DGDNConfig:
    """Load configuration from environment variables."""
    config_dict = {
        'model': {},
        'training': {},
        'security': {},
        'monitoring': {}
    }
    
    # Model config from env
    if os.getenv('DGDN_NODE_DIM'):
        config_dict['model']['node_dim'] = int(os.getenv('DGDN_NODE_DIM'))
    if os.getenv('DGDN_HIDDEN_DIM'):
        config_dict['model']['hidden_dim'] = int(os.getenv('DGDN_HIDDEN_DIM'))
    if os.getenv('DGDN_NUM_LAYERS'):
        config_dict['model']['num_layers'] = int(os.getenv('DGDN_NUM_LAYERS'))
    
    # Training config from env
    if os.getenv('DGDN_LEARNING_RATE'):
        config_dict['training']['learning_rate'] = float(os.getenv('DGDN_LEARNING_RATE'))
    if os.getenv('DGDN_BATCH_SIZE'):
        config_dict['training']['batch_size'] = int(os.getenv('DGDN_BATCH_SIZE'))
    if os.getenv('DGDN_EPOCHS'):
        config_dict['training']['epochs'] = int(os.getenv('DGDN_EPOCHS'))
    
    # Security config from env
    if os.getenv('DGDN_ENABLE_VALIDATION'):
        config_dict['security']['enable_input_validation'] = os.getenv('DGDN_ENABLE_VALIDATION').lower() == 'true'
    
    # Monitoring config from env
    if os.getenv('DGDN_LOG_LEVEL'):
        config_dict['monitoring']['log_level'] = os.getenv('DGDN_LOG_LEVEL')
    
    return DGDNConfig.from_dict(config_dict)


def get_default_config() -> DGDNConfig:
    """Get default configuration."""
    return DGDNConfig()


def merge_configs(base_config: DGDNConfig, override_config: DGDNConfig) -> DGDNConfig:
    """Merge two configurations, with override taking precedence."""
    base_dict = base_config.to_dict()
    override_dict = override_config.to_dict()
    
    def deep_merge(base: dict, override: dict) -> dict:
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = deep_merge(result[key], value)
            else:
                result[key] = value
        return result
    
    merged_dict = deep_merge(base_dict, override_dict)
    return DGDNConfig.from_dict(merged_dict)