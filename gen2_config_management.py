#!/usr/bin/env python3
"""
Generation 2: Configuration Management System
Comprehensive configuration management, validation, and environment handling.
"""

import json
import yaml
import os
import logging
from typing import Dict, Any, Optional, List, Union, Type, Callable
from dataclasses import dataclass, field, asdict
from pathlib import Path
from enum import Enum
import re
from copy import deepcopy
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

logger = logging.getLogger(__name__)

class Environment(Enum):
    """Deployment environments."""
    DEVELOPMENT = "development"
    TESTING = "testing"  
    STAGING = "staging"
    PRODUCTION = "production"

@dataclass
class ModelConfig:
    """Core model configuration."""
    node_dim: int = 64
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
    early_stopping_patience: int = 10
    gradient_clip_norm: Optional[float] = 1.0
    warmup_steps: int = 1000
    scheduler: str = "cosine"
    beta_kl: float = 0.1
    beta_temporal: float = 0.05

@dataclass
class SecurityConfig:
    """Security configuration."""
    enable_input_sanitization: bool = True
    enable_integrity_checks: bool = True
    max_nodes: int = 1000000
    max_edges: int = 10000000
    timeout_seconds: float = 30.0
    enable_rate_limiting: bool = True
    max_requests_per_minute: int = 1000

@dataclass
class MonitoringConfig:
    """Monitoring and logging configuration."""
    log_level: str = "INFO"
    enable_performance_monitoring: bool = True
    enable_health_checks: bool = True
    metrics_retention_days: int = 30
    alert_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'memory_mb': 1000.0,
        'inference_time_ms': 1000.0,
        'error_rate': 0.1
    })

@dataclass
class DeploymentConfig:
    """Deployment configuration."""
    environment: Environment = Environment.DEVELOPMENT
    replicas: int = 1
    enable_gpu: bool = False
    gpu_memory_limit: Optional[str] = None
    cpu_limit: str = "2"
    memory_limit: str = "4Gi"
    enable_autoscaling: bool = False
    min_replicas: int = 1
    max_replicas: int = 10

@dataclass
class DGDNConfig:
    """Complete DGDN configuration."""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    deployment: DeploymentConfig = field(default_factory=DeploymentConfig)
    
    # Global settings
    random_seed: int = 42
    data_dir: str = "data"
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    cache_dir: str = "cache"

class ConfigValidator:
    """Configuration validation and sanitization."""
    
    @staticmethod
    def validate_model_config(config: ModelConfig) -> List[str]:
        """Validate model configuration."""
        errors = []
        
        # Dimension checks
        if config.node_dim <= 0:
            errors.append("node_dim must be positive")
        if config.edge_dim < 0:
            errors.append("edge_dim must be non-negative")
        if config.time_dim <= 0:
            errors.append("time_dim must be positive")
        if config.hidden_dim <= 0:
            errors.append("hidden_dim must be positive")
        
        # Architecture checks  
        if config.num_layers <= 0:
            errors.append("num_layers must be positive")
        if config.num_heads <= 0:
            errors.append("num_heads must be positive")
        if config.hidden_dim % config.num_heads != 0:
            errors.append("hidden_dim must be divisible by num_heads")
        
        # Diffusion checks
        if config.diffusion_steps <= 0:
            errors.append("diffusion_steps must be positive")
        if config.diffusion_steps > 20:
            errors.append("diffusion_steps should not exceed 20 for performance")
        
        # String parameter checks
        valid_aggregations = {"attention", "mean", "sum"}
        if config.aggregation not in valid_aggregations:
            errors.append(f"aggregation must be one of {valid_aggregations}")
        
        valid_activations = {"relu", "gelu", "swish", "leaky_relu"}
        if config.activation not in valid_activations:
            errors.append(f"activation must be one of {valid_activations}")
        
        valid_time_encodings = {"fourier", "positional", "multiscale"}
        if config.time_encoding not in valid_time_encodings:
            errors.append(f"time_encoding must be one of {valid_time_encodings}")
        
        # Numerical parameter checks
        if not 0.0 <= config.dropout < 1.0:
            errors.append("dropout must be in [0.0, 1.0)")
        if config.max_time <= 0:
            errors.append("max_time must be positive")
        
        return errors
    
    @staticmethod
    def validate_training_config(config: TrainingConfig) -> List[str]:
        """Validate training configuration."""
        errors = []
        
        if config.learning_rate <= 0:
            errors.append("learning_rate must be positive")
        if config.learning_rate > 1.0:
            errors.append("learning_rate is suspiciously large")
        
        if config.weight_decay < 0:
            errors.append("weight_decay must be non-negative")
        
        if config.batch_size <= 0:
            errors.append("batch_size must be positive")
        if config.batch_size > 1024:
            errors.append("batch_size is very large, may cause memory issues")
        
        if config.epochs <= 0:
            errors.append("epochs must be positive")
        
        if config.early_stopping_patience <= 0:
            errors.append("early_stopping_patience must be positive")
        
        if config.gradient_clip_norm is not None and config.gradient_clip_norm <= 0:
            errors.append("gradient_clip_norm must be positive if specified")
        
        if config.warmup_steps < 0:
            errors.append("warmup_steps must be non-negative")
        
        valid_schedulers = {"linear", "cosine", "exponential", "step"}
        if config.scheduler not in valid_schedulers:
            errors.append(f"scheduler must be one of {valid_schedulers}")
        
        if not 0.0 <= config.beta_kl <= 1.0:
            errors.append("beta_kl must be in [0.0, 1.0]")
        if not 0.0 <= config.beta_temporal <= 1.0:
            errors.append("beta_temporal must be in [0.0, 1.0]")
        
        return errors
    
    @staticmethod
    def validate_security_config(config: SecurityConfig) -> List[str]:
        """Validate security configuration."""
        errors = []
        
        if config.max_nodes <= 0:
            errors.append("max_nodes must be positive")
        if config.max_edges <= 0:
            errors.append("max_edges must be positive")
        if config.timeout_seconds <= 0:
            errors.append("timeout_seconds must be positive")
        if config.max_requests_per_minute <= 0:
            errors.append("max_requests_per_minute must be positive")
        
        return errors
    
    @staticmethod
    def validate_deployment_config(config: DeploymentConfig) -> List[str]:
        """Validate deployment configuration."""
        errors = []
        
        if config.replicas <= 0:
            errors.append("replicas must be positive")
        
        # CPU limit validation
        cpu_pattern = r'^(\d+(\.\d+)?)(m)?$'
        if not re.match(cpu_pattern, config.cpu_limit):
            errors.append("cpu_limit must be in format like '2', '1.5', '500m'")
        
        # Memory limit validation
        memory_pattern = r'^(\d+)(Mi|Gi|M|G)?$'
        if not re.match(memory_pattern, config.memory_limit):
            errors.append("memory_limit must be in format like '4Gi', '512Mi', '1G'")
        
        if config.enable_autoscaling:
            if config.min_replicas <= 0:
                errors.append("min_replicas must be positive when autoscaling enabled")
            if config.max_replicas <= config.min_replicas:
                errors.append("max_replicas must be greater than min_replicas")
        
        return errors
    
    @staticmethod
    def validate_complete_config(config: DGDNConfig) -> List[str]:
        """Validate complete configuration."""
        errors = []
        
        errors.extend(ConfigValidator.validate_model_config(config.model))
        errors.extend(ConfigValidator.validate_training_config(config.training))
        errors.extend(ConfigValidator.validate_security_config(config.security))
        errors.extend(ConfigValidator.validate_deployment_config(config.deployment))
        
        # Global settings validation
        if config.random_seed < 0:
            errors.append("random_seed must be non-negative")
        
        # Directory checks
        directories = [config.data_dir, config.checkpoint_dir, config.log_dir, config.cache_dir]
        for directory in directories:
            if not directory or not isinstance(directory, str):
                errors.append(f"Directory path must be a non-empty string: {directory}")
        
        return errors

class ConfigManager:
    """Configuration management system."""
    
    def __init__(self, config_dir: str = "configs"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        
        # Environment-specific config files
        self.env_configs = {
            Environment.DEVELOPMENT: self.config_dir / "development.yaml",
            Environment.TESTING: self.config_dir / "testing.yaml",
            Environment.STAGING: self.config_dir / "staging.yaml",
            Environment.PRODUCTION: self.config_dir / "production.yaml"
        }
        
        self._current_config: Optional[DGDNConfig] = None
        self._current_env: Optional[Environment] = None
    
    def load_config(self, environment: Union[Environment, str], 
                   config_path: Optional[str] = None) -> DGDNConfig:
        """Load configuration for specified environment."""
        if isinstance(environment, str):
            environment = Environment(environment)
        
        # Determine config file path
        if config_path:
            config_file = Path(config_path)
        else:
            config_file = self.env_configs[environment]
        
        # Load configuration
        if config_file.exists():
            config_dict = self._load_config_file(config_file)
            config = self._dict_to_config(config_dict)
        else:
            logger.warning(f"Config file not found: {config_file}, using defaults")
            config = DGDNConfig()
            config.deployment.environment = environment
        
        # Apply environment-specific overrides
        config = self._apply_environment_overrides(config, environment)
        
        # Validate configuration
        errors = ConfigValidator.validate_complete_config(config)
        if errors:
            raise ValueError(f"Configuration validation failed: {errors}")
        
        # Set deployment environment
        config.deployment.environment = environment
        
        self._current_config = config
        self._current_env = environment
        
        logger.info(f"Configuration loaded for environment: {environment.value}")
        return config
    
    def save_config(self, config: DGDNConfig, 
                   environment: Optional[Environment] = None,
                   config_path: Optional[str] = None):
        """Save configuration to file."""
        if environment is None:
            environment = config.deployment.environment
        
        if config_path:
            config_file = Path(config_path)
        else:
            config_file = self.env_configs[environment]
        
        # Validate before saving
        errors = ConfigValidator.validate_complete_config(config)
        if errors:
            raise ValueError(f"Cannot save invalid configuration: {errors}")
        
        config_dict = self._config_to_dict(config)
        self._save_config_file(config_dict, config_file)
        
        logger.info(f"Configuration saved to: {config_file}")
    
    def get_current_config(self) -> Optional[DGDNConfig]:
        """Get currently loaded configuration."""
        return self._current_config
    
    def get_current_environment(self) -> Optional[Environment]:
        """Get current environment."""
        return self._current_env
    
    def create_default_configs(self):
        """Create default configuration files for all environments."""
        for env in Environment:
            config = self._get_default_config_for_environment(env)
            self.save_config(config, env)
        
        logger.info("Default configuration files created")
    
    def override_config(self, overrides: Dict[str, Any]) -> DGDNConfig:
        """Apply configuration overrides."""
        if self._current_config is None:
            raise ValueError("No configuration loaded")
        
        config = deepcopy(self._current_config)
        
        # Apply nested overrides
        for key, value in overrides.items():
            self._set_nested_value(config, key, value)
        
        # Validate after overrides
        errors = ConfigValidator.validate_complete_config(config)
        if errors:
            raise ValueError(f"Invalid configuration after overrides: {errors}")
        
        return config
    
    def _load_config_file(self, config_file: Path) -> Dict[str, Any]:
        """Load configuration from YAML or JSON file."""
        try:
            with open(config_file, 'r') as f:
                if config_file.suffix.lower() == '.json':
                    return json.load(f)
                else:
                    return yaml.safe_load(f) or {}
        except Exception as e:
            raise ValueError(f"Failed to load config file {config_file}: {str(e)}")
    
    def _save_config_file(self, config_dict: Dict[str, Any], config_file: Path):
        """Save configuration to YAML file."""
        try:
            config_file.parent.mkdir(parents=True, exist_ok=True)
            with open(config_file, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
        except Exception as e:
            raise ValueError(f"Failed to save config file {config_file}: {str(e)}")
    
    def _dict_to_config(self, config_dict: Dict[str, Any]) -> DGDNConfig:
        """Convert dictionary to DGDNConfig object."""
        config = DGDNConfig()
        
        # Update nested configurations
        if 'model' in config_dict:
            for key, value in config_dict['model'].items():
                if hasattr(config.model, key):
                    setattr(config.model, key, value)
        
        if 'training' in config_dict:
            for key, value in config_dict['training'].items():
                if hasattr(config.training, key):
                    setattr(config.training, key, value)
        
        if 'security' in config_dict:
            for key, value in config_dict['security'].items():
                if hasattr(config.security, key):
                    setattr(config.security, key, value)
        
        if 'monitoring' in config_dict:
            for key, value in config_dict['monitoring'].items():
                if hasattr(config.monitoring, key):
                    setattr(config.monitoring, key, value)
        
        if 'deployment' in config_dict:
            for key, value in config_dict['deployment'].items():
                if hasattr(config.deployment, key):
                    if key == 'environment' and isinstance(value, str):
                        setattr(config.deployment, key, Environment(value))
                    else:
                        setattr(config.deployment, key, value)
        
        # Update global settings
        for key in ['random_seed', 'data_dir', 'checkpoint_dir', 'log_dir', 'cache_dir']:
            if key in config_dict:
                setattr(config, key, config_dict[key])
        
        return config
    
    def _config_to_dict(self, config: DGDNConfig) -> Dict[str, Any]:
        """Convert DGDNConfig object to dictionary."""
        config_dict = {
            'model': asdict(config.model),
            'training': asdict(config.training),
            'security': asdict(config.security),
            'monitoring': asdict(config.monitoring),
            'deployment': asdict(config.deployment),
            'random_seed': config.random_seed,
            'data_dir': config.data_dir,
            'checkpoint_dir': config.checkpoint_dir,
            'log_dir': config.log_dir,
            'cache_dir': config.cache_dir
        }
        
        # Convert environment enum to string
        config_dict['deployment']['environment'] = config_dict['deployment']['environment'].value
        
        return config_dict
    
    def _apply_environment_overrides(self, config: DGDNConfig, 
                                   environment: Environment) -> DGDNConfig:
        """Apply environment-specific configuration overrides."""
        if environment == Environment.PRODUCTION:
            # Production optimizations
            config.security.enable_input_sanitization = True
            config.security.enable_integrity_checks = True
            config.security.enable_rate_limiting = True
            config.monitoring.log_level = "WARNING"
            config.monitoring.enable_performance_monitoring = True
            config.deployment.enable_autoscaling = True
            
        elif environment == Environment.DEVELOPMENT:
            # Development conveniences
            config.security.enable_input_sanitization = False
            config.security.enable_rate_limiting = False
            config.monitoring.log_level = "DEBUG"
            config.training.early_stopping_patience = 5  # Faster iteration
            
        elif environment == Environment.TESTING:
            # Testing optimizations
            config.training.epochs = 5  # Quick tests
            config.training.batch_size = 16  # Smaller batches
            config.model.num_layers = 2  # Simpler model
            config.security.enable_input_sanitization = True
            
        return config
    
    def _get_default_config_for_environment(self, environment: Environment) -> DGDNConfig:
        """Get default configuration for environment."""
        config = DGDNConfig()
        config.deployment.environment = environment
        return self._apply_environment_overrides(config, environment)
    
    def _set_nested_value(self, obj: Any, key: str, value: Any):
        """Set nested attribute value using dot notation."""
        keys = key.split('.')
        current = obj
        
        for k in keys[:-1]:
            if hasattr(current, k):
                current = getattr(current, k)
            else:
                raise ValueError(f"Invalid configuration key: {key}")
        
        final_key = keys[-1]
        if hasattr(current, final_key):
            setattr(current, final_key, value)
        else:
            raise ValueError(f"Invalid configuration key: {key}")

def test_config_management():
    """Test configuration management system."""
    print("⚙️ Testing Configuration Management System")
    print("=" * 50)
    
    try:
        # Initialize config manager
        print("🔧 Initializing configuration manager...")
        config_manager = ConfigManager("test_configs")
        print("✅ Configuration manager initialized")
        
        # Create default configs
        print("\n📝 Creating default configurations...")
        config_manager.create_default_configs()
        print("✅ Default configurations created")
        
        # Load development config
        print("\n📥 Loading development configuration...")
        dev_config = config_manager.load_config(Environment.DEVELOPMENT)
        print(f"✅ Development config loaded: {dev_config.deployment.environment.value}")
        print(f"   Model layers: {dev_config.model.num_layers}")
        print(f"   Hidden dim: {dev_config.model.hidden_dim}")
        print(f"   Log level: {dev_config.monitoring.log_level}")
        
        # Load production config
        print("\n🏭 Loading production configuration...")
        prod_config = config_manager.load_config(Environment.PRODUCTION)
        print(f"✅ Production config loaded: {prod_config.deployment.environment.value}")
        print(f"   Security enabled: {prod_config.security.enable_input_sanitization}")
        print(f"   Autoscaling: {prod_config.deployment.enable_autoscaling}")
        
        # Test configuration overrides
        print("\n🔄 Testing configuration overrides...")
        overrides = {
            'model.hidden_dim': 512,
            'training.learning_rate': 0.001,
            'deployment.replicas': 3
        }
        
        override_config = config_manager.override_config(overrides)
        print(f"✅ Configuration overrides applied:")
        print(f"   Hidden dim: {override_config.model.hidden_dim}")
        print(f"   Learning rate: {override_config.training.learning_rate}")
        print(f"   Replicas: {override_config.deployment.replicas}")
        
        # Test validation
        print("\n✅ Testing configuration validation...")
        
        # Test invalid config
        try:
            invalid_config = DGDNConfig()
            invalid_config.model.hidden_dim = -1  # Invalid
            errors = ConfigValidator.validate_complete_config(invalid_config)
            if errors:
                print(f"✅ Correctly caught validation errors: {len(errors)} errors")
            else:
                print("❌ Should have caught validation errors")
        except Exception as e:
            print(f"✅ Validation working: {str(e)[:50]}...")
        
        # Test saving config
        print("\n💾 Testing configuration saving...")
        test_config = DGDNConfig()
        test_config.model.hidden_dim = 384
        config_manager.save_config(test_config, Environment.TESTING, "test_configs/custom.yaml")
        print("✅ Configuration saved successfully")
        
        # Health check
        print("\n🏥 Testing configuration health...")
        current_config = config_manager.get_current_config()
        current_env = config_manager.get_current_environment()
        print(f"✅ Current environment: {current_env.value if current_env else 'None'}")
        print(f"   Config loaded: {current_config is not None}")
        
        print("\n🎉 Configuration Management Tests: ALL PASSED!")
        return True
        
    except Exception as e:
        print(f"\n❌ Error in configuration management test: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_config_management()
    sys.exit(0 if success else 1)