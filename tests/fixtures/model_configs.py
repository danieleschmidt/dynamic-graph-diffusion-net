"""Model configuration fixtures for testing."""

import pytest
from typing import Dict, Any


@pytest.fixture
def minimal_dgdn_config() -> Dict[str, Any]:
    """Minimal DGDN configuration for fast testing.
    
    Returns:
        Dict containing minimal model configuration
    """
    return {
        'node_dim': 16,
        'edge_dim': 8,
        'time_dim': 4,
        'hidden_dim': 32,
        'num_layers': 2,
        'diffusion_steps': 3,
        'attention_heads': 2,
        'dropout': 0.1,
        'activation': 'relu'
    }


@pytest.fixture
def standard_dgdn_config() -> Dict[str, Any]:
    """Standard DGDN configuration for regular testing.
    
    Returns:
        Dict containing standard model configuration
    """
    return {
        'node_dim': 64,
        'edge_dim': 32,
        'time_dim': 16,
        'hidden_dim': 128,
        'num_layers': 3,
        'diffusion_steps': 5,
        'attention_heads': 4,
        'dropout': 0.2,
        'activation': 'relu',
        'aggregation': 'attention'
    }


@pytest.fixture
def large_dgdn_config() -> Dict[str, Any]:
    """Large DGDN configuration for performance testing.
    
    Returns:
        Dict containing large model configuration
    """
    return {
        'node_dim': 128,
        'edge_dim': 64,
        'time_dim': 32,
        'hidden_dim': 256,
        'num_layers': 4,
        'diffusion_steps': 10,
        'attention_heads': 8,
        'dropout': 0.1,
        'activation': 'gelu',
        'aggregation': 'attention',
        'use_layer_norm': True,
        'residual_connections': True
    }


@pytest.fixture
def training_config() -> Dict[str, Any]:
    """Training configuration for testing.
    
    Returns:
        Dict containing training parameters
    """
    return {
        'learning_rate': 0.001,
        'weight_decay': 1e-5,
        'batch_size': 32,
        'max_epochs': 10,
        'patience': 3,
        'min_delta': 1e-4,
        'gradient_clip_norm': 1.0,
        'scheduler': 'cosine',
        'warmup_steps': 100,
        'loss_weights': {
            'reconstruction': 1.0,
            'variational': 0.1,
            'temporal': 0.05,
            'diffusion': 0.1
        }
    }


@pytest.fixture
def diffusion_config() -> Dict[str, Any]:
    """Diffusion-specific configuration for testing.
    
    Returns:
        Dict containing diffusion parameters
    """
    return {
        'diffusion_steps': 5,
        'noise_schedule': 'cosine',
        'beta_start': 0.0001,
        'beta_end': 0.02,
        'variance_type': 'learned',
        'loss_type': 'mse',
        'parameterization': 'eps',
        'clip_denoised': True,
        'use_timestep_embedding': True
    }


@pytest.fixture
def attention_config() -> Dict[str, Any]:
    """Attention mechanism configuration for testing.
    
    Returns:
        Dict containing attention parameters
    """
    return {
        'attention_heads': 4,
        'attention_dim': 64,
        'attention_dropout': 0.1,
        'use_bias': True,
        'temperature': 1.0,
        'use_relative_position': False,
        'max_relative_position': 32
    }


@pytest.fixture
def temporal_encoding_config() -> Dict[str, Any]:
    """Temporal encoding configuration for testing.
    
    Returns:
        Dict containing temporal encoding parameters
    """
    return {
        'time_dim': 16,
        'num_bases': 32,
        'encoding_type': 'fourier',
        'learnable_bases': True,
        'max_timescale': 10000.0,
        'use_positional_encoding': True
    }


@pytest.fixture 
def evaluation_config() -> Dict[str, Any]:
    """Evaluation configuration for testing.
    
    Returns:
        Dict containing evaluation parameters
    """
    return {
        'metrics': ['auc', 'ap', 'mar', 'hits@1', 'hits@3', 'hits@10'],
        'k_values': [1, 3, 10, 50],
        'negative_sampling_ratio': 1.0,
        'evaluation_frequency': 5,
        'save_predictions': False,
        'compute_embeddings': True
    }


@pytest.fixture
def optimizer_configs() -> Dict[str, Dict[str, Any]]:
    """Different optimizer configurations for testing.
    
    Returns:
        Dict mapping optimizer names to their configurations
    """
    return {
        'adam': {
            'type': 'adam',
            'learning_rate': 0.001,
            'betas': [0.9, 0.999],
            'eps': 1e-8,
            'weight_decay': 1e-5,
            'amsgrad': False
        },
        'adamw': {
            'type': 'adamw',
            'learning_rate': 0.001,
            'betas': [0.9, 0.999],
            'eps': 1e-8,
            'weight_decay': 0.01,
            'amsgrad': False
        },
        'sgd': {
            'type': 'sgd',
            'learning_rate': 0.01,
            'momentum': 0.9,
            'weight_decay': 1e-4,
            'nesterov': True
        },
        'rmsprop': {
            'type': 'rmsprop',
            'learning_rate': 0.001,
            'alpha': 0.99,
            'eps': 1e-8,
            'weight_decay': 0,
            'momentum': 0,
            'centered': False
        }
    }


@pytest.fixture
def scheduler_configs() -> Dict[str, Dict[str, Any]]:
    """Different scheduler configurations for testing.
    
    Returns:
        Dict mapping scheduler names to their configurations
    """
    return {
        'cosine': {
            'type': 'cosine_annealing',
            'T_max': 100,
            'eta_min': 1e-6,
            'last_epoch': -1
        },
        'step': {
            'type': 'step_lr',
            'step_size': 30,
            'gamma': 0.1,
            'last_epoch': -1
        },
        'exponential': {
            'type': 'exponential_lr',
            'gamma': 0.95,
            'last_epoch': -1
        },
        'plateau': {
            'type': 'reduce_lr_on_plateau',
            'mode': 'min',
            'factor': 0.5,
            'patience': 10,
            'verbose': False,
            'threshold': 1e-4
        },
        'warmup_cosine': {
            'type': 'warmup_cosine',
            'warmup_steps': 100,
            'T_max': 1000,
            'eta_min': 1e-6
        }
    }


@pytest.fixture
def device_configs() -> Dict[str, str]:
    """Different device configurations for testing.
    
    Returns:
        Dict mapping config names to device strings
    """
    import torch
    
    configs = {'cpu': 'cpu'}
    
    if torch.cuda.is_available():
        configs['cuda'] = 'cuda'
        if torch.cuda.device_count() > 1:
            configs['cuda_multi'] = 'cuda'
    
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        configs['mps'] = 'mps'
    
    return configs