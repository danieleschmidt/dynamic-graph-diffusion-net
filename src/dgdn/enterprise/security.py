"""Advanced security features for enterprise DGDN deployment."""

import torch
import torch.nn as nn
import hashlib
import hmac
import secrets
import time
from typing import Dict, List, Optional, Any, Tuple
import json
import logging
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64

from ..models.dgdn import DynamicGraphDiffusionNet


class SecurityManager:
    """Comprehensive security manager for DGDN systems."""
    
    def __init__(self, security_config: Dict[str, Any] = None):
        self.config = security_config or {}
        self.logger = logging.getLogger('DGDN.Security')
        
        # Security settings
        self.encryption_enabled = self.config.get('encryption', True)
        self.audit_enabled = self.config.get('auditing', True) 
        self.access_control_enabled = self.config.get('access_control', True)
        
        # Initialize components
        self.encryption_key = self._generate_encryption_key()
        self.cipher_suite = Fernet(self.encryption_key)
        self.audit_logger = AuditLogger() if self.audit_enabled else None
        self.access_controller = AccessController() if self.access_control_enabled else None
        
    def _generate_encryption_key(self):
        """Generate encryption key from password/config."""
        password = self.config.get('password', 'default_dgdn_key').encode()
        salt = self.config.get('salt', b'dgdn_salt_2025')
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        
        key = base64.urlsafe_b64encode(kdf.derive(password))
        return key
        
    def encrypt_model_weights(self, model: nn.Module) -> Dict[str, bytes]:
        """Encrypt model weights for secure storage."""
        encrypted_weights = {}
        
        for name, param in model.named_parameters():
            # Serialize parameter
            param_bytes = param.detach().cpu().numpy().tobytes()
            
            # Encrypt
            encrypted_param = self.cipher_suite.encrypt(param_bytes)
            encrypted_weights[name] = encrypted_param
            
        self.logger.info(f"Encrypted {len(encrypted_weights)} model parameters")
        return encrypted_weights
        
    def decrypt_model_weights(self, encrypted_weights: Dict[str, bytes], model: nn.Module):
        """Decrypt and load model weights."""
        decrypted_state = {}
        
        for name, encrypted_param in encrypted_weights.items():
            # Decrypt
            param_bytes = self.cipher_suite.decrypt(encrypted_param)
            
            # Deserialize
            original_shape = model.state_dict()[name].shape
            param_array = torch.frombuffer(param_bytes, dtype=torch.float32).reshape(original_shape)
            decrypted_state[name] = param_array
            
        model.load_state_dict(decrypted_state)
        self.logger.info(f"Decrypted and loaded {len(decrypted_state)} model parameters")
        
    def secure_inference(self, model: nn.Module, data: Any, user_id: str = None):
        """Perform secure inference with access control and auditing."""
        # Access control check
        if self.access_controller and not self.access_controller.check_access(user_id, 'inference'):
            raise PermissionError(f"User {user_id} not authorized for inference")
            
        # Audit log
        if self.audit_logger:
            self.audit_logger.log_inference_request(user_id, data)
            
        start_time = time.time()
        
        try:
            # Secure inference
            with torch.no_grad():
                output = model(data)
                
            # Audit successful inference
            if self.audit_logger:
                self.audit_logger.log_inference_success(user_id, time.time() - start_time)
                
            return output
            
        except Exception as e:
            # Audit failed inference
            if self.audit_logger:
                self.audit_logger.log_inference_failure(user_id, str(e))
            raise
            
    def compute_model_hash(self, model: nn.Module) -> str:
        """Compute cryptographic hash of model for integrity checking."""
        # Concatenate all parameters
        param_concat = b''
        for param in model.parameters():
            param_concat += param.detach().cpu().numpy().tobytes()
            
        # Compute SHA-256 hash
        model_hash = hashlib.sha256(param_concat).hexdigest()
        return model_hash
        
    def verify_model_integrity(self, model: nn.Module, expected_hash: str) -> bool:
        """Verify model hasn't been tampered with."""
        current_hash = self.compute_model_hash(model)
        is_valid = hmac.compare_digest(current_hash, expected_hash)
        
        if self.audit_logger:
            self.audit_logger.log_integrity_check(current_hash, expected_hash, is_valid)
            
        return is_valid
        
    def apply_differential_privacy(self, gradients: List[torch.Tensor], 
                                  epsilon: float = 1.0, delta: float = 1e-5):
        """Apply differential privacy to gradients."""
        sensitivity = 1.0  # L2 sensitivity
        noise_scale = sensitivity * (2 * np.log(1.25 / delta)) ** 0.5 / epsilon
        
        noised_gradients = []
        for grad in gradients:
            # Add Gaussian noise
            noise = torch.normal(0, noise_scale, size=grad.shape)
            noised_grad = grad + noise
            noised_gradients.append(noised_grad)
            
        self.logger.info(f"Applied DP noise with ε={epsilon}, δ={delta}")
        return noised_gradients


class EncryptedDGDN(DynamicGraphDiffusionNet):
    """DGDN with built-in encryption capabilities."""
    
    def __init__(self, *args, security_config: Dict = None, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.security_manager = SecurityManager(security_config)
        self.model_hash = None
        
    def save_encrypted(self, filepath: str, password: str = None):
        """Save model with encryption."""
        # Update security config with password if provided
        if password:
            self.security_manager.config['password'] = password
            self.security_manager.encryption_key = self.security_manager._generate_encryption_key()
            self.security_manager.cipher_suite = Fernet(self.security_manager.encryption_key)
            
        # Encrypt weights
        encrypted_weights = self.security_manager.encrypt_model_weights(self)
        
        # Compute integrity hash
        self.model_hash = self.security_manager.compute_model_hash(self)
        
        # Save encrypted data
        save_data = {
            'encrypted_weights': encrypted_weights,
            'model_hash': self.model_hash,
            'architecture_config': {
                'node_dim': self.node_dim,
                'edge_dim': self.edge_dim,
                'hidden_dim': self.hidden_dim,
                'num_layers': self.num_layers,
                # Add other necessary config
            }
        }
        
        # Serialize and encrypt the entire save data
        serialized_data = json.dumps(save_data, default=str).encode()
        encrypted_save_data = self.security_manager.cipher_suite.encrypt(serialized_data)
        
        with open(filepath, 'wb') as f:
            f.write(encrypted_save_data)
            
    def load_encrypted(self, filepath: str, password: str = None, verify_integrity: bool = True):
        """Load encrypted model."""
        # Update password if provided
        if password:
            self.security_manager.config['password'] = password
            self.security_manager.encryption_key = self.security_manager._generate_encryption_key()
            self.security_manager.cipher_suite = Fernet(self.security_manager.encryption_key)
            
        # Load and decrypt save data
        with open(filepath, 'rb') as f:
            encrypted_save_data = f.read()
            
        decrypted_data = self.security_manager.cipher_suite.decrypt(encrypted_save_data)
        save_data = json.loads(decrypted_data.decode())
        
        # Load encrypted weights
        encrypted_weights = {k: v.encode() for k, v in save_data['encrypted_weights'].items()}
        self.security_manager.decrypt_model_weights(encrypted_weights, self)
        
        # Verify integrity if requested
        if verify_integrity:
            expected_hash = save_data['model_hash']
            if not self.security_manager.verify_model_integrity(self, expected_hash):
                raise RuntimeError("Model integrity check failed - possible tampering detected")
                
        self.model_hash = save_data['model_hash']
        
    def secure_forward(self, data, user_id: str = None):
        """Secure forward pass with auditing."""
        return self.security_manager.secure_inference(self, data, user_id)


class AuditLogger:
    """Comprehensive audit logging for DGDN operations."""
    
    def __init__(self, log_file: str = "dgdn_audit.log"):
        self.logger = logging.getLogger('DGDN.Audit')
        
        # File handler for audit logs
        handler = logging.FileHandler(log_file)
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
        
    def log_inference_request(self, user_id: str, data_info: Any):
        """Log inference request."""
        self.logger.info(f"INFERENCE_REQUEST - User: {user_id}, Data size: {self._get_data_size(data_info)}")
        
    def log_inference_success(self, user_id: str, duration: float):
        """Log successful inference."""
        self.logger.info(f"INFERENCE_SUCCESS - User: {user_id}, Duration: {duration:.3f}s")
        
    def log_inference_failure(self, user_id: str, error: str):
        """Log failed inference."""
        self.logger.error(f"INFERENCE_FAILURE - User: {user_id}, Error: {error}")
        
    def log_model_training(self, user_id: str, config: Dict):
        """Log model training session."""
        self.logger.info(f"TRAINING_START - User: {user_id}, Config: {json.dumps(config)}")
        
    def log_model_update(self, user_id: str, model_hash: str):
        """Log model parameter updates."""
        self.logger.info(f"MODEL_UPDATE - User: {user_id}, New hash: {model_hash}")
        
    def log_integrity_check(self, current_hash: str, expected_hash: str, is_valid: bool):
        """Log integrity verification."""
        status = "PASS" if is_valid else "FAIL"
        self.logger.info(f"INTEGRITY_CHECK - Status: {status}, Current: {current_hash[:16]}..., Expected: {expected_hash[:16]}...")
        
    def log_access_attempt(self, user_id: str, operation: str, granted: bool):
        """Log access control attempts."""
        status = "GRANTED" if granted else "DENIED"
        self.logger.info(f"ACCESS_ATTEMPT - User: {user_id}, Operation: {operation}, Status: {status}")
        
    def log_security_event(self, event_type: str, details: Dict):
        """Log security-related events."""
        self.logger.warning(f"SECURITY_EVENT - Type: {event_type}, Details: {json.dumps(details)}")
        
    def _get_data_size(self, data):
        """Extract data size information safely."""
        if hasattr(data, 'x') and hasattr(data.x, 'shape'):
            return f"Nodes: {data.x.shape[0]}, Features: {data.x.shape[1]}"
        elif hasattr(data, 'edge_index') and hasattr(data.edge_index, 'shape'):
            return f"Edges: {data.edge_index.shape[1]}"
        else:
            return "Unknown"


class AccessController:
    """Role-based access control for DGDN operations."""
    
    def __init__(self):
        # Default roles and permissions
        self.roles = {
            'admin': ['inference', 'training', 'model_management', 'user_management'],
            'data_scientist': ['inference', 'training', 'model_management'],
            'analyst': ['inference'],
            'viewer': ['inference']  # Read-only access
        }
        
        # User-role mapping (in production, this would be external)
        self.user_roles = {}
        
        # Session tokens
        self.active_sessions = {}
        
    def add_user(self, user_id: str, role: str):
        """Add user with specified role."""
        if role in self.roles:
            self.user_roles[user_id] = role
        else:
            raise ValueError(f"Unknown role: {role}")
            
    def check_access(self, user_id: str, operation: str) -> bool:
        """Check if user has permission for operation."""
        if user_id not in self.user_roles:
            return False
            
        user_role = self.user_roles[user_id]
        return operation in self.roles.get(user_role, [])
        
    def create_session(self, user_id: str, duration: int = 3600) -> str:
        """Create authenticated session."""
        session_token = secrets.token_urlsafe(32)
        expiry_time = time.time() + duration
        
        self.active_sessions[session_token] = {
            'user_id': user_id,
            'expires': expiry_time
        }
        
        return session_token
        
    def validate_session(self, session_token: str) -> Optional[str]:
        """Validate session and return user_id if valid."""
        if session_token not in self.active_sessions:
            return None
            
        session = self.active_sessions[session_token]
        if time.time() > session['expires']:
            del self.active_sessions[session_token]
            return None
            
        return session['user_id']
        
    def revoke_session(self, session_token: str):
        """Revoke session token."""
        if session_token in self.active_sessions:
            del self.active_sessions[session_token]


class ThreatDetector:
    """Detect potential security threats in DGDN usage."""
    
    def __init__(self):
        self.request_history = {}  # Track request patterns
        self.anomaly_threshold = 0.95
        
    def detect_adversarial_inputs(self, data) -> Tuple[bool, float]:
        """Detect potential adversarial examples."""
        # Simple statistical anomaly detection
        if hasattr(data, 'x'):
            node_features = data.x
            
            # Check for extreme values
            mean_vals = torch.mean(node_features, dim=0)
            std_vals = torch.std(node_features, dim=0)
            
            # Z-score based anomaly detection
            z_scores = torch.abs((node_features - mean_vals) / (std_vals + 1e-8))
            max_z_score = torch.max(z_scores).item()
            
            is_anomalous = max_z_score > 3.0  # 3-sigma rule
            confidence = min(max_z_score / 10.0, 1.0)  # Normalize confidence
            
            return is_anomalous, confidence
            
        return False, 0.0
        
    def detect_inference_attacks(self, user_id: str) -> bool:
        """Detect potential model inference attacks."""
        current_time = time.time()
        
        # Track request frequency
        if user_id not in self.request_history:
            self.request_history[user_id] = []
            
        self.request_history[user_id].append(current_time)
        
        # Remove old requests (last 5 minutes)
        cutoff_time = current_time - 300
        self.request_history[user_id] = [
            t for t in self.request_history[user_id] if t > cutoff_time
        ]
        
        # Check for excessive requests (potential membership inference attack)
        request_count = len(self.request_history[user_id])
        if request_count > 100:  # More than 100 requests in 5 minutes
            return True
            
        return False
        
    def detect_model_extraction(self, queries: List[Any]) -> bool:
        """Detect potential model extraction attempts."""
        if len(queries) < 10:
            return False
            
        # Check for systematic querying patterns
        # This is a simplified heuristic
        
        # Look for structured/grid-like queries
        if self._has_systematic_pattern(queries):
            return True
            
        return False
        
    def _has_systematic_pattern(self, queries: List[Any]) -> bool:
        """Check if queries follow systematic pattern."""
        # Simplified pattern detection
        # In practice, this would be more sophisticated
        return len(queries) > 50  # Simple threshold