#!/usr/bin/env python3
"""
Generation 2: Security Enhancements for DGDN

Advanced security features including input sanitization, access control,
audit logging, and protection against adversarial attacks.
"""

import torch
import hashlib
import hmac
import time
import json
import uuid
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import sys

# Add src to path for imports
sys.path.insert(0, 'src')

from dgdn import TemporalData


@dataclass
class SecurityConfig:
    """Security configuration for DGDN operations."""
    enable_input_sanitization: bool = True
    enable_access_control: bool = True
    enable_audit_logging: bool = True
    enable_rate_limiting: bool = True
    max_requests_per_minute: int = 100
    enable_data_encryption: bool = False
    secret_key: str = "dgdn_default_key_change_in_production"
    session_timeout_minutes: int = 60
    enable_adversarial_detection: bool = True


class AccessControl:
    """Role-based access control system."""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.sessions = {}
        self.permissions = {
            "admin": ["read", "write", "execute", "admin"],
            "researcher": ["read", "execute"],
            "viewer": ["read"]
        }
    
    def create_session(self, user_id: str, role: str) -> str:
        """Create authenticated session."""
        session_id = str(uuid.uuid4())
        expiry = datetime.now() + timedelta(minutes=self.config.session_timeout_minutes)
        
        self.sessions[session_id] = {
            "user_id": user_id,
            "role": role,
            "created_at": datetime.now(),
            "expires_at": expiry,
            "permissions": self.permissions.get(role, [])
        }
        
        return session_id
    
    def validate_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Validate session and return session info."""
        if session_id not in self.sessions:
            return None
        
        session = self.sessions[session_id]
        if datetime.now() > session["expires_at"]:
            del self.sessions[session_id]
            return None
        
        return session
    
    def check_permission(self, session_id: str, required_permission: str) -> bool:
        """Check if session has required permission."""
        session = self.validate_session(session_id)
        if not session:
            return False
        
        return required_permission in session["permissions"]
    
    def revoke_session(self, session_id: str):
        """Revoke session."""
        if session_id in self.sessions:
            del self.sessions[session_id]


class RateLimiter:
    """Rate limiting for API protection."""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.request_history = {}
    
    def is_allowed(self, identifier: str) -> bool:
        """Check if request is allowed under rate limits."""
        now = time.time()
        minute_ago = now - 60
        
        # Clean old entries
        if identifier in self.request_history:
            self.request_history[identifier] = [
                timestamp for timestamp in self.request_history[identifier]
                if timestamp > minute_ago
            ]
        else:
            self.request_history[identifier] = []
        
        # Check limit
        if len(self.request_history[identifier]) >= self.config.max_requests_per_minute:
            return False
        
        # Record request
        self.request_history[identifier].append(now)
        return True


class AuditLogger:
    """Security audit logging system."""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.audit_log = []
    
    def log_access(self, session_id: str, action: str, resource: str, success: bool, details: str = ""):
        """Log access attempt."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "session_id": session_id[:8] + "..." if session_id else "anonymous",
            "action": action,
            "resource": resource,
            "success": success,
            "details": details,
            "event_id": str(uuid.uuid4())
        }
        
        self.audit_log.append(entry)
        
        # Print to console for immediate visibility
        status = "✅" if success else "❌"
        print(f"🔐 AUDIT {status} {action} on {resource} - {details}")
    
    def get_audit_trail(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent audit entries."""
        return self.audit_log[-limit:]
    
    def export_audit_log(self, filepath: str):
        """Export audit log to file."""
        with open(filepath, 'w') as f:
            json.dump(self.audit_log, f, indent=2)


class InputSanitizer:
    """Input sanitization and validation."""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
    
    def sanitize_temporal_data(self, data: TemporalData) -> Tuple[TemporalData, List[str]]:
        """Sanitize temporal data and return warnings."""
        warnings = []
        
        # Clone data for safe modification
        sanitized_data = TemporalData(
            edge_index=data.edge_index.clone(),
            timestamps=data.timestamps.clone(),
            edge_attr=data.edge_attr.clone() if data.edge_attr is not None else None,
            node_features=data.node_features.clone() if data.node_features is not None else None,
            y=data.y.clone() if data.y is not None else None,
            num_nodes=data.num_nodes
        )
        
        # Sanitize edge indices
        if torch.any(sanitized_data.edge_index < 0):
            warnings.append("Negative edge indices detected and clamped")
            sanitized_data.edge_index = torch.clamp(sanitized_data.edge_index, min=0)
        
        # Sanitize timestamps
        if torch.any(torch.isnan(sanitized_data.timestamps)):
            warnings.append("NaN timestamps detected and replaced")
            nan_mask = torch.isnan(sanitized_data.timestamps)
            sanitized_data.timestamps[nan_mask] = 0.0
        
        if torch.any(torch.isinf(sanitized_data.timestamps)):
            warnings.append("Infinite timestamps detected and replaced")
            inf_mask = torch.isinf(sanitized_data.timestamps)
            sanitized_data.timestamps[inf_mask] = 1000.0
        
        # Sanitize node features
        if sanitized_data.node_features is not None:
            if torch.any(torch.isnan(sanitized_data.node_features)):
                warnings.append("NaN node features detected and replaced")
                sanitized_data.node_features = torch.nan_to_num(sanitized_data.node_features)
            
            # Clamp extreme values
            if torch.any(torch.abs(sanitized_data.node_features) > 1000):
                warnings.append("Extreme node feature values detected and clamped")
                sanitized_data.node_features = torch.clamp(sanitized_data.node_features, -1000, 1000)
        
        # Sanitize edge attributes
        if sanitized_data.edge_attr is not None:
            if torch.any(torch.isnan(sanitized_data.edge_attr)):
                warnings.append("NaN edge attributes detected and replaced")
                sanitized_data.edge_attr = torch.nan_to_num(sanitized_data.edge_attr)
            
            if torch.any(torch.abs(sanitized_data.edge_attr) > 1000):
                warnings.append("Extreme edge attribute values detected and clamped")
                sanitized_data.edge_attr = torch.clamp(sanitized_data.edge_attr, -1000, 1000)
        
        return sanitized_data, warnings


class AdversarialDetector:
    """Detection of adversarial attacks on graph data."""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.baseline_stats = None
    
    def compute_graph_statistics(self, data: TemporalData) -> Dict[str, float]:
        """Compute statistical features of graph data."""
        stats = {}
        
        # Basic graph statistics
        stats["num_nodes"] = float(data.num_nodes)
        stats["num_edges"] = float(data.edge_index.shape[1])
        stats["density"] = stats["num_edges"] / (stats["num_nodes"] * (stats["num_nodes"] - 1))
        
        # Degree statistics
        degrees = torch.bincount(data.edge_index[0], minlength=data.num_nodes).float()
        stats["avg_degree"] = torch.mean(degrees).item()
        stats["max_degree"] = torch.max(degrees).item()
        stats["degree_std"] = torch.std(degrees).item()
        
        # Temporal statistics
        stats["time_span"] = (data.timestamps.max() - data.timestamps.min()).item()
        stats["temporal_variance"] = torch.var(data.timestamps).item()
        
        # Feature statistics if available
        if data.node_features is not None:
            stats["node_feature_mean"] = torch.mean(data.node_features).item()
            stats["node_feature_std"] = torch.std(data.node_features).item()
            stats["node_feature_max"] = torch.max(torch.abs(data.node_features)).item()
        
        if data.edge_attr is not None:
            stats["edge_attr_mean"] = torch.mean(data.edge_attr).item()
            stats["edge_attr_std"] = torch.std(data.edge_attr).item()
            stats["edge_attr_max"] = torch.max(torch.abs(data.edge_attr)).item()
        
        return stats
    
    def set_baseline(self, data: TemporalData):
        """Set baseline statistics for anomaly detection."""
        self.baseline_stats = self.compute_graph_statistics(data)
    
    def detect_anomalies(self, data: TemporalData) -> Tuple[bool, List[str]]:
        """Detect potential adversarial modifications."""
        if self.baseline_stats is None:
            return False, ["No baseline set for anomaly detection"]
        
        current_stats = self.compute_graph_statistics(data)
        anomalies = []
        
        # Define thresholds for detection
        thresholds = {
            "density": 2.0,  # 2x change in density
            "avg_degree": 2.0,
            "max_degree": 5.0,
            "node_feature_max": 10.0,
            "edge_attr_max": 10.0
        }
        
        for key, threshold in thresholds.items():
            if key in self.baseline_stats and key in current_stats:
                baseline_val = self.baseline_stats[key]
                current_val = current_stats[key]
                
                if baseline_val > 0:  # Avoid division by zero
                    ratio = current_val / baseline_val
                    if ratio > threshold or ratio < (1.0 / threshold):
                        anomalies.append(f"{key} changed by {ratio:.2f}x (baseline: {baseline_val:.4f}, current: {current_val:.4f})")
        
        is_anomalous = len(anomalies) > 0
        return is_anomalous, anomalies


class SecureDGDNFramework:
    """Secure framework wrapping DGDN operations."""
    
    def __init__(self, config: SecurityConfig = None):
        self.config = config or SecurityConfig()
        self.access_control = AccessControl(self.config)
        self.rate_limiter = RateLimiter(self.config)
        self.audit_logger = AuditLogger(self.config)
        self.input_sanitizer = InputSanitizer(self.config)
        self.adversarial_detector = AdversarialDetector(self.config)
        
        print("🛡️ Secure DGDN Framework initialized")
        print(f"   Input sanitization: {self.config.enable_input_sanitization}")
        print(f"   Access control: {self.config.enable_access_control}")
        print(f"   Rate limiting: {self.config.enable_rate_limiting}")
        print(f"   Audit logging: {self.config.enable_audit_logging}")
        print(f"   Adversarial detection: {self.config.enable_adversarial_detection}")
    
    def authenticate_user(self, user_id: str, role: str) -> str:
        """Authenticate user and create session."""
        session_id = self.access_control.create_session(user_id, role)
        
        self.audit_logger.log_access(
            session_id=session_id,
            action="authenticate",
            resource="system",
            success=True,
            details=f"User {user_id} authenticated with role {role}"
        )
        
        return session_id
    
    def secure_process_data(
        self,
        data: TemporalData,
        session_id: str,
        client_identifier: str = "default"
    ) -> Optional[TemporalData]:
        """Securely process temporal data with all security checks."""
        
        # Rate limiting check
        if self.config.enable_rate_limiting:
            if not self.rate_limiter.is_allowed(client_identifier):
                self.audit_logger.log_access(
                    session_id=session_id,
                    action="process_data",
                    resource="temporal_data",
                    success=False,
                    details="Rate limit exceeded"
                )
                print("🚨 Rate limit exceeded")
                return None
        
        # Access control check
        if self.config.enable_access_control:
            if not self.access_control.check_permission(session_id, "execute"):
                self.audit_logger.log_access(
                    session_id=session_id,
                    action="process_data",
                    resource="temporal_data",
                    success=False,
                    details="Insufficient permissions"
                )
                print("🚨 Access denied - insufficient permissions")
                return None
        
        # Input sanitization
        processed_data = data
        sanitization_warnings = []
        
        if self.config.enable_input_sanitization:
            processed_data, sanitization_warnings = self.input_sanitizer.sanitize_temporal_data(data)
            
            if sanitization_warnings:
                warning_msg = f"Sanitization warnings: {', '.join(sanitization_warnings)}"
                self.audit_logger.log_access(
                    session_id=session_id,
                    action="sanitize_input",
                    resource="temporal_data",
                    success=True,
                    details=warning_msg
                )
                print(f"⚠️ {warning_msg}")
        
        # Adversarial detection
        if self.config.enable_adversarial_detection:
            is_anomalous, anomalies = self.adversarial_detector.detect_anomalies(processed_data)
            
            if is_anomalous:
                anomaly_msg = f"Potential adversarial data detected: {', '.join(anomalies)}"
                self.audit_logger.log_access(
                    session_id=session_id,
                    action="adversarial_detection",
                    resource="temporal_data",
                    success=False,
                    details=anomaly_msg
                )
                print(f"🚨 {anomaly_msg}")
                return None
        
        # Success audit
        self.audit_logger.log_access(
            session_id=session_id,
            action="process_data",
            resource="temporal_data",
            success=True,
            details=f"Data processed successfully (nodes: {data.num_nodes}, edges: {data.edge_index.shape[1]})"
        )
        
        return processed_data
    
    def get_security_report(self) -> Dict[str, Any]:
        """Generate comprehensive security report."""
        return {
            "timestamp": datetime.now().isoformat(),
            "active_sessions": len(self.access_control.sessions),
            "audit_entries": len(self.audit_logger.audit_log),
            "rate_limiter_active": len(self.rate_limiter.request_history),
            "config": {
                "input_sanitization": self.config.enable_input_sanitization,
                "access_control": self.config.enable_access_control,
                "rate_limiting": self.config.enable_rate_limiting,
                "audit_logging": self.config.enable_audit_logging,
                "adversarial_detection": self.config.enable_adversarial_detection
            },
            "recent_audit_trail": self.audit_logger.get_audit_trail(limit=10)
        }


def demo_security_features():
    """Demonstrate security features."""
    print("🛡️ DGDN Security Enhancements Demo")
    print("=" * 50)
    
    # Initialize secure framework
    config = SecurityConfig()
    secure_framework = SecureDGDNFramework(config)
    
    # Create test users
    admin_session = secure_framework.authenticate_user("admin_user", "admin")
    researcher_session = secure_framework.authenticate_user("researcher_user", "researcher")
    viewer_session = secure_framework.authenticate_user("viewer_user", "viewer")
    
    print(f"\n👥 Created sessions:")
    print(f"   Admin: {admin_session[:8]}...")
    print(f"   Researcher: {researcher_session[:8]}...")
    print(f"   Viewer: {viewer_session[:8]}...")
    
    # Create test data
    print("\n🧪 Creating test data...")
    
    # Normal data
    normal_data = TemporalData(
        edge_index=torch.randint(0, 100, (2, 500)),
        timestamps=torch.sort(torch.rand(500) * 100)[0],
        node_features=torch.randn(100, 64),
        edge_attr=torch.randn(500, 32),
        num_nodes=100
    )
    
    # Set baseline for adversarial detection
    secure_framework.adversarial_detector.set_baseline(normal_data)
    
    # Malicious data with extreme values
    malicious_data = TemporalData(
        edge_index=torch.randint(0, 100, (2, 500)),
        timestamps=torch.cat([torch.rand(400) * 100, torch.tensor([float('inf')] * 100)]),  # Contains inf
        node_features=torch.randn(100, 64) * 1000,  # Extreme values
        edge_attr=torch.cat([torch.randn(400, 32), torch.full((100, 32), float('nan'))]),  # Contains NaN
        num_nodes=100
    )
    
    # Test scenarios
    test_cases = [
        ("Normal data with admin session", normal_data, admin_session),
        ("Normal data with researcher session", normal_data, researcher_session),
        ("Normal data with viewer session", normal_data, viewer_session),
        ("Malicious data with admin session", malicious_data, admin_session),
        ("Normal data with invalid session", normal_data, "invalid_session")
    ]
    
    print(f"\n🔍 Running security tests...")
    
    for test_name, data, session in test_cases:
        print(f"\n📋 Test: {test_name}")
        
        result = secure_framework.secure_process_data(
            data=data,
            session_id=session,
            client_identifier="test_client"
        )
        
        if result is not None:
            print(f"✅ Test passed - Data processed successfully")
        else:
            print(f"❌ Test failed - Data processing blocked")
    
    # Test rate limiting
    print(f"\n🚦 Testing rate limiting...")
    
    for i in range(105):  # Exceed rate limit
        result = secure_framework.secure_process_data(
            data=normal_data,
            session_id=admin_session,
            client_identifier="rate_limit_test"
        )
        
        if result is None:
            print(f"🚨 Rate limit triggered after {i} requests")
            break
    
    # Generate security report
    print(f"\n📊 Security Report:")
    report = secure_framework.get_security_report()
    print(f"   Active sessions: {report['active_sessions']}")
    print(f"   Audit entries: {report['audit_entries']}")
    print(f"   Rate limiter tracked clients: {report['rate_limiter_active']}")
    
    # Export audit log
    secure_framework.audit_logger.export_audit_log("security_audit.json")
    print(f"📁 Audit log exported to security_audit.json")
    
    print(f"\n🎉 Security demonstration completed!")


if __name__ == "__main__":
    demo_security_features()