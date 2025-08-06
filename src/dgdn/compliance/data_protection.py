"""Data protection manager for multi-region compliance."""

import logging
import hashlib
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Set, Union
from enum import Enum
import torch


class DataProtectionLevel(Enum):
    """Data protection levels."""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"


class EncryptionMethod(Enum):
    """Supported encryption methods."""
    AES_256 = "aes_256"
    RSA_2048 = "rsa_2048"
    HASH_SHA256 = "sha_256"
    DIFFERENTIAL_PRIVACY = "differential_privacy"


class DataProtectionManager:
    """Multi-region data protection manager."""
    
    def __init__(self, region: str = "global", compliance_regimes: List[str] = None):
        """Initialize data protection manager."""
        self.region = region
        self.compliance_regimes = compliance_regimes or ["gdpr", "ccpa", "pdpa"]
        self.logger = logging.getLogger(__name__)
        
        # Protection policies by region and data type
        self.protection_policies = self._initialize_protection_policies()
        
        # Encryption keys (in production, use proper key management)
        self.encryption_keys = self._initialize_encryption()
        
        # Audit trail
        self.audit_log = []
        
        self.logger.info(f"Data protection manager initialized for region: {region}")
    
    def _initialize_protection_policies(self) -> Dict[str, Dict[str, Any]]:
        """Initialize protection policies for different regions."""
        return {
            "eu": {  # GDPR requirements
                "min_protection_level": DataProtectionLevel.CONFIDENTIAL,
                "required_encryption": [EncryptionMethod.AES_256],
                "pseudonymization_required": True,
                "retention_limits": {"default": 1095},  # 3 years
                "cross_border_restrictions": True,
                "consent_requirements": "explicit",
                "data_subject_rights": ["access", "rectification", "erasure", "portability"]
            },
            "us": {  # CCPA requirements
                "min_protection_level": DataProtectionLevel.INTERNAL,
                "required_encryption": [EncryptionMethod.AES_256],
                "pseudonymization_required": False,
                "retention_limits": {"default": 1095},
                "cross_border_restrictions": False,
                "consent_requirements": "opt_out",
                "data_subject_rights": ["access", "deletion", "opt_out", "correct"]
            },
            "sg": {  # PDPA requirements
                "min_protection_level": DataProtectionLevel.CONFIDENTIAL,
                "required_encryption": [EncryptionMethod.AES_256],
                "pseudonymization_required": True,
                "retention_limits": {"default": 1095},
                "cross_border_restrictions": True,
                "consent_requirements": "explicit",
                "data_subject_rights": ["access", "correction", "withdraw_consent"]
            },
            "global": {  # Most restrictive combination
                "min_protection_level": DataProtectionLevel.CONFIDENTIAL,
                "required_encryption": [EncryptionMethod.AES_256, EncryptionMethod.HASH_SHA256],
                "pseudonymization_required": True,
                "retention_limits": {"default": 1095},
                "cross_border_restrictions": True,
                "consent_requirements": "explicit",
                "data_subject_rights": ["access", "rectification", "erasure", "portability", "opt_out"]
            }
        }
    
    def _initialize_encryption(self) -> Dict[str, Any]:
        """Initialize encryption system."""
        # In production, use proper key management service
        return {
            "aes_key": "mock_aes_key_256_bit",
            "rsa_key": "mock_rsa_key_2048_bit",
            "initialized": datetime.utcnow().isoformat()
        }
    
    def classify_data_protection_level(self, data: Any, context: Dict[str, Any] = None) -> DataProtectionLevel:
        """Classify data protection level based on content and context."""
        context = context or {}
        
        # Default classification
        protection_level = DataProtectionLevel.INTERNAL
        
        # Check for sensitive indicators
        if self._contains_personal_identifiers(data, context):
            protection_level = DataProtectionLevel.CONFIDENTIAL
        
        if self._contains_sensitive_personal_data(data, context):
            protection_level = DataProtectionLevel.RESTRICTED
        
        if context.get('data_category') == 'public':
            protection_level = DataProtectionLevel.PUBLIC
        
        # Apply regional policy
        regional_policy = self.protection_policies.get(self.region, self.protection_policies["global"])
        min_level = regional_policy["min_protection_level"]
        
        # Use higher protection level
        levels = [DataProtectionLevel.PUBLIC, DataProtectionLevel.INTERNAL, 
                 DataProtectionLevel.CONFIDENTIAL, DataProtectionLevel.RESTRICTED]
        
        current_index = levels.index(protection_level)
        min_index = levels.index(min_level)
        
        final_level = levels[max(current_index, min_index)]
        
        self._log_audit_event("data_classification", {
            "protection_level": final_level.value,
            "data_type": type(data).__name__,
            "region": self.region
        })
        
        return final_level
    
    def apply_data_protection(self, data: Any, protection_level: DataProtectionLevel, 
                            purpose: str) -> Dict[str, Any]:
        """Apply appropriate data protection measures."""
        protected_data = {
            "original_data": data,
            "protection_level": protection_level.value,
            "protections_applied": [],
            "metadata": {
                "protection_date": datetime.utcnow().isoformat(),
                "purpose": purpose,
                "region": self.region
            }
        }
        
        # Apply protections based on level
        if protection_level in [DataProtectionLevel.CONFIDENTIAL, DataProtectionLevel.RESTRICTED]:
            # Apply encryption
            encrypted_data = self._encrypt_data(data, EncryptionMethod.AES_256)
            protected_data["protected_data"] = encrypted_data["encrypted_data"]
            protected_data["encryption_metadata"] = encrypted_data["metadata"]
            protected_data["protections_applied"].append("encryption")
            
            # Apply pseudonymization if required
            regional_policy = self.protection_policies.get(self.region, self.protection_policies["global"])
            if regional_policy["pseudonymization_required"]:
                pseudonym_data = self._pseudonymize_data(data)
                protected_data["pseudonymized_data"] = pseudonym_data["pseudonymized"]
                protected_data["pseudonym_mapping"] = pseudonym_data["mapping"]
                protected_data["protections_applied"].append("pseudonymization")
        
        if protection_level == DataProtectionLevel.RESTRICTED:
            # Apply additional protections for restricted data
            anonymized_data = self._anonymize_data(data)
            protected_data["anonymized_data"] = anonymized_data["anonymized"]
            protected_data["anonymization_metadata"] = anonymized_data["metadata"]
            protected_data["protections_applied"].append("anonymization")
            
            # Apply differential privacy
            if isinstance(data, torch.Tensor):
                dp_data = self._apply_differential_privacy(data, epsilon=1.0)
                protected_data["differential_privacy_data"] = dp_data["noisy_data"]
                protected_data["privacy_budget"] = dp_data["epsilon"]
                protected_data["protections_applied"].append("differential_privacy")
        
        # Log protection application
        self._log_audit_event("protection_applied", {
            "protection_level": protection_level.value,
            "protections": protected_data["protections_applied"],
            "purpose": purpose
        })
        
        return protected_data
    
    def _encrypt_data(self, data: Any, method: EncryptionMethod) -> Dict[str, Any]:
        """Encrypt data using specified method."""
        if isinstance(data, torch.Tensor):
            # For tensors, encrypt the serialized form
            data_bytes = data.cpu().numpy().tobytes()
            encrypted_hash = hashlib.sha256(data_bytes + b"encryption_key").hexdigest()
            
            return {
                "encrypted_data": encrypted_hash,
                "metadata": {
                    "method": method.value,
                    "original_shape": data.shape,
                    "original_dtype": str(data.dtype),
                    "encrypted_at": datetime.utcnow().isoformat()
                }
            }
        else:
            # For other data types
            data_str = str(data)
            encrypted_hash = hashlib.sha256(data_str.encode() + b"encryption_key").hexdigest()
            
            return {
                "encrypted_data": encrypted_hash,
                "metadata": {
                    "method": method.value,
                    "original_type": type(data).__name__,
                    "encrypted_at": datetime.utcnow().isoformat()
                }
            }
    
    def _pseudonymize_data(self, data: Any) -> Dict[str, Any]:
        """Create pseudonymized version of data."""
        if isinstance(data, torch.Tensor):
            # Create pseudonymous mapping for tensor indices
            unique_values = torch.unique(data.flatten())
            pseudonym_mapping = {}
            
            for i, value in enumerate(unique_values):
                pseudonym_hash = hashlib.sha256(f"pseudonym_{value.item()}".encode()).hexdigest()[:8]
                pseudonym_mapping[value.item()] = pseudonym_hash
            
            # Apply pseudonymization (simplified)
            pseudonymized = data.clone().float()
            for original, pseudo in pseudonym_mapping.items():
                mask = data == original
                pseudonymized[mask] = hash(pseudo) % 10000  # Simple numeric pseudonym
            
            return {
                "pseudonymized": pseudonymized,
                "mapping": pseudonym_mapping,
                "method": "hash_based"
            }
        else:
            # Simple hash-based pseudonymization for other data
            data_hash = hashlib.sha256(str(data).encode()).hexdigest()[:16]
            return {
                "pseudonymized": f"pseudo_{data_hash}",
                "mapping": {str(data): data_hash},
                "method": "hash_based"
            }
    
    def _anonymize_data(self, data: Any) -> Dict[str, Any]:
        """Create anonymized version of data."""
        if isinstance(data, torch.Tensor):
            # K-anonymity approach (simplified)
            k = 3  # K-anonymity parameter
            anonymized = data.clone()
            
            # Generalize values to achieve k-anonymity
            if data.numel() > k:
                # Group values and generalize
                sorted_data, indices = torch.sort(data.flatten())
                group_size = len(sorted_data) // k
                
                for i in range(0, len(sorted_data), group_size):
                    end_idx = min(i + group_size, len(sorted_data))
                    group_mean = torch.mean(sorted_data[i:end_idx])
                    
                    # Replace group values with mean
                    for j in range(i, end_idx):
                        original_pos = indices[j]
                        anonymized.flatten()[original_pos] = group_mean
            
            return {
                "anonymized": anonymized,
                "metadata": {
                    "method": "k_anonymity",
                    "k_value": k,
                    "anonymized_at": datetime.utcnow().isoformat()
                }
            }
        else:
            # Simple anonymization for other data types
            return {
                "anonymized": f"<anonymized_{type(data).__name__}>",
                "metadata": {
                    "method": "type_replacement",
                    "anonymized_at": datetime.utcnow().isoformat()
                }
            }
    
    def _apply_differential_privacy(self, data: torch.Tensor, epsilon: float = 1.0) -> Dict[str, Any]:
        """Apply differential privacy to tensor data."""
        sensitivity = 1.0  # L2 sensitivity
        noise_scale = sensitivity / epsilon
        
        # Add calibrated Gaussian noise
        noise = torch.randn_like(data) * noise_scale
        noisy_data = data + noise
        
        return {
            "noisy_data": noisy_data,
            "epsilon": epsilon,
            "delta": 1e-5,  # Standard delta value
            "sensitivity": sensitivity,
            "noise_scale": noise_scale,
            "method": "gaussian_mechanism"
        }
    
    def check_cross_border_transfer_compliance(self, target_region: str, 
                                             data_category: str) -> Dict[str, Any]:
        """Check if cross-border transfer is compliant."""
        source_policy = self.protection_policies.get(self.region, self.protection_policies["global"])
        target_policy = self.protection_policies.get(target_region, self.protection_policies["global"])
        
        compliance_check = {
            "transfer_allowed": True,
            "requirements": [],
            "safeguards_needed": [],
            "adequacy_decision": False
        }
        
        # Check if source region has cross-border restrictions
        if source_policy.get("cross_border_restrictions"):
            compliance_check["requirements"].append("adequacy_decision_or_safeguards")
            
            # Check adequacy (simplified - would use real adequacy decisions)
            adequate_regions = {
                "eu": ["sg"],  # EU-Singapore adequacy decision
                "sg": ["eu"]   # Reciprocal
            }
            
            if target_region in adequate_regions.get(self.region, []):
                compliance_check["adequacy_decision"] = True
            else:
                compliance_check["safeguards_needed"].extend([
                    "standard_contractual_clauses",
                    "binding_corporate_rules",
                    "certification_mechanism"
                ])
        
        # Additional protections for sensitive data
        if data_category in ["sensitive_personal_data", "special_category"]:
            compliance_check["safeguards_needed"].extend([
                "encryption_in_transit",
                "access_controls",
                "audit_logging"
            ])
        
        self._log_audit_event("transfer_compliance_check", {
            "source_region": self.region,
            "target_region": target_region,
            "data_category": data_category,
            "allowed": compliance_check["transfer_allowed"]
        })
        
        return compliance_check
    
    def apply_retention_policy(self, data_age: timedelta, data_category: str, 
                             purpose: str) -> Dict[str, Any]:
        """Apply data retention policy."""
        regional_policy = self.protection_policies.get(self.region, self.protection_policies["global"])
        retention_limits = regional_policy.get("retention_limits", {})
        
        # Determine retention period
        retention_period = timedelta(days=retention_limits.get(data_category, 
                                                             retention_limits.get("default", 1095)))
        
        retention_decision = {
            "retain": data_age < retention_period,
            "retention_period_days": retention_period.days,
            "data_age_days": data_age.days,
            "action_required": "none"
        }
        
        if data_age >= retention_period:
            retention_decision["action_required"] = "delete_or_anonymize"
            retention_decision["deadline"] = (datetime.utcnow() + timedelta(days=30)).isoformat()
        elif data_age >= retention_period * 0.9:  # 90% of retention period
            retention_decision["action_required"] = "review_retention_need"
            retention_decision["review_date"] = (datetime.utcnow() + timedelta(days=30)).isoformat()
        
        self._log_audit_event("retention_policy_applied", {
            "data_category": data_category,
            "purpose": purpose,
            "retain": retention_decision["retain"],
            "action": retention_decision["action_required"]
        })
        
        return retention_decision
    
    def _contains_personal_identifiers(self, data: Any, context: Dict[str, Any]) -> bool:
        """Check if data contains personal identifiers."""
        # Simplified heuristics
        if context.get("contains_pii", False):
            return True
        
        if isinstance(data, torch.Tensor):
            # Check for patterns that might indicate IDs
            if data.dtype in [torch.long, torch.int32, torch.int64]:
                if context.get("data_type") in ["user_ids", "node_ids"]:
                    return True
        
        return False
    
    def _contains_sensitive_personal_data(self, data: Any, context: Dict[str, Any]) -> bool:
        """Check if data contains sensitive personal information."""
        sensitive_indicators = [
            "biometric", "health", "genetic", "racial", "ethnic", 
            "political", "religious", "sexual", "criminal"
        ]
        
        data_description = context.get("description", "").lower()
        return any(indicator in data_description for indicator in sensitive_indicators)
    
    def _log_audit_event(self, event_type: str, details: Dict[str, Any]):
        """Log audit event."""
        audit_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": event_type,
            "region": self.region,
            "details": details
        }
        
        self.audit_log.append(audit_entry)
        
        # Keep only recent audit entries
        if len(self.audit_log) > 1000:
            self.audit_log = self.audit_log[-1000:]
        
        self.logger.debug(f"Audit event logged: {event_type}")
    
    def get_protection_report(self) -> Dict[str, Any]:
        """Generate data protection compliance report."""
        return {
            "region": self.region,
            "compliance_regimes": self.compliance_regimes,
            "protection_policies": self.protection_policies.get(self.region, {}),
            "audit_entries": len(self.audit_log),
            "recent_events": self.audit_log[-10:] if self.audit_log else [],
            "encryption_status": "active",
            "report_generated": datetime.utcnow().isoformat()
        }
    
    def validate_compliance_configuration(self) -> Dict[str, Any]:
        """Validate current compliance configuration."""
        validation_results = {
            "valid": True,
            "warnings": [],
            "errors": [],
            "recommendations": []
        }
        
        # Check regional policy exists
        if self.region not in self.protection_policies:
            validation_results["warnings"].append(f"No specific policy for region '{self.region}', using global policy")
        
        # Check encryption is available
        if not self.encryption_keys.get("initialized"):
            validation_results["errors"].append("Encryption system not properly initialized")
            validation_results["valid"] = False
        
        # Check compliance regime coverage
        supported_regimes = ["gdpr", "ccpa", "pdpa"]
        for regime in self.compliance_regimes:
            if regime not in supported_regimes:
                validation_results["warnings"].append(f"Compliance regime '{regime}' not fully supported")
        
        # Recommendations
        validation_results["recommendations"].extend([
            "Regularly review and update protection policies",
            "Implement proper key management system",
            "Set up automated data retention workflows",
            "Enable real-time compliance monitoring"
        ])
        
        return validation_results