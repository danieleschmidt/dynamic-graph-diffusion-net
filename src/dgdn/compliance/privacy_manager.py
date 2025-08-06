"""Unified privacy management for DGDN compliance."""

import logging
import hashlib
import json
import uuid
from typing import Dict, List, Optional, Any, Set, Union
from datetime import datetime, timedelta
from enum import Enum
import torch

from .gdpr import GDPRCompliance
from .ccpa import CCPACompliance
from .pdpa import PDPACompliance


class PrivacyRegime(Enum):
    """Supported privacy regimes."""
    GDPR = "gdpr"
    CCPA = "ccpa"
    PDPA = "pdpa"
    ALL = "all"


class DataCategory(Enum):
    """Categories of data for privacy classification."""
    PERSONAL_IDENTIFIABLE = "pii"
    SENSITIVE_PERSONAL = "sensitive"
    BEHAVIORAL = "behavioral"
    TECHNICAL = "technical"
    ANONYMOUS = "anonymous"


class ProcessingPurpose(Enum):
    """Legal purposes for data processing."""
    RESEARCH = "research"
    MACHINE_LEARNING = "ml_training"
    ANALYTICS = "analytics"
    PERFORMANCE_OPTIMIZATION = "performance"
    SECURITY = "security"


class PrivacyManager:
    """Unified privacy management system for DGDN."""
    
    def __init__(self, active_regimes: List[PrivacyRegime] = None):
        """Initialize privacy manager with compliance systems."""
        self.active_regimes = active_regimes or [PrivacyRegime.GDPR]
        self.logger = logging.getLogger(__name__)
        
        # Initialize compliance systems
        self.compliance_systems = {}
        if PrivacyRegime.GDPR in self.active_regimes or PrivacyRegime.ALL in self.active_regimes:
            self.compliance_systems[PrivacyRegime.GDPR] = GDPRCompliance()
        if PrivacyRegime.CCPA in self.active_regimes or PrivacyRegime.ALL in self.active_regimes:
            self.compliance_systems[PrivacyRegime.CCPA] = CCPACompliance()
        if PrivacyRegime.PDPA in self.active_regimes or PrivacyRegime.ALL in self.active_regimes:
            self.compliance_systems[PrivacyRegime.PDPA] = PDPACompliance()
        
        # Privacy tracking
        self.processing_records = []
        self.consent_records = {}
        self.data_minimization_policies = {}
        
        self.logger.info(f"Privacy manager initialized with regimes: {[r.value for r in self.active_regimes]}")
    
    def classify_data(self, data: Any, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Classify data according to privacy categories."""
        context = context or {}
        
        classification = {
            'category': DataCategory.TECHNICAL,  # Default safe classification
            'contains_pii': False,
            'contains_sensitive': False,
            'risk_level': 'low',
            'required_protections': []
        }
        
        # Analyze data structure and content
        if isinstance(data, torch.Tensor):
            # Tensor data analysis
            if self._contains_potential_identifiers(data, context):
                classification['category'] = DataCategory.PERSONAL_IDENTIFIABLE
                classification['contains_pii'] = True
                classification['risk_level'] = 'high'
                classification['required_protections'] = ['encryption', 'access_control', 'audit_logging']
        
        elif isinstance(data, dict):
            # Dictionary data analysis
            pii_keys = {'name', 'email', 'id', 'user_id', 'phone', 'address'}
            if any(key.lower() in str(data.keys()).lower() for key in pii_keys):
                classification['contains_pii'] = True
                classification['category'] = DataCategory.PERSONAL_IDENTIFIABLE
                classification['risk_level'] = 'high'
        
        # Apply additional context-based classification
        if context.get('data_source') == 'user_input':
            classification['risk_level'] = 'medium'
        
        self.logger.debug(f"Data classified as: {classification}")
        return classification
    
    def _contains_potential_identifiers(self, tensor: torch.Tensor, context: Dict[str, Any]) -> bool:
        """Heuristic check for potential identifiers in tensor data."""
        # This is a simplified heuristic - in production, use more sophisticated methods
        
        # Check tensor properties that might indicate PII
        if tensor.dtype in [torch.long, torch.int32, torch.int64]:
            # Integer tensors might contain IDs
            if context.get('contains_node_ids', False):
                return True
            
            # Large integer values might be IDs
            if tensor.numel() > 0 and torch.max(torch.abs(tensor)) > 1000000:
                return True
        
        # Check for structured data patterns
        if len(tensor.shape) == 2 and tensor.shape[1] in [64, 128, 256]:
            # Common embedding dimensions might contain encoded PII
            if context.get('data_type') == 'user_embeddings':
                return True
        
        return False
    
    def request_consent(self, purpose: ProcessingPurpose, data_subject_id: str, 
                       data_categories: List[DataCategory]) -> Dict[str, Any]:
        """Request and record consent for data processing."""
        consent_id = str(uuid.uuid4())
        consent_record = {
            'consent_id': consent_id,
            'data_subject_id': data_subject_id,
            'purpose': purpose.value,
            'data_categories': [cat.value for cat in data_categories],
            'timestamp': datetime.utcnow().isoformat(),
            'status': 'granted',  # In real implementation, this would be user-driven
            'withdrawal_instructions': self._generate_withdrawal_instructions(consent_id),
            'expiry_date': (datetime.utcnow() + timedelta(days=365)).isoformat(),
            'legal_basis': self._determine_legal_basis(purpose)
        }
        
        # Store consent record
        self.consent_records[consent_id] = consent_record
        
        # Notify compliance systems
        for regime, system in self.compliance_systems.items():
            if hasattr(system, 'record_consent'):
                system.record_consent(consent_record)
        
        self.logger.info(f"Consent recorded: {consent_id} for purpose {purpose.value}")
        return consent_record
    
    def check_processing_lawfulness(self, data: Any, purpose: ProcessingPurpose, 
                                  data_subject_id: Optional[str] = None) -> Dict[str, Any]:
        """Check if data processing is lawful under active regimes."""
        classification = self.classify_data(data)
        
        lawfulness_check = {
            'lawful': True,
            'legal_basis': [],
            'required_actions': [],
            'compliance_notes': []
        }
        
        # Check each active compliance regime
        for regime, system in self.compliance_systems.items():
            regime_check = system.check_processing_lawfulness(
                data_category=classification['category'].value,
                purpose=purpose.value,
                data_subject_id=data_subject_id
            )
            
            if not regime_check.get('lawful', True):
                lawfulness_check['lawful'] = False
            
            lawfulness_check['legal_basis'].extend(regime_check.get('legal_basis', []))
            lawfulness_check['required_actions'].extend(regime_check.get('required_actions', []))
            lawfulness_check['compliance_notes'].append(f"{regime.value}: {regime_check.get('note', 'OK')}")
        
        # Record processing activity
        self._record_processing_activity(data, purpose, classification, lawfulness_check)
        
        return lawfulness_check
    
    def apply_data_minimization(self, data: Any, purpose: ProcessingPurpose) -> Any:
        """Apply data minimization principles."""
        if not isinstance(data, torch.Tensor):
            return data  # Only handle tensor data for now
        
        # Get minimization policy for purpose
        policy = self.data_minimization_policies.get(purpose, {})
        
        minimized_data = data.clone()
        
        # Apply dimension reduction if specified
        if 'max_dimensions' in policy:
            max_dim = policy['max_dimensions']
            if len(data.shape) > 1 and data.shape[1] > max_dim:
                # Simple truncation - in production, use PCA or other methods
                minimized_data = data[:, :max_dim]
                self.logger.info(f"Applied dimension reduction: {data.shape} -> {minimized_data.shape}")
        
        # Apply noise for differential privacy if specified
        if policy.get('add_noise', False):
            noise_scale = policy.get('noise_scale', 0.01)
            noise = torch.randn_like(minimized_data) * noise_scale
            minimized_data = minimized_data + noise
            self.logger.info(f"Applied differential privacy noise (scale: {noise_scale})")
        
        return minimized_data
    
    def anonymize_data(self, data: Any, method: str = 'hash') -> Dict[str, Any]:
        """Anonymize data while preserving utility."""
        if isinstance(data, torch.Tensor):
            if method == 'hash':
                # Hash-based anonymization
                data_bytes = data.cpu().numpy().tobytes()
                data_hash = hashlib.sha256(data_bytes).hexdigest()
                return {
                    'anonymized_data': data_hash,
                    'method': 'hash',
                    'original_shape': data.shape,
                    'preserves_structure': False
                }
            elif method == 'differential_privacy':
                # Add calibrated noise
                epsilon = 1.0  # Privacy parameter
                sensitivity = torch.std(data).item()
                noise_scale = sensitivity / epsilon
                
                noisy_data = data + torch.randn_like(data) * noise_scale
                
                return {
                    'anonymized_data': noisy_data,
                    'method': 'differential_privacy',
                    'epsilon': epsilon,
                    'preserves_structure': True
                }
        
        # Fallback for other data types
        return {
            'anonymized_data': hashlib.sha256(str(data).encode()).hexdigest(),
            'method': 'simple_hash',
            'preserves_structure': False
        }
    
    def handle_data_subject_request(self, request_type: str, data_subject_id: str) -> Dict[str, Any]:
        """Handle data subject rights requests (access, deletion, portability)."""
        request_id = str(uuid.uuid4())
        
        response = {
            'request_id': request_id,
            'request_type': request_type,
            'data_subject_id': data_subject_id,
            'timestamp': datetime.utcnow().isoformat(),
            'status': 'processing'
        }
        
        if request_type == 'access':
            # Right to access
            response.update(self._handle_access_request(data_subject_id))
        elif request_type == 'deletion':
            # Right to deletion/erasure
            response.update(self._handle_deletion_request(data_subject_id))
        elif request_type == 'portability':
            # Right to data portability
            response.update(self._handle_portability_request(data_subject_id))
        elif request_type == 'rectification':
            # Right to rectification
            response.update(self._handle_rectification_request(data_subject_id))
        
        # Forward to all compliance systems
        for regime, system in self.compliance_systems.items():
            if hasattr(system, 'handle_data_subject_request'):
                system.handle_data_subject_request(request_type, data_subject_id)
        
        self.logger.info(f"Data subject request handled: {request_id}")
        return response
    
    def _handle_access_request(self, data_subject_id: str) -> Dict[str, Any]:
        """Handle right to access request."""
        # Find all data related to the subject
        related_consents = [
            consent for consent in self.consent_records.values()
            if consent['data_subject_id'] == data_subject_id
        ]
        
        related_processing = [
            record for record in self.processing_records
            if record.get('data_subject_id') == data_subject_id
        ]
        
        return {
            'data_found': len(related_consents) + len(related_processing) > 0,
            'consent_records': related_consents,
            'processing_records': related_processing,
            'data_categories': list(set([
                cat for consent in related_consents 
                for cat in consent.get('data_categories', [])
            ])),
            'status': 'completed'
        }
    
    def _handle_deletion_request(self, data_subject_id: str) -> Dict[str, Any]:
        """Handle right to deletion/erasure request."""
        # Remove consent records
        deleted_consents = [
            consent_id for consent_id, consent in self.consent_records.items()
            if consent['data_subject_id'] == data_subject_id
        ]
        
        for consent_id in deleted_consents:
            del self.consent_records[consent_id]
        
        # Mark processing records for deletion
        deleted_processing = 0
        for record in self.processing_records:
            if record.get('data_subject_id') == data_subject_id:
                record['deleted'] = True
                record['deletion_date'] = datetime.utcnow().isoformat()
                deleted_processing += 1
        
        return {
            'deleted_consents': len(deleted_consents),
            'deleted_processing_records': deleted_processing,
            'status': 'completed',
            'note': 'Data marked for deletion. Physical deletion may take up to 30 days.'
        }
    
    def _handle_portability_request(self, data_subject_id: str) -> Dict[str, Any]:
        """Handle right to data portability request."""
        # Collect portable data
        portable_data = self._handle_access_request(data_subject_id)
        
        # Format for portability (JSON format)
        export_data = {
            'data_subject_id': data_subject_id,
            'export_date': datetime.utcnow().isoformat(),
            'data': portable_data,
            'format': 'json',
            'schema_version': '1.0'
        }
        
        return {
            'portable_data': json.dumps(export_data, indent=2),
            'format': 'json',
            'size_bytes': len(json.dumps(export_data)),
            'status': 'completed'
        }
    
    def _handle_rectification_request(self, data_subject_id: str) -> Dict[str, Any]:
        """Handle right to rectification request."""
        return {
            'status': 'manual_review_required',
            'note': 'Rectification requests require manual review to determine appropriate action.',
            'contact_info': 'Please provide specific details about incorrect data.'
        }
    
    def _record_processing_activity(self, data: Any, purpose: ProcessingPurpose, 
                                  classification: Dict[str, Any], lawfulness_check: Dict[str, Any]):
        """Record processing activity for compliance audit trail."""
        record = {
            'timestamp': datetime.utcnow().isoformat(),
            'purpose': purpose.value,
            'data_category': classification['category'].value,
            'risk_level': classification['risk_level'],
            'lawful': lawfulness_check['lawful'],
            'legal_basis': lawfulness_check['legal_basis'],
            'data_size': self._estimate_data_size(data),
            'processing_id': str(uuid.uuid4())
        }
        
        self.processing_records.append(record)
        
        # Keep only recent records (last 1000)
        if len(self.processing_records) > 1000:
            self.processing_records = self.processing_records[-1000:]
    
    def _estimate_data_size(self, data: Any) -> int:
        """Estimate data size in bytes."""
        if isinstance(data, torch.Tensor):
            return data.numel() * data.element_size()
        elif isinstance(data, str):
            return len(data.encode('utf-8'))
        elif isinstance(data, (dict, list)):
            return len(str(data).encode('utf-8'))
        else:
            return len(str(data).encode('utf-8'))
    
    def _generate_withdrawal_instructions(self, consent_id: str) -> str:
        """Generate instructions for withdrawing consent."""
        return f"To withdraw consent {consent_id}, contact privacy@dgdn.ai or use the privacy dashboard."
    
    def _determine_legal_basis(self, purpose: ProcessingPurpose) -> str:
        """Determine legal basis for processing purpose."""
        legal_basis_map = {
            ProcessingPurpose.RESEARCH: "legitimate_interest",
            ProcessingPurpose.MACHINE_LEARNING: "consent",
            ProcessingPurpose.ANALYTICS: "legitimate_interest",
            ProcessingPurpose.PERFORMANCE_OPTIMIZATION: "legitimate_interest",
            ProcessingPurpose.SECURITY: "legitimate_interest"
        }
        return legal_basis_map.get(purpose, "consent")
    
    def set_data_minimization_policy(self, purpose: ProcessingPurpose, policy: Dict[str, Any]):
        """Set data minimization policy for a processing purpose."""
        self.data_minimization_policies[purpose] = policy
        self.logger.info(f"Data minimization policy set for {purpose.value}: {policy}")
    
    def get_compliance_report(self) -> Dict[str, Any]:
        """Generate compliance report."""
        return {
            'active_regimes': [regime.value for regime in self.active_regimes],
            'total_consent_records': len(self.consent_records),
            'total_processing_records': len(self.processing_records),
            'data_subject_count': len(set(
                consent['data_subject_id'] for consent in self.consent_records.values()
            )),
            'processing_purposes': list(set(
                record['purpose'] for record in self.processing_records
            )),
            'compliance_systems': {
                regime.value: system.__class__.__name__ 
                for regime, system in self.compliance_systems.items()
            },
            'report_date': datetime.utcnow().isoformat()
        }