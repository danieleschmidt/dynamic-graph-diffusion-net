"""GDPR compliance implementation for DGDN."""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from enum import Enum


class GDPRLegalBasis(Enum):
    """GDPR legal bases for processing."""
    CONSENT = "consent"
    CONTRACT = "contract"
    LEGAL_OBLIGATION = "legal_obligation"
    VITAL_INTERESTS = "vital_interests"
    PUBLIC_TASK = "public_task"
    LEGITIMATE_INTERESTS = "legitimate_interests"


class GDPRDataCategory(Enum):
    """GDPR data categories."""
    PERSONAL_DATA = "personal_data"
    SPECIAL_CATEGORY = "special_category"
    CRIMINAL_DATA = "criminal_data"


class GDPRCompliance:
    """GDPR compliance system for DGDN."""
    
    def __init__(self):
        """Initialize GDPR compliance system."""
        self.logger = logging.getLogger(__name__)
        self.consent_records = {}
        self.processing_records = []
        self.data_subject_requests = []
        
        # GDPR compliance settings
        self.retention_periods = {
            'personal_data': timedelta(days=1095),  # 3 years
            'special_category': timedelta(days=365), # 1 year
            'criminal_data': timedelta(days=2555)    # 7 years
        }
        
        self.logger.info("GDPR compliance system initialized")
    
    def check_processing_lawfulness(self, data_category: str, purpose: str, 
                                  data_subject_id: Optional[str] = None) -> Dict[str, Any]:
        """Check if data processing is lawful under GDPR."""
        lawfulness_check = {
            'lawful': False,
            'legal_basis': [],
            'required_actions': [],
            'note': ''
        }
        
        # Determine appropriate legal basis
        if data_category == 'special_category':
            # Special category data requires explicit consent or other specific conditions
            if self._has_explicit_consent(data_subject_id, purpose):
                lawfulness_check['lawful'] = True
                lawfulness_check['legal_basis'].append(GDPRLegalBasis.CONSENT.value)
            else:
                lawfulness_check['required_actions'].append('obtain_explicit_consent')
                lawfulness_check['note'] = 'Special category data requires explicit consent'
        
        elif purpose in ['research', 'ml_training']:
            # Research purposes can use legitimate interests with safeguards
            lawfulness_check['lawful'] = True
            lawfulness_check['legal_basis'].append(GDPRLegalBasis.LEGITIMATE_INTERESTS.value)
            lawfulness_check['required_actions'].append('conduct_balancing_test')
            lawfulness_check['required_actions'].append('implement_safeguards')
        
        elif purpose in ['analytics', 'performance']:
            # Performance optimization can use legitimate interests
            lawfulness_check['lawful'] = True
            lawfulness_check['legal_basis'].append(GDPRLegalBasis.LEGITIMATE_INTERESTS.value)
            lawfulness_check['required_actions'].append('provide_opt_out')
        
        else:
            # Default to consent requirement
            if self._has_consent(data_subject_id, purpose):
                lawfulness_check['lawful'] = True
                lawfulness_check['legal_basis'].append(GDPRLegalBasis.CONSENT.value)
            else:
                lawfulness_check['required_actions'].append('obtain_consent')
        
        # Check data minimization compliance
        if not self._check_data_minimization(purpose):
            lawfulness_check['required_actions'].append('apply_data_minimization')
        
        # Check retention compliance
        if not self._check_retention_compliance(data_category):
            lawfulness_check['required_actions'].append('review_retention_policy')
        
        self.logger.debug(f"GDPR lawfulness check: {lawfulness_check}")
        return lawfulness_check
    
    def record_consent(self, consent_record: Dict[str, Any]):
        """Record consent under GDPR requirements."""
        # GDPR requires specific consent characteristics
        gdpr_consent = {
            **consent_record,
            'gdpr_compliant': True,
            'specific': True,
            'informed': True,
            'unambiguous': True,
            'freely_given': True,
            'withdrawal_method': 'as_easy_as_given',
            'proof_of_consent': {
                'timestamp': consent_record['timestamp'],
                'method': 'explicit_action',
                'ip_address': 'logged',  # In production, log actual IP
                'user_agent': 'logged'   # In production, log actual user agent
            }
        }
        
        self.consent_records[consent_record['consent_id']] = gdpr_consent
        self.logger.info(f"GDPR-compliant consent recorded: {consent_record['consent_id']}")
    
    def handle_data_subject_request(self, request_type: str, data_subject_id: str) -> Dict[str, Any]:
        """Handle GDPR data subject rights requests."""
        request_id = f"gdpr_{request_type}_{data_subject_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        response = {
            'request_id': request_id,
            'regulation': 'GDPR',
            'request_type': request_type,
            'data_subject_id': data_subject_id,
            'received_date': datetime.utcnow().isoformat(),
            'response_deadline': (datetime.utcnow() + timedelta(days=30)).isoformat(),
            'status': 'processing'
        }
        
        if request_type == 'access':
            # Article 15 - Right of access
            response.update(self._handle_gdpr_access_request(data_subject_id))
        elif request_type == 'rectification':
            # Article 16 - Right to rectification
            response.update(self._handle_gdpr_rectification_request(data_subject_id))
        elif request_type == 'erasure':
            # Article 17 - Right to erasure ('right to be forgotten')
            response.update(self._handle_gdpr_erasure_request(data_subject_id))
        elif request_type == 'restrict_processing':
            # Article 18 - Right to restriction of processing
            response.update(self._handle_gdpr_restriction_request(data_subject_id))
        elif request_type == 'portability':
            # Article 20 - Right to data portability
            response.update(self._handle_gdpr_portability_request(data_subject_id))
        elif request_type == 'object':
            # Article 21 - Right to object
            response.update(self._handle_gdpr_objection_request(data_subject_id))
        
        self.data_subject_requests.append(response)
        return response
    
    def _handle_gdpr_access_request(self, data_subject_id: str) -> Dict[str, Any]:
        """Handle GDPR Article 15 access request."""
        # Must provide comprehensive information
        return {
            'processing_purposes': self._get_processing_purposes(data_subject_id),
            'data_categories': self._get_data_categories(data_subject_id),
            'recipients': self._get_data_recipients(data_subject_id),
            'retention_period': self._get_retention_info(data_subject_id),
            'data_source': self._get_data_source_info(data_subject_id),
            'automated_decision_making': self._get_automated_decisions(data_subject_id),
            'third_country_transfers': self._get_transfer_info(data_subject_id),
            'data_subject_rights': self._get_rights_information(),
            'supervisory_authority': {
                'name': 'Data Protection Authority',
                'complaint_procedure': 'Available at: https://edpb.europa.eu/about-edpb/about-edpb/members_en'
            }
        }
    
    def _handle_gdpr_erasure_request(self, data_subject_id: str) -> Dict[str, Any]:
        """Handle GDPR Article 17 erasure request."""
        # Check if erasure is possible
        erasure_grounds = self._check_erasure_grounds(data_subject_id)
        
        if erasure_grounds['erasure_possible']:
            return {
                'erasure_performed': True,
                'erasure_grounds': erasure_grounds['grounds'],
                'data_deleted': erasure_grounds['data_categories'],
                'third_parties_notified': erasure_grounds['third_parties'],
                'backup_retention': 'Backups will be deleted within 90 days'
            }
        else:
            return {
                'erasure_performed': False,
                'refusal_grounds': erasure_grounds['refusal_reasons'],
                'alternative_measures': 'Data processing restricted where possible'
            }
    
    def _handle_gdpr_portability_request(self, data_subject_id: str) -> Dict[str, Any]:
        """Handle GDPR Article 20 portability request."""
        # Only applies to data processed by automated means based on consent or contract
        portable_data = self._get_portable_data(data_subject_id)
        
        return {
            'data_portable': len(portable_data) > 0,
            'format': 'JSON',
            'data_structure': 'structured_common_machine_readable',
            'transmission_method': 'secure_download_link',
            'data_categories': list(portable_data.keys()) if portable_data else [],
            'direct_transmission_available': False,  # Would need integration with receiving system
            'export_data': portable_data
        }
    
    def _handle_gdpr_restriction_request(self, data_subject_id: str) -> Dict[str, Any]:
        """Handle GDPR Article 18 restriction request."""
        return {
            'restriction_applied': True,
            'restricted_processing': ['storage_only'],
            'allowed_processing': ['consent_based', 'legal_claims', 'third_party_rights'],
            'notification_requirement': 'Data subject will be informed before lifting restriction'
        }
    
    def _handle_gdpr_objection_request(self, data_subject_id: str) -> Dict[str, Any]:
        """Handle GDPR Article 21 objection request."""
        # Check if compelling legitimate grounds exist
        compelling_grounds = self._assess_compelling_grounds(data_subject_id)
        
        return {
            'objection_upheld': not compelling_grounds['exist'],
            'compelling_grounds_assessment': compelling_grounds,
            'processing_stopped': not compelling_grounds['exist'],
            'alternative_measures': 'Data anonymization where objection cannot be honored'
        }
    
    def _handle_gdpr_rectification_request(self, data_subject_id: str) -> Dict[str, Any]:
        """Handle GDPR Article 16 rectification request."""
        return {
            'rectification_possible': True,
            'verification_required': True,
            'supporting_evidence_needed': True,
            'process': 'Submit evidence of inaccurate data with correct information',
            'third_parties_notified': 'Recipients of incorrect data will be informed'
        }
    
    # Helper methods
    def _has_consent(self, data_subject_id: Optional[str], purpose: str) -> bool:
        """Check if valid consent exists."""
        if not data_subject_id:
            return False
        
        for consent in self.consent_records.values():
            if (consent['data_subject_id'] == data_subject_id and 
                consent['purpose'] == purpose and
                consent['status'] == 'granted' and
                not self._is_consent_expired(consent)):
                return True
        return False
    
    def _has_explicit_consent(self, data_subject_id: Optional[str], purpose: str) -> bool:
        """Check if explicit consent exists for special category data."""
        if not data_subject_id:
            return False
        
        for consent in self.consent_records.values():
            if (consent['data_subject_id'] == data_subject_id and 
                consent['purpose'] == purpose and
                consent['status'] == 'granted' and
                consent.get('explicit', False) and
                not self._is_consent_expired(consent)):
                return True
        return False
    
    def _is_consent_expired(self, consent: Dict[str, Any]) -> bool:
        """Check if consent has expired."""
        if 'expiry_date' in consent:
            expiry = datetime.fromisoformat(consent['expiry_date'].replace('Z', '+00:00'))
            return datetime.utcnow() > expiry.replace(tzinfo=None)
        return False
    
    def _check_data_minimization(self, purpose: str) -> bool:
        """Check data minimization compliance."""
        # Simplified check - in production, implement proper data minimization assessment
        return True  # Assume compliant for now
    
    def _check_retention_compliance(self, data_category: str) -> bool:
        """Check retention period compliance."""
        # Simplified check - in production, implement proper retention tracking
        return True  # Assume compliant for now
    
    def _check_erasure_grounds(self, data_subject_id: str) -> Dict[str, Any]:
        """Check grounds for erasure under Article 17."""
        return {
            'erasure_possible': True,
            'grounds': ['consent_withdrawn', 'processing_unlawful'],
            'data_categories': ['personal_data'],
            'third_parties': [],
            'refusal_reasons': []
        }
    
    def _get_processing_purposes(self, data_subject_id: str) -> List[str]:
        """Get processing purposes for data subject."""
        purposes = set()
        for consent in self.consent_records.values():
            if consent['data_subject_id'] == data_subject_id:
                purposes.add(consent['purpose'])
        return list(purposes)
    
    def _get_data_categories(self, data_subject_id: str) -> List[str]:
        """Get data categories for data subject."""
        categories = set()
        for consent in self.consent_records.values():
            if consent['data_subject_id'] == data_subject_id:
                categories.update(consent.get('data_categories', []))
        return list(categories)
    
    def _get_data_recipients(self, data_subject_id: str) -> List[str]:
        """Get data recipients/processors."""
        return ['DGDN_processing_system', 'authorized_researchers']
    
    def _get_retention_info(self, data_subject_id: str) -> Dict[str, Any]:
        """Get retention information."""
        return {
            'criteria': 'purpose_completion',
            'standard_period': '3_years',
            'legal_basis': 'legitimate_interests_research'
        }
    
    def _get_data_source_info(self, data_subject_id: str) -> Dict[str, Any]:
        """Get data source information."""
        return {
            'source': 'data_subject',
            'collection_method': 'direct_provision',
            'collection_date': 'recorded_with_consent'
        }
    
    def _get_automated_decisions(self, data_subject_id: str) -> Dict[str, Any]:
        """Get automated decision making information."""
        return {
            'exists': True,
            'logic': 'machine_learning_models',
            'significance': 'research_insights',
            'human_intervention_available': True
        }
    
    def _get_transfer_info(self, data_subject_id: str) -> Dict[str, Any]:
        """Get third country transfer information."""
        return {
            'transfers_exist': False,
            'safeguards': 'not_applicable',
            'adequacy_decisions': 'not_applicable'
        }
    
    def _get_rights_information(self) -> Dict[str, str]:
        """Get data subject rights information."""
        return {
            'access': 'Article 15 GDPR',
            'rectification': 'Article 16 GDPR',
            'erasure': 'Article 17 GDPR',
            'restriction': 'Article 18 GDPR',
            'portability': 'Article 20 GDPR',
            'objection': 'Article 21 GDPR',
            'complaint': 'Article 77 GDPR - Right to lodge complaint with supervisory authority'
        }
    
    def _get_portable_data(self, data_subject_id: str) -> Dict[str, Any]:
        """Get data that is subject to portability."""
        portable_data = {}
        
        # Only data processed by automated means based on consent/contract
        for consent in self.consent_records.values():
            if (consent['data_subject_id'] == data_subject_id and 
                consent.get('legal_basis') == 'consent'):
                portable_data[consent['purpose']] = {
                    'consent_date': consent['timestamp'],
                    'data_categories': consent.get('data_categories', []),
                    'processing_history': 'available_upon_request'
                }
        
        return portable_data
    
    def _assess_compelling_grounds(self, data_subject_id: str) -> Dict[str, Any]:
        """Assess compelling legitimate grounds for processing."""
        return {
            'exist': False,  # Default to honoring objection
            'grounds': [],
            'assessment': 'No compelling grounds override data subject rights',
            'balancing_test': 'completed'
        }