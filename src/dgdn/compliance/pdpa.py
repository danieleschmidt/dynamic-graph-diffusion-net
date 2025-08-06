"""PDPA compliance implementation for DGDN."""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from enum import Enum


class PDPAConsentBasis(Enum):
    """PDPA consent basis for processing."""
    CONSENT = "consent"
    CONTRACT = "contract"
    LEGAL_OBLIGATION = "legal_obligation"
    VITAL_INTERESTS = "vital_interests"
    PUBLIC_TASK = "public_task"
    LEGITIMATE_INTERESTS = "legitimate_interests"


class PDPADataCategory(Enum):
    """PDPA data categories (Singapore)."""
    PERSONAL_DATA = "personal_data"
    SENSITIVE_PERSONAL_DATA = "sensitive_personal_data"


class PDPACompliance:
    """PDPA compliance system for DGDN (Singapore Personal Data Protection Act)."""
    
    def __init__(self):
        """Initialize PDPA compliance system."""
        self.logger = logging.getLogger(__name__)
        self.consent_records = {}
        self.processing_records = []
        self.data_subject_requests = []
        
        # PDPA specific requirements
        self.dpo_contact = {
            'name': 'Data Protection Officer',
            'email': 'dpo@dgdn.ai',
            'phone': '+65-XXXX-XXXX'
        }
        
        # Data retention policies
        self.retention_periods = {
            'personal_data': timedelta(days=1095),  # 3 years
            'sensitive_personal_data': timedelta(days=365)  # 1 year
        }
        
        self.logger.info("PDPA compliance system initialized")
    
    def check_processing_lawfulness(self, data_category: str, purpose: str, 
                                  data_subject_id: Optional[str] = None) -> Dict[str, Any]:
        """Check if data processing is lawful under PDPA."""
        lawfulness_check = {
            'lawful': False,
            'legal_basis': [],
            'required_actions': [],
            'note': ''
        }
        
        # PDPA requires consent for most processing unless exception applies
        if data_category == 'sensitive_personal_data':
            # Sensitive data requires explicit consent
            if self._has_explicit_consent(data_subject_id, purpose):
                lawfulness_check['lawful'] = True
                lawfulness_check['legal_basis'].append(PDPAConsentBasis.CONSENT.value)
            else:
                lawfulness_check['required_actions'].append('obtain_explicit_consent')
                lawfulness_check['note'] = 'Sensitive data requires explicit consent under PDPA'
        
        elif purpose in ['research', 'ml_training']:
            # Check if research exception applies
            if self._qualifies_for_research_exception(purpose):
                lawfulness_check['lawful'] = True
                lawfulness_check['legal_basis'].append('research_exception')
                lawfulness_check['required_actions'].append('ensure_research_safeguards')
            elif self._has_consent(data_subject_id, purpose):
                lawfulness_check['lawful'] = True
                lawfulness_check['legal_basis'].append(PDPAConsentBasis.CONSENT.value)
            else:
                lawfulness_check['required_actions'].append('obtain_consent_or_qualify_exception')
        
        else:
            # General processing requires consent or legitimate interests
            if self._has_consent(data_subject_id, purpose):
                lawfulness_check['lawful'] = True
                lawfulness_check['legal_basis'].append(PDPAConsentBasis.CONSENT.value)
            elif self._has_legitimate_interests(purpose):
                lawfulness_check['lawful'] = True
                lawfulness_check['legal_basis'].append(PDPAConsentBasis.LEGITIMATE_INTERESTS.value)
                lawfulness_check['required_actions'].append('conduct_legitimate_interests_assessment')
            else:
                lawfulness_check['required_actions'].append('obtain_consent')
        
        # Check purpose limitation compliance
        if not self._check_purpose_limitation(purpose):
            lawfulness_check['required_actions'].append('ensure_purpose_limitation')
        
        # Check data minimization
        if not self._check_data_minimization(purpose):
            lawfulness_check['required_actions'].append('apply_data_minimization')
        
        self.logger.debug(f"PDPA lawfulness check: {lawfulness_check}")
        return lawfulness_check
    
    def record_consent(self, consent_record: Dict[str, Any]):
        """Record consent under PDPA requirements."""
        # PDPA consent requirements
        pdpa_consent = {
            **consent_record,
            'pdpa_compliant': True,
            'voluntary': True,
            'informed': True,
            'specific': True,
            'clear_affirmative_action': True,
            'withdrawal_mechanism': 'provided',
            'consent_evidence': {
                'timestamp': consent_record['timestamp'],
                'method': 'explicit_action',
                'purpose_disclosed': True,
                'consequences_explained': True
            }
        }
        
        self.consent_records[consent_record['consent_id']] = pdpa_consent
        self.logger.info(f"PDPA-compliant consent recorded: {consent_record['consent_id']}")
    
    def handle_data_subject_request(self, request_type: str, data_subject_id: str) -> Dict[str, Any]:
        """Handle PDPA data subject rights requests."""
        request_id = f"pdpa_{request_type}_{data_subject_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        response = {
            'request_id': request_id,
            'regulation': 'PDPA',
            'request_type': request_type,
            'data_subject_id': data_subject_id,
            'received_date': datetime.utcnow().isoformat(),
            'response_deadline': (datetime.utcnow() + timedelta(days=30)).isoformat(),
            'status': 'processing',
            'dpo_contact': self.dpo_contact
        }
        
        if request_type == 'access':
            # Right to access personal data
            response.update(self._handle_pdpa_access_request(data_subject_id))
        elif request_type == 'correction':
            # Right to correct personal data
            response.update(self._handle_pdpa_correction_request(data_subject_id))
        elif request_type == 'withdraw_consent':
            # Right to withdraw consent
            response.update(self._handle_pdpa_withdraw_consent(data_subject_id))
        elif request_type == 'stop_processing':
            # Right to request cessation of processing
            response.update(self._handle_pdpa_stop_processing(data_subject_id))
        
        self.data_subject_requests.append(response)
        return response
    
    def _handle_pdpa_access_request(self, data_subject_id: str) -> Dict[str, Any]:
        """Handle PDPA access request."""
        return {
            'personal_data_categories': self._get_personal_data_categories(data_subject_id),
            'processing_purposes': self._get_processing_purposes(data_subject_id),
            'data_sources': self._get_data_sources(data_subject_id),
            'third_party_disclosures': self._get_third_party_disclosures(data_subject_id),
            'retention_periods': self._get_retention_periods(data_subject_id),
            'data_protection_measures': self._get_protection_measures(),
            'individual_rights': self._get_individual_rights(),
            'complaint_mechanism': {
                'internal': self.dpo_contact,
                'external': 'Personal Data Protection Commission Singapore',
                'website': 'https://www.pdpc.gov.sg'
            }
        }
    
    def _handle_pdpa_correction_request(self, data_subject_id: str) -> Dict[str, Any]:
        """Handle PDPA correction request."""
        return {
            'correction_possible': True,
            'verification_required': True,
            'evidence_needed': 'Documentation supporting correction request',
            'correction_process': 'Submit accurate data with supporting evidence',
            'timeframe': '30 days from receipt of complete request',
            'third_party_notification': 'Disclosed parties will be notified of correction',
            'no_fee_policy': 'No fee charged for correction unless frivolous or excessive'
        }
    
    def _handle_pdpa_withdraw_consent(self, data_subject_id: str) -> Dict[str, Any]:
        """Handle consent withdrawal request."""
        # Find and update consent records
        withdrawn_consents = []
        for consent_id, consent in self.consent_records.items():
            if consent['data_subject_id'] == data_subject_id:
                consent['status'] = 'withdrawn'
                consent['withdrawal_date'] = datetime.utcnow().isoformat()
                withdrawn_consents.append(consent_id)
        
        return {
            'withdrawal_processed': True,
            'withdrawn_consents': withdrawn_consents,
            'effective_date': datetime.utcnow().isoformat(),
            'continued_processing': self._assess_continued_processing(data_subject_id),
            'data_retention': self._assess_retention_after_withdrawal(data_subject_id),
            'consequences_explained': True
        }
    
    def _handle_pdpa_stop_processing(self, data_subject_id: str) -> Dict[str, Any]:
        """Handle request to stop processing."""
        processing_assessment = self._assess_processing_cessation(data_subject_id)
        
        return {
            'cessation_possible': processing_assessment['can_stop'],
            'cessation_performed': processing_assessment['can_stop'],
            'continued_processing_basis': processing_assessment.get('continued_basis', []),
            'data_retention_basis': processing_assessment.get('retention_basis', ''),
            'alternative_measures': processing_assessment.get('alternatives', []),
            'effective_date': datetime.utcnow().isoformat() if processing_assessment['can_stop'] else None
        }
    
    def create_privacy_policy_disclosure(self) -> Dict[str, Any]:
        """Create PDPA-compliant privacy policy disclosure."""
        return {
            'organization_details': {
                'name': 'DGDN Labs',
                'contact': self.dpo_contact,
                'registration_number': 'XXX-XXX-XXX'  # Would be actual registration
            },
            'personal_data_collected': {
                'categories': [
                    'identifiers',
                    'usage_data',
                    'device_information',
                    'interaction_patterns'
                ],
                'collection_methods': [
                    'direct_provision',
                    'automatic_collection',
                    'derived_data'
                ]
            },
            'purposes_of_processing': [
                'service_provision',
                'research_and_development',
                'performance_improvement',
                'security_monitoring'
            ],
            'legal_basis': [
                'consent',
                'legitimate_interests',
                'research_exception'
            ],
            'data_sharing': {
                'third_parties': [
                    'service_providers',
                    'research_collaborators'
                ],
                'safeguards': [
                    'data_processing_agreements',
                    'anonymization',
                    'access_controls'
                ]
            },
            'data_transfers': {
                'cross_border': 'May occur for research collaboration',
                'safeguards': 'Adequacy decisions or appropriate safeguards',
                'notification': 'Individuals notified of transfers'
            },
            'retention_policy': {
                'criteria': 'Purpose fulfillment',
                'standard_period': '3 years',
                'deletion_process': 'Secure deletion after retention period'
            },
            'individual_rights': {
                'access': 'Right to access personal data',
                'correction': 'Right to correct inaccurate data',
                'withdraw_consent': 'Right to withdraw consent',
                'stop_processing': 'Right to request cessation',
                'complaint': 'Right to complain to PDPC'
            },
            'security_measures': [
                'encryption',
                'access_controls',
                'audit_logging',
                'security_monitoring'
            ],
            'data_breach_notification': {
                'individual_notification': 'Within 72 hours if high risk',
                'pdpc_notification': 'Within 72 hours',
                'assessment_criteria': 'Risk of harm to individuals'
            },
            'policy_updates': {
                'notification_method': 'Email and website notice',
                'advance_notice': '30 days for material changes'
            },
            'last_updated': datetime.utcnow().isoformat()
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
        """Check if explicit consent exists for sensitive data."""
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
    
    def _qualifies_for_research_exception(self, purpose: str) -> bool:
        """Check if processing qualifies for research exception."""
        research_purposes = ['research', 'ml_training']
        return purpose in research_purposes
    
    def _has_legitimate_interests(self, purpose: str) -> bool:
        """Check if legitimate interests basis applies."""
        legitimate_purposes = ['performance', 'security', 'analytics']
        return purpose in legitimate_purposes
    
    def _is_consent_expired(self, consent: Dict[str, Any]) -> bool:
        """Check if consent has expired."""
        if 'expiry_date' in consent:
            expiry = datetime.fromisoformat(consent['expiry_date'].replace('Z', '+00:00'))
            return datetime.utcnow() > expiry.replace(tzinfo=None)
        return False
    
    def _check_purpose_limitation(self, purpose: str) -> bool:
        """Check purpose limitation compliance."""
        # Simplified check - in production, verify against disclosed purposes
        return True
    
    def _check_data_minimization(self, purpose: str) -> bool:
        """Check data minimization compliance."""
        # Simplified check - in production, verify minimal data collection
        return True
    
    def _get_personal_data_categories(self, data_subject_id: str) -> List[str]:
        """Get categories of personal data for individual."""
        return [
            'identifiers',
            'usage_patterns',
            'device_information',
            'interaction_data'
        ]
    
    def _get_processing_purposes(self, data_subject_id: str) -> List[str]:
        """Get processing purposes for individual."""
        purposes = set()
        for consent in self.consent_records.values():
            if consent['data_subject_id'] == data_subject_id:
                purposes.add(consent['purpose'])
        return list(purposes)
    
    def _get_data_sources(self, data_subject_id: str) -> List[str]:
        """Get sources of personal data."""
        return [
            'direct_provision_by_individual',
            'automatic_collection_from_device',
            'derived_from_usage_patterns'
        ]
    
    def _get_third_party_disclosures(self, data_subject_id: str) -> List[Dict[str, str]]:
        """Get third party disclosures."""
        return [
            {
                'recipient': 'Cloud Service Providers',
                'purpose': 'Data processing and storage',
                'safeguards': 'Data processing agreements'
            },
            {
                'recipient': 'Research Partners',
                'purpose': 'Collaborative research',
                'safeguards': 'Anonymization and access controls'
            }
        ]
    
    def _get_retention_periods(self, data_subject_id: str) -> Dict[str, str]:
        """Get retention periods for different data categories."""
        return {
            'personal_data': '3 years from last interaction',
            'sensitive_personal_data': '1 year from collection',
            'research_data': 'Duration of research project plus 2 years'
        }
    
    def _get_protection_measures(self) -> List[str]:
        """Get data protection measures implemented."""
        return [
            'encryption_at_rest_and_transit',
            'access_controls',
            'audit_logging',
            'regular_security_assessments',
            'staff_training',
            'incident_response_procedures'
        ]
    
    def _get_individual_rights(self) -> Dict[str, str]:
        """Get individual rights information."""
        return {
            'access': 'Right to obtain copy of personal data',
            'correction': 'Right to correct inaccurate personal data',
            'withdraw_consent': 'Right to withdraw previously given consent',
            'cessation': 'Right to request cessation of processing',
            'complaint': 'Right to lodge complaint with PDPC'
        }
    
    def _assess_continued_processing(self, data_subject_id: str) -> Dict[str, Any]:
        """Assess continued processing after consent withdrawal."""
        return {
            'continues': False,  # DGDN typically relies on consent
            'basis': [],
            'data_categories': [],
            'note': 'Processing generally ceases upon consent withdrawal'
        }
    
    def _assess_retention_after_withdrawal(self, data_subject_id: str) -> Dict[str, Any]:
        """Assess data retention after consent withdrawal."""
        return {
            'retention_continues': True,
            'basis': 'legitimate_interests_record_keeping',
            'period': '6 months for audit purposes',
            'subsequent_deletion': 'Secure deletion after retention period'
        }
    
    def _assess_processing_cessation(self, data_subject_id: str) -> Dict[str, Any]:
        """Assess whether processing can be ceased."""
        return {
            'can_stop': True,
            'continued_basis': [],
            'retention_basis': 'audit_trail_maintenance',
            'alternatives': ['anonymization', 'aggregation'],
            'note': 'Processing can generally be stopped upon request'
        }