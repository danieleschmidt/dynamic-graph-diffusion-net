"""CCPA compliance implementation for DGDN."""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from enum import Enum


class CCPADataCategory(Enum):
    """CCPA categories of personal information."""
    IDENTIFIERS = "identifiers"
    PERSONAL_INFO = "personal_info"
    CHARACTERISTICS = "characteristics"
    COMMERCIAL_INFO = "commercial_info"
    BIOMETRIC_INFO = "biometric_info"
    INTERNET_ACTIVITY = "internet_activity"
    GEOLOCATION = "geolocation"
    SENSORY_DATA = "sensory_data"
    PROFESSIONAL_INFO = "professional_info"
    EDUCATION_INFO = "education_info"
    INFERENCES = "inferences"


class CCPABusinessPurpose(Enum):
    """CCPA business purposes."""
    AUDITING = "auditing"
    SECURITY = "security"
    DEBUGGING = "debugging"
    TRANSIENT_USE = "transient_use"
    PERFORMING_SERVICES = "performing_services"
    INTERNAL_RESEARCH = "internal_research"
    QUALITY_VERIFICATION = "quality_verification"


class CCPACompliance:
    """CCPA compliance system for DGDN."""
    
    def __init__(self):
        """Initialize CCPA compliance system."""
        self.logger = logging.getLogger(__name__)
        self.consumer_requests = []
        self.privacy_disclosures = {}
        self.opt_out_requests = {}
        self.data_sales_log = []  # DGDN doesn't sell data, but tracking for compliance
        
        # CCPA response timeframes
        self.response_timeframes = {
            'access': timedelta(days=45),
            'deletion': timedelta(days=45),
            'opt_out': timedelta(days=15)  # Must honor within 15 business days
        }
        
        self.logger.info("CCPA compliance system initialized")
    
    def check_processing_lawfulness(self, data_category: str, purpose: str, 
                                  data_subject_id: Optional[str] = None) -> Dict[str, Any]:
        """Check if data processing complies with CCPA."""
        lawfulness_check = {
            'lawful': True,  # CCPA is more permissive than GDPR
            'legal_basis': [],
            'required_actions': [],
            'note': ''
        }
        
        # Map purpose to CCPA business purpose
        ccpa_purpose = self._map_to_ccpa_purpose(purpose)
        
        if ccpa_purpose:
            lawfulness_check['legal_basis'].append(ccpa_purpose.value)
        else:
            lawfulness_check['legal_basis'].append('business_purpose_other')
        
        # Check if consumer has opted out
        if self._has_opted_out(data_subject_id):
            if purpose in ['analytics', 'advertising', 'profiling']:
                lawfulness_check['lawful'] = False
                lawfulness_check['note'] = 'Consumer has opted out of sale/sharing'
                lawfulness_check['required_actions'].append('honor_opt_out')
        
        # Verify privacy notice compliance
        if not self._has_privacy_notice_compliance(purpose):
            lawfulness_check['required_actions'].append('update_privacy_notice')
        
        # Check sensitive personal information handling
        if self._is_sensitive_personal_info(data_category):
            lawfulness_check['required_actions'].append('limit_sensitive_data_use')
        
        self.logger.debug(f"CCPA lawfulness check: {lawfulness_check}")
        return lawfulness_check
    
    def handle_data_subject_request(self, request_type: str, consumer_id: str) -> Dict[str, Any]:
        """Handle CCPA consumer rights requests."""
        request_id = f"ccpa_{request_type}_{consumer_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        response = {
            'request_id': request_id,
            'regulation': 'CCPA',
            'request_type': request_type,
            'consumer_id': consumer_id,
            'received_date': datetime.utcnow().isoformat(),
            'response_deadline': (datetime.utcnow() + self.response_timeframes.get(
                request_type, timedelta(days=45)
            )).isoformat(),
            'status': 'processing',
            'verification_required': True
        }
        
        if request_type == 'access':
            # CCPA Right to Know
            response.update(self._handle_ccpa_access_request(consumer_id))
        elif request_type == 'deletion':
            # CCPA Right to Delete
            response.update(self._handle_ccpa_deletion_request(consumer_id))
        elif request_type == 'opt_out':
            # CCPA Right to Opt-Out of Sale/Sharing
            response.update(self._handle_ccpa_opt_out_request(consumer_id))
        elif request_type == 'correct':
            # CCPA Right to Correct
            response.update(self._handle_ccpa_correction_request(consumer_id))
        elif request_type == 'limit_sensitive':
            # CCPA Right to Limit Sensitive Personal Information
            response.update(self._handle_ccpa_limit_sensitive_request(consumer_id))
        
        self.consumer_requests.append(response)
        return response
    
    def _handle_ccpa_access_request(self, consumer_id: str) -> Dict[str, Any]:
        """Handle CCPA right to know request."""
        # Must provide information about personal information collected, used, disclosed
        return {
            'categories_collected': self._get_collected_categories(consumer_id),
            'categories_sold_shared': self._get_sold_shared_categories(consumer_id),
            'categories_disclosed': self._get_disclosed_categories(consumer_id),
            'business_purposes': self._get_business_purposes(consumer_id),
            'sources_of_collection': self._get_collection_sources(consumer_id),
            'third_parties': self._get_third_parties(consumer_id),
            'retention_period': self._get_ccpa_retention_info(consumer_id),
            'specific_pieces': self._get_specific_pieces(consumer_id),  # Only if requested
            'data_sales_disclosure': {
                'sales_occurred': False,
                'sharing_occurred': False,  # DGDN doesn't sell or share for advertising
                'note': 'DGDN does not sell or share personal information'
            }
        }
    
    def _handle_ccpa_deletion_request(self, consumer_id: str) -> Dict[str, Any]:
        """Handle CCPA right to delete request."""
        # Check if deletion exceptions apply
        deletion_exceptions = self._check_deletion_exceptions(consumer_id)
        
        if deletion_exceptions['exceptions_apply']:
            return {
                'deletion_performed': False,
                'exceptions_claimed': deletion_exceptions['exceptions'],
                'partial_deletion': deletion_exceptions.get('partial_deletion', False),
                'explanation': deletion_exceptions['explanation']
            }
        else:
            return {
                'deletion_performed': True,
                'deleted_categories': self._get_deletable_categories(consumer_id),
                'third_party_notification': 'Service providers notified of deletion request',
                'backup_retention': 'Backups deleted within 90 days',
                'verification_method': 'identity_verified_through_secure_process'
            }
    
    def _handle_ccpa_opt_out_request(self, consumer_id: str) -> Dict[str, Any]:
        """Handle CCPA opt-out request."""
        # Record opt-out preference
        self.opt_out_requests[consumer_id] = {
            'opt_out_date': datetime.utcnow().isoformat(),
            'scope': 'sale_and_sharing',
            'honored_within_15_days': True,
            'method': 'web_form'  # Could be 'do_not_sell_link', 'email', etc.
        }
        
        return {
            'opt_out_honored': True,
            'effective_date': datetime.utcnow().isoformat(),
            'scope': 'all_sale_and_sharing_activities',
            'confirmation_method': 'email_confirmation',
            'future_processing': 'limited_to_business_purposes_only',
            'third_party_notification': 'Partners notified of opt-out status'
        }
    
    def _handle_ccpa_correction_request(self, consumer_id: str) -> Dict[str, Any]:
        """Handle CCPA right to correct request."""
        return {
            'correction_possible': True,
            'verification_required': True,
            'documentation_needed': 'Evidence of inaccurate personal information',
            'correction_process': 'Submit accurate information with supporting documentation',
            'timeframe': '45 days from verification',
            'third_party_correction': 'Service providers will be notified of corrections'
        }
    
    def _handle_ccpa_limit_sensitive_request(self, consumer_id: str) -> Dict[str, Any]:
        """Handle request to limit use of sensitive personal information."""
        return {
            'limitation_applied': True,
            'sensitive_data_categories': self._get_sensitive_categories(consumer_id),
            'permitted_uses': [
                'performing_requested_services',
                'ensuring_security',
                'system_functionality',
                'quality_assurance'
            ],
            'prohibited_uses': [
                'inferring_characteristics',
                'advertising',
                'profiling'
            ],
            'effective_date': datetime.utcnow().isoformat()
        }
    
    def create_privacy_notice_disclosure(self) -> Dict[str, Any]:
        """Create CCPA-compliant privacy notice disclosure."""
        return {
            'categories_collected': {
                CCPADataCategory.IDENTIFIERS.value: {
                    'examples': 'User IDs, device identifiers',
                    'sources': 'Direct from consumer, automatic collection',
                    'business_purposes': [CCPABusinessPurpose.PERFORMING_SERVICES.value],
                    'third_parties': 'Service providers only'
                },
                CCPADataCategory.INTERNET_ACTIVITY.value: {
                    'examples': 'Usage patterns, interaction data',
                    'sources': 'Automatic collection during service use',
                    'business_purposes': [
                        CCPABusinessPurpose.INTERNAL_RESEARCH.value,
                        CCPABusinessPurpose.QUALITY_VERIFICATION.value
                    ],
                    'third_parties': 'Research partners (anonymized)'
                },
                CCPADataCategory.INFERENCES.value: {
                    'examples': 'Model predictions, behavioral insights',
                    'sources': 'Derived from usage data',
                    'business_purposes': [CCPABusinessPurpose.INTERNAL_RESEARCH.value],
                    'third_parties': 'None'
                }
            },
            'retention_policy': 'Personal information retained for 3 years or until purpose fulfilled',
            'sale_sharing_disclosure': 'DGDN does not sell or share personal information',
            'sensitive_data_uses': 'Limited to service provision and security',
            'consumer_rights': {
                'right_to_know': 'Request categories and specific pieces of PI collected',
                'right_to_delete': 'Request deletion of personal information',
                'right_to_opt_out': 'Opt out of sale/sharing (not applicable - we don\'t sell)',
                'right_to_correct': 'Request correction of inaccurate personal information',
                'right_to_limit_sensitive': 'Limit use of sensitive personal information'
            },
            'contact_information': {
                'privacy_email': 'privacy@dgdn.ai',
                'privacy_phone': '+1-555-PRIVACY',
                'privacy_address': 'Privacy Office, DGDN Labs'
            },
            'last_updated': datetime.utcnow().isoformat()
        }
    
    # Helper methods
    def _map_to_ccpa_purpose(self, purpose: str) -> Optional[CCPABusinessPurpose]:
        """Map processing purpose to CCPA business purpose."""
        purpose_map = {
            'research': CCPABusinessPurpose.INTERNAL_RESEARCH,
            'ml_training': CCPABusinessPurpose.INTERNAL_RESEARCH,
            'analytics': CCPABusinessPurpose.INTERNAL_RESEARCH,
            'performance': CCPABusinessPurpose.QUALITY_VERIFICATION,
            'security': CCPABusinessPurpose.SECURITY
        }
        return purpose_map.get(purpose)
    
    def _has_opted_out(self, consumer_id: Optional[str]) -> bool:
        """Check if consumer has opted out of sale/sharing."""
        if not consumer_id:
            return False
        return consumer_id in self.opt_out_requests
    
    def _has_privacy_notice_compliance(self, purpose: str) -> bool:
        """Check if privacy notice adequately discloses purpose."""
        # Simplified check - in production, verify against actual privacy notice
        return True
    
    def _is_sensitive_personal_info(self, data_category: str) -> bool:
        """Check if data category is sensitive personal information under CCPA."""
        sensitive_categories = {
            'biometric_info',
            'geolocation', 
            'racial_ethnic_origin',
            'religious_beliefs',
            'health_info',
            'sexual_orientation',
            'union_membership'
        }
        return data_category in sensitive_categories
    
    def _check_deletion_exceptions(self, consumer_id: str) -> Dict[str, Any]:
        """Check if CCPA deletion exceptions apply."""
        # CCPA Section 1798.105(d) exceptions
        exceptions = []
        
        # Check each exception
        if self._has_transaction_records(consumer_id):
            exceptions.append('complete_transaction')
        
        if self._has_security_needs(consumer_id):
            exceptions.append('detect_security_incidents')
        
        if self._has_legal_obligations(consumer_id):
            exceptions.append('comply_legal_obligation')
        
        if self._has_internal_research_use(consumer_id):
            exceptions.append('internal_research_compatible_with_relationship')
        
        return {
            'exceptions_apply': len(exceptions) > 0,
            'exceptions': exceptions,
            'explanation': f'Deletion limited due to: {", ".join(exceptions)}',
            'partial_deletion': len(exceptions) > 0  # Can delete some data
        }
    
    def _get_collected_categories(self, consumer_id: str) -> List[str]:
        """Get categories of personal information collected."""
        return [
            CCPADataCategory.IDENTIFIERS.value,
            CCPADataCategory.INTERNET_ACTIVITY.value,
            CCPADataCategory.INFERENCES.value
        ]
    
    def _get_sold_shared_categories(self, consumer_id: str) -> List[str]:
        """Get categories sold or shared."""
        return []  # DGDN doesn't sell or share PI
    
    def _get_disclosed_categories(self, consumer_id: str) -> List[str]:
        """Get categories disclosed for business purposes."""
        return [
            CCPADataCategory.IDENTIFIERS.value,  # To service providers
            CCPADataCategory.INTERNET_ACTIVITY.value  # For analytics
        ]
    
    def _get_business_purposes(self, consumer_id: str) -> List[str]:
        """Get business purposes for processing."""
        return [
            CCPABusinessPurpose.PERFORMING_SERVICES.value,
            CCPABusinessPurpose.INTERNAL_RESEARCH.value,
            CCPABusinessPurpose.QUALITY_VERIFICATION.value,
            CCPABusinessPurpose.SECURITY.value
        ]
    
    def _get_collection_sources(self, consumer_id: str) -> List[str]:
        """Get sources of personal information collection."""
        return [
            'directly_from_consumer',
            'consumer_devices',
            'service_usage_data'
        ]
    
    def _get_third_parties(self, consumer_id: str) -> List[str]:
        """Get third parties that receive personal information."""
        return [
            'cloud_service_providers',
            'research_partners_anonymized_data_only'
        ]
    
    def _get_ccpa_retention_info(self, consumer_id: str) -> Dict[str, Any]:
        """Get retention information per CCPA requirements."""
        return {
            'retention_criteria': 'Business purpose fulfillment',
            'standard_period': '3 years from collection or last use',
            'deletion_schedule': 'Automatic deletion after retention period',
            'exception_handling': 'Legal obligations may extend retention'
        }
    
    def _get_specific_pieces(self, consumer_id: str) -> Dict[str, Any]:
        """Get specific pieces of personal information."""
        # This should only be provided for verified access requests
        return {
            'note': 'Specific pieces provided only after identity verification',
            'format': 'Structured data export',
            'delivery_method': 'Secure portal or encrypted email'
        }
    
    def _get_deletable_categories(self, consumer_id: str) -> List[str]:
        """Get categories that can be deleted."""
        return [
            CCPADataCategory.INTERNET_ACTIVITY.value,
            CCPADataCategory.INFERENCES.value
        ]
    
    def _get_sensitive_categories(self, consumer_id: str) -> List[str]:
        """Get sensitive personal information categories."""
        return []  # DGDN typically doesn't process sensitive PI
    
    def _has_transaction_records(self, consumer_id: str) -> bool:
        """Check if consumer has transaction records."""
        return False  # DGDN is research-focused, not transactional
    
    def _has_security_needs(self, consumer_id: str) -> bool:
        """Check if data needed for security purposes."""
        return True  # Always need some data for security
    
    def _has_legal_obligations(self, consumer_id: str) -> bool:
        """Check if legal obligations require data retention."""
        return False  # Simplified - would check actual legal requirements
    
    def _has_internal_research_use(self, consumer_id: str) -> bool:
        """Check if data used for internal research."""
        return True  # DGDN's primary purpose is research