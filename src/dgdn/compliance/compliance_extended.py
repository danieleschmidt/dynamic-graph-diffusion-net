"""Extended compliance framework for global DGDN deployment."""

import json
import hashlib
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set
from enum import Enum
import logging


class ComplianceRegion(Enum):
    """Supported compliance regions."""
    EU = "eu"           # European Union (GDPR)
    US = "us"           # United States (CCPA, etc.)
    UK = "uk"           # United Kingdom (UK GDPR)
    CANADA = "ca"       # Canada (PIPEDA)
    SINGAPORE = "sg"    # Singapore (PDPA)
    AUSTRALIA = "au"    # Australia (Privacy Act)
    BRAZIL = "br"       # Brazil (LGPD)
    JAPAN = "jp"        # Japan (APPI)
    SOUTH_KOREA = "kr"  # South Korea (PIPA)
    CHINA = "cn"        # China (PIPL)


class DataCategory(Enum):
    """Categories of data for compliance classification."""
    PERSONAL_IDENTIFIABLE = "pii"
    SENSITIVE_PERSONAL = "sensitive_pii"
    BIOMETRIC = "biometric"
    FINANCIAL = "financial"
    HEALTH = "health"
    BEHAVIORAL = "behavioral"
    TECHNICAL = "technical"
    STATISTICAL = "statistical"


class ProcessingPurpose(Enum):
    """Lawful purposes for data processing."""
    RESEARCH = "research"
    COMMERCIAL = "commercial"
    SECURITY = "security"
    ANALYTICS = "analytics"
    MODEL_TRAINING = "model_training"
    PERSONALIZATION = "personalization"
    FRAUD_DETECTION = "fraud_detection"
    QUALITY_ASSURANCE = "quality_assurance"


class GlobalComplianceFramework:
    """Comprehensive global compliance framework for DGDN."""
    
    def __init__(self, default_region: ComplianceRegion = ComplianceRegion.EU):
        self.default_region = default_region
        self.logger = logging.getLogger('DGDN.GlobalCompliance')
        
        # Compliance configurations by region
        self.region_configs = self._initialize_region_configs()
        
        # Data processing records
        self.processing_records = []
        
        # Consent management
        self.consent_records = {}
        
        # Audit trail
        self.audit_trail = []
        
    def _initialize_region_configs(self) -> Dict[ComplianceRegion, Dict]:
        """Initialize region-specific compliance configurations."""
        return {
            ComplianceRegion.EU: {
                'regulation': 'GDPR',
                'data_retention_months': 24,
                'consent_required': True,
                'right_to_deletion': True,
                'right_to_portability': True,
                'data_protection_officer_required': True,
                'cross_border_restrictions': True,
                'pseudonymization_required': True,
                'encryption_required': True,
                'breach_notification_hours': 72,
                'lawful_bases': ['consent', 'legitimate_interest', 'vital_interest', 'public_task', 'contract']
            },
            ComplianceRegion.US: {
                'regulation': 'CCPA/CPRA',
                'data_retention_months': 12,
                'consent_required': False,  # Opt-out model
                'right_to_deletion': True,
                'right_to_portability': True,
                'data_protection_officer_required': False,
                'cross_border_restrictions': False,
                'pseudonymization_required': False,
                'encryption_required': True,
                'breach_notification_hours': 72,
                'lawful_bases': ['business_purpose', 'commercial_purpose']
            },
            ComplianceRegion.UK: {
                'regulation': 'UK GDPR',
                'data_retention_months': 24,
                'consent_required': True,
                'right_to_deletion': True,
                'right_to_portability': True,
                'data_protection_officer_required': True,
                'cross_border_restrictions': True,
                'pseudonymization_required': True,
                'encryption_required': True,
                'breach_notification_hours': 72,
                'lawful_bases': ['consent', 'legitimate_interest', 'vital_interest', 'public_task', 'contract']
            },
            ComplianceRegion.SINGAPORE: {
                'regulation': 'PDPA',
                'data_retention_months': 12,
                'consent_required': True,
                'right_to_deletion': False,
                'right_to_portability': False,
                'data_protection_officer_required': True,
                'cross_border_restrictions': True,
                'pseudonymization_required': False,
                'encryption_required': True,
                'breach_notification_hours': 72,
                'lawful_bases': ['consent', 'legitimate_interest']
            },
            ComplianceRegion.CANADA: {
                'regulation': 'PIPEDA',
                'data_retention_months': 12,
                'consent_required': True,
                'right_to_deletion': False,
                'right_to_portability': True,
                'data_protection_officer_required': False,
                'cross_border_restrictions': True,
                'pseudonymization_required': False,
                'encryption_required': True,
                'breach_notification_hours': 72,
                'lawful_bases': ['consent', 'legitimate_interest']
            },
            ComplianceRegion.BRAZIL: {
                'regulation': 'LGPD',
                'data_retention_months': 12,
                'consent_required': True,
                'right_to_deletion': True,
                'right_to_portability': True,
                'data_protection_officer_required': True,
                'cross_border_restrictions': True,
                'pseudonymization_required': True,
                'encryption_required': True,
                'breach_notification_hours': 72,
                'lawful_bases': ['consent', 'legitimate_interest', 'vital_interest', 'public_interest']
            }
        }
        
    def register_data_processing(
        self,
        purpose: ProcessingPurpose,
        data_categories: List[DataCategory],
        regions: List[ComplianceRegion],
        lawful_basis: str,
        retention_period_months: int = None,
        data_subjects_count: int = None
    ) -> str:
        """Register a data processing activity."""
        
        # Generate processing ID
        processing_id = hashlib.sha256(
            f"{purpose.value}_{time.time()}".encode()
        ).hexdigest()[:16]
        
        # Validate lawful basis for each region
        for region in regions:
            config = self.region_configs.get(region, {})
            valid_bases = config.get('lawful_bases', [])
            
            if lawful_basis not in valid_bases:
                raise ValueError(
                    f"Lawful basis '{lawful_basis}' not valid for region {region.value}. "
                    f"Valid bases: {valid_bases}"
                )
        
        # Create processing record
        processing_record = {
            'processing_id': processing_id,
            'purpose': purpose.value,
            'data_categories': [cat.value for cat in data_categories],
            'regions': [region.value for region in regions],
            'lawful_basis': lawful_basis,
            'retention_period_months': retention_period_months,
            'data_subjects_count': data_subjects_count,
            'created_at': datetime.utcnow().isoformat(),
            'status': 'active'
        }
        
        self.processing_records.append(processing_record)
        
        # Log to audit trail
        self._audit_log('data_processing_registered', {
            'processing_id': processing_id,
            'purpose': purpose.value,
            'regions': [r.value for r in regions]
        })
        
        self.logger.info(f"Data processing registered: {processing_id}")
        return processing_id
        
    def record_consent(
        self,
        data_subject_id: str,
        processing_purposes: List[ProcessingPurpose],
        consent_given: bool,
        consent_method: str = 'explicit',
        region: ComplianceRegion = None
    ) -> str:
        """Record consent for data processing."""
        
        region = region or self.default_region
        config = self.region_configs.get(region, {})
        
        # Check if consent is required for this region
        if not config.get('consent_required', False) and consent_method == 'explicit':
            self.logger.warning(f"Explicit consent may not be required for region {region.value}")
        
        consent_id = hashlib.sha256(
            f"{data_subject_id}_{time.time()}".encode()
        ).hexdigest()[:16]
        
        consent_record = {
            'consent_id': consent_id,
            'data_subject_id': data_subject_id,
            'processing_purposes': [p.value for p in processing_purposes],
            'consent_given': consent_given,
            'consent_method': consent_method,
            'region': region.value,
            'timestamp': datetime.utcnow().isoformat(),
            'ip_address': None,  # Would be populated in real implementation
            'user_agent': None   # Would be populated in real implementation
        }
        
        if data_subject_id not in self.consent_records:
            self.consent_records[data_subject_id] = []
        
        self.consent_records[data_subject_id].append(consent_record)
        
        # Log to audit trail
        self._audit_log('consent_recorded', {
            'consent_id': consent_id,
            'data_subject_id': data_subject_id,
            'consent_given': consent_given,
            'region': region.value
        })
        
        return consent_id
        
    def handle_deletion_request(
        self,
        data_subject_id: str,
        region: ComplianceRegion = None,
        verification_method: str = 'email'
    ) -> Dict[str, Any]:
        """Handle right to deletion request (GDPR Article 17, CCPA, etc.)."""
        
        region = region or self.default_region
        config = self.region_configs.get(region, {})
        
        if not config.get('right_to_deletion', False):
            return {
                'status': 'not_supported',
                'message': f"Right to deletion not supported in region {region.value}"
            }
        
        # Verify identity (simplified)
        if not self._verify_data_subject_identity(data_subject_id, verification_method):
            return {
                'status': 'verification_failed',
                'message': 'Data subject identity verification failed'
            }
        
        # Check for legitimate reasons to refuse deletion
        refusal_reasons = self._check_deletion_refusal_reasons(data_subject_id, region)
        
        if refusal_reasons:
            return {
                'status': 'deletion_refused',
                'reasons': refusal_reasons,
                'message': 'Deletion request refused due to legitimate interests'
            }
        
        # Generate deletion request ID
        request_id = hashlib.sha256(
            f"deletion_{data_subject_id}_{time.time()}".encode()
        ).hexdigest()[:16]
        
        # Schedule data deletion (in practice, this would trigger actual deletion)
        deletion_scheduled_time = datetime.utcnow() + timedelta(days=30)  # Grace period
        
        # Log to audit trail
        self._audit_log('deletion_request_received', {
            'request_id': request_id,
            'data_subject_id': data_subject_id,
            'region': region.value,
            'scheduled_deletion': deletion_scheduled_time.isoformat()
        })
        
        return {
            'status': 'accepted',
            'request_id': request_id,
            'scheduled_deletion': deletion_scheduled_time.isoformat(),
            'message': 'Deletion request accepted and scheduled'
        }
        
    def handle_portability_request(
        self,
        data_subject_id: str,
        export_format: str = 'json',
        region: ComplianceRegion = None
    ) -> Dict[str, Any]:
        """Handle right to data portability request (GDPR Article 20)."""
        
        region = region or self.default_region
        config = self.region_configs.get(region, {})
        
        if not config.get('right_to_portability', False):
            return {
                'status': 'not_supported',
                'message': f"Right to portability not supported in region {region.value}"
            }
        
        # Verify identity
        if not self._verify_data_subject_identity(data_subject_id, 'email'):
            return {
                'status': 'verification_failed',
                'message': 'Data subject identity verification failed'
            }
        
        # Generate export (simplified - would fetch actual data)
        export_data = {
            'data_subject_id': data_subject_id,
            'export_timestamp': datetime.utcnow().isoformat(),
            'data_categories': [],  # Would populate with actual data
            'processing_history': [],  # Would populate with processing records
            'consent_history': self.consent_records.get(data_subject_id, [])
        }
        
        # Generate export ID
        export_id = hashlib.sha256(
            f"export_{data_subject_id}_{time.time()}".encode()
        ).hexdigest()[:16]
        
        # Log to audit trail
        self._audit_log('portability_request_fulfilled', {
            'export_id': export_id,
            'data_subject_id': data_subject_id,
            'export_format': export_format,
            'region': region.value
        })
        
        return {
            'status': 'completed',
            'export_id': export_id,
            'export_data': export_data,
            'format': export_format
        }
        
    def assess_cross_border_transfer(
        self,
        source_region: ComplianceRegion,
        destination_region: ComplianceRegion,
        data_categories: List[DataCategory],
        transfer_mechanism: str = None
    ) -> Dict[str, Any]:
        """Assess compliance for cross-border data transfers."""
        
        source_config = self.region_configs.get(source_region, {})
        
        # Check if source region has cross-border restrictions
        if not source_config.get('cross_border_restrictions', False):
            return {
                'allowed': True,
                'mechanism_required': False,
                'message': 'No cross-border restrictions in source region'
            }
        
        # Determine if destination is adequate
        adequate_destinations = self._get_adequate_destinations(source_region)
        
        if destination_region in adequate_destinations:
            return {
                'allowed': True,
                'mechanism_required': False,
                'adequacy_decision': True,
                'message': f'Adequacy decision exists for transfer to {destination_region.value}'
            }
        
        # Check for appropriate safeguards
        required_mechanisms = self._get_required_transfer_mechanisms(
            source_region, destination_region, data_categories
        )
        
        if transfer_mechanism in required_mechanisms:
            return {
                'allowed': True,
                'mechanism_required': True,
                'mechanism_used': transfer_mechanism,
                'message': f'Transfer allowed with {transfer_mechanism}'
            }
        
        return {
            'allowed': False,
            'required_mechanisms': required_mechanisms,
            'message': 'Transfer not allowed without appropriate safeguards'
        }
        
    def generate_compliance_report(
        self,
        region: ComplianceRegion = None,
        start_date: datetime = None,
        end_date: datetime = None
    ) -> Dict[str, Any]:
        """Generate comprehensive compliance report."""
        
        region = region or self.default_region
        end_date = end_date or datetime.utcnow()
        start_date = start_date or (end_date - timedelta(days=30))
        
        # Filter records by date range
        filtered_processing = [
            record for record in self.processing_records
            if start_date <= datetime.fromisoformat(record['created_at']) <= end_date
        ]
        
        filtered_audit = [
            entry for entry in self.audit_trail
            if start_date <= datetime.fromisoformat(entry['timestamp']) <= end_date
        ]
        
        # Compliance statistics
        stats = {
            'processing_activities': len(filtered_processing),
            'consent_records': sum(len(consents) for consents in self.consent_records.values()),
            'audit_entries': len(filtered_audit),
            'data_subjects': len(self.consent_records),
        }
        
        # Risk assessment
        risks = self._assess_compliance_risks(region, filtered_processing)
        
        # Recommendations
        recommendations = self._generate_compliance_recommendations(region, stats, risks)
        
        report = {
            'region': region.value,
            'regulation': self.region_configs[region]['regulation'],
            'reporting_period': {
                'start': start_date.isoformat(),
                'end': end_date.isoformat()
            },
            'statistics': stats,
            'risks': risks,
            'recommendations': recommendations,
            'processing_activities': filtered_processing,
            'generated_at': datetime.utcnow().isoformat()
        }
        
        # Log report generation
        self._audit_log('compliance_report_generated', {
            'region': region.value,
            'report_period_days': (end_date - start_date).days
        })
        
        return report
        
    def _verify_data_subject_identity(self, data_subject_id: str, method: str) -> bool:
        """Verify data subject identity (simplified implementation)."""
        # In practice, this would involve robust identity verification
        return True
        
    def _check_deletion_refusal_reasons(
        self,
        data_subject_id: str,
        region: ComplianceRegion
    ) -> List[str]:
        """Check for legitimate reasons to refuse deletion."""
        reasons = []
        
        # Example reasons (would be more sophisticated in practice)
        # - Legal obligations
        # - Public interest
        # - Freedom of expression and information
        # - Archiving purposes in the public interest
        
        return reasons
        
    def _get_adequate_destinations(self, source_region: ComplianceRegion) -> List[ComplianceRegion]:
        """Get list of adequate destination regions for data transfer."""
        adequacy_map = {
            ComplianceRegion.EU: [ComplianceRegion.UK, ComplianceRegion.CANADA],
            ComplianceRegion.UK: [ComplianceRegion.EU],
            ComplianceRegion.US: [],  # Generally requires specific mechanisms
        }
        
        return adequacy_map.get(source_region, [])
        
    def _get_required_transfer_mechanisms(
        self,
        source: ComplianceRegion,
        destination: ComplianceRegion,
        data_categories: List[DataCategory]
    ) -> List[str]:
        """Get required transfer mechanisms for cross-border data transfer."""
        
        mechanisms = []
        
        # Standard Contractual Clauses (SCCs)
        mechanisms.append('standard_contractual_clauses')
        
        # Binding Corporate Rules (BCRs) for intra-group transfers
        mechanisms.append('binding_corporate_rules')
        
        # Certification schemes
        mechanisms.append('certification_scheme')
        
        # Codes of conduct
        mechanisms.append('approved_code_of_conduct')
        
        # For sensitive data, additional mechanisms may be required
        sensitive_categories = [DataCategory.BIOMETRIC, DataCategory.HEALTH, DataCategory.FINANCIAL]
        
        if any(cat in sensitive_categories for cat in data_categories):
            mechanisms.append('additional_safeguards_required')
        
        return mechanisms
        
    def _assess_compliance_risks(
        self,
        region: ComplianceRegion,
        processing_records: List[Dict]
    ) -> List[Dict[str, Any]]:
        """Assess compliance risks."""
        risks = []
        
        config = self.region_configs.get(region, {})
        
        # Check for missing lawful basis
        for record in processing_records:
            if not record.get('lawful_basis'):
                risks.append({
                    'type': 'missing_lawful_basis',
                    'severity': 'high',
                    'processing_id': record['processing_id'],
                    'description': 'Processing activity lacks lawful basis'
                })
        
        # Check for excessive retention periods
        max_retention = config.get('data_retention_months', 12)
        for record in processing_records:
            retention = record.get('retention_period_months', 0)
            if retention > max_retention:
                risks.append({
                    'type': 'excessive_retention',
                    'severity': 'medium',
                    'processing_id': record['processing_id'],
                    'description': f'Retention period ({retention} months) exceeds guideline ({max_retention} months)'
                })
        
        return risks
        
    def _generate_compliance_recommendations(
        self,
        region: ComplianceRegion,
        stats: Dict,
        risks: List[Dict]
    ) -> List[str]:
        """Generate compliance recommendations."""
        recommendations = []
        
        config = self.region_configs.get(region, {})
        
        # Data Protection Officer recommendation
        if config.get('data_protection_officer_required') and stats['data_subjects'] > 1000:
            recommendations.append(
                "Consider appointing a Data Protection Officer due to scale of processing"
            )
        
        # Encryption recommendation
        if config.get('encryption_required'):
            recommendations.append(
                "Ensure all personal data is encrypted both in transit and at rest"
            )
        
        # Risk-based recommendations
        high_risks = [r for r in risks if r['severity'] == 'high']
        if high_risks:
            recommendations.append(
                f"Address {len(high_risks)} high-severity compliance risks immediately"
            )
        
        # Regular audit recommendation
        recommendations.append(
            "Conduct regular compliance audits and update privacy policies"
        )
        
        return recommendations
        
    def _audit_log(self, event_type: str, details: Dict[str, Any]):
        """Add entry to audit trail."""
        audit_entry = {
            'event_type': event_type,
            'timestamp': datetime.utcnow().isoformat(),
            'details': details
        }
        
        self.audit_trail.append(audit_entry)
        
        # Keep audit trail size manageable
        if len(self.audit_trail) > 10000:
            self.audit_trail = self.audit_trail[-5000:]


class ComplianceValidator:
    """Validator for DGDN model compliance across regions."""
    
    def __init__(self, compliance_framework: GlobalComplianceFramework):
        self.framework = compliance_framework
        self.logger = logging.getLogger('DGDN.ComplianceValidator')
        
    def validate_model_deployment(
        self,
        model_metadata: Dict[str, Any],
        deployment_regions: List[ComplianceRegion],
        data_categories: List[DataCategory]
    ) -> Dict[str, Any]:
        """Validate model deployment compliance across regions."""
        
        validation_results = {
            'overall_compliant': True,
            'region_results': {},
            'blocking_issues': [],
            'warnings': [],
            'recommendations': []
        }
        
        for region in deployment_regions:
            region_result = self._validate_region_deployment(
                model_metadata, region, data_categories
            )
            validation_results['region_results'][region.value] = region_result
            
            if not region_result['compliant']:
                validation_results['overall_compliant'] = False
                validation_results['blocking_issues'].extend(region_result['issues'])
            
            validation_results['warnings'].extend(region_result['warnings'])
            validation_results['recommendations'].extend(region_result['recommendations'])
        
        return validation_results
        
    def _validate_region_deployment(
        self,
        model_metadata: Dict[str, Any],
        region: ComplianceRegion,
        data_categories: List[DataCategory]
    ) -> Dict[str, Any]:
        """Validate deployment for a specific region."""
        
        config = self.framework.region_configs.get(region, {})
        
        result = {
            'compliant': True,
            'issues': [],
            'warnings': [],
            'recommendations': []
        }
        
        # Check encryption requirements
        if config.get('encryption_required', False):
            if not model_metadata.get('encrypted', False):
                result['compliant'] = False
                result['issues'].append(
                    f"Model encryption required for region {region.value}"
                )
        
        # Check pseudonymization requirements
        if config.get('pseudonymization_required', False):
            if DataCategory.PERSONAL_IDENTIFIABLE in data_categories:
                if not model_metadata.get('pseudonymized_training_data', False):
                    result['warnings'].append(
                        f"Consider pseudonymizing training data for region {region.value}"
                    )
        
        # Check data retention compliance
        retention_months = model_metadata.get('training_data_retention_months')
        max_retention = config.get('data_retention_months', 12)
        
        if retention_months and retention_months > max_retention:
            result['warnings'].append(
                f"Training data retention ({retention_months} months) exceeds "
                f"recommended period ({max_retention} months) for region {region.value}"
            )
        
        # Add region-specific recommendations
        if config.get('data_protection_officer_required', False):
            result['recommendations'].append(
                f"Ensure Data Protection Officer oversight for region {region.value}"
            )
        
        return result


# Example usage and testing
def demonstrate_global_compliance():
    """Demonstrate global compliance framework capabilities."""
    
    print("🌍 Global Compliance Framework Demo")
    print("=" * 50)
    
    # Initialize compliance framework
    compliance = GlobalComplianceFramework(ComplianceRegion.EU)
    
    # Register data processing activity
    processing_id = compliance.register_data_processing(
        purpose=ProcessingPurpose.MODEL_TRAINING,
        data_categories=[DataCategory.BEHAVIORAL, DataCategory.TECHNICAL],
        regions=[ComplianceRegion.EU, ComplianceRegion.US],
        lawful_basis='legitimate_interest',
        retention_period_months=24,
        data_subjects_count=10000
    )
    
    print(f"✅ Data processing registered: {processing_id}")
    
    # Record consent
    consent_id = compliance.record_consent(
        data_subject_id='user123',
        processing_purposes=[ProcessingPurpose.MODEL_TRAINING],
        consent_given=True,
        region=ComplianceRegion.EU
    )
    
    print(f"✅ Consent recorded: {consent_id}")
    
    # Test cross-border transfer assessment
    transfer_result = compliance.assess_cross_border_transfer(
        source_region=ComplianceRegion.EU,
        destination_region=ComplianceRegion.US,
        data_categories=[DataCategory.BEHAVIORAL],
        transfer_mechanism='standard_contractual_clauses'
    )
    
    print(f"✅ Cross-border transfer: {'Allowed' if transfer_result['allowed'] else 'Blocked'}")
    
    # Generate compliance report
    report = compliance.generate_compliance_report(ComplianceRegion.EU)
    print(f"✅ Compliance report generated with {len(report['recommendations'])} recommendations")
    
    # Validate model deployment
    validator = ComplianceValidator(compliance)
    model_metadata = {
        'encrypted': True,
        'pseudonymized_training_data': True,
        'training_data_retention_months': 18
    }
    
    validation = validator.validate_model_deployment(
        model_metadata=model_metadata,
        deployment_regions=[ComplianceRegion.EU, ComplianceRegion.US],
        data_categories=[DataCategory.BEHAVIORAL, DataCategory.TECHNICAL]
    )
    
    print(f"✅ Model deployment validation: {'Compliant' if validation['overall_compliant'] else 'Non-compliant'}")
    print(f"   Issues: {len(validation['blocking_issues'])}")
    print(f"   Warnings: {len(validation['warnings'])}")
    print(f"   Recommendations: {len(validation['recommendations'])}")
    
    return compliance, validator


if __name__ == "__main__":
    demonstrate_global_compliance()