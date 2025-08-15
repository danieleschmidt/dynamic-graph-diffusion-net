"""
Enhanced global compliance framework for DGDN supporting multiple regions and regulations.
"""

import json
import hashlib
import time
import logging
from typing import Dict, Any, List, Optional, Set, Callable
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import uuid

class ComplianceRegion(Enum):
    """Supported compliance regions."""
    EU = "eu"  # European Union (GDPR)
    US_CALIFORNIA = "us_ca"  # California (CCPA/CPRA)
    SINGAPORE = "sg"  # Singapore (PDPA)
    CANADA = "ca"  # Canada (PIPEDA)
    BRAZIL = "br"  # Brazil (LGPD)
    JAPAN = "jp"  # Japan (APPI)
    AUSTRALIA = "au"  # Australia (Privacy Act)
    UK = "uk"  # United Kingdom (UK GDPR)
    GLOBAL = "global"  # Global baseline

class DataCategory(Enum):
    """Data categories for compliance tracking."""
    PERSONAL_IDENTIFIABLE = "pii"
    SENSITIVE_PERSONAL = "sensitive"
    BIOMETRIC = "biometric"
    HEALTH = "health"
    FINANCIAL = "financial"
    BEHAVIORAL = "behavioral"
    TECHNICAL = "technical"
    ANONYMOUS = "anonymous"

class ProcessingPurpose(Enum):
    """Data processing purposes."""
    MODEL_TRAINING = "model_training"
    MODEL_INFERENCE = "model_inference"
    PERFORMANCE_OPTIMIZATION = "performance_optimization"
    RESEARCH = "research"
    ANALYTICS = "analytics"
    SECURITY = "security"
    COMPLIANCE = "compliance"

@dataclass
class DataProcessingRecord:
    """Record of data processing activity."""
    id: str
    timestamp: float
    region: ComplianceRegion
    data_categories: List[DataCategory]
    purposes: List[ProcessingPurpose]
    legal_basis: str
    data_subject_count: int
    retention_period: int  # days
    third_party_transfers: List[str]
    security_measures: List[str]
    consent_obtained: bool
    anonymized: bool
    
class GlobalComplianceManager:
    """Comprehensive global compliance management system."""
    
    def __init__(self, default_region: ComplianceRegion = ComplianceRegion.GLOBAL):
        self.default_region = default_region
        self.active_regions = {default_region}
        self.processing_records = []
        self.consent_records = {}
        self.data_retention_policies = {}
        self.audit_log = []
        self.logger = logging.getLogger(f'{__name__}.GlobalComplianceManager')
        
        # Initialize compliance frameworks
        self.compliance_frameworks = self._initialize_compliance_frameworks()
        self.data_minimization_rules = self._initialize_data_minimization()
        self.retention_policies = self._initialize_retention_policies()
        
    def _initialize_compliance_frameworks(self) -> Dict[ComplianceRegion, Dict[str, Any]]:
        """Initialize compliance frameworks for different regions."""
        return {
            ComplianceRegion.EU: {
                "name": "General Data Protection Regulation (GDPR)",
                "authority": "European Data Protection Authorities",
                "key_principles": [
                    "lawfulness", "fairness", "transparency",
                    "purpose_limitation", "data_minimization",
                    "accuracy", "storage_limitation",
                    "integrity_confidentiality", "accountability"
                ],
                "rights": [
                    "access", "rectification", "erasure", "restrict_processing",
                    "data_portability", "object", "not_subject_to_automated_decision"
                ],
                "lawful_bases": [
                    "consent", "contract", "legal_obligation",
                    "vital_interests", "public_task", "legitimate_interests"
                ],
                "penalties": {"max_fine": "20M EUR or 4% annual turnover"},
                "data_transfer_mechanisms": ["adequacy_decision", "safeguards", "derogations"],
                "breach_notification": {"authority": 72, "data_subject": "without_delay"},
                "dpo_required": ["public_authority", "large_scale_monitoring", "sensitive_data"]
            },
            ComplianceRegion.US_CALIFORNIA: {
                "name": "California Consumer Privacy Act (CCPA/CPRA)",
                "authority": "California Privacy Protection Agency",
                "key_principles": [
                    "transparency", "consumer_control", "data_minimization"
                ],
                "rights": [
                    "know", "delete", "opt_out", "non_discrimination",
                    "correct", "limit_use", "opt_out_sensitive"
                ],
                "categories": [
                    "identifiers", "personal_info", "protected_classifications",
                    "commercial_info", "biometric", "internet_activity",
                    "geolocation", "audio_visual", "professional_info",
                    "education_info", "inferences"
                ],
                "penalties": {"max_fine": "7500 USD per violation"},
                "verification_required": True,
                "sale_opt_out": True
            },
            ComplianceRegion.SINGAPORE: {
                "name": "Personal Data Protection Act (PDPA)",
                "authority": "Personal Data Protection Commission",
                "key_principles": [
                    "consent", "purpose_limitation", "notification",
                    "access_correction", "accuracy", "protection",
                    "retention_limitation", "transfer_limitation"
                ],
                "penalties": {"max_fine": "1M SGD"},
                "consent_requirements": ["clear", "unambiguous", "informed"],
                "breach_notification": {"authority": "as_soon_as_practicable"}
            },
            ComplianceRegion.BRAZIL: {
                "name": "Lei Geral de Proteção de Dados (LGPD)",
                "authority": "Autoridade Nacional de Proteção de Dados",
                "key_principles": [
                    "purpose", "adequacy", "necessity", "free_access",
                    "data_quality", "transparency", "security",
                    "prevention", "non_discrimination", "accountability"
                ],
                "legal_bases": [
                    "consent", "legal_obligation", "public_administration",
                    "research", "contract", "judicial_process",
                    "life_protection", "health_protection", "legitimate_interest",
                    "credit_protection"
                ],
                "penalties": {"max_fine": "50M BRL or 2% revenue"}
            }
        }
    
    def _initialize_data_minimization(self) -> Dict[ProcessingPurpose, Dict[str, Any]]:
        """Initialize data minimization rules."""
        return {
            ProcessingPurpose.MODEL_TRAINING: {
                "allowed_categories": [
                    DataCategory.TECHNICAL, DataCategory.BEHAVIORAL,
                    DataCategory.ANONYMOUS
                ],
                "forbidden_categories": [
                    DataCategory.PERSONAL_IDENTIFIABLE,
                    DataCategory.SENSITIVE_PERSONAL,
                    DataCategory.BIOMETRIC, DataCategory.HEALTH
                ],
                "anonymization_required": True,
                "aggregation_threshold": 1000
            },
            ProcessingPurpose.MODEL_INFERENCE: {
                "allowed_categories": [
                    DataCategory.TECHNICAL, DataCategory.BEHAVIORAL,
                    DataCategory.ANONYMOUS
                ],
                "forbidden_categories": [
                    DataCategory.PERSONAL_IDENTIFIABLE,
                    DataCategory.SENSITIVE_PERSONAL
                ],
                "anonymization_required": True,
                "retention_limit": 30  # days
            },
            ProcessingPurpose.ANALYTICS: {
                "allowed_categories": [
                    DataCategory.TECHNICAL, DataCategory.BEHAVIORAL,
                    DataCategory.ANONYMOUS
                ],
                "anonymization_required": True,
                "aggregation_threshold": 100
            }
        }
    
    def _initialize_retention_policies(self) -> Dict[ComplianceRegion, Dict[DataCategory, int]]:
        """Initialize data retention policies by region and data category."""
        return {
            ComplianceRegion.EU: {
                DataCategory.PERSONAL_IDENTIFIABLE: 365,  # 1 year
                DataCategory.SENSITIVE_PERSONAL: 90,      # 3 months
                DataCategory.TECHNICAL: 1095,             # 3 years
                DataCategory.ANONYMOUS: -1                # No limit
            },
            ComplianceRegion.US_CALIFORNIA: {
                DataCategory.PERSONAL_IDENTIFIABLE: 730,  # 2 years
                DataCategory.SENSITIVE_PERSONAL: 365,     # 1 year
                DataCategory.TECHNICAL: 1095,             # 3 years
                DataCategory.ANONYMOUS: -1                # No limit
            },
            ComplianceRegion.SINGAPORE: {
                DataCategory.PERSONAL_IDENTIFIABLE: 365,  # 1 year
                DataCategory.SENSITIVE_PERSONAL: 180,     # 6 months
                DataCategory.TECHNICAL: 730,              # 2 years
                DataCategory.ANONYMOUS: -1                # No limit
            }
        }
    
    def register_processing_activity(self,
                                   region: ComplianceRegion,
                                   data_categories: List[DataCategory],
                                   purposes: List[ProcessingPurpose],
                                   legal_basis: str,
                                   data_subject_count: int,
                                   retention_period: Optional[int] = None,
                                   third_party_transfers: Optional[List[str]] = None,
                                   security_measures: Optional[List[str]] = None,
                                   consent_obtained: bool = False,
                                   anonymized: bool = False) -> str:
        """Register a data processing activity."""
        
        # Validate data minimization
        self._validate_data_minimization(data_categories, purposes)
        
        # Calculate retention period if not provided
        if retention_period is None:
            retention_period = self._calculate_retention_period(region, data_categories)
        
        # Create processing record
        record = DataProcessingRecord(
            id=str(uuid.uuid4()),
            timestamp=time.time(),
            region=region,
            data_categories=data_categories,
            purposes=purposes,
            legal_basis=legal_basis,
            data_subject_count=data_subject_count,
            retention_period=retention_period,
            third_party_transfers=third_party_transfers or [],
            security_measures=security_measures or [],
            consent_obtained=consent_obtained,
            anonymized=anonymized
        )
        
        self.processing_records.append(record)
        
        # Log the activity
        self._log_audit_event("processing_registered", {
            "record_id": record.id,
            "region": region.value,
            "purposes": [p.value for p in purposes],
            "data_categories": [c.value for c in data_categories]
        })
        
        self.logger.info(f"Registered processing activity {record.id} for region {region.value}")
        
        return record.id
    
    def _validate_data_minimization(self, 
                                   data_categories: List[DataCategory],
                                   purposes: List[ProcessingPurpose]) -> None:
        """Validate data minimization principles."""
        for purpose in purposes:
            if purpose in self.data_minimization_rules:
                rules = self.data_minimization_rules[purpose]
                
                # Check forbidden categories
                forbidden = set(rules.get("forbidden_categories", [])) & set(data_categories)
                if forbidden:
                    raise ValueError(
                        f"Data categories {forbidden} are forbidden for purpose {purpose.value}"
                    )
                
                # Check allowed categories
                allowed = set(rules.get("allowed_categories", []))
                if allowed and not set(data_categories).issubset(allowed):
                    disallowed = set(data_categories) - allowed
                    raise ValueError(
                        f"Data categories {disallowed} are not allowed for purpose {purpose.value}"
                    )
    
    def _calculate_retention_period(self, 
                                   region: ComplianceRegion,
                                   data_categories: List[DataCategory]) -> int:
        """Calculate retention period based on region and data categories."""
        if region not in self.retention_policies:
            region = ComplianceRegion.GLOBAL
        
        policies = self.retention_policies.get(region, {})
        
        # Use the shortest retention period among all categories
        periods = []
        for category in data_categories:
            period = policies.get(category, 365)  # Default 1 year
            if period > 0:  # Ignore unlimited (-1)
                periods.append(period)
        
        return min(periods) if periods else 365
    
    def obtain_consent(self, 
                      data_subject_id: str,
                      purposes: List[ProcessingPurpose],
                      region: ComplianceRegion,
                      consent_text: str,
                      granular_consent: Optional[Dict[str, bool]] = None) -> str:
        """Record user consent."""
        consent_id = str(uuid.uuid4())
        
        consent_record = {
            "id": consent_id,
            "data_subject_id": data_subject_id,
            "timestamp": time.time(),
            "region": region.value,
            "purposes": [p.value for p in purposes],
            "consent_text": consent_text,
            "granular_consent": granular_consent or {},
            "withdrawn": False,
            "withdrawal_timestamp": None
        }
        
        self.consent_records[consent_id] = consent_record
        
        self._log_audit_event("consent_obtained", {
            "consent_id": consent_id,
            "data_subject_id": data_subject_id,
            "purposes": [p.value for p in purposes]
        })
        
        return consent_id
    
    def withdraw_consent(self, consent_id: str) -> bool:
        """Withdraw user consent."""
        if consent_id in self.consent_records:
            self.consent_records[consent_id]["withdrawn"] = True
            self.consent_records[consent_id]["withdrawal_timestamp"] = time.time()
            
            self._log_audit_event("consent_withdrawn", {
                "consent_id": consent_id
            })
            
            return True
        return False
    
    def anonymize_data(self, data: Dict[str, Any], 
                      method: str = "k_anonymity",
                      k: int = 5) -> Dict[str, Any]:
        """Anonymize data according to compliance requirements."""
        if method == "k_anonymity":
            return self._apply_k_anonymity(data, k)
        elif method == "differential_privacy":
            return self._apply_differential_privacy(data)
        elif method == "generalization":
            return self._apply_generalization(data)
        else:
            raise ValueError(f"Unknown anonymization method: {method}")
    
    def _apply_k_anonymity(self, data: Dict[str, Any], k: int) -> Dict[str, Any]:
        """Apply k-anonymity to data (simplified implementation)."""
        # This is a simplified implementation
        # In practice, you'd use specialized anonymization libraries
        
        anonymized_data = data.copy()
        
        # Remove direct identifiers
        identifiers = ["id", "email", "phone", "ssn", "name"]
        for identifier in identifiers:
            if identifier in anonymized_data:
                anonymized_data[identifier] = self._hash_value(str(anonymized_data[identifier]))
        
        # Generalize quasi-identifiers
        if "age" in anonymized_data:
            age = anonymized_data["age"]
            anonymized_data["age_group"] = f"{(age // 10) * 10}-{(age // 10) * 10 + 9}"
            del anonymized_data["age"]
        
        if "zipcode" in anonymized_data:
            zipcode = str(anonymized_data["zipcode"])
            anonymized_data["region"] = zipcode[:3] + "XX"
            del anonymized_data["zipcode"]
        
        return anonymized_data
    
    def _apply_differential_privacy(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply differential privacy (simplified implementation)."""
        # Simplified implementation - add noise to numerical values
        import random
        
        anonymized_data = data.copy()
        
        for key, value in anonymized_data.items():
            if isinstance(value, (int, float)):
                # Add Laplace noise
                noise = random.laplace(0, 1.0)  # Scale parameter of 1.0
                anonymized_data[key] = value + noise
        
        return anonymized_data
    
    def _apply_generalization(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply data generalization."""
        anonymized_data = data.copy()
        
        # Remove or generalize specific fields
        if "timestamp" in anonymized_data:
            # Generalize to date only
            import datetime
            ts = anonymized_data["timestamp"]
            if isinstance(ts, (int, float)):
                dt = datetime.datetime.fromtimestamp(ts)
                anonymized_data["date"] = dt.strftime("%Y-%m-%d")
                del anonymized_data["timestamp"]
        
        return anonymized_data
    
    def _hash_value(self, value: str) -> str:
        """Hash a value for anonymization."""
        return hashlib.sha256(value.encode()).hexdigest()[:8]
    
    def check_cross_border_transfer_compliance(self, 
                                             source_region: ComplianceRegion,
                                             target_region: ComplianceRegion,
                                             data_categories: List[DataCategory]) -> Dict[str, Any]:
        """Check compliance for cross-border data transfers."""
        compliance_result = {
            "allowed": False,
            "requirements": [],
            "risks": [],
            "safeguards_needed": []
        }
        
        # EU transfers
        if source_region == ComplianceRegion.EU:
            if target_region in [ComplianceRegion.EU, ComplianceRegion.UK]:
                compliance_result["allowed"] = True
            else:
                # Requires adequacy decision or appropriate safeguards
                compliance_result["requirements"].append("adequacy_decision_or_safeguards")
                compliance_result["safeguards_needed"].extend([
                    "standard_contractual_clauses",
                    "binding_corporate_rules",
                    "certification_schemes"
                ])
        
        # California transfers
        elif source_region == ComplianceRegion.US_CALIFORNIA:
            # CCPA doesn't restrict transfers but requires disclosure
            compliance_result["allowed"] = True
            compliance_result["requirements"].append("disclosure_to_consumers")
        
        # Add risk assessment
        sensitive_categories = {
            DataCategory.SENSITIVE_PERSONAL,
            DataCategory.BIOMETRIC,
            DataCategory.HEALTH,
            DataCategory.FINANCIAL
        }
        
        if any(cat in sensitive_categories for cat in data_categories):
            compliance_result["risks"].append("sensitive_data_transfer")
            compliance_result["safeguards_needed"].append("enhanced_protection_measures")
        
        return compliance_result
    
    def generate_privacy_impact_assessment(self, 
                                         processing_purposes: List[ProcessingPurpose],
                                         data_categories: List[DataCategory],
                                         region: ComplianceRegion) -> Dict[str, Any]:
        """Generate a Privacy Impact Assessment (PIA)."""
        pia = {
            "assessment_id": str(uuid.uuid4()),
            "timestamp": time.time(),
            "region": region.value,
            "processing_purposes": [p.value for p in processing_purposes],
            "data_categories": [c.value for c in data_categories],
            "risk_level": "low",
            "risks_identified": [],
            "mitigation_measures": [],
            "recommendation": "proceed"
        }
        
        # Risk assessment
        high_risk_purposes = {
            ProcessingPurpose.RESEARCH,
            ProcessingPurpose.ANALYTICS
        }
        
        sensitive_categories = {
            DataCategory.SENSITIVE_PERSONAL,
            DataCategory.BIOMETRIC,
            DataCategory.HEALTH,
            DataCategory.FINANCIAL
        }
        
        # Determine risk level
        if any(purpose in high_risk_purposes for purpose in processing_purposes):
            pia["risk_level"] = "medium"
            pia["risks_identified"].append("research_or_analytics_processing")
        
        if any(cat in sensitive_categories for cat in data_categories):
            pia["risk_level"] = "high"
            pia["risks_identified"].append("sensitive_data_processing")
        
        # Add mitigation measures based on risks
        if pia["risk_level"] in ["medium", "high"]:
            pia["mitigation_measures"].extend([
                "implement_data_minimization",
                "use_anonymization_techniques",
                "implement_access_controls",
                "conduct_regular_audits"
            ])
        
        if pia["risk_level"] == "high":
            pia["mitigation_measures"].extend([
                "obtain_explicit_consent",
                "implement_encryption",
                "conduct_dpia",
                "designate_dpo"
            ])
            pia["recommendation"] = "proceed_with_caution"
        
        return pia
    
    def _log_audit_event(self, event_type: str, details: Dict[str, Any]) -> None:
        """Log an audit event."""
        audit_entry = {
            "id": str(uuid.uuid4()),
            "timestamp": time.time(),
            "event_type": event_type,
            "details": details
        }
        
        self.audit_log.append(audit_entry)
        
        # Keep only last 10000 entries
        if len(self.audit_log) > 10000:
            self.audit_log = self.audit_log[-10000:]
    
    def get_compliance_status(self) -> Dict[str, Any]:
        """Get overall compliance status."""
        return {
            "active_regions": [r.value for r in self.active_regions],
            "total_processing_activities": len(self.processing_records),
            "total_consent_records": len(self.consent_records),
            "audit_events": len(self.audit_log),
            "compliance_frameworks": list(self.compliance_frameworks.keys()),
            "data_retention_active": True,
            "anonymization_enabled": True
        }
    
    def export_compliance_report(self, region: Optional[ComplianceRegion] = None) -> Dict[str, Any]:
        """Export comprehensive compliance report."""
        filtered_records = self.processing_records
        if region:
            filtered_records = [r for r in self.processing_records if r.region == region]
        
        return {
            "report_id": str(uuid.uuid4()),
            "generation_timestamp": time.time(),
            "region_filter": region.value if region else "all",
            "processing_activities": [asdict(r) for r in filtered_records],
            "consent_summary": {
                "total_consents": len(self.consent_records),
                "active_consents": len([c for c in self.consent_records.values() if not c["withdrawn"]]),
                "withdrawn_consents": len([c for c in self.consent_records.values() if c["withdrawn"]])
            },
            "compliance_status": self.get_compliance_status(),
            "audit_trail": self.audit_log[-100:]  # Last 100 events
        }