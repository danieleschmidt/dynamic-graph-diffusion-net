"""Compliance and privacy features for DGDN library."""

from .gdpr import GDPRCompliance
from .ccpa import CCPACompliance  
from .pdpa import PDPACompliance
from .privacy_manager import (
    PrivacyManager, 
    DataCategory, 
    ProcessingPurpose, 
    PrivacyRegime
)
from .data_protection import DataProtectionManager

__all__ = [
    "GDPRCompliance", 
    "CCPACompliance", 
    "PDPACompliance",
    "PrivacyManager",
    "DataCategory",
    "ProcessingPurpose", 
    "PrivacyRegime",
    "DataProtectionManager"
]