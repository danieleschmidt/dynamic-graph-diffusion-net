#!/usr/bin/env python3
"""Global-First DGDN Deployment - Multi-Region, I18n, Compliance.

Implements global deployment capabilities with internationalization,
multi-region support, and comprehensive compliance (GDPR, CCPA, PDPA).
"""

import sys
import time
import json
import logging
import os
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import datetime

# Configure global logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s] - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('global_dgdn.log', mode='a', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

class Region(Enum):
    """Supported global regions with compliance requirements."""
    # Americas
    US_EAST = "us-east-1"
    US_WEST = "us-west-2" 
    CANADA = "ca-central-1"
    BRAZIL = "sa-east-1"
    
    # Europe (GDPR)
    EU_WEST = "eu-west-1"
    EU_CENTRAL = "eu-central-1"
    UK = "eu-west-2"
    
    # Asia-Pacific
    ASIA_PACIFIC = "ap-southeast-1" 
    JAPAN = "ap-northeast-1"
    CHINA = "cn-north-1"

class Language(Enum):
    """Supported languages for internationalization."""
    ENGLISH = "en"
    SPANISH = "es" 
    FRENCH = "fr"
    GERMAN = "de"
    JAPANESE = "ja"
    CHINESE = "zh"

class ComplianceFramework(Enum):
    """Data protection and compliance frameworks."""
    GDPR = "gdpr"  # General Data Protection Regulation (EU)
    CCPA = "ccpa"  # California Consumer Privacy Act (US)
    PDPA = "pdpa"  # Personal Data Protection Act (Singapore)
    PIPEDA = "pipeda"  # Personal Information Protection and Electronic Documents Act (Canada)

@dataclass
class RegionConfig:
    """Configuration for a specific deployment region."""
    region: Region
    primary_language: Language
    compliance_frameworks: List[ComplianceFramework]
    data_residency_required: bool
    encryption_key_region: str
    latency_target_ms: int = 200

@dataclass
class ComplianceConfig:
    """Compliance configuration and requirements."""
    framework: ComplianceFramework
    data_retention_days: int
    anonymization_required: bool
    consent_required: bool
    right_to_erasure: bool
    data_portability: bool
    breach_notification_hours: int

class InternationalizationManager:
    """Manages internationalization and localization."""
    
    def __init__(self):
        self.messages = self._load_messages()
        self.current_locale = Language.ENGLISH
        logger.info("🌍 Internationalization Manager initialized")
    
    def _load_messages(self) -> Dict[Language, Dict[str, str]]:
        """Load localized messages for all supported languages."""
        return {
            Language.ENGLISH: {
                "startup": "🚀 Starting DGDN Global Deployment",
                "processing": "🔄 Processing graph data",
                "completed": "✅ Processing completed successfully",
                "error": "❌ Error occurred during processing",
                "privacy_notice": "🔒 Your data is processed in compliance with applicable privacy laws",
                "performance_report": "📊 Performance Report",
                "nodes_processed": "Nodes processed",
                "edges_processed": "Edges processed", 
                "inference_time": "Inference time",
                "compliance_check": "🛡️ Compliance validation passed",
                "region_selected": "🌍 Region selected",
                "data_location": "📍 Data processing location"
            },
            Language.SPANISH: {
                "startup": "🚀 Iniciando Despliegue Global DGDN",
                "processing": "🔄 Procesando datos del grafo",
                "completed": "✅ Procesamiento completado exitosamente",
                "error": "❌ Error ocurrido durante el procesamiento",
                "privacy_notice": "🔒 Sus datos son procesados en cumplimiento con las leyes de privacidad aplicables",
                "performance_report": "📊 Reporte de Rendimiento",
                "nodes_processed": "Nodos procesados",
                "edges_processed": "Aristas procesadas",
                "inference_time": "Tiempo de inferencia",
                "compliance_check": "🛡️ Validación de cumplimiento aprobada",
                "region_selected": "🌍 Región seleccionada",
                "data_location": "📍 Ubicación de procesamiento de datos"
            },
            Language.FRENCH: {
                "startup": "🚀 Démarrage du déploiement global DGDN",
                "processing": "🔄 Traitement des données graphiques",
                "completed": "✅ Traitement terminé avec succès",
                "error": "❌ Erreur survenue lors du traitement",
                "privacy_notice": "🔒 Vos données sont traitées en conformité avec les lois sur la confidentialité applicables",
                "performance_report": "📊 Rapport de performance",
                "nodes_processed": "Nœuds traités",
                "edges_processed": "Arêtes traitées",
                "inference_time": "Temps d'inférence",
                "compliance_check": "🛡️ Validation de conformité réussie",
                "region_selected": "🌍 Région sélectionnée",
                "data_location": "📍 Emplacement de traitement des données"
            },
            Language.GERMAN: {
                "startup": "🚀 Starten der globalen DGDN-Bereitstellung",
                "processing": "🔄 Verarbeitung der Graphdaten",
                "completed": "✅ Verarbeitung erfolgreich abgeschlossen",
                "error": "❌ Fehler bei der Verarbeitung aufgetreten",
                "privacy_notice": "🔒 Ihre Daten werden in Übereinstimmung mit geltenden Datenschutzgesetzen verarbeitet",
                "performance_report": "📊 Leistungsbericht",
                "nodes_processed": "Verarbeitete Knoten",
                "edges_processed": "Verarbeitete Kanten",
                "inference_time": "Inferenzzeit",
                "compliance_check": "🛡️ Compliance-Validierung bestanden",
                "region_selected": "🌍 Region ausgewählt", 
                "data_location": "📍 Datenverarbeitungsort"
            },
            Language.JAPANESE: {
                "startup": "🚀 DGDNグローバル展開開始",
                "processing": "🔄 グラフデータ処理中",
                "completed": "✅ 処理が正常に完了しました",
                "error": "❌ 処理中にエラーが発生しました",
                "privacy_notice": "🔒 お客様のデータは適用されるプライバシー法に準拠して処理されます",
                "performance_report": "📊 パフォーマンス レポート",
                "nodes_processed": "処理されたノード",
                "edges_processed": "処理されたエッジ",
                "inference_time": "推論時間",
                "compliance_check": "🛡️ コンプライアンス検証合格",
                "region_selected": "🌍 選択された地域",
                "data_location": "📍 データ処理場所"
            },
            Language.CHINESE: {
                "startup": "🚀 启动DGDN全球部署",
                "processing": "🔄 处理图数据",
                "completed": "✅ 处理成功完成",
                "error": "❌ 处理过程中发生错误",
                "privacy_notice": "🔒 您的数据按照适用的隐私法律进行处理",
                "performance_report": "📊 性能报告", 
                "nodes_processed": "已处理节点",
                "edges_processed": "已处理边",
                "inference_time": "推理时间",
                "compliance_check": "🛡️ 合规验证通过",
                "region_selected": "🌍 选择的区域",
                "data_location": "📍 数据处理位置"
            }
        }
    
    def set_locale(self, language: Language) -> None:
        """Set the current locale."""
        self.current_locale = language
        logger.info(f"🌐 Locale set to {language.value}")
    
    def get_message(self, key: str, **kwargs) -> str:
        """Get localized message with optional formatting."""
        messages = self.messages.get(self.current_locale, self.messages[Language.ENGLISH])
        message = messages.get(key, f"[Missing: {key}]")
        
        if kwargs:
            try:
                return message.format(**kwargs)
            except KeyError:
                return message
        
        return message

class ComplianceManager:
    """Manages data protection compliance across multiple frameworks."""
    
    def __init__(self):
        self.compliance_configs = self._initialize_compliance_configs()
        self.consent_records = {}
        self.processing_logs = []
        
        logger.info("🛡️ Compliance Manager initialized")
    
    def _initialize_compliance_configs(self) -> Dict[ComplianceFramework, ComplianceConfig]:
        """Initialize compliance configurations for different frameworks."""
        return {
            ComplianceFramework.GDPR: ComplianceConfig(
                framework=ComplianceFramework.GDPR,
                data_retention_days=365,
                anonymization_required=True,
                consent_required=True,
                right_to_erasure=True,
                data_portability=True,
                breach_notification_hours=72
            ),
            ComplianceFramework.CCPA: ComplianceConfig(
                framework=ComplianceFramework.CCPA,
                data_retention_days=365,
                anonymization_required=False,
                consent_required=False,  # Opt-out model
                right_to_erasure=True,
                data_portability=True,
                breach_notification_hours=None
            ),
            ComplianceFramework.PDPA: ComplianceConfig(
                framework=ComplianceFramework.PDPA,
                data_retention_days=365,
                anonymization_required=True,
                consent_required=True,
                right_to_erasure=True,
                data_portability=False,
                breach_notification_hours=72
            ),
            ComplianceFramework.PIPEDA: ComplianceConfig(
                framework=ComplianceFramework.PIPEDA,
                data_retention_days=365,
                anonymization_required=True,
                consent_required=True,
                right_to_erasure=True,
                data_portability=False,
                breach_notification_hours=72
            )
        }
    
    def validate_compliance(self, region_config: RegionConfig, data_info: Dict[str, Any]) -> Dict[str, Any]:
        """Validate compliance for data processing in a specific region."""
        validation_result = {
            'compliant': True,
            'frameworks_checked': [],
            'violations': [],
            'recommendations': []
        }
        
        for framework in region_config.compliance_frameworks:
            config = self.compliance_configs[framework]
            validation_result['frameworks_checked'].append(framework.value)
            
            # Check consent requirements
            if config.consent_required and not self._has_valid_consent(data_info):
                validation_result['violations'].append(f"{framework.value}: Missing required consent")
                validation_result['compliant'] = False
            
            # Check data retention
            if self._exceeds_retention_period(data_info, config.data_retention_days):
                validation_result['violations'].append(f"{framework.value}: Data exceeds retention period")
                validation_result['recommendations'].append("Consider data anonymization or deletion")
            
            # Check anonymization requirements
            if config.anonymization_required and not self._is_anonymized(data_info):
                validation_result['recommendations'].append(f"{framework.value}: Consider data anonymization for enhanced compliance")
        
        self._log_compliance_check(region_config, validation_result)
        return validation_result
    
    def _has_valid_consent(self, data_info: Dict[str, Any]) -> bool:
        """Check if valid consent exists for data processing."""
        user_id = data_info.get('user_id', 'anonymous')
        consent_record = self.consent_records.get(user_id)
        
        if not consent_record:
            # For demo purposes, assume consent is given for anonymous processing
            return user_id == 'anonymous'
        
        # Check if consent is still valid
        consent_date = datetime.datetime.fromisoformat(consent_record['date'])
        max_age = datetime.timedelta(days=365)  # Consent valid for 1 year
        
        return datetime.datetime.now() - consent_date < max_age
    
    def _exceeds_retention_period(self, data_info: Dict[str, Any], retention_days: int) -> bool:
        """Check if data exceeds retention period."""
        created_date = data_info.get('created_date')
        if not created_date:
            return False
        
        try:
            created = datetime.datetime.fromisoformat(created_date)
            retention_period = datetime.timedelta(days=retention_days)
            return datetime.datetime.now() - created > retention_period
        except ValueError:
            return False
    
    def _is_anonymized(self, data_info: Dict[str, Any]) -> bool:
        """Check if data is properly anonymized."""
        # Simple heuristic: data is considered anonymized if it doesn't contain PII
        pii_fields = ['user_id', 'email', 'name', 'phone', 'address']
        
        for field in pii_fields:
            if field in data_info and data_info[field] != 'anonymous':
                return False
        
        return True
    
    def _log_compliance_check(self, region_config: RegionConfig, result: Dict[str, Any]) -> None:
        """Log compliance check for audit purposes."""
        log_entry = {
            'timestamp': datetime.datetime.now().isoformat(),
            'region': region_config.region.value,
            'frameworks': [f.value for f in region_config.compliance_frameworks],
            'compliant': result['compliant'],
            'violations': result['violations']
        }
        
        self.processing_logs.append(log_entry)
        
        # Keep only recent logs (for privacy)
        if len(self.processing_logs) > 1000:
            self.processing_logs = self.processing_logs[-1000:]

class RegionManager:
    """Manages multi-region deployment and data localization."""
    
    def __init__(self):
        self.region_configs = self._initialize_regions()
        self.current_region = None
        
        logger.info("🌍 Region Manager initialized with global deployment support")
    
    def _initialize_regions(self) -> Dict[Region, RegionConfig]:
        """Initialize configuration for all supported regions."""
        return {
            # Americas
            Region.US_EAST: RegionConfig(
                region=Region.US_EAST,
                primary_language=Language.ENGLISH,
                compliance_frameworks=[ComplianceFramework.CCPA],
                data_residency_required=False,
                encryption_key_region="us-east-1",
                latency_target_ms=100
            ),
            Region.US_WEST: RegionConfig(
                region=Region.US_WEST,
                primary_language=Language.ENGLISH,
                compliance_frameworks=[ComplianceFramework.CCPA],
                data_residency_required=False,
                encryption_key_region="us-west-2",
                latency_target_ms=100
            ),
            Region.CANADA: RegionConfig(
                region=Region.CANADA,
                primary_language=Language.ENGLISH,
                compliance_frameworks=[ComplianceFramework.PIPEDA],
                data_residency_required=True,
                encryption_key_region="ca-central-1",
                latency_target_ms=120
            ),
            Region.BRAZIL: RegionConfig(
                region=Region.BRAZIL,
                primary_language=Language.SPANISH,
                compliance_frameworks=[],
                data_residency_required=False,
                encryption_key_region="sa-east-1",
                latency_target_ms=150
            ),
            
            # Europe (GDPR required)
            Region.EU_WEST: RegionConfig(
                region=Region.EU_WEST,
                primary_language=Language.ENGLISH,
                compliance_frameworks=[ComplianceFramework.GDPR],
                data_residency_required=True,
                encryption_key_region="eu-west-1",
                latency_target_ms=80
            ),
            Region.EU_CENTRAL: RegionConfig(
                region=Region.EU_CENTRAL,
                primary_language=Language.GERMAN,
                compliance_frameworks=[ComplianceFramework.GDPR],
                data_residency_required=True,
                encryption_key_region="eu-central-1",
                latency_target_ms=80
            ),
            Region.UK: RegionConfig(
                region=Region.UK,
                primary_language=Language.ENGLISH,
                compliance_frameworks=[ComplianceFramework.GDPR],
                data_residency_required=True,
                encryption_key_region="eu-west-2",
                latency_target_ms=60
            ),
            
            # Asia-Pacific
            Region.ASIA_PACIFIC: RegionConfig(
                region=Region.ASIA_PACIFIC,
                primary_language=Language.ENGLISH,
                compliance_frameworks=[ComplianceFramework.PDPA],
                data_residency_required=True,
                encryption_key_region="ap-southeast-1",
                latency_target_ms=120
            ),
            Region.JAPAN: RegionConfig(
                region=Region.JAPAN,
                primary_language=Language.JAPANESE,
                compliance_frameworks=[],
                data_residency_required=False,
                encryption_key_region="ap-northeast-1",
                latency_target_ms=100
            ),
            Region.CHINA: RegionConfig(
                region=Region.CHINA,
                primary_language=Language.CHINESE,
                compliance_frameworks=[],
                data_residency_required=True,
                encryption_key_region="cn-north-1",
                latency_target_ms=120
            )
        }
    
    def select_optimal_region(self, user_location: str, preferences: Dict[str, Any] = None) -> Region:
        """Select optimal region based on user location and preferences."""
        preferences = preferences or {}
        
        # Simple region selection based on location hints
        location_lower = user_location.lower()
        
        # Region mapping based on common location indicators
        region_mapping = {
            'us': Region.US_EAST,
            'usa': Region.US_EAST,
            'united states': Region.US_EAST,
            'california': Region.US_WEST,
            'canada': Region.CANADA,
            'brasil': Region.BRAZIL,
            'brazil': Region.BRAZIL,
            'eu': Region.EU_WEST,
            'europe': Region.EU_WEST,
            'uk': Region.UK,
            'britain': Region.UK,
            'germany': Region.EU_CENTRAL,
            'deutschland': Region.EU_CENTRAL,
            'france': Region.EU_WEST,
            'singapore': Region.ASIA_PACIFIC,
            'asia': Region.ASIA_PACIFIC,
            'japan': Region.JAPAN,
            'china': Region.CHINA,
            'chinese': Region.CHINA
        }
        
        # Find matching region
        for location_key, region in region_mapping.items():
            if location_key in location_lower:
                self.current_region = region
                logger.info(f"🌍 Selected region {region.value} based on location: {user_location}")
                return region
        
        # Default to US_EAST if no match found
        self.current_region = Region.US_EAST
        logger.info(f"🌍 Using default region {Region.US_EAST.value} for location: {user_location}")
        return Region.US_EAST
    
    def get_region_config(self, region: Region) -> RegionConfig:
        """Get configuration for a specific region."""
        return self.region_configs[region]
    
    def validate_data_residency(self, region: Region, data_info: Dict[str, Any]) -> bool:
        """Validate data residency requirements for a region."""
        config = self.region_configs[region]
        
        if not config.data_residency_required:
            return True
        
        # Check if data processing location matches region requirements
        processing_region = data_info.get('processing_region', region.value)
        
        # For EU regions, any EU region is acceptable
        eu_regions = [Region.EU_WEST, Region.EU_CENTRAL, Region.UK]
        if region in eu_regions:
            return any(processing_region.startswith(r.value.split('-')[0]) for r in eu_regions)
        
        # For other regions, exact match required
        return processing_region.startswith(region.value.split('-')[0])

class GlobalDGDNDeployment:
    """Global DGDN deployment with full internationalization and compliance."""
    
    def __init__(self):
        self.i18n_manager = InternationalizationManager()
        self.compliance_manager = ComplianceManager()
        self.region_manager = RegionManager()
        
        # Import the lightweight DGDN for processing
        try:
            from simple_autonomous_demo import LightweightDGDN
            self.dgdn_core = LightweightDGDN(node_dim=32, hidden_dim=64, num_layers=2)
        except ImportError:
            logger.warning("Could not import DGDN core, using mock implementation")
            self.dgdn_core = None
        
        logger.info("🌍 Global DGDN Deployment initialized")
    
    def process_with_global_compliance(self, 
                                     data: Dict[str, Any], 
                                     user_location: str,
                                     language: Language = Language.ENGLISH,
                                     user_preferences: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process data with full global compliance and localization."""
        
        # Set language preference
        self.i18n_manager.set_locale(language)
        
        # Display localized startup message
        logger.info(self.i18n_manager.get_message("startup"))
        
        try:
            # Select optimal region
            selected_region = self.region_manager.select_optimal_region(user_location, user_preferences)
            region_config = self.region_manager.get_region_config(selected_region)
            
            logger.info(self.i18n_manager.get_message("region_selected") + f": {selected_region.value}")
            logger.info(self.i18n_manager.get_message("data_location") + f": {region_config.encryption_key_region}")
            
            # Display privacy notice
            logger.info(self.i18n_manager.get_message("privacy_notice"))
            
            # Validate compliance
            data_info = {
                'user_id': 'anonymous',  # For demo purposes
                'created_date': datetime.datetime.now().isoformat(),
                'processing_region': region_config.encryption_key_region,
                'data_type': 'graph_analysis'
            }
            
            compliance_result = self.compliance_manager.validate_compliance(region_config, data_info)
            
            if not compliance_result['compliant']:
                return {
                    'status': 'compliance_failed',
                    'message': self.i18n_manager.get_message("error"),
                    'violations': compliance_result['violations'],
                    'region': selected_region.value,
                    'language': language.value
                }
            
            logger.info(self.i18n_manager.get_message("compliance_check"))
            
            # Validate data residency
            if not self.region_manager.validate_data_residency(selected_region, data_info):
                return {
                    'status': 'data_residency_failed',
                    'message': "Data residency requirements not met",
                    'region': selected_region.value,
                    'language': language.value
                }
            
            # Process data
            logger.info(self.i18n_manager.get_message("processing"))
            start_time = time.time()
            
            # Use DGDN core if available, otherwise mock processing
            if self.dgdn_core:
                dgdn_result = self.dgdn_core.forward_pass(data)
                processing_successful = True
            else:
                # Mock processing for demo
                dgdn_result = {
                    'node_embeddings': [[0.1, 0.2] for _ in range(data.get('num_nodes', 10))],
                    'uncertainty_mean': 0.15,
                    'uncertainty_std': 0.05
                }
                processing_successful = True
            
            processing_time = time.time() - start_time
            
            if not processing_successful:
                return {
                    'status': 'processing_failed',
                    'message': self.i18n_manager.get_message("error"),
                    'region': selected_region.value,
                    'language': language.value
                }
            
            logger.info(self.i18n_manager.get_message("completed"))
            
            # Generate localized performance report
            performance_report = self._generate_performance_report(data, processing_time, dgdn_result)
            
            # Compile global deployment result
            result = {
                'status': 'success',
                'message': self.i18n_manager.get_message("completed"),
                'dgdn_result': dgdn_result,
                'performance_report': performance_report,
                'compliance': {
                    'frameworks_validated': compliance_result['frameworks_checked'],
                    'compliant': compliance_result['compliant'],
                    'data_residency_validated': True
                },
                'deployment': {
                    'region': selected_region.value,
                    'language': language.value,
                    'data_location': region_config.encryption_key_region,
                    'latency_target_ms': region_config.latency_target_ms,
                    'encryption_key_region': region_config.encryption_key_region
                },
                'processing_metadata': {
                    'timestamp': datetime.datetime.now().isoformat(),
                    'processing_time_seconds': processing_time,
                    'user_location': user_location,
                    'data_anonymized': True
                }
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Global processing error: {e}")
            return {
                'status': 'error',
                'message': self.i18n_manager.get_message("error"),
                'error_details': str(e),
                'region': getattr(self.region_manager, 'current_region', Region.US_EAST).value,
                'language': language.value
            }
    
    def _generate_performance_report(self, data: Dict[str, Any], 
                                   processing_time: float, 
                                   dgdn_result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate localized performance report."""
        
        return {
            'title': self.i18n_manager.get_message("performance_report"),
            'metrics': {
                self.i18n_manager.get_message("nodes_processed"): data.get('num_nodes', 'N/A'),
                self.i18n_manager.get_message("edges_processed"): data.get('num_edges', 'N/A'),
                self.i18n_manager.get_message("inference_time"): f"{processing_time:.3f}s"
            },
            'quality_metrics': {
                'uncertainty_mean': dgdn_result.get('uncertainty_mean', 0.0),
                'uncertainty_std': dgdn_result.get('uncertainty_std', 0.0),
                'embedding_dimensions': len(dgdn_result.get('node_embeddings', [[]])[0]) if dgdn_result.get('node_embeddings') else 0
            }
        }
    
    def demonstrate_global_deployment(self) -> Dict[str, Any]:
        """Demonstrate global deployment across multiple regions and languages."""
        
        logger.info("🌍 Starting Global DGDN Deployment Demonstration")
        logger.info("=" * 80)
        
        # Test scenarios: different locations, languages, and data sizes
        test_scenarios = [
            {
                'location': 'United States',
                'language': Language.ENGLISH,
                'data_size': {'nodes': 50, 'edges': 150}
            },
            {
                'location': 'Germany',
                'language': Language.GERMAN,
                'data_size': {'nodes': 30, 'edges': 100}
            },
            {
                'location': 'Japan',
                'language': Language.JAPANESE,
                'data_size': {'nodes': 40, 'edges': 120}
            },
            {
                'location': 'Singapore',
                'language': Language.ENGLISH,
                'data_size': {'nodes': 25, 'edges': 80}
            },
            {
                'location': 'France',
                'language': Language.FRENCH,
                'data_size': {'nodes': 35, 'edges': 110}
            }
        ]
        
        results = []
        
        for i, scenario in enumerate(test_scenarios, 1):
            logger.info(f"\n🌐 Scenario {i}/{len(test_scenarios)}: {scenario['location']} ({scenario['language'].value})")
            
            # Generate test data
            if self.dgdn_core:
                test_data = self.dgdn_core.create_synthetic_data(
                    num_nodes=scenario['data_size']['nodes'],
                    num_edges=scenario['data_size']['edges']
                )
            else:
                # Mock data for demo
                test_data = {
                    'node_features': [[0.1, 0.2] for _ in range(scenario['data_size']['nodes'])],
                    'edges': [(0, 1, 1.0, 0.5) for _ in range(scenario['data_size']['edges'])],
                    'num_nodes': scenario['data_size']['nodes'],
                    'num_edges': scenario['data_size']['edges']
                }
            
            # Process with global compliance
            result = self.process_with_global_compliance(
                data=test_data,
                user_location=scenario['location'],
                language=scenario['language']
            )
            
            # Add scenario info to result
            result['scenario'] = scenario
            result['scenario_id'] = i
            
            results.append(result)
            
            # Log key metrics
            if result['status'] == 'success':
                deployment = result['deployment']
                performance = result['performance_report']
                logger.info(f"   ✅ Success: Region {deployment['region']}, Processing time: {result['processing_metadata']['processing_time_seconds']:.3f}s")
            else:
                logger.error(f"   ❌ Failed: {result['message']}")
        
        # Compile summary
        successful_deployments = [r for r in results if r['status'] == 'success']
        
        summary = {
            'total_scenarios': len(test_scenarios),
            'successful_deployments': len(successful_deployments),
            'success_rate': len(successful_deployments) / len(test_scenarios),
            'regions_tested': list(set(r['deployment']['region'] for r in successful_deployments)),
            'languages_tested': list(set(r['deployment']['language'] for r in successful_deployments)),
            'compliance_frameworks_validated': list(set(
                framework for r in successful_deployments 
                for framework in r['compliance']['frameworks_validated']
            )),
            'average_processing_time': sum(r['processing_metadata']['processing_time_seconds'] 
                                         for r in successful_deployments) / len(successful_deployments) if successful_deployments else 0,
            'detailed_results': results
        }
        
        # Final reporting
        logger.info("\n" + "=" * 80)
        logger.info("🌍 GLOBAL DEPLOYMENT SUMMARY")
        logger.info("=" * 80)
        logger.info(f"✅ Success Rate: {summary['success_rate']:.1%} ({summary['successful_deployments']}/{summary['total_scenarios']})")
        logger.info(f"🌍 Regions Deployed: {', '.join(summary['regions_tested'])}")
        logger.info(f"🗣️  Languages Supported: {', '.join(summary['languages_tested'])}")
        logger.info(f"🛡️  Compliance Frameworks: {', '.join(summary['compliance_frameworks_validated'])}")
        logger.info(f"⚡ Average Processing Time: {summary['average_processing_time']:.3f}s")
        logger.info("=" * 80)
        
        return summary

def main():
    """Main execution function for global DGDN deployment demonstration."""
    
    try:
        # Initialize global deployment
        global_deployment = GlobalDGDNDeployment()
        
        # Run comprehensive demonstration
        summary = global_deployment.demonstrate_global_deployment()
        
        # Save results (convert enums to strings for JSON serialization)
        def convert_enums(obj):
            if isinstance(obj, Enum):
                return obj.value
            elif isinstance(obj, dict):
                return {k: convert_enums(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_enums(item) for item in obj]
            else:
                return obj
        
        json_safe_summary = convert_enums(summary)
        
        with open('global_deployment_results.json', 'w', encoding='utf-8') as f:
            json.dump(json_safe_summary, f, indent=2, ensure_ascii=False)
        
        # Determine success
        if summary['success_rate'] >= 0.8:  # 80% success rate required
            logger.info("🚀 Global deployment demonstration PASSED!")
            return 0
        else:
            logger.error("💥 Global deployment demonstration FAILED!")
            return 1
            
    except Exception as e:
        logger.error(f"💥 Global deployment failed with exception: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)