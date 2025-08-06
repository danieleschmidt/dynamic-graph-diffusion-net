#!/usr/bin/env python3
"""
DGDN Global-First Features Demo

Demonstrates internationalization, compliance, and multi-region capabilities.
"""

import torch
from dgdn import (
    DynamicGraphDiffusionNet, 
    TemporalData,
    get_translator, 
    set_global_locale,
    PrivacyManager,
    RegionManager,
    DeploymentRegion
)
from dgdn.compliance import DataCategory, ProcessingPurpose, PrivacyRegime


def demonstrate_internationalization():
    """Demonstrate I18n support across multiple languages."""
    print("🌍 INTERNATIONALIZATION DEMO")
    print("=" * 60)
    
    # Test multiple languages
    languages = ['en', 'es', 'fr', 'de', 'ja', 'zh']
    
    for lang in languages:
        set_global_locale(lang)
        translator = get_translator()
        
        print(f"\n📍 {lang.upper()} ({translator.get_locale_info()['native_name']}):")
        
        # Training messages
        print(f"  {translator.t('training.started', epochs=10)}")
        print(f"  {translator.t('training.epoch_progress', epoch=5, total_epochs=10, loss=0.1234)}")
        print(f"  {translator.t('model.created', layers=3, hidden_dim=256)}")
        
        # Performance metrics with proper formatting
        speed_improvement = 0.27
        print(f"  {translator.t('perf.speed_improvement', improvement=translator.format_percentage(speed_improvement))}")
        
        # Success message
        print(f"  {translator.t('success.tests_passed', passed=38, total=45)}")


def demonstrate_privacy_compliance():
    """Demonstrate privacy compliance across multiple regimes."""
    print("\n🛡️ PRIVACY COMPLIANCE DEMO")
    print("=" * 60)
    
    # Initialize privacy manager for multiple regimes
    privacy_manager = PrivacyManager([
        PrivacyRegime.GDPR, 
        PrivacyRegime.CCPA, 
        PrivacyRegime.PDPA
    ])
    
    # Create sample data
    sample_data = torch.randn(100, 64)  # User embeddings
    
    # Classify data
    classification = privacy_manager.classify_data(
        sample_data, 
        context={
            'data_type': 'user_embeddings',
            'contains_node_ids': True,
            'data_source': 'user_input'
        }
    )
    
    print(f"📊 Data Classification:")
    print(f"  Category: {classification['category'].value}")
    print(f"  Contains PII: {classification['contains_pii']}")
    print(f"  Risk Level: {classification['risk_level']}")
    print(f"  Required Protections: {', '.join(classification['required_protections'])}")
    
    # Request consent for processing
    consent = privacy_manager.request_consent(
        purpose=ProcessingPurpose.MACHINE_LEARNING,
        data_subject_id="user_12345",
        data_categories=[DataCategory.PERSONAL_IDENTIFIABLE, DataCategory.BEHAVIORAL]
    )
    
    print(f"\n✅ Consent Recorded:")
    print(f"  Consent ID: {consent['consent_id']}")
    print(f"  Purpose: {consent['purpose']}")
    print(f"  Legal Basis: {consent['legal_basis']}")
    print(f"  Valid Until: {consent['expiry_date']}")
    
    # Check processing lawfulness
    lawfulness = privacy_manager.check_processing_lawfulness(
        data=sample_data,
        purpose=ProcessingPurpose.MACHINE_LEARNING,
        data_subject_id="user_12345"
    )
    
    print(f"\n⚖️ Processing Lawfulness Check:")
    print(f"  Lawful: {lawfulness['lawful']}")
    print(f"  Legal Basis: {', '.join(lawfulness['legal_basis'])}")
    if lawfulness['required_actions']:
        print(f"  Required Actions: {', '.join(lawfulness['required_actions'])}")
    
    # Apply data minimization
    minimized_data = privacy_manager.apply_data_minimization(
        data=sample_data,
        purpose=ProcessingPurpose.MACHINE_LEARNING
    )
    
    print(f"\n🔒 Data Minimization Applied:")
    print(f"  Original Shape: {sample_data.shape}")
    print(f"  Minimized Shape: {minimized_data.shape}")
    
    # Demonstrate anonymization
    anonymized = privacy_manager.anonymize_data(sample_data, method='differential_privacy')
    
    print(f"\n🎭 Data Anonymization:")
    print(f"  Method: {anonymized['method']}")
    print(f"  Preserves Structure: {anonymized['preserves_structure']}")
    if anonymized['method'] == 'differential_privacy':
        print(f"  Privacy Budget (ε): {anonymized['epsilon']}")
    
    # Handle data subject request
    access_request = privacy_manager.handle_data_subject_request(
        request_type="access",
        data_subject_id="user_12345"
    )
    
    print(f"\n📋 Data Subject Access Request:")
    print(f"  Request ID: {access_request['request_id']}")
    print(f"  Status: {access_request['status']}")
    print(f"  Data Found: {access_request.get('data_found', False)}")
    print(f"  Consent Records: {len(access_request.get('consent_records', []))}")


def demonstrate_multi_region_deployment():
    """Demonstrate multi-region deployment capabilities."""
    print("\n🌐 MULTI-REGION DEPLOYMENT DEMO")
    print("=" * 60)
    
    # Initialize region manager
    region_manager = RegionManager()
    
    # Get deployment status
    status = region_manager.get_deployment_status()
    
    print(f"🗺️ Global Deployment Status:")
    print(f"  Total Regions: {status['total_regions']}")
    print(f"  Active Regions: {status['active_regions']}")
    print(f"  Active Region List: {', '.join(status['active_region_list'])}")
    print(f"  Compliance Coverage: GDPR={status['compliance_coverage']['gdpr']}, "
          f"CCPA={status['compliance_coverage']['ccpa']}, PDPA={status['compliance_coverage']['pdpa']}")
    print(f"  Supported Languages: {', '.join(status['supported_languages'])}")
    
    # Get optimal region for different users
    user_scenarios = [
        {
            "name": "European User",
            "location": "eu",
            "compliance_requirements": ["gdpr"],
            "language_preference": "de"
        },
        {
            "name": "US User", 
            "location": "us",
            "compliance_requirements": ["ccpa"],
            "language_preference": "en"
        },
        {
            "name": "Singapore User",
            "location": "asia",
            "compliance_requirements": ["pdpa"],
            "language_preference": "zh"
        }
    ]
    
    print(f"\n🎯 Optimal Region Selection:")
    for scenario in user_scenarios:
        optimal_region = region_manager.get_optimal_region(
            user_location=scenario["location"],
            compliance_requirements=scenario["compliance_requirements"],
            language_preference=scenario["language_preference"]
        )
        
        print(f"  {scenario['name']}: {optimal_region.value}")
    
    # Get region recommendations
    print(f"\n📊 Region Recommendations for European User:")
    recommendations = region_manager.get_region_recommendations({
        "location": "europe",
        "language": "fr",
        "compliance_requirements": ["gdpr"]
    })
    
    for i, rec in enumerate(recommendations[:3], 1):
        print(f"  {i}. {rec['region_name']} (Score: {rec['score']})")
        print(f"     Reasons: {', '.join(rec['reasons'])}")
        print(f"     Languages: {', '.join(rec['languages'])}")
        print(f"     Compliance: {rec['compliance'].upper()}")
    
    # Simulate deployment
    print(f"\n🚀 Simulating Deployment to EU-West-1:")
    deployment_config = {
        "version": "1.0.0",
        "instance_type": "ml.p3.2xlarge",
        "auto_scaling": True,
        "monitoring": True,
        "compliance_mode": "gdpr"
    }
    
    deployment_result = region_manager.deploy_to_region(
        DeploymentRegion.EU_WEST_1,
        deployment_config
    )
    
    print(f"  Deployment ID: {deployment_result['deployment_id']}")
    print(f"  Status: {deployment_result['status']}")
    print(f"  Compliance Regime: {deployment_result['compliance_regime']}")
    print(f"  Data Residency: {deployment_result['data_residency']}")
    
    if deployment_result['status'] == 'completed':
        print(f"  API Endpoint: {deployment_result['endpoints']['api']}")
        print(f"  Training Endpoint: {deployment_result['endpoints']['training']}")
        print(f"  Monitoring: {deployment_result['monitoring_dashboard']}")


def demonstrate_integrated_training_workflow():
    """Demonstrate training with global-first features integrated."""
    print("\n🧠 INTEGRATED TRAINING WORKFLOW DEMO")
    print("=" * 60)
    
    # Set locale for German user
    set_global_locale('de')
    translator = get_translator()
    
    print(f"🇩🇪 {translator.t('training.started', epochs=5)}")
    
    # Initialize privacy manager
    privacy_manager = PrivacyManager([PrivacyRegime.GDPR])
    
    # Create DGDN model
    model = DynamicGraphDiffusionNet(
        node_dim=64,
        edge_dim=32, 
        hidden_dim=128,
        num_layers=2
    )
    
    print(f"📊 {translator.t('model.created', layers=2, hidden_dim=128)}")
    
    # Create sample temporal data
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
    timestamps = torch.tensor([0.1, 0.2, 0.3])
    node_features = torch.randn(3, 64)
    edge_attr = torch.randn(3, 32)
    
    data = TemporalData(
        edge_index=edge_index,
        timestamps=timestamps,
        node_features=node_features,
        edge_attr=edge_attr,
        num_nodes=3
    )
    
    # Check data compliance before processing
    lawfulness = privacy_manager.check_processing_lawfulness(
        data=node_features,
        purpose=ProcessingPurpose.MACHINE_LEARNING
    )
    
    if lawfulness['lawful']:
        print(f"✅ {translator.t('security.validation.passed')}")
        
        # Apply data protection if needed
        if lawfulness.get('required_actions'):
            print(f"🔒 Applying required protections...")
            protected_data = privacy_manager.apply_data_minimization(
                data=node_features,
                purpose=ProcessingPurpose.MACHINE_LEARNING
            )
            print(f"✅ Data protection applied")
        
        # Run forward pass
        with torch.no_grad():
            output = model(data)
        
        print(f"🎯 {translator.t('success.operation_completed')}")
        print(f"📈 Node embeddings shape: {output['node_embeddings'].shape}")
        print(f"📊 KL divergence: {output['kl_loss'].item():.4f}")
        
        # Show uncertainty quantification
        if 'uncertainty' in output:
            print(f"🎲 Mean uncertainty: {translator.format_number(output['uncertainty'].mean().item(), 4)}")
    else:
        print(f"❌ {translator.t('error.security.path_traversal')}")


def main():
    """Run comprehensive global-first demo."""
    print("🌍 DGDN GLOBAL-FIRST FEATURES DEMONSTRATION")
    print("=" * 80)
    print("Showcasing internationalization, compliance, and multi-region deployment")
    print("=" * 80)
    
    # Run all demonstrations
    demonstrate_internationalization()
    demonstrate_privacy_compliance()
    demonstrate_multi_region_deployment() 
    demonstrate_integrated_training_workflow()
    
    print("\n" + "=" * 80)
    print("✅ GLOBAL-FIRST DEMO COMPLETED!")
    print("=" * 80)
    print("🌍 DGDN is now ready for global deployment with:")
    print("  • Multi-language support (en, es, fr, de, ja, zh)")
    print("  • GDPR/CCPA/PDPA compliance")
    print("  • Multi-region deployment capabilities")
    print("  • Privacy-preserving data processing")
    print("  • Automated compliance workflows")


if __name__ == "__main__":
    main()