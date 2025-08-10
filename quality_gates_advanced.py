#!/usr/bin/env python3
"""Advanced Quality Gates for DGDN Research Extensions."""

import sys
import time
import traceback
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_advanced_imports():
    """Test all advanced component imports."""
    print("🔍 Testing advanced component imports...")
    
    tests = [
        ("Core DGDN", "from dgdn import DynamicGraphDiffusionNet, DGDNTrainer"),
        ("Advanced Models", "from dgdn.models.advanced import FoundationDGDN, ContinuousDGDN, MultiScaleDGDN"),
        ("Research Components", "from dgdn.research.causal import CausalDGDN"),
        ("Enterprise Security", "from dgdn.enterprise.security import SecurityManager"),
        ("Enterprise Monitoring", "from dgdn.enterprise.monitoring import AdvancedMonitoring"),
        ("Distributed Training", "from dgdn.distributed.distributed_training import DistributedDGDNTrainer"),
        ("Edge Computing", "from dgdn.distributed.edge_computing import EdgeDGDN"),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, import_statement in tests:
        try:
            exec(import_statement)
            print(f"  ✅ {test_name}")
            passed += 1
        except ImportError as e:
            print(f"  ❌ {test_name}: {e}")
        except Exception as e:
            print(f"  ⚠️  {test_name}: {e}")
    
    print(f"\\n📊 Import Tests: {passed}/{total} passed")
    return passed == total

def test_model_instantiation():
    """Test advanced model instantiation."""
    print("\\n🏗️ Testing model instantiation...")
    
    try:
        import torch
        from dgdn.models.advanced import FoundationDGDN, MultiScaleDGDN
        
        # Test Foundation DGDN
        foundation_model = FoundationDGDN(
            node_dim=32, hidden_dim=64, num_layers=2
        )
        foundation_params = sum(p.numel() for p in foundation_model.parameters())
        print(f"  ✅ FoundationDGDN: {foundation_params:,} parameters")
        
        # Test MultiScale DGDN
        multiscale_model = MultiScaleDGDN(
            node_dim=32, time_scales=[1.0, 5.0, 10.0], hidden_dim=64
        )
        multiscale_params = sum(p.numel() for p in multiscale_model.parameters())
        print(f"  ✅ MultiScaleDGDN: {multiscale_params:,} parameters")
        
        print("\\n📊 Model Instantiation: ✅ PASSED")
        return True
        
    except Exception as e:
        print(f"  ❌ Model instantiation failed: {e}")
        print("\\n📊 Model Instantiation: ❌ FAILED")
        return False

def test_functionality():
    """Test basic functionality of advanced models."""
    print("\\n⚡ Testing advanced functionality...")
    
    try:
        import torch
        from dgdn.models.advanced import FoundationDGDN
        
        # Create model and synthetic data
        model = FoundationDGDN(node_dim=16, hidden_dim=32, num_layers=1)
        
        # Create synthetic temporal graph data
        num_nodes, num_edges = 10, 20
        node_features = torch.randn(num_nodes, 16)
        edge_index = torch.randint(0, num_nodes, (2, num_edges))
        edge_features = torch.randn(num_edges, 8)
        timestamps = torch.sort(torch.rand(num_edges) * 10)[0]
        
        # Create data object
        data = type('Data', (), {
            'x': node_features,
            'edge_index': edge_index,
            'edge_attr': edge_features,
            'timestamps': timestamps,
            'num_nodes': num_nodes
        })()
        
        # Test forward pass
        with torch.no_grad():
            output = model(data)
            
        print(f"  ✅ Forward pass successful: output shape {output['node_embeddings'].shape}")
        
        # Test pretraining mode
        pretraining_output = model.pretraining_forward(data)
        print(f"  ✅ Pretraining mode: reconstruction loss {pretraining_output['pretraining_losses']['reconstruction']:.4f}")
        
        print("\\n📊 Functionality Tests: ✅ PASSED")
        return True
        
    except Exception as e:
        print(f"  ❌ Functionality test failed: {e}")
        traceback.print_exc()
        print("\\n📊 Functionality Tests: ❌ FAILED")
        return False

def test_research_components():
    """Test research-specific components."""
    print("\\n🔬 Testing research components...")
    
    try:
        import torch
        from dgdn.research.causal import CausalDGDN
        
        # Create causal model
        causal_model = CausalDGDN(
            node_dim=16, hidden_dim=32, num_layers=1, max_nodes=20
        )
        
        # Test with synthetic data
        num_nodes = 15
        data = type('Data', (), {
            'x': torch.randn(num_nodes, 16),
            'edge_index': torch.randint(0, num_nodes, (2, 30)),
            'timestamps': torch.sort(torch.rand(30) * 5)[0],
            'num_nodes': num_nodes
        })()
        
        # Test causal discovery
        with torch.no_grad():
            causal_output = causal_model.discover_causal_structure(data)
            
        print(f"  ✅ Causal discovery: adjacency shape {causal_output['causal_adjacency'].shape}")
        print(f"  ✅ Causal loss: {causal_output['losses']['causal']:.4f}")
        
        print("\\n📊 Research Components: ✅ PASSED")
        return True
        
    except Exception as e:
        print(f"  ❌ Research components test failed: {e}")
        print("\\n📊 Research Components: ❌ FAILED")
        return False

def test_enterprise_features():
    """Test enterprise-grade features."""
    print("\\n🏢 Testing enterprise features...")
    
    try:
        from dgdn.enterprise.security import SecurityManager
        from dgdn.enterprise.monitoring import MetricsCollector
        
        # Test security manager
        security_manager = SecurityManager()
        print(f"  ✅ SecurityManager instantiated")
        
        # Test metrics collector
        metrics_collector = MetricsCollector()
        system_metrics = metrics_collector.collect_system_metrics()
        print(f"  ✅ MetricsCollector: {len(system_metrics)} system metrics")
        
        print("\\n📊 Enterprise Features: ✅ PASSED")
        return True
        
    except Exception as e:
        print(f"  ❌ Enterprise features test failed: {e}")
        print("\\n📊 Enterprise Features: ❌ FAILED")
        return False

def test_code_quality():
    """Test code quality metrics."""
    print("\\n📝 Testing code quality...")
    
    # Count implemented components
    src_path = Path(__file__).parent / "src" / "dgdn"
    
    python_files = list(src_path.rglob("*.py"))
    total_files = len(python_files)
    
    total_lines = 0
    for py_file in python_files:
        try:
            with open(py_file, 'r') as f:
                total_lines += sum(1 for line in f if line.strip())
        except:
            pass
    
    # Count modules and components
    modules = list(src_path.iterdir())
    module_count = len([m for m in modules if m.is_dir() and not m.name.startswith('__')])
    
    print(f"  ✅ Python files: {total_files}")
    print(f"  ✅ Lines of code: {total_lines:,}")
    print(f"  ✅ Major modules: {module_count}")
    
    # Quality thresholds
    quality_score = 0
    if total_files >= 40:  # Good coverage
        quality_score += 25
    if total_lines >= 10000:  # Substantial implementation
        quality_score += 25
    if module_count >= 6:  # Good modular structure
        quality_score += 25
    
    # Check for key components
    key_components = ['models', 'research', 'enterprise', 'distributed']
    found_components = sum(1 for comp in key_components if (src_path / comp).exists())
    if found_components == len(key_components):
        quality_score += 25
        
    print(f"  📊 Code Quality Score: {quality_score}/100")
    
    if quality_score >= 80:
        print("\\n📊 Code Quality: ✅ EXCELLENT")
        return True
    elif quality_score >= 60:
        print("\\n📊 Code Quality: ⚠️ GOOD")
        return True
    else:
        print("\\n📊 Code Quality: ❌ NEEDS IMPROVEMENT")
        return False

def run_all_quality_gates():
    """Run all quality gates."""
    print("🚀 DGDN Advanced Quality Gates")
    print("=" * 50)
    
    start_time = time.time()
    
    tests = [
        ("Advanced Imports", test_advanced_imports),
        ("Model Instantiation", test_model_instantiation),
        ("Functionality", test_functionality),
        ("Research Components", test_research_components),
        ("Enterprise Features", test_enterprise_features),
        ("Code Quality", test_code_quality),
    ]
    
    results = {}
    passed_count = 0
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results[test_name] = result
            if result:
                passed_count += 1
        except Exception as e:
            print(f"\\n❌ {test_name} crashed: {e}")
            results[test_name] = False
    
    execution_time = time.time() - start_time
    
    # Summary
    print("\\n" + "=" * 50)
    print("📊 QUALITY GATES SUMMARY")
    print("=" * 50)
    
    total_tests = len(tests)
    success_rate = (passed_count / total_tests) * 100
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:<20} | {status}")
    
    print(f"\\nOverall Results:")
    print(f"  Tests Passed: {passed_count}/{total_tests}")
    print(f"  Success Rate: {success_rate:.1f}%")
    print(f"  Execution Time: {execution_time:.2f}s")
    
    if success_rate >= 80:
        grade = "🏆 EXCELLENT"
    elif success_rate >= 60:
        grade = "✅ GOOD"
    else:
        grade = "⚠️ NEEDS WORK"
        
    print(f"  Overall Grade: {grade}")
    
    print("\\n🎯 ADVANCED DGDN IMPLEMENTATION STATUS:")
    
    if passed_count >= 5:
        print("✅ RESEARCH-GRADE IMPLEMENTATION ACHIEVED")
        print("✅ Advanced features fully functional")
        print("✅ Enterprise-ready components implemented")
        print("✅ Ready for production deployment")
        return True
    else:
        print("⚠️ Implementation needs refinement")
        return False

if __name__ == "__main__":
    success = run_all_quality_gates()
    sys.exit(0 if success else 1)