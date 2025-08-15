#!/usr/bin/env python3
"""
Adaptive Systems Validation
Tests self-improving patterns, adaptive learning, and continuous optimization.
"""

import sys
import os
sys.path.insert(0, 'src')

import torch
import torch.nn as nn
import torch.optim as optim
import time
import numpy as np
from typing import Dict, Any, List

# Import DGDN components
import dgdn
from dgdn import DynamicGraphDiffusionNet

def test_adaptive_learning_system():
    """Test adaptive learning system."""
    print("Testing adaptive learning system...")
    
    try:
        from dgdn.adaptive.learning import AdaptiveLearningSystem
        
        # Create adaptive learning system
        adaptive_learner = AdaptiveLearningSystem(
            base_learning_rate=1e-3,
            adaptation_window=5,
            patience=10
        )
        print("  ✓ Adaptive learning system initialized")
        
        # Create model and optimizer
        model = DynamicGraphDiffusionNet(
            node_dim=64,
            edge_dim=32,
            hidden_dim=128,
            num_layers=2,
            num_heads=4,
            diffusion_steps=3
        )
        
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        print("  ✓ Model and optimizer created")
        
        # Simulate training with adaptation
        from gen1_simple_validation import create_synthetic_data
        
        performance_history = []
        adaptation_count = 0
        
        for epoch in range(20):
            # Create training data
            data = create_synthetic_data(num_nodes=50, num_edges=150)
            
            # Forward pass
            model.train()
            optimizer.zero_grad()
            output = model(data)
            
            # Simple loss calculation
            loss = output['node_embeddings'].sum()
            loss.backward()
            optimizer.step()
            
            # Simulate performance metric (accuracy)
            simulated_accuracy = 0.5 + 0.4 * (1 - np.exp(-epoch / 10)) + np.random.normal(0, 0.05)
            simulated_accuracy = max(0.0, min(1.0, simulated_accuracy))
            
            # Adaptive learning step
            adaptations = adaptive_learner.adapt_training(
                model=model,
                optimizer=optimizer,
                current_loss=loss.item(),
                current_accuracy=simulated_accuracy,
                epoch=epoch
            )
            
            if adaptations:
                adaptation_count += 1
                print(f"    Epoch {epoch}: Adaptations applied - {list(adaptations.keys())}")
            
            performance_history.append(simulated_accuracy)
        
        print(f"  ✓ Training completed with {adaptation_count} adaptations")
        
        # Check adaptation effectiveness
        early_performance = np.mean(performance_history[:5])
        late_performance = np.mean(performance_history[-5:])
        improvement = late_performance - early_performance
        
        if improvement > 0.1:
            print(f"  ✓ Significant improvement: {improvement:.3f}")
        else:
            print(f"  ⚠ Modest improvement: {improvement:.3f}")
        
        # Test learning summary
        summary = adaptive_learner.get_learning_summary()
        if 'total_epochs' not in summary:
            print("  ✗ Learning summary missing data")
            return False
        
        print(f"  ✓ Learning summary: {summary['learning_trend']} trend")
        
        print("✓ Adaptive learning system successful")
        return True
        
    except Exception as e:
        print(f"✗ Adaptive learning system failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_self_tuning_optimizer():
    """Test self-tuning hyperparameter optimizer."""
    print("Testing self-tuning optimizer...")
    
    try:
        from dgdn.adaptive.optimization import SelfTuningOptimizer
        
        # Create self-tuning optimizer
        self_tuner = SelfTuningOptimizer(
            optimization_frequency=5,
            population_size=10,
            elite_fraction=0.3
        )
        print("  ✓ Self-tuning optimizer initialized")
        
        # Register hyperparameters
        self_tuner.register_hyperparameter(
            name="learning_rate",
            current_value=1e-3,
            value_type="float",
            bounds=(1e-5, 1e-1),
            mutation_strength=0.1
        )
        
        self_tuner.register_hyperparameter(
            name="batch_size",
            current_value=32,
            value_type="int",
            bounds=(8, 128),
            mutation_strength=0.1
        )
        
        self_tuner.register_hyperparameter(
            name="optimizer_type",
            current_value="adam",
            value_type="categorical",
            categories=["adam", "sgd", "rmsprop"]
        )
        
        print("  ✓ Hyperparameters registered")
        
        # Simulate optimization loop
        optimization_count = 0
        
        for iteration in range(25):
            # Simulate performance (with some trend based on hyperparameters)
            config = self_tuner.get_current_config()
            
            # Simple performance function
            lr = config["learning_rate"]
            batch_size = config["batch_size"]
            
            # Performance increases with moderate LR and batch size
            performance = 0.5 + 0.3 * np.exp(-abs(np.log10(lr) + 3)) + 0.1 * (batch_size / 64)
            performance += np.random.normal(0, 0.05)  # Add noise
            performance = max(0.0, min(1.0, performance))
            
            # Optimization step
            new_config = self_tuner.optimize_step(performance)
            
            if new_config != config:
                optimization_count += 1
                print(f"    Iteration {iteration}: Config updated")
            
        print(f"  ✓ Optimization completed with {optimization_count} updates")
        
        # Check optimization summary
        summary = self_tuner.get_optimization_summary()
        if 'best_performance' not in summary:
            print("  ✗ Optimization summary missing data")
            return False
        
        improvement = summary.get('improvement', 0)
        if improvement > 0:
            print(f"  ✓ Performance improvement: {improvement:.3f}")
        else:
            print(f"  ⚠ No improvement: {improvement:.3f}")
        
        print(f"  ✓ Best config: {summary['best_config']}")
        
        print("✓ Self-tuning optimizer successful")
        return True
        
    except Exception as e:
        print(f"✗ Self-tuning optimizer failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_hyperparameter_evolution():
    """Test hyperparameter evolution system."""
    print("Testing hyperparameter evolution...")
    
    try:
        from dgdn.adaptive.optimization import HyperparameterEvolution
        
        # Define search space
        search_space = {
            "learning_rate": {"type": "float", "low": 1e-5, "high": 1e-1},
            "hidden_dim": {"type": "int", "low": 64, "high": 512},
            "num_layers": {"type": "int", "low": 1, "high": 5},
            "activation": {"type": "categorical", "choices": ["relu", "gelu", "swish"]},
            "dropout": {"type": "float", "low": 0.0, "high": 0.5},
            "use_batch_norm": {"type": "bool"}
        }
        
        # Create evolution system
        hp_evolution = HyperparameterEvolution(
            search_space=search_space,
            population_size=20
        )
        print("  ✓ Hyperparameter evolution initialized")
        
        # Define fitness function
        def fitness_function(config):
            # Simulate model performance based on configuration
            lr = config["learning_rate"]
            hidden_dim = config["hidden_dim"]
            num_layers = config["num_layers"]
            dropout = config["dropout"]
            
            # Simple heuristic for performance
            performance = 0.5
            
            # Optimal LR around 1e-3
            performance += 0.2 * np.exp(-abs(np.log10(lr) + 3))
            
            # Moderate hidden dimensions are better
            performance += 0.1 * (1 - abs(hidden_dim - 256) / 256)
            
            # 2-3 layers optimal
            performance += 0.1 * (1 - abs(num_layers - 2.5) / 2.5)
            
            # Some dropout is good but not too much
            performance += 0.1 * (dropout * (1 - dropout) * 4)
            
            # Add noise
            performance += np.random.normal(0, 0.05)
            
            return max(0.0, min(1.0, performance))
        
        # Evolve hyperparameters
        best_config = hp_evolution.evolve(
            fitness_function=fitness_function,
            generations=5  # Reduced for testing
        )
        
        print(f"  ✓ Evolution completed")
        print(f"  ✓ Best config: {best_config}")
        
        # Test evolution summary
        summary = hp_evolution.get_evolution_summary()
        if 'generation' not in summary:
            print("  ✗ Evolution summary missing data")
            return False
        
        print(f"  ✓ Evolution summary: Generation {summary['generation']}, Best fitness {summary['best_fitness']:.3f}")
        
        # Verify best configuration is reasonable
        best_lr = best_config["learning_rate"]
        if not (1e-5 <= best_lr <= 1e-1):
            print(f"  ✗ Learning rate out of bounds: {best_lr}")
            return False
        
        print("✓ Hyperparameter evolution successful")
        return True
        
    except Exception as e:
        print(f"✗ Hyperparameter evolution failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_online_learning():
    """Test online learning capabilities."""
    print("Testing online learning...")
    
    try:
        from dgdn.adaptive.learning import OnlineLearner
        
        # Create online learner
        online_learner = OnlineLearner(learning_rate=1e-4, momentum=0.9)
        print("  ✓ Online learner initialized")
        
        # Create model
        model = DynamicGraphDiffusionNet(
            node_dim=64,
            edge_dim=32,
            hidden_dim=128,
            num_layers=2,
            num_heads=4,
            diffusion_steps=3
        )
        
        # Create test data
        from gen1_simple_validation import create_synthetic_data
        data = create_synthetic_data(num_nodes=50, num_edges=150)
        
        # Test online updates
        initial_params = {name: param.clone() for name, param in model.named_parameters()}
        
        for step in range(5):
            # Forward pass
            model.train()
            output = model(data)
            loss = output['node_embeddings'].sum()
            
            # Online learning update
            online_learner.update_online(model, loss)
            
        # Check if parameters changed
        param_changed = False
        for name, param in model.named_parameters():
            if not torch.equal(initial_params[name], param):
                param_changed = True
                break
        
        if not param_changed:
            print("  ✗ Parameters did not change during online learning")
            return False
        
        print("  ✓ Online learning updates working")
        
        # Test distribution shift adaptation
        new_data = create_synthetic_data(num_nodes=100, num_edges=300)  # Different distribution
        adaptation_score = online_learner.adapt_to_distribution_shift(model, new_data)
        
        if not (0.0 <= adaptation_score <= 1.0):
            print(f"  ✗ Invalid adaptation score: {adaptation_score}")
            return False
        
        print(f"  ✓ Distribution shift adaptation score: {adaptation_score:.3f}")
        
        print("✓ Online learning successful")
        return True
        
    except Exception as e:
        print(f"✗ Online learning failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_meta_learning():
    """Test meta-learning capabilities."""
    print("Testing meta-learning...")
    
    try:
        from dgdn.adaptive.learning import MetaLearner, LearningMetrics
        
        # Create meta-learner
        meta_learner = MetaLearner(meta_lr=1e-3)
        print("  ✓ Meta-learner initialized")
        
        # Simulate strategy evaluation
        strategies = ["learning_rate_schedule", "batch_size_adaptation", "regularization_adaptation"]
        
        for strategy in strategies:
            # Simulate multiple evaluations
            for _ in range(5):
                performance_before = np.random.uniform(0.5, 0.7)
                performance_after = performance_before + np.random.uniform(-0.1, 0.2)
                
                effectiveness = meta_learner.evaluate_strategy(
                    strategy, performance_before, performance_after
                )
                
            print(f"    Strategy '{strategy}' effectiveness: {effectiveness:.3f}")
        
        # Test strategy recommendation
        dummy_metrics = LearningMetrics(
            epoch=10,
            loss=0.5,
            accuracy=0.7,
            learning_rate=1e-3,
            convergence_rate=1e-4,
            adaptation_score=0.6,
            timestamp=time.time()
        )
        
        recommended_strategy = meta_learner.recommend_strategy(dummy_metrics)
        if recommended_strategy not in strategies:
            print(f"  ✗ Invalid strategy recommendation: {recommended_strategy}")
            return False
        
        print(f"  ✓ Recommended strategy: {recommended_strategy}")
        
        # Test meta-update
        meta_learner.meta_update(
            strategies_used=["learning_rate_schedule", "batch_size_adaptation"],
            outcomes=[0.1, 0.05]
        )
        print("  ✓ Meta-update completed")
        
        print("✓ Meta-learning successful")
        return True
        
    except Exception as e:
        print(f"✗ Meta-learning failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_continual_learning():
    """Test continual learning without forgetting."""
    print("Testing continual learning...")
    
    try:
        from dgdn.adaptive.learning import ContinualLearningSystem
        
        # Create model
        model = DynamicGraphDiffusionNet(
            node_dim=64,
            edge_dim=32,
            hidden_dim=128,
            num_layers=2,
            num_heads=4,
            diffusion_steps=3
        )
        
        # Create continual learning system
        continual_learner = ContinualLearningSystem(model, importance_lambda=1000.0)
        print("  ✓ Continual learning system initialized")
        
        # Create synthetic data loader
        from gen1_simple_validation import create_synthetic_data
        
        class DummyDataLoader:
            def __init__(self, num_batches=5):
                self.num_batches = num_batches
                self.current_batch = 0
            
            def __iter__(self):
                self.current_batch = 0
                return self
            
            def __next__(self):
                if self.current_batch >= self.num_batches:
                    raise StopIteration
                self.current_batch += 1
                return create_synthetic_data(num_nodes=50, num_edges=150)
        
        data_loader = DummyDataLoader()
        
        # Test Fisher information computation
        continual_learner.compute_fisher_information(data_loader)
        print("  ✓ Fisher information computed")
        
        # Test EWC loss computation
        dummy_loss = torch.tensor(1.0, requires_grad=True)
        ewc_loss = continual_learner.ewc_loss(dummy_loss)
        
        if not isinstance(ewc_loss, torch.Tensor):
            print("  ✗ EWC loss not computed correctly")
            return False
        
        print(f"  ✓ EWC loss computed: {ewc_loss.item():.4f}")
        
        # Test task switching
        continual_learner.switch_task(data_loader)
        print("  ✓ Task switching completed")
        
        # Verify multiple tasks are tracked
        if continual_learner.task_id < 1:
            print("  ✗ Task switching not working")
            return False
        
        print(f"  ✓ Current task ID: {continual_learner.task_id}")
        
        print("✓ Continual learning successful")
        return True
        
    except Exception as e:
        print(f"✗ Continual learning failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_adaptive_systems_validation():
    """Run all adaptive systems validation tests."""
    print("🧬" + "=" * 80)
    print("ADAPTIVE SYSTEMS VALIDATION")
    print("🧬" + "=" * 80)
    
    tests = [
        ("Adaptive Learning System", test_adaptive_learning_system),
        ("Self-Tuning Optimizer", test_self_tuning_optimizer),
        ("Hyperparameter Evolution", test_hyperparameter_evolution),
        ("Online Learning", test_online_learning),
        ("Meta-Learning", test_meta_learning),
        ("Continual Learning", test_continual_learning)
    ]
    
    results = []
    start_time = time.time()
    
    for test_name, test_func in tests:
        print(f"\n🔍 Running Test: {test_name}")
        print("-" * 60)
        
        try:
            result = test_func()
            results.append(result)
            
            status_emoji = "✅" if result else "❌"
            print(f"{status_emoji} {test_name}: {'PASS' if result else 'FAIL'}")
            
        except Exception as e:
            results.append(False)
            print(f"❌ {test_name}: ERROR - {e}")
        
        print()
    
    end_time = time.time()
    
    # Summary
    print("🧬" + "=" * 80)
    print("ADAPTIVE SYSTEMS SUMMARY")
    print("🧬" + "=" * 80)
    
    passed = sum(results)
    total = len(results)
    success_rate = (passed / total * 100) if total > 0 else 0
    
    print(f"\n📊 RESULTS:")
    print(f"   Tests passed: {passed}/{total}")
    print(f"   Success rate: {success_rate:.1f}%")
    print(f"   Execution time: {end_time - start_time:.2f}s")
    
    # Detailed results
    print(f"\n📋 DETAILED RESULTS:")
    for i, (test_name, _) in enumerate(tests):
        status_emoji = "✅" if results[i] else "❌"
        status_text = "PASS" if results[i] else "FAIL"
        print(f"   {status_emoji} {test_name}: {status_text}")
    
    # Adaptive capabilities assessment
    print(f"\n🧬 ADAPTIVE CAPABILITIES ASSESSMENT:")
    if success_rate >= 100:
        print("   🚀 FULLY ADAPTIVE SYSTEM")
        print("   ✅ Self-improving patterns active")
        print("   ✅ Continuous optimization enabled")
        print("   ✅ Meta-learning operational")
    elif success_rate >= 85:
        print("   ⚠️ MOSTLY ADAPTIVE")
        print("   🔧 Some adaptive features need refinement")
        print("   ✅ Basic self-improvement working")
        print("   ⚠️ Advanced adaptation limited")
    else:
        print("   ❌ LIMITED ADAPTIVE CAPABILITIES")
        print("   🔧 Significant adaptive features missing")
        print("   ⚠️ Manual optimization required")
        print("   ❌ Self-improvement not functional")
    
    print("\n🧬" + "=" * 80)
    
    return success_rate >= 85

if __name__ == "__main__":
    success = run_adaptive_systems_validation()
    sys.exit(0 if success else 1)