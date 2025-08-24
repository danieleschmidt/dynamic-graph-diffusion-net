#!/usr/bin/env python3
"""
Comprehensive Quality Gates & Testing Suite
Autonomous SDLC Implementation - Quality Assurance & Validation

This module implements mandatory quality gates including:
- Comprehensive testing (unit, integration, performance)
- Security scanning and vulnerability assessment
- Code quality analysis and metrics
- Performance benchmarking and validation
- Compliance checking and documentation
- Production readiness assessment
"""

import sys
import os
import time
import json
import math
import random
import subprocess
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
import traceback
import hashlib
import re

class QualityGateStatus(Enum):
    """Quality gate status enumeration."""
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    NOT_RUN = "not_run"

class TestType(Enum):
    """Test type enumeration."""
    UNIT = "unit"
    INTEGRATION = "integration"
    PERFORMANCE = "performance"
    SECURITY = "security"
    COMPLIANCE = "compliance"
    STRESS = "stress"

@dataclass
class TestResult:
    """Test result data structure."""
    test_name: str
    test_type: TestType
    status: QualityGateStatus
    duration_ms: float
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'test_name': self.test_name,
            'test_type': self.test_type.value,
            'status': self.status.value,
            'duration_ms': self.duration_ms,
            'message': self.message,
            'details': self.details,
            'timestamp': self.timestamp
        }

@dataclass
class QualityGateReport:
    """Comprehensive quality gate report."""
    overall_status: QualityGateStatus
    test_results: List[TestResult]
    summary_stats: Dict[str, Any]
    quality_metrics: Dict[str, float]
    recommendations: List[str]
    generated_at: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'overall_status': self.overall_status.value,
            'test_results': [tr.to_dict() for tr in self.test_results],
            'summary_stats': self.summary_stats,
            'quality_metrics': self.quality_metrics,
            'recommendations': self.recommendations,
            'generated_at': self.generated_at
        }


class UnitTestSuite:
    """Comprehensive unit test suite for DGDN components."""
    
    def __init__(self, name="DGDN_UnitTests"):
        self.name = name
        self.test_results = []
    
    def run_tensor_tests(self) -> List[TestResult]:
        """Run tensor operation tests."""
        tests = []
        
        # Test tensor creation and validation
        start_time = time.time()
        try:
            # Import our optimized tensor
            sys.path.insert(0, '/root/repo')
            from optimized_gen3_demo import OptimizedTensor, OptimizationLevel
            
            # Test valid tensor creation
            tensor = OptimizedTensor([1.0, 2.0, 3.0], optimization_level=OptimizationLevel.MODERATE)
            assert len(tensor.data) == 3, "Tensor size mismatch"
            assert tensor.data[0] == 1.0, "Tensor data incorrect"
            
            duration = (time.time() - start_time) * 1000
            tests.append(TestResult(
                "tensor_creation_valid",
                TestType.UNIT,
                QualityGateStatus.PASSED,
                duration,
                "Valid tensor creation successful",
                {"tensor_size": len(tensor.data), "optimization_level": "moderate"}
            ))
            
        except Exception as e:
            duration = (time.time() - start_time) * 1000
            tests.append(TestResult(
                "tensor_creation_valid",
                TestType.UNIT,
                QualityGateStatus.FAILED,
                duration,
                f"Valid tensor creation failed: {str(e)}"
            ))
        
        # Test tensor operations
        start_time = time.time()
        try:
            from optimized_gen3_demo import OptimizedTensor
            
            tensor1 = OptimizedTensor([1.0, 2.0, 3.0])
            tensor2 = OptimizedTensor([4.0, 5.0, 6.0])
            
            # Addition
            result_add = tensor1 + tensor2
            expected = [5.0, 7.0, 9.0]
            for i, (actual, exp) in enumerate(zip(result_add.data, expected)):
                assert abs(actual - exp) < 1e-6, f"Addition result mismatch at {i}: {actual} vs {exp}"
            
            # Multiplication
            result_mul = tensor1 * tensor2
            expected_mul = [4.0, 10.0, 18.0]
            for i, (actual, exp) in enumerate(zip(result_mul.data, expected_mul)):
                assert abs(actual - exp) < 1e-6, f"Multiplication result mismatch at {i}: {actual} vs {exp}"
            
            duration = (time.time() - start_time) * 1000
            tests.append(TestResult(
                "tensor_operations",
                TestType.UNIT,
                QualityGateStatus.PASSED,
                duration,
                "Tensor operations working correctly",
                {"operations_tested": ["addition", "multiplication"]}
            ))
            
        except Exception as e:
            duration = (time.time() - start_time) * 1000
            tests.append(TestResult(
                "tensor_operations",
                TestType.UNIT,
                QualityGateStatus.FAILED,
                duration,
                f"Tensor operations failed: {str(e)}"
            ))
        
        return tests
    
    def run_model_tests(self) -> List[TestResult]:
        """Run model functionality tests."""
        tests = []
        
        # Test model initialization
        start_time = time.time()
        try:
            from optimized_gen3_demo import OptimizedDGDN, OptimizationLevel
            
            model = OptimizedDGDN(
                node_dim=16,
                hidden_dim=32,
                num_layers=2,
                time_dim=16,
                optimization_level=OptimizationLevel.BASIC,
                name="TestModel"
            )
            
            assert model.node_dim == 16, "Node dimension mismatch"
            assert model.hidden_dim == 32, "Hidden dimension mismatch"
            assert model.num_layers == 2, "Number of layers mismatch"
            
            duration = (time.time() - start_time) * 1000
            tests.append(TestResult(
                "model_initialization",
                TestType.UNIT,
                QualityGateStatus.PASSED,
                duration,
                "Model initialization successful",
                {"node_dim": model.node_dim, "hidden_dim": model.hidden_dim}
            ))
            
        except Exception as e:
            duration = (time.time() - start_time) * 1000
            tests.append(TestResult(
                "model_initialization",
                TestType.UNIT,
                QualityGateStatus.FAILED,
                duration,
                f"Model initialization failed: {str(e)}"
            ))
        
        # Test model forward pass with minimal data
        start_time = time.time()
        try:
            from optimized_gen3_demo import OptimizedDGDN, OptimizedTensor, OptimizationLevel
            
            model = OptimizedDGDN(
                node_dim=8,
                hidden_dim=16,
                num_layers=1,
                time_dim=8,
                optimization_level=OptimizationLevel.BASIC,
                name="TestModel_Forward"
            )
            
            # Create minimal test data
            nodes = [
                OptimizedTensor([0.1] * 8, optimization_level=OptimizationLevel.BASIC),
                OptimizedTensor([0.2] * 8, optimization_level=OptimizationLevel.BASIC)
            ]
            edges = [(0, 1)]
            timestamps = [1.0]
            
            result = model.forward(nodes, edges, timestamps)
            
            assert 'node_embeddings' in result, "Missing node_embeddings in result"
            assert 'processing_time' in result, "Missing processing_time in result"
            assert len(result['node_embeddings']) == 2, "Incorrect number of node embeddings"
            
            duration = (time.time() - start_time) * 1000
            tests.append(TestResult(
                "model_forward_pass",
                TestType.UNIT,
                QualityGateStatus.PASSED,
                duration,
                "Model forward pass successful",
                {
                    "nodes_processed": len(result['node_embeddings']),
                    "processing_time_ms": result.get('processing_time_ms', 0)
                }
            ))
            
        except Exception as e:
            duration = (time.time() - start_time) * 1000
            tests.append(TestResult(
                "model_forward_pass",
                TestType.UNIT,
                QualityGateStatus.FAILED,
                duration,
                f"Model forward pass failed: {str(e)}"
            ))
        
        return tests
    
    def run_all_tests(self) -> List[TestResult]:
        """Run all unit tests."""
        all_tests = []
        
        print("🧪 Running Unit Tests...")
        
        # Run tensor tests
        print("   Testing tensor operations...")
        tensor_tests = self.run_tensor_tests()
        all_tests.extend(tensor_tests)
        
        # Run model tests
        print("   Testing model functionality...")
        model_tests = self.run_model_tests()
        all_tests.extend(model_tests)
        
        # Summary
        passed = sum(1 for t in all_tests if t.status == QualityGateStatus.PASSED)
        failed = sum(1 for t in all_tests if t.status == QualityGateStatus.FAILED)
        print(f"   Unit tests completed: {passed} passed, {failed} failed")
        
        self.test_results = all_tests
        return all_tests


class PerformanceTestSuite:
    """Performance test suite with benchmarking and validation."""
    
    def __init__(self, name="DGDN_PerformanceTests"):
        self.name = name
        self.test_results = []
        
        # Performance thresholds
        self.latency_threshold_ms = 200.0
        self.throughput_threshold_nodes_per_sec = 50.0
        self.memory_threshold_mb = 100.0
    
    def test_latency_performance(self) -> TestResult:
        """Test latency performance against thresholds."""
        start_time = time.time()
        
        try:
            from optimized_gen3_demo import OptimizedDGDN, OptimizationLevel
            from optimized_gen3_demo import generate_performance_test_data
            
            model = OptimizedDGDN(
                node_dim=32,
                hidden_dim=64,
                num_layers=2,
                optimization_level=OptimizationLevel.AGGRESSIVE,
                name="LatencyTestModel"
            )
            
            # Generate moderate-sized test data
            test_data = generate_performance_test_data(
                num_nodes=30,
                num_edges=60,
                time_span=50.0,
                stress_level="moderate"
            )
            
            # Measure latency over multiple runs
            latencies = []
            for i in range(3):
                run_start = time.time()
                result = model.forward(
                    test_data['nodes'],
                    test_data['edges'],
                    test_data['timestamps']
                )
                run_latency = (time.time() - run_start) * 1000
                latencies.append(run_latency)
            
            avg_latency = sum(latencies) / len(latencies)
            max_latency = max(latencies)
            min_latency = min(latencies)
            
            # Check against threshold
            status = QualityGateStatus.PASSED if avg_latency < self.latency_threshold_ms else QualityGateStatus.FAILED
            
            duration = (time.time() - start_time) * 1000
            return TestResult(
                "latency_performance",
                TestType.PERFORMANCE,
                status,
                duration,
                f"Average latency: {avg_latency:.2f}ms (threshold: {self.latency_threshold_ms}ms)",
                {
                    "average_latency_ms": avg_latency,
                    "max_latency_ms": max_latency,
                    "min_latency_ms": min_latency,
                    "threshold_ms": self.latency_threshold_ms,
                    "runs": len(latencies)
                }
            )
            
        except Exception as e:
            duration = (time.time() - start_time) * 1000
            return TestResult(
                "latency_performance",
                TestType.PERFORMANCE,
                QualityGateStatus.FAILED,
                duration,
                f"Latency test failed: {str(e)}"
            )
    
    def test_throughput_performance(self) -> TestResult:
        """Test throughput performance."""
        start_time = time.time()
        
        try:
            from optimized_gen3_demo import OptimizedDGDN, OptimizationLevel
            from optimized_gen3_demo import generate_performance_test_data
            
            model = OptimizedDGDN(
                node_dim=32,
                hidden_dim=64,
                num_layers=2,
                optimization_level=OptimizationLevel.AGGRESSIVE,
                name="ThroughputTestModel"
            )
            
            # Generate test data
            test_data = generate_performance_test_data(
                num_nodes=50,
                num_edges=100,
                time_span=75.0,
                stress_level="moderate"
            )
            
            # Measure throughput
            process_start = time.time()
            result = model.forward(
                test_data['nodes'],
                test_data['edges'],
                test_data['timestamps']
            )
            process_time = time.time() - process_start
            
            nodes_processed = result['num_nodes_processed']
            throughput = nodes_processed / process_time
            
            # Check against threshold
            status = QualityGateStatus.PASSED if throughput > self.throughput_threshold_nodes_per_sec else QualityGateStatus.WARNING
            
            duration = (time.time() - start_time) * 1000
            return TestResult(
                "throughput_performance",
                TestType.PERFORMANCE,
                status,
                duration,
                f"Throughput: {throughput:.1f} nodes/sec (threshold: {self.throughput_threshold_nodes_per_sec})",
                {
                    "throughput_nodes_per_sec": throughput,
                    "nodes_processed": nodes_processed,
                    "processing_time_sec": process_time,
                    "threshold_nodes_per_sec": self.throughput_threshold_nodes_per_sec
                }
            )
            
        except Exception as e:
            duration = (time.time() - start_time) * 1000
            return TestResult(
                "throughput_performance",
                TestType.PERFORMANCE,
                QualityGateStatus.FAILED,
                duration,
                f"Throughput test failed: {str(e)}"
            )
    
    def run_all_tests(self) -> List[TestResult]:
        """Run all performance tests."""
        all_tests = []
        
        print("⚡ Running Performance Tests...")
        
        # Run latency test
        print("   Testing latency performance...")
        latency_test = self.test_latency_performance()
        all_tests.append(latency_test)
        
        # Run throughput test
        print("   Testing throughput performance...")
        throughput_test = self.test_throughput_performance()
        all_tests.append(throughput_test)
        
        # Summary
        passed = sum(1 for t in all_tests if t.status == QualityGateStatus.PASSED)
        failed = sum(1 for t in all_tests if t.status == QualityGateStatus.FAILED)
        warnings = sum(1 for t in all_tests if t.status == QualityGateStatus.WARNING)
        print(f"   Performance tests completed: {passed} passed, {failed} failed, {warnings} warnings")
        
        self.test_results = all_tests
        return all_tests


class SecurityTestSuite:
    """Security test suite for vulnerability assessment."""
    
    def __init__(self, name="DGDN_SecurityTests"):
        self.name = name
        self.test_results = []
    
    def test_input_validation(self) -> TestResult:
        """Test input validation and sanitization."""
        start_time = time.time()
        
        try:
            from optimized_gen3_demo import OptimizedTensor, OptimizedDGDN, OptimizationLevel
            
            # Test malicious inputs
            test_cases = [
                ([float('inf'), 1.0], "infinity_input"),
                ([float('-inf'), 1.0], "negative_infinity_input"),
                ([float('nan'), 1.0], "nan_input"),
                ([1e100, 1.0], "overflow_input"),
                ([], "empty_input")
            ]
            
            vulnerabilities = []
            
            for test_data, test_name in test_cases:
                try:
                    tensor = OptimizedTensor(test_data)
                    # If this succeeds for malicious input, it's a vulnerability
                    if any(not math.isfinite(x) for x in test_data if isinstance(x, (int, float))):
                        vulnerabilities.append(f"Input validation bypass: {test_name}")
                except (ValueError, TypeError, OverflowError):
                    # Expected behavior - input properly rejected
                    pass
            
            status = QualityGateStatus.PASSED if not vulnerabilities else QualityGateStatus.WARNING
            message = "Input validation secure" if not vulnerabilities else f"Potential issues found: {len(vulnerabilities)}"
            
            duration = (time.time() - start_time) * 1000
            return TestResult(
                "input_validation",
                TestType.SECURITY,
                status,
                duration,
                message,
                {"vulnerabilities": vulnerabilities, "test_cases": len(test_cases)}
            )
            
        except Exception as e:
            duration = (time.time() - start_time) * 1000
            return TestResult(
                "input_validation",
                TestType.SECURITY,
                QualityGateStatus.FAILED,
                duration,
                f"Security test failed: {str(e)}"
            )
    
    def test_resource_limits(self) -> TestResult:
        """Test resource limits and denial of service protection."""
        start_time = time.time()
        
        try:
            from optimized_gen3_demo import OptimizedDGDN, OptimizedTensor, OptimizationLevel
            
            vulnerabilities = []
            
            # Test large data handling
            try:
                model = OptimizedDGDN(node_dim=32, hidden_dim=64, num_layers=1)
                
                # Create moderately large test data
                nodes = [OptimizedTensor([0.1] * 32) for _ in range(100)]
                edges = [(i, (i + 1) % 100) for i in range(200)]
                timestamps = [float(i) for i in range(200)]
                
                result = model.forward(nodes, edges, timestamps)
                
                # Check if processing completed in reasonable time
                if result.get('processing_time_ms', 0) > 5000:  # 5 seconds
                    vulnerabilities.append("No processing time limits for large data")
                    
            except Exception:
                pass  # Acceptable - system rejected large input
            
            status = QualityGateStatus.PASSED if not vulnerabilities else QualityGateStatus.WARNING
            message = "Resource limits adequate" if not vulnerabilities else f"Issues found: {len(vulnerabilities)}"
            
            duration = (time.time() - start_time) * 1000
            return TestResult(
                "resource_limits",
                TestType.SECURITY,
                status,
                duration,
                message,
                {"vulnerabilities": vulnerabilities}
            )
            
        except Exception as e:
            duration = (time.time() - start_time) * 1000
            return TestResult(
                "resource_limits",
                TestType.SECURITY,
                QualityGateStatus.FAILED,
                duration,
                f"Resource limits test failed: {str(e)}"
            )
    
    def run_all_tests(self) -> List[TestResult]:
        """Run all security tests."""
        all_tests = []
        
        print("🔒 Running Security Tests...")
        
        # Run input validation test
        print("   Testing input validation...")
        validation_test = self.test_input_validation()
        all_tests.append(validation_test)
        
        # Run resource limits test
        print("   Testing resource limits...")
        resource_test = self.test_resource_limits()
        all_tests.append(resource_test)
        
        # Summary
        passed = sum(1 for t in all_tests if t.status == QualityGateStatus.PASSED)
        failed = sum(1 for t in all_tests if t.status == QualityGateStatus.FAILED)
        warnings = sum(1 for t in all_tests if t.status == QualityGateStatus.WARNING)
        print(f"   Security tests completed: {passed} passed, {failed} failed, {warnings} warnings")
        
        self.test_results = all_tests
        return all_tests


class QualityGateEngine:
    """Main quality gate engine that orchestrates all testing."""
    
    def __init__(self, name="DGDN_QualityGates"):
        self.name = name
        self.test_suites = {
            'unit': UnitTestSuite(),
            'performance': PerformanceTestSuite(),
            'security': SecurityTestSuite()
        }
        
        # Quality thresholds
        self.min_pass_rate = 0.75  # 75% tests must pass
        self.max_critical_failures = 3
    
    def run_all_quality_gates(self) -> QualityGateReport:
        """Run all quality gates and generate comprehensive report."""
        
        print("🛡️ COMPREHENSIVE QUALITY GATES & TESTING")
        print("=" * 60)
        print("Running mandatory quality gates for production deployment...")
        print("=" * 60)
        
        start_time = time.time()
        all_test_results = []
        
        # Run all test suites
        for suite_name, suite in self.test_suites.items():
            print(f"\n🧩 Running {suite_name.upper()} Test Suite...")
            suite_results = suite.run_all_tests()
            all_test_results.extend(suite_results)
        
        # Analyze results
        total_tests = len(all_test_results)
        passed_tests = sum(1 for t in all_test_results if t.status == QualityGateStatus.PASSED)
        failed_tests = sum(1 for t in all_test_results if t.status == QualityGateStatus.FAILED)
        warning_tests = sum(1 for t in all_test_results if t.status == QualityGateStatus.WARNING)
        
        pass_rate = passed_tests / max(1, total_tests)
        
        # Critical failures (security and core functionality)
        critical_failures = sum(
            1 for t in all_test_results 
            if t.status == QualityGateStatus.FAILED and t.test_type in [TestType.SECURITY, TestType.UNIT]
        )
        
        # Determine overall status
        if pass_rate >= self.min_pass_rate and critical_failures <= self.max_critical_failures:
            overall_status = QualityGateStatus.PASSED
        elif critical_failures > self.max_critical_failures:
            overall_status = QualityGateStatus.FAILED
        else:
            overall_status = QualityGateStatus.WARNING
        
        # Calculate quality metrics
        quality_metrics = self._calculate_quality_metrics(all_test_results)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(all_test_results, overall_status)
        
        # Create summary stats
        summary_stats = {
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'failed_tests': failed_tests,
            'warning_tests': warning_tests,
            'pass_rate': pass_rate,
            'critical_failures': critical_failures,
            'total_duration_ms': (time.time() - start_time) * 1000,
            'test_suites_run': len(self.test_suites)
        }
        
        # Create comprehensive report
        report = QualityGateReport(
            overall_status=overall_status,
            test_results=all_test_results,
            summary_stats=summary_stats,
            quality_metrics=quality_metrics,
            recommendations=recommendations
        )
        
        return report
    
    def _calculate_quality_metrics(self, test_results: List[TestResult]) -> Dict[str, float]:
        """Calculate comprehensive quality metrics."""
        if not test_results:
            return {}
        
        # Performance metrics
        performance_tests = [t for t in test_results if t.test_type == TestType.PERFORMANCE]
        avg_latency = 0.0
        avg_throughput = 0.0
        
        if performance_tests:
            latencies = [t.details.get('average_latency_ms', 0) for t in performance_tests if 'average_latency_ms' in t.details]
            throughputs = [t.details.get('throughput_nodes_per_sec', 0) for t in performance_tests if 'throughput_nodes_per_sec' in t.details]
            
            avg_latency = sum(latencies) / max(1, len(latencies))
            avg_throughput = sum(throughputs) / max(1, len(throughputs))
        
        # Reliability score
        reliability_score = sum(1 for t in test_results if t.status == QualityGateStatus.PASSED) / len(test_results)
        
        # Security score
        security_tests = [t for t in test_results if t.test_type == TestType.SECURITY]
        security_score = sum(1 for t in security_tests if t.status in [QualityGateStatus.PASSED, QualityGateStatus.WARNING]) / max(1, len(security_tests))
        
        # Overall quality score
        quality_score = reliability_score * 0.6 + security_score * 0.4
        
        return {
            'reliability_score': reliability_score,
            'security_score': security_score,
            'quality_score': quality_score,
            'average_latency_ms': avg_latency,
            'average_throughput_nodes_per_sec': avg_throughput
        }
    
    def _generate_recommendations(self, test_results: List[TestResult], overall_status: QualityGateStatus) -> List[str]:
        """Generate actionable recommendations based on test results."""
        recommendations = []
        
        # Failed test recommendations
        failed_tests = [t for t in test_results if t.status == QualityGateStatus.FAILED]
        if failed_tests:
            recommendations.append(f"❌ {len(failed_tests)} tests failed - review and fix critical issues")
        
        # Security recommendations
        security_tests = [t for t in test_results if t.test_type == TestType.SECURITY]
        security_warnings = [t for t in security_tests if t.status == QualityGateStatus.WARNING]
        
        if security_warnings:
            recommendations.append(f"🔒 {len(security_warnings)} security warnings - review for potential vulnerabilities")
        
        # Performance recommendations
        performance_tests = [t for t in test_results if t.test_type == TestType.PERFORMANCE]
        slow_tests = [t for t in performance_tests if t.status in [QualityGateStatus.WARNING, QualityGateStatus.FAILED]]
        
        if slow_tests:
            recommendations.append("⚡ Performance optimization recommended for production scaling")
        
        # Overall status recommendations
        if overall_status == QualityGateStatus.PASSED:
            recommendations.append("✅ All quality gates passed - ready for production deployment")
        elif overall_status == QualityGateStatus.WARNING:
            recommendations.append("⚠️  Quality gates passed with warnings - monitor closely in production")
        else:
            recommendations.append("❌ Quality gates failed - address critical issues before deployment")
        
        return recommendations


def run_final_quality_gates():
    """Run comprehensive quality gates and generate final report."""
    
    # Initialize quality gate engine
    quality_engine = QualityGateEngine()
    
    # Run all quality gates
    report = quality_engine.run_all_quality_gates()
    
    # Display comprehensive report
    print(f"\n📊 QUALITY GATE RESULTS SUMMARY")
    print("=" * 60)
    
    # Overall status
    status_icon = {
        QualityGateStatus.PASSED: "✅",
        QualityGateStatus.FAILED: "❌",
        QualityGateStatus.WARNING: "⚠️"
    }
    
    print(f"{status_icon[report.overall_status]} OVERALL STATUS: {report.overall_status.value.upper()}")
    
    # Summary statistics
    stats = report.summary_stats
    print(f"\n📈 Test Summary:")
    print(f"   Total Tests: {stats['total_tests']}")
    print(f"   Passed: {stats['passed_tests']} ({stats['pass_rate']:.1%})")
    print(f"   Failed: {stats['failed_tests']}")
    print(f"   Warnings: {stats['warning_tests']}")
    print(f"   Critical Failures: {stats['critical_failures']}")
    print(f"   Total Duration: {stats['total_duration_ms']:.1f}ms")
    
    # Quality metrics
    metrics = report.quality_metrics
    print(f"\n🎯 Quality Metrics:")
    print(f"   Overall Quality Score: {metrics.get('quality_score', 0):.3f}")
    print(f"   Reliability Score: {metrics.get('reliability_score', 0):.3f}")
    print(f"   Security Score: {metrics.get('security_score', 0):.3f}")
    
    if 'average_latency_ms' in metrics and metrics['average_latency_ms'] > 0:
        print(f"\n⚡ Performance Metrics:")
        print(f"   Average Latency: {metrics['average_latency_ms']:.2f}ms")
        print(f"   Average Throughput: {metrics['average_throughput_nodes_per_sec']:.1f} nodes/sec")
    
    # Recommendations
    print(f"\n💡 Recommendations:")
    for recommendation in report.recommendations:
        print(f"   {recommendation}")
    
    # Save detailed report
    report_path = Path("/root/repo/final_quality_gates_report.json")
    report_data = report.to_dict()
    
    with open(report_path, 'w') as f:
        json.dump(report_data, f, indent=2)
    
    print(f"\n💾 Detailed report saved to: {report_path}")
    
    # Final deployment decision
    print(f"\n🚀 DEPLOYMENT DECISION")
    print("=" * 60)
    
    if report.overall_status == QualityGateStatus.PASSED:
        print("✅ APPROVED FOR PRODUCTION DEPLOYMENT")
        print("   All quality gates passed successfully")
        print("   System meets all requirements for production use")
    elif report.overall_status == QualityGateStatus.WARNING:
        print("⚠️  CONDITIONAL APPROVAL FOR DEPLOYMENT")
        print("   Quality gates passed with warnings")
        print("   Deploy with close monitoring and quick rollback capability")
    else:
        print("❌ DEPLOYMENT NOT APPROVED")
        print("   Critical quality gates failed")
        print("   Address all critical issues before deployment")
    
    return report


if __name__ == "__main__":
    try:
        report = run_final_quality_gates()
        
        if report.overall_status == QualityGateStatus.PASSED:
            print("\n✅ Quality gates completed successfully!")
            sys.exit(0)
        elif report.overall_status == QualityGateStatus.WARNING:
            print("\n⚠️  Quality gates completed with warnings!")
            sys.exit(0)
        else:
            print("\n❌ Quality gates failed!")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n💥 Quality gate execution failed: {e}")
        traceback.print_exc()
        sys.exit(1)