#!/usr/bin/env python3
"""
Deployment Verification Suite

Comprehensive verification of production deployment including
API endpoints, performance, security, and monitoring.
"""

import requests
import time
import json
import sys
from typing import Dict, Any, List
import concurrent.futures


class DeploymentVerification:
    """Production deployment verification."""
    
    def __init__(self, base_url: str = "http://localhost"):
        self.base_url = base_url
        self.results = {}
        self.errors = []
    
    def verify_api_health(self) -> bool:
        """Verify API health endpoint."""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=10)
            if response.status_code == 200:
                health_data = response.json()
                return health_data.get("status") == "healthy"
            return False
        except Exception as e:
            self.errors.append(f"Health check failed: {e}")
            return False
    
    def verify_api_endpoints(self) -> bool:
        """Verify all API endpoints."""
        endpoints = [
            ("/", 200),
            ("/health", 200),
            ("/metrics", 200),
            ("/docs", 200)
        ]
        
        all_passed = True
        for endpoint, expected_status in endpoints:
            try:
                response = requests.get(f"{self.base_url}{endpoint}", timeout=10)
                if response.status_code != expected_status:
                    self.errors.append(f"Endpoint {endpoint} returned {response.status_code}, expected {expected_status}")
                    all_passed = False
            except Exception as e:
                self.errors.append(f"Endpoint {endpoint} failed: {e}")
                all_passed = False
        
        return all_passed
    
    def verify_inference_api(self) -> bool:
        """Verify inference API functionality."""
        try:
            # Test data
            test_data = {
                "graph_data": {
                    "edge_index": [[0, 1, 2], [1, 2, 0]],
                    "timestamps": [1.0, 2.0, 3.0],
                    "node_features": [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
                    "num_nodes": 3
                },
                "return_attention": False,
                "return_uncertainty": False
            }
            
            response = requests.post(
                f"{self.base_url}/inference",
                json=test_data,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return "node_embeddings" in result and "inference_time" in result
            else:
                self.errors.append(f"Inference API returned {response.status_code}: {response.text}")
                return False
                
        except Exception as e:
            self.errors.append(f"Inference API failed: {e}")
            return False
    
    def verify_performance(self) -> bool:
        """Verify performance requirements."""
        try:
            # Performance test data
            test_data = {
                "graph_data": {
                    "edge_index": [[i for i in range(100)], [(i+1) % 100 for i in range(100)]],
                    "timestamps": [float(i) for i in range(100)],
                    "num_nodes": 100
                }
            }
            
            # Run multiple requests to test performance
            times = []
            for _ in range(5):
                start_time = time.time()
                response = requests.post(
                    f"{self.base_url}/inference",
                    json=test_data,
                    timeout=60
                )
                end_time = time.time()
                
                if response.status_code == 200:
                    times.append(end_time - start_time)
                else:
                    return False
            
            avg_time = sum(times) / len(times)
            return avg_time < 10.0  # 10 second limit
            
        except Exception as e:
            self.errors.append(f"Performance test failed: {e}")
            return False
    
    def verify_monitoring(self) -> bool:
        """Verify monitoring endpoints."""
        monitoring_endpoints = [
            ("http://localhost:9090/-/healthy", "Prometheus"),
            ("http://localhost:3000/api/health", "Grafana")
        ]
        
        all_passed = True
        for url, service in monitoring_endpoints:
            try:
                response = requests.get(url, timeout=10)
                if response.status_code != 200:
                    self.errors.append(f"{service} not healthy")
                    all_passed = False
            except Exception as e:
                self.errors.append(f"{service} check failed: {e}")
                all_passed = False
        
        return all_passed
    
    def verify_security(self) -> bool:
        """Verify basic security measures."""
        try:
            # Check security headers
            response = requests.get(f"{self.base_url}/", timeout=10)
            headers = response.headers
            
            security_checks = [
                ("X-Frame-Options" in headers, "X-Frame-Options header"),
                ("X-Content-Type-Options" in headers, "X-Content-Type-Options header"),
                (response.status_code != 500, "No server errors")
            ]
            
            all_passed = True
            for check, description in security_checks:
                if not check:
                    self.errors.append(f"Security check failed: {description}")
                    all_passed = False
            
            return all_passed
            
        except Exception as e:
            self.errors.append(f"Security verification failed: {e}")
            return False
    
    def run_verification(self) -> Dict[str, Any]:
        """Run complete deployment verification."""
        print("🔍 Starting deployment verification...")
        
        verifications = [
            ("API Health", self.verify_api_health),
            ("API Endpoints", self.verify_api_endpoints),
            ("Inference API", self.verify_inference_api),
            ("Performance", self.verify_performance),
            ("Monitoring", self.verify_monitoring),
            ("Security", self.verify_security)
        ]
        
        for name, verify_func in verifications:
            print(f"   Testing {name}...")
            passed = verify_func()
            self.results[name] = passed
            if passed:
                print(f"   ✅ {name} passed")
            else:
                print(f"   ❌ {name} failed")
        
        # Summary
        passed_count = sum(1 for result in self.results.values() if result)
        total_count = len(self.results)
        success_rate = passed_count / total_count if total_count > 0 else 0
        
        verification_result = {
            "overall_status": "PASSED" if passed_count == total_count else "FAILED",
            "passed": passed_count,
            "total": total_count,
            "success_rate": success_rate,
            "results": self.results,
            "errors": self.errors,
            "timestamp": time.time()
        }
        
        return verification_result


if __name__ == "__main__":
    verifier = DeploymentVerification()
    result = verifier.run_verification()
    
    print(f"\n📊 Verification Summary:")
    print(f"   Status: {result['overall_status']}")
    print(f"   Passed: {result['passed']}/{result['total']}")
    print(f"   Success Rate: {result['success_rate']:.1%}")
    
    if result['errors']:
        print(f"\n❌ Errors:")
        for error in result['errors']:
            print(f"   {error}")
    
    # Save results
    with open("deployment_verification_results.json", "w") as f:
        json.dump(result, f, indent=2)
    
    print(f"\n📁 Results saved to deployment_verification_results.json")
    
    sys.exit(0 if result['overall_status'] == 'PASSED' else 1)
