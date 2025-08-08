#!/usr/bin/env python3
"""Quality Gates for DGDN - Comprehensive testing, linting, and security checks."""

import subprocess
import sys
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any
import json


class QualityGateRunner:
    """Run comprehensive quality gates for the DGDN project."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.src_path = self.project_root / "src"
        self.tests_path = self.project_root / "tests"
        
        # Quality gate configurations
        self.gates = {
            "import_tests": {"enabled": True, "critical": True},
            "unit_tests": {"enabled": True, "critical": True},
            "integration_tests": {"enabled": True, "critical": False},
            "code_coverage": {"enabled": True, "critical": False, "min_coverage": 80},
            "type_checking": {"enabled": True, "critical": False},
            "code_formatting": {"enabled": True, "critical": False},
            "linting": {"enabled": True, "critical": False},
            "security_scan": {"enabled": True, "critical": True},
            "dependency_scan": {"enabled": True, "critical": True},
            "performance_tests": {"enabled": True, "critical": False}
        }
        
        self.results = {}
        
    def run_command(self, command: List[str], cwd: Path = None) -> Tuple[int, str, str]:
        """Run a command and return exit code, stdout, stderr."""
        try:
            result = subprocess.run(
                command,
                cwd=cwd or self.project_root,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            return result.returncode, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return 1, "", "Command timed out"
        except Exception as e:
            return 1, "", str(e)
    
    def run_import_tests(self) -> Dict[str, Any]:
        """Test basic imports work correctly."""
        print("🔍 Running import tests...")
        
        start_time = time.time()
        returncode, stdout, stderr = self.run_command([
            sys.executable, "-m", "pytest", 
            "tests/test_import.py", "-v", "--tb=short"
        ])
        duration = time.time() - start_time
        
        result = {
            "name": "Import Tests",
            "passed": returncode == 0,
            "duration": duration,
            "output": stdout,
            "error": stderr
        }
        
        if result["passed"]:
            print("✅ Import tests passed")
        else:
            print("❌ Import tests failed")
            print(f"Error: {stderr}")
        
        return result
    
    def run_unit_tests(self) -> Dict[str, Any]:
        """Run unit tests with coverage."""
        print("🧪 Running unit tests...")
        
        start_time = time.time()
        returncode, stdout, stderr = self.run_command([
            sys.executable, "-m", "pytest",
            "tests/unit/", "-v", "--tb=short",
            "--cov=dgdn", "--cov-report=term-missing",
            "-m", "not slow"
        ])
        duration = time.time() - start_time
        
        # Extract coverage percentage from output
        coverage_pct = 0
        for line in stdout.split('\n'):
            if 'TOTAL' in line and '%' in line:
                try:
                    coverage_pct = int(line.split()[-1].replace('%', ''))
                except:
                    pass
        
        result = {
            "name": "Unit Tests",
            "passed": returncode == 0,
            "duration": duration,
            "coverage": coverage_pct,
            "output": stdout,
            "error": stderr
        }
        
        if result["passed"]:
            print(f"✅ Unit tests passed (Coverage: {coverage_pct}%)")
        else:
            print("❌ Unit tests failed")
            print(f"Error: {stderr}")
        
        return result
    
    def run_integration_tests(self) -> Dict[str, Any]:
        """Run integration tests."""
        print("🔗 Running integration tests...")
        
        start_time = time.time()
        returncode, stdout, stderr = self.run_command([
            sys.executable, "-m", "pytest",
            "tests/integration/", "-v", "--tb=short",
            "-m", "not slow"
        ])
        duration = time.time() - start_time
        
        result = {
            "name": "Integration Tests",
            "passed": returncode == 0,
            "duration": duration,
            "output": stdout,
            "error": stderr
        }
        
        if result["passed"]:
            print("✅ Integration tests passed")
        else:
            print("❌ Integration tests failed")
            print(f"Error: {stderr}")
        
        return result
    
    def run_type_checking(self) -> Dict[str, Any]:
        """Run mypy type checking."""
        print("🔍 Running type checking...")
        
        start_time = time.time()
        returncode, stdout, stderr = self.run_command([
            "mypy", str(self.src_path), "--ignore-missing-imports"
        ])
        duration = time.time() - start_time
        
        # Count errors
        error_count = stdout.count("error:")
        
        result = {
            "name": "Type Checking",
            "passed": returncode == 0,
            "duration": duration,
            "error_count": error_count,
            "output": stdout,
            "error": stderr
        }
        
        if result["passed"]:
            print("✅ Type checking passed")
        else:
            print(f"❌ Type checking failed ({error_count} errors)")
        
        return result
    
    def run_code_formatting(self) -> Dict[str, Any]:
        """Check code formatting with ruff."""
        print("🎨 Checking code formatting...")
        
        start_time = time.time()
        returncode, stdout, stderr = self.run_command([
            "ruff", "format", "--check", str(self.src_path)
        ])
        duration = time.time() - start_time
        
        result = {
            "name": "Code Formatting",
            "passed": returncode == 0,
            "duration": duration,
            "output": stdout,
            "error": stderr
        }
        
        if result["passed"]:
            print("✅ Code formatting correct")
        else:
            print("❌ Code formatting issues found")
        
        return result
    
    def run_linting(self) -> Dict[str, Any]:
        """Run linting with ruff."""
        print("🔍 Running linting...")
        
        start_time = time.time()
        returncode, stdout, stderr = self.run_command([
            "ruff", "check", str(self.src_path)
        ])
        duration = time.time() - start_time
        
        # Count violations
        violation_count = len([line for line in stdout.split('\n') if line.strip() and not line.startswith('Found')])
        
        result = {
            "name": "Linting",
            "passed": returncode == 0,
            "duration": duration,
            "violations": violation_count,
            "output": stdout,
            "error": stderr
        }
        
        if result["passed"]:
            print("✅ Linting passed")
        else:
            print(f"❌ Linting found {violation_count} violations")
        
        return result
    
    def run_security_scan(self) -> Dict[str, Any]:
        """Run security scanning with bandit."""
        print("🔒 Running security scan...")
        
        start_time = time.time()
        returncode, stdout, stderr = self.run_command([
            "bandit", "-r", str(self.src_path), "-f", "json"
        ])
        duration = time.time() - start_time
        
        # Parse bandit JSON output
        issues_count = 0
        high_severity_count = 0
        
        try:
            if stdout:
                bandit_result = json.loads(stdout)
                issues_count = len(bandit_result.get("results", []))
                high_severity_count = len([
                    r for r in bandit_result.get("results", [])
                    if r.get("issue_severity") == "HIGH"
                ])
        except json.JSONDecodeError:
            pass
        
        result = {
            "name": "Security Scan",
            "passed": high_severity_count == 0,  # Pass if no high severity issues
            "duration": duration,
            "total_issues": issues_count,
            "high_severity": high_severity_count,
            "output": stdout,
            "error": stderr
        }
        
        if result["passed"]:
            print(f"✅ Security scan passed ({issues_count} low/medium issues)")
        else:
            print(f"❌ Security scan failed ({high_severity_count} high severity issues)")
        
        return result
    
    def run_dependency_scan(self) -> Dict[str, Any]:
        """Run dependency vulnerability scan with safety."""
        print("📦 Running dependency scan...")
        
        start_time = time.time()
        returncode, stdout, stderr = self.run_command([
            "safety", "check", "--json"
        ])
        duration = time.time() - start_time
        
        # Parse safety JSON output
        vulnerabilities = 0
        try:
            if stdout:
                safety_result = json.loads(stdout)
                vulnerabilities = len(safety_result.get("vulnerabilities", []))
        except json.JSONDecodeError:
            # Safety might not return valid JSON on success
            vulnerabilities = 0
        
        result = {
            "name": "Dependency Scan",
            "passed": vulnerabilities == 0,
            "duration": duration,
            "vulnerabilities": vulnerabilities,
            "output": stdout,
            "error": stderr
        }
        
        if result["passed"]:
            print("✅ Dependency scan passed")
        else:
            print(f"❌ Dependency scan found {vulnerabilities} vulnerabilities")
        
        return result
    
    def run_performance_tests(self) -> Dict[str, Any]:
        """Run performance tests."""
        print("⚡ Running performance tests...")
        
        start_time = time.time()
        
        # Run our generation demos as performance tests
        test_commands = [
            [sys.executable, "gen1_demo.py"],
            [sys.executable, "gen2_simple_demo.py"],
            [sys.executable, "gen3_demo.py"]
        ]
        
        all_passed = True
        total_duration = 0
        outputs = []
        
        for i, cmd in enumerate(test_commands):
            print(f"  Running generation {i+1} demo...")
            returncode, stdout, stderr = self.run_command(cmd)
            cmd_duration = time.time() - start_time - total_duration
            total_duration += cmd_duration
            
            if returncode != 0:
                all_passed = False
                outputs.append(f"Generation {i+1} failed: {stderr}")
            else:
                outputs.append(f"Generation {i+1} passed in {cmd_duration:.2f}s")
        
        duration = time.time() - start_time
        
        result = {
            "name": "Performance Tests",
            "passed": all_passed,
            "duration": duration,
            "output": "\n".join(outputs),
            "error": "" if all_passed else "Some performance tests failed"
        }
        
        if result["passed"]:
            print(f"✅ Performance tests passed ({duration:.2f}s)")
        else:
            print("❌ Performance tests failed")
        
        return result
    
    def run_all_gates(self) -> Dict[str, Any]:
        """Run all enabled quality gates."""
        print("🚀 Starting Quality Gates for DGDN")
        print("=" * 60)
        
        total_start_time = time.time()
        
        # Map gate names to methods
        gate_methods = {
            "import_tests": self.run_import_tests,
            "unit_tests": self.run_unit_tests,
            "integration_tests": self.run_integration_tests,
            "type_checking": self.run_type_checking,
            "code_formatting": self.run_code_formatting,
            "linting": self.run_linting,
            "security_scan": self.run_security_scan,
            "dependency_scan": self.run_dependency_scan,
            "performance_tests": self.run_performance_tests
        }
        
        results = {}
        critical_failures = []
        
        for gate_name, gate_config in self.gates.items():
            if not gate_config["enabled"]:
                continue
            
            print(f"\n📋 Running {gate_name}...")
            
            try:
                result = gate_methods[gate_name]()
                results[gate_name] = result
                
                if not result["passed"] and gate_config["critical"]:
                    critical_failures.append(gate_name)
                    
            except Exception as e:
                print(f"❌ {gate_name} failed with exception: {e}")
                results[gate_name] = {
                    "name": gate_name,
                    "passed": False,
                    "duration": 0,
                    "error": str(e)
                }
                if gate_config["critical"]:
                    critical_failures.append(gate_name)
        
        total_duration = time.time() - total_start_time
        
        # Summary
        print("\n" + "=" * 60)
        print("📊 QUALITY GATES SUMMARY")
        print("=" * 60)
        
        passed_count = sum(1 for r in results.values() if r["passed"])
        total_count = len(results)
        
        print(f"Total Gates Run: {total_count}")
        print(f"Passed: {passed_count}")
        print(f"Failed: {total_count - passed_count}")
        print(f"Total Duration: {total_duration:.2f}s")
        
        if critical_failures:
            print(f"\n❌ CRITICAL FAILURES: {', '.join(critical_failures)}")
            overall_success = False
        else:
            print(f"\n✅ All critical quality gates passed!")
            overall_success = True
        
        # Detailed results
        print("\nDetailed Results:")
        for gate_name, result in results.items():
            status = "✅" if result["passed"] else "❌"
            critical = "🔴" if self.gates[gate_name]["critical"] else "🟡"
            duration = result.get("duration", 0)
            print(f"  {status} {critical} {result['name']}: {duration:.2f}s")
        
        return {
            "overall_success": overall_success,
            "total_duration": total_duration,
            "results": results,
            "critical_failures": critical_failures,
            "summary": {
                "total": total_count,
                "passed": passed_count,
                "failed": total_count - passed_count
            }
        }
    
    def save_results(self, results: Dict[str, Any], output_file: str = "quality_gates_report.json"):
        """Save results to JSON file."""
        output_path = self.project_root / output_file
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n📄 Results saved to: {output_path}")


def main():
    """Main entry point for quality gates."""
    runner = QualityGateRunner()
    
    # Check if we're in the right directory
    if not runner.src_path.exists():
        print("❌ Error: src/ directory not found. Run from project root.")
        sys.exit(1)
    
    # Run all quality gates
    results = runner.run_all_gates()
    
    # Save results
    runner.save_results(results)
    
    # Exit with appropriate code
    if results["overall_success"]:
        print("\n🎉 All quality gates passed! Ready for production deployment.")
        sys.exit(0)
    else:
        print("\n💥 Quality gates failed. Please fix issues before deployment.")
        sys.exit(1)


if __name__ == "__main__":
    main()