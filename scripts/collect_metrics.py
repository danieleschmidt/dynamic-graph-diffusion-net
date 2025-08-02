#!/usr/bin/env python3
"""
Automated metrics collection script for DGDN project.

This script collects various project metrics including code quality,
performance, security, and repository health metrics.
"""

import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional
import os
import sys

class MetricsCollector:
    """Collects and aggregates project metrics."""
    
    def __init__(self, config_path: str = ".github/project-metrics.json"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.metrics = {}
        self.timestamp = datetime.now(timezone.utc).isoformat()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load metrics configuration."""
        try:
            with open(self.config_path) as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Config file {self.config_path} not found. Using defaults.")
            return self._default_config()
    
    def _default_config(self) -> Dict[str, Any]:
        """Return default configuration."""
        return {
            "metrics": {
                "code_quality": {},
                "performance": {},
                "development": {},
                "repository": {},
                "security": {}
            }
        }
    
    def collect_all_metrics(self) -> Dict[str, Any]:
        """Collect all available metrics."""
        print("🔍 Collecting project metrics...")
        
        self.collect_code_quality_metrics()
        self.collect_performance_metrics()
        self.collect_development_metrics()
        self.collect_repository_metrics()
        self.collect_security_metrics()
        
        return self.metrics
    
    def collect_code_quality_metrics(self):
        """Collect code quality metrics."""
        print("  📊 Collecting code quality metrics...")
        
        self.metrics["code_quality"] = {
            "timestamp": self.timestamp,
            "test_coverage": self._get_test_coverage(),
            "code_complexity": self._get_code_complexity(),
            "documentation_coverage": self._get_documentation_coverage(),
            "linting_violations": self._get_linting_violations(),
            "type_coverage": self._get_type_coverage(),
            "duplicate_code": self._get_duplicate_code_percentage()
        }
    
    def collect_performance_metrics(self):
        """Collect performance metrics."""
        print("  🚀 Collecting performance metrics...")
        
        self.metrics["performance"] = {
            "timestamp": self.timestamp,
            "training_speed": self._get_training_speed(),
            "inference_latency": self._get_inference_latency(),
            "memory_usage": self._get_memory_usage(),
            "model_accuracy": self._get_model_accuracy(),
            "gpu_utilization": self._get_gpu_utilization(),
            "disk_io": self._get_disk_io_metrics()
        }
    
    def collect_development_metrics(self):
        """Collect development workflow metrics."""
        print("  🛠️ Collecting development metrics...")
        
        self.metrics["development"] = {
            "timestamp": self.timestamp,
            "build_time": self._get_build_time(),
            "test_execution_time": self._get_test_execution_time(),
            "deployment_time": self._get_deployment_time(),
            "docker_build_time": self._get_docker_build_time(),
            "dependency_count": self._get_dependency_count()
        }
    
    def collect_repository_metrics(self):
        """Collect repository health metrics."""
        print("  📈 Collecting repository metrics...")
        
        self.metrics["repository"] = {
            "timestamp": self.timestamp,
            "commit_frequency": self._get_commit_frequency(),
            "pr_merge_time": self._get_pr_merge_time(),
            "issue_resolution_time": self._get_issue_resolution_time(),
            "active_contributors": self._get_active_contributors(),
            "code_churn": self._get_code_churn(),
            "branch_count": self._get_branch_count()
        }
    
    def collect_security_metrics(self):
        """Collect security metrics."""
        print("  🔒 Collecting security metrics...")
        
        self.metrics["security"] = {
            "timestamp": self.timestamp,
            "vulnerability_count": self._get_vulnerability_count(),
            "dependency_freshness": self._get_dependency_freshness(),
            "security_scan_frequency": self._get_security_scan_frequency(),
            "license_compliance": self._check_license_compliance(),
            "secrets_scan": self._check_secrets_scan()
        }
    
    def _run_command(self, command: str, cwd: Optional[str] = None) -> str:
        """Run shell command and return output."""
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                cwd=cwd,
                timeout=60
            )
            if result.returncode == 0:
                return result.stdout.strip()
            else:
                print(f"Command failed: {command}")
                print(f"Error: {result.stderr}")
                return ""
        except subprocess.TimeoutExpired:
            print(f"Command timed out: {command}")
            return ""
        except Exception as e:
            print(f"Error running command {command}: {e}")
            return ""
    
    def _get_test_coverage(self) -> float:
        """Get test coverage percentage."""
        try:
            output = self._run_command("pytest --cov=dgdn --cov-report=json --no-cov-report")
            if os.path.exists("coverage.json"):
                with open("coverage.json") as f:
                    coverage_data = json.load(f)
                    return coverage_data.get("totals", {}).get("percent_covered", 0.0)
        except Exception as e:
            print(f"Error getting test coverage: {e}")
        return 0.0
    
    def _get_code_complexity(self) -> float:
        """Get average cyclomatic complexity."""
        try:
            output = self._run_command("radon cc src/ --json")
            if output:
                complexity_data = json.loads(output)
                total_complexity = 0
                total_functions = 0
                
                for file_data in complexity_data.values():
                    for item in file_data:
                        if item.get("type") == "function":
                            total_complexity += item.get("complexity", 0)
                            total_functions += 1
                
                return total_complexity / total_functions if total_functions > 0 else 0.0
        except Exception as e:
            print(f"Error getting code complexity: {e}")
        return 0.0
    
    def _get_documentation_coverage(self) -> float:
        """Get documentation coverage percentage."""
        try:
            output = self._run_command("interrogate src/ --format json")
            if output:
                doc_data = json.loads(output)
                return doc_data.get("result", {}).get("coverage", 0.0)
        except Exception as e:
            print(f"Error getting documentation coverage: {e}")
        return 0.0
    
    def _get_linting_violations(self) -> int:
        """Get number of linting violations."""
        try:
            output = self._run_command("flake8 src/ --format=json")
            if output:
                violations = json.loads(output)
                return len(violations)
        except Exception as e:
            print(f"Error getting linting violations: {e}")
        return 0
    
    def _get_type_coverage(self) -> float:
        """Get type annotation coverage."""
        try:
            output = self._run_command("mypy src/ --json-report mypy-report")
            if os.path.exists("mypy-report/index.txt"):
                with open("mypy-report/index.txt") as f:
                    content = f.read()
                    # Parse mypy output for type coverage
                    return 0.0  # Placeholder
        except Exception as e:
            print(f"Error getting type coverage: {e}")
        return 0.0
    
    def _get_duplicate_code_percentage(self) -> float:
        """Get duplicate code percentage."""
        try:
            output = self._run_command("jscpd src/ --format json")
            if output:
                duplicate_data = json.loads(output)
                return duplicate_data.get("statistics", {}).get("percentage", 0.0)
        except Exception as e:
            print(f"Error getting duplicate code: {e}")
        return 0.0
    
    def _get_training_speed(self) -> float:
        """Get training speed in samples per second."""
        # This would integrate with actual training metrics
        return 0.0
    
    def _get_inference_latency(self) -> float:
        """Get inference latency in milliseconds."""
        # This would integrate with actual inference metrics
        return 0.0
    
    def _get_memory_usage(self) -> float:
        """Get memory usage in GB."""
        try:
            output = self._run_command("ps -o pid,ppid,%mem,rss,comm -ax | grep python")
            # Parse memory usage
            return 0.0
        except Exception as e:
            print(f"Error getting memory usage: {e}")
        return 0.0
    
    def _get_model_accuracy(self) -> float:
        """Get model accuracy percentage."""
        # This would integrate with actual model evaluation
        return 0.0
    
    def _get_gpu_utilization(self) -> float:
        """Get GPU utilization percentage."""
        try:
            output = self._run_command("nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits")
            if output:
                return float(output.split("\\n")[0])
        except Exception as e:
            print(f"Error getting GPU utilization: {e}")
        return 0.0
    
    def _get_disk_io_metrics(self) -> Dict[str, float]:
        """Get disk I/O metrics."""
        return {"read_bytes": 0.0, "write_bytes": 0.0}
    
    def _get_build_time(self) -> float:
        """Get average build time in seconds."""
        start_time = time.time()
        self._run_command("python -m build")
        return time.time() - start_time
    
    def _get_test_execution_time(self) -> float:
        """Get test execution time in seconds."""
        start_time = time.time()
        self._run_command("pytest tests/")
        return time.time() - start_time
    
    def _get_deployment_time(self) -> float:
        """Get deployment time in seconds."""
        start_time = time.time()
        self._run_command("docker build -t dgdn:test .")
        return time.time() - start_time
    
    def _get_docker_build_time(self) -> float:
        """Get Docker build time in seconds."""
        start_time = time.time()
        self._run_command("docker build --no-cache -t dgdn:metrics-test .")
        return time.time() - start_time
    
    def _get_dependency_count(self) -> int:
        """Get number of dependencies."""
        try:
            if os.path.exists("pyproject.toml"):
                with open("pyproject.toml") as f:
                    content = f.read()
                    # Count dependencies from pyproject.toml
                    return content.count("==") + content.count(">=") + content.count("~=")
        except Exception as e:
            print(f"Error getting dependency count: {e}")
        return 0
    
    def _get_commit_frequency(self) -> float:
        """Get commits per week over last month."""
        try:
            output = self._run_command("git log --since='4 weeks ago' --oneline")
            commits = len(output.split("\\n")) if output else 0
            return commits / 4.0  # Average per week
        except Exception as e:
            print(f"Error getting commit frequency: {e}")
        return 0.0
    
    def _get_pr_merge_time(self) -> float:
        """Get average PR merge time in hours."""
        # This would require GitHub API integration
        return 0.0
    
    def _get_issue_resolution_time(self) -> float:
        """Get average issue resolution time in hours."""
        # This would require GitHub API integration
        return 0.0
    
    def _get_active_contributors(self) -> int:
        """Get number of active contributors in last month."""
        try:
            output = self._run_command("git log --since='1 month ago' --format='%ae' | sort | uniq")
            return len(output.split("\\n")) if output else 0
        except Exception as e:
            print(f"Error getting active contributors: {e}")
        return 0
    
    def _get_code_churn(self) -> Dict[str, int]:
        """Get code churn metrics."""
        try:
            output = self._run_command("git log --since='1 week ago' --numstat --format=''")
            additions = 0
            deletions = 0
            for line in output.split("\\n"):
                if line.strip():
                    parts = line.split("\\t")
                    if len(parts) >= 2:
                        additions += int(parts[0]) if parts[0].isdigit() else 0
                        deletions += int(parts[1]) if parts[1].isdigit() else 0
            return {"additions": additions, "deletions": deletions}
        except Exception as e:
            print(f"Error getting code churn: {e}")
        return {"additions": 0, "deletions": 0}
    
    def _get_branch_count(self) -> int:
        """Get number of branches."""
        try:
            output = self._run_command("git branch -a")
            return len(output.split("\\n")) if output else 0
        except Exception as e:
            print(f"Error getting branch count: {e}")
        return 0
    
    def _get_vulnerability_count(self) -> int:
        """Get number of security vulnerabilities."""
        try:
            output = self._run_command("safety check --json")
            if output:
                safety_data = json.loads(output)
                return len(safety_data.get("vulnerabilities", []))
        except Exception as e:
            print(f"Error getting vulnerability count: {e}")
        return 0
    
    def _get_dependency_freshness(self) -> float:
        """Get average age of dependencies in days."""
        # This would require checking package registries
        return 0.0
    
    def _get_security_scan_frequency(self) -> float:
        """Get security scan frequency."""
        # This would track scan history
        return 0.0
    
    def _check_license_compliance(self) -> bool:
        """Check license compliance."""
        try:
            output = self._run_command("pip-licenses --format=json")
            if output:
                licenses = json.loads(output)
                # Check for problematic licenses
                problematic = ["GPL", "AGPL", "SSPL"]
                for pkg in licenses:
                    license_name = pkg.get("License", "")
                    if any(prob in license_name for prob in problematic):
                        return False
                return True
        except Exception as e:
            print(f"Error checking license compliance: {e}")
        return True
    
    def _check_secrets_scan(self) -> bool:
        """Check for exposed secrets."""
        try:
            output = self._run_command("truffleHog --json .")
            if output:
                secrets = json.loads(output)
                return len(secrets) == 0
        except Exception as e:
            print(f"Error checking secrets: {e}")
        return True
    
    def save_metrics(self, output_path: str = "metrics-report.json"):
        """Save collected metrics to file."""
        metrics_report = {
            "collection_timestamp": self.timestamp,
            "metrics": self.metrics,
            "metadata": {
                "collector_version": "1.0.0",
                "collection_duration": time.time() - time.time(),  # Would track actual duration
                "environment": {
                    "python_version": sys.version,
                    "platform": sys.platform,
                    "cwd": os.getcwd()
                }
            }
        }
        
        with open(output_path, "w") as f:
            json.dump(metrics_report, f, indent=2)
        
        print(f"✅ Metrics saved to {output_path}")
    
    def update_config(self):
        """Update configuration with current metrics."""
        for category, metrics in self.metrics.items():
            if category in self.config.get("metrics", {}):
                for metric_name, value in metrics.items():
                    if metric_name in self.config["metrics"][category]:
                        self.config["metrics"][category][metric_name]["current"] = value
                        self.config["metrics"][category][metric_name]["last_measured"] = self.timestamp
        
        with open(self.config_path, "w") as f:
            json.dump(self.config, f, indent=2)
        
        print(f"✅ Configuration updated: {self.config_path}")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Collect DGDN project metrics")
    parser.add_argument("--config", default=".github/project-metrics.json", help="Config file path")
    parser.add_argument("--output", default="metrics-report.json", help="Output file path")
    parser.add_argument("--update-config", action="store_true", help="Update config with current values")
    parser.add_argument("--category", choices=["code_quality", "performance", "development", "repository", "security"], help="Collect specific category only")
    
    args = parser.parse_args()
    
    collector = MetricsCollector(args.config)
    
    if args.category:
        print(f"🎯 Collecting {args.category} metrics only...")
        getattr(collector, f"collect_{args.category}_metrics")()
    else:
        collector.collect_all_metrics()
    
    collector.save_metrics(args.output)
    
    if args.update_config:
        collector.update_config()
    
    print("🎉 Metrics collection completed successfully!")


if __name__ == "__main__":
    main()