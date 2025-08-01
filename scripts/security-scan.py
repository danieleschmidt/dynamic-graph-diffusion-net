#!/usr/bin/env python3
"""
Automated security scanning script for DGDN project.
Integrates multiple security tools and generates comprehensive reports.
"""

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

class SecurityScanner:
    """Comprehensive security scanner for Python projects."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.security_dir = project_root / ".security"
        self.reports_dir = self.security_dir / "reports"
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
    def run_safety_check(self) -> Dict[str, Any]:
        """Run safety check for known security vulnerabilities."""
        print("Running safety check...")
        
        try:
            result = subprocess.run(
                ["safety", "check", "--json", "--full-report"],
                capture_output=True,
                text=True,
                cwd=self.project_root
            )
            
            if result.returncode == 0:
                return {"status": "success", "vulnerabilities": []}
            else:
                # Parse safety JSON output
                try:
                    vulnerabilities = json.loads(result.stdout)
                    return {
                        "status": "vulnerabilities_found",
                        "vulnerabilities": vulnerabilities,
                        "count": len(vulnerabilities)
                    }
                except json.JSONDecodeError:
                    return {
                        "status": "error",
                        "error": "Failed to parse safety output",
                        "output": result.stdout
                    }
                    
        except FileNotFoundError:
            return {
                "status": "error",
                "error": "Safety tool not installed. Install with: pip install safety"
            }
    
    def run_bandit_scan(self) -> Dict[str, Any]:
        """Run bandit security linting."""
        print("Running bandit security scan...")
        
        try:
            config_file = self.security_dir / "bandit.yml"
            result = subprocess.run([
                "bandit",
                "-r", "src/",
                "-f", "json",
                "-c", str(config_file) if config_file.exists() else "",
            ], capture_output=True, text=True, cwd=self.project_root)
            
            try:
                report = json.loads(result.stdout)
                return {
                    "status": "success",
                    "results": report.get("results", []),
                    "metrics": report.get("metrics", {}),
                    "errors": report.get("errors", [])
                }
            except json.JSONDecodeError:
                return {
                    "status": "error",
                    "error": "Failed to parse bandit output",
                    "output": result.stdout
                }
                
        except FileNotFoundError:
            return {
                "status": "error", 
                "error": "Bandit tool not installed. Install with: pip install bandit"
            }
    
    def run_pip_audit(self) -> Dict[str, Any]:
        """Run pip-audit for Python package vulnerabilities."""
        print("Running pip-audit...")
        
        try:
            result = subprocess.run([
                "pip-audit",
                "--format=json",
                "--desc"
            ], capture_output=True, text=True, cwd=self.project_root)
            
            if result.returncode == 0:
                try:
                    audit_results = json.loads(result.stdout)
                    return {
                        "status": "success",
                        "vulnerabilities": audit_results,
                        "count": len(audit_results)
                    }
                except json.JSONDecodeError:
                    return {"status": "success", "vulnerabilities": [], "count": 0}
            else:
                return {
                    "status": "error",
                    "error": "pip-audit failed",
                    "output": result.stderr
                }
                
        except FileNotFoundError:
            return {
                "status": "error",
                "error": "pip-audit not installed. Install with: pip install pip-audit"
            }
    
    def check_secrets(self) -> Dict[str, Any]:
        """Check for potential secrets in code."""
        print("Checking for secrets...")
        
        # Simple regex-based secret detection
        import re
        
        secret_patterns = [
            (r'api[_-]?key["\s]*[:=]["\s]*[a-zA-Z0-9]{10,}', "API Key"),
            (r'secret[_-]?key["\s]*[:=]["\s]*[a-zA-Z0-9]{10,}', "Secret Key"),
            (r'password["\s]*[:=]["\s]*[a-zA-Z0-9]{8,}', "Password"),
            (r'token["\s]*[:=]["\s]*[a-zA-Z0-9]{10,}', "Token"),
            (r'["\']?AWS_ACCESS_KEY_ID["\']?\s*[:=]\s*["\'][A-Z0-9]{20}["\']', "AWS Access Key"),
            (r'["\']?AWS_SECRET_ACCESS_KEY["\']?\s*[:=]\s*["\'][A-Za-z0-9/+=]{40}["\']', "AWS Secret Key"),
        ]
        
        findings = []
        
        for py_file in self.project_root.rglob("*.py"):
            if ".git" in str(py_file) or "__pycache__" in str(py_file):
                continue
                
            try:
                content = py_file.read_text()
                for pattern, secret_type in secret_patterns:
                    matches = re.finditer(pattern, content, re.IGNORECASE)
                    for match in matches:
                        line_num = content[:match.start()].count('\n') + 1
                        findings.append({
                            "file": str(py_file.relative_to(self.project_root)),
                            "line": line_num,
                            "type": secret_type,
                            "pattern": pattern
                        })
            except UnicodeDecodeError:
                continue
        
        return {
            "status": "success",
            "findings": findings,
            "count": len(findings)
        }
    
    def generate_report(self, scan_results: Dict[str, Any]) -> Path:
        """Generate comprehensive security report."""
        timestamp = datetime.now().isoformat()
        
        report = {
            "scan_timestamp": timestamp,
            "project": "Dynamic Graph Diffusion Net",
            "scan_results": scan_results,
            "summary": self._generate_summary(scan_results)
        }
        
        # Save JSON report
        json_report_path = self.reports_dir / f"security-report-{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
        with open(json_report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Generate markdown report
        md_report_path = self.reports_dir / f"security-report-{datetime.now().strftime('%Y%m%d-%H%M%S')}.md"
        self._generate_markdown_report(report, md_report_path)
        
        return json_report_path
    
    def _generate_summary(self, scan_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate executive summary of security scan."""
        total_issues = 0
        critical_issues = 0
        high_issues = 0
        medium_issues = 0
        
        # Count issues from different tools
        for tool, results in scan_results.items():
            if results.get("status") == "success":
                if tool == "safety":
                    total_issues += results.get("count", 0)
                    critical_issues += results.get("count", 0)  # Safety issues are critical
                elif tool == "bandit":
                    bandit_results = results.get("results", [])
                    total_issues += len(bandit_results)
                    for issue in bandit_results:
                        severity = issue.get("issue_severity", "").upper()
                        if severity == "HIGH":
                            high_issues += 1
                        elif severity == "MEDIUM":
                            medium_issues += 1
                elif tool == "secrets":
                    secret_count = results.get("count", 0)
                    total_issues += secret_count
                    critical_issues += secret_count  # Secrets are critical
        
        return {
            "total_issues": total_issues,
            "critical_issues": critical_issues,
            "high_issues": high_issues,
            "medium_issues": medium_issues,
            "overall_risk": self._calculate_risk_level(critical_issues, high_issues, medium_issues)
        }
    
    def _calculate_risk_level(self, critical: int, high: int, medium: int) -> str:
        """Calculate overall risk level."""
        if critical > 0:
            return "CRITICAL"
        elif high > 3:
            return "HIGH"
        elif high > 0 or medium > 10:
            return "MEDIUM"
        elif medium > 0:
            return "LOW"
        else:
            return "MINIMAL"
    
    def _generate_markdown_report(self, report: Dict[str, Any], output_path: Path):
        """Generate markdown security report."""
        summary = report["summary"]
        
        md_content = f"""# Security Scan Report
        
**Project**: {report["project"]}  
**Scan Date**: {report["scan_timestamp"]}

## Executive Summary

- **Total Issues**: {summary["total_issues"]}
- **Critical Issues**: {summary["critical_issues"]}
- **High Priority Issues**: {summary["high_issues"]}
- **Medium Priority Issues**: {summary["medium_issues"]}
- **Overall Risk Level**: {summary["overall_risk"]}

## Detailed Results

### Dependency Vulnerabilities (Safety)
"""
        
        safety_results = report["scan_results"].get("safety", {})
        if safety_results.get("status") == "success":
            if safety_results.get("count", 0) == 0:
                md_content += "✅ No known vulnerabilities found in dependencies.\n\n"
            else:
                md_content += f"⚠️ Found {safety_results['count']} vulnerabilities:\n\n"
                for vuln in safety_results.get("vulnerabilities", []):
                    md_content += f"- **{vuln.get('package_name', 'Unknown')}**: {vuln.get('advisory', 'No details')}\n"
                md_content += "\n"
        
        # Add other tool results...
        
        md_content += """
## Recommendations

1. **Immediate Actions**:
   - Review and fix all critical and high priority issues
   - Update vulnerable dependencies
   - Remove any hardcoded secrets

2. **Preventive Measures**:
   - Enable automated security scanning in CI/CD
   - Implement pre-commit hooks for secret detection
   - Regular dependency updates
   - Security code reviews

3. **Monitoring**:
   - Set up continuous monitoring for new vulnerabilities
   - Subscribe to security advisories for used packages
   - Regular security assessments

---
*Report generated by Terragon Autonomous SDLC Security Scanner*
"""
        
        with open(output_path, 'w') as f:
            f.write(md_content)
    
    def run_full_scan(self) -> Path:
        """Run comprehensive security scan."""
        print("Starting comprehensive security scan...")
        
        scan_results = {
            "safety": self.run_safety_check(),
            "bandit": self.run_bandit_scan(),
            "pip_audit": self.run_pip_audit(),
            "secrets": self.check_secrets()
        }
        
        report_path = self.generate_report(scan_results)
        
        print(f"Security scan completed. Report saved to: {report_path}")
        return report_path

def main():
    """Main entry point."""
    project_root = Path(__file__).parent.parent
    scanner = SecurityScanner(project_root)
    
    try:
        report_path = scanner.run_full_scan()
        print(f"Security scan completed successfully: {report_path}")
        sys.exit(0)
    except Exception as e:
        print(f"Security scan failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()