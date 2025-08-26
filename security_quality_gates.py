#!/usr/bin/env python3
"""
DGDN Security & Quality Gates Validation
Terragon Labs Autonomous SDLC - Production Security & Quality Assurance
"""

import os
import sys
import json
import time
import hashlib
import logging
import traceback
import subprocess
import tempfile
import ast
import inspect
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Set
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SecurityScanner:
    """Comprehensive security vulnerability scanner."""
    
    def __init__(self):
        self.vulnerabilities = []
        self.security_patterns = {
            # Input validation vulnerabilities
            'unsafe_eval': [r'eval\(', r'exec\(', r'compile\('],
            'unsafe_input': [r'input\(\)', r'raw_input\(\)'],
            'unsafe_pickle': [r'pickle\.loads?', r'cPickle\.loads?'],
            'unsafe_subprocess': [r'subprocess\.call', r'os\.system', r'os\.popen'],
            'hardcoded_secrets': [r'password\s*=\s*["\'][^"\']+["\']', r'api_key\s*=\s*["\'][^"\']+["\']'],
            
            # Injection vulnerabilities
            'sql_injection': [r'execute\([^)]*%[^)]*\)', r'cursor\.execute.*format'],
            'command_injection': [r'shell=True', r'os\.system.*format'],
            
            # Crypto vulnerabilities
            'weak_crypto': [r'md5\(', r'sha1\(', r'random\.random\(\)'],
            'insecure_random': [r'random\.randint', r'random\.choice'],
            
            # Network security
            'unsafe_http': [r'http://', r'urllib\.request\.urlopen'],
            'ssl_verification': [r'verify=False', r'ssl\._create_unverified_context'],
            
            # File operations
            'unsafe_file_ops': [r'open\([^)]*["\']w["\'][^)]*\)', r'tempfile\.mktemp'],
            'path_traversal': [r'\.\./', r'os\.path\.join.*\.\./']
        }
    
    def scan_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """Scan single file for security vulnerabilities."""
        vulnerabilities = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.splitlines()
            
            # Pattern-based scanning
            import re
            for vuln_type, patterns in self.security_patterns.items():
                for pattern in patterns:
                    matches = re.finditer(pattern, content, re.IGNORECASE)
                    for match in matches:
                        line_num = content[:match.start()].count('\n') + 1
                        vulnerabilities.append({
                            'type': vuln_type,
                            'file': str(file_path),
                            'line': line_num,
                            'code': lines[line_num - 1].strip() if line_num <= len(lines) else '',
                            'severity': self._get_severity(vuln_type),
                            'description': self._get_description(vuln_type)
                        })
            
            # AST-based analysis for Python files
            if file_path.suffix == '.py':
                try:
                    tree = ast.parse(content)
                    ast_vulns = self._ast_security_scan(tree, file_path)
                    vulnerabilities.extend(ast_vulns)
                except SyntaxError:
                    # File has syntax errors, skip AST analysis
                    pass
                    
        except Exception as e:
            logger.warning(f"Failed to scan {file_path}: {e}")
        
        return vulnerabilities
    
    def _ast_security_scan(self, tree: ast.AST, file_path: Path) -> List[Dict[str, Any]]:
        """AST-based security analysis."""
        vulnerabilities = []
        
        class SecurityVisitor(ast.NodeVisitor):
            def __init__(self):
                self.vulns = []
            
            def visit_Call(self, node):
                # Check for dangerous function calls
                if isinstance(node.func, ast.Name):
                    if node.func.id in ['eval', 'exec', 'compile']:
                        self.vulns.append({
                            'type': 'dangerous_function_call',
                            'file': str(file_path),
                            'line': node.lineno,
                            'code': f"{node.func.id}(...)",
                            'severity': 'HIGH',
                            'description': f'Use of dangerous function: {node.func.id}'
                        })
                
                # Check for subprocess with shell=True
                if isinstance(node.func, ast.Attribute):
                    if (hasattr(node.func.value, 'id') and 
                        node.func.value.id == 'subprocess' and
                        node.func.attr in ['call', 'run', 'Popen']):
                        for keyword in node.keywords:
                            if keyword.arg == 'shell' and isinstance(keyword.value, ast.Constant):
                                if keyword.value.value:
                                    self.vulns.append({
                                        'type': 'shell_injection_risk',
                                        'file': str(file_path),
                                        'line': node.lineno,
                                        'code': f"subprocess.{node.func.attr}(..., shell=True)",
                                        'severity': 'HIGH',
                                        'description': 'Subprocess call with shell=True is dangerous'
                                    })
                
                self.generic_visit(node)
        
        visitor = SecurityVisitor()
        visitor.visit(tree)
        vulnerabilities.extend(visitor.vulns)
        
        return vulnerabilities
    
    def _get_severity(self, vuln_type: str) -> str:
        """Get vulnerability severity."""
        high_severity = ['unsafe_eval', 'sql_injection', 'command_injection', 'hardcoded_secrets']
        medium_severity = ['unsafe_pickle', 'weak_crypto', 'unsafe_file_ops']
        
        if vuln_type in high_severity:
            return 'HIGH'
        elif vuln_type in medium_severity:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def _get_description(self, vuln_type: str) -> str:
        """Get vulnerability description."""
        descriptions = {
            'unsafe_eval': 'Use of eval/exec can lead to code injection',
            'unsafe_input': 'Direct use of input() without validation',
            'unsafe_pickle': 'Pickle deserialization can execute arbitrary code',
            'unsafe_subprocess': 'Subprocess calls may allow command injection',
            'hardcoded_secrets': 'Hardcoded credentials found in source code',
            'sql_injection': 'Potential SQL injection vulnerability',
            'command_injection': 'Potential command injection vulnerability',
            'weak_crypto': 'Use of weak cryptographic functions',
            'insecure_random': 'Use of predictable random number generation',
            'unsafe_http': 'Use of insecure HTTP protocol',
            'ssl_verification': 'SSL certificate verification disabled',
            'unsafe_file_ops': 'Unsafe file operations detected',
            'path_traversal': 'Potential path traversal vulnerability'
        }
        return descriptions.get(vuln_type, 'Security vulnerability detected')
    
    def scan_directory(self, directory: Path) -> Dict[str, Any]:
        """Scan entire directory for vulnerabilities."""
        logger.info(f"🔒 Starting security scan of {directory}")
        
        all_vulnerabilities = []
        files_scanned = 0
        
        # Scan Python files
        for py_file in directory.rglob('*.py'):
            if 'test' not in str(py_file).lower():  # Skip test files
                vulns = self.scan_file(py_file)
                all_vulnerabilities.extend(vulns)
                files_scanned += 1
        
        # Analyze results
        severity_counts = {'HIGH': 0, 'MEDIUM': 0, 'LOW': 0}
        vulnerability_types = {}
        
        for vuln in all_vulnerabilities:
            severity = vuln['severity']
            vuln_type = vuln['type']
            
            severity_counts[severity] += 1
            vulnerability_types[vuln_type] = vulnerability_types.get(vuln_type, 0) + 1
        
        security_score = max(0, 100 - (severity_counts['HIGH'] * 10 + 
                                     severity_counts['MEDIUM'] * 5 + 
                                     severity_counts['LOW'] * 1))
        
        return {
            'files_scanned': files_scanned,
            'total_vulnerabilities': len(all_vulnerabilities),
            'severity_breakdown': severity_counts,
            'vulnerability_types': vulnerability_types,
            'security_score': security_score,
            'vulnerabilities': all_vulnerabilities,
            'scan_timestamp': time.time()
        }

class CodeQualityAnalyzer:
    """Code quality analysis and metrics."""
    
    def __init__(self):
        self.quality_metrics = {}
    
    def analyze_complexity(self, file_path: Path) -> Dict[str, Any]:
        """Analyze cyclomatic complexity."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            class ComplexityVisitor(ast.NodeVisitor):
                def __init__(self):
                    self.complexity = 1  # Base complexity
                    self.functions = {}
                    self.current_function = None
                
                def visit_FunctionDef(self, node):
                    old_function = self.current_function
                    self.current_function = node.name
                    self.functions[node.name] = {'complexity': 1, 'line': node.lineno}
                    
                    self.generic_visit(node)
                    self.current_function = old_function
                
                def visit_If(self, node):
                    self._increment_complexity()
                    self.generic_visit(node)
                
                def visit_For(self, node):
                    self._increment_complexity()
                    self.generic_visit(node)
                
                def visit_While(self, node):
                    self._increment_complexity()
                    self.generic_visit(node)
                
                def visit_Try(self, node):
                    self._increment_complexity()
                    self.generic_visit(node)
                
                def visit_ExceptHandler(self, node):
                    self._increment_complexity()
                    self.generic_visit(node)
                
                def _increment_complexity(self):
                    if self.current_function:
                        self.functions[self.current_function]['complexity'] += 1
                    else:
                        self.complexity += 1
            
            visitor = ComplexityVisitor()
            visitor.visit(tree)
            
            return {
                'overall_complexity': visitor.complexity,
                'function_complexities': visitor.functions,
                'max_function_complexity': max([f['complexity'] for f in visitor.functions.values()] + [0])
            }
            
        except Exception as e:
            logger.warning(f"Failed to analyze complexity for {file_path}: {e}")
            return {'overall_complexity': 0, 'function_complexities': {}, 'max_function_complexity': 0}
    
    def analyze_maintainability(self, file_path: Path) -> Dict[str, Any]:
        """Analyze code maintainability metrics."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Basic metrics
            total_lines = len(lines)
            code_lines = len([l for l in lines if l.strip() and not l.strip().startswith('#')])
            comment_lines = len([l for l in lines if l.strip().startswith('#')])
            blank_lines = total_lines - code_lines - comment_lines
            
            # Documentation ratio
            doc_ratio = comment_lines / max(code_lines, 1) * 100
            
            # Average line length
            avg_line_length = np.mean([len(l.rstrip()) for l in lines if l.strip()]) if code_lines > 0 else 0
            
            # Long lines count (>88 characters)
            long_lines = len([l for l in lines if len(l.rstrip()) > 88])
            
            maintainability_index = max(0, 100 - long_lines * 2)
            if doc_ratio > 10:
                maintainability_index += 10
            if avg_line_length < 80:
                maintainability_index += 5
            
            maintainability_index = min(100, maintainability_index)
            
            return {
                'total_lines': total_lines,
                'code_lines': code_lines,
                'comment_lines': comment_lines,
                'blank_lines': blank_lines,
                'documentation_ratio': doc_ratio,
                'average_line_length': float(avg_line_length),
                'long_lines': long_lines,
                'maintainability_index': maintainability_index
            }
            
        except Exception as e:
            logger.warning(f"Failed to analyze maintainability for {file_path}: {e}")
            return {'maintainability_index': 0}
    
    def analyze_directory(self, directory: Path) -> Dict[str, Any]:
        """Analyze code quality for entire directory."""
        logger.info(f"📊 Starting code quality analysis of {directory}")
        
        all_complexities = []
        all_maintainability = []
        files_analyzed = 0
        
        for py_file in directory.rglob('*.py'):
            if 'test' not in str(py_file).lower():
                complexity = self.analyze_complexity(py_file)
                maintainability = self.analyze_maintainability(py_file)
                
                all_complexities.append(complexity['max_function_complexity'])
                all_maintainability.append(maintainability['maintainability_index'])
                files_analyzed += 1
        
        # Aggregate metrics
        avg_complexity = np.mean(all_complexities) if all_complexities else 0
        max_complexity = max(all_complexities) if all_complexities else 0
        avg_maintainability = np.mean(all_maintainability) if all_maintainability else 0
        
        # Quality scores
        complexity_score = max(0, 100 - max_complexity * 5)  # Penalize high complexity
        maintainability_score = avg_maintainability
        
        overall_quality_score = (complexity_score + maintainability_score) / 2
        
        return {
            'files_analyzed': files_analyzed,
            'average_complexity': float(avg_complexity),
            'max_complexity': float(max_complexity),
            'average_maintainability': float(avg_maintainability),
            'complexity_score': float(complexity_score),
            'maintainability_score': float(maintainability_score),
            'overall_quality_score': float(overall_quality_score),
            'analysis_timestamp': time.time()
        }

class DependencyScanner:
    """Scan dependencies for known vulnerabilities."""
    
    def __init__(self):
        self.known_vulnerabilities = {
            # Known vulnerable package versions
            'numpy': {'<1.20.0': 'CVE-2021-33430'},
            'requests': {'<2.20.0': 'CVE-2018-18074'},
            'urllib3': {'<1.24.2': 'CVE-2019-11324'},
            'jinja2': {'<2.10.1': 'CVE-2019-10906'},
            'pillow': {'<6.2.0': 'CVE-2019-16865'}
        }
    
    def scan_requirements(self, requirements_file: Path) -> Dict[str, Any]:
        """Scan requirements.txt for vulnerable dependencies."""
        logger.info(f"🔍 Scanning dependencies in {requirements_file}")
        
        vulnerabilities = []
        packages_scanned = 0
        
        try:
            with open(requirements_file, 'r') as f:
                lines = f.readlines()
            
            for line in lines:
                line = line.strip()
                if line and not line.startswith('#'):
                    # Parse package name and version
                    if '>=' in line:
                        package, version = line.split('>=')
                        operator = '>='
                    elif '==' in line:
                        package, version = line.split('==')
                        operator = '=='
                    elif '>' in line:
                        package, version = line.split('>')
                        operator = '>'
                    else:
                        package = line
                        version = None
                        operator = None
                    
                    package = package.strip()
                    packages_scanned += 1
                    
                    # Check for known vulnerabilities
                    if package in self.known_vulnerabilities:
                        for vuln_version, cve in self.known_vulnerabilities[package].items():
                            if version and self._is_vulnerable_version(version, vuln_version):
                                vulnerabilities.append({
                                    'package': package,
                                    'version': version,
                                    'vulnerability': cve,
                                    'vulnerable_version': vuln_version,
                                    'severity': 'MEDIUM'
                                })
        
        except Exception as e:
            logger.warning(f"Failed to scan requirements: {e}")
        
        return {
            'packages_scanned': packages_scanned,
            'vulnerable_packages': len(vulnerabilities),
            'vulnerabilities': vulnerabilities,
            'dependency_score': max(0, 100 - len(vulnerabilities) * 10)
        }
    
    def _is_vulnerable_version(self, current_version: str, vulnerable_pattern: str) -> bool:
        """Check if current version matches vulnerable pattern."""
        # Simplified version comparison - in production, use proper semver
        if vulnerable_pattern.startswith('<'):
            # Extract version number
            try:
                vuln_ver = vulnerable_pattern[1:]
                return current_version < vuln_ver
            except:
                return False
        return False

class LicenseChecker:
    """Check license compatibility and compliance."""
    
    def __init__(self):
        self.approved_licenses = {
            'MIT', 'BSD', 'Apache', 'Apache-2.0', 'Apache 2.0', 
            'ISC', 'BSD-3-Clause', 'BSD-2-Clause', 'MPL-2.0'
        }
        self.restrictive_licenses = {
            'GPL', 'GPL-2.0', 'GPL-3.0', 'AGPL', 'AGPL-3.0', 'LGPL'
        }
    
    def check_project_license(self, directory: Path) -> Dict[str, Any]:
        """Check project license compliance."""
        logger.info(f"⚖️ Checking license compliance for {directory}")
        
        license_files = []
        for pattern in ['LICENSE', 'LICENSE.txt', 'LICENSE.md', 'COPYING']:
            for license_file in directory.glob(pattern):
                license_files.append(license_file)
        
        if not license_files:
            return {
                'has_license': False,
                'license_type': 'UNKNOWN',
                'compliance_score': 0,
                'recommendation': 'Add a LICENSE file to the project'
            }
        
        # Read and analyze license content
        license_content = ''
        for license_file in license_files:
            try:
                with open(license_file, 'r', encoding='utf-8') as f:
                    license_content += f.read().upper()
            except:
                continue
        
        # Detect license type
        license_type = 'UNKNOWN'
        compliance_score = 50  # Base score for having a license
        
        if 'MIT' in license_content:
            license_type = 'MIT'
            compliance_score = 100
        elif 'APACHE' in license_content:
            license_type = 'Apache'
            compliance_score = 95
        elif 'BSD' in license_content:
            license_type = 'BSD'
            compliance_score = 95
        elif 'GPL' in license_content:
            license_type = 'GPL'
            compliance_score = 60  # May have compatibility issues
        
        return {
            'has_license': True,
            'license_type': license_type,
            'license_files': [str(f) for f in license_files],
            'compliance_score': compliance_score,
            'is_approved': license_type in self.approved_licenses or any(approved in license_type for approved in self.approved_licenses)
        }

class QualityGateValidator:
    """Validate all quality gates for production readiness."""
    
    def __init__(self):
        self.security_scanner = SecurityScanner()
        self.quality_analyzer = CodeQualityAnalyzer()
        self.dependency_scanner = DependencyScanner()
        self.license_checker = LicenseChecker()
        
        # Quality gate thresholds
        self.thresholds = {
            'security_score_min': 80,
            'quality_score_min': 70,
            'dependency_score_min': 90,
            'license_compliance_min': 80,
            'max_high_vulnerabilities': 0,
            'max_medium_vulnerabilities': 3,
            'max_complexity': 15,
            'min_documentation_ratio': 5
        }
    
    def run_all_gates(self, project_dir: Path) -> Dict[str, Any]:
        """Run all quality gates and generate comprehensive report."""
        logger.info("🛡️ RUNNING COMPREHENSIVE SECURITY & QUALITY GATES")
        logger.info("=" * 60)
        
        start_time = time.time()
        
        # Run security scan
        logger.info("Running security vulnerability scan...")
        security_results = self.security_scanner.scan_directory(project_dir)
        
        # Run code quality analysis
        logger.info("Running code quality analysis...")
        quality_results = self.quality_analyzer.analyze_directory(project_dir)
        
        # Run dependency scan
        logger.info("Running dependency vulnerability scan...")
        requirements_file = project_dir / 'requirements.txt'
        if requirements_file.exists():
            dependency_results = self.dependency_scanner.scan_requirements(requirements_file)
        else:
            dependency_results = {'packages_scanned': 0, 'vulnerable_packages': 0, 'dependency_score': 100}
        
        # Run license check
        logger.info("Running license compliance check...")
        license_results = self.license_checker.check_project_license(project_dir)
        
        # Validate quality gates
        gate_results = self._validate_gates(security_results, quality_results, dependency_results, license_results)
        
        total_time = time.time() - start_time
        
        # Comprehensive results
        results = {
            'scan_metadata': {
                'timestamp': time.time(),
                'duration_seconds': total_time,
                'project_directory': str(project_dir),
                'scanner_version': '1.0.0'
            },
            
            'security_scan': security_results,
            'code_quality': quality_results,
            'dependency_scan': dependency_results,
            'license_check': license_results,
            'quality_gates': gate_results,
            
            'overall_assessment': {
                'security_passed': gate_results['security_gate']['passed'],
                'quality_passed': gate_results['quality_gate']['passed'],
                'dependency_passed': gate_results['dependency_gate']['passed'],
                'license_passed': gate_results['license_gate']['passed'],
                'all_gates_passed': gate_results['overall_passed'],
                'production_ready': gate_results['overall_passed'] and gate_results['overall_score'] >= 80,
                'overall_score': gate_results['overall_score'],
                'risk_level': self._get_risk_level(gate_results['overall_score'])
            },
            
            'recommendations': self._generate_recommendations(security_results, quality_results, dependency_results, license_results)
        }
        
        return results
    
    def _validate_gates(self, security_results, quality_results, dependency_results, license_results) -> Dict[str, Any]:
        """Validate all quality gates against thresholds."""
        
        # Security gate validation
        security_gate = {
            'passed': (
                security_results['security_score'] >= self.thresholds['security_score_min'] and
                security_results['severity_breakdown']['HIGH'] <= self.thresholds['max_high_vulnerabilities'] and
                security_results['severity_breakdown']['MEDIUM'] <= self.thresholds['max_medium_vulnerabilities']
            ),
            'score': security_results['security_score'],
            'details': {
                'high_vulnerabilities': security_results['severity_breakdown']['HIGH'],
                'medium_vulnerabilities': security_results['severity_breakdown']['MEDIUM'],
                'total_vulnerabilities': security_results['total_vulnerabilities']
            }
        }
        
        # Code quality gate validation
        quality_gate = {
            'passed': (
                quality_results['overall_quality_score'] >= self.thresholds['quality_score_min'] and
                quality_results['max_complexity'] <= self.thresholds['max_complexity']
            ),
            'score': quality_results['overall_quality_score'],
            'details': {
                'max_complexity': quality_results['max_complexity'],
                'maintainability_score': quality_results['maintainability_score']
            }
        }
        
        # Dependency gate validation
        dependency_gate = {
            'passed': dependency_results['dependency_score'] >= self.thresholds['dependency_score_min'],
            'score': dependency_results['dependency_score'],
            'details': {
                'vulnerable_packages': dependency_results['vulnerable_packages'],
                'packages_scanned': dependency_results['packages_scanned']
            }
        }
        
        # License gate validation
        license_gate = {
            'passed': (
                license_results.get('has_license', False) and
                license_results.get('compliance_score', 0) >= self.thresholds['license_compliance_min']
            ),
            'score': license_results.get('compliance_score', 0),
            'details': {
                'has_license': license_results.get('has_license', False),
                'license_type': license_results.get('license_type', 'UNKNOWN'),
                'is_approved': license_results.get('is_approved', False)
            }
        }
        
        # Overall assessment
        gates_passed = sum([
            security_gate['passed'],
            quality_gate['passed'], 
            dependency_gate['passed'],
            license_gate['passed']
        ])
        
        overall_score = (
            security_gate['score'] * 0.4 +
            quality_gate['score'] * 0.3 +
            dependency_gate['score'] * 0.2 +
            license_gate['score'] * 0.1
        )
        
        return {
            'security_gate': security_gate,
            'quality_gate': quality_gate,
            'dependency_gate': dependency_gate,
            'license_gate': license_gate,
            'gates_passed_count': gates_passed,
            'total_gates': 4,
            'overall_passed': gates_passed >= 4,  # All gates must pass
            'overall_score': overall_score
        }
    
    def _get_risk_level(self, score: float) -> str:
        """Determine risk level based on overall score."""
        if score >= 90:
            return 'LOW'
        elif score >= 70:
            return 'MEDIUM'
        elif score >= 50:
            return 'HIGH'
        else:
            return 'CRITICAL'
    
    def _generate_recommendations(self, security_results, quality_results, dependency_results, license_results) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []
        
        # Security recommendations
        if security_results['severity_breakdown']['HIGH'] > 0:
            recommendations.append("⚠️ CRITICAL: Fix all HIGH severity security vulnerabilities immediately")
        if security_results['severity_breakdown']['MEDIUM'] > 3:
            recommendations.append("⚠️ Address MEDIUM severity security vulnerabilities")
        if security_results['security_score'] < 80:
            recommendations.append("🔒 Improve overall security posture - consider security code review")
        
        # Code quality recommendations
        if quality_results['max_complexity'] > 15:
            recommendations.append("📊 Refactor high-complexity functions to improve maintainability")
        if quality_results['maintainability_score'] < 70:
            recommendations.append("🔧 Improve code documentation and structure")
        
        # Dependency recommendations
        if dependency_results['vulnerable_packages'] > 0:
            recommendations.append("📦 Update vulnerable dependencies to secure versions")
        
        # License recommendations
        if not license_results.get('has_license', False):
            recommendations.append("⚖️ Add a LICENSE file to clarify project licensing")
        elif not license_results.get('is_approved', False):
            recommendations.append("⚖️ Consider using a more permissive license for broader adoption")
        
        if not recommendations:
            recommendations.append("✅ All quality gates passed - project is production ready!")
        
        return recommendations

def run_security_quality_gates():
    """Execute comprehensive security and quality gates validation."""
    logger.info("🛡️ TERRAGON AUTONOMOUS SDLC - SECURITY & QUALITY GATES")
    logger.info("Running production-ready security and quality validation...")
    
    try:
        project_dir = Path("/root/repo")
        
        # Initialize validator
        validator = QualityGateValidator()
        
        # Run all quality gates
        results = validator.run_all_gates(project_dir)
        
        # Log summary results
        logger.info("=" * 60)
        logger.info("🛡️ SECURITY & QUALITY GATES RESULTS SUMMARY")
        logger.info("=" * 60)
        
        assessment = results['overall_assessment']
        logger.info(f"Overall Score: {assessment['overall_score']:.1f}/100")
        logger.info(f"Risk Level: {assessment['risk_level']}")
        logger.info(f"Production Ready: {'✅ YES' if assessment['production_ready'] else '❌ NO'}")
        
        logger.info("\n📊 Individual Gate Results:")
        gates = results['quality_gates']
        logger.info(f"  Security Gate: {'✅ PASS' if gates['security_gate']['passed'] else '❌ FAIL'} ({gates['security_gate']['score']:.1f}/100)")
        logger.info(f"  Code Quality Gate: {'✅ PASS' if gates['quality_gate']['passed'] else '❌ FAIL'} ({gates['quality_gate']['score']:.1f}/100)")
        logger.info(f"  Dependency Gate: {'✅ PASS' if gates['dependency_gate']['passed'] else '❌ FAIL'} ({gates['dependency_gate']['score']:.1f}/100)")
        logger.info(f"  License Gate: {'✅ PASS' if gates['license_gate']['passed'] else '❌ FAIL'} ({gates['license_gate']['score']:.1f}/100)")
        
        logger.info(f"\n📈 Detailed Metrics:")
        security = results['security_scan']
        quality = results['code_quality']
        logger.info(f"  Files Scanned: {security['files_scanned']} (security), {quality['files_analyzed']} (quality)")
        logger.info(f"  Security Vulnerabilities: {security['total_vulnerabilities']} (H:{security['severity_breakdown']['HIGH']}, M:{security['severity_breakdown']['MEDIUM']}, L:{security['severity_breakdown']['LOW']})")
        logger.info(f"  Code Complexity: Max={quality['max_complexity']:.1f}, Avg={quality['average_complexity']:.1f}")
        logger.info(f"  Dependencies: {results['dependency_scan']['packages_scanned']} scanned, {results['dependency_scan']['vulnerable_packages']} vulnerable")
        
        logger.info(f"\n💡 Recommendations:")
        for rec in results['recommendations']:
            logger.info(f"  {rec}")
        
        # Save detailed results
        results_file = Path("security_quality_gates_results.json")
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"\n💾 Detailed results saved to: {results_file}")
        
        # Determine overall success
        if assessment['all_gates_passed'] and assessment['production_ready']:
            logger.info("\n🎉 ALL SECURITY & QUALITY GATES PASSED!")
            logger.info("✅ Project meets all production readiness criteria")
            logger.info("✅ Ready for deployment preparation")
            return True, results
        else:
            logger.warning("\n⚠️ SECURITY & QUALITY GATES NOT FULLY PASSED")
            logger.warning("Some quality gates failed - review recommendations before production deployment")
            return False, results
            
    except Exception as e:
        logger.error(f"❌ SECURITY & QUALITY GATES FAILED: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False, {'status': 'failed', 'error': str(e)}

if __name__ == "__main__":
    # Add numpy import for calculations
    import numpy as np
    
    success, results = run_security_quality_gates()
    
    if success:
        print("\n" + "="*50)
        print("🎉 SECURITY & QUALITY GATES SUCCESS!")
        print("="*50)
        assessment = results['overall_assessment']
        print(f"✅ Overall Score: {assessment['overall_score']:.1f}/100")
        print(f"✅ Risk Level: {assessment['risk_level']}")
        print(f"✅ Production Ready: YES")
        print("✅ All security and quality criteria met!")
        print("✅ Ready for production deployment preparation!")
    else:
        print("\n" + "="*50)
        print("❌ SECURITY & QUALITY GATES INCOMPLETE")
        print("="*50)
        if isinstance(results, dict) and 'overall_assessment' in results:
            assessment = results['overall_assessment']
            print(f"Overall Score: {assessment['overall_score']:.1f}/100")
            print(f"Risk Level: {assessment['risk_level']}")
        print("Additional security/quality work needed before production")
    
    sys.exit(0 if success else 1)