#!/usr/bin/env python3
"""
Autonomous Value Discovery Engine for DGDN project.
Continuously discovers, scores, and prioritizes work items for maximum value delivery.
"""

import json
import re
import subprocess
import yaml
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
from enum import Enum

class Priority(Enum):
    LOW = "low"
    MEDIUM = "medium" 
    HIGH = "high"
    CRITICAL = "critical"

class Category(Enum):
    SECURITY = "security"
    PERFORMANCE = "performance"
    TECHNICAL_DEBT = "technical_debt"
    FEATURE = "feature"
    DOCUMENTATION = "documentation"
    TESTING = "testing"
    INFRASTRUCTURE = "infrastructure"
    MAINTENANCE = "maintenance"

@dataclass
class ValueItem:
    """Represents a discovered work item with value scoring."""
    id: str
    title: str
    description: str
    category: Category
    priority: Priority
    estimated_effort_hours: float
    wsjf_score: float
    ice_score: float
    technical_debt_score: float
    composite_score: float
    source: str
    file_path: str = ""
    line_number: int = 0
    discovered_at: str = ""
    tags: List[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if not self.discovered_at:
            self.discovered_at = datetime.now().isoformat()

class ValueDiscoveryEngine:
    """Main engine for discovering and scoring value items."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.config_file = project_root / ".terragon" / "config.yaml"
        self.metrics_file = project_root / ".terragon" / "value-metrics.json"
        
        # Load configuration
        with open(self.config_file) as f:
            self.config = yaml.safe_load(f)
        
        # Load existing metrics
        self.metrics = self._load_metrics()
        
        self.discovered_items: List[ValueItem] = []
    
    def _load_metrics(self) -> Dict[str, Any]:
        """Load existing value metrics."""
        if self.metrics_file.exists():
            with open(self.metrics_file) as f:
                return json.load(f)
        return {}
    
    def _save_metrics(self):
        """Save updated metrics."""
        self.metrics["last_updated"] = datetime.now().isoformat()
        with open(self.metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)
    
    def discover_from_git_history(self) -> List[ValueItem]:
        """Discover work items from git history and comments."""
        items = []
        
        # Search for TODO, FIXME, XXX, HACK patterns in code
        patterns = {
            r'TODO(?:\s*\([^)]+\))?\s*:?\s*(.+)': ("TODO", Category.MAINTENANCE),
            r'FIXME(?:\s*\([^)]+\))?\s*:?\s*(.+)': ("FIXME", Category.TECHNICAL_DEBT),
            r'XXX(?:\s*\([^)]+\))?\s*:?\s*(.+)': ("XXX", Category.TECHNICAL_DEBT),
            r'HACK(?:\s*\([^)]+\))?\s*:?\s*(.+)': ("HACK", Category.TECHNICAL_DEBT),
            r'DEPRECATED(?:\s*\([^)]+\))?\s*:?\s*(.+)': ("DEPRECATED", Category.MAINTENANCE),
            r'PERFORMANCE(?:\s*\([^)]+\))?\s*:?\s*(.+)': ("PERFORMANCE", Category.PERFORMANCE),
            r'SECURITY(?:\s*\([^)]+\))?\s*:?\s*(.+)': ("SECURITY", Category.SECURITY),
        }
        
        for py_file in self.project_root.rglob("*.py"):
            if ".git" in str(py_file) or "__pycache__" in str(py_file):
                continue
            
            try:
                content = py_file.read_text()
                lines = content.split('\n')
                
                for line_num, line in enumerate(lines, 1):
                    for pattern, (marker_type, category) in patterns.items():
                        match = re.search(pattern, line, re.IGNORECASE)
                        if match:
                            description = match.group(1).strip()
                            
                            # Create value item
                            item = ValueItem(
                                id=f"git-{marker_type.lower()}-{hash(f'{py_file}:{line_num}') % 10000}",
                                title=f"{marker_type}: {description[:50]}...",
                                description=description,
                                category=category,
                                priority=self._estimate_priority(description, marker_type),
                                estimated_effort_hours=self._estimate_effort(description, marker_type),
                                wsjf_score=0,  # Will be calculated later
                                ice_score=0,   # Will be calculated later  
                                technical_debt_score=0,  # Will be calculated later
                                composite_score=0,  # Will be calculated later
                                source="git_history",
                                file_path=str(py_file.relative_to(self.project_root)),
                                line_number=line_num,
                                tags=[marker_type.lower(), category.value]
                            )
                            
                            items.append(item)
                            
            except (UnicodeDecodeError, PermissionError):
                continue
        
        return items
    
    def discover_from_static_analysis(self) -> List[ValueItem]:
        """Discover items from static analysis tools."""
        items = []
        
        # Run ruff and parse output
        try:
            result = subprocess.run(
                ["ruff", "check", ".", "--format=json"],
                capture_output=True,
                text=True,
                cwd=self.project_root
            )
            
            if result.stdout:
                ruff_issues = json.loads(result.stdout)
                
                for issue in ruff_issues:
                    item = ValueItem(
                        id=f"ruff-{hash(f'{issue[\"filename\"]}:{issue[\"location\"][\"row\"]}:{issue[\"code\"]}') % 10000}",
                        title=f"Ruff {issue['code']}: {issue['message'][:50]}...",
                        description=issue['message'],
                        category=self._categorize_ruff_issue(issue['code']),
                        priority=self._prioritize_ruff_issue(issue['code']),
                        estimated_effort_hours=0.5,  # Most linting issues are quick fixes
                        wsjf_score=0,
                        ice_score=0,
                        technical_debt_score=0,
                        composite_score=0,
                        source="static_analysis_ruff",
                        file_path=issue['filename'],
                        line_number=issue['location']['row'],
                        tags=["ruff", issue['code'], "static_analysis"]
                    )
                    items.append(item)
                    
        except (FileNotFoundError, json.JSONDecodeError, subprocess.CalledProcessError):
            pass
        
        # Run mypy and parse output
        try:
            result = subprocess.run(
                ["mypy", "src/", "--json-report", "/tmp/mypy-report"],
                capture_output=True,
                text=True,
                cwd=self.project_root
            )
            
            # Parse mypy JSON report if available
            mypy_report_file = Path("/tmp/mypy-report/index.json")
            if mypy_report_file.exists():
                with open(mypy_report_file) as f:
                    mypy_data = json.load(f)
                    
                # Process mypy results
                for file_path, file_data in mypy_data.get("files", {}).items():
                    for error in file_data.get("errors", []):
                        item = ValueItem(
                            id=f"mypy-{hash(f'{file_path}:{error[\"line\"]}') % 10000}",
                            title=f"Type issue: {error['message'][:50]}...",
                            description=error['message'],
                            category=Category.TECHNICAL_DEBT,
                            priority=Priority.MEDIUM,
                            estimated_effort_hours=1.0,
                            wsjf_score=0,
                            ice_score=0,
                            technical_debt_score=0,
                            composite_score=0,
                            source="static_analysis_mypy",
                            file_path=file_path,
                            line_number=error['line'],
                            tags=["mypy", "types", "static_analysis"]
                        )
                        items.append(item)
                        
        except (FileNotFoundError, json.JSONDecodeError, subprocess.CalledProcessError):
            pass
        
        return items
    
    def discover_from_dependencies(self) -> List[ValueItem]:
        """Discover work items from dependency analysis."""
        items = []
        
        # Check for outdated dependencies
        try:
            result = subprocess.run(
                ["pip", "list", "--outdated", "--format=json"],
                capture_output=True,
                text=True,
                cwd=self.project_root
            )
            
            if result.stdout:
                outdated_packages = json.loads(result.stdout)
                
                for package in outdated_packages:
                    # Calculate priority based on version gap
                    current_version = package['version']
                    latest_version = package['latest_version']
                    
                    priority = Priority.LOW
                    if self._is_security_update(package):
                        priority = Priority.CRITICAL
                    elif self._is_major_version_update(current_version, latest_version):
                        priority = Priority.MEDIUM
                    
                    item = ValueItem(
                        id=f"dep-update-{package['name']}",
                        title=f"Update {package['name']} from {current_version} to {latest_version}",
                        description=f"Dependency update for {package['name']}",
                        category=Category.MAINTENANCE,
                        priority=priority,
                        estimated_effort_hours=self._estimate_update_effort(package),
                        wsjf_score=0,
                        ice_score=0,
                        technical_debt_score=0,
                        composite_score=0,
                        source="dependency_analysis",
                        tags=["dependency", "update", package['name']]
                    )
                    items.append(item)
                    
        except (FileNotFoundError, json.JSONDecodeError, subprocess.CalledProcessError):
            pass
        
        return items
    
    def discover_from_testing(self) -> List[ValueItem]:
        """Discover work items from test analysis."""
        items = []
        
        # Analyze test coverage
        try:
            result = subprocess.run(
                ["pytest", "--cov=src", "--cov-report=json", "--quiet"],
                capture_output=True,
                text=True,
                cwd=self.project_root
            )
            
            coverage_file = self.project_root / "coverage.json"
            if coverage_file.exists():
                with open(coverage_file) as f:
                    coverage_data = json.load(f)
                
                # Find files with low coverage
                for file_path, file_data in coverage_data.get("files", {}).items():
                    coverage_percent = file_data.get("summary", {}).get("percent_covered", 100)
                    
                    if coverage_percent < 80:  # Below 80% coverage threshold
                        item = ValueItem(
                            id=f"coverage-{hash(file_path) % 10000}",
                            title=f"Improve test coverage for {Path(file_path).name}",
                            description=f"File has {coverage_percent:.1f}% test coverage (target: 80%+)",
                            category=Category.TESTING,
                            priority=Priority.MEDIUM if coverage_percent < 50 else Priority.LOW,
                            estimated_effort_hours=2.0 + (80 - coverage_percent) * 0.1,
                            wsjf_score=0,
                            ice_score=0,
                            technical_debt_score=0,
                            composite_score=0,
                            source="test_analysis",
                            file_path=file_path,
                            tags=["testing", "coverage", "quality"]
                        )
                        items.append(item)
                        
        except (FileNotFoundError, json.JSONDecodeError, subprocess.CalledProcessError):
            pass
        
        return items
    
    def calculate_scores(self, item: ValueItem) -> ValueItem:
        """Calculate WSJF, ICE, and composite scores for an item."""
        
        # WSJF Calculation (Weighted Shortest Job First)
        user_business_value = self._score_business_value(item)
        time_criticality = self._score_time_criticality(item)  
        risk_reduction = self._score_risk_reduction(item)
        opportunity_enablement = self._score_opportunity_enablement(item)
        
        cost_of_delay = (
            user_business_value * self.config["scoring"]["wsjf_components"]["user_business_value"] +
            time_criticality * self.config["scoring"]["wsjf_components"]["time_criticality"] +
            risk_reduction * self.config["scoring"]["wsjf_components"]["risk_reduction"] +
            opportunity_enablement * self.config["scoring"]["wsjf_components"]["opportunity_enablement"]
        )
        
        job_size = max(item.estimated_effort_hours, 0.5)  # Avoid division by zero
        item.wsjf_score = cost_of_delay / job_size
        
        # ICE Calculation (Impact, Confidence, Ease)
        impact = self._score_impact(item)
        confidence = self._score_confidence(item)
        ease = 10 - min(item.estimated_effort_hours / 2, 10)  # Easier tasks get higher scores
        
        item.ice_score = impact * confidence * ease
        
        # Technical Debt Score
        if item.category in [Category.TECHNICAL_DEBT, Category.MAINTENANCE]:
            debt_impact = self._calculate_debt_impact(item)
            debt_interest = self._calculate_debt_interest(item)
            hotspot_multiplier = self._get_hotspot_multiplier(item)
            
            item.technical_debt_score = (debt_impact + debt_interest) * hotspot_multiplier
        else:
            item.technical_debt_score = 0
        
        # Composite Score
        weights = self.config["scoring"]["weights"]
        
        normalized_wsjf = min(item.wsjf_score / 50, 1.0)  # Normalize to 0-1 range
        normalized_ice = min(item.ice_score / 1000, 1.0)  # Normalize to 0-1 range  
        normalized_debt = min(item.technical_debt_score / 100, 1.0)  # Normalize to 0-1 range
        
        item.composite_score = (
            weights["wsjf"] * normalized_wsjf +
            weights["ice"] * normalized_ice +
            weights["technical_debt"] * normalized_debt
        )
        
        # Apply boosts and penalties
        if item.category == Category.SECURITY:
            item.composite_score *= self.config["scoring"]["thresholds"]["security_priority_boost"]
        
        return item
    
    def _estimate_priority(self, description: str, marker_type: str) -> Priority:
        """Estimate priority based on description and marker type."""
        description_lower = description.lower()
        
        if marker_type in ["SECURITY", "FIXME"] or any(word in description_lower for word in ["critical", "urgent", "security", "vulnerability"]):
            return Priority.HIGH
        elif marker_type in ["HACK", "XXX"] or any(word in description_lower for word in ["performance", "slow", "memory"]):
            return Priority.MEDIUM
        else:
            return Priority.LOW
    
    def _estimate_effort(self, description: str, marker_type: str) -> float:
        """Estimate effort in hours based on description and marker type."""
        if marker_type == "TODO":
            return 1.0
        elif marker_type in ["FIXME", "PERFORMANCE"]:
            return 2.0
        elif marker_type in ["HACK", "XXX", "SECURITY"]:
            return 4.0
        else:
            return 1.5
    
    def _categorize_ruff_issue(self, code: str) -> Category:
        """Categorize ruff issue by error code."""
        if code.startswith("S"):  # Security issues
            return Category.SECURITY
        elif code.startswith("E") or code.startswith("W"):  # Style issues
            return Category.MAINTENANCE
        elif code.startswith("F"):  # Logic issues
            return Category.TECHNICAL_DEBT
        else:
            return Category.MAINTENANCE
    
    def _prioritize_ruff_issue(self, code: str) -> Priority:
        """Prioritize ruff issue by error code."""
        if code.startswith("S"):  # Security issues
            return Priority.HIGH
        elif code.startswith("F"):  # Logic issues
            return Priority.MEDIUM
        else:
            return Priority.LOW
    
    def _is_security_update(self, package: Dict[str, str]) -> bool:
        """Check if package update contains security fixes."""
        # This would typically check against vulnerability databases
        # For now, use heuristics
        security_packages = ["urllib3", "requests", "django", "flask", "cryptography", "pycryptodome"]
        return package["name"].lower() in security_packages
    
    def _is_major_version_update(self, current: str, latest: str) -> bool:
        """Check if update is a major version change."""
        try:
            current_major = int(current.split('.')[0])
            latest_major = int(latest.split('.')[0])
            return latest_major > current_major
        except (ValueError, IndexError):
            return False
    
    def _estimate_update_effort(self, package: Dict[str, str]) -> float:
        """Estimate effort for dependency update."""
        if self._is_major_version_update(package["version"], package["latest_version"]):
            return 4.0  # Major updates need more testing
        else:
            return 0.5  # Minor updates are usually safe
    
    def _score_business_value(self, item: ValueItem) -> int:
        """Score business value (1-10 scale)."""
        if item.category == Category.SECURITY:
            return 10
        elif item.category == Category.PERFORMANCE:
            return 8
        elif item.category == Category.FEATURE:
            return 7
        elif item.category == Category.TECHNICAL_DEBT:
            return 6
        elif item.category == Category.TESTING:
            return 5
        else:
            return 4
    
    def _score_time_criticality(self, item: ValueItem) -> int:
        """Score time criticality (1-10 scale)."""
        if item.priority == Priority.CRITICAL:
            return 10
        elif item.priority == Priority.HIGH:
            return 8
        elif item.priority == Priority.MEDIUM:
            return 5
        else:
            return 3
    
    def _score_risk_reduction(self, item: ValueItem) -> int:
        """Score risk reduction value (1-10 scale)."""
        if item.category == Category.SECURITY:
            return 10
        elif item.category == Category.TECHNICAL_DEBT:
            return 7
        elif item.category == Category.TESTING:
            return 6
        else:
            return 4
    
    def _score_opportunity_enablement(self, item: ValueItem) -> int:
        """Score opportunity enablement (1-10 scale)."""
        if item.category == Category.INFRASTRUCTURE:
            return 9
        elif item.category == Category.FEATURE:
            return 8
        elif item.category == Category.PERFORMANCE:
            return 6
        else:
            return 4
    
    def _score_impact(self, item: ValueItem) -> int:
        """Score impact (1-10 scale)."""
        return self._score_business_value(item)
    
    def _score_confidence(self, item: ValueItem) -> int:
        """Score confidence in execution (1-10 scale)."""
        if item.source == "static_analysis_ruff":
            return 9  # High confidence in linting fixes
        elif item.category == Category.DOCUMENTATION:
            return 8  # Documentation is usually straightforward
        elif item.category == Category.TESTING:
            return 7  # Testing has some uncertainty
        else:
            return 6  # Default confidence
    
    def _calculate_debt_impact(self, item: ValueItem) -> float:
        """Calculate technical debt impact score."""
        # Base impact on category and estimated effort
        base_impact = item.estimated_effort_hours * 5
        
        # Boost for certain types of debt
        if "security" in item.tags:
            base_impact *= 2
        elif "performance" in item.tags:
            base_impact *= 1.5
        
        return base_impact
    
    def _calculate_debt_interest(self, item: ValueItem) -> float:
        """Calculate technical debt interest (future cost if not addressed)."""
        # Estimate compounding cost over time
        return item.estimated_effort_hours * 0.1 * 365  # Daily compound cost
    
    def _get_hotspot_multiplier(self, item: ValueItem) -> float:
        """Get hotspot multiplier based on file churn/complexity."""
        # For now, use simple heuristics
        if item.file_path:
            if "models" in item.file_path or "core" in item.file_path:
                return 2.0  # Core files are hotspots
            elif "test" in item.file_path:
                return 0.8  # Test files are less critical hotspots
        
        return 1.0  # Default multiplier
    
    def run_discovery(self) -> List[ValueItem]:
        """Run full value discovery process."""
        print("Starting autonomous value discovery...")
        
        all_items = []
        
        # Discover from various sources
        print("- Analyzing git history and code comments...")
        all_items.extend(self.discover_from_git_history())
        
        print("- Running static analysis...")
        all_items.extend(self.discover_from_static_analysis())
        
        print("- Analyzing dependencies...")
        all_items.extend(self.discover_from_dependencies())
        
        print("- Analyzing test coverage...")
        all_items.extend(self.discover_from_testing())
        
        # Calculate scores for all items
        print("- Calculating value scores...")
        for item in all_items:
            self.calculate_scores(item)
        
        # Sort by composite score (highest first)
        all_items.sort(key=lambda x: x.composite_score, reverse=True)
        
        # Filter items below minimum threshold
        min_score = self.config["scoring"]["thresholds"]["min_composite_score"]
        filtered_items = [item for item in all_items if item.composite_score >= min_score]
        
        self.discovered_items = filtered_items
        
        # Update metrics
        self._update_discovery_metrics(len(all_items), len(filtered_items))
        
        print(f"Discovery complete. Found {len(filtered_items)} actionable items (filtered from {len(all_items)} total).")
        
        return filtered_items
    
    def _update_discovery_metrics(self, total_discovered: int, actionable_items: int):
        """Update discovery metrics."""
        if "discovery_stats" not in self.metrics:
            self.metrics["discovery_stats"] = {"sources": {}}
        
        self.metrics["discovery_stats"]["items_discovered_today"] = actionable_items
        self.metrics["discovery_stats"]["discovery_rate_per_day"] = actionable_items
        
        # Update backlog metrics
        self.metrics["backlog_metrics"]["total_items"] = actionable_items
        high_priority = len([item for item in self.discovered_items if item.priority in [Priority.HIGH, Priority.CRITICAL]])
        medium_priority = len([item for item in self.discovered_items if item.priority == Priority.MEDIUM])
        low_priority = len([item for item in self.discovered_items if item.priority == Priority.LOW])
        
        self.metrics["backlog_metrics"]["high_priority_items"] = high_priority
        self.metrics["backlog_metrics"]["medium_priority_items"] = medium_priority
        self.metrics["backlog_metrics"]["low_priority_items"] = low_priority
        
        self._save_metrics()
    
    def get_next_best_value_item(self) -> ValueItem:
        """Get the next highest value item to work on."""
        if not self.discovered_items:
            self.run_discovery()
        
        if self.discovered_items:
            return self.discovered_items[0]
        else:
            # Return a housekeeping task if no items found
            return ValueItem(
                id="housekeeping-cleanup",
                title="Code cleanup and maintenance",
                description="General code cleanup and maintenance tasks",
                category=Category.MAINTENANCE,
                priority=Priority.LOW,
                estimated_effort_hours=1.0,
                wsjf_score=5.0,
                ice_score=100.0,
                technical_debt_score=10.0,
                composite_score=20.0,
                source="housekeeping"
            )

def main():
    """Main entry point."""
    project_root = Path(__file__).parent.parent
    engine = ValueDiscoveryEngine(project_root)
    
    items = engine.run_discovery()
    
    if items:
        print(f"\nTop 5 Value Items:")
        for i, item in enumerate(items[:5], 1):
            print(f"{i}. [{item.priority.value.upper()}] {item.title}")
            print(f"   Score: {item.composite_score:.1f} | Effort: {item.estimated_effort_hours}h | Category: {item.category.value}")
            print(f"   Source: {item.source}")
            if item.file_path:
                print(f"   File: {item.file_path}:{item.line_number}")
            print()
    else:
        print("No actionable value items discovered.")

if __name__ == "__main__":
    main()