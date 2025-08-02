#!/usr/bin/env python3
"""
Automated dependency update script for DGDN project.

This script checks for outdated dependencies, creates update branches,
and optionally creates pull requests for dependency updates.
"""

import subprocess
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import argparse


class DependencyUpdater:
    """Manages automated dependency updates."""
    
    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run
        self.project_root = Path.cwd()
        self.pyproject_path = self.project_root / "pyproject.toml"
        self.requirements_path = self.project_root / "requirements.txt"
        self.outdated_packages = []
        self.update_summary = {
            "timestamp": datetime.now().isoformat(),
            "updates": [],
            "security_updates": [],
            "breaking_changes": [],
            "failed_updates": []
        }
    
    def check_outdated_packages(self) -> List[Dict]:
        """Check for outdated packages using pip."""
        print("🔍 Checking for outdated packages...")
        
        try:
            result = subprocess.run(
                ["pip", "list", "--outdated", "--format=json"],
                capture_output=True,
                text=True,
                check=True
            )
            outdated = json.loads(result.stdout)
            self.outdated_packages = outdated
            
            if outdated:
                print(f"Found {len(outdated)} outdated packages:")
                for pkg in outdated:
                    print(f"  {pkg['name']}: {pkg['version']} → {pkg['latest_version']}")
            else:
                print("✅ All packages are up to date!")
            
            return outdated
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Error checking outdated packages: {e}")
            return []
        except json.JSONDecodeError as e:
            print(f"❌ Error parsing pip output: {e}")
            return []
    
    def check_security_vulnerabilities(self) -> List[Dict]:
        """Check for security vulnerabilities using safety."""
        print("🔒 Checking for security vulnerabilities...")
        
        try:
            result = subprocess.run(
                ["safety", "check", "--json"],
                capture_output=True,
                text=True
            )
            
            if result.stdout:
                vulnerabilities = json.loads(result.stdout)
                if vulnerabilities:
                    print(f"⚠️ Found {len(vulnerabilities)} security vulnerabilities!")
                    for vuln in vulnerabilities:
                        print(f"  {vuln.get('package_name', 'Unknown')}: {vuln.get('vulnerability_id', 'N/A')}")
                    return vulnerabilities
                else:
                    print("✅ No security vulnerabilities found!")
            
            return []
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Error checking security vulnerabilities: {e}")
            return []
        except json.JSONDecodeError as e:
            print(f"❌ Error parsing safety output: {e}")
            return []
    
    def categorize_updates(self) -> Dict[str, List[Dict]]:
        """Categorize updates by type (major, minor, patch, security)."""
        categories = {
            "security": [],
            "major": [],
            "minor": [],
            "patch": []
        }
        
        security_packages = {vuln.get("package_name", "").lower() 
                           for vuln in self.check_security_vulnerabilities()}
        
        for pkg in self.outdated_packages:
            current_version = self._parse_version(pkg["version"])
            latest_version = self._parse_version(pkg["latest_version"])
            
            # Check if this is a security update
            if pkg["name"].lower() in security_packages:
                categories["security"].append(pkg)
            # Determine update type based on semantic versioning
            elif latest_version[0] > current_version[0]:
                categories["major"].append(pkg)
            elif latest_version[1] > current_version[1]:
                categories["minor"].append(pkg)
            else:
                categories["patch"].append(pkg)
        
        return categories
    
    def _parse_version(self, version_str: str) -> Tuple[int, int, int]:
        """Parse version string into tuple of integers."""
        # Clean version string (remove pre-release suffixes)
        clean_version = re.match(r'^(\d+)\.(\d+)\.(\d+)', version_str)
        if clean_version:
            return tuple(map(int, clean_version.groups()))
        else:
            # Fallback for non-standard versions
            parts = version_str.split('.')
            try:
                return (int(parts[0]), int(parts[1]) if len(parts) > 1 else 0, 
                       int(parts[2]) if len(parts) > 2 else 0)
            except (ValueError, IndexError):
                return (0, 0, 0)
    
    def create_update_branch(self, branch_name: str) -> bool:
        """Create a new branch for updates."""
        if self.dry_run:
            print(f"[DRY RUN] Would create branch: {branch_name}")
            return True
        
        try:
            # Ensure we're on main branch
            subprocess.run(["git", "checkout", "main"], check=True, capture_output=True)
            subprocess.run(["git", "pull", "origin", "main"], check=True, capture_output=True)
            
            # Create and checkout new branch
            subprocess.run(["git", "checkout", "-b", branch_name], check=True, capture_output=True)
            print(f"✅ Created and switched to branch: {branch_name}")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Error creating branch {branch_name}: {e}")
            return False
    
    def update_package(self, package_name: str, target_version: Optional[str] = None) -> bool:
        """Update a specific package."""
        if self.dry_run:
            version_spec = f"=={target_version}" if target_version else ""
            print(f"[DRY RUN] Would update: {package_name}{version_spec}")
            return True
        
        try:
            cmd = ["pip", "install", "--upgrade"]
            if target_version:
                cmd.append(f"{package_name}=={target_version}")
            else:
                cmd.append(package_name)
            
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(f"✅ Updated {package_name}")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Error updating {package_name}: {e}")
            print(f"   stderr: {e.stderr}")
            return False
    
    def update_requirements_file(self):
        """Update requirements.txt or pyproject.toml with new versions."""
        if self.dry_run:
            print("[DRY RUN] Would update requirements file")
            return
        
        try:
            # Generate new requirements.txt
            result = subprocess.run(
                ["pip", "freeze"],
                capture_output=True,
                text=True,
                check=True
            )
            
            requirements_content = result.stdout
            
            # Filter out editable packages and unnecessary entries
            filtered_lines = []
            for line in requirements_content.split('\\n'):
                if line and not line.startswith('-e ') and not line.startswith('# '):
                    filtered_lines.append(line)
            
            if self.requirements_path.exists():
                with open(self.requirements_path, 'w') as f:
                    f.write('\\n'.join(filtered_lines))
                print(f"✅ Updated {self.requirements_path}")
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Error updating requirements file: {e}")
    
    def run_tests(self) -> bool:
        """Run tests to verify updates don't break anything."""
        if self.dry_run:
            print("[DRY RUN] Would run tests")
            return True
        
        print("🧪 Running tests to verify updates...")
        try:
            result = subprocess.run(
                ["python", "-m", "pytest", "tests/", "-v", "--tb=short"],
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout
            )
            
            if result.returncode == 0:
                print("✅ All tests passed!")
                return True
            else:
                print("❌ Tests failed!")
                print(f"stdout: {result.stdout}")
                print(f"stderr: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            print("❌ Tests timed out!")
            return False
        except Exception as e:
            print(f"❌ Error running tests: {e}")
            return False
    
    def commit_updates(self, message: str) -> bool:
        """Commit the updates."""
        if self.dry_run:
            print(f"[DRY RUN] Would commit with message: {message}")
            return True
        
        try:
            # Add all changes
            subprocess.run(["git", "add", "."], check=True, capture_output=True)
            
            # Check if there are changes to commit
            result = subprocess.run(["git", "diff", "--cached", "--name-only"], 
                                  capture_output=True, text=True)
            if not result.stdout.strip():
                print("ℹ️ No changes to commit")
                return True
            
            # Commit changes
            subprocess.run(["git", "commit", "-m", message], check=True, capture_output=True)
            print(f"✅ Committed changes: {message}")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Error committing changes: {e}")
            return False
    
    def push_branch(self, branch_name: str) -> bool:
        """Push the update branch to remote."""
        if self.dry_run:
            print(f"[DRY RUN] Would push branch: {branch_name}")
            return True
        
        try:
            subprocess.run(["git", "push", "-u", "origin", branch_name], 
                          check=True, capture_output=True)
            print(f"✅ Pushed branch: {branch_name}")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Error pushing branch {branch_name}: {e}")
            return False
    
    def process_security_updates(self) -> bool:
        """Process security updates with high priority."""
        print("🔒 Processing security updates...")
        
        security_vulns = self.check_security_vulnerabilities()
        if not security_vulns:
            print("✅ No security updates needed")
            return True
        
        branch_name = f"deps/security-updates-{datetime.now().strftime('%Y%m%d')}"
        
        if not self.create_update_branch(branch_name):
            return False
        
        success = True
        updated_packages = []
        
        for vuln in security_vulns:
            pkg_name = vuln.get("package_name", "")
            if pkg_name and self.update_package(pkg_name):
                updated_packages.append(pkg_name)
                self.update_summary["security_updates"].append({
                    "package": pkg_name,
                    "vulnerability_id": vuln.get("vulnerability_id", ""),
                    "status": "updated"
                })
            else:
                success = False
                self.update_summary["failed_updates"].append({
                    "package": pkg_name,
                    "reason": "security_update_failed",
                    "vulnerability_id": vuln.get("vulnerability_id", "")
                })
        
        if updated_packages:
            self.update_requirements_file()
            
            if self.run_tests():
                commit_msg = f"security: update packages for security vulnerabilities\\n\\nUpdated packages:\\n" + \
                           "\\n".join(f"- {pkg}" for pkg in updated_packages)
                
                if self.commit_updates(commit_msg):
                    self.push_branch(branch_name)
                    print(f"🔒 Security updates completed on branch: {branch_name}")
                else:
                    success = False
            else:
                print("❌ Security updates failed tests")
                success = False
        
        return success
    
    def process_regular_updates(self, update_type: str, packages: List[Dict]) -> bool:
        """Process regular dependency updates by type."""
        if not packages:
            print(f"✅ No {update_type} updates needed")
            return True
        
        print(f"📦 Processing {update_type} updates...")
        
        branch_name = f"deps/{update_type}-updates-{datetime.now().strftime('%Y%m%d')}"
        
        if not self.create_update_branch(branch_name):
            return False
        
        success = True
        updated_packages = []
        
        for pkg in packages:
            pkg_name = pkg["name"]
            target_version = pkg["latest_version"]
            
            if self.update_package(pkg_name, target_version):
                updated_packages.append(f"{pkg_name} ({pkg['version']} → {target_version})")
                self.update_summary["updates"].append({
                    "package": pkg_name,
                    "old_version": pkg["version"],
                    "new_version": target_version,
                    "update_type": update_type,
                    "status": "updated"
                })
            else:
                success = False
                self.update_summary["failed_updates"].append({
                    "package": pkg_name,
                    "reason": f"{update_type}_update_failed",
                    "old_version": pkg["version"],
                    "target_version": target_version
                })
        
        if updated_packages:
            self.update_requirements_file()
            
            if self.run_tests():
                commit_msg = f"deps: {update_type} dependency updates\\n\\nUpdated packages:\\n" + \
                           "\\n".join(f"- {pkg}" for pkg in updated_packages)
                
                if self.commit_updates(commit_msg):
                    self.push_branch(branch_name)
                    print(f"📦 {update_type.title()} updates completed on branch: {branch_name}")
                else:
                    success = False
            else:
                print(f"❌ {update_type.title()} updates failed tests")
                success = False
        
        return success
    
    def generate_update_report(self, output_file: str = "dependency-update-report.json"):
        """Generate a report of all updates performed."""
        with open(output_file, 'w') as f:
            json.dump(self.update_summary, f, indent=2)
        
        print(f"📊 Update report saved to: {output_file}")
        
        # Print summary to console
        print("\\n📋 Update Summary:")
        print(f"  Security updates: {len(self.update_summary['security_updates'])}")
        print(f"  Regular updates: {len(self.update_summary['updates'])}")
        print(f"  Failed updates: {len(self.update_summary['failed_updates'])}")
    
    def run_full_update_cycle(self):
        """Run the complete dependency update cycle."""
        print("🚀 Starting dependency update cycle...")
        
        # Check for outdated packages
        if not self.check_outdated_packages():
            print("ℹ️ No updates needed")
            return
        
        # Categorize updates
        categories = self.categorize_updates()
        
        # Process security updates first (highest priority)
        self.process_security_updates()
        
        # Process patch updates (low risk)
        self.process_regular_updates("patch", categories["patch"])
        
        # Process minor updates (medium risk)
        self.process_regular_updates("minor", categories["minor"])
        
        # Skip major updates in automated runs (high risk)
        if categories["major"]:
            print(f"⚠️ Found {len(categories['major'])} major updates that require manual review:")
            for pkg in categories["major"]:
                print(f"  {pkg['name']}: {pkg['version']} → {pkg['latest_version']}")
            
            for pkg in categories["major"]:
                self.update_summary["breaking_changes"].append({
                    "package": pkg["name"],
                    "old_version": pkg["version"],
                    "new_version": pkg["latest_version"],
                    "status": "manual_review_required"
                })
        
        # Generate report
        self.generate_update_report()
        
        print("✅ Dependency update cycle completed!")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Automated dependency updater")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without making changes")
    parser.add_argument("--security-only", action="store_true", help="Only process security updates")
    parser.add_argument("--type", choices=["security", "patch", "minor", "major"], help="Process specific update type only")
    parser.add_argument("--package", help="Update specific package only")
    parser.add_argument("--check-only", action="store_true", help="Only check for updates, don't perform them")
    
    args = parser.parse_args()
    
    updater = DependencyUpdater(dry_run=args.dry_run)
    
    if args.check_only:
        updater.check_outdated_packages()
        updater.check_security_vulnerabilities()
        return
    
    if args.package:
        # Update specific package
        branch_name = f"deps/update-{args.package}-{datetime.now().strftime('%Y%m%d')}"
        if updater.create_update_branch(branch_name):
            if updater.update_package(args.package):
                updater.update_requirements_file()
                if updater.run_tests():
                    updater.commit_updates(f"deps: update {args.package}")
                    updater.push_branch(branch_name)
        return
    
    if args.security_only:
        updater.process_security_updates()
        return
    
    if args.type:
        updater.check_outdated_packages()
        categories = updater.categorize_updates()
        if args.type == "security":
            updater.process_security_updates()
        else:
            updater.process_regular_updates(args.type, categories[args.type])
        return
    
    # Run full update cycle
    updater.run_full_update_cycle()


if __name__ == "__main__":
    main()