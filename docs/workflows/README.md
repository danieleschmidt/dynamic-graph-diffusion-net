# GitHub Workflows Documentation

This directory contains workflow templates and documentation for setting up comprehensive CI/CD pipelines for the Dynamic Graph Diffusion Net project.

## Required Setup

Since GitHub workflow files cannot be created programmatically in this environment, you'll need to manually create these workflow files in `.github/workflows/` directory.

## Workflow Files to Create

### 1. Main CI/CD Pipeline
**File:** `.github/workflows/ci.yml`
- Runs on: Push to main, PRs, and release tags
- Tests across Python 3.8-3.11 and PyTorch versions
- Includes security scanning and dependency checks
- See: [ci-workflow-template.yml](./ci-workflow-template.yml)

### 2. Security Scanning
**File:** `.github/workflows/security.yml`
- Dependency vulnerability scanning
- Code security analysis
- SBOM generation
- See: [security-workflow-template.yml](./security-workflow-template.yml)

### 3. Release Automation
**File:** `.github/workflows/release.yml`
- Automated PyPI publishing
- GitHub release creation
- Documentation deployment
- See: [release-workflow-template.yml](./release-workflow-template.yml)

### 4. Performance Benchmarking
**File:** `.github/workflows/benchmark.yml`
- Automated performance testing
- Benchmark result tracking
- Memory usage analysis
- See: [benchmark-workflow-template.yml](./benchmark-workflow-template.yml)

## Setup Instructions

1. Create `.github/workflows/` directory in your repository root
2. Copy the template files from this directory and rename them (remove `-template` suffix)
3. Customize the workflows for your specific needs
4. Set up required secrets in repository settings
5. Enable workflows in Actions tab

## Required Secrets

Set these in Repository Settings > Secrets and variables > Actions:

- `PYPI_API_TOKEN`: For publishing to PyPI
- `TEST_PYPI_API_TOKEN`: For publishing to Test PyPI
- `CODECOV_TOKEN`: For code coverage reporting (optional)

## Workflow Triggers

- **CI Pipeline**: All pushes and PRs
- **Security Scans**: Daily schedule + manual trigger
- **Release**: On version tags (v*)
- **Benchmarks**: Weekly schedule + PR comments

## Monitoring

All workflows include comprehensive logging and artifact collection for debugging and monitoring purposes.