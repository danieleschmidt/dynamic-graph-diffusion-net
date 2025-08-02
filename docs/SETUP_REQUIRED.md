# 🔧 Manual Setup Required

This document outlines the manual setup steps required to complete the SDLC implementation due to GitHub App permission limitations.

## 📋 Overview

The automated SDLC implementation has completed all possible configurations, but some GitHub-specific features require manual setup by repository maintainers with appropriate permissions.

## 🚨 Required Actions

### 1. GitHub Workflows Setup

**Status:** ⚠️ Manual Action Required  
**Priority:** High

#### Create Workflow Files

Copy the following workflow templates from `docs/workflows/` to `.github/workflows/`:

```bash
# Create workflows directory
mkdir -p .github/workflows

# Copy workflow templates
cp docs/workflows/ci-workflow-template.yml .github/workflows/ci.yml
cp docs/workflows/security-workflow-template.yml .github/workflows/security.yml
cp docs/workflows/release-workflow-template.yml .github/workflows/release.yml
cp docs/workflows/benchmark-workflow-template.yml .github/workflows/benchmark.yml
```

#### Required GitHub Secrets

Set these secrets in Repository Settings > Secrets and variables > Actions:

- `PYPI_API_TOKEN`: For publishing to PyPI
- `TEST_PYPI_API_TOKEN`: For publishing to Test PyPI  
- `CODECOV_TOKEN`: For code coverage reporting (optional)

### 2. Branch Protection Rules

**Status:** ⚠️ Manual Action Required  
**Priority:** High

Configure branch protection for `main` branch:

1. Go to Settings > Branches
2. Add protection rule for `main` branch:
   - ✅ Require a pull request before merging
   - ✅ Require approvals (minimum 1)
   - ✅ Dismiss stale PR approvals when new commits are pushed
   - ✅ Require review from code owners
   - ✅ Require status checks to pass before merging
   - ✅ Require branches to be up to date before merging
   - ✅ Require conversation resolution before merging
   - ✅ Include administrators
   - ✅ Allow force pushes (for maintainers only)
   - ✅ Allow deletions (disabled for safety)

#### Required Status Checks

Add these status checks (they will appear after first workflow runs):
- `build-and-test`
- `security-scan`
- `lint-and-format`
- `type-check`

### 3. Repository Settings

**Status:** ⚠️ Manual Action Required  
**Priority:** Medium

Update repository settings:

1. **General Settings:**
   - Description: "Advanced variational diffusion model for dynamic graph neural networks"
   - Website: Add documentation URL when available
   - Topics: `machine-learning`, `graph-neural-networks`, `diffusion-models`, `pytorch`, `variational-inference`

2. **Features:**
   - ✅ Issues
   - ✅ Projects
   - ✅ Wiki (if needed)
   - ✅ Discussions (optional)

3. **Pull Requests:**
   - ✅ Allow merge commits
   - ✅ Allow squash merging
   - ✅ Allow rebase merging
   - ✅ Always suggest updating pull request branches
   - ✅ Allow auto-merge
   - ✅ Automatically delete head branches

### 4. GitHub Pages Setup

**Status:** 🔄 Optional  
**Priority:** Low

If you want to host documentation:

1. Go to Settings > Pages
2. Source: Deploy from a branch
3. Branch: `gh-pages` (will be created by release workflow)
4. Folder: `/ (root)`

### 5. Issue and PR Templates

**Status:** ✅ Completed  
**Templates are already created in `.github/`**

The following templates are already configured:
- Bug report template
- Feature request template  
- Pull request template

### 6. Security Settings

**Status:** ⚠️ Manual Action Required  
**Priority:** High

Configure security settings:

1. **Security & Analysis:**
   - ✅ Dependency graph
   - ✅ Dependabot alerts
   - ✅ Dependabot security updates
   - ✅ Code scanning alerts
   - ✅ Secret scanning alerts

2. **Dependabot Configuration:**
   Create `.github/dependabot.yml`:
   ```yaml
   version: 2
   updates:
     - package-ecosystem: "pip"
       directory: "/"
       schedule:
         interval: "weekly"
       reviewers:
         - "danieleschmidt"
       assignees:
         - "danieleschmidt"
   ```

### 7. Environment Configuration

**Status:** ⚠️ Manual Action Required  
**Priority:** Medium

Create deployment environments:

1. Go to Settings > Environments
2. Create environments:
   - `test-pypi` (for Test PyPI deployments)
   - `pypi` (for production PyPI deployments)
3. Configure environment protection rules and secrets

## 🔍 Verification Checklist

After completing the manual setup, verify:

- [ ] All workflow files are created and enabled
- [ ] Required secrets are configured
- [ ] Branch protection rules are active
- [ ] Repository description and topics are set
- [ ] Issue and PR templates work correctly
- [ ] Dependabot is configured and active
- [ ] Security scanning is enabled
- [ ] CODEOWNERS file is recognized

## 🚀 Testing the Setup

1. **Create a test PR:**
   ```bash
   git checkout -b test/setup-verification
   echo "# Test setup" > test-setup.md
   git add test-setup.md
   git commit -m "test: verify SDLC setup"
   git push -u origin test/setup-verification
   ```

2. **Verify workflows run:**
   - Check Actions tab for workflow executions
   - Ensure status checks appear on the PR
   - Verify branch protection prevents direct pushes to main

3. **Test issue creation:**
   - Create a test issue using the templates
   - Verify CODEOWNERS assignment works

## 📞 Support

If you encounter issues during setup:

1. Check the [troubleshooting guide](./troubleshooting.md)
2. Review GitHub's documentation on repository settings
3. Create an issue in this repository for SDLC-specific questions

## 📈 Next Steps

Once manual setup is complete:

1. Monitor workflow executions and fix any configuration issues
2. Customize workflows for project-specific needs
3. Set up additional integrations (Codecov, SonarQube, etc.)
4. Configure monitoring and alerting
5. Train team members on the new SDLC processes

---

🤖 *This document was generated automatically as part of the SDLC implementation*