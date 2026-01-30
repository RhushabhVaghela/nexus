# Changelog Maintenance Process

This document describes the version numbering, release process, and automated changelog generation for Nexus.

## Table of Contents

- [Version Numbering](#version-numbering)
- [Release Process](#release-process)
- [Changelog Format](#changelog-format)
- [Automated Generation](#automated-generation)
- [Manual Updates](#manual-updates)

---

## Version Numbering

Nexus follows [Semantic Versioning 2.0.0](https://semver.org/):

```
MAJOR.MINOR.PATCH
```

### Version Components

| Component | When to Increment | Example |
|-----------|-------------------|---------|
| **MAJOR** | Breaking API changes | `6.x.x` → `7.0.0` |
| **MINOR** | New features (backward compatible) | `6.1.x` → `6.2.0` |
| **PATCH** | Bug fixes (backward compatible) | `6.1.0` → `6.1.1` |

### Pre-release Versions

```
6.2.0-alpha.1    # Early testing
6.2.0-beta.2     # Feature complete, testing
6.2.0-rc.1       # Release candidate
```

### Version Tags

- **Latest:** Always points to most recent stable release
- **Stable:** Last known good version
- **LTS:** Long-term support versions (even major numbers)

---

## Release Process

### 1. Pre-Release Checklist

- [ ] All tests passing
- [ ] Documentation updated
- [ ] Changelog updated
- [ ] Version bumped
- [ ] Release notes drafted

### 2. Version Bump

Using `bump2version`:

```bash
# Patch release (bug fixes)
bump2version patch

# Minor release (new features)
bump2version minor

# Major release (breaking changes)
bump2version major

# Pre-release
bump2version prerelease --tag-name='{new_version}'
```

This updates:

- `pyproject.toml`
- `src/__init__.py`
- `docs/conf.py`
- Git tags

### 3. Changelog Update

Move items from `[Unreleased]` to new version section:

```markdown
## [6.2.0] - 2024-01-30

### Added
- Multimodal training support for video data
- New TTS integration with voice cloning
- Distributed training improvements

### Changed
- Optimized memory usage in inference pipeline
- Updated default learning rates

### Fixed
- Resolved OOM issues in large model training
- Fixed tokenizer alignment bugs

### Deprecated
- Old model loading API (will be removed in 7.0.0)

### Security
- Updated dependencies for CVE-2024-XXXX
```

### 4. Create Release PR

```bash
# Create release branch
git checkout -b release/v6.2.0

# Commit changes
git add -A
git commit -m "chore(release): prepare v6.2.0"

# Push and create PR
git push origin release/v6.2.0
```

PR Template:

```markdown
## Release v6.2.0

### Highlights
- New multimodal video support
- Performance improvements
- Bug fixes

### Full Changelog
See [CHANGELOG.md](../CHANGELOG.md)

### Migration Guide
No breaking changes - direct upgrade supported.
```

### 5. Merge and Tag

After PR approval:

```bash
# Merge to main
git checkout main
git merge release/v6.2.0

# Create tag
git tag -a v6.2.0 -m "Release version 6.2.0"
git push origin v6.2.0
```

### 6. GitHub Release

Create release on GitHub:

```bash
gh release create v6.2.0 \
  --title "Nexus v6.2.0" \
  --notes-file RELEASE_NOTES.md \
  --draft
```

### 7. Post-Release

- [ ] Deploy documentation
- [ ] Update Docker images
- [ ] Announce on social media
- [ ] Notify mailing list

---

## Changelog Format

Nexus uses [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) format:

```markdown
# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- New features

### Changed
- Changes in existing functionality

### Deprecated
- Soon-to-be removed features

### Removed
- Now removed features

### Fixed
- Bug fixes

### Security
- Security improvements

## [6.1.0] - 2024-01-15

### Added
- Feature description (#123)

### Fixed
- Bug fix description (#456)

## [6.0.0] - 2023-12-01
...
```

### Categories

| Category | Description |
|----------|-------------|
| `Added` | New features |
| `Changed` | Changes to existing functionality |
| `Deprecated` | Features marked for removal |
| `Removed` | Removed features |
| `Fixed` | Bug fixes |
| `Security` | Security-related changes |

---

## Automated Generation

### GitHub Actions Workflow

```yaml
name: Generate Changelog

on:
  push:
    branches: [main]
  pull_request:
    types: [closed]

jobs:
  changelog:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0
      
      - name: Generate changelog
        uses: github-changelog-generator/github-changelog-generator@v1
        with:
          token: ${{ secrets.GITHUB_TOKEN }}
          output: CHANGELOG.md
      
      - name: Commit changelog
        run: |
          git config user.name "GitHub Actions"
          git config user.email "actions@github.com"
          git add CHANGELOG.md
          git diff --staged --quiet || git commit -m "docs: update changelog"
          git push
```

### Conventional Commits

Use conventional commit format for automatic categorization:

```bash
# Added
feat: add new multimodal encoder
git commit -m "feat(multimodal): add video processing support"

# Fixed
fix: resolve memory leak in training
git commit -m "fix(training): resolve OOM in distributed setup"

# Changed
refactor: optimize data loading
git commit -m "refactor(data): optimize batch processing"

# Security
security: update vulnerable dependency
git commit -m "security(deps): patch CVE-2024-XXXX"
```

### Auto-Categorization Rules

| Commit Type | Changelog Category |
|-------------|-------------------|
| `feat:` | Added |
| `fix:` | Fixed |
| `refactor:` | Changed |
| `perf:` | Changed |
| `docs:` | (skip) |
| `style:` | (skip) |
| `test:` | (skip) |
| `chore:` | (skip) |
| `security:` | Security |
| `BREAKING CHANGE:` | Changed (breaking) |

### Release Drafter

```yaml
# .github/release-drafter.yml
template: |
  ## What's Changed

  $CHANGES

categories:
  - title: 🚀 Features
    labels:
      - feature
      - enhancement
  - title: 🐛 Bug Fixes
    labels:
      - fix
      - bugfix
      - bug
  - title: 🧰 Maintenance
    labels:
      - chore
      - refactor
  - title: 📚 Documentation
    labels:
      - docs
      - documentation
  - title: 🔒 Security
    labels:
      - security

change-template: '- $TITLE @$AUTHOR (#$NUMBER)'
version-resolver:
  major:
    labels:
      - major
      - breaking
  minor:
    labels:
      - minor
      - feature
      - enhancement
  patch:
    labels:
      - patch
      - fix
      - bugfix
      - bug
  default: patch
```

---

## Manual Updates

### When to Update Manually

- Breaking changes need detailed migration notes
- Security issues need explanation
- New features need usage examples
- Deprecations need timelines

### Template for Manual Entries

```markdown
### Added
- **Feature Name**: Brief description
  
  Detailed explanation of what was added, why it was needed,
  and how to use it.
  
  ```python
  # Usage example
  from nexus import new_feature
  result = new_feature.process(data)
  ```
  
  See [documentation](link) for more details.
  
  Contributed by @username in #123

```

### Breaking Changes

```markdown
### Changed
- **⚠️ BREAKING**: Old API method deprecated
  
  The old method `load_model_old()` has been replaced with
  `load_model()`. Migration:
  
  ```python
  # Before
  model = load_model_old(path)
  
  # After
  model = load_model(path, use_safetensors=True)
  ```
  
  The old method will be removed in version 7.0.0.

```

---

## Version Support

### Support Matrix

| Version | Status | Support Until |
|---------|--------|---------------|
| 6.2.x | Active | Current + 6 months |
| 6.1.x | Maintenance | 2024-06-30 |
| 6.0.x | End of Life | 2024-03-31 |
| 5.x.x | End of Life | 2023-12-31 |

### LTS Policy

Even major versions (6.x, 8.x) are LTS releases:
- 2 years of security updates
- 1 year of bug fixes
- Backward compatibility guaranteed

---

## Tools and Scripts

### Generate Release Notes

```bash
#!/bin/bash
# scripts/generate_release_notes.sh

VERSION=$1
PREVIOUS_TAG=$(git describe --tags --abbrev=0)

echo "## Changes since $PREVIOUS_TAG"
echo ""
git log $PREVIOUS_TAG..HEAD --pretty=format:"- %s (%h)" --reverse | grep -E "^(feat|fix|refactor|security):"
```

### Validate Changelog

```bash
#!/bin/bash
# scripts/validate_changelog.sh

# Check format
if ! grep -q "^## \[Unreleased\]" CHANGELOG.md; then
    echo "Error: Missing [Unreleased] section"
    exit 1
fi

# Check links
if ! grep -q "\[6.2.0\]:" CHANGELOG.md; then
    echo "Warning: Missing version comparison link"
fi

echo "Changelog validation passed"
```

### Update Version References

```python
#!/usr/bin/env python3
# scripts/update_version.py

import re
import sys

def update_version_in_file(filepath, pattern, replacement):
    with open(filepath, 'r') as f:
        content = f.read()
    
    new_content = re.sub(pattern, replacement, content)
    
    with open(filepath, 'w') as f:
        f.write(new_content)

if __name__ == "__main__":
    new_version = sys.argv[1]
    
    # Update files
    update_version_in_file(
        'src/__init__.py',
        r'__version__ = ".*"',
        f'__version__ = "{new_version}"'
    )
    
    print(f"Updated version to {new_version}")
```

---

## Best Practices

1. **Update as you go**: Add changelog entries with each PR
2. **Be specific**: Include issue numbers and descriptions
3. **User-focused**: Write for users, not developers
4. **Link references**: Link to docs, issues, and PRs
5. **Review carefully**: Changelogs are permanent records

---

## Questions?

- Check [Semantic Versioning](https://semver.org/)
- See [Keep a Changelog](https://keepachangelog.com/)
- Ask in [Discussions](https://github.com/yourusername/nexus/discussions)
