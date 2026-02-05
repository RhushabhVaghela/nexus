# Configuration Files Audit Report

**Audit Date:** 2026-02-03  
**Auditor:** Nexus Code Audit System  
**Scope:** All YAML, JSON, and configuration files in the project

---

## Executive Summary

| Metric | Count |
|--------|-------|
| Total Config Files | 21 |
| Valid JSON Files | 6/6 (100%) |
| Valid YAML Files | 14/14 (100%) |
| Files with Placeholders | 0 |
| Files with Hardcoded Paths | 13 |
| Missing Referenced Configs | 1 |
| Critical Issues | 2 |
| Warnings | 8 |

**Overall Status:** ⚠️ **NEEDS ATTENTION** - Multiple hardcoded paths and one missing config reference found.

---

## 1. Config Files Inventory

### 1.1 Core Config Files (`/config/`)

| File | Type | Purpose | Status |
|------|------|---------|--------|
| `model_config.yaml` | YAML | Model training configuration | ✅ Valid |
| `training_config.yaml` | YAML | Training pipeline settings | ✅ Valid |
| `production.yaml` | YAML | Production hardening config | ✅ Valid |
| `ds_config.json` | JSON | DeepSpeed configuration (standard) | ✅ Valid |
| `ds_config_ultra.json` | JSON | DeepSpeed configuration (ultra) | ✅ Valid |

### 1.2 Dataset/Model Registry (`/configs/`)

| File | Type | Purpose | Status |
|------|------|---------|--------|
| `datasets.yaml` | YAML | Dataset mappings by capability | ⚠️ Hardcoded paths |
| `encoders.yaml` | YAML | Vision/audio encoder paths | ⚠️ Hardcoded paths |
| `decoders.yaml` | YAML | Generation decoder paths | ⚠️ Hardcoded paths |
| `outputs.yaml` | YAML | Output directory structure | ⚠️ Hardcoded paths |
| `global_config.json` | JSON | Global settings (seed, hardware) | ✅ Valid |
| `teacher_registry.json` | JSON | Teacher model registry | ⚠️ Missing path |
| `seed.txt` | Text | Random seed value | ✅ Valid |

### 1.3 Deployment Configs (`/deployment/`)

| File | Type | Purpose | Status |
|------|------|---------|--------|
| `vllm_config.json` | JSON | vLLM serving configuration | ⚠️ Relative path |
| `docker-compose.yml` | YAML | Docker Compose services | ✅ Valid |
| `k8s_deployment.yaml` | YAML | Kubernetes deployment | ✅ Valid |
| `helm/nexus/values.yaml` | YAML | Helm chart values | ✅ Valid |
| `helm/nexus/values-production.yaml` | YAML | Helm production values | ✅ Valid |
| `helm/nexus/Chart.yaml` | YAML | Helm chart metadata | ✅ Valid |
| `monitoring/prometheus.yml` | YAML | Prometheus scraping config | ⚠️ Hardcoded targets |
| `monitoring/alerts.yml` | YAML | Alerting rules | ✅ Valid |

### 1.4 Project Configs (Root)

| File | Type | Purpose | Status |
|------|------|---------|--------|
| `pyproject.toml` | TOML | Python package configuration | ✅ Valid |
| `pytest.ini` | INI | Pytest configuration | ✅ Valid |

---

## 2. Syntax Validation Results

### 2.1 JSON Files

```
✓ configs/global_config.json - Valid JSON
✓ configs/teacher_registry.json - Valid JSON
✓ config/ds_config.json - Valid JSON
✓ config/ds_config_ultra.json - Valid JSON
✓ deployment/vllm_config.json - Valid JSON
✓ deployment/monitoring/grafana-dashboard.json - Valid JSON
```

### 2.2 YAML Files

```
✓ configs/datasets.yaml - Valid YAML
✓ configs/decoders.yaml - Valid YAML
✓ configs/encoders.yaml - Valid YAML
✓ configs/outputs.yaml - Valid YAML
✓ config/model_config.yaml - Valid YAML
✓ config/production.yaml - Valid YAML
✓ config/training_config.yaml - Valid YAML
✓ deployment/docker-compose.yml - Valid YAML
✓ deployment/k8s_deployment.yaml - Valid YAML
✓ deployment/helm/nexus/values.yaml - Valid YAML
✓ deployment/helm/nexus/values-production.yaml - Valid YAML
✓ deployment/helm/nexus/Chart.yaml - Valid YAML
✓ deployment/monitoring/prometheus.yml - Valid YAML
✓ deployment/monitoring/alerts.yml - Valid YAML
```

**Result:** All configuration files have valid syntax. ✅

---

## 3. Placeholder Values Check

### 3.1 Search Results

Searched for: `TODO`, `FIXME`, `CHANGE_ME`, `PLACEHOLDER`, `XXX`

| File | Placeholders Found |
|------|-------------------|
| All config files | None ✅ |

**Result:** No placeholder values found in configuration files. ✅

---

## 4. Hardcoded Paths Analysis

### 4.1 Critical Issues (System-Specific Paths)

| File | Path | Issue | Recommendation |
|------|------|-------|----------------|
| `configs/datasets.yaml:76` | `/mnt/e/data/datasets` | Windows WSL path | Use environment variable |
| `configs/encoders.yaml:6` | `/mnt/e/data/encoders/...` | Windows WSL path | Use environment variable |
| `configs/encoders.yaml:14` | `/mnt/e/data/encoders/...` | Windows WSL path | Use environment variable |
| `configs/encoders.yaml:21` | `/mnt/e/data/encoders/...` | Windows WSL path | Use environment variable |
| `configs/encoders.yaml:28` | `/mnt/e/data/models/...` | Windows WSL path | Use environment variable |
| `configs/decoders.yaml:6` | `/mnt/e/data/decoders/...` | Windows WSL path | Use environment variable |
| `configs/decoders.yaml:13` | `/mnt/e/data/decoders/...` | Windows WSL path | Use environment variable |
| `configs/decoders.yaml:20` | `/mnt/e/data/decoders/...` | Windows WSL path | Use environment variable |
| `configs/outputs.yaml:20` | `/mnt/e/data/output` | Windows WSL path | Use environment variable |
| `config/model_config.yaml:2` | `/mnt/e/data/models/...` | Windows WSL path | Use environment variable |
| `config/model_config.yaml:23` | `/mnt/e/data/downloaded` | Windows WSL path | Use environment variable |
| `config/model_config.yaml:26` | `/mnt/e/models/...` | Windows WSL path | Use environment variable |
| `config/training_config.yaml:7` | `/mnt/e/data/models/...` | Windows WSL path | Use environment variable |
| `config/training_config.yaml:15` | `/mnt/d/Research Experiments/nexus/data` | Absolute path | Use relative path |
| `config/training_config.yaml:18` | `/mnt/e/data/downloaded` | Windows WSL path | Use environment variable |
| `config/training_config.yaml:21` | `/mnt/e/data/unified_multimodal` | Windows WSL path | Use environment variable |
| `config/training_config.yaml:25` | `/mnt/d/Research Experiments/nexus/data` | Absolute path | Use relative path |
| `config/training_config.yaml:26` | `/mnt/e/data/unified_multimodal` | Windows WSL path | Use environment variable |
| `config/training_config.yaml:72` | `/mnt/e/models/...` | Windows WSL path | Use environment variable |
| `config/training_config.yaml:73` | `/mnt/e/checkpoints/...` | Windows WSL path | Use environment variable |
| `config/training_config.yaml:74` | `/mnt/d/Research Experiments/nexus/logs` | Absolute path | Use relative path |

### 4.2 Teacher Registry Issues

| File | Entry | Issue |
|------|-------|-------|
| `configs/teacher_registry.json:110` | `parakeet-tdt-0.6b-v3` | Path: "MISSING" - Model not available |

### 4.3 Deployment Config Issues

| File | Path | Issue | Recommendation |
|------|------|-------|----------------|
| `deployment/vllm_config.json:2` | `checkpoints/stage3_grpo/final` | Relative path may not exist | Verify path or make configurable |
| `deployment/docker-compose.yml:13` | `./checkpoints:/app/checkpoints` | Relative path | Ensure checkpoints directory exists |
| `deployment/monitoring/prometheus.yml:17` | `alertmanager:9093` | Hardcoded service name | Make configurable via env vars |
| `deployment/monitoring/prometheus.yml:36` | `nexus-api:9090` | Hardcoded service name | Make configurable via env vars |

---

## 5. Config Reference Verification

### 5.1 Code References to Configs

| Config File | Referenced In | Status |
|-------------|---------------|--------|
| `config/model_config.yaml` | `src/nexus/training/scripts/10_sft_training.py` | ✅ Found |
| `config/model_config.yaml` | `src/nexus/training/scripts/11_continued_pretraining.py` | ✅ Found |
| `config/model_config.yaml` | `src/nexus/training/scripts/13_safety_finetuning.py` | ✅ Found |
| `configs/datasets.yaml` | `src/nexus/data/scripts/01_download_real_datasets.py` | ✅ Found |
| `configs/datasets.yaml` | `src/nexus/cli/completion.py` | ✅ Found |
| `config/ds_config.json` | Generated dynamically | ✅ Created at runtime |
| `config/ds_config_ultra.json` | Generated dynamically | ✅ Created at runtime |

### 5.2 Missing Config References

| Referenced In | Missing Config | Issue |
|---------------|----------------|-------|
| `tests/test_pipeline_integration.py:143` | `src/config/multimodal_datasets.yaml` | ❌ File does not exist |

**Code snippet:**

```python
config_path = Path(__file__).parent.parent / "src" / "config" / "multimodal_datasets.yaml"
```

**Expected location:** `src/config/multimodal_datasets.yaml`  
**Actual:** Directory `src/config/` does not exist

---

## 6. Security and Production Readiness

### 6.1 Security Configuration (`config/production.yaml`)

| Setting | Value | Status |
|---------|-------|--------|
| `security.enabled` | `true` | ✅ |
| `security.block_on_violation` | `true` | ✅ |
| `rate_limiting.enabled` | `true` | ✅ |
| `metrics.enabled` | `true` | ✅ |
| `health_checks.enabled` | `true` | ✅ |
| `alerting.enabled` | `false` | ⚠️ Disabled |
| `logging.handlers.file.path` | `/var/log/nexus/app.log` | ⚠️ May need permissions setup |

### 6.2 Default/Example Values

| File | Setting | Current Value | Production Ready? |
|------|---------|---------------|-------------------|
| `config/production.yaml:36` | `redis.password` | `null` | ⚠️ Should be set |
| `config/production.yaml:220` | `alerting.channels.email.smtp_host` | `null` | ⚠️ Should be configured |
| `config/production.yaml:223` | `alerting.channels.email.from_address` | `null` | ⚠️ Should be configured |
| `config/production.yaml:227` | `alerting.channels.slack.webhook_url` | `null` | ⚠️ Should be configured |
| `config/production.yaml:232` | `alerting.channels.pagerduty.service_key` | `null` | ⚠️ Should be configured |

### 6.3 Helm Values Production

| Setting | Default | Recommendation |
|---------|---------|----------------|
| `ingress.hosts[0].host` | `nexus.local` | Change to actual domain |
| `tls.secretName` | `nexus-tls` | Ensure TLS cert exists |

---

## 7. Recommendations

### 7.1 Critical (Fix Before Production)

1. **Fix Missing Config Reference**
   - Create `src/config/multimodal_datasets.yaml` or update test reference
   - Priority: HIGH

2. **Replace Hardcoded WSL Paths**
   - Use environment variables: `NEXUS_DATA_DIR`, `NEXUS_MODEL_DIR`, `NEXUS_OUTPUT_DIR`
   - Example: `${NEXUS_DATA_DIR:-/mnt/e/data}/datasets`
   - Priority: HIGH

3. **Fix Missing Teacher Model Path**
   - Update `parakeet-tdt-0.6b-v3` entry with correct path or remove
   - Priority: MEDIUM

### 7.2 High Priority

1. **Configure Production Alerting**
   - Set up email SMTP settings
   - Configure Slack webhook or remove
   - Set up PagerDuty integration or remove

2. **Add Path Validation**
   - Add startup checks to verify all configured paths exist
   - Provide clear error messages for missing directories

3. **Environment-Based Configuration**
   - Create `config/production.yaml.template` with env var placeholders
   - Document required environment variables

### 7.3 Medium Priority

1. **Standardize Path Configuration**
   - Use consistent path structure across all configs
   - Consider using XDG Base Directory specification

2. **Add Config Schema Validation**
   - Implement JSON Schema validation for all configs
   - Add validation step to CI/CD pipeline

3. **Documentation**
   - Create `CONFIGURATION.md` documenting all config options
   - Add inline comments explaining each setting

### 7.4 Low Priority

1. **Config Hot-Reload**
    - Implement file watching for config changes
    - Allow runtime updates without restart

---

## 8. Action Items

| # | Action | Owner | Priority | Status |
|---|--------|-------|----------|--------|
| 1 | Create missing `src/config/multimodal_datasets.yaml` | Dev Team | Critical | ⬜ |
| 2 | Replace hardcoded `/mnt/e/` paths with env vars | Dev Team | Critical | ⬜ |
| 3 | Replace hardcoded `/mnt/d/` paths with env vars | Dev Team | Critical | ⬜ |
| 4 | Fix teacher_registry.json "MISSING" path | Dev Team | High | ⬜ |
| 5 | Create production.yaml.template | DevOps | High | ⬜ |
| 6 | Configure alerting channels in production.yaml | DevOps | High | ⬜ |
| 7 | Add path validation to startup scripts | Dev Team | Medium | ⬜ |
| 8 | Create CONFIGURATION.md documentation | Tech Writers | Medium | ⬜ |
| 9 | Implement config schema validation | Dev Team | Medium | ⬜ |
| 10 | Update Helm values for production domain | DevOps | Medium | ⬜ |

---

## 9. Appendix

### 9.1 Environment Variables to Define

```bash
# Required
export NEXUS_DATA_DIR="/mnt/e/data"
export NEXUS_MODEL_DIR="/mnt/e/models"
export NEXUS_OUTPUT_DIR="/mnt/e/data/output"
export NEXUS_LOG_DIR="/var/log/nexus"

# Optional (for alerting)
export NEXUS_ALERT_EMAIL_SMTP_HOST="smtp.example.com"
export NEXUS_ALERT_EMAIL_FROM="alerts@example.com"
export NEXUS_ALERT_SLACK_WEBHOOK="https://hooks.slack.com/..."
```

### 9.2 Config File Dependencies

```
config/model_config.yaml
    └── Referenced by: training scripts

configs/datasets.yaml
    └── Referenced by: data download scripts, CLI

configs/teacher_registry.json
    └── Referenced by: distillation pipeline

config/production.yaml
    └── Referenced by: production deployment
```

### 9.3 Validation Commands

```bash
# Validate JSON files
python3 -c "import json; json.load(open('config/ds_config.json'))"

# Validate YAML files
python3 -c "import yaml; yaml.safe_load(open('config/production.yaml'))"

# Check for hardcoded paths
grep -r "/mnt/[de]/" configs/ config/
```

---

**End of Report**

*Generated by Nexus Configuration Audit System*  
*Version: 1.0*  
*Date: 2026-02-03*
