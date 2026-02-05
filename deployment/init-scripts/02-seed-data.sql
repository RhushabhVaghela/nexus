-- =============================================================================
-- Nexus Platform Initial Seed Data
# =============================================================================
-- This script seeds the database with initial data for the Nexus platform.

-- =============================================================================
-- Seed Default Organization
# =============================================================================

INSERT INTO api.organizations (id, name, slug, description, owner_id, settings)
VALUES (
    '550e8400-e29b-41d4-a716-446655440000',
    'Nexus Platform',
    'nexus-platform',
    'The main Nexus Platform organization for managing AI/ML workflows.',
    '550e8400-e29b-41d4-a716-446655440001',
    '{
        "features": {
            "multi_gpu": true,
            "distributed_training": true,
            "inference_scaling": true,
            "custom_domains": true
        },
        "limits": {
            "max_projects": 100,
            "max_models_per_project": 50,
            "max_concurrent_training_jobs": 10,
            "max_storage_bytes": 107374182400
        },
        "defaults": {
            "training_gpu": "A100",
            "inference_gpu": "T4",
            "storage_backend": "s3"
        }
    }'
)
ON CONFLICT (slug) DO NOTHING;

-- =============================================================================
-- Seed Default User
# =============================================================================

INSERT INTO api.users (id, email, username, password_hash, full_name, role, is_active, email_verified)
VALUES (
    '550e8400-e29b-41d4-a716-446655440001',
    'admin@nexus-platform.io',
    'admin',
    -- This is a placeholder - in production, use proper password hashing
    '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/X4aYJGYxMnC6C5.Ky',
    'Nexus Admin',
    'admin',
    true,
    true
)
ON CONFLICT (email) DO NOTHING;

-- Add admin to Nexus Platform organization
INSERT INTO api.organization_members (organization_id, user_id, role)
SELECT
    '550e8400-e29b-41d4-a716-446655440000',
    '550e8400-e29b-41d4-a716-446655440001',
    'owner'
WHERE EXISTS (
    SELECT 1 FROM api.organizations WHERE id = '550e8400-e29b-41d4-a716-446655440000'
)
ON CONFLICT (organization_id, user_id) DO NOTHING;

-- =============================================================================
-- Seed Default Project
# =============================================================================

INSERT INTO api.projects (id, name, slug, description, organization_id, owner_id, visibility, settings)
VALUES (
    '660e8400-e29b-41d4-a716-446655440000',
    'Demo Project',
    'demo-project',
    'A demonstration project showcasing Nexus platform capabilities.',
    '550e8400-e29b-41d4-a716-446655440000',
    '550e8400-e29b-41d4-a716-446655440001',
    'public',
    '{
        "features": {
            "auto_scaling": true,
            "caching": true,
            "monitoring": true
        },
        "integrations": {
            "slack": {
                "enabled": true,
                "channel": "#nexus-demo"
            }
        }
    }'
)
ON CONFLICT (organization_id, slug) DO NOTHING;

-- =============================================================================
-- Seed Default Models
# =============================================================================

INSERT INTO training.models (id, name, description, project_id, owner_id, architecture, base_model_type, parameter_count, context_length, quantization, huggingface_id, license, tags)
VALUES
(
    '770e8400-e29b-41d4-a716-446655440000',
    'Llama-3.2-1B-Instruct-Demo',
    'Demo instance of Llama 3.2 1B Instruct model for testing inference capabilities.',
    '660e8400-e29b-41d4-a716-446655440000',
    '550e8400-e29b-41d4-a716-446655440001',
    'transformer',
    'LlamaForCausalLM',
    1000000000,
    131072,
    'Q4_K_M',
    'unsloth/Llama-3.2-1B-Instruct',
    'llama3',
    ARRAY['demo', 'small', 'instruct', 'causal-lm'],
    NOW(), NOW()
)
ON CONFLICT DO NOTHING;

INSERT INTO training.models (id, name, description, project_id, owner_id, architecture, base_model_type, parameter_count, context_length, quantization, huggingface_id, license, tags)
VALUES
(
    '770e8400-e29b-41d4-a716-446655440001',
    'Llama-3.2-3B-Instruct-Demo',
    'Demo instance of Llama 3.2 3B Instruct model for testing inference capabilities.',
    '660e8400-e29b-41d4-a716-446655440000',
    '550e8400-e29b-41d4-a716-446655440001',
    'transformer',
    'LlamaForCausalLM',
    3000000000,
    131072,
    'Q4_K_M',
    'unsloth/Llama-3.2-3B-Instruct',
    'llama3',
    ARRAY['demo', 'medium', 'instruct', 'causal-lm'],
    NOW(), NOW()
)
ON CONFLICT DO NOTHING;

-- =============================================================================
-- Seed Model Versions
# =============================================================================

INSERT INTO models.model_versions (id, model_id, version, description, commit_hash, branch, is_default, is_latest, released_at)
VALUES
(
    '880e8400-e29b-41d4-a716-446655440000',
    '770e8400-e29b-41d4-a716-446655440000',
    '1.0.0',
    'Initial release of the demo model.',
    'abc123def456',
    'main',
    true,
    true,
    NOW()
)
ON CONFLICT DO NOTHING;

INSERT INTO models.model_versions (id, model_id, version, description, commit_hash, branch, is_default, is_latest, released_at)
VALUES
(
    '880e8400-e29b-41d4-a716-446655440001',
    '770e8400-e29b-41d4-a716-446655440001',
    '1.0.0',
    'Initial release of the demo model.',
    'abc123def456',
    'main',
    true,
    true,
    NOW()
)
ON CONFLICT DO NOTHING;

-- =============================================================================
-- Seed Inference Endpoints
# =============================================================================

INSERT INTO inference.endpoints (id, name, description, project_id, model_id, owner_id, status, url, replica_count, auto_scaling, resources, deployed_at)
VALUES
(
    '990e8400-e29b-41d4-a716-446655440000',
    'demo-llama-1b',
    'Demo endpoint for Llama 3.2 1B Instruct model.',
    '660e8400-e29b-41d4-a716-446655440000',
    '770e8400-e29b-41d4-a716-446655440000',
    '550e8400-e29b-41d4-a716-446655440001',
    'ready',
    'https://demo-llama-1b.nexus-platform.io',
    1,
    '{"enabled": true, "min_replicas": 1, "max_replicas": 5, "target_cpu_utilization": 70}',
    '{"gpu": 0.5, "memory": "8Gi", "cpu": 2}',
    NOW()
)
ON CONFLICT DO NOTHING;

INSERT INTO inference.endpoints (id, name, description, project_id, model_id, owner_id, status, url, replica_count, auto_scaling, resources, deployed_at)
VALUES
(
    '990e8400-e29b-41d4-a716-446655440001',
    'demo-llama-3b',
    'Demo endpoint for Llama 3.2 3B Instruct model.',
    '660e8400-e29b-41d4-a716-446655440000',
    '770e8400-e29b-41d4-a716-446655440001',
    '550e8400-e29b-41d4-a716-446655440001',
    'ready',
    'https://demo-llama-3b.nexus-platform.io',
    1,
    '{"enabled": true, "min_replicas": 1, "max_replicas": 5, "target_cpu_utilization": 70}',
    '{"gpu": 1, "memory": "16Gi", "cpu": 4}',
    NOW()
)
ON CONFLICT DO NOTHING;

-- =============================================================================
-- Seed Default Datasets
# =============================================================================

INSERT INTO training.datasets (id, name, description, project_id, owner_id, source_type, row_count, format, metadata)
VALUES
(
    '110e8400-e29b-41d4-a716-446655440000',
    'Demo Dataset',
    'A demonstration dataset for testing training workflows.',
    '660e8400-e29b-41d4-a716-446655440000',
    '550e8400-e29b-41d4-a716-446655440001',
    'synthetic',
    10000,
    'parquet',
    '{
        "columns": ["input_text", "target_text", "category"],
        "categories": ["general", "technical", "creative"],
        "average_length": 256
    }'
)
ON CONFLICT DO NOTHING;

-- =============================================================================
-- Seed Sample Training Configuration
# =============================================================================

INSERT INTO training.training_configs (id, name, description, project_id, model_id, owner_id, hyperparameters, resources)
VALUES
(
    '220e8400-e29b-41d4-a716-446655440000',
    'Demo Fine-tuning Config',
    'Default configuration for demo fine-tuning jobs.',
    '660e8400-e29b-41d4-a716-446655440000',
    '770e8400-e29b-41d4-a716-446655440000',
    '550e8400-e29b-41d4-a716-446655440001',
    '{
        "learning_rate": 0.0002,
        "batch_size": 4,
        "epochs": 3,
        "warmup_steps": 100,
        "weight_decay": 0.01,
        "gradient_accumulation_steps": 4,
        "max_seq_length": 2048,
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"]
    }',
    '{
        "gpu": 1,
        "memory": "32Gi",
        "cpu": 8,
        "storage": "100Gi"
    }'
)
ON CONFLICT DO NOTHING;

-- =============================================================================
-- Seed Sample API Keys (for demo purposes - use secure values in production)
# =============================================================================

INSERT INTO api.api_keys (id, user_id, name, key_hash, key_prefix, scopes, expires_at)
VALUES
(
    '330e8400-e29b-41d4-a716-446655440000',
    '550e8400-e29b-41d4-a716-446655440001',
    'Demo API Key',
    -- Placeholder hash - in production, generate proper keys
    '$2b$12$DemoHashPlaceholderValueHereXXX',
    'nx_demo_',
    ARRAY['read:models', 'read:endpoints', 'create:inference'],
    NOW() + INTERVAL '30 days'
)
ON CONFLICT DO NOTHING;

-- =============================================================================
-- Seed Monitoring Settings
# =============================================================================

INSERT INTO monitoring.alerts (id, name, description, severity, status, labels, annotations)
VALUES
(
    '440e8400-e29b-41d4-a716-446655440000',
    'DemoAlert',
    'Sample alert for demo purposes.',
    'info',
    'resolved',
    '{"service": "demo", "component": "demo"}',
    '{"summary": "Demo alert configuration", "runbook_url": "https://wiki.nexus-platform.io/runbooks/demo"}'
)
ON CONFLICT DO NOTHING;

-- =============================================================================
-- Seed Audit Log Entry
# =============================================================================

INSERT INTO monitoring.audit_logs (user_id, action, resource_type, resource_id, details)
VALUES
(
    '550e8400-e29b-41d4-a716-446655440001',
    'system.initialize',
    'system',
    NULL,
    '{
        "version": "1.0.0",
        "initialized_by": "system",
        "initialized_at": "' || NOW()::TEXT || '",
        "seed_data_version": "1.0"
    }'
);

-- =============================================================================
-- Create Views for Common Queries
# =============================================================================

CREATE OR REPLACE VIEW api.user_projects AS
SELECT 
    p.*,
    o.name as organization_name
FROM api.projects p
LEFT JOIN api.organizations o ON p.organization_id = o.id
WHERE p.owner_id = api.current_user_id() 
   OR p.id IN (
       SELECT project_id 
       FROM api.organization_members 
       WHERE user_id = api.current_user_id()
   )
ORDER BY p.updated_at DESC;

CREATE OR REPLACE VIEW training.job_status_summary AS
SELECT 
    project_id,
    status,
    COUNT(*) as count,
    MIN(created_at) as oldest_job,
    MAX(created_at) as newest_job
FROM training.training_jobs
GROUP BY project_id, status;

CREATE OR REPLACE VIEW inference.endpoint_metrics_summary AS
SELECT 
    e.id,
    e.name,
    e.status,
    e.replica_count,
    m.requests_total,
    m.requests_failed,
    m.input_tokens_total,
    m.output_tokens_total,
    m.latency_p95_ms,
    m.period_start,
    m.period_end
FROM inference.endpoints e
LEFT JOIN inference.request_metrics m ON e.id = m.endpoint_id
AND m.period_start = (
    SELECT MAX(period_start) 
    FROM inference.request_metrics 
    WHERE endpoint_id = e.id
);

-- =============================================================================
-- Set Sequence Values
# =============================================================================

SELECT setval('api.users_id_seq', 1000000001, true);
SELECT setval('api.organizations_id_seq', 1000000001, true);
SELECT setval('api.projects_id_seq', 1000000001, true);
SELECT setval('training.datasets_id_seq', 1000000001, true);
SELECT setval('training.models_id_seq', 1000000001, true);
SELECT setval('training.training_jobs_id_seq', 1000000001, true);
SELECT setval('inference.endpoints_id_seq', 1000000001, true);
SELECT setval('models.model_versions_id_seq', 1000000001, true);

-- =============================================================================
-- Final Comments
# =============================================================================

COMMENT ON SCHEMA api IS 'API and user management schema';
COMMENT ON SCHEMA training IS 'Training job and model management schema';
COMMENT ON SCHEMA inference IS 'Inference serving and endpoint management schema';
COMMENT ON SCHEMA models IS 'Model registry and versioning schema';
COMMENT ON SCHEMA monitoring IS 'Observability and audit logging schema';

COMMENT ON TABLE api.users IS 'Nexus platform users';
COMMENT ON TABLE api.organizations IS 'Organizations for multi-tenant support';
COMMENT ON TABLE api.projects IS 'Projects for organizing work';
COMMENT ON TABLE training.datasets IS 'Training datasets';
COMMENT ON TABLE training.models IS 'Registered models';
COMMENT ON TABLE training.training_jobs IS 'Training job records';
COMMENT ON TABLE inference.endpoints IS 'Inference endpoints for model serving';
COMMENT ON TABLE inference.request_logs IS 'Inference request logs';
COMMENT ON TABLE monitoring.audit_logs IS 'Audit trail for compliance';
