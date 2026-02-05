-- =============================================================================
-- Nexus Platform Database Schema Initialization
-- =============================================================================
-- This script creates the initial database schema for the Nexus platform.
-- Run this script after creating the database.

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- Create custom types
DO $$ BEGIN
    CREATE TYPE job_status AS ENUM ('pending', 'running', 'completed', 'failed', 'cancelled');
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

DO $$ BEGIN
    CREATE TYPE pipeline_status AS ENUM ('created', 'running', 'paused', 'completed', 'failed', 'cancelled');
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

DO $$ BEGIN
    CREATE TYPE model_architecture AS ENUM ('transformer', 'lstm', 'cnn', 'hybrid', 'other');
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

DO $$ BEGIN
    CREATE TYPE quantization_type AS ENUM ('fp32', 'fp16', 'bf16', 'int8', 'int4', 'Q4_K_M', 'Q5_K_M', 'Q8_0');
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

DO $$ BEGIN
    CREATE TYPE deployment_status AS ENUM ('deploying', 'ready', 'updating', 'failed', 'unhealthy');
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

-- Create schemas
CREATE SCHEMA IF NOT EXISTS api;
CREATE SCHEMA IF NOT EXISTS training;
CREATE SCHEMA IF NOT EXISTS inference;
CREATE SCHEMA IF NOT EXISTS models;
CREATE SCHEMA IF NOT EXISTS monitoring;

-- =============================================================================
-- API Schema - User and Project Management
-- =============================================================================

CREATE TABLE IF NOT EXISTS api.users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    email VARCHAR(255) UNIQUE NOT NULL,
    username VARCHAR(100) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    full_name VARCHAR(200),
    avatar_url TEXT,
    role VARCHAR(50) DEFAULT 'user',
    is_active BOOLEAN DEFAULT true,
    email_verified BOOLEAN DEFAULT false,
    last_login TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS api.organizations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(200) NOT NULL,
    slug VARCHAR(100) UNIQUE NOT NULL,
    description TEXT,
    logo_url TEXT,
    owner_id UUID REFERENCES api.users(id),
    settings JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS api.organization_members (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    organization_id UUID REFERENCES api.organizations(id) ON DELETE CASCADE,
    user_id UUID REFERENCES api.users(id) ON DELETE CASCADE,
    role VARCHAR(50) DEFAULT 'member',
    joined_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(organization_id, user_id)
);

CREATE TABLE IF NOT EXISTS api.projects (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(200) NOT NULL,
    slug VARCHAR(100) NOT NULL,
    description TEXT,
    organization_id UUID REFERENCES api.organizations(id) ON DELETE CASCADE,
    owner_id UUID REFERENCES api.users(id),
    visibility VARCHAR(50) DEFAULT 'private',
    settings JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(organization_id, slug)
);

CREATE TABLE IF NOT EXISTS api.api_keys (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES api.users(id) ON DELETE CASCADE,
    name VARCHAR(200) NOT NULL,
    key_hash VARCHAR(255) NOT NULL,
    key_prefix VARCHAR(20) NOT NULL,
    scopes TEXT[] DEFAULT '{}',
    expires_at TIMESTAMP WITH TIME ZONE,
    last_used_at TIMESTAMP WITH TIME ZONE,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_api_keys_user_id ON api.api_keys(user_id);
CREATE INDEX IF NOT EXISTS idx_api_keys_prefix ON api.api_keys(key_prefix);

-- =============================================================================
-- Training Schema - Training Job Management
# ==============================================================================

CREATE TABLE IF NOT EXISTS training.datasets (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(200) NOT NULL,
    description TEXT,
    project_id UUID REFERENCES api.projects(id) ON DELETE CASCADE,
    owner_id UUID REFERENCES api.users(id),
    source_type VARCHAR(50),
    source_path TEXT,
    schema JSONB DEFAULT '{}',
    statistics JSONB DEFAULT '{}',
    row_count BIGINT,
    file_count INTEGER,
    total_size_bytes BIGINT,
    format VARCHAR(50) DEFAULT 'parquet',
    storage_backend VARCHAR(50) DEFAULT 's3',
    storage_path TEXT,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS training.models (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(200) NOT NULL,
    description TEXT,
    project_id UUID REFERENCES api.projects(id) ON DELETE CASCADE,
    owner_id UUID REFERENCES api.users(id),
    architecture model_architecture DEFAULT 'transformer',
    base_model_id UUID REFERENCES training.models(id),
    base_model_type VARCHAR(100),
    parameter_count BIGINT,
    context_length INTEGER DEFAULT 2048,
    quantization quantization_type,
    huggingface_id VARCHAR(200),
    license VARCHAR(200),
    tags TEXT[],
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS training.training_configs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(200) NOT NULL,
    description TEXT,
    project_id UUID REFERENCES api.projects(id) ON DELETE CASCADE,
    model_id UUID REFERENCES training.models(id),
    owner_id UUID REFERENCES api.users(id),
    hyperparameters JSONB DEFAULT '{}',
    environment_variables JSONB DEFAULT '{}',
    resources JSONB DEFAULT '{"gpu": 1, "memory": "32Gi", "cpu": 8}',
    stopping_conditions JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS training.training_jobs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(200) NOT NULL,
    project_id UUID REFERENCES api.projects(id) ON DELETE CASCADE,
    model_id UUID REFERENCES training.models(id),
    config_id UUID REFERENCES training.training_configs(id),
    owner_id UUID REFERENCES api.users(id),
    status job_status DEFAULT 'pending',
    dataset_id UUID REFERENCES training.datasets(id),
    hyperparameters JSONB DEFAULT '{}',
    metrics JSONB DEFAULT '{}',
    progress JSONB DEFAULT '{"epoch": 0, "total_epochs": null, "step": 0, "total_steps": null}',
    resource_usage JSONB DEFAULT '{}',
    logs TEXT,
    error_message TEXT,
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_training_jobs_project ON training.training_jobs(project_id);
CREATE INDEX IF NOT EXISTS idx_training_jobs_status ON training.training_jobs(status);
CREATE INDEX IF NOT EXISTS idx_training_jobs_owner ON training.training_jobs(owner_id);
CREATE INDEX IF NOT EXISTS idx_training_jobs_created ON training.training_jobs(created_at DESC);

CREATE TABLE IF NOT EXISTS training.training_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    job_id UUID REFERENCES training.training_jobs(id) ON DELETE CASCADE,
    epoch INTEGER,
    step INTEGER,
    loss FLOAT,
    accuracy FLOAT,
    learning_rate FLOAT,
    gpu_memory_mb FLOAT,
    cpu_memory_mb FLOAT,
    throughput_samples_per_sec FLOAT,
    recorded_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_training_metrics_job ON training.training_metrics(job_id);
CREATE INDEX IF NOT EXISTS idx_training_metrics_recorded ON training.training_metrics(recorded_at);

-- =============================================================================
-- Inference Schema - Model Serving
# =============================================================================

CREATE TABLE IF NOT EXISTS inference.endpoints (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(200) NOT NULL,
    description TEXT,
    project_id UUID REFERENCES api.projects(id) ON DELETE CASCADE,
    model_id UUID REFERENCES training.models(id),
    owner_id UUID REFERENCES api.users(id),
    status deployment_status DEFAULT 'deploying',
    url TEXT,
    replica_count INTEGER DEFAULT 1,
    auto_scaling JSONB DEFAULT '{"enabled": true, "min_replicas": 1, "max_replicas": 10}',
    resources JSONB DEFAULT '{"gpu": 1, "memory": "16Gi", "cpu": 4}',
    traffic_split JSONB DEFAULT '{}',
    caching JSONB DEFAULT '{"enabled": true, "ttl": 3600}',
    metadata JSONB DEFAULT '{}',
    deployed_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS inference.endpoint_versions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    endpoint_id UUID REFERENCES inference.endpoints(id) ON DELETE CASCADE,
    model_id UUID REFERENCES training.models(id),
    version INTEGER NOT NULL,
    model_version VARCHAR(100),
    weight FLOAT DEFAULT 0.0,
    status deployment_status DEFAULT 'ready',
    deployment_started_at TIMESTAMP WITH TIME ZONE,
    deployment_completed_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_inference_endpoints_project ON inference.endpoints(project_id);
CREATE INDEX IF NOT EXISTS idx_inference_endpoints_status ON inference.endpoints(status);

CREATE TABLE IF NOT EXISTS inference.request_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    endpoint_id UUID REFERENCES inference.endpoints(id) ON DELETE CASCADE,
    request_id UUID DEFAULT uuid_generate_v4(),
    model_id UUID,
    input_tokens INTEGER,
    output_tokens INTEGER,
    latency_ms FLOAT,
    status_code INTEGER,
    error_message TEXT,
    client_ip VARCHAR(50),
    user_agent TEXT,
    request_size_bytes BIGINT,
    response_size_bytes BIGINT,
    cached BOOLEAN DEFAULT false,
    recorded_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_inference_requests_endpoint ON inference.request_logs(endpoint_id);
CREATE INDEX IF NOT EXISTS idx_inference_requests_recorded ON inference.request_logs(recorded_at DESC);

CREATE TABLE IF NOT EXISTS inference.request_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    endpoint_id UUID REFERENCES inference.endpoints(id) ON DELETE CASCADE,
    requests_total BIGINT,
    requests_failed BIGINT,
    input_tokens_total BIGINT,
    output_tokens_total BIGINT,
    latency_p50_ms FLOAT,
    latency_p95_ms FLOAT,
    latency_p99_ms FLOAT,
    period_start TIMESTAMP WITH TIME ZONE NOT NULL,
    period_end TIMESTAMP WITH TIME ZONE NOT NULL,
    UNIQUE(endpoint_id, period_start)
);

CREATE INDEX IF NOT EXISTS idx_inference_metrics_period ON inference.request_metrics(period_start DESC);

-- =============================================================================
-- Models Schema - Model Registry
# =============================================================================

CREATE TABLE IF NOT EXISTS models.model_versions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_id UUID REFERENCES training.models(id) ON DELETE CASCADE,
    version VARCHAR(100) NOT NULL,
    description TEXT,
    commit_hash VARCHAR(40),
    branch VARCHAR(200),
    training_job_id UUID REFERENCES training.training_jobs(id),
    parameters JSONB DEFAULT '{}',
    files JSONB DEFAULT '{}',
    storage_backend VARCHAR(50) DEFAULT 's3',
    storage_path TEXT,
    size_bytes BIGINT,
    shasum VARCHAR(64),
    metadata JSONB DEFAULT '{}',
    is_default BOOLEAN DEFAULT false,
    is_latest BOOLEAN DEFAULT true,
    released_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_models_versions_model ON models.model_versions(model_id);
CREATE INDEX IF NOT EXISTS idx_models_versions_created ON models.model_versions(created_at DESC);

CREATE TABLE IF NOT EXISTS models.model_files (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_version_id UUID REFERENCES models.model_versions(id) ON DELETE CASCADE,
    path VARCHAR(500) NOT NULL,
    size_bytes BIGINT,
    file_type VARCHAR(50),
    shasum VARCHAR(64),
    storage_backend VARCHAR(50) DEFAULT 's3',
    storage_path TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_models_files_version ON models.model_files(model_version_id);

-- =============================================================================
-- Monitoring Schema - Observability
# =============================================================================

CREATE TABLE IF NOT EXISTS monitoring.system_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    node_name VARCHAR(200) NOT NULL,
    cpu_usage_percent FLOAT,
    memory_usage_percent FLOAT,
    disk_usage_percent FLOAT,
    disk_io_read_bytes BIGINT,
    disk_io_write_bytes BIGINT,
    network_rx_bytes BIGINT,
    network_tx_bytes BIGINT,
    gpu_utilization_percent FLOAT,
    gpu_memory_percent FLOAT,
    gpu_memory_used_bytes BIGINT,
    gpu_memory_total_bytes BIGINT,
    recorded_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_monitoring_metrics_node ON monitoring.system_metrics(node_name);
CREATE INDEX IF NOT EXISTS idx_monitoring_metrics_recorded ON monitoring.system_metrics(recorded_at DESC);

CREATE TABLE IF NOT EXISTS monitoring.alerts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(200) NOT NULL,
    description TEXT,
    severity VARCHAR(50) NOT NULL,
    status VARCHAR(50) NOT NULL,
    source VARCHAR(200),
    labels JSONB DEFAULT '{}',
    annotations JSONB DEFAULT '{}',
    starts_at TIMESTAMP WITH TIME ZONE,
    ends_at TIMESTAMP WITH TIME ZONE,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(name, starts_at)
);

CREATE INDEX IF NOT EXISTS idx_monitoring_alerts_status ON monitoring.alerts(status);
CREATE INDEX IF NOT EXISTS idx_monitoring_alerts_severity ON monitoring.alerts(severity);
CREATE INDEX IF NOT EXISTS idx_monitoring_alerts_recorded ON monitoring.alerts(updated_at DESC);

CREATE TABLE IF NOT EXISTS monitoring.audit_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES api.users(id),
    action VARCHAR(100) NOT NULL,
    resource_type VARCHAR(100),
    resource_id UUID,
    details JSONB DEFAULT '{}',
    ip_address VARCHAR(50),
    user_agent TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_audit_logs_user ON monitoring.audit_logs(user_id);
CREATE INDEX IF NOT EXISTS idx_audit_logs_action ON monitoring.audit_logs(action);
CREATE INDEX IF NOT EXISTS idx_audit_logs_created ON monitoring.audit_logs(created_at DESC);

-- =============================================================================
-- Create Functions and Triggers
# =============================================================================

CREATE OR REPLACE FUNCTION api.update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Apply updated_at triggers
CREATE TRIGGER update_users_updated_at
    BEFORE UPDATE ON api.users
    FOR EACH ROW EXECUTE FUNCTION api.update_updated_at();

CREATE TRIGGER update_organizations_updated_at
    BEFORE UPDATE ON api.organizations
    FOR EACH ROW EXECUTE FUNCTION api.update_updated_at();

CREATE TRIGGER update_projects_updated_at
    BEFORE UPDATE ON api.projects
    FOR EACH ROW EXECUTE FUNCTION api.update_updated_at();

CREATE TRIGGER update_training_jobs_updated_at
    BEFORE UPDATE ON training.training_jobs
    FOR EACH ROW EXECUTE FUNCTION api.update_updated_at();

CREATE TRIGGER update_endpoints_updated_at
    BEFORE UPDATE ON inference.endpoints
    FOR EACH ROW EXECUTE FUNCTION api.update_updated_at();

CREATE TRIGGER update_models_updated_at
    BEFORE UPDATE ON training.models
    FOR EACH ROW EXECUTE FUNCTION api.update_updated_at();

-- =============================================================================
-- Create Indexes for Performance
# =============================================================================

CREATE INDEX IF NOT EXISTS idx_users_email ON api.users(email);
CREATE INDEX IF NOT EXISTS idx_users_username ON api.users(username);
CREATE INDEX IF NOT EXISTS idx_organizations_slug ON api.organizations(slug);
CREATE INDEX IF NOT EXISTS idx_projects_org ON api.projects(organization_id);
CREATE INDEX IF NOT EXISTS idx_datasets_project ON training.datasets(project_id);
CREATE INDEX IF NOT EXISTS idx_models_project ON training.models(project_id);
CREATE INDEX IF NOT EXISTS idx_training_configs_model ON training.training_configs(model_id);
CREATE INDEX IF NOT EXISTS idx_endpoints_model ON inference.endpoints(model_id);

-- Create composite indexes for common queries
CREATE INDEX IF NOT EXISTS idx_training_jobs_status_created ON training.training_jobs(status, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_inference_requests_endpoint_recorded ON inference.request_logs(endpoint_id, recorded_at DESC);
CREATE INDEX IF NOT EXISTS idx_monitoring_metrics_recorded_node ON monitoring.system_metrics(recorded_at DESC, node_name);

-- =============================================================================
-- Set Permissions
# =============================================================================

ALTER DEFAULT PRIVILEGES IN SCHEMA api GRANT ALL ON TABLES TO nexus;
ALTER DEFAULT PRIVILEGES IN SCHEMA training GRANT ALL ON TABLES TO nexus;
ALTER DEFAULT PRIVILEGES IN SCHEMA inference GRANT ALL ON TABLES TO nexus;
ALTER DEFAULT PRIVILEGES IN SCHEMA models GRANT ALL ON TABLES TO nexus;
ALTER DEFAULT PRIVILEGES IN SCHEMA monitoring GRANT ALL ON TABLES TO nexus;

-- Grant SELECT on monitoring tables to readonly role
GRANT SELECT ON ALL TABLES IN SCHEMA monitoring TO readonly;
