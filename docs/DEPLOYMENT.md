# Nexus Deployment Guide

## Overview

This comprehensive deployment guide covers all aspects of deploying the Nexus platform to production environments. It provides detailed instructions for containerized deployments using Docker, container orchestration with Kubernetes, Helm chart configuration, and monitoring setup. The guide is structured to take you from initial environment preparation through a fully operational production deployment with monitoring, logging, and alerting.

The Nexus platform is designed for flexible deployment across various infrastructures including on-premises data centers, cloud providers (AWS, GCP, Azure), and hybrid environments. The deployment architecture follows cloud-native principles with support for horizontal scaling, high availability, and disaster recovery. All deployment configurations are version-controlled and repeatable, enabling consistent deployments across development, staging, and production environments.

This guide covers Docker deployment for simple environments, Kubernetes deployment for production-grade orchestration, Helm chart usage for declarative deployments, and comprehensive monitoring setup using Prometheus and Grafana. For security configuration, see the Security Documentation.

## Installation

### Prerequisites

#### System Requirements

```bash
# Minimum Requirements
- CPU: 8 cores (16+ recommended for production)
- Memory: 32 GB RAM (64+ GB recommended)
- Storage: 100 GB SSD (500+ GB recommended for production)
- GPU: Optional - NVIDIA GPU with CUDA support for inference acceleration
  - Minimum: NVIDIA T4 or equivalent
  - Recommended: NVIDIA A100 or H100 for production
- Network: 1 Gbps connectivity (10 Gbps recommended)

# Software Prerequisites
- Docker: 24.0+ with containerd
- Docker Compose: 2.20+ (for Docker deployments)
- Kubernetes: 1.27+ (for K8s deployments)
- Helm: 3.12+ (for Helm deployments)
- kubectl: 1.27+ (for K8s management)
- NVIDIA Container Toolkit (for GPU support)
```

#### Container Runtime Setup

```bash
# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
    sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker

# Configure Docker for GPU access
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

#### Kubernetes Tools Setup

```bash
# Install kubectl
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
chmod +x kubectl
sudo mv kubectl /usr/local/bin/

# Install Helm
curl -fsSL https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash

# Install additional tools
kubectl krew install ctx ns hull
```

### Clone Repository and Dependencies

```bash
# Clone Nexus repository
git clone https://github.com/nexus-ai/nexus.git
cd nexus

# Install Python dependencies
python -m pip install -r requirements.txt

# Install Nexus CLI
python -m pip install -e .

# Verify installation
nexus --version
```

## Docker Deployment

### Basic Docker Setup

```bash
# Create deployment directory
mkdir -p /opt/nexus
cd /opt/nexus

# Copy configuration
cp -r /path/to/nexus/deployment/docker/* .

# Create environment file
cat > .env << EOF
# Nexus Environment Configuration
NEXUS_VERSION=1.0.0
NEXUS_ENV=production

# API Configuration
NEXUS_API_HOST=0.0.0.0
NEXUS_API_PORT=8080
NEXUS_WORKERS=4

# Model Configuration
NEXUS_MODEL_PATH=/models
NEXUS_CACHE_PATH=/cache

# Security Configuration
NEXUS_SECRET_KEY=your-secret-key-change-in-production
NEXUS_AUTH_ENABLED=true

# Database Configuration
NEXUS_DATABASE_URL=postgresql://user:pass@postgres:5432/nexus
NEXUS_REDIS_URL=redis://redis:6379

# Monitoring Configuration
NEXUS_METRICS_ENABLED=true
NEXUS_METRICS_PORT=9090

# Logging Configuration
NEXUS_LOG_LEVEL=INFO
NEXUS_LOG_FORMAT=json
EOF
```

### Docker Compose Configuration

```yaml
# docker-compose.yml
version: '3.8'

services:
  # Nexus API Server
  nexus-api:
    image: nexus-ai/nexus-api:${NEXUS_VERSION:-latest}
    container_name: nexus-api
    restart: unless-stopped
    ports:
      - "${NEXUS_API_PORT:-8080}:8080"
    volumes:
      - ${NEXUS_MODEL_PATH:-./models}:/models:ro
      - ${NEXUS_CACHE_PATH:-./cache}:/cache
      - ./config:/app/config:ro
    environment:
      - NEXUS_ENV=${NEXUS_ENV:-production}
      - NEXUS_API_HOST=${NEXUS_API_HOST:-0.0.0.0}
      - NEXUS_API_PORT=${NEXUS_API_PORT:-8080}
      - NEXUS_WORKERS=${NEXUS_WORKERS:-4}
      - NEXUS_SECRET_KEY=${NEXUS_SECRET_KEY}
      - NEXUS_AUTH_ENABLED=${NEXUS_AUTH_ENABLED:-true}
      - NEXUS_DATABASE_URL=${NEXUS_DATABASE_URL}
      - NEXUS_REDIS_URL=${NEXUS_REDIS_URL}
      - NEXUS_METRICS_ENABLED=${NEXUS_METRICS_ENABLED:-true}
      - NEXUS_LOG_LEVEL=${NEXUS_LOG_LEVEL:-INFO}
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s
    networks:
      - nexus-network
    logging:
      driver: "json-file"
      options:
        max-size: "100m"
        max-file: "5"

  # PostgreSQL Database
  postgres:
    image: postgres:15-alpine
    container_name: nexus-postgres
    restart: unless-stopped
    volumes:
      - postgres-data:/var/lib/postgresql/data
    environment:
      - POSTGRES_USER=nexus
      - POSTGRES_PASSWORD=${POSTGRES_PASSWORD:-nexus-secret}
      - POSTGRES_DB=nexus
    command: >
      postgres
      -c max_connections=200
      -c shared_buffers=1GB
      -c effective_cache_size=3GB
      -c maintenance_work_mem=256MB
      -c checkpoint_completion_target=0.9
      -c wal_buffers=64MB
      -c max_wal_size=2GB
      -c min_wal_size=512MB
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U nexus -d nexus"]
      interval: 10s
      timeout: 5s
      retries: 5
    networks:
      - nexus-network
    volumes:
      - postgres-data:/var/lib/postgresql/data

  # Redis for caching and rate limiting
  redis:
    image: redis:7-alpine
    container_name: nexus-redis
    restart: unless-stopped
    command: redis-server --appendonly yes --maxmemory 2gb --maxmemory-policy allkeys-lru
    volumes:
      - redis-data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5
    networks:
      - nexus-network
    volumes:
      - redis-data:/data

  # Prometheus for metrics
  prometheus:
    image: prom/prometheus:v2.47.0
    container_name: nexus-prometheus
    restart: unless-stopped
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--storage.tsdb.retention.time=15d'
      - '--storage.tsdb.retention.size=50GB'
      - '--web.enable-lifecycle'
    volumes:
      - ./monitoring/prometheus/prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - prometheus-data:/prometheus
    ports:
      - "9090:9090"
    networks:
      - nexus-network
    volumes:
      - prometheus-data:/prometheus

  # Grafana for visualization
  grafana:
    image: grafana/grafana:10.1.0
    container_name: nexus-grafana
    restart: unless-stopped
    environment:
      - GF_SECURITY_ADMIN_USER=admin
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD:-nexus-grafana-secret}
      - GF_USERS_ALLOW_SIGN_UP=false
    volumes:
      - ./monitoring/grafana/provisioning:/etc/grafana/provisioning:ro
      - grafana-data:/var/lib/grafana
    ports:
      - "3000:3000"
    depends_on:
      - prometheus
    networks:
      - nexus-network
    volumes:
      - grafana-data:/var/lib/grafana

  # Alertmanager for alerts
  alertmanager:
    image: prom/alertmanager:v0.25.0
    container_name: nexus-alertmanager
    restart: unless-stopped
    command:
      - '--config.file=/etc/alertmanager/alertmanager.yml'
      - '--storage.path=/alertmanager'
    volumes:
      - ./monitoring/alertmanager/alertmanager.yml:/etc/alertmanager/alertmanager.yml:ro
      - alertmanager-data:/alertmanager
    ports:
      - "9093:9093"
    networks:
      - nexus-network
    volumes:
      - alertmanager-data:/alertmanager

networks:
  nexus-network:
    driver: bridge

volumes:
  postgres-data:
    driver: local
  redis-data:
    driver: local
  prometheus-data:
    driver: local
  grafana-data:
    driver: local
  alertmanager-data:
    driver: local
```

### Advanced Docker Configuration

```yaml
# docker-compose.prod.yml - Production configuration with scaling
version: '3.8'

services:
  nexus-api:
    image: nexus-ai/nexus-api:${NEXUS_VERSION:-latest}
    deploy:
      replicas: 3
      update_config:
        parallelism: 1
        delay: 10s
      restart_policy:
        condition: on-failure
        delay: 5s
        max_attempts: 3
      resources:
        limits:
          cpus: '8'
          memory: 32G
        reservations:
          cpus: '4'
          memory: 16G
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    environment:
      - NEXUS_ENV=production
      - NEXUS_SHARD=shard-1
    depends_on:
      - redis
      - postgres
    networks:
      - nexus-network
      - loadbalancer-network

  nginx:
    image: nginx:alpine
    container_name: nexus-nginx
    restart: unless-stopped
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf:ro
      - ./nginx/ssl:/etc/nginx/ssl:ro
    ports:
      - "80:80"
      - "443:443"
    depends_on:
      - nexus-api
    networks:
      - loadbalancer-network
    healthcheck:
      test: ["CMD", "nginx", "-t"]
      interval: 30s
      timeout: 10s
      retries: 3

networks:
  loadbalancer-network:
    driver: bridge
```

### Docker Build Configuration

```dockerfile
# Dockerfile
FROM nvidia/cuda:12.1-base-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3-pip \
    python3-venv \
    curl \
    git \
    vim \
    htop \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Copy Nexus application
COPY src/ /app/src/
COPY config/ /app/config/
COPY deployment/docker/entrypoint.sh /entrypoint.sh

WORKDIR /app

# Create non-root user
RUN useradd -m -s /bin/bash nexus && \
    chown -R nexus:nexus /app && \
    chown -R nexus:nexus /opt/venv

USER nexus

# Expose ports
EXPOSE 8080 9090

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Entrypoint
ENTRYPOINT ["/entrypoint.sh"]
CMD ["nexus", "serve"]
```

### Docker Deployment Commands

```bash
#!/bin/bash
# deploy.sh - Docker deployment script

set -e

# Configuration
VERSION=${1:-latest}
ENV_FILE=".env"
COMPOSE_FILE="docker-compose.yml"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Pre-deployment checks
pre_deploy_checks() {
    log_info "Running pre-deployment checks..."
    
    # Check Docker is running
    if ! docker info > /dev/null 2>&1; then
        log_error "Docker is not running"
        exit 1
    fi
    
    # Check GPU availability
    if nvidia-smi > /dev/null 2>&1; then
        log_info "NVIDIA GPU detected"
        nvidia-smi --query-gpu=name,memory.total --format=csv
    else
        log_warn "No NVIDIA GPU detected - inference will use CPU"
    fi
    
    # Check disk space (minimum 50GB)
    available=$(df -BG / | awk 'NR==2 {print $4}' | sed 's/G//')
    if [ "$available" -lt 50 ]; then
        log_error "Insufficient disk space: ${available}GB available"
        exit 1
    fi
    
    log_info "Pre-deployment checks passed"
}

# Build and deploy
deploy() {
    log_info "Building Nexus containers..."
    docker build -t nexus-ai/nexus-api:${VERSION} .
    
    log_info "Starting Nexus services..."
    docker-compose -f ${COMPOSE_FILE} up -d
    
    log_info "Waiting for services to be healthy..."
    wait_for_health
    
    log_info "Deployment complete!"
}

# Wait for services to be healthy
wait_for_health() {
    local services=("nexus-api" "postgres" "redis" "prometheus" "grafana")
    local max_wait=300
    local waited=0
    
    for service in "${services[@]}"; do
        log_info "Waiting for $service..."
        while ! docker inspect --format='{{.State.Health.Status}}' $service 2>/dev/null | grep -q "healthy"; do
            if [ $waited -ge $max_wait ]; then
                log_error "$service failed to become healthy within ${max_wait}s"
                exit 1
            fi
            sleep 5
            waited=$((waited + 5))
            echo -n "."
        done
        echo ""
        log_info "$service is healthy"
    done
}

# View logs
logs() {
    local service=${1:-nexus-api}
    local lines=${2:-100}
    docker-compose -f ${COMPOSE_FILE} logs --tail=$lines -f $service
}

# Stop deployment
stop() {
    log_info "Stopping Nexus services..."
    docker-compose -f ${COMPOSE_FILE} down
    
    log_info "Services stopped"
}

# Restart deployment
restart() {
    log_info "Restarting Nexus services..."
    docker-compose -f ${COMPOSE_FILE} restart
    
    log_info "Waiting for services to be healthy..."
    wait_for_health
    
    log_info "Restart complete"
}

# Update deployment
update() {
    log_info "Pulling latest images..."
    docker-compose -f ${COMPOSE_FILE} pull
    
    log_info "Updating services..."
    docker-compose -f ${COMPOSE_FILE} up -d
    
    log_info "Waiting for services to be healthy..."
    wait_for_health
    
    log_info "Update complete"
}

# View status
status() {
    docker-compose -f ${COMPOSE_FILE} ps
    echo ""
    echo "GPU Usage:"
    nvidia-smi 2>/dev/null || echo "No GPU available"
}

# Show usage
usage() {
    echo "Usage: $0 <command> [options]"
    echo ""
    echo "Commands:"
    echo "  deploy [version]    Deploy Nexus (default: latest)"
    echo "  stop                Stop all services"
    echo "  restart             Restart all services"
    echo "  update              Update to latest version"
    echo "  logs [service]      View logs (default: nexus-api)"
    echo "  status              View service status"
    echo "  health              Check service health"
}

# Main entrypoint
case "${1:-deploy}" in
    deploy)
        pre_deploy_checks
        deploy
        ;;
    stop)
        stop
        ;;
    restart)
        restart
        ;;
    update)
        update
        ;;
    logs)
        logs "${2:-nexus-api}" "${3:-100}"
        ;;
    status)
        status
        ;;
    health)
        wait_for_health
        ;;
    *)
        usage
        exit 1
        ;;
esac
```

## Kubernetes Deployment

### Kubernetes Cluster Setup

```bash
#!/bin/bash
# setup-cluster.sh - Kubernetes cluster setup script

# Create Kubernetes namespace
kubectl create namespace nexus

# Create service account
kubectl create serviceaccount nexus -n nexus

# Create RBAC resources
cat << EOF | kubectl apply -f -
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: nexus-role
rules:
- apiGroups: [""]
  resources: ["pods", "services", "endpoints", "configmaps", "secrets"]
  verbs: ["get", "list", "watch", "create", "update", "patch", "delete"]
- apiGroups: ["apps"]
  resources: ["deployments", "statefulsets"]
  verbs: ["get", "list", "watch", "create", "update", "patch", "delete"]
- apiGroups: ["autoscaling"]
  resources: ["horizontalpodautoscalers"]
  verbs: ["get", "list", "watch", "create", "update", "patch", "delete"]
- apiGroups: [""]
  resources: ["pods/exec"]
  verbs: ["create"]
EOF

# Bind role to service account
kubectl create clusterrolebinding nexus-binding \
    --clusterrole=nexus-role \
    --serviceaccount=nexus:nexus
```

### Kubernetes Deployment Configuration

```yaml
# deployment/k8s_deployment.yaml
---
# Nexus API Deployment
apiVersion: apps/v1
kind: Deployment
metadata:
  name: nexus-api
  namespace: nexus
  labels:
    app: nexus-api
    version: v1
spec:
  replicas: 3
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
  selector:
    matchLabels:
      app: nexus-api
  template:
    metadata:
      labels:
        app: nexus-api
        version: v1
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "9090"
        prometheus.io/path: "/metrics"
    spec:
      serviceAccountName: nexus
      restartPolicy: Always
      terminationGracePeriodSeconds: 30
      securityContext:
        runAsNonRoot: true
        runAsUser: 1000
        fsGroup: 1000
      containers:
      - name: nexus-api
        image: nexus-ai/nexus-api:latest
        imagePullPolicy: Always
        ports:
        - containerPort: 8080
          name: http
        - containerPort: 9090
          name: metrics
        env:
        - name: NEXUS_ENV
          value: "production"
        - name: NEXUS_API_HOST
          value: "0.0.0.0"
        - name: NEXUS_API_PORT
          value: "8080"
        - name: NEXUS_WORKERS
          value: "4"
        - name: NEXUS_SECRET_KEY
          valueFrom:
            secretKeyRef:
              name: nexus-secrets
              key: secret-key
        - name: NEXUS_DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: nexus-secrets
              key: database-url
        - name: NEXUS_REDIS_URL
          valueFrom:
            configMapKeyRef:
              name: nexus-config
              key: redis-url
        resources:
          requests:
            memory: "16Gi"
            cpu: "4"
            nvidia.com/gpu: 1
          limits:
            memory: "32Gi"
            cpu: "8"
            nvidia.com/gpu: 1
        volumeMounts:
        - name: model-storage
          mountPath: /models
        - name: cache-storage
          mountPath: /cache
        - name: config-volume
          mountPath: /app/config
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 60
          periodSeconds: 30
          timeoutSeconds: 10
          failureThreshold: 3
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3
      volumes:
      - name: model-storage
        persistentVolumeClaim:
          claimName: nexus-models-pvc
      - name: cache-storage
        persistentVolumeClaim:
          claimName: nexus-cache-pvc
      - name: config-volume
        configMap:
          name: nexus-config
      affinity:
        podAntiAffinity:
          preferredDuringSchedulingIgnoredDuringExecution:
          - weight: 100
            podAffinityTerm:
              labelSelector:
                matchLabels:
                  app: nexus-api
              topologyKey: kubernetes.io/hostname
      topologySpreadConstraints:
      - maxSkew: 1
        topologyKey: topology.kubernetes.io/zone
        whenUnsatisfiable: DoNotSchedule
        labelSelector:
          matchLabels:
            app: nexus-api
---
# Nexus API Service
apiVersion: v1
kind: Service
metadata:
  name: nexus-api
  namespace: nexus
  labels:
    app: nexus-api
spec:
  type: ClusterIP
  ports:
  - port: 80
    targetPort: 8080
    protocol: TCP
    name: http
  selector:
    app: nexus-api
---
# Horizontal Pod Autoscaler
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: nexus-api-hpa
  namespace: nexus
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: nexus-api
  minReplicas: 3
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 10
        periodSeconds: 60
    scaleUp:
      stabilizationWindowSeconds: 0
      policies:
      - type: Percent
        value: 100
        periodSeconds: 15
---
# Pod Disruption Budget
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: nexus-api-pdb
  namespace: nexus
spec:
  minAvailable: 2
  selector:
    matchLabels:
      app: nexus-api
```

### Kubernetes StatefulSet for Data Services

```yaml
# deployment/k8s_statefulset.yaml
---
# PostgreSQL StatefulSet
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: nexus-postgres
  namespace: nexus
spec:
  serviceName: nexus-postgres
  replicas: 1
  selector:
    matchLabels:
      app: nexus-postgres
  template:
    metadata:
      labels:
        app: nexus-postgres
    spec:
      terminationGracePeriodSeconds: 30
      containers:
      - name: postgres
        image: postgres:15-alpine
        env:
        - name: POSTGRES_USER
          valueFrom:
            secretKeyRef:
              name: nexus-secrets
              key: postgres-user
        - name: POSTGRES_PASSWORD
          valueFrom:
            secretKeyRef:
              name: nexus-secrets
              key: postgres-password
        - name: POSTGRES_DB
          value: nexus
        ports:
        - containerPort: 5432
        resources:
          requests:
            memory: "4Gi"
            cpu: "2"
          limits:
            memory: "8Gi"
            cpu: "4"
        volumeMounts:
        - name: postgres-data
          mountPath: /var/lib/postgresql/data
        - name: postgres-config
          mountPath: /etc/postgresql
      volumes:
      - name: postgres-config
        configMap:
          name: postgres-config
  volumeClaimTemplates:
  - metadata:
      name: postgres-data
    spec:
      accessModes: ["ReadWriteOnce"]
      storageClassName: fast-storage
      resources:
        requests:
          storage: 100Gi
---
# Redis StatefulSet
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: nexus-redis
  namespace: nexus
spec:
  serviceName: nexus-redis
  replicas: 1
  selector:
    matchLabels:
      app: nexus-redis
  template:
    metadata:
      labels:
        app: nexus-redis
    spec:
      terminationGracePeriodSeconds: 30
      containers:
      - name: redis
        image: redis:7-alpine
        command: ["redis-server"]
        args: ["--appendonly", "yes", "--maxmemory", "2gb", "--maxmemory-policy", "allkeys-lru"]
        ports:
        - containerPort: 6379
        resources:
          requests:
            memory: "2Gi"
            cpu: "1"
          limits:
            memory: "4Gi"
            cpu: "2"
        volumeMounts:
        - name: redis-data
          mountPath: /data
      volumes:
      - name: redis-data
        persistentVolumeClaim:
          claimName: redis-data-pvc
---
# Redis Service
apiVersion: v1
kind: Service
metadata:
  name: nexus-redis
  namespace: nexus
  labels:
    app: nexus-redis
spec:
  clusterIP: None
  ports:
  - port: 6379
    targetPort: 6379
  selector:
    app: nexus-redis
```

### Kubernetes ConfigMaps and Secrets

```yaml
# deployment/k8s_configmap.yaml
---
# Nexus Configuration ConfigMap
apiVersion: v1
kind: ConfigMap
metadata:
  name: nexus-config
  namespace: nexus
data:
  redis-url: "redis://nexus-redis-0.nexus-redis:6379"
  model-path: "/models"
  cache-path: "/cache"
  log-level: "INFO"
  log-format: "json"
  metrics-enabled: "true"
  metrics-port: "9090"
---
# PostgreSQL Configuration ConfigMap
apiVersion: v1
kind: ConfigMap
metadata:
  name: postgres-config
  namespace: nexus
data:
  postgresql.conf: |
    max_connections = 200
    shared_buffers = 2GB
    effective_cache_size = 6GB
    maintenance_work_mem = 512MB
    checkpoint_completion_target = 0.9
    wal_buffers = 64MB
    max_wal_size = 4GB
    min_wal_size = 1GB
    work_mem = 64MB
    effective_io_concurrency = 200
    max_parallel_workers_per_gather = 2
---
# Nexus Secrets
apiVersion: v1
kind: Secret
metadata:
  name: nexus-secrets
  namespace: nexus
type: Opaque
stringData:
  secret-key: "your-production-secret-key-min-32-chars"
  database-url: "postgresql://nexus:password@nexus-postgres-0.nexus-postgres:5432/nexus"
  postgres-user: "nexus"
  postgres-password: "secure-password-here"
  grafana-admin-password: "grafana-admin-password"
```

### Kubernetes Ingress Configuration

```yaml
# deployment/k8s_ingress.yaml
---
# Ingress Configuration
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: nexus-ingress
  namespace: nexus
  annotations:
    kubernetes.io/ingress.class: nginx
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    nginx.ingress.kubernetes.io/proxy-body-size: "100m"
    nginx.ingress.kubernetes.io/proxy-read-timeout: "300"
    nginx.ingress.kubernetes.io/proxy-send-timeout: "300"
    nginx.ingress.kubernetes.io/proxy-connect-timeout: "60"
    nginx.ingress.kubernetes.io/configuration-snippet: |
      more_set_headers "X-Request-ID: $req_id";
      more_set_headers "X-Response-Time: $upstream_response_time";
    cert-manager.io/cluster-issuer: letsencrypt-prod
spec:
  tls:
  - hosts:
    - api.nexus.example.com
    - grafana.nexus.example.com
    - prometheus.nexus.example.com
    secretName: nexus-tls
  rules:
  - host: api.nexus.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: nexus-api
            port:
              number: 80
  - host: grafana.nexus.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: nexus-grafana
            port:
              number: 80
  - host: prometheus.nexus.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: nexus-prometheus
            port:
              number: 9090
---
# Certificate (using cert-manager)
apiVersion: cert-manager.io/v1
kind: Certificate
metadata:
  name: nexus-tls
  namespace: nexus
spec:
  secretName: nexus-tls
  issuerRef:
    name: letsencrypt-prod
    kind: ClusterIssuer
  dnsNames:
  - api.nexus.example.com
  - grafana.nexus.example.com
  - prometheus.nexus.example.com
```

### Kubernetes Network Policies

```yaml
# deployment/k8s_networkpolicy.yaml
---
# Default deny all ingress
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: nexus-default-deny
  namespace: nexus
spec:
  podSelector: {}
  policyTypes:
  - Ingress
---
# Allow traffic within namespace
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: nexus-allow-internal
  namespace: nexus
spec:
  podSelector:
    matchLabels:
      app: nexus-api
  policyTypes:
  - Ingress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: nexus
---
# Allow traffic from ingress controller
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: nexus-allow-ingress
  namespace: nexus
spec:
  podSelector:
    matchLabels:
      app: nexus-api
  policyTypes:
  - Ingress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: ingress-nginx
    ports:
    - protocol: TCP
      port: 8080
---
# Allow traffic to PostgreSQL
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: nexus-allow-postgres
  namespace: nexus
spec:
  podSelector:
    matchLabels:
      app: nexus-postgres
  policyTypes:
  - Ingress
  ingress:
  - from:
    - podSelector:
        matchLabels:
          app: nexus-api
    ports:
    - protocol: TCP
      port: 5432
---
# Allow traffic to Redis
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: nexus-allow-redis
  namespace: nexus
spec:
  podSelector:
    matchLabels:
      app: nexus-redis
  policyTypes:
  - Ingress
  ingress:
  - from:
    - podSelector:
        matchLabels:
          app: nexus-api
    ports:
    - protocol: TCP
      port: 6379
```

## Helm Chart Deployment

### Helm Chart Structure

```
nexus/
├── Chart.yaml
├── values.yaml
├── values-production.yaml
├── values-development.yaml
├── templates/
│   ├── _helpers.tpl
│   ├── deployment.yaml
│   ├── service.yaml
│   ├── hpa.yaml
│   ├── pdb.yaml
│   ├── configmap.yaml
│   ├── secret.yaml
│   ├── ingress.yaml
│   ├── pvc.yaml
│   └── networkpolicy.yaml
└── charts/
    ├── postgresql/
    ├── redis/
    ├── prometheus/
    └── grafana/
```

### Helm Chart Values Configuration

```yaml
# values.yaml - Default values
replicaCount: 1

image:
  repository: nexus-ai/nexus-api
  tag: latest
  pullPolicy: IfNotPresent

imagePullSecrets:
  - name: nexus-registry-secret

nameOverride: nexus
fullnameOverride: nexus

service:
  type: ClusterIP
  port: 8080
  metricsPort: 9090

ingress:
  enabled: true
  className: nginx
  annotations:
    kubernetes.io/tls-acme: "true"
  hosts:
    - host: api.nexus.example.com
      paths:
        - path: /
          pathType: Prefix
  tls:
    - secretName: nexus-tls
      hosts:
        - api.nexus.example.com

resources:
  requests:
    memory: "16Gi"
    cpu: "4"
    nvidia.com/gpu: 1
  limits:
    memory: "32Gi"
    cpu: "8"
    nvidia.com/gpu: 1

autoscaling:
  enabled: true
  minReplicas: 1
  maxReplicas: 10
  targetCPUUtilizationPercentage: 70
  targetMemoryUtilizationPercentage: 80

nodeSelector:
  node-type: gpu

tolerations:
  - key: nvidia.com/gpu
    operator: Exists
    effect: NoSchedule

affinity:
  podAntiAffinity:
    preferredDuringSchedulingIgnoredDuringExecution:
      - weight: 100
        podAffinityTerm:
          labelSelector:
            matchLabels:
              app.kubernetes.io/name: nexus
          topologyKey: kubernetes.io/hostname

config:
  NEXUS_ENV: production
  NEXUS_API_HOST: "0.0.0.0"
  NEXUS_API_PORT: "8080"
  NEXUS_WORKERS: "4"
  NEXUS_LOG_LEVEL: INFO
  NEXUS_LOG_FORMAT: json
  NEXUS_METRICS_ENABLED: "true"
  NEXUS_METRICS_PORT: "9090"

secrets:
  existingSecret: nexus-secrets

persistence:
  models:
    enabled: true
    existingClaim: nexus-models-pvc
    mountPath: /models
    subPath: models
  cache:
    enabled: true
    existingClaim: nexus-cache-pvc
    mountPath: /cache
    subPath: cache

postgresql:
  enabled: true
  auth:
    database: nexus
    existingSecret: nexus-secrets
  primary:
    persistence:
      size: 100Gi
    resources:
      requests:
        memory: "4Gi"
        cpu: "2"

redis:
  enabled: true
  architecture: standalone
  master:
    persistence:
      size: 10Gi
    resources:
      requests:
        memory: "2Gi"
        cpu: "1"
```

```yaml
# values-production.yaml - Production configuration
replicaCount: 3

autoscaling:
  enabled: true
  minReplicas: 3
  maxReplicas: 20
  targetCPUUtilizationPercentage: 70
  targetMemoryUtilizationPercentage: 80

resources:
  requests:
    memory: "32Gi"
    cpu: "8"
    nvidia.com/gpu: 1
  limits:
    memory: "64Gi"
    cpu: "16"
    nvidia.com/gpu: 1

config:
  NEXUS_ENV: production
  NEXUS_WORKERS: "8"

ingress:
  annotations:
    nginx.ingress.kubernetes.io/proxy-body-size: "500m"
    nginx.ingress.kubernetes.io/proxy-read-timeout: "300"
    nginx.ingress.kubernetes.io/proxy-send-timeout: "300"
    cert-manager.io/cluster-issuer: letsencrypt-prod

postgresql:
  primary:
    replicaCount: 1
    resources:
      requests:
        memory: "8Gi"
        cpu: "4"

redis:
  master:
    resources:
      requests:
        memory: "4Gi"
        cpu: "2"

monitoring:
  enabled: true
  prometheus:
    enabled: true
    retention: 30d
  grafana:
    enabled: true
    adminPassword: "secure-grafana-password"
```

### Helm Deployment Commands

```bash
#!/bin/bash
# deploy-helm.sh - Helm deployment script

# Add Helm repositories
helm repo add nexus https://nexus-ai.github.io/helm-charts
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo add grafana https://grafana.github.io/helm-charts
helm repo update

# Create namespace
kubectl create namespace nexus

# Create secrets
kubectl create secret generic nexus-secrets \
    --from-literal=secret-key="your-production-secret-key" \
    --from-literal=database-url="postgresql://user:pass@host:5432/nexus" \
    --from-literal=postgres-password="secure-password" \
    --from-literal=grafana-password="grafana-password" \
    -n nexus

# Install dependencies
helm dependency update ./charts/nexus

# Deploy with Helm
helm upgrade --install nexus ./charts/nexus \
    --namespace nexus \
    --values values-production.yaml \
    --wait \
    --timeout 10m \
    --debug

# Verify deployment
kubectl get pods -n nexus
kubectl get svc -n nexus

# Check logs
kubectl logs -l app.kubernetes.io/name=nexus -n nexus --tail=100

# Scale deployment
kubectl scale deployment nexus -n nexus --replicas=5

# View status
helm status nexus -n nexus

# Uninstall
helm uninstall nexus -n nexus
```

## Monitoring Setup

### Prometheus Configuration

```yaml
# monitoring/prometheus/prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s
  external_labels:
    cluster: 'nexus-production'
    env: 'production'

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093

rule_files:
  - /etc/prometheus/rules/*.yml

scrape_configs:
  # Nexus API metrics
  - job_name: nexus-api
    metrics_path: /metrics
    static_configs:
      - targets:
          - nexus-api:9090
    relabel_configs:
      - source_labels: [__address__]
        target_label: instance
        regex: '([^:]+):\\d+'
        replacement: '${1}'

  # PostgreSQL metrics
  - job_name: postgres
    static_configs:
      - targets:
          - postgres-exporter:9187

  # Redis metrics
  - job_name: redis
    static_configs:
      - targets:
          - redis-exporter:9121

  # Node metrics
  - job_name: node
    static_configs:
      - targets:
          - node-exporter:9100

  # NVIDIA GPU metrics
  - job_name: nvidia-gpu
    static_configs:
      - targets:
          - nvidia-dcgm-exporter:9400
```

### Prometheus Alert Rules

```yaml
# monitoring/prometheus/rules/nexus-alerts.yml
groups:
  - name: nexus.rules
    rules:
      - alert: NexusAPIHighErrorRate
        expr: rate(nexus_api_requests_total{status=~"5.."}[5m]) / rate(nexus_api_requests_total[5m]) > 0.05
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Nexus API high error rate"
          description: "Error rate is {{ $value | humanizePercentage }} for the last 5 minutes"
          
      - alert: NexusAPILatencyHigh
        expr: histogram_quantile(0.95, rate(nexus_api_request_duration_seconds_bucket[5m])) > 5
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Nexus API high latency"
          description: "P95 latency is {{ $value | humanizeDuration }}"
          
      - alert: NexusGPUUtilizationHigh
        expr: rate(nvidia_gpu_utilization[5m]) > 0.9
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "High GPU utilization"
          description: "GPU utilization is {{ $value | humanizePercentage }}"
          
      - alert: NexusGPUMemoryHigh
        expr: nvidia_gpu_memory_used / nvidia_gpu_memory_total > 0.9
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "High GPU memory usage"
          description: "GPU memory usage is {{ $value | humanizePercentage }}"
          
      - alert: NexusRateLimitExceeded
        expr: rate(nexus_rate_limit_exceeded_total[5m]) > 10
        for: 5m
        labels:
          severity: info
        annotations:
          summary: "Rate limit exceeded"
          description: "{{ $value | humanizeRate }} rate limit violations"
          
      - alert: NexusInferenceFailed
        expr: rate(nexus_inference_total{status="failed"}[5m]) > 0.01
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Inference failures"
          description: "Inference failure rate is {{ $value | humanizePercentage }}"
```

### Grafana Dashboard Configuration

```yaml
# monitoring/grafana/provisioning/dashboards/nexus-dashboard.yml
apiVersion: 1
providers:
  - name: 'Nexus'
    orgId: 1
    folder: 'Nexus'
    type: file
    disableDeletion: false
    editable: true
    allowUiUpdates: true
    options:
      path: /etc/grafana/provisioning/dashboards
---
# monitoring/grafana/provisioning/datasources/prometheus.yml
apiVersion: 1
datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
    editable: false
```

### Grafana Dashboard JSON

```json
{
  "dashboard": {
    "title": "Nexus Overview",
    "panels": [
      {
        "title": "API Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "sum(rate(nexus_api_requests_total[5m])) by (endpoint)",
            "legendFormat": "{{endpoint}}"
          }
        ],
        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0}
      },
      {
        "title": "API Error Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "sum(rate(nexus_api_requests_total{status=~\"5..\"}[5m])) by (endpoint)",
            "legendFormat": "{{endpoint}}"
          }
        ],
        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0}
      },
      {
        "title": "P95 Latency",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, sum(rate(nexus_api_request_duration_seconds_bucket[5m])) by (le, endpoint))",
            "legendFormat": "{{endpoint}}"
          }
        ],
        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 8}
      },
      {
        "title": "GPU Utilization",
        "type": "graph",
        "targets": [
          {
            "expr": "avg(rate(nvidia_gpu_utilization[5m]))",
            "legendFormat": "Average GPU"
          }
        ],
        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 8}
      },
      {
        "title": "Inference Throughput",
        "type": "graph",
        "targets": [
          {
            "expr": "sum(rate(nexus_inference_total[5m]))",
            "legendFormat": "Inferences/sec"
          }
        ],
        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 16}
      },
      {
        "title": "Token Generation Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "sum(rate(nexus_tokens_generated[5m]))",
            "legendFormat": "Tokens/sec"
          }
        ],
        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 16}
      },
      {
        "title": "Cache Hit Rate",
        "type": "gauge",
        "targets": [
          {
            "expr": "sum(rate(nexus_cache_hits[5m])) / sum(rate(nexus_cache_requests[5m])) * 100",
            "legendFormat": "Cache Hit Rate"
          }
        ],
        "gridPos": {"h": 8, "w": 6, "x": 0, "y": 24}
      },
      {
        "title": "Active Requests",
        "type": "stat",
        "targets": [
          {
            "expr": "sum(nexus_active_requests)",
            "legendFormat": "Active Requests"
          }
        ],
        "gridPos": {"h": 4, "w": 6, "x": 6, "y": 24}
      }
    ]
  }
}
```

### AlertManager Configuration

```yaml
# monitoring/alertmanager/alertmanager.yml
global:
  resolve_timeout: 5m
  smtp_smarthost: 'smtp.example.com:587'
  smtp_from: 'alertmanager@nexus.example.com'
  smtp_auth_username: 'alertmanager@nexus.example.com'
  smtp_auth_password: '${SMTP_PASSWORD}'

route:
  group_by: ['alertname', 'severity']
  group_wait: 30s
  group_interval: 5m
  repeat_interval: 4h
  receiver: 'default-receiver'
  routes:
    - match:
        severity: critical
      receiver: 'critical-receiver'
      continue: true
    - match:
        severity: warning
      receiver: 'warning-receiver'

receivers:
  - name: 'default-receiver'
    email_configs:
      - to: 'team@nexus.example.com'
        send_resolved: true
    slack_configs:
      - channel: '#alerts-nexus'
        send_resolved: true

  - name: 'critical-receiver'
    email_configs:
      - to: 'pager@nexus.example.com'
        send_resolved: true
    slack_configs:
      - channel: '#alerts-critical'
        send_resolved: true
    pagerduty_configs:
      - service_key: '${PAGERDUTY_SERVICE_KEY}'
        severity: critical

  - name: 'warning-receiver'
    email_configs:
      - to: 'team@nexus.example.com'
        send_resolved: true
    slack_configs:
      - channel: '#alerts-nexus'
        send_resolved: true

inhibit_rules:
  - source_match:
      severity: 'critical'
    target_match:
      severity: 'warning'
    equal: ['alertname', 'instance']
```

## Deployment Commands and Verification

### Complete Deployment Command Sequence

```bash
#!/bin/bash
# deploy-complete.sh - Complete deployment script

set -e

echo "======================================"
echo "Nexus Platform Deployment"
echo "======================================"

# Step 1: Pre-flight checks
echo "[1/8] Running pre-flight checks..."
./scripts/preflight-checks.sh

# Step 2: Create namespace and RBAC
echo "[2/8] Creating Kubernetes namespace and RBAC..."
kubectl apply -f deployment/k8s_namespace.yaml
kubectl apply -f deployment/k8s_rbac.yaml

# Step 3: Create secrets and configmaps
echo "[3/8] Creating secrets and configmaps..."
kubectl apply -f deployment/k8s_configmap.yaml
kubectl apply -f deployment/k8s_secrets.yaml

# Step 4: Install dependencies
echo "[4/8] Installing dependencies (PostgreSQL, Redis)..."
helm upgrade --install nexus-postgresql bitnami/postgresql \
    --namespace nexus \
    --values deployment/helm/postgresql-values.yaml \
    --wait

helm upgrade --install nexus-redis bitnami/redis \
    --namespace nexus \
    --values deployment/helm/redis-values.yaml \
    --wait

# Step 5: Install monitoring stack
echo "[5/8] Installing monitoring stack..."
helm upgrade --install nexus-monitoring prometheus-community/kube-prometheus-stack \
    --namespace monitoring \
    --create-namespace \
    --values deployment/helm/monitoring-values.yaml \
    --wait

# Step 6: Deploy Nexus application
echo "[6/8] Deploying Nexus application..."
helm upgrade --install nexus ./deployment/helm/nexus \
    --namespace nexus \
    --values deployment/helm/nexus-production-values.yaml \
    --wait

# Step 7: Configure ingress
echo "[7/8] Configuring ingress..."
kubectl apply -f deployment/k8s_ingress.yaml

# Step 8: Verify deployment
echo "[8/8] Verifying deployment..."
./scripts/verify-deployment.sh

echo "======================================"
echo "Deployment complete!"
echo "======================================"
echo ""
echo "Next steps:"
echo "  1. Configure DNS records"
echo "  2. Verify TLS certificates"
echo "  3. Test API endpoints"
echo "  4. Configure backup jobs"
```

### Verification Commands

```bash
#!/bin/bash
# verify-deployment.sh - Deployment verification script

echo "Verifying Nexus deployment..."

# Check namespace
echo "[1] Checking namespace..."
kubectl get namespace nexus
kubectl get namespace monitoring

# Check pods
echo "[2] Checking pods..."
kubectl get pods -n nexus -o wide
kubectl get pods -n monitoring -o wide

# Check services
echo "[3] Checking services..."
kubectl get svc -n nexus
kubectl get svc -n monitoring

# Check deployments
echo "[4] Checking deployments..."
kubectl get deployment -n nexus
kubectl get deployment -n monitoring

# Check HPA
echo "[5] Checking HPA..."
kubectl get hpa -n nexus

# Check logs
echo "[6] Checking API logs..."
kubectl logs -l app=nexus-api -n nexus --tail=50

# Check metrics
echo "[7] Checking metrics endpoint..."
kubectl port-forward svc/nexus-api 9090:9090 &
sleep 5
curl -s http://localhost:9090/metrics | head -20
kill %1 2>/dev/null || true

# Test API health
echo "[8] Testing API health..."
kubectl port-forward svc/nexus-api 8080:80 &
sleep 5
curl -s http://localhost:8080/health
kill %1 2>/dev/null || true

# Check GPU availability
echo "[9] Checking GPU availability..."
kubectl get nodes -l node-type=gpu
kubectl describe node $(kubectl get nodes -l node-type=gpu -o jsonpath='{.items[0].metadata.name}') | grep -A5 "nvidia.com/gpu"

echo "======================================"
echo "Deployment verification complete!"
echo "======================================"
```

## Scaling and Performance

### Horizontal Scaling

```bash
# Manual scaling
kubectl scale deployment nexus --replicas=10 -n nexus

# Auto-scaling with HPA
kubectl get hpa nexus -n nexus -o yaml

# View scaling events
kubectl describe hpa nexus -n nexus
```

### Vertical Scaling

```bash
# Update resource requests
kubectl set resources deployment nexus \
    --requests=cpu=8,memory=32Gi \
    --limits=cpu=16,memory=64Gi \
    -n nexus
```

### GPU Scaling

```bash
# Scale GPU nodes (cloud provider specific)
# AWS
aws eks update-nodegroup-config \
    --cluster-name nexus-cluster \
    --nodegroup-name gpu-nodes \
    --scaling desiredSize=3,minSize=1,maxSize=10

# Scale NVIDIA GPU operator
kubectl scale nvidiaGPUOperator -n nvidia-gpu-operator --replicas=3
```

## Backup and Recovery

### Database Backup

```bash
#!/bin/bash
# backup.sh - Database backup script

BACKUP_DIR="/backups/nexus"
DATE=$(date +%Y%m%d_%H%M%S)

# Create backup directory
mkdir -p ${BACKUP_DIR}

# Backup PostgreSQL
echo "Backing up PostgreSQL..."
PGPASSWORD=${POSTGRES_PASSWORD} pg_dump \
    -h nexus-postgresql-0.nexus-postgresql \
    -U nexus \
    -d nexus \
    -Fc \
    -f ${BACKUP_DIR}/nexus_backup_${DATE}.dump

# Upload to S3
aws s3 cp ${BACKUP_DIR}/nexus_backup_${DATE}.dump \
    s3://nexus-backups/postgres/

# Cleanup old backups (keep last 30 days)
find ${BACKUP_DIR} -name "nexus_backup_*.dump" -mtime +30 -delete

echo "Backup complete: nexus_backup_${DATE}.dump"
```

### Restore from Backup

```bash
#!/bin/bash
# restore.sh - Database restore script

BACKUP_FILE=$1
BACKUP_DIR="/backups/nexus"

if [ -z "$BACKUP_FILE" ]; then
    echo "Usage: $0 <backup_file>"
    exit 1
fi

echo "Restoring from backup: $BACKUP_FILE"

# Stop Nexus
kubectl scale deployment nexus --replicas=0 -n nexus

# Restore PostgreSQL
PGPASSWORD=${POSTGRES_PASSWORD} pg_restore \
    -h nexus-postgresql-0.nexus-postgresql \
    -U nexus \
    -d nexus \
    -c \
    ${BACKUP_DIR}/${BACKUP_FILE}

# Start Nexus
kubectl scale deployment nexus --replicas=3 -n nexus

echo "Restore complete"
```

## See Also

- **[Architecture Overview](ARCHITECTURE.md)** - System architecture details
- **[API Reference](API_REFERENCE.md)** - Detailed API documentation
- **[Pipeline Guide](PIPELINE_GUIDE.md)** - Pipeline configuration
- **[Security Documentation](SECURITY.md)** - Security implementation
- **[Training Methods](TRAINING_METHODS.md)** - Training pipeline details
- **[Troubleshooting Guide](TROUBLESHOOTING.md)** - Common issues and solutions
