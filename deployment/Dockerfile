# =============================================================================
# Nexus AI Platform - Multi-Stage Dockerfile
# =============================================================================
# Stages:
#   base       - Python + system deps (shared)
#   builder    - Build the nexus wheel + install deps
#   production - Minimal runtime image for deployment
#   dev        - Development image with extra tools
# =============================================================================

# ---------------------------------------------------------------------------
# Stage 1: Base image with system dependencies
# ---------------------------------------------------------------------------
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 \
        python3.11-venv \
        python3.11-dev \
        python3-pip \
        libsndfile1 \
        ffmpeg \
        curl \
        && rm -rf /var/lib/apt/lists/*

# Make python3.11 the default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

RUN python -m pip install --upgrade pip setuptools wheel

# ---------------------------------------------------------------------------
# Stage 2: Builder - install deps and build the package
# ---------------------------------------------------------------------------
FROM base AS builder

WORKDIR /build

# Install dependencies first (better layer caching)
COPY requirements.txt .
RUN pip install --prefix=/install -r requirements.txt

# Copy source and build the nexus package
COPY pyproject.toml setup.py ./
COPY src/ ./src/

RUN pip install --prefix=/install --no-deps .

# ---------------------------------------------------------------------------
# Stage 3: Production image (default target)
# ---------------------------------------------------------------------------
FROM base AS production

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy config files if needed at runtime
COPY src/nexus/config/ /app/config/

# Create non-root user for security
RUN groupadd -r nexus && useradd -r -g nexus -d /app -s /sbin/nologin nexus && \
    mkdir -p /data/models /data/cache /data/logs && \
    chown -R nexus:nexus /app /data

USER nexus

# Health check endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health/live || exit 1

EXPOSE 8000

# Default: run the Nexus API server via uvicorn
CMD ["python", "-m", "uvicorn", "nexus.api:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "4", \
     "--log-level", "info"]

# ---------------------------------------------------------------------------
# Stage 4: Development image (optional target)
# ---------------------------------------------------------------------------
FROM production AS dev

USER root

RUN pip install \
    pytest pytest-cov \
    black isort flake8 mypy \
    ipython ipdb \
    pre-commit

USER nexus

CMD ["python", "-m", "uvicorn", "nexus.api:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--reload", \
     "--log-level", "debug"]
