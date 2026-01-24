# =============================================================================
# Stage 1: Builder - Install dependencies
# =============================================================================
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency resolution
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Create virtual environment and install dependencies
# Install base dependencies + docker extras (redis, celery, gunicorn, eventlet)
RUN uv venv /opt/venv && \
    uv pip install --python /opt/venv/bin/python \
    -r pyproject.toml && \
    uv pip install --python /opt/venv/bin/python \
    redis>=5.0 \
    "celery[redis]>=5.3" \
    gunicorn>=21.0 \
    eventlet>=0.35 \
    watchdog>=4.0

# =============================================================================
# Stage 2: Runtime
# =============================================================================
FROM python:3.11-slim AS runtime

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash gmagent

WORKDIR /app

# Install runtime dependencies (curl for healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Set PATH to use the virtual environment
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code
COPY --chown=gmagent:gmagent . /app/

# Create data directories
RUN mkdir -p /data/campaigns /data/pf2e-rag && \
    chown -R gmagent:gmagent /data

# Switch to non-root user
USER gmagent

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    CAMPAIGNS_DIR=/data/campaigns \
    RAG_DB_PATH=/data/pf2e-rag/pathfinder_search.db \
    PF2E_RAG_PATH=/data/pf2e-rag

# Health check for the API
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/api/health || exit 1

EXPOSE 5000

# Default command - single eventlet worker (required for WebSocket state)
CMD ["gunicorn", "--worker-class", "eventlet", "-w", "1", "-b", "0.0.0.0:5000", "--timeout", "120", "wsgi:app"]
