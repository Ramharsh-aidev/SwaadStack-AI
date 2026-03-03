# ==============================================================================
# SwaadStack AI - Modular Production Dockerfile
# ==============================================================================
# Multi-stage build for optimized image size
# Base: Python 3.9-slim for minimal footprint
# Uses CPU-only PyTorch to reduce image size (~2GB vs ~6GB)
# ==============================================================================

# ── Stage 1: Builder ──
FROM python:3.9-slim as builder

WORKDIR /build

# Install system dependencies for building
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .

# Install PyTorch CPU-only first (saves ~4GB)
RUN pip install --no-cache-dir \
    torch==2.1.0+cpu \
    torchvision==0.16.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
RUN pip install --no-cache-dir -r requirements.txt

# ── Stage 2: Runtime ──
FROM python:3.9-slim as runtime

WORKDIR /app

# Install runtime system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && adduser --disabled-password --gecos '' appuser

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY swaadstack/ ./swaadstack/
COPY scripts/ ./scripts/

# Create directories including artifacts and logs
RUN mkdir -p swaadstack/artifacts/data swaadstack/artifacts/models swaadstack/logs && chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    API_HOST=0.0.0.0 \
    API_PORT=8000 \
    PYTHONPATH=/app

# Run with uvicorn
CMD ["uvicorn", "swaadstack.api.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
