# UltraTrader - Institutional-Grade Trading System
# Multi-stage Docker build for production deployment

# Stage 1: Base image with Python and system dependencies
FROM python:3.10-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Stage 2: Builder image
FROM base as builder

WORKDIR /build

# Copy requirements and install Python dependencies
COPY requirements_ultra.txt .
RUN pip install --user -r requirements_ultra.txt

# Stage 3: Runtime image
FROM base as runtime

# Create non-root user for security
RUN useradd -m -u 1000 trader && \
    mkdir -p /app /app/logs /app/checkpoints /app/reports /app/data && \
    chown -R trader:trader /app

# Switch to non-root user
USER trader
WORKDIR /app

# Copy Python packages from builder
COPY --from=builder --chown=trader:trader /root/.local /home/trader/.local

# Add Python packages to PATH
ENV PATH=/home/trader/.local/bin:$PATH

# Copy application code
COPY --chown=trader:trader . .

# Expose ports (if serving model)
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.exit(0)"

# Default command
CMD ["python", "train_ultra.py"]
