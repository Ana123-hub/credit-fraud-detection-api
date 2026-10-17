# Stage 1: Build the application wheel and dependency wheels.
FROM python:3.14.5-slim AS builder

WORKDIR /build

# Install system dependencies needed for building wheels
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-runtime.txt ./

RUN mkdir -p /wheels \
    && python -m pip wheel \
        --no-cache-dir \
        --wheel-dir=/wheels \
        -r requirements-runtime.txt

# Copy package dependency definitions and all inputs declared by the package build configuration.
COPY pyproject.toml README.md ./
COPY src/ ./src/

RUN python -m pip install --no-cache-dir --upgrade setuptools wheel \
    && python -m pip wheel \
        --no-deps \
        --no-cache-dir \
        --wheel-dir=/wheels .

# Stage 2: Runtime-only inference image
FROM python:3.14.5-slim AS runner

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    CREDIT_FRAUD_PROJECT_ROOT=/app

WORKDIR /app

RUN groupadd --system appuser \
    && useradd --system --gid appuser appuser

COPY --from=builder /wheels /wheels
COPY requirements-runtime.txt /tmp/requirements-runtime.txt

# Install the exact locked runtime dependencies without contacting PyPI.
RUN python -m pip install \
        --no-index \
        --find-links=/wheels \
        -r /tmp/requirements-runtime.txt \
    && python -m pip install \
        --no-index \
        --no-deps \
        --find-links=/wheels \
        credit-fraud-detection==1.0.0 \
    && rm -rf /wheels /tmp/requirements-runtime.txt

# Only the configuration and model bundle are required for inference.
COPY config.yaml ./config.yaml
COPY models/ ./models/

# Ensure non-root user owns the app directory
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose FastAPI port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)" || exit 1

# Launch production Uvicorn server with multiple workers
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]