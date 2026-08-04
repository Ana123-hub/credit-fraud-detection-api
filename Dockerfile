# Stage 1: Build Dependencies
FROM python:3.11-slim AS builder

WORKDIR /app

# Install system dependencies needed for building wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy package dependency definitions
COPY pyproject.toml .

# Install dependencies into a wheels directory
RUN pip install --no-cache-dir --upgrade pip wheel && \
    pip wheel --no-cache-dir --no-deps --wheel-dir /app/wheels .

# Stage 2: Runtime Environment
FROM python:3.11-slim AS runner

# Create a non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser

WORKDIR /app

# Copy compiled wheels from builder stage and install
COPY --from=builder /app/wheels /wheels
RUN pip install --no-cache-dir /wheels/* && rm -rf /wheels

# Copy application source code, config, and trained artifacts
COPY config.yaml .
COPY src/ ./src/
COPY data/processed/ ./data/processed/
COPY models/ ./models/

# Ensure non-root user owns the app directory
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose FastAPI port
EXPOSE 8000

# Health check using python urllib (no curl dependency required)
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Launch production Uvicorn server with multiple workers
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]