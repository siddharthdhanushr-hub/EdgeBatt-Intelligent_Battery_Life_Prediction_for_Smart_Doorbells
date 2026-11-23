# ==============================================================================
# Dockerfile -- Production Container for EDGI AI Team 01 (EdgeBatt API)
# Optimized for GitHub Codespaces & Cloud Container Deployment
# ==============================================================================
FROM python:3.11-slim

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

# Install curl for container health check
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory inside container
WORKDIR /app

# Create non-root user for security best practices
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Copy requirements and install CPU-optimized PyTorch and dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source code, models, static dashboard, and dataset
COPY app/ ./app/
COPY model/ ./model/
COPY static/ ./static/
COPY data/ ./data/
COPY client/ ./client/
COPY *.py ./

# Set ownership of /app to appuser
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose FastAPI service port
EXPOSE 8000

# Container Healthcheck (pings /health endpoint)
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Launch production Uvicorn server with 1 worker
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
