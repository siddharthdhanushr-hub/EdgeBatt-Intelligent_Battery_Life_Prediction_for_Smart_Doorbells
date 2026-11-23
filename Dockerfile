# ==============================================================================
# Dockerfile -- Production Container for EdgeBatt Battery Prediction API
# Containerization Specification
# ==============================================================================
FROM python:3.11-slim

# Prevent Python from writing pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

# Install curl for container healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set /app as working directory
WORKDIR /app

# Create non-root appuser (UID 10001)
RUN groupadd -g 10001 appuser && \
    useradd -u 10001 -g appuser -s /bin/bash -m appuser

# Copy runtime-only dependencies and install
COPY requirements-runtime.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements-runtime.txt

# Copy entrypoint, app package, model artifacts, and static UI
COPY app.py .
COPY app/ ./app/
COPY model/ ./model/
COPY static/ ./static/

# Set ownership to appuser
RUN chown -R appuser:appuser /app

# Run as non-root appuser
USER appuser

# Expose port 8000
EXPOSE 8000

# Define Docker /health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://127.0.0.1:8000/health || exit 1

# Starts with python app.py
CMD ["python", "app.py"]
