# ==============================================================================
# Stock Prediction API - Dockerfile
# Updated for new backend/ structure
# ==============================================================================
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Copy project files
COPY pyproject.toml uv.lock ./
COPY backend/ ./backend/

# Install dependencies
RUN uv sync --frozen --no-dev

# Create directories for volumes
RUN mkdir -p data models Result

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/ || exit 1

# Run the application
CMD ["uv", "run", "uvicorn", "backend.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
