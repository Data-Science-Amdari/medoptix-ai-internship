FROM python:3.11-slim AS builder

LABEL maintainer="Richard Gidi"
LABEL created="2025-07-30"

# Set the working directory
WORKDIR /app    

ENV PYTHONUNBUFFERED=1\
    PYTHONDONTWRITEBYTECODE=1\
    PIP_NO_CACHE_DIR=1

# Install system dependencies needed for building
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file
COPY requirements.txt . 

# Install dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Runtime stage
FROM python:3.11-slim

LABEL maintainer="Richard Gidi"
LABEL created="2025-07-30"

# Set the working directory
WORKDIR /app

ENV PYTHONUNBUFFERED=1\
    PYTHONDONTWRITEBYTECODE=1\
    PIP_NO_CACHE_DIR=1

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy the installed packages from the builder stage
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash --uid 1000 medoptix

# Copy application files (be specific to avoid copying unnecessary files)
COPY ./app ./app
COPY ./models ./models

# Set ownership to non-root user
RUN chown -R medoptix:medoptix /app

# Switch to non-root user
USER medoptix

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Command to run the application
# Use fewer workers for development, increase for production
CMD ["python", "-m", "gunicorn", "-k", "uvicorn.workers.UvicornWorker", "-w", "2", "app.main:app", "--bind", "0.0.0.0:8000", "--timeout", "120"]