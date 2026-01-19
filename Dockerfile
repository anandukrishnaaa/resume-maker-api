# Use Python 3.13 slim image
FROM python:3.13-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    wget \
    xz-utils \
    && rm -rf /var/lib/apt/lists/*

# Install Typst (required for rendercv PDF generation)
RUN wget -q https://github.com/typst/typst/releases/download/v0.12.0/typst-x86_64-unknown-linux-musl.tar.xz \
    && tar -xf typst-x86_64-unknown-linux-musl.tar.xz \
    && cp typst-x86_64-unknown-linux-musl/typst /usr/local/bin/ \
    && chmod +x /usr/local/bin/typst \
    && rm -rf typst-x86_64-unknown-linux-musl* \
    && typst --version

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_SYSTEM_PYTHON=1

# Copy dependency files
COPY pyproject.toml ./

# Install dependencies using uv
RUN uv pip install --system --no-cache -r pyproject.toml

# Copy application code
COPY *.py ./

# Create necessary directories
RUN mkdir -p /app/tmp /app/output /app/data

# Expose FastAPI port
EXPOSE 8000

# Default command
CMD ["uvicorn", "resume_generator:app", "--host", "0.0.0.0", "--port", "8000"]