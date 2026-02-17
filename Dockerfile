FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency metadata first for better caching
COPY pyproject.toml ./

# Install PDM + uv, then project dependencies
RUN pip install --no-cache-dir pdm uv \
    && pdm install --prod --no-editable --no-self

# Copy application code
COPY . .

# Create non-root user for security
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose ports (8888 for compatibility, 7860 for Hugging Face)
EXPOSE 8888 7860

# Set environment variables
ENV PYTHONPATH=/app
ENV HOST=0.0.0.0
ENV PORT=7860

# Health check (use PORT environment variable)
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Run the application using app.py (Hugging Face compatible entry point)
CMD ["pdm", "run", "python", "app.py"]
