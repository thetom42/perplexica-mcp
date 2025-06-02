# Use Python 3.13 Alpine image for better security and smaller size
FROM python:3.13-alpine

# Set working directory
WORKDIR /app

# Install system dependencies and uv
RUN apk add --no-cache \
    gcc \
    musl-dev \
    && pip install --no-cache-dir uv

# Copy requirements first for better caching
COPY pyproject.toml ./

# Install Python dependencies using uv
RUN uv pip install --system --no-cache -e .

# Copy application code
COPY src/ ./src

# Create non-root user for security (Alpine compatible)
RUN adduser -D -u 1000 mcpuser && chown -R mcpuser:mcpuser /app
USER mcpuser

# Expose ports for SSE and HTTP transports
EXPOSE 3001 3002

# Set default environment variables
ENV PERPLEXICA_BACKEND_URL=http://localhost:3000/api/search

# Default command (can be overridden)
CMD ["python", "src/perplexica_mcp_tool.py", "--transport", "all", "--host", "0.0.0.0", "--sse-port", "3001", "--http-port", "3002"]
