# Use Python 3.12 slim image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY pyproject.toml ./

# Install Python dependencies
RUN pip install --no-cache-dir -e .

# Copy application code
COPY . .

# Create non-root user for security
RUN useradd -m -u 1000 mcpuser && chown -R mcpuser:mcpuser /app
USER mcpuser

# Expose ports for SSE and HTTP transports
EXPOSE 3001 3002

# Set default environment variables
ENV PERPLEXICA_BACKEND_URL=http://localhost:3000/api/search

# Default command (can be overridden)
CMD ["python", "perplexica_mcp_tool.py", "--transport", "all", "--host", "0.0.0.0", "--sse-port", "3001", "--http-port", "3002"]