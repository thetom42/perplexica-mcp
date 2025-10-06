# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Architecture Overview

Perplexica MCP Server is a Model Context Protocol (MCP) implementation that bridges AI-powered search capabilities from Perplexica with MCP clients. The codebase has a unified architecture supporting multiple transport modes (stdio, SSE, HTTP) through the FastMCP framework.

### Core Components

- **Single Server Implementation**: `src/perplexica_mcp/server.py` contains the complete MCP server with all transport modes
- **FastMCP Framework**: Uses FastMCP for robust MCP protocol compliance and built-in transport support
- **Unified Transport Design**: All three transport modes (stdio, SSE, Streamable HTTP) are handled by the same server code
- **Environment-driven Configuration**: Uses `.env` files and environment variables for flexible deployment

### Project Structure

```
src/perplexica_mcp/
├── server.py          # Main MCP server with all transport modes
├── __init__.py        # Package exports and entry point
└── __main__.py        # CLI entry point
src/test_transports.py # Comprehensive transport testing suite
config/                # Configuration examples and templates
docs/                  # Documentation and architecture guides
```

## Development Commands

### Environment Setup
```bash
# Install dependencies using uv
uv sync

# Create .env from template
cp .sample.env .env
# Edit .env with your Perplexica backend URL and model configurations
```

### Running the Server

```bash
# Stdio transport (for MCP clients like Claude Desktop)
uv run src/perplexica_mcp/server.py stdio

# SSE transport (default: localhost:3001)
uv run src/perplexica_mcp/server.py sse [host] [port]

# HTTP transport (default: localhost:3002)
uv run src/perplexica_mcp/server.py http [host] [port]

# Using PyPI installation
uvx perplexica-mcp stdio|sse|http
```

### Testing

```bash
# Run comprehensive transport tests
uv run src/test_transports.py

# Test individual components with pytest (when dev dependencies installed)
uv run pytest
```

### Docker Development

```bash
# Create external network (required for integration)
docker network create backend

# HTTP transport
docker-compose up -d

# SSE transport
docker-compose -f docker-compose-sse.yml up -d

# Build and run manually
docker build -t perplexica-mcp .
docker run -p 3001:3001 -e PERPLEXICA_BACKEND_URL=http://host:3000/api/search perplexica-mcp
```

## Key Implementation Details

### Environment Variables

The server supports both required and optional environment variables:

- `PERPLEXICA_BACKEND_URL`: Backend Perplexica API URL (required)
- `PERPLEXICA_CHAT_MODEL_PROVIDER/NAME`: Default chat model (optional)
- `PERPLEXICA_EMBEDDING_MODEL_PROVIDER/NAME`: Default embedding model (optional)
- `PERPLEXICA_READ_TIMEOUT`: API timeout in seconds (default: 60)

### Transport Modes

1. **Stdio**: Standard MCP protocol over stdin/stdout - used by most MCP clients
2. **SSE**: Server-Sent Events for real-time streaming - includes ping/pong for connection health
3. **HTTP**: Streamable HTTP with 307 redirects - REST API compatible

### MCP Tool Definition

The `search` tool is the primary interface, supporting:
- Multiple focus modes (webSearch, academicSearch, writingAssistant, etc.)
- Optional model overrides per request
- Conversation history and system instructions
- Streaming and optimization modes

### Error Handling

The codebase implements fail-fast validation:
- Checks for required models at request time
- Provides detailed error messages for missing configurations
- Uses httpx for robust HTTP client operations with proper timeout handling

## Configuration Examples

### Claude Desktop (stdio)
```json
{
  "mcpServers": {
    "perplexica": {
      "command": "uvx",
      "args": ["perplexica-mcp", "stdio"],
      "env": {
        "PERPLEXICA_BACKEND_URL": "http://localhost:3000/api/search",
        "PERPLEXICA_CHAT_MODEL_PROVIDER": "openai",
        "PERPLEXICA_CHAT_MODEL_NAME": "gpt-4o-mini"
      }
    }
  }
}
```


## Testing Strategy

The `test_transports.py` provides comprehensive testing:

- **Stdio**: Full MCP handshake (initialize → initialized → tools/list)
- **HTTP**: Streamable HTTP compliance with 307 redirects
- **SSE**: Endpoint accessibility and streaming capability
- **Background servers**: Automatic server startup/cleanup for integration tests
- **Timeout handling**: Robust timeout and retry logic for all transport modes

## Dependencies and Framework Choices

- **FastMCP**: Chosen for built-in transport support and MCP protocol compliance
- **httpx**: Async HTTP client with excellent timeout and error handling
- **uvicorn**: ASGI server for SSE and HTTP transports
- **pydantic**: Type validation and schema definition
- **python-dotenv**: Environment variable management

The unified architecture allows a single codebase to serve all transport modes while maintaining protocol compliance and production readiness.
