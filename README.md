# Perplexica MCP Server

A Model Context Protocol (MCP) server that provides search functionality using Perplexica's AI-powered search engine.

## Features

- **Search Tool**: AI-powered web search with multiple focus modes
- **Multiple Transport Support**: stdio, SSE, and Streamable HTTP transports
- **FastMCP Integration**: Built using FastMCP for robust MCP protocol compliance
- **Unified Architecture**: Single server implementation supporting all transport modes
- **Production Ready**: Docker support with security best practices

## Development Environment

### For Claude Code Users

**Important**: If you are using Claude Code for development, this project requires the use of the `container-use` MCP server for all development operations. All file operations, code changes, and shell commands must be executed within container-use environments.

#### Working with Container-Use (Claude Code Only)

When contributing to this project using Claude Code, you must:

1. **Use Container-Use Only**: All file operations, code editing, and shell commands must be performed using container-use environments
2. **View Your Work**: After making changes, inform others how to access your work:
   - Use `container-use log <env_id>` to view the development log
   - Use `container-use checkout <env_id>` to check out your environment
3. **No Local Operations**: Do not perform file operations directly on the local filesystem

#### Example Development Workflow (Claude Code)

```bash
# Create a new environment for your work
container-use create --title "Your feature description"

# Make your changes using container-use tools
# (All file operations handled by container-use)

# Share your work with others
container-use log <your-env-id>
container-use checkout <your-env-id>
```

This ensures consistency, reproducibility, and proper version control for all development activities when using Claude Code.

### For Other Development Environments

If you are not using Claude Code, you can develop normally using your preferred tools and IDE. The container-use requirement does not apply to regular development workflows.

## Installation

### From PyPI (Recommended)

```bash
# Install directly from PyPI
pip install perplexica-mcp

# Or using uvx for isolated execution
uvx perplexica-mcp --help
```

### From Source

```bash
# Clone the repository
git clone https://github.com/thetom42/perplexica-mcp.git
cd perplexica-mcp

# Install dependencies
uv sync
```

## MCP Client Configuration

To use this server with MCP clients, you need to configure the client to connect to the Perplexica MCP server. Below are configuration examples for popular MCP clients.

> **Important**: All transport modes require proper environment variable configuration, especially:
> - `PERPLEXICA_BACKEND_URL`: URL to your Perplexica backend API
> - `PERPLEXICA_CHAT_MODEL_PROVIDER` and `PERPLEXICA_CHAT_MODEL_NAME`: Chat model configuration
> - `PERPLEXICA_EMBEDDING_MODEL_PROVIDER` and `PERPLEXICA_EMBEDDING_MODEL_NAME`: Embedding model configuration
> 
> These variables must be set either in your environment or provided in the MCP client configuration.

### Claude Desktop

#### Stdio Transport (Recommended)

Add the following to your Claude Desktop configuration file:

**Location**: `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS) or `%APPDATA%\Claude\claude_desktop_config.json` (Windows)

```json
{
  "mcpServers": {
    "perplexica": {
      "command": "uvx",
      "args": ["perplexica-mcp", "stdio"],
      "env": {
        "PERPLEXICA_BACKEND_URL": "http://localhost:3000/api/search",
        "PERPLEXICA_CHAT_MODEL_PROVIDER": "openai",
        "PERPLEXICA_CHAT_MODEL_NAME": "gpt-4o-mini",
        "PERPLEXICA_EMBEDDING_MODEL_PROVIDER": "openai",
        "PERPLEXICA_EMBEDDING_MODEL_NAME": "text-embedding-3-small"
      }
    }
  }
}
```

**Alternative (from source):**

```json
{
  "mcpServers": {
    "perplexica": {
      "command": "uv",
      "args": ["run", "python", "-m", "perplexica_mcp", "stdio"],
      "cwd": "/path/to/perplexica-mcp",
      "env": {
        "PERPLEXICA_BACKEND_URL": "http://localhost:3000/api/search",
        "PERPLEXICA_CHAT_MODEL_PROVIDER": "openai",
        "PERPLEXICA_CHAT_MODEL_NAME": "gpt-4o-mini",
        "PERPLEXICA_EMBEDDING_MODEL_PROVIDER": "openai",
        "PERPLEXICA_EMBEDDING_MODEL_NAME": "text-embedding-3-small"
      }
    }
  }
}
```

> **Note**: When running from source, ensure all required environment variables are set. The stdio transport requires proper model provider and model name configuration to communicate with the Perplexica backend.
```

#### SSE Transport

For SSE transport, first start the server:

```bash
uv run src/perplexica_mcp/server.py sse
```

Then configure Claude Desktop:

```json
{
  "mcpServers": {
    "perplexica": {
      "url": "http://localhost:3001/sse"
    }
  }
}
```

### Cursor IDE

Add to your Cursor MCP configuration:

```json
{
  "servers": {
    "perplexica": {
      "command": "uvx",
      "args": ["perplexica-mcp", "stdio"],
      "env": {
        "PERPLEXICA_BACKEND_URL": "http://localhost:3000/api/search",
        "PERPLEXICA_CHAT_MODEL_PROVIDER": "openai",
        "PERPLEXICA_CHAT_MODEL_NAME": "gpt-4o-mini",
        "PERPLEXICA_EMBEDDING_MODEL_PROVIDER": "openai",
        "PERPLEXICA_EMBEDDING_MODEL_NAME": "text-embedding-3-small"
      }
    }
  }
}
```

**Alternative (from source):**

```json
{
  "servers": {
    "perplexica": {
      "command": "uv",
      "args": ["run", "python", "-m", "perplexica_mcp", "stdio"],
      "cwd": "/path/to/perplexica-mcp",
      "env": {
        "PERPLEXICA_BACKEND_URL": "http://localhost:3000/api/search",
        "PERPLEXICA_CHAT_MODEL_PROVIDER": "openai",
        "PERPLEXICA_CHAT_MODEL_NAME": "gpt-4o-mini",
        "PERPLEXICA_EMBEDDING_MODEL_PROVIDER": "openai",
        "PERPLEXICA_EMBEDDING_MODEL_NAME": "text-embedding-3-small"
      }
    }
  }
}
```

### VS Code (with MCP Extension)

Add to your VS Code MCP configuration file (`.vscode/mcp.json`):

```json
{
  "servers": {
    "perplexica": {
      "type": "stdio",
      "command": "uv",
      "args": ["run", "python", "-m", "perplexica_mcp", "stdio"],
      "cwd": "/path/to/perplexica-mcp",
      "env": {
        "PERPLEXICA_BACKEND_URL": "http://localhost:3000/api/search",
        "PERPLEXICA_CHAT_MODEL_PROVIDER": "openai",
        "PERPLEXICA_CHAT_MODEL_NAME": "gpt-4o-mini",
        "PERPLEXICA_EMBEDDING_MODEL_PROVIDER": "openai",
        "PERPLEXICA_EMBEDDING_MODEL_NAME": "text-embedding-3-small"
      }
    }
  }
}
```

### Generic MCP Client Configuration

For any MCP client supporting stdio transport:

```bash
# Command to run the server (PyPI installation)
uvx perplexica-mcp stdio

# Command to run the server with .env file (PyPI installation)
uvx --env-file .env perplexica-mcp stdio

# Command to run the server (from source)
uv run python -m perplexica_mcp stdio

# Environment variables (can be exported or set inline)
export PERPLEXICA_BACKEND_URL=http://localhost:3000/api/search
export PERPLEXICA_CHAT_MODEL_PROVIDER=openai
export PERPLEXICA_CHAT_MODEL_NAME=gpt-4o-mini
export PERPLEXICA_EMBEDDING_MODEL_PROVIDER=openai
export PERPLEXICA_EMBEDDING_MODEL_NAME=text-embedding-3-small

# Or set inline for single execution (all required vars)
PERPLEXICA_BACKEND_URL=http://localhost:3000/api/search \
PERPLEXICA_CHAT_MODEL_PROVIDER=openai \
PERPLEXICA_CHAT_MODEL_NAME=gpt-4o-mini \
PERPLEXICA_EMBEDDING_MODEL_PROVIDER=openai \
PERPLEXICA_EMBEDDING_MODEL_NAME=text-embedding-3-small \
uvx perplexica-mcp stdio
```

For HTTP/SSE transport clients:

```bash
# Start the server (PyPI installation)
uvx perplexica-mcp sse  # or 'http'

# Start the server (from source)
uv run /path/to/perplexica-mcp/src/perplexica_mcp/server.py sse  # or 'http'

# Connect to endpoints
SSE: http://localhost:3001/sse
HTTP: http://localhost:3002/mcp/
```

### Configuration Notes

1. **Path Configuration**: Replace `/path/to/perplexica-mcp/` with the actual path to your installation
2. **Perplexica URL**: Ensure `PERPLEXICA_BACKEND_URL` points to your running Perplexica instance
3. **Transport Selection**:
   - Use **stdio** for most MCP clients (Claude Desktop, Cursor)
   - Use **SSE** for web-based clients or real-time applications
   - Use **HTTP** for REST API integrations
4. **Dependencies**: Ensure `uvx` is installed and available in your PATH (or `uv` for source installations)

### Troubleshooting

- **Server not starting**: Check that `uvx` (or `uv` for source) is installed and the path is correct
- **Connection refused**: Verify Perplexica is running and accessible at the configured URL
- **Permission errors**: Ensure the MCP client has permission to execute the server command
- **Environment variables**: Check that `PERPLEXICA_BACKEND_URL` is properly set

## Server Configuration

Create a `.env` file in the project root with your Perplexica configuration:

```env
# Perplexica Backend Configuration
PERPLEXICA_BACKEND_URL=http://localhost:3000/api/search

# Default Model Configuration (Optional)
# If set, these models will be used as defaults when no model is specified in the search request

# Chat Model Configuration
PERPLEXICA_CHAT_MODEL_PROVIDER=openai
PERPLEXICA_CHAT_MODEL_NAME=gpt-4o-mini

# Embedding Model Configuration  
PERPLEXICA_EMBEDDING_MODEL_PROVIDER=openai
PERPLEXICA_EMBEDDING_MODEL_NAME=text-embedding-3-small
```

### Environment Variables

| Variable | Description | Default | Example |
|----------|-------------|---------|---------|
| `PERPLEXICA_BACKEND_URL` | URL to Perplexica search API | `http://localhost:3000/api/search` | `http://localhost:3000/api/search` |
| `PERPLEXICA_CHAT_MODEL_PROVIDER` | Default chat model provider | None | `openai`, `ollama`, `anthropic` |
| `PERPLEXICA_CHAT_MODEL_NAME` | Default chat model name | None | `gpt-4o-mini`, `claude-3-sonnet` |
| `PERPLEXICA_EMBEDDING_MODEL_PROVIDER` | Default embedding model provider | None | `openai`, `ollama` |
| `PERPLEXICA_EMBEDDING_MODEL_NAME` | Default embedding model name | None | `text-embedding-3-small` |

**Note**: The model environment variables are optional. If not set, you'll need to specify models in each search request. When set, they provide convenient defaults that can still be overridden per request.

## Usage

The server supports three transport modes:

### 1. Stdio Transport

```bash
# PyPI installation
uvx perplexica-mcp stdio

# From source
uv run src/perplexica_mcp/server.py stdio
```

### 2. SSE Transport

```bash
# PyPI installation
uvx perplexica-mcp sse [host] [port]

# From source
uv run src/perplexica_mcp/server.py sse [host] [port]
# Default: localhost:3001, endpoint: /sse
```

### 3. Streamable HTTP Transport

```bash
# PyPI installation
uvx perplexica-mcp http [host] [port]

# From source
uv run src/perplexica_mcp/server.py http [host] [port]
# Default: localhost:3002, endpoint: /mcp
```

## Docker Deployment

The server includes Docker support with multiple transport configurations for containerized deployments.

### Prerequisites

- Docker and Docker Compose installed
- External Docker network named `backend` (for integration with Perplexica)

### Create External Network

```bash
docker network create backend
```

### Build and Run

#### Option 1: HTTP Transport (Streamable HTTP)

```bash
# Build and run with HTTP transport
docker-compose up -d

# Or build first, then run
docker-compose build
docker-compose up -d
```

#### Option 2: SSE Transport (Server-Sent Events)

```bash
# Build and run with SSE transport
docker-compose -f docker-compose-sse.yml up -d

# Or build first, then run
docker-compose -f docker-compose-sse.yml build
docker-compose -f docker-compose-sse.yml up -d
```

### Environment Configuration

Both Docker configurations support environment variables:

```bash
# Create .env file for Docker
cat > .env << EOF
PERPLEXICA_BACKEND_URL=http://perplexica-app:3000/api/search
EOF

# Uncomment env_file in docker-compose.yml to use .env file
```

Or set environment variables directly in the compose file:

```yaml
environment:
  - PERPLEXICA_BACKEND_URL=http://your-perplexica-host:3000/api/search
```

### Container Details

| Transport | Container Name | Port | Endpoint | Health Check |
|-----------|----------------|------|----------|--------------|
| HTTP      | `perplexica-mcp-http` | 3001 | `/mcp/` | MCP initialize request |
| SSE       | `perplexica-mcp-sse`  | 3001 | `/sse`  | SSE endpoint check |

### Health Monitoring

Both containers include health checks:

```bash
# Check container health
docker ps
docker-compose ps

# View health check logs
docker logs perplexica-mcp-http
docker logs perplexica-mcp-sse
```

### Integration with Perplexica

The Docker setup assumes Perplexica is running in the same Docker network:

```yaml
# Example Perplexica service in the same compose file
services:
  perplexica-app:
    # ... your Perplexica configuration
    networks:
      - backend
  
  perplexica-mcp:
    # ... MCP server configuration
    environment:
      - PERPLEXICA_BACKEND_URL=http://perplexica-app:3000/api/search
    networks:
      - backend
```

### Production Considerations

- Both containers use `restart: unless-stopped` for reliability
- Health checks ensure service availability
- External network allows integration with existing Perplexica deployments
- Security best practices implemented in Dockerfile

## Available Tools

### search

Performs AI-powered web search using Perplexica.

**Parameters:**

- `query` (string, required): Search query
- `focus_mode` (string, required): One of 'webSearch', 'academicSearch', 'writingAssistant', 'wolframAlphaSearch', 'youtubeSearch', 'redditSearch'
- `chat_model` (string, optional): Chat model configuration
- `embedding_model` (string, optional): Embedding model configuration
- `optimization_mode` (string, optional): 'speed' or 'balanced'
- `history` (array, optional): Conversation history
- `system_instructions` (string, optional): Custom instructions
- `stream` (boolean, optional): Whether to stream responses

## Testing

Run the comprehensive test suite to verify all transports:

```bash
uv run src/test_transports.py
```

This will test:

- ✓ Stdio transport with MCP protocol handshake
- ✓ HTTP transport with Streamable HTTP compliance
- ✓ SSE transport endpoint accessibility

## Transport Details

### Stdio Transport

- Uses FastMCP's built-in stdio server
- Supports full MCP protocol including initialization and tool listing
- Ideal for MCP client integration

### SSE Transport

- Server-Sent Events for real-time communication
- Endpoint: `http://host:port/sse`
- Includes periodic ping messages for connection health

### Streamable HTTP Transport

- Compliant with MCP Streamable HTTP specification
- Endpoint: `http://host:port/mcp`
- Returns 307 redirect to `/mcp/` as per protocol
- Uses StreamableHTTPSessionManager for proper session handling

## Development

The server is built using:

- **FastMCP**: Modern MCP server framework with built-in transport support
- **Uvicorn**: ASGI server for SSE and HTTP transports
- **httpx**: HTTP client for Perplexica API communication
- **python-dotenv**: Environment variable management

## Architecture

```none
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   MCP Client    │◄──►│ Perplexica MCP   │◄──►│   Perplexica    │
│                 │    │     Server       │    │   Search API    │
│  (stdio/SSE/    │    │   (FastMCP)      │    │                 │
│   HTTP)         │    │                  │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                              │
                              ▼
                       ┌──────────────┐
                       │   FastMCP    │
                       │  Framework   │
                       │ ┌──────────┐ │
                       │ │  stdio   │ │
                       │ │   SSE    │ │
                       │ │  HTTP    │ │
                       │ └──────────┘ │
                       └──────────────┘
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch (using container-use environments if using Claude Code)
3. Make your changes (within container-use environment if using Claude Code)
4. Add tests if applicable
5. Submit a pull request
6. If using Claude Code, provide access to your work via `container-use log <env_id>` and `container-use checkout <env_id>`

## Support

For issues and questions:

- Check the troubleshooting section
- Review the Perplexica documentation
- Open an issue on GitHub
