# Perplexica MCP Server - Docker Compose Installation Guide

This guide provides step-by-step instructions for setting up the Perplexica MCP Server with a local Perplexica backend using Docker Compose.

## Overview

The Perplexica MCP Server connects to a Perplexica search backend to provide AI-powered search capabilities through the Model Context Protocol (MCP). Both services run in Docker containers on a shared network.

## Prerequisites

- **Docker & Docker Compose**: Latest version
- **Git**: For cloning repositories
- **Python 3.9+**: For verification commands (optional)

## Step 1: Setup Local Perplexica Backend

### 1.1 Clone and Configure Perplexica

```bash
# Clone the main Perplexica repository
git clone https://github.com/ItzCrazyKns/Perplexica.git
cd Perplexica

# Copy and configure the settings
cp sample.config.toml config.toml
# Edit config.toml with your API keys (OpenAI, etc.)
```

### 1.2 Create Docker Network

```bash
# Create a shared Docker network for both services
docker network create perplexica-network
```

### 1.3 Configure Perplexica Network

The default Perplexica `docker-compose.yml` needs to be modified to use the external network:

```bash
# Edit the docker-compose.yml in the Perplexica directory
# Find the networks section at the bottom and modify it:
```

**Before:**

```yaml
networks:
  perplexica-network:
```

**After:**

```yaml
networks:
  perplexica-network:
    external: true
```

Alternatively, you can use this command to make the change automatically:

```bash
# In the Perplexica directory
sed -i.bak '/^networks:$/,/^[[:space:]]*perplexica-network:$/c\
networks:\
  perplexica-network:\
    external: true\
' docker-compose.yaml
```

### 1.4 Start Perplexica Backend

```bash
# Start Perplexica with Docker Compose
docker compose up -d

# Wait for services to start, then verify 
# NOTE: Both chatModel and embeddingModel are REQUIRED by the Perplexica API
# Replace the provider and model names with your configured models
curl -v "http://localhost:3000/api/search" \
  -H "Content-Type: application/json" \
  -H "Accept: text/event-stream" \
  -d '{
    "query": "What is artificial intelligence?",
    "focusMode": "webSearch",
    "optimizationMode": "balanced",
    "history": [],
    "stream": false,
    "chatModel": {
      "provider": "ollama",
      "name": "qwen3:4b"
    },
    "embeddingModel": {
      "provider": "ollama",
      "name": "nomic-embed-text:latest"
    }
  }'
# Expected response: lengthy model response, eg.
# {"message":"<think>\nOkay, the user is asking, ..."}
```

## Step 2: Setup Perplexica MCP Server

### 2.1 Clone MCP Server Repository

```bash
# In a separate directory
git clone https://github.com/thetom42/perplexica-mcp.git
cd perplexica-mcp
```

### 2.2 Configure Environment Variables

The Perplexica MCP server requires configuration of chat and embedding models. Copy the sample environment file and configure it:

```bash
# Copy the sample environment file
cp .sample.env .env

# Edit the .env file with your model configuration
nano .env  # or use your preferred editor
```

**Required Configuration (.env file):**

```env
# Perplexica Backend Configuration
PERPLEXICA_BACKEND_URL=http://perplexica-app-1:3000/api/search

# Default Model Configuration (Required)
# The Perplexica API requires both chatModel and embeddingModel to be specified.
# These will be used as defaults when no model is explicitly provided in the search request.

# Chat Model Configuration
PERPLEXICA_CHAT_MODEL_PROVIDER=ollama
PERPLEXICA_CHAT_MODEL_NAME=qwen3:4b

# Embedding Model Configuration  
PERPLEXICA_EMBEDDING_MODEL_PROVIDER=ollama
PERPLEXICA_EMBEDDING_MODEL_NAME=nomic-embed-text:latest

# Timeout Configuration (Optional)
PERPLEXICA_READ_TIMEOUT=60
```

**Important Notes:**

- Both `chatModel` and `embeddingModel` are **required** by the Perplexica API
- The model names must match exactly what's available in your Perplexica instance
- For Ollama models, include the tag (e.g., `:latest`, `:4b`) in the model name
- Use `http://perplexica-app-1:3000/api/search` for Docker container networking

### 2.3 Start MCP Server with Docker Compose

```bash
# Start the MCP server (HTTP transport)
docker compose up -d

# Or for SSE transport
# docker compose -f docker-compose-sse.yml up -d
```

## Step 3: Verify Docker Network Configuration

### 3.1 Check Network Connectivity

```bash
# Verify both containers are on the perplexica-network
docker network inspect perplexica-network --format '{{json .Containers}}' | python3 -m json.tool
```

You should see at least the following containers listed:

- `perplexica-app-1` (Perplexica backend)
- `perplexica-mcp-http` (MCP server)

### 3.2 Test Inter-Container Communication

```bash
# Test connection from MCP server to Perplexica backend
docker exec perplexica-mcp-http python -c "
import urllib.request, json
try:
    req = urllib.request.Request(
        'http://perplexica-app-1:3000/api/search',
        data=json.dumps({'query':'test','focusMode':'webSearch'}).encode(),
        headers={'Content-Type': 'application/json'}
    )
    response = urllib.request.urlopen(req, timeout=10)
    print('✓ Connection successful:', response.status)
except Exception as e:
    print('✗ Connection failed:', e)
"
```

### 3.3 Verify MCP Server

```bash
# Test MCP server initialization
curl -X POST http://localhost:3001/mcp \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -d '{
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {
      "protocolVersion": "2024-11-05",
      "capabilities": {},
      "clientInfo": {"name": "test", "version": "1.0"}
    }
  }'
```

Expected response should include `"serverInfo":{"name":"Perplexica"...}`

### 3.4 Network Troubleshooting

If containers are not appearing on the same network, check these common issues:

```bash
# 1. Verify the external network exists
docker network ls | grep perplexica-network

# 2. Check if Perplexica docker-compose.yaml has external: true
grep -A 2 "networks:" /path/to/Perplexica/docker-compose.yaml

# 3. If you see containers on different networks, restart them:
cd /path/to/Perplexica && docker compose down && docker compose up -d
cd /path/to/perplexica-mcp && docker compose down && docker compose up -d

# 4. Check actual network names (Docker may add prefixes)
docker network ls
docker ps --format "table {{.Names}}\t{{.Networks}}"
```

## Step 4: IDE Configuration

### 4.1 VS Code

Create `.vscode/mcp.json` in your project:

```json
{
  "servers": {
    "perplexica": {
      "type": "http",
      "url": "http://localhost:3001/mcp",
      "headers": {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream"
      }
    }
  }
}
```

### 4.2 Roocode

Create `.roo/mcp.json` in your project:

```json
{
  "mcpServers": {
    "perplexica": {
      "type": "streamable-http",
      "url": "http://localhost:3001/mcp",
      "headers": {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream"
      }
    }
  }
}
```

### 4.3 Kilocode

Create `.kilocode/mcp.json` in your project:

```json
{
  "mcpServers": {
    "perplexica": {
      "type": "streamable-http",
      "url": "http://localhost:3001/mcp",
      "alwaysAllow": ["search"],
      "disabled": false
    }
  }
}
```

### 4.4 Cursor

Create `.cursor/mcp.json` in your project:

```json
{
  "mcpServers": {
    "perplexica": {
      "type": "http",
      "url": "http://localhost:3001/mcp",
      "headers": {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream"
      }
    }
  }
}
```

### 4.5 Windsurf

Add this to your `~/.codeium/windsurf/mcp_config.json`:

```json
    "perplexica": {
      "serverUrl": "http://localhost:3001/mcp",
      "headers": {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream"
      }
    }
```

## Troubleshooting

### Check Container Status

```bash
# List running containers
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

# Check logs
docker logs perplexica-app-1
docker logs perplexica-mcp-http
```

### Network Issues

```bash
# List Docker networks
docker network ls

# Check network details
docker network inspect perplexica-network

# Verify containers are connected
docker network inspect perplexica-network --format '{{range .Containers}}{{.Name}} - {{.IPv4Address}}{{"\n"}}{{end}}'
```

### Common Issues

1. **Port conflicts**: Ensure ports 3000 and 3001 are available
2. **Network isolation**: Both containers must be on `perplexica-network`
3. **Startup timing**: Perplexica backend must start before MCP server connects

## Remote Perplexica Backend Configuration

If your Perplexica backend is hosted remotely (e.g., `https://perplexica.my-domain.com`), modify the MCP server configuration:

### Update Docker Compose

Edit `docker-compose.yml`:

```yaml
services:
  perplexica-mcp:
    build: .
    image: perplexica-mcp-http
    container_name: perplexica-mcp-http
    ports:
      - "3001:3001"
    environment:
      - PERPLEXICA_BACKEND_URL=https://perplexica.my-domain.com/api/search
      - PERPLEXICA_READ_TIMEOUT=60  # Increase timeout for remote connections
    command: python src/perplexica_mcp/server.py http
    restart: unless-stopped
    # Remove networks section since no local Perplexica network needed
```

### Restart MCP Server

```bash
# Stop current container
docker compose down

# Start with new configuration
docker compose up -d
```

### Verify Remote Connection

```bash
# Test remote Perplexica directly
curl https://perplexica.my-domain.com/api/search

# Test MCP server with remote backend
docker exec perplexica-mcp-http python -c "
import urllib.request, json
try:
    req = urllib.request.Request(
        'https://perplexica.my-domain.com/api/search',
        data=json.dumps({'query':'test','focusMode':'webSearch'}).encode(),
        headers={'Content-Type': 'application/json'}
    )
    response = urllib.request.urlopen(req, timeout=60)
    print('✓ Remote connection successful:', response.status)
except Exception as e:
    print('✗ Remote connection failed:', e)
"
```

### Security Considerations for Remote Backend

- Ensure HTTPS is used for remote connections
- Verify SSL certificates are valid
- Consider authentication if required by your remote Perplexica instance
- Increase timeout values for network latency

## Testing Your Setup

Once configured, test the MCP search tool in your IDE:

1. Open a project in your configured IDE
2. Look for the Perplexica search tool in available MCP tools
3. Try a search query like "What is artificial intelligence?"
4. Verify you receive results with source citations

The search tool supports various focus modes:

- `webSearch` - General web search
- `academicSearch` - Academic papers and research
- `writingAssistant` - Writing assistance
- `youtubeSearch` - YouTube video content
- `redditSearch` - Reddit discussions
- `wolframAlphaSearch` - Mathematical queries
