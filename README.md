# Perplexica MCP Server

A Model Context Protocol (MCP) server that provides access to Perplexica's AI-powered search engine. This server supports multiple transport protocols: stdio, Server-Sent Events (SSE), and HTTP.

## Features

- **Multi-Transport Support**: Choose between stdio, SSE, or HTTP transports
- **AI-Powered Search**: Access Perplexica's intelligent search capabilities
- **Multiple Focus Modes**: Support for web search, academic search, writing assistant, and more
- **Flexible Configuration**: Customizable chat models, embedding models, and optimization modes
- **Conversation History**: Maintain context across search queries

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd perplexica-mcp
```

2. Install dependencies:
```bash
pip install -e .
```

3. Set up environment variables:
```bash
cp .sample.env .env
# Edit .env with your Perplexica backend URL
```

## Usage

### Transport Options

The server supports three different transport protocols:

#### 1. Stdio Transport (Default)
Best for MCP clients that communicate via standard input/output:
```bash
python perplexica_mcp_tool.py --transport stdio
```

#### 2. SSE Transport
Best for web-based clients that need real-time communication:
```bash
python perplexica_mcp_tool.py --transport sse --host 0.0.0.0 --sse-port 3001
```

#### 3. HTTP Transport
Best for REST API integration:
```bash
python perplexica_mcp_tool.py --transport http --host 0.0.0.0 --http-port 3002
```

#### 4. All Transports
Run all transports simultaneously:
```bash
python perplexica_mcp_tool.py --transport all --host 0.0.0.0 --sse-port 3001 --http-port 3002
```

### Command Line Options

- `--transport`: Choose transport type (`stdio`, `sse`, `http`, `all`)
- `--host`: Host to bind to for SSE/HTTP transports (default: `0.0.0.0`)
- `--sse-port`: Port for SSE transport (default: `3001`)
- `--http-port`: Port for HTTP transport (default: `3002`)

### MCP Client Configuration

#### For Claude Desktop (stdio)
Add to your Claude Desktop configuration:
```json
{
  "mcpServers": {
    "perplexica": {
      "command": "python",
      "args": ["/path/to/perplexica_mcp_tool.py", "--transport", "stdio"]
    }
  }
}
```

#### For SSE Clients
Connect to: `http://localhost:3001/sse`

#### For HTTP Clients
Make POST requests to: `http://localhost:3002/search`

## API Reference

### Search Tool

The server provides a `search` tool with the following parameters:

#### Required Parameters
- `query` (string): The search query or question
- `focus_mode` (string): Focus mode, must be one of:
  - `webSearch`: General web search
  - `academicSearch`: Academic and research-focused search
  - `writingAssistant`: Writing and content creation assistance
  - `wolframAlphaSearch`: Mathematical and computational queries
  - `youtubeSearch`: Video content search
  - `redditSearch`: Reddit community discussions

#### Optional Parameters
- `chat_model` (string): Chat model configuration
- `embedding_model` (string): Embedding model configuration
- `optimization_mode` (string): `speed` or `balanced`
- `history` (array): Conversation history as `[role, message]` pairs
- `system_instructions` (string): Custom instructions for AI response
- `stream` (boolean): Whether to stream responses (default: false)

### HTTP API Endpoints

When using HTTP transport, the following endpoints are available:

#### POST /search
Search using Perplexica's AI-powered search engine.

**Request Body:**
```json
{
  "query": "What is quantum computing?",
  "focus_mode": "academicSearch",
  "optimization_mode": "balanced",
  "system_instructions": "Provide a technical but accessible explanation"
}
```

**Response:**
```json
{
  "message": "Quantum computing is...",
  "sources": [...]
}
```

#### GET /health
Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "service": "perplexica-mcp"
}
```

## Configuration

### Environment Variables

Create a `.env` file with the following variables:

```env
PERPLEXICA_BACKEND_URL=http://localhost:3000/api/search
```

### Focus Modes

- **webSearch**: General web search for any topic
- **academicSearch**: Academic papers and research content
- **writingAssistant**: Help with writing and content creation
- **wolframAlphaSearch**: Mathematical computations and data analysis
- **youtubeSearch**: Video content and tutorials
- **redditSearch**: Community discussions and opinions

## Examples

### Basic Search
```python
# Using the MCP tool
result = search(
    query="What are the benefits of renewable energy?",
    focus_mode="webSearch"
)
```

### Academic Research
```python
# Academic search with conversation history
result = search(
    query="Latest developments in machine learning",
    focus_mode="academicSearch",
    optimization_mode="balanced",
    history=[
        ["human", "I'm researching AI trends"],
        ["assistant", "I can help you find recent research papers"]
    ]
)
```

### HTTP API Usage
```bash
curl -X POST http://localhost:3002/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Climate change solutions",
    "focus_mode": "academicSearch",
    "optimization_mode": "speed"
  }'
```

## Development

### Running Tests
```bash
# Test stdio transport
python perplexica_mcp_tool.py --transport stdio

# Test SSE transport
python perplexica_mcp_tool.py --transport sse --sse-port 3001

# Test HTTP transport
python perplexica_mcp_tool.py --transport http --http-port 3002
```

### Architecture

The server implementation uses:
- **FastMCP**: For stdio transport (simplified MCP server)
- **Standard MCP Server**: For SSE and HTTP transports
- **FastAPI**: For HTTP REST API endpoints
- **Uvicorn**: ASGI server for SSE and HTTP transports

## Troubleshooting

### Common Issues

1. **Connection Refused**: Ensure Perplexica backend is running
2. **Port Already in Use**: Change port numbers using command line options
3. **Import Errors**: Install all dependencies with `pip install -e .`

### Debug Mode
Add logging to debug issues:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Support

For issues and questions:
- Check the troubleshooting section
- Review the Perplexica documentation
- Open an issue on GitHub
