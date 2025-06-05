# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Model Context Protocol (MCP) server that acts as a proxy to Perplexica's AI-powered search engine. The server exposes Perplexica's search functionality as an MCP tool for AI assistants and other MCP clients.

## Key Architecture

- **Single Module Structure**: The entire MCP server is implemented in `perplexica_mcp.py`
- **FastMCP Framework**: Uses the FastMCP library for MCP server implementation
- **API Proxy Pattern**: Acts as a pass-through proxy to the Perplexica backend at `/api/search`
- **Environment-based Configuration**: Backend URL configured via `PERPLEXICA_BACKEND_URL` environment variable

## Development Commands

### Installation and Setup

```bash
pip install -e .
```

### Running the MCP Server

```bash
python perplexica_mcp.py
```

### Running via uv (recommended for MCP clients)

```bash
uv run perplexica_mcp.py
```

## Configuration

### Environment Variables

- `PERPLEXICA_BACKEND_URL`: The Perplexica backend endpoint (defaults to `http://localhost:3000/api/search`)

### Dependencies

- `mcp[cli]`: Model Context Protocol framework
- `requests`: HTTP client for backend communication
- `python-dotenv`: Environment variable management
- `pydantic`: Data validation and type hints

## MCP Tool Implementation

The server exposes a single `search` tool with the following required parameters:

- `query`: Search query string
- `focus_mode`: Must be one of: `webSearch`, `academicSearch`, `writingAssistant`, `wolframAlphaSearch`, `youtubeSearch`, `redditSearch`

Optional parameters are passed through to the Perplexica backend including chat models, embedding models, optimization modes, conversation history, and system instructions.

## Integration Points

- **Backend Communication**: All requests are forwarded to the configured Perplexica backend via POST requests
- **Parameter Mapping**: MCP tool parameters are mapped to Perplexica API format (e.g., `focus_mode` becomes `focusMode`)
- **Error Handling**: HTTP errors from the backend are propagated via `response.raise_for_status()`
