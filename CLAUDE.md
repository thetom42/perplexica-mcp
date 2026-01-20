# CLAUDE.md

This file provides guidance for AI assistants working with this codebase.

## Project Overview

Perplexica MCP Server - A Model Context Protocol (MCP) server providing AI-powered search functionality via Perplexica.

## Key Files

- `src/perplexica_mcp/server.py` - Main server implementation
- `pyproject.toml` - Project configuration and dependencies
- `README.md` - User documentation

## Development Guidelines

- Use feature branches for changes
- Follow existing code style
- Update CHANGELOG.md for notable changes
- Test changes before submitting PRs

## Testing

```bash
# Run transport tests
uv run python src/test_transports.py
```
