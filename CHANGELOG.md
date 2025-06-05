# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2025-06-04

### Changed

- **Architecture Migration**: Complete rewrite using FastMCP framework for unified transport handling
- **Simplified Implementation**: Single server implementation replacing previous multi-transport architecture
- **Dependencies Update**: Migrated from FastAPI to FastMCP, removed Starlette dependency
- **Transport Handling**: Unified transport selection through FastMCP's built-in capabilities
- **Configuration**: Streamlined settings using FastMCP's Settings class

### Improved

- **Code Maintainability**: Significantly reduced codebase complexity with FastMCP integration
- **Transport Reliability**: Enhanced transport stability through FastMCP's proven implementation
- **Error Handling**: Better error handling and timeout management
- **Documentation**: Updated README to reflect current FastMCP-based architecture

### Technical Details

- **FastMCP Integration**: Leverages FastMCP's built-in stdio, SSE, and Streamable HTTP transport support
- **Unified Server**: Single server instance handles all transport modes through FastMCP
- **Settings Management**: Centralized configuration using FastMCP Settings
- **Async Implementation**: Maintained async/await patterns for optimal performance
- **Docker Compatibility**: Updated Dockerfile for FastMCP-based deployment

### Removed

- **FastAPI Dependency**: No longer needed with FastMCP's built-in HTTP support
- **Custom Transport Logic**: Replaced with FastMCP's proven transport implementations
- **Complex Configuration**: Simplified setup with FastMCP's unified approach

## [0.2.0] - 2025-06-02

### Added

- **Multi-Transport Support**: Added support for stdio, SSE (Server-Sent Events), and HTTP transports
- **Command Line Interface**: Added CLI arguments for transport selection and configuration
- **HTTP REST API**: Added FastAPI-based HTTP endpoints for direct API access
- **SSE Transport**: Added Server-Sent Events support for real-time communication
- **Health Check Endpoint**: Added `/health` endpoint for monitoring and load balancing
- **Docker Support**: Added Dockerfile and Docker Compose configuration
- **Nginx Configuration**: Added reverse proxy configuration for load balancing
- **Test Suite**: Added comprehensive transport testing script
- **Configuration Examples**: Added MCP client configuration examples for various scenarios
- **Production Deployment**: Added Kubernetes, systemd, and container deployment examples

### Changed

- **Architecture Refactor**: Restructured codebase to support multiple transport protocols
- **Dependencies**: Added FastAPI, Uvicorn for HTTP/SSE transport support
- **Documentation**: Completely updated README with multi-transport usage instructions
- **Server Implementation**: Enhanced server to handle concurrent transport protocols

### Technical Details

- **Stdio Transport**: Uses FastMCP for simplified MCP server implementation
- **SSE Transport**: Uses standard MCP Server with SSE transport layer
- **HTTP Transport**: Uses FastAPI for REST API endpoints with JSON responses
- **Concurrent Operation**: Supports running all transports simultaneously
- **Load Balancing**: Nginx configuration for distributing requests across multiple instances
- **Health Monitoring**: Built-in health checks for container orchestration
- **Security**: Non-root user in Docker containers, proper error handling

### Deployment Options

- **Development**: Single transport mode for local testing
- **Production**: Multi-transport mode with load balancing
- **Container**: Docker and Docker Compose support
- **Orchestration**: Kubernetes deployment configurations
- **Service**: Systemd service configuration for daemon operation

## [Unreleased]

## [0.1.0] - 2025-05-31

### Added

- Initial implementation of Perplexica MCP Server
- Model Context Protocol (MCP) server acting as a proxy to Perplexica API
- Single search tool that forwards requests to Perplexica endpoint
- Parameter passthrough for all Perplexica search options and configurations
- MCP-compatible interface for AI assistants and other MCP clients

### Changed

- Replaced hardcoded Perplexica backend URL with environment variable configuration
- Updated documentation to explain environment variable configuration
- Added MIT License file
- Added `requires-python` value to pyproject.toml
