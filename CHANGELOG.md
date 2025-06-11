# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.4] - 2025-06-11

### Changed

- **Package Structure**: Refactored package organization to follow Python best practices
- **Code Organization**: Moved main server implementation from `__init__.py` to dedicated `server.py` module
- **Clean Package Init**: Updated `__init__.py` to properly export main function without containing implementation code

### Added

- **Module Execution Support**: Added `__main__.py` to enable `python -m perplexica_mcp` execution
- **Test Compatibility**: Updated test script to work with new package structure

### Fixed

- **Test Script**: Fixed `src/test_transports.py` to use new module execution format
- **Module Import**: Ensured package can be executed both as console script and Python module

### Technical Details

- **Module Separation**: Main server code now in `src/perplexica_mcp/server.py`
- **Package Exports**: Clean `__init__.py` that only exports necessary functions
- **Entry Point**: Maintained proper console script entry point through package imports
- **Module Execution**: Added `__main__.py` for `python -m` compatibility

## [0.3.3] - 2025-06-11

### Fixed

- **PyPI Installation Issue**: Fixed ModuleNotFoundError when installing from PyPI
- **Package Structure**: Restructured package to use proper `perplexica_mcp` module name (underscores instead of hyphens)
- **Entry Point**: Corrected console script entry point to match package structure
- **Module Import**: Fixed import path for proper module resolution in installed packages

### Changed

- **Package Layout**: Moved main module to `src/perplexica_mcp/__init__.py` for proper package structure
- **Build Configuration**: Updated hatchling configuration to handle hyphenated package names correctly

### Technical Details

- **Module Resolution**: Package now correctly resolves as `perplexica_mcp` module when installed
- **Console Script**: Entry point now properly references `perplexica_mcp:main`
- **Package Discovery**: Fixed package discovery for PyPI installations using uvx

## [0.3.2] - 2025-06-11

### Added

- **PyPI Publishing Ready**: Complete package structure for PyPI publication
- **uvx Compatibility**: Full support for uvx execution with proper entry points
- **Package Structure**: Added proper `__init__.py` with module exports
- **MANIFEST.in**: Added source distribution control file
- **Publishing Documentation**: Added comprehensive `PUBLISHING.md` guide

### Changed

- **Entry Point Configuration**: Fixed console script entry point for proper PyPI installation
- **Package Build Configuration**: Enhanced pyproject.toml with proper sdist settings
- **Documentation**: Updated README.md with PyPI installation instructions and uvx usage examples
- **Installation Methods**: Added both PyPI and source installation options throughout documentation

### Fixed

- **Console Script**: Corrected entry point from `"perplexica_mcp:main"` to `"src.perplexica_mcp:main"`
- **Package Discovery**: Fixed package structure for proper Python module resolution
- **Build System**: Enhanced hatchling configuration for reliable package building

### Technical Details

- **uvx Support**: Package now works seamlessly with `uvx perplexica-mcp` commands
- **PyPI Metadata**: Complete metadata configuration for package discovery and installation
- **Build Verification**: Tested successful package building and uvx execution
- **Source Distribution**: Proper file inclusion for source distributions

## [0.3.1] - 2025-06-06

### Added

- **PyPI Deployment Support**: Complete pyproject.toml configuration for PyPI publishing
- **Build System Configuration**: Added Hatchling build backend with proper file inclusion
- **Project Metadata**: Added comprehensive metadata including description, authors, classifiers, and URLs
- **Development Dependencies**: Added optional development dependencies for testing and linting

### Changed

- **Package Structure**: Configured Hatchling to properly handle src/ directory layout
- **Version**: Bumped to 0.3.1 for PyPI deployment readiness

### Technical Details

- **Build Backend**: Uses Hatchling for modern Python packaging standards
- **PyPI Classifiers**: Added appropriate trove classifiers for package discovery
- **Project URLs**: Added links to homepage, repository, issues, and documentation
- **License**: Properly configured MIT license for PyPI display

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
