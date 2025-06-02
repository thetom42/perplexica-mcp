# Roocode Rules for Perplexica MCP Server

## Project Overview

This is a Model Context Protocol (MCP) server that provides AI assistants with access to Perplexica's AI-powered search engine. The server acts as a proxy between MCP clients and the Perplexica API.

## General

- Maintain a task.md file for high level tasks that need step by step implementation
- Use context7 MCP tool to always gather latest documentation / knowledge about a library
- Use searxng tool to search web pages when necessary
- Use mcp-server-firecrawl tool to scrape web pages when necessary
- Use github tool for actions on remote repositories
- Use git tool for actions on local repositories
- Use perplexica tool for research and information gathering

## Testing

- Write unit tests for all core functionality
- Test MCP tool registration and execution
- Mock external API calls in tests
- Maintain test coverage above 80%

## Miscellaneous

- Keep [`README.md`](README.md:1) up-to-date with installation and usage instructions
- Document all configuration options
- Update [`CHANGELOG.md`](CHANGELOG.md:1) for all releases
- Use conventional commit format: `type(scope): description`
- Use [`.env`](README.md:51) files for local development
- Document all environment variables in [`README.md`](README.md:47)
- Follow semantic versioning (SemVer)
- Update version in [`pyproject.toml`](pyproject.toml:3)
- Tag releases in Git
- Update [`CHANGELOG.md`](CHANGELOG.md:1) with release notes
- Use `uv` for dependency management
- Always use `uv run` for running Python implementations
