# Publishing to PyPI

This document outlines the steps to publish the Perplexica MCP Server to PyPI.

## Prerequisites

1. Install build tools:

   ```bash
   pip install build twine
   ```

2. Set up PyPI credentials:
   - Create account on [PyPI](https://pypi.org/)
   - Generate API token in account settings
   - Configure credentials in `~/.pypirc` or use environment variables

## Build Process

1. Clean previous builds:

   ```bash
   rm -rf dist/ build/ src/*.egg-info/
   ```

2. Build the package:

   ```bash
   uv build
   ```

3. Verify the build:

   ```bash
   twine check dist/*
   ```

## Publishing

### Test PyPI (Recommended first)

1. Upload to Test PyPI:

   ```bash
   twine upload --repository testpypi dist/*
   ```

2. Test installation:

   ```bash
   uvx --index-url https://test.pypi.org/simple/ perplexica-mcp --help
   ```

### Production PyPI

1. Upload to PyPI:

   ```bash
   twine upload dist/*
   ```

2. Verify installation:

   ```bash
   uvx perplexica-mcp --help
   ```

## Version Management

1. Update version in `pyproject.toml`
2. Update `CHANGELOG.md`
3. Create git tag:

   ```bash
   git tag v0.3.2
   git push origin v0.3.2
   ```

## Automated Publishing

Consider setting up GitHub Actions for automated publishing on tag creation.

## uvx Compatibility

The package is configured to work with `uvx` out of the box:

- Entry point: `perplexica-mcp`
- All dependencies are properly declared
- Console script is correctly configured

Users can run:

```bash
uvx perplexica-mcp stdio
uvx perplexica-mcp sse
uvx perplexica-mcp http
```
