# Repository Structure

Here is the suggested new structure for the repository:

```mermaid
graph TD
    A[perplexica-mcp/]
    A --> B[src/]
    B --> B1[perplexica_mcp.py]
    B --> B2[test_transports.py]
    A --> C[config/]
    C --> C1[nginx.conf]
    C --> C2[mcp-config-examples.json]
    A --> D[docs/]
    D --> D1[CLAUDE.md]
    A --> E[docker-compose.yml]
    A --> F[Dockerfile]
    A --> G[.gitignore]
    A --> H[.env]
    A --> I[.sample.env]
    A --> J[LICENSE]
    A --> K[pyproject.toml]
    A --> L[README.md]
```

## Explanation of Changes

* **`src/` (or `app/`)**: This directory would contain all the primary application source code.
  * `perplexica_mcp.py`
  * `test_transports.py`
* **`config/`**: This directory would hold configuration files that are not directly part of the application's source code but configure its environment or external services.
  * `nginx.conf`
  * `mcp-config-examples.json`
* **`docs/`**: A dedicated directory for documentation files.
  * `CLAUDE.md`
* **Root Directory**: Files that are essential for the project's overall management or are typically found at the root level remain here.
  * `docker-compose.yml`
  * `Dockerfile`
  * `.gitignore`
  * `.env` (the actual environment file, which should be ignored by Git)
  * `.sample.env` (a template for the `.env` file)
  * `LICENSE`
  * `pyproject.toml`
  * `README.md`
  * `CHANGELOG.md`
