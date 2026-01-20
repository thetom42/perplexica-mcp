# Perplexica Backend Configuration
export PERPLEXICA_BACKEND_URL="http://[ip]:[port]/api/search" # Replace with the actual backend URL

# Default Model Configuration
export PERPLEXICA_CHAT_MODEL_PROVIDER="openai"
export PERPLEXICA_CHAT_MODEL_NAME="gpt-3.5-turbo"

export PERPLEXICA_EMBEDDING_MODEL_PROVIDER="openai"
export PERPLEXICA_EMBEDDING_MODEL_NAME="text-embedding-3-small"

uv run src/perplexica_mcp/server.py http 0.0.0.0 12345
