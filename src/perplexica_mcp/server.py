#!/usr/bin/env python3

import argparse
import os
from typing import Annotated, Optional, Union

import httpx
from dotenv import load_dotenv
from fastmcp import FastMCP
from pydantic import BaseModel, ConfigDict, Field
from urllib.parse import urlparse, urlunparse


# Structured output models for MCP 2025-06-18 compliance
class Source(BaseModel):
    """A source document from search results."""
    model_config = ConfigDict(extra="allow")  # Allow additional fields from Perplexica

    title: Optional[str] = None
    url: Optional[str] = None
    content: Optional[str] = None


class SearchResult(BaseModel):
    """Successful search result from Perplexica."""
    model_config = ConfigDict(extra="allow")  # Allow additional fields from Perplexica

    answer: Optional[str] = Field(default=None, description="The AI-generated answer")
    sources: list[Source] = Field(default_factory=list, description="Source documents")


class SearchError(BaseModel):
    """Error response from search."""
    error: str = Field(description="Error message")


SearchResponse = Union[SearchResult, SearchError]

# Load environment variables from .env file
load_dotenv()

# Get the backend URL from environment variable or use default
PERPLEXICA_BACKEND_URL = os.getenv(
    "PERPLEXICA_BACKEND_URL", "http://localhost:3000/api/search"
)
PERPLEXICA_READ_TIMEOUT = int(os.getenv("PERPLEXICA_READ_TIMEOUT", 60))

# Default model configurations from environment variables
DEFAULT_CHAT_MODEL = None
if os.getenv("PERPLEXICA_CHAT_MODEL_PROVIDER") and os.getenv(
    "PERPLEXICA_CHAT_MODEL_NAME"
):
    DEFAULT_CHAT_MODEL = {
        "provider": os.getenv("PERPLEXICA_CHAT_MODEL_PROVIDER"),
        "name": os.getenv("PERPLEXICA_CHAT_MODEL_NAME"),
    }

DEFAULT_EMBEDDING_MODEL = None
if os.getenv("PERPLEXICA_EMBEDDING_MODEL_PROVIDER") and os.getenv(
    "PERPLEXICA_EMBEDDING_MODEL_NAME"
):
    DEFAULT_EMBEDDING_MODEL = {
        "provider": os.getenv("PERPLEXICA_EMBEDDING_MODEL_PROVIDER"),
        "name": os.getenv("PERPLEXICA_EMBEDDING_MODEL_NAME"),
    }

# Create FastMCP server with default settings
mcp = FastMCP("Perplexica")


def _providers_url_from_search_url(search_url: str) -> str:
    """Convert search URL to providers URL."""
    parsed = urlparse(search_url)
    path = parsed.path or ""
    if path.endswith("/api/search"):
        base = path[: -len("/api/search")]
        new_path = base + "/api/providers"
    else:
        new_path = "/api/providers"
    return urlunparse((parsed.scheme, parsed.netloc, new_path, "", "", ""))


async def _fetch_providers(client: httpx.AsyncClient) -> list:
    """Fetch providers from Perplexica backend."""
    providers_url = _providers_url_from_search_url(PERPLEXICA_BACKEND_URL)
    resp = await client.get(providers_url, timeout=PERPLEXICA_READ_TIMEOUT)
    resp.raise_for_status()
    data = resp.json()
    return data.get("providers", [])


def _find_provider(providers: list, provider_identifier: str) -> Optional[dict]:
    """Find provider by UUID or name (case-insensitive)."""
    if not provider_identifier:
        return None
    # Try by id exact
    for p in providers:
        if str(p.get("id")) == str(provider_identifier):
            return p
    # Try by name (case-insensitive)
    for p in providers:
        if str(p.get("name", "")).lower() == str(provider_identifier).lower():
            return p
    return None


def _find_model_key(models: list, model_identifier: str) -> Optional[str]:
    """Find model key by key or display name."""
    if not models or not model_identifier:
        return None
    for m in models:
        if str(m.get("key")) == str(model_identifier):
            return m.get("key")
    for m in models:
        if str(m.get("name", "")).lower() == str(model_identifier).lower():
            return m.get("key")
    return None


async def _normalize_model_spec(client: httpx.AsyncClient, model_spec: dict, is_embedding: bool):
    """Normalize model spec to providerId/key format."""
    if model_spec is None:
        return None

    # If already has providerId and key, return as-is
    if "providerId" in model_spec and "key" in model_spec:
        return {"providerId": str(model_spec["providerId"]), "key": str(model_spec["key"])}

    providers = await _fetch_providers(client)

    # Get provider identifier
    provider_identifier = model_spec.get("provider") or model_spec.get("providerId")
    if not provider_identifier:
        raise ValueError("Model spec must include provider or providerId")

    provider = _find_provider(providers, provider_identifier)
    if not provider:
        raise ValueError(f"Provider '{provider_identifier}' not found")

    # Get models list
    models_list = provider.get("embeddingModels" if is_embedding else "chatModels", [])

    # If key is provided, validate it
    if "key" in model_spec:
        key = str(model_spec["key"])
        if _find_model_key(models_list, key) is None:
            raise ValueError(f"Model key '{key}' not found for provider '{provider.get('name')}'")
        return {"providerId": provider.get("id"), "key": key}

    # Try to resolve by name
    model_identifier = model_spec.get("name")
    if not model_identifier:
        raise ValueError("Model spec must include model name or key")

    key = _find_model_key(models_list, model_identifier)
    if not key:
        raise ValueError(f"Model '{model_identifier}' not found for provider '{provider.get('name')}'")
    return {"providerId": provider.get("id"), "key": key}


async def perplexica_search(
    query,
    sources,
    chat_model=None,
    embedding_model=None,
    optimization_mode=None,
    history=None,
    system_instructions=None,
    stream=False,
) -> SearchResponse:
    """
    Search using the Perplexica API

    Args:
        query (str): The search query
        sources (list): Search sources - list containing: "web", "academic", "discussions"
        chat_model (dict, optional): Chat model configuration with:
            provider: Provider name (e.g., openai, ollama)
            name: Model name (e.g., gpt-4o-mini)
        embedding_model (dict, optional): Embedding model configuration with:
            provider: Provider name (e.g., openai)
            name: Model name (e.g., text-embedding-3-small)
        optimization_mode (str, optional): Optimization mode (speed, balanced, quality)
        history (list, optional): Conversation history as [["human", "text"], ["assistant", "text"]] pairs
        system_instructions (str, optional): Custom system instructions
        stream (bool, optional): Whether to stream responses

    Returns:
        dict: Search results from Perplexica
    """

    # Prepare the request payload
    payload = {"query": query, "sources": sources}

    # Add optional parameters if provided
    if chat_model:
        payload["chatModel"] = chat_model
    if embedding_model:
        payload["embeddingModel"] = embedding_model
    if optimization_mode:
        payload["optimizationMode"] = optimization_mode
    else:
        payload["optimizationMode"] = "balanced"
    if history is not None:
        payload["history"] = history
    else:
        payload["history"] = []
    if system_instructions:
        payload["systemInstructions"] = system_instructions
    if stream is not None:
        payload["stream"] = stream

    try:
        async with httpx.AsyncClient() as client:
            # Normalize model specifications to providerId/key format
            try:
                if "chatModel" in payload and payload["chatModel"] is not None:
                    normalized_chat = await _normalize_model_spec(client, payload["chatModel"], is_embedding=False)
                    payload["chatModel"] = normalized_chat
                if "embeddingModel" in payload and payload["embeddingModel"] is not None:
                    normalized_embed = await _normalize_model_spec(client, payload["embeddingModel"], is_embedding=True)
                    payload["embeddingModel"] = normalized_embed
            except ValueError as ve:
                return SearchError(error=f"Invalid model configuration: {ve!s}")

            response = await client.post(
                PERPLEXICA_BACKEND_URL, json=payload, timeout=PERPLEXICA_READ_TIMEOUT
            )
            response.raise_for_status()
            data = response.json()
            # Parse response into structured format
            if "error" in data:
                return SearchError(error=data["error"])
            return SearchResult(**data)
    except httpx.HTTPError as e:
        return SearchError(error=f"HTTP error occurred: {e!s}")
    except Exception as e:  # noqa: BLE001 - ensure tool always returns SearchError
        return SearchError(error=f"An error occurred: {e!s}")


@mcp.tool()
async def search(
    query: Annotated[str, Field(description="Search query")],
    sources: Annotated[
        list,
        Field(
            description="Search sources array. Valid values: 'web' (general web search), 'academic' (scholarly articles), 'discussions' (forums like Reddit)"
        ),
    ],
    chat_model: Annotated[
        Optional[dict], Field(description="Chat model configuration")
    ] = DEFAULT_CHAT_MODEL,
    embedding_model: Annotated[
        Optional[dict], Field(description="Embedding model configuration")
    ] = DEFAULT_EMBEDDING_MODEL,
    optimization_mode: Annotated[
        Optional[str], Field(description="Optimization mode: speed, balanced, or quality")
    ] = None,
    history: Annotated[
        Optional[list], Field(description="Conversation history as [[role, text], ...] pairs")
    ] = None,
    system_instructions: Annotated[
        Optional[str], Field(description="Custom system instructions")
    ] = None,
    stream: Annotated[bool, Field(description="Whether to stream responses")] = False,
) -> SearchResponse:
    """
    Search using Perplexica's AI-powered search engine.

    This tool provides access to Perplexica's search capabilities with multiple source types
    that can be combined: web search, academic search, and discussions (forums).
    """
    # Normalize model defaults before validation
    if chat_model is None:
        chat_model = DEFAULT_CHAT_MODEL
    if embedding_model is None:
        embedding_model = DEFAULT_EMBEDDING_MODEL
    # Fail fast if required models are absent
    if chat_model is None or embedding_model is None:
        return SearchError(
            error="Both chatModel and embeddingModel are required. Configure PERPLEXICA_* model env vars or pass them in the request."
        )

    return await perplexica_search(
        query=query,
        sources=sources,
        chat_model=chat_model,
        embedding_model=embedding_model,
        optimization_mode=optimization_mode,
        history=history,
        system_instructions=system_instructions,
        stream=stream,
    )


def main():
    """Main entry point for the Perplexica MCP server."""
    parser = argparse.ArgumentParser(description="Perplexica MCP Server")
    parser.add_argument(
        "transport", choices=["stdio", "sse", "http"], help="Transport type to use"
    )
    parser.add_argument(
        "host",
        nargs="?",
        default="0.0.0.0",
        help="Host to bind to for SSE/HTTP transports (default: 0.0.0.0)",
    )
    parser.add_argument(
        "port",
        nargs="?",
        type=int,
        default=3001,
        help="Port for SSE/HTTP transports (default: 3001)",
    )

    args = parser.parse_args()

    if args.transport == "stdio":
        mcp.run(transport="stdio")
    elif args.transport == "sse":
        print(
            f"Starting Perplexica MCP server with SSE transport on {args.host}:{args.port}"
        )
        print(f"SSE endpoint: http://{args.host}:{args.port}/sse")
        mcp.run(transport="sse", host=args.host, port=args.port)
    elif args.transport == "http":
        print(
            f"Starting Perplexica MCP server with Streamable HTTP transport on {args.host}:{args.port}"
        )
        print(f"HTTP endpoint: http://{args.host}:{args.port}/mcp")
        mcp.run(transport="http", host=args.host, port=args.port, path="/mcp")


if __name__ == "__main__":
    main()
