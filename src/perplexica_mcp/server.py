#!/usr/bin/env python3

import os
import argparse
import httpx
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.server import Settings
from pydantic import Field
from typing import Annotated
import uvicorn

# Load environment variables from .env file
load_dotenv()

# Get the backend URL from environment variable or use default
PERPLEXICA_BACKEND_URL = os.getenv('PERPLEXICA_BACKEND_URL', 'http://localhost:3000/api/search')
PERPLEXICA_READ_TIMEOUT = int(os.getenv('PERPLEXICA_READ_TIMEOUT', 30))

# Default model configurations from environment variables
DEFAULT_CHAT_MODEL = None
if os.getenv("PERPLEXICA_CHAT_MODEL_PROVIDER") and os.getenv("PERPLEXICA_CHAT_MODEL_NAME"):
    DEFAULT_CHAT_MODEL = {
        "provider": os.getenv("PERPLEXICA_CHAT_MODEL_PROVIDER"),
        "name": os.getenv("PERPLEXICA_CHAT_MODEL_NAME"),
    }

DEFAULT_EMBEDDING_MODEL = None
if os.getenv("PERPLEXICA_EMBEDDING_MODEL_PROVIDER") and os.getenv("PERPLEXICA_EMBEDDING_MODEL_NAME"):
    DEFAULT_EMBEDDING_MODEL = {
        "provider": os.getenv("PERPLEXICA_EMBEDDING_MODEL_PROVIDER"),
        "name": os.getenv("PERPLEXICA_EMBEDDING_MODEL_NAME"),
    }

# Create FastMCP server with settings for all transports
settings = Settings(
    host="0.0.0.0",
    port=3001,
    sse_path="/sse",
    message_path="/messages/",
    streamable_http_path="/mcp"
)

mcp = FastMCP("Perplexica", dependencies=["httpx", "mcp", "python-dotenv", "uvicorn"], settings=settings)

async def perplexica_search(
    query, focus_mode,
    chat_model=None,
    embedding_model=None,
    optimization_mode=None,
    history=None,
    system_instructions=None,
    stream=False
) -> dict:
    """
    Search using the Perplexica API

    Args:
        query (str): The search query
        chat_model (dict, optional): Chat model configuration with:
            provider: Provider name (e.g., openai, ollama)
            name: Model name (e.g., gpt-4o-mini)
            customOpenAIBaseURL: Optional custom OpenAI base URL
            customOpenAIKey: Optional custom OpenAI API key
        embedding_model (dict, optional): Embedding model configuration with:
            provider: Provider name (e.g., openai)
            name: Model name (e.g., text-embedding-3-small)
            customOpenAIBaseURL: Optional custom OpenAI base URL
            customOpenAIKey: Optional custom OpenAI API key
        focus_mode (str): Search focus mode (webSearch, academicSearch, etc.)
        optimization_mode (str, optional): Optimization mode (speed, balanced)
        history (list, optional): Conversation history
        system_instructions (str, optional): Custom system instructions
        stream (bool, optional): Whether to stream responses

    Returns:
        dict: Search results from Perplexica
    """
    
    # Prepare the request payload
    payload = {
        "query": query,
        "focusMode": focus_mode
    }
    
    # Add optional parameters if provided
    if chat_model:
        payload["chatModel"] = chat_model
    if embedding_model:
        payload["embeddingModel"] = embedding_model
    if optimization_mode:
        payload["optimizationMode"] = optimization_mode
    if history:
        payload["history"] = history
    if system_instructions:
        payload["systemInstructions"] = system_instructions
    if stream is not None:
        payload["stream"] = stream
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                PERPLEXICA_BACKEND_URL,
                json=payload,
                timeout=PERPLEXICA_READ_TIMEOUT
            )
            response.raise_for_status()
            return response.json()
    except httpx.HTTPError as e:
        return {"error": f"HTTP error occurred: {str(e)}"}
    except Exception as e:
        return {"error": f"An error occurred: {str(e)}"}

@mcp.tool()
async def search(
    query: Annotated[str, Field(description="Search query")],
    focus_mode: Annotated[str, Field(description="Focus mode: webSearch, academicSearch, writingAssistant, wolframAlphaSearch, youtubeSearch, redditSearch")],
    chat_model: Annotated[dict, Field(description="Chat model configuration")] = DEFAULT_CHAT_MODEL,
    embedding_model: Annotated[dict, Field(description="Embedding model configuration")] = DEFAULT_EMBEDDING_MODEL,
    optimization_mode: Annotated[str, Field(description="Optimization mode: speed or balanced")] = None,
    history: Annotated[list, Field(description="Conversation history")] = None,
    system_instructions: Annotated[str, Field(description="Custom system instructions")] = None,
    stream: Annotated[bool, Field(description="Whether to stream responses")] = False
) -> dict:
    """
    Search using Perplexica's AI-powered search engine.
    
    This tool provides access to Perplexica's search capabilities with various focus modes
    for different types of searches including web search, academic search, writing assistance,
    and specialized searches for platforms like YouTube and Reddit.
    """
    return await perplexica_search(
        query=query,
        focus_mode=focus_mode,
        chat_model=chat_model,
        embedding_model=embedding_model,
        optimization_mode=optimization_mode,
        history=history,
        system_instructions=system_instructions,
        stream=stream
    )

def main():
    """Main entry point for the Perplexica MCP server."""
    parser = argparse.ArgumentParser(description="Perplexica MCP Server")
    parser.add_argument(
        "transport",
        choices=["stdio", "sse", "http"],
        help="Transport type to use"
    )
    parser.add_argument(
        "host",
        nargs="?",
        default="0.0.0.0",
        help="Host to bind to for SSE/HTTP transports (default: 0.0.0.0)"
    )
    parser.add_argument(
        "port",
        nargs="?",
        type=int,
        default=3001,
        help="Port for SSE/HTTP transports (default: 3001)"
    )
    
    args = parser.parse_args()
    
    if args.transport == "stdio":
        # Use FastMCP's stdio transport
        mcp.run()
    elif args.transport == "sse":
        # Use FastMCP's SSE transport
        print(f"Starting Perplexica MCP server with SSE transport on {args.host}:{args.port}")
        print(f"SSE endpoint: http://{args.host}:{args.port}/sse")
        uvicorn.run(mcp.sse_app(), host=args.host, port=args.port)
    elif args.transport == "http":
        # Use FastMCP's Streamable HTTP transport
        print(f"Starting Perplexica MCP server with Streamable HTTP transport on {args.host}:{args.port}")
        print(f"HTTP endpoint: http://{args.host}:{args.port}/mcp")
        uvicorn.run(mcp.streamable_http_app(), host=args.host, port=args.port)

if __name__ == "__main__":
    main()