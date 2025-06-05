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
            name: Model name (e.g., text-embedding-3-large)
        optimization_mode (str, optional): Optimization mode ('speed' or 'balanced')
        focus_mode (str, required): Focus mode, must be one of:
            'webSearch', 'academicSearch', 'writingAssistant',
            'wolframAlphaSearch', 'youtubeSearch', 'redditSearch'
        history (list, optional): Conversation history as list of [role, message] pairs
        system_instructions (str, optional): Custom instructions for AI response
        stream (bool, optional): Whether to stream responses (default: False)

    Returns:
        dict: The search results
    """
    url = PERPLEXICA_BACKEND_URL
    payload = {
        'query': query,
        'focusMode': focus_mode
    }

    if chat_model:
        payload['chatModel'] = chat_model
    if embedding_model:
        payload['embeddingModel'] = embedding_model
    if optimization_mode:
        payload['optimizationMode'] = optimization_mode
    if history:
        payload['history'] = history
    if system_instructions:
        payload['systemInstructions'] = system_instructions
    if stream:
        payload['stream'] = stream

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(url, json=payload)
        response.raise_for_status()
        return response.json()

# Add the search tool to FastMCP (for stdio)
@mcp.tool(
    name="search",
    description="Search using Perplexica's AI-powered search engine"
)
async def search(
    query: Annotated[str, Field(description="Search query string")],
    focus_mode: Annotated[str, Field(description="Focus mode, must be one of: 'webSearch', 'academicSearch', 'writingAssistant', 'wolframAlphaSearch', 'youtubeSearch', 'redditSearch'")],
    chat_model: Annotated[str, Field(description="Chat model configuration with provider (e.g., openai, ollama), name (e.g., gpt-4o-mini), and optional customOpenAIBaseURL and customOpenAIKey", default=None)],
    embedding_model: Annotated[str, Field(description="Embedding model configuration with provider (e.g., openai) and name (e.g., text-embedding-3-large)", default=None)],
    optimization_mode: Annotated[str, Field(description="Optimization mode: 'speed' (prioritize speed) or 'balanced' (balance speed and quality)", default=None)],
    history: Annotated[list[tuple[str, str]], Field(description="Conversation history as list of [role, message] pairs where role is 'human' or 'assistant'", default=None)],
    system_instructions: Annotated[str, Field(description="Custom instructions to guide the AI's response style, format, or focus area", default=None)],
    stream: Annotated[bool, Field(description="Whether to stream responses (default: False)", default=False)]
) -> dict:
    """
    Search using Perplexica's AI-powered search engine

    Args:
        query: The search query or question
        focus_mode: Focus mode, must be one of: 'webSearch', 'academicSearch', 'writingAssistant', 'wolframAlphaSearch', 'youtubeSearch', 'redditSearch'
        chat_model: Chat model configuration with provider (e.g., openai, ollama), name (e.g., gpt-4o-mini), and optional customOpenAIBaseURL and customOpenAIKey
        embedding_model: Embedding model configuration with provider (e.g., openai) and name (e.g., text-embedding-3-large)
        optimization_mode: Optimization mode: 'speed' (prioritize speed) or 'balanced' (balance speed and quality)
        history: Conversation history as list of [role, message] pairs where role is 'human' or 'assistant'
        system_instructions: Custom instructions to guide the AI's response style, format, or focus area
        stream: Whether to stream responses (default: False)

    Returns:
        dict: The search results
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
    """Main entry point with transport selection."""
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