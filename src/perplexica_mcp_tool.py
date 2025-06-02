#!/usr/bin/env python3

import os
import asyncio
import argparse
import requests
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.server.sse import SseServerTransport
from mcp.types import Tool, TextContent
from pydantic import Field
from typing import Annotated, Sequence
import uvicorn
from uvicorn.config import Config
from fastapi import FastAPI
import json

# Load environment variables from .env file
load_dotenv()

# Get the backend URL from environment variable or use default
PERPLEXICA_BACKEND_URL = os.getenv('PERPLEXICA_BACKEND_URL', 'http://localhost:3000/api/search')

# Create FastMCP server for stdio transport
mcp = FastMCP("Perplexica", dependencies=["requests", "mcp", "python-dotenv", "uvicorn", "fastapi"])

# Create standard MCP server for SSE and HTTP transports
server = Server("perplexica-mcp")

def perplexica_search(
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

    response = requests.post(url, json=payload)
    response.raise_for_status()
    return response.json()

# Add the search tool to FastMCP (for stdio)
@mcp.tool(
    name="search",
    description="Search using Perplexica's AI-powered search engine"
)
def search(
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
    return perplexica_search(
        query=query,
        focus_mode=focus_mode,
        chat_model=chat_model,
        embedding_model=embedding_model,
        optimization_mode=optimization_mode,
        history=history,
        system_instructions=system_instructions,
        stream=stream
    )

# Add the search tool to standard MCP server (for SSE and HTTP)
@server.list_tools()
async def handle_list_tools() -> list[Tool]:
    """List available tools."""
    return [
        Tool(
            name="search",
            description="Search using Perplexica's AI-powered search engine",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query string"
                    },
                    "focus_mode": {
                        "type": "string",
                        "description": "Focus mode, must be one of: 'webSearch', 'academicSearch', 'writingAssistant', 'wolframAlphaSearch', 'youtubeSearch', 'redditSearch'"
                    },
                    "chat_model": {
                        "type": "string",
                        "description": "Chat model configuration with provider (e.g., openai, ollama), name (e.g., gpt-4o-mini), and optional customOpenAIBaseURL and customOpenAIKey",
                        "default": None
                    },
                    "embedding_model": {
                        "type": "string",
                        "description": "Embedding model configuration with provider (e.g., openai) and name (e.g., text-embedding-3-large)",
                        "default": None
                    },
                    "optimization_mode": {
                        "type": "string",
                        "description": "Optimization mode: 'speed' (prioritize speed) or 'balanced' (balance speed and quality)",
                        "default": None
                    },
                    "history": {
                        "type": "array",
                        "items": {
                            "type": "array",
                            "minItems": 2,
                            "maxItems": 2,
                            "prefixItems": [
                                {"type": "string"},
                                {"type": "string"}
                            ]
                        },
                        "description": "Conversation history as list of [role, message] pairs where role is 'human' or 'assistant'",
                        "default": None
                    },
                    "system_instructions": {
                        "type": "string",
                        "description": "Custom instructions to guide the AI's response style, format, or focus area",
                        "default": None
                    },
                    "stream": {
                        "type": "boolean",
                        "description": "Whether to stream responses (default: False)",
                        "default": False
                    }
                },
                "required": ["query", "focus_mode"]
            }
        )
    ]

@server.call_tool()
async def handle_call_tool(name: str, arguments: dict) -> Sequence[TextContent]:
    """Handle tool calls."""
    if name == "search":
        try:
            result = perplexica_search(
                query=arguments["query"],
                focus_mode=arguments["focus_mode"],
                chat_model=arguments.get("chat_model"),
                embedding_model=arguments.get("embedding_model"),
                optimization_mode=arguments.get("optimization_mode"),
                history=arguments.get("history"),
                system_instructions=arguments.get("system_instructions"),
                stream=arguments.get("stream", False)
            )
            return [TextContent(type="text", text=json.dumps(result, indent=2))]
        except Exception as e:
            return [TextContent(type="text", text=f"Error: {str(e)}")]
    else:
        raise ValueError(f"Unknown tool: {name}")

# FastAPI app for HTTP transport
app = FastAPI(title="Perplexica MCP Server")

@app.post("/search")
async def http_search(request: dict):
    """HTTP endpoint for search functionality."""
    try:
        result = perplexica_search(
            query=request["query"],
            focus_mode=request["focus_mode"],
            chat_model=request.get("chat_model"),
            embedding_model=request.get("embedding_model"),
            optimization_mode=request.get("optimization_mode"),
            history=request.get("history"),
            system_instructions=request.get("system_instructions"),
            stream=request.get("stream", False)
        )
        return result
    except Exception as e:
        return {"error": str(e)}

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "perplexica-mcp"}

async def run_stdio():
    """Run the server with stdio transport."""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )

async def run_sse_server(host: str = "0.0.0.0", port: int = 3001):
    """Run the server with SSE transport."""
    # TODO: SSE transport implementation needs to be fixed
    # The current MCP SSE transport API is not clear from documentation
    print(f"SSE transport not yet implemented properly. Would run on {host}:{port}")
    print("SSE transport requires additional research into the correct MCP SSE API usage.")
    
    # For now, just run a simple HTTP server that indicates SSE is not ready
    from fastapi import FastAPI
    app = FastAPI()
    
    @app.get("/")
    async def root():
        return {"message": "SSE transport not yet implemented", "status": "error"}
    
    @app.get("/sse")
    async def sse_endpoint():
        return {"message": "SSE transport not yet implemented", "status": "error"}
    
    config = Config(app, host=host, port=port, log_level="info")
    server_instance = uvicorn.Server(config)
    await server_instance.serve()

async def run_http_server(host: str = "0.0.0.0", port: int = 3002):
    """Run the server with HTTP transport."""
    config = Config(app, host=host, port=port, log_level="info")
    server = uvicorn.Server(config)
    await server.serve()

def main():
    """Main entry point with transport selection."""
    parser = argparse.ArgumentParser(description="Perplexica MCP Server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse", "http", "all"],
        default="stdio",
        help="Transport type to use (default: stdio)"
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host to bind to for SSE/HTTP transports (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--sse-port",
        type=int,
        default=3001,
        help="Port for SSE transport (default: 3001)"
    )
    parser.add_argument(
        "--http-port",
        type=int,
        default=3002,
        help="Port for HTTP transport (default: 3002)"
    )
    
    args = parser.parse_args()
    
    if args.transport == "stdio":
        # Use FastMCP for stdio
        mcp.run()
    elif args.transport == "sse":
        asyncio.run(run_sse_server(args.host, args.sse_port))
    elif args.transport == "http":
        asyncio.run(run_http_server(args.host, args.http_port))
    elif args.transport == "all":
        # Run all transports concurrently
        async def run_all_servers():
            await asyncio.gather(
                run_stdio(),
                run_sse_server(args.host, args.sse_port),
                run_http_server(args.host, args.http_port)
            )
        asyncio.run(run_all_servers())

if __name__ == "__main__":
    main()