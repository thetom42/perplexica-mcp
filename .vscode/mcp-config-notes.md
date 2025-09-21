# VS Code MCP Configuration Examples

## Current Configuration (.vscode/mcp.json)
The current configuration uses the streamable HTTP transport which VS Code supports natively.

## Alternative Configurations

### Option 1: HTTP with explicit headers
```json
{
  "servers": {
    "perplexica": {
      "type": "http",
      "url": "http://localhost:3001/mcp",
      "headers": {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream"
      }
    }
  }
}
```

### Option 2: SSE transport (legacy but more stable)
```json
{
  "servers": {
    "perplexica": {
      "type": "sse",
      "url": "http://localhost:3001/sse"
    }
  }
}
```

## Troubleshooting Steps

1. **Restart VS Code** after creating/modifying the mcp.json file
2. **Restart the MCP server** to ensure it's running the latest version
3. **Check the MCP server logs** using "MCP: Show Installed Servers" command
4. **Verify the server is accessible** by opening http://localhost:3001/mcp in a browser

## Server Status Commands

- `MCP: Show Installed Servers` - View configured servers
- `MCP: List Servers` - List and manage servers
- `MCP: Reset Cached Tools` - Clear tool cache if needed
- `MCP: Reset Trust` - Reset server trust if needed