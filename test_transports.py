#!/usr/bin/env python3
"""
Test script for verifying all transport modes of the Perplexica MCP server.
"""

import asyncio
import json
import requests
import subprocess
import time
import sys

def test_http_transport(host: str = "localhost", port: int = 3002) -> bool:
    """Test HTTP transport by making a direct API call."""
    print(f"Testing HTTP transport on {host}:{port}...")
    
    try:
        # Test health endpoint
        health_response = requests.get(f"http://{host}:{port}/health", timeout=5)
        if health_response.status_code == 200:
            print("✓ Health endpoint working")
        else:
            print(f"✗ Health endpoint failed: {health_response.status_code}")
            return False
        
        # Test search endpoint
        search_payload = {
            "query": "What is artificial intelligence?",
            "focus_mode": "webSearch",
            "optimization_mode": "speed"
        }
        
        search_response = requests.post(
            f"http://{host}:{port}/search",
            json=search_payload,
            timeout=30
        )
        
        if search_response.status_code == 200:
            result = search_response.json()
            if "message" in result or "error" not in result:
                print("✓ Search endpoint working")
                return True
            else:
                print(f"✗ Search endpoint returned error: {result}")
                return False
        else:
            print(f"✗ Search endpoint failed: {search_response.status_code}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"✗ HTTP transport test failed: {e}")
        return False

def test_stdio_transport() -> bool:
    """Test stdio transport by running the server in stdio mode."""
    print("Testing stdio transport...")
    
    try:
        # Create a simple MCP request
        mcp_request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/list",
            "params": {}
        }
        
        # Start the server process
        process = subprocess.Popen(
            [sys.executable, "perplexica_mcp_tool.py", "--transport", "stdio"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Send the request
        request_line = json.dumps(mcp_request) + "\n"
        stdout, stderr = process.communicate(input=request_line, timeout=10)
        
        if process.returncode == 0 or "search" in stdout:
            print("✓ Stdio transport working")
            return True
        else:
            print(f"✗ Stdio transport failed: {stderr}")
            return False
            
    except (subprocess.TimeoutExpired, subprocess.SubprocessError) as e:
        print(f"✗ Stdio transport test failed: {e}")
        if 'process' in locals():
            process.kill()
        return False

async def test_sse_transport(host: str = "localhost", port: int = 3001) -> bool:
    """Test SSE transport by checking the SSE endpoint."""
    print(f"Testing SSE transport on {host}:{port}...")
    
    try:
        # For SSE, we'll just check if the endpoint is accessible
        # A full SSE test would require an SSE client
        response = requests.get(f"http://{host}:{port}/sse", timeout=5, stream=True)
        
        if response.status_code == 200:
            print("✓ SSE endpoint accessible")
            return True
        else:
            print(f"✗ SSE endpoint failed: {response.status_code}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"✗ SSE transport test failed: {e}")
        return False

def run_server_background(transport: str, **kwargs) -> subprocess.Popen:
    """Start a server in the background for testing."""
    cmd = [sys.executable, "perplexica_mcp_tool.py", "--transport", transport]
    
    if transport in ["sse", "http", "all"]:
        cmd.extend(["--host", kwargs.get("host", "localhost")])
        
    if transport in ["sse", "all"]:
        cmd.extend(["--sse-port", str(kwargs.get("sse_port", 3001))])
        
    if transport in ["http", "all"]:
        cmd.extend(["--http-port", str(kwargs.get("http_port", 3002))])
    
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

def main():
    """Run all transport tests."""
    print("Perplexica MCP Server Transport Tests")
    print("=" * 40)
    
    results = {}
    
    # Test stdio transport
    results["stdio"] = test_stdio_transport()
    
    # Test HTTP transport
    print("\nStarting HTTP server for testing...")
    http_server = run_server_background("http", host="localhost", http_port=3002)
    time.sleep(3)  # Give server time to start
    
    try:
        results["http"] = test_http_transport("localhost", 3002)
    finally:
        http_server.terminate()
        http_server.wait()
    
    # Test SSE transport
    print("\nStarting SSE server for testing...")
    sse_server = run_server_background("sse", host="localhost", sse_port=3001)
    time.sleep(3)  # Give server time to start
    
    try:
        results["sse"] = asyncio.run(test_sse_transport("localhost", 3001))
    finally:
        sse_server.terminate()
        sse_server.wait()
    
    # Print results
    print("\n" + "=" * 40)
    print("Test Results:")
    print("=" * 40)
    
    for transport, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{transport.upper()} transport: {status}")
    
    all_passed = all(results.values())
    print(f"\nOverall: {'✓ ALL TESTS PASSED' if all_passed else '✗ SOME TESTS FAILED'}")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())