#!/usr/bin/env python3
"""
Test script for verifying all transport modes of the Perplexica MCP server.
"""

import asyncio
import json
import httpx
import subprocess
import time
import sys
import os

async def test_http_transport(host: str = "localhost", port: int = 3002) -> bool:
    """Test HTTP transport by checking MCP endpoint."""
    print(f"Testing HTTP transport on {host}:{port}...")
    
    async with httpx.AsyncClient() as client:
        try:
            # Test MCP endpoint - should return 307 redirect for Streamable HTTP
            mcp_response = await client.get(f"http://{host}:{port}/mcp", timeout=5, follow_redirects=False)
            if mcp_response.status_code == 307:
                print("✓ MCP endpoint working (307 redirect as expected)")
                return True
            else:
                print(f"✗ MCP endpoint failed: {mcp_response.status_code}")
                return False
                
        except httpx.RequestError as e:
            print(f"✗ HTTP transport test failed: {e}")
            return False

def _read_line_with_timeout(pipe, timeout=10):
    """Reads a line from a pipe with a timeout."""
    import threading
    import queue
    
    result_queue = queue.Queue()
    
    def read_line():
        try:
            line = pipe.readline()
            result_queue.put(('success', line))
        except Exception as e:
            result_queue.put(('error', str(e)))
    
    thread = threading.Thread(target=read_line)
    thread.daemon = True
    thread.start()
    
    try:
        result_type, result = result_queue.get(timeout=timeout)
        if result_type == 'success':
            return result
        else:
            raise RuntimeError(f"Error reading line: {result}")
    except queue.Empty:
        raise TimeoutError(f"Timed out after {timeout} seconds waiting for line.")
    
def test_stdio_transport() -> bool:
    """Test stdio transport by running the server in stdio mode and performing a handshake."""
    print("Testing stdio transport...")
    process = None
    try:
        cmd = [sys.executable, "src/perplexica_mcp.py", "stdio"]
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1" # Ensure Python's output is unbuffered

        process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env
        )

        # 1. Send initialize request (FastMCP format)
        init_request = {
            "jsonrpc": "2.0",
            "id": 0,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "roots": {
                        "listChanged": True
                    },
                    "sampling": {}
                },
                "clientInfo": {
                    "name": "test-client",
                    "version": "1.0.0"
                },
                "initializationOptions": {}
            }
        }
        process.stdin.write(json.dumps(init_request) + "\n")
        process.stdin.flush()
        
        # 2. Read initialize response
        init_response_line = _read_line_with_timeout(process.stdout, timeout=15) # Increased timeout for init
        try:
            init_response = json.loads(init_response_line)
            if init_response.get("result", {}).get("capabilities"):
                print("✓ Stdio: Server initialized successfully")
            else:
                print(f"✗ Stdio: Initialization failed. Response: {init_response_line}")
                return False
        except json.JSONDecodeError as e:
            print(f"✗ Stdio: Invalid JSON in initialization response: {e}")
            print(f"Raw response: {init_response_line}")
            return False

        # 3. Send initialized notification (required by MCP protocol)
        initialized_notification = {
            "jsonrpc": "2.0",
            "method": "notifications/initialized",
            "params": {}
        }
        process.stdin.write(json.dumps(initialized_notification) + "\n")
        process.stdin.flush()
        
        # Small delay to ensure initialization is complete
        time.sleep(0.1)

        # 4. Send tools/list request
        tools_list_request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/list",
            "params": {}
        }
        process.stdin.write(json.dumps(tools_list_request) + "\n")
        process.stdin.flush()

        # 5. Read tools/list response
        tools_list_response_line = _read_line_with_timeout(process.stdout, timeout=10)
        try:
            tools_list_response = json.loads(tools_list_response_line)
            
            # FastMCP returns tools directly in result, not nested under "tools"
            if "result" in tools_list_response:
                tools = tools_list_response["result"]
                # Handle both FastMCP format (direct list) and standard MCP format (nested under "tools")
                if isinstance(tools, dict) and "tools" in tools:
                    tools = tools["tools"]
                
                if isinstance(tools, list) and any("search" in tool.get("name", "") for tool in tools):
                    print("✓ Stdio: Tools listed successfully (found 'search' tool)")
                    return True
                else:
                    print(f"✗ Stdio: No 'search' tool found in tools list: {tools}")
                    return False
            else:
                print(f"✗ Stdio: Tools list failed. Response: {tools_list_response_line}")
                return False
        except json.JSONDecodeError as e:
            print(f"✗ Stdio: Invalid JSON in tools list response: {e}")
            print(f"Raw response: {tools_list_response_line}")
            return False

    except (subprocess.TimeoutExpired, TimeoutError, json.JSONDecodeError, ValueError) as e:
        print(f"✗ Stdio transport test failed: {e}")
        if process and process.poll() is None: # Check if process is still running
            process.kill()
        # Print stderr if available for debugging
        if process and process.stderr:
            try:
                # Use communicate with timeout to avoid blocking
                _, stderr_output = process.communicate(timeout=2)
                if stderr_output:
                    print(f"STDERR: {stderr_output}")
            except subprocess.TimeoutExpired:
                print("STDERR: Could not read stderr (timeout)")
            except Exception as stderr_e:
                print(f"STDERR: Error reading stderr: {stderr_e}")
        return False
    finally:
        if process:
            try:
                if process.poll() is None:  # Process is still running
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        print("Process didn't terminate gracefully, force killing...")
                        process.kill()
                        process.wait(timeout=2)  # Give it a moment to die
            except Exception as cleanup_e:
                print(f"Error during process cleanup: {cleanup_e}")
                try:
                    process.kill()
                except (OSError, ProcessLookupError):
                    pass  # Process might already be dead

async def test_sse_transport(host: str = "localhost", port: int = 3001) -> bool:
    """Test SSE transport by checking the SSE endpoint."""
    print(f"Testing SSE transport on {host}:{port}...")
    
    max_retries = 5
    retry_delay = 2  # Increased delay for SSE server startup
    
    async with httpx.AsyncClient() as client:
        for i in range(max_retries):
            try:
                # Check the SSE endpoint with streaming
                async with client.stream("GET", f"http://{host}:{port}/sse", timeout=5) as response:
                    if response.status_code == 200:
                        # For SSE, we expect a streaming response
                        # Try to read the first few bytes to confirm it's working
                        try:
                            # Read a small chunk to see if we get SSE data
                            async for chunk in response.aiter_text(chunk_size=100):
                                if chunk and ("ping" in chunk or "data:" in chunk or "event:" in chunk):
                                    print("✓ SSE endpoint accessible and streaming")
                                    return True
                                else:
                                    print("✓ SSE endpoint accessible")
                                    return True
                                break  # Only check first chunk
                        except Exception:
                            # No data received, but 200 status means endpoint exists
                            print("✓ SSE endpoint accessible")
                            return True
                    else:
                        print(f"✗ SSE endpoint failed: {response.status_code}")
                        if i == max_retries - 1:  # Last attempt
                            return False
            except httpx.RequestError as e:
                print(f"Attempt {i+1}/{max_retries}: SSE transport test failed: {e}")
                if i < max_retries - 1:
                    print(f"Waiting {retry_delay} seconds before retry...")
                    await asyncio.sleep(retry_delay)
                else:
                    return False
        
        return False  # All retries exhausted

def run_server_background(transport: str, **kwargs) -> subprocess.Popen:
    """Start a server in the background for testing."""
    cmd = [sys.executable, "src/perplexica_mcp.py", transport]
    
    if transport in ["sse", "http", "all"]:
        cmd.extend([kwargs.get("host", "localhost")])
        
    if transport in ["sse", "all"]:
        cmd.extend([str(kwargs.get("sse_port", 3001))])
        
    if transport in ["http", "all"]:
        cmd.extend([str(kwargs.get("http_port", 3002))])
    
    # Pass current environment variables to subprocess so it can access .env file
    env = os.environ.copy()
    
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)

async def wait_for_server_ready(host: str, port: int, max_retries: int = 10, retry_delay: float = 0.5) -> bool:
    """Wait for server to be ready by polling the MCP endpoint."""
    async with httpx.AsyncClient() as client:
        for i in range(max_retries):
            try:
                # For HTTP transport, check the MCP endpoint which should return 307 redirect
                response = await client.get(f"http://{host}:{port}/mcp", timeout=2, follow_redirects=False)
                if response.status_code == 307:  # Expected redirect for Streamable HTTP
                    print(f"✓ Server ready on {host}:{port} after {i+1} attempts")
                    return True
            except httpx.RequestError:
                # Server not ready yet, continue polling
                pass
            
            if i < max_retries - 1:  # Don't sleep on the last attempt
                await asyncio.sleep(retry_delay)
        
        print(f"✗ Server on {host}:{port} not ready after {max_retries} attempts")
        return False

async def main():
    """Run all transport tests."""
    print("Perplexica MCP Server Transport Tests")
    print("=" * 40)
    
    results = {}
    
    # Test stdio transport
    results["stdio"] = test_stdio_transport()
    
    # Test HTTP transport
    print("\nStarting HTTP server for testing...")
    http_server = run_server_background("http", host="localhost", http_port=3002)
    
    # Wait for server to be ready instead of fixed sleep
    server_ready = await wait_for_server_ready("localhost", 3002, max_retries=20, retry_delay=0.5)
    
    if not server_ready:
        # Check if server process failed
        if http_server.poll() is not None:
            print(f"✗ HTTP server failed to start. Exit code: {http_server.returncode}")
            try:
                stdout, stderr = http_server.communicate(timeout=2)
                if stderr:
                    print(f"Server stderr: {stderr}")
            except (subprocess.TimeoutExpired, OSError):
                pass
        results["http"] = False
    else:
        results["http"] = await test_http_transport("localhost", 3002)
    
    # Always clean up the HTTP server
    try:
        http_server.terminate()
        http_server.wait(timeout=5)
    except subprocess.TimeoutExpired:
        print("HTTP server didn't terminate gracefully, force killing...")
        http_server.kill()
    except Exception as e:
        print(f"Error terminating HTTP server: {e}")
    
    # Test SSE transport
    print("\nStarting SSE server for testing...")
    sse_server = run_server_background("sse", host="localhost", sse_port=3001)
    
    # Wait for SSE server to be ready instead of fixed sleep
    # Note: SSE endpoint might not have a health endpoint, so we'll use the existing test function
    # which already has retry logic built in
    results["sse"] = await test_sse_transport("localhost", 3001)
    
    # Always clean up the SSE server
    try:
        sse_server.terminate()
        sse_server.wait(timeout=5)
    except subprocess.TimeoutExpired:
        print("SSE server didn't terminate gracefully, force killing...")
        sse_server.kill()
    except Exception as e:
        print(f"Error terminating SSE server: {e}")
    
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
    sys.exit(asyncio.run(main()))