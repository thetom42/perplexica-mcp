from fastmcp import Client
import asyncio

SERVER_IP = "..."  # Replace with the server side IP address. Should test both localhost and actual IP.
SERVER_PORT = 12345  # Replace with the port number set by test_serverside.sh

async def main():
    async with Client(f"http://{SERVER_IP}:{SERVER_PORT}/mcp") as client:
        tools = await client.list_tools()
        print(tools)
        result = await client.call_tool('search', {'query': 'How much is iPhone 17 pro', 'focus_mode': 'webSearch', 'system_instructions': 'Your response should be less than 100 words.'})
        print(result)
    
asyncio.run(main())
