# list_tools_test.py
import asyncio
from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport

async def main():
    async with Client(
        StreamableHttpTransport("http://127.0.0.1:9000/mcp")
    ) as cli:
        tools = await cli.list_tools()          # await inside the session
        print("Tools exposed by server:\n", [t.name for t in tools])

asyncio.run(main())
