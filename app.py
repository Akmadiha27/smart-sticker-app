

import os
from typing import Annotated
from fastmcp import FastMCP
from google.oauth2 import service_account
from google.auth.transport.requests import Request
import gspread
from pydantic import BaseModel, Field
from mcp import ErrorData, McpError
from mcp.server.auth.provider import AccessToken
from mcp.types import INTERNAL_ERROR, INVALID_PARAMS, TextContent ,ImageContent
from  google import genai
from google.genai import types 
from  google.genai.types import Content,Part
from app import add_event_final

MY_NUMBER = "YOUR NUMBER"# Insert your number {91}{Your number}
mcp = FastMCP(
    "YOUR SERVER NAME",debug=True
   
)

class RichToolDescription(BaseModel):
    description: str
    use_when: str
    side_effects: str | None


#ADD YOUR CALENDAR TOOL HERE

@mcp.tool()
async def validate() -> str:
    """
    NOTE: This tool must be present in an MCP server used by puch.
    """

    return MY_NUMBER

FetchToolDescription = RichToolDescription(
    description="Fetch a URL and return its content.",
    use_when="Use this tool when the user provides a URL and asks for its content, or when the user wants to fetch a webpage.",
    side_effects="The user will receive the content of the requested URL in a simplified format, or raw HTML if requested.",
)

async def main():
    await mcp.run_async(
        "streamable-http",
        host="0.0.0.0",
        port=8085,
    )


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())