

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
from googlecalendar_service import add_event_final

MY_NUMBER = "YOUR NUMBER"# Insert your number {91}{Your number}
mcp = FastMCP(
    "YOUR SERVER NAME",debug=True
   
)

class RichToolDescription(BaseModel):
    description: str
    use_when: str
    side_effects: str | None


#ADD YOUR CALENDAR TOOL HERE


class CalendarToolDescription(BaseModel):
    description: str
    use_when: str
    side_effects: str | None

CalendarTool=CalendarToolDescription(
    description="This is used to add events to google calendar",
    use_when="whenever user /add-event and send event details then use this to add event to google calendar",
    side_effects=None
)
class DateTimeWithTimeZone(BaseModel):
    dateTime: str = Field(description="Date and time in YYYY-MM-DDTHH:MM:SS+HH:MM format")
    timeZone: str = Field(description="Time zone, e.g., Asia/Kolkata")

class CalendarEvent(BaseModel):
    summary: str = Field(description="Name or summary of the event")
    location: str = Field(description="Location of the event")
    description: str = Field(description="Description of the event")
    start: DateTimeWithTimeZone = Field(description="Start date, time, and time zone")
    end: DateTimeWithTimeZone = Field(description="End date, time, and time zone")

@mcp.tool(description=CalendarTool.model_dump_json())
async def add_event(event: CalendarEvent):
    event_dict = {
            "summary": event.summary,
            "location": event.location,
            "description": event.description,
            "start": {
                "dateTime": event.start.dateTime,
                "timeZone": event.start.timeZone
            },
            "end": {
                "dateTime": event.end.dateTime,
                "timeZone": event.end.timeZone
            }}
    print(event_dict)
    result=await add_event_final(event_dict)
    return result

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
