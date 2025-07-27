import os
import json
from fastapi import FastAPI, Request
from fastapi.responses import RedirectResponse
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
from starlette.responses import HTMLResponse

app = FastAPI()
import os
os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'

# Define the scopes and redirect URI
SCOPES = [
    "https://www.googleapis.com/auth/calendar",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/userinfo.profile",
    "openid"
]
REDIRECT_URI = "http://localhost:8000/oauth2callback"


def get_flow():
    return Flow.from_client_secrets_file(
        "credentials.json",  # This should be downloaded from Google Cloud Console
        scopes=SCOPES,
        redirect_uri=REDIRECT_URI
    )


@app.get("/")
async def root():
    return HTMLResponse("<a href='/authorize'>Connect Google Calendar</a>")


@app.get("/authorize")
async def authorize():
    flow = get_flow()
    auth_url, state = flow.authorization_url(access_type="offline", include_granted_scopes="true", prompt="consent")
    return RedirectResponse(auth_url)


@app.get("/oauth2callback")
async def oauth2callback(request: Request):
    flow = get_flow()
    flow.fetch_token(authorization_response=str(request.url))

    credentials = flow.credentials
    token_data = {
        "token": credentials.token,
        "refresh_token": credentials.refresh_token,
        "token_uri": credentials.token_uri,
        "client_id": credentials.client_id,
        "client_secret": credentials.client_secret,
        "scopes": credentials.scopes
    }

    with open("token.json", "w") as token_file:
        json.dump(token_data, token_file)

    return HTMLResponse("<h2>Google Calendar Connected!</h2><a href='/add-event'>Add Event</a>")


def get_calendar_service():

    creds = Credentials.from_authorized_user_file("token.json", SCOPES)
    return build("calendar", "v3", credentials=creds)


#this takes in event details and adds it to the calendar
async def add_event_final(event):
    service = get_calendar_service()
    print(event)

   
    created_event = service.events().insert(calendarId="primary", body=event).execute()
    return {"message": "Event created", "event_link": created_event.get("htmlLink")}
@app.get("/add-event")
async def add_event():
    service = get_calendar_service()
    #add your PERSONALISED MEETING DETAILS
    event = {
        "summary": "AI Meeting with Debdut",
        "location": "Online",
        "description": "Discussing MCP project and calendar API integration.",
        "start": {
            "dateTime": "2025-07-28T10:00:00+05:30",
            "timeZone": "Asia/Kolkata",
        },
        "end": {
            "dateTime": "2025-07-28T11:00:00+05:30",
            "timeZone": "Asia/Kolkata",
        },
    }

    created_event = service.events().insert(calendarId="primary", body=event).execute()
    return {"message": "Event created", "event_link": created_event.get("htmlLink")}
