import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Type

from dateutil import parser, tz
from dotenv import load_dotenv
from google.oauth2 import service_account
from googleapiclient.discovery import build
from langchain_google_community import CalendarToolkit
from pydantic import BaseModel, Field

load_dotenv()

# Path of service account JSON key
SERVICE_ACCOUNT_FILE = "ai-summerschool-2025-30d03eb0fedc.json"

# scope of calender access
SCOPES = ["https://www.googleapis.com/auth/calendar"]

# Test Calender ID
CALENDAR_ID = os.getenv("GOOGLE_CALENDAR_ID")

# create credential using service account
credentials = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)

# build teh api resource manually
api_resource = build("calendar", "v3", credentials=credentials)

# initialize the toolkit with the api_resource
toolkit = CalendarToolkit(api_resource=api_resource)


class GoogleCalendarBaseTool:
    """Base class for Google Calendar tools using a service account."""

    api_resource = None

    def __init__(self, api_resource):
        self.api_resource = api_resource

    @classmethod
    def from_api_resource(cls, api_resource):
        return cls(api_resource=api_resource)


class GetEventsSchema(BaseModel):
    start_datetime: str = Field(..., description="Start datetime (YYYY-MM-DDTHH:MM:SS)")
    end_datetime: str = Field(..., description="End datetime (YYYY-MM-DDTHH:MM:SS)")
    max_results: int = Field(default=10, description="Max results to return")
    timezone: str = Field(default="Europe/Berlin", description="Timezone (TZ database name)")


class ListGoogleCalendarEvents(GoogleCalendarBaseTool):
    def _parse_event(self, event, timezone):
        start = event["start"].get("dateTime", event["start"].get("date"))
        start = parser.parse(start).astimezone(tz.gettz(timezone)).strftime("%Y/%m/%d %H:%M:%S")
        end = event["end"].get("dateTime", event["end"].get("date"))
        end = parser.parse(end).astimezone(tz.gettz(timezone)).strftime("%Y/%m/%d %H:%M:%S")
        event_parsed = dict(start=start, end=end)
        for field in ["summary", "description", "location", "hangoutLink"]:
            event_parsed[field] = event.get(field, None)
        # ADD THIS LINE:
        event_parsed["id"] = event.get("id")
        return event_parsed

    def _run(self, start_datetime, end_datetime, max_results=10, timezone="Europe/Berlin"):
        calendar_id = CALENDAR_ID
        events = []
        start = datetime.strptime(start_datetime, "%Y-%m-%dT%H:%M:%S")
        start = start.replace(tzinfo=tz.gettz(timezone)).isoformat()
        end = datetime.strptime(end_datetime, "%Y-%m-%dT%H:%M:%S")
        end = end.replace(tzinfo=tz.gettz(timezone)).isoformat()
        events_result = self.api_resource.events().list(calendarId=calendar_id, timeMin=start, timeMax=end, maxResults=max_results, singleEvents=True, orderBy="startTime").execute()
        cal_events = events_result.get("items", [])
        events.extend(cal_events)
        events = sorted(events, key=lambda x: x["start"].get("dateTime", x["start"].get("date")))
        return [self._parse_event(e, timezone) for e in events]


class CreateEventSchema(BaseModel):
    start_datetime: str = Field(..., description="Start datetime (YYYY-MM-DDTHH:MM:SS)")
    end_datetime: str = Field(..., description="End datetime (YYYY-MM-DDTHH:MM:SS)")
    summary: str = Field(..., description="Event title")
    location: Optional[str] = Field(default="", description="Event location")
    description: Optional[str] = Field(default="", description="Event description")
    timezone: str = Field(default="Europe/Berlin", description="Timezone (TZ database name)")


class CreateGoogleCalendarEvent(GoogleCalendarBaseTool):
    def _run(self, start_datetime, end_datetime, summary, location="", description="", timezone="Europe/Berlin"):
        calendar_id = CALENDAR_ID
        start = datetime.strptime(start_datetime, "%Y-%m-%dT%H:%M:%S")
        start = start.replace(tzinfo=tz.gettz(timezone)).isoformat()
        end = datetime.strptime(end_datetime, "%Y-%m-%dT%H:%M:%S")
        end = end.replace(tzinfo=tz.gettz(timezone)).isoformat()
        body = {"summary": summary, "start": {"dateTime": start}, "end": {"dateTime": end}}
        if location:
            body["location"] = location
        if description:
            body["description"] = description
        event = self.api_resource.events().insert(calendarId=calendar_id, body=body).execute()
        return "Event created: " + event.get("htmlLink", "Failed to create event")


class DeleteEventSchema(BaseModel):
    event_id: str = Field(..., description="The event ID to delete.")
    calendar_id: Optional[str] = Field(default=None, description="The calendar ID. Defaults to your test calendar.")


class DeleteGoogleCalendarEvent:
    """
    Delete an event from your Google Calendar.

    - Requires only the event ID.
    - Always uses your test calendar ID unless another is specified.
    """

    def __init__(self, api_resource):
        self.api_resource = api_resource
        self.calendar_id = CALENDAR_ID

    def _run(self, event_id, calendar_id=None):
        calendar_id = calendar_id or self.calendar_id
        try:
            self.api_resource.events().delete(calendarId=calendar_id, eventId=event_id).execute()
            return f"Event {event_id} deleted from calendar {calendar_id}."
        except Exception as e:
            return f"Failed to delete event: {e}"


class PostponeEventSchema(BaseModel):
    event_id: str = Field(..., description="The event ID to postpone.")
    new_start_datetime: str = Field(..., description="New start datetime (YYYY-MM-DDTHH:MM:SS)")
    new_end_datetime: str = Field(..., description="New end datetime (YYYY-MM-DDTHH:MM:SS)")
    timezone: str = Field(default="Europe/Berlin", description="Timezone (TZ database name)")
    calendar_id: Optional[str] = Field(default=None, description="The calendar ID. Defaults to your test calendar.")


class PostponeGoogleCalendarEvent:
    """
    Postpone (reschedule) an event in your Google Calendar.

    - Requires event ID and new start/end datetimes.
    - Always uses your test calendar ID unless another is specified.
    """

    def __init__(self, api_resource):
        self.api_resource = api_resource
        self.calendar_id = CALENDAR_ID

    def _run(self, event_id, new_start_datetime, new_end_datetime, timezone="Europe/Berlin", calendar_id=None):
        calendar_id = calendar_id or self.calendar_id
        try:
            # Get the existing event
            event = self.api_resource.events().get(calendarId=calendar_id, eventId=event_id).execute()

            # Format new times
            start = datetime.strptime(new_start_datetime, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=tz.gettz(timezone)).isoformat()
            end = datetime.strptime(new_end_datetime, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=tz.gettz(timezone)).isoformat()

            event["start"]["dateTime"] = start
            event["end"]["dateTime"] = end
            event["start"]["timeZone"] = timezone
            event["end"]["timeZone"] = timezone

            updated_event = self.api_resource.events().update(calendarId=calendar_id, eventId=event_id, body=event).execute()

            return f"Event postponed: {updated_event.get('htmlLink', 'No link')}"
        except Exception as e:
            return f"Failed to postpone event: {e}"


if __name__ == "__main__":
    print("All okay")
