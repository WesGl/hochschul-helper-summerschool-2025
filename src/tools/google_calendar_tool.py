import os
import sys

# Add project root to Python path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
import logging
from datetime import datetime, timedelta
from typing import TypedDict, cast

from dotenv import load_dotenv
from langchain.tools import tool

from src.models import LLM
from src.utils.google_calendar_utils import (
    CreateGoogleCalendarEvent,
    DeleteGoogleCalendarEvent,
    ListGoogleCalendarEvents,
    PostponeGoogleCalendarEvent,
    api_resource,
)

# from langchain_openai import ChatOpenAI


load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

CALENDAR_AGENT_MODEL = os.getenv("CALENDAR_AGENT_MODEL", "deepseek/deepseek-chat-v3.1:free")
llm = LLM(CALENDAR_AGENT_MODEL)


@tool
def create_event_tool(
    start_datetime,
    end_datetime,
    summary,
    location="",
    description="",
):
    """
    Create a Google Calendar event.

    Args:
        start_datetime (str): Start datetime (YYYY-MM-DDTHH:MM:SS).
        end_datetime (str): End datetime (YYYY-MM-DDTHH:MM:SS).
        summary (str): Event title.
        location (str, optional): Event location.
        description (str, optional): Event description.
        timezone (str): Timezone.

    Returns:
        str: Confirmation message with event link.
    """
    timezone = "Europe/Berlin"
    try:
        tool = CreateGoogleCalendarEvent(api_resource)
        result = tool._run(start_datetime=start_datetime, end_datetime=end_datetime, summary=summary, location=location, description=description, timezone=timezone)
        logger.info(f"Created event: {summary} from {start_datetime} to {end_datetime}")
        return result
    except Exception as e:
        logger.error(f"Error creating event: {e}")
        return f"❌ Error creating event: {e}"


@tool
def list_events_tool(
    start_datetime,
    end_datetime,
    max_results=10,
):
    """
    List Google Calendar events in a date range.

    Args:
        start_datetime (str): Start datetime (YYYY-MM-DDTHH:MM:SS).
        end_datetime (str): End datetime (YYYY-MM-DDTHH:MM:SS).
        max_results (int): Maximum results to return.
        timezone (str): Timezone.

    Returns:
        list: List of event dicts (each includes event ID, summary, times, etc.).
    """
    timezone = "Europe/Berlin"
    try:
        tool = ListGoogleCalendarEvents(api_resource)
        events = tool._run(start_datetime=start_datetime, end_datetime=end_datetime, max_results=max_results, timezone=timezone)
        logger.info(f"Listed {len(events)} events from {start_datetime} to {end_datetime}")
        return events
    except Exception as e:
        logger.error(f"Error listing events: {e}")
        return []


@tool
def postpone_event_tool(user_query: str) -> str:
    """
    Postpone a Google Calendar event of which in next 7 days based on a natural language user query.
    Automatically extracts the event to postpone and the time adjustment from the query.

    Args:
        user_query (str): Natural language query like "postpone meeting with Bob by 2 hours"

    Returns:
        str: Confirmation message or error.
    """
    timezone = "Europe/Berlin"

    # Get upcoming events (next 7 days)
    start_search = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    end_search = (datetime.now() + timedelta(days=7)).strftime("%Y-%m-%dT%H:%M:%S")

    events = list_events_tool.invoke({"start_datetime": start_search, "end_datetime": end_search, "max_results": 50})

    if not events:
        return "No upcoming events found."

    # Prepare event options for the LLM
    event_options = [f"{idx+1}. {e.get('summary', 'No Title')} at {e.get('start')} (ID: {e.get('id')})" for idx, e in enumerate(events)]
    options_text = "\n".join(event_options)

    # LLM extracts both event selection AND time adjustment
    class PostponeOutput(TypedDict):
        event_id: str
        hours_to_add: int
        minutes_to_add: int

    structured_prompt = (
        f"User query: '{user_query}'\n"
        f"Available events:\n{options_text}\n\n"
        "Extract:\n"
        "1. Which event ID matches the user's description\n"
        "2. How many hours to postpone (can be negative for earlier)\n"
        "3. How many minutes to postpone (can be negative for earlier)\n\n"
        "Examples:\n"
        "- 'by 2 hours' = hours_to_add: 2, minutes_to_add: 0\n"
        "- 'by 30 minutes' = hours_to_add: 0, minutes_to_add: 30\n"
        "- 'by 1.5 hours' = hours_to_add: 1, minutes_to_add: 30\n\n"
        'Respond with JSON: {{"event_id": "abc123", "hours_to_add": 2, "minutes_to_add": 0}}'
    )

    try:
        import json

        import dateutil.parser as parser

        # Use the chat method with proper message format
        messages = [{"role": "user", "content": structured_prompt}]
        llm_response_text = llm.chat(messages)
        llm_response_json = json.loads(llm_response_text.strip())

        event_id = llm_response_json.get("event_id")
        hours_to_add = llm_response_json.get("hours_to_add", 0)
        minutes_to_add = llm_response_json.get("minutes_to_add", 0)

    except (json.JSONDecodeError, Exception) as e:
        logger.error(f"Error parsing LLM response: {e}")
        return f"❌ Could not understand the postponement request: {e}"

    # Find the selected event
    event = next((e for e in events if e.get("id") == event_id), None)
    if not event:
        return f"❌ Event not found: {event_id}"

    # Calculate new times based on original event times
    try:
        # Parse original start/end times
        original_start = parser.parse(event.get("start").replace("/", "-"))
        original_end = parser.parse(event.get("end").replace("/", "-"))

        # Add the postponement delta
        time_delta = timedelta(hours=hours_to_add, minutes=minutes_to_add)
        new_start = original_start + time_delta
        new_end = original_end + time_delta

        # Format for API
        new_start_datetime = new_start.strftime("%Y-%m-%dT%H:%M:%S")
        new_end_datetime = new_end.strftime("%Y-%m-%dT%H:%M:%S")

        # Postpone the event
        tool = PostponeGoogleCalendarEvent(api_resource)
        result = tool._run(event_id=event_id, new_start_datetime=new_start_datetime, new_end_datetime=new_end_datetime, timezone=timezone)

        return f"✅ Postponed '{event.get('summary')}' by {hours_to_add}h {minutes_to_add}m → {result}"

    except Exception as e:
        logger.error(f"Error postponing event: {e}")
        return f"❌ Error postponing event: {e}"


@tool
def delete_event_tool(user_query: str) -> str:
    """
    Delete a Google Calendar event based on a natural language user query.
    Automatically finds events in the next 7 days and selects the correct one to delete.

    Args:
        user_query (str): Natural language query like "delete meeting with Bob" or "cancel kickoff with Alice"

    Returns:
        str: Confirmation message or error.
    """
    # Get upcoming events (next 7 days)
    start_search = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    end_search = (datetime.now() + timedelta(days=7)).strftime("%Y-%m-%dT%H:%M:%S")

    events = list_events_tool.invoke({"start_datetime": start_search, "end_datetime": end_search, "max_results": 50})

    if not events:
        return "No upcoming events found."

    # Prepare event options for the LLM
    event_options = [f"{idx+1}. {e.get('summary', 'No Title')} at {e.get('start')} (ID: {e.get('id')})" for idx, e in enumerate(events)]
    options_text = "\n".join(event_options)

    # Create a prompt that asks for structured JSON output
    structured_prompt = (
        f"User query: '{user_query}'\n"
        f"Available events:\n{options_text}\n\n"
        "Based on the user's query, which event ID(s) best match the intent for deletion? "
        'Respond with a JSON object in this exact format: {{"event_id": ["id1", "id2"]}}. '
        "Only return the JSON, no other text."
    )

    try:
        import json

        # Use the chat method with proper message format
        messages = [{"role": "user", "content": structured_prompt}]
        llm_response_text = llm.chat(messages)
        llm_response_json = json.loads(llm_response_text.strip())
        selected_event_ids = llm_response_json.get("event_id", [])

    except (json.JSONDecodeError, Exception) as e:
        logger.error(f"Error parsing LLM response: {e}")
        # Fallback: return first event if parsing fails
        selected_event_ids = [events[0].get("id")] if events else []

    logger.info(f"Selected event IDs for deletion: {selected_event_ids}")

    # Delete all selected events
    deleted_events = []
    for event_id in selected_event_ids:
        event = next((e for e in events if e.get("id") == event_id), None)
        if not event:
            msg = f"❌ Event ID `{event_id}` not found."
            logger.warning(msg)
            deleted_events.append(msg)
            continue

        try:
            tool = DeleteGoogleCalendarEvent(api_resource)
            result = tool._run(event_id=event_id, calendar_id=None)
            msg = f"✅ Deleted event: **{event.get('summary', 'No Title')}** → {result}"
            logger.info(msg)
            deleted_events.append(msg)
        except Exception as e:
            msg = f"❌ Error deleting event `{event_id}`: {e}"
            logger.error(msg)
            deleted_events.append(msg)

    return "\n".join(deleted_events)


calendar_tools = [create_event_tool, list_events_tool, postpone_event_tool, delete_event_tool]


def test_calendar_tools():
    # --- Test creation tool ---
    # First Event
    start_time_1 = datetime.now() + timedelta(hours=1)
    end_time_1 = start_time_1 + timedelta(hours=1, minutes=30)  # Meeting duration 1.5 hour

    # Format to ISO 8601 format
    start_datetime_1 = start_time_1.strftime("%Y-%m-%dT%H:%M:%S")
    end_datetime_1 = end_time_1.strftime("%Y-%m-%dT%H:%M:%S")

    result = create_event_tool.invoke(
        {
            "start_datetime": start_datetime_1,
            "end_datetime": end_datetime_1,
            "summary": "Meeting with Bob",
            "location": "Conference Room A",
            "description": "Discuss project updates.",
        }
    )
    logger.info(f"Result Create: {result}")

    # Second Event
    start_time_2 = datetime.now() + timedelta(hours=4)
    end_time_2 = start_time_2 + timedelta(hours=1, minutes=45)  # Meeting duration 1.75 hour

    # Format to ISO 8601 format
    start_datetime_2 = start_time_2.strftime("%Y-%m-%dT%H:%M:%S")
    end_datetime_2 = end_time_2.strftime("%Y-%m-%dT%H:%M:%S")

    result = create_event_tool.invoke(
        {
            "start_datetime": start_datetime_2,
            "end_datetime": end_datetime_2,
            "summary": "Kickoff with Alice",
            "location": "Conference Room 6",
            "description": "Project kickoff meeting.",
        }
    )
    logger.info(f"Result Create: {result}")

    # --- Test listing tool ---
    # Use a wider time range to capture both created events
    today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    tomorrow_end = today_start + timedelta(days=1)

    result = list_events_tool.invoke(
        {
            "start_datetime": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
            "end_datetime": (datetime.now() + timedelta(weeks=1)).strftime("%Y-%m-%dT%H:%M:%S"),
            "max_results": 20,
        }
    )
    logger.info(f"Result List: {result}")

    # --- Test postpone and delete tools ---
    # Test postponing the first event (Meeting with Bob)
    # Calculate new times: postpone by 2 hours
    new_start_time_1 = start_time_1 + timedelta(hours=2)
    new_end_time_1 = end_time_1 + timedelta(hours=2)

    result = postpone_event_tool.invoke(
        {
            "user_query": "postpone Meeting with Bob by 2 hours",
        }
    )
    logger.info(f"Result Postpone: {result}")

    # Test deleting the second event (Kickoff with Alice)
    result = delete_event_tool.invoke(
        {
            "user_query": "delete Kickoff with Alice meeting",
        }
    )
    logger.info(f"Result Delete: {result}")

    logger.info("Tests completed.")


def process_calendar_request(query: str, hka_context: str, user_intent: str = None) -> dict:
    """
    Main calendar agent function that processes user intent with HKA context.
    """
    if user_intent is None:
        user_intent = query

    # Enhanced intent detection with better keyword matching
    intent_lower = user_intent.lower()

    # TIMETABLE/SCHEDULE QUERY intents - NEW
    timetable_keywords = ["vorlesung", "veranstaltung", "stundenplan", "termine", "wann ist", "welche vorlesung", "kurs"]
    if any(keyword in intent_lower for keyword in timetable_keywords) and hka_context:
        # Return timetable information from HKA context
        return {"message": f"📅 Stundenplan-Informationen:\n{hka_context}", "events": [], "confidence": 0.9}  # Could be enhanced to parse events from hka_context

    # CREATE intents
    create_keywords = ["erstelle", "plane", "trage ein", "add", "create", "hinzufügen", "eintragen", "importiere"]
    if any(keyword in intent_lower for keyword in create_keywords):
        return _handle_create_from_hka(hka_context, user_intent)

    # POSTPONE intents
    postpone_keywords = ["verschiebe", "postpone", "später", "verlege", "ändere zeit"]
    if any(keyword in intent_lower for keyword in postpone_keywords):
        result = postpone_event_tool.invoke({"user_query": user_intent})
        return {"message": result, "events": [], "confidence": 0.9}

    # DELETE intents
    delete_keywords = ["lösche", "cancel", "delete", "absage", "entferne", "storniere"]
    if any(keyword in intent_lower for keyword in delete_keywords):
        result = delete_event_tool.invoke({"user_query": user_intent})
        return {"message": result, "events": [], "confidence": 0.9}

    # LIST intents
    list_keywords = ["zeige", "list", "welche termine", "übersicht", "termine", "anzeigen", "auflisten"]
    if any(keyword in intent_lower for keyword in list_keywords):
        # Check if asking for timetable or calendar events
        if any(kw in intent_lower for kw in timetable_keywords) and hka_context:
            return {"message": f"📅 Stundenplan-Übersicht:\n{hka_context}", "events": [], "confidence": 0.9}
        else:
            return _handle_list_events(user_intent)

    # Default: provide HKA information with suggestion for calendar actions
    suggestion = "\n\nMögliche Aktionen: 'Erstelle Termine', 'Zeige meine Termine', 'Lösche Termin X'"
    return {"message": f"HKA-Stundenplan Informationen:\n{hka_context}{suggestion}", "events": [], "confidence": 0.7}


def _handle_create_from_hka(hka_context: str, user_intent: str) -> dict:
    """Extract events from HKA context and create calendar entries"""
    import re
    from datetime import datetime, timedelta

    try:
        # Parse HKA context for event details using LLM
        extraction_prompt = (
            f"Extract calendar events from this HKA timetable information:\n"
            f"You can only plan events for the upcoming week"
            f"To create events ignore the date information and only use the day of the week and time.\n"
            f"{hka_context}\n\n"
            f"User intent: {user_intent}\n\n"
            f"Extract event details and respond ONLY with valid JSON array (no other text):\n"
            f'[{{"title": "Course Name", "start_date": "YYYY-MM-DD", "start_time": "HH:MM", "end_time": "HH:MM", "location": "Room", "description": "Details"}}]'
            f"if no events found, respond with empty array []"
        )

        messages = [{"role": "user", "content": extraction_prompt}]
        llm_response = llm.chat(messages)

        import json

        events_data = json.loads(llm_response.strip())

        created_events = []
        for event in events_data:
            # Convert to datetime format
            start_datetime = f"{event['start_date']}T{event['start_time']}:00"
            end_datetime = f"{event['start_date']}T{event['end_time']}:00"

            result = create_event_tool.invoke(
                {"start_datetime": start_datetime, "end_datetime": end_datetime, "summary": event["title"], "location": event.get("location", ""), "description": event.get("description", "")}
            )
            created_events.append(result)

        return {"message": f"✅ Created {len(created_events)} events from HKA timetable", "events": created_events, "confidence": 0.9}

    except Exception as e:
        return {"message": f"❌ Error creating events from HKA data: {str(e)}", "events": [], "confidence": 0.3}


def _handle_list_events(user_intent: str) -> dict:
    """List calendar events based on user intent"""
    try:
        # Determine time range from user intent
        now = datetime.now()

        if any(word in user_intent.lower() for word in ["heute", "today"]):
            start_time = now.replace(hour=0, minute=0, second=0)
            end_time = start_time + timedelta(days=1)
        elif any(word in user_intent.lower() for word in ["morgen", "tomorrow"]):
            start_time = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0)
            end_time = start_time + timedelta(days=1)
        elif any(word in user_intent.lower() for word in ["woche", "week"]):
            start_time = now
            end_time = now + timedelta(weeks=1)
        else:
            # Default: next 7 days
            start_time = now
            end_time = now + timedelta(days=7)

        events = list_events_tool.invoke({"start_datetime": start_time.strftime("%Y-%m-%dT%H:%M:%S"), "end_datetime": end_time.strftime("%Y-%m-%dT%H:%M:%S"), "max_results": 20})

        if not events:
            message = "Keine Termine im angegebenen Zeitraum gefunden."
        else:
            event_list = "\n".join([f"• {event.get('summary', 'Kein Titel')} - {event.get('start')} bis {event.get('end')}" for event in events])
            message = f"Gefundene Termine ({len(events)}):\n{event_list}"

        return {"message": message, "events": events, "confidence": 0.9}

    except Exception as e:
        return {"message": f"❌ Fehler beim Abrufen der Termine: {str(e)}", "events": [], "confidence": 0.3}


if __name__ == "__main__":
    test_calendar_tools()
