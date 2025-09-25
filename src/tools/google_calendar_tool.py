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
    print(f"TOOL create_event_tool was called")
    timezone = "Europe/Berlin"
    try:
        tool = CreateGoogleCalendarEvent(api_resource)
        result = tool._run(start_datetime=start_datetime, end_datetime=end_datetime, summary=summary, location=location, description=description, timezone=timezone)
        logger.info(f"Created event: {summary} from {start_datetime} to {end_datetime}")
        print(f"TOOL create_event_tool finished")
        return result
    except Exception as e:
        logger.error(f"Error creating event: {e}")
        print(f"TOOL create_event_tool finished with error")
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
    print(f"TOOL list_events_tool was called")
    timezone = "Europe/Berlin"
    try:
        tool = ListGoogleCalendarEvents(api_resource)
        events = tool._run(start_datetime=start_datetime, end_datetime=end_datetime, max_results=max_results, timezone=timezone)
        logger.info(f"Listed {len(events)} events from {start_datetime} to {end_datetime}")
        print(f"TOOL list_events_tool finished")
        return events
    except Exception as e:
        logger.error(f"Error listing events: {e}")
        print(f"TOOL list_events_tool finished with error")
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
    print(f"TOOL postpone_event_tool was called")
    timezone = "Europe/Berlin"

    # Get upcoming events (next 7 days)
    start_search = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    end_search = (datetime.now() + timedelta(days=7)).strftime("%Y-%m-%dT%H:%M:%S")

    events = list_events_tool.invoke({"start_datetime": start_search, "end_datetime": end_search, "max_results": 50})

    if not events:
        print(f"TOOL postpone_event_tool finished - no events found")
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
        print(f"TOOL postpone_event_tool finished with parsing error")
        return f"❌ Could not understand the postponement request: {e}"

    # Find the selected event
    event = next((e for e in events if e.get("id") == event_id), None)
    if not event:
        print(f"TOOL postpone_event_tool finished - event not found")
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

        print(f"TOOL postpone_event_tool finished")
        return f"✅ Postponed '{event.get('summary')}' by {hours_to_add}h {minutes_to_add}m → {result}"

    except Exception as e:
        logger.error(f"Error postponing event: {e}")
        print(f"TOOL postpone_event_tool finished with error")
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
    print(f"TOOL delete_event_tool was called")
    # Get upcoming events (next 7 days)
    start_search = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    end_search = (datetime.now() + timedelta(days=7)).strftime("%Y-%m-%dT%H:%M:%S")

    events = list_events_tool.invoke({"start_datetime": start_search, "end_datetime": end_search, "max_results": 50})

    if not events:
        print(f"TOOL delete_event_tool finished - no events found")
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

    print(f"TOOL delete_event_tool finished")
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
    result = list_events_tool.invoke(
        {
            "start_datetime": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
            "end_datetime": (datetime.now() + timedelta(weeks=1)).strftime("%Y-%m-%dT%H:%M:%S"),
            "max_results": 20,
        }
    )
    logger.info(f"Result List: {result}")

    # --- Test postpone and delete tools ---
    result = postpone_event_tool.invoke(
        {
            "user_query": "postpone Meeting with Bob by 2 hours",
        }
    )
    logger.info(f"Result Postpone: {result}")

    result = delete_event_tool.invoke(
        {
            "user_query": "delete Kickoff with Alice meeting",
        }
    )
    logger.info(f"Result Delete: {result}")

    logger.info("Tests completed.")


if __name__ == "__main__":
    test_calendar_tools()
