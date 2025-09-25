# src/agent.py
from __future__ import annotations

import json
import os
from datetime import datetime, timedelta
from typing import Literal, Optional, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

from src.models import LLM
from src.tools import google_calendar_tool, rag, search
from src.tools.rag_calender import answer as calendar_rag_answer

GUARD_MODEL = os.getenv("GUARD_MODEL", "deepseek/deepseek-chat-v3.1:free")
SUPERVISOR_MODEL = os.getenv("SUPERVISOR_MODEL", "deepseek/deepseek-chat-v3.1:free")
CALENDAR_AGENT_MODEL = os.getenv("CALENDAR_AGENT_MODEL", "deepseek/deepseek-chat-v3.1:free")
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", 0.6))

_guard = LLM(GUARD_MODEL)
_supervisor = LLM(SUPERVISOR_MODEL)
_calendar_agent = LLM(CALENDAR_AGENT_MODEL)


# ---------- Strukturausgaben ----------
class GuardResult(BaseModel):
    valid: bool
    reason: Optional[str] = None


class Plan(BaseModel):
    tool: Literal["rag", "web", "rag_calendar"] = "rag"
    query: str = Field(..., description="Kanonische Such-/RAG-Query")


# ---------- Agent-State ----------
class AgentState(TypedDict, total=False):
    user_msg: str
    guard: GuardResult
    plan: Plan
    answer: str
    confidence: float
    citations: list[str]
    calendar_events: Optional[list]
    hka_rag_results: Optional[dict]
    done: bool


# ---------- Prompts ----------
GUARD_PROMPT = (
    "Beurteile knapp, ob die Nutzerfrage legitime HKA-Informationen betrifft. "
    "Legitime Themen (Termine, Stundenplan, Kalender, Prüfungen, HKA-Infos, Termine in Kalender eintragen, Vorlesungsinformationen) -> true. "
    "Missbrauch/Off-Topic (Code, allgemeine LLM-Fragen) -> false. "
    "Antworte als kompaktes JSON {valid: bool, reason: string?}."
)


# SUPERVISOR_PROMPT = (
#     "Du bist ein Tool-Router für HKA-Anfragen. "
#     "Das heißt du kannst Fragen zu allen Hochschul-Themen beantworten, inklusive Stundenplan, Vorlesungen, Prüfungen, allgemeine HKA-Infos und Kalenderfunktionen sowie zum studentischen Leben oder Übergeordneten Informationen zu Karlsruhe, solange sie Bezug zur Hoschulkontext haben. "
#     "Sieh dir die Anfrage genau an und überlege, was der Benutzer von dir möchte. Bedenke Schritt-für-Schritt und wähle demnach das beste Tool aus.\n"
#     "- 'rag_calendar': Für Stundenplan, Vorlesungstermine, Veranstaltungen, 'wann ist', Kalenderfragen, Termine in Kalender eintragen, Termine ausgeben, Termine verschieben\n"
#     "- 'rag': Für allgemeine HKA-Informationen mit Web-Fallback. Besonders gut geeignet für Dokumente wie die Studien- und Prüfungsordnungen (SPO), Modulhandbücher, Rechenzentrum (rz) Flyer (Anleitungen) und Zulassungssatzungen. Nicht geeignet für konkrete Termine, Kalenderanfragen und Vorlesungstermine. Hat langfristig relevante Informationen zur Hochschule und Studiengängen, keine tagesaktuellen Informationen.\n"
#     "- 'web': Für aktuelle/spezifische Infos mit RAG-Fallback\n"
#     "Antworte nur mit valider JSON {tool: rag|web|rag_calendar, query: string}."
# )
SUPERVISOR_PROMPT = (
    "Du bist ein Tool-Router für HKA-Anfragen. Analysiere die Nutzeranfrage GENAU und wähle das passende Tool:\n\n"
    "**rag_calendar** - Verwende für ALLE Anfragen zu:\n"
    "• Stundenpläne, Vorlesungszeiten, Veranstaltungszeiten\n"
    "• Wann/wo findet etwas statt (Vorlesungen, Übungen, Praktika)\n"
    "• Termine erstellen/eintragen/hinzufügen in Kalender\n"
    "• Kalenderoperationen (verschieben, löschen, auflisten)\n"
    "• Raumbelegungen, Raumsuche für Termine\n"
    "• Professor-Sprechzeiten, Öffnungszeiten\n"
    "• Prüfungstermine, Klausurtermine\n"
    "• Schlüsselwörter: 'wann', 'wo', 'Uhrzeit', 'Termin', 'Kalender', 'Zeit', 'Datum', 'Stundenplan'\n\n"
    "**rag** - Verwende für allgemeine HKA-Informationen OHNE Zeitbezug:\n"
    "• Studien-/Prüfungsordnungen (SPO), Modulhandbücher\n"
    "• Zulassungsvoraussetzungen, Bewerbungsverfahren\n"
    "• Rechenzentrum-Anleitungen, technische Hilfen\n"
    "• Studienganginformationen, Modulinhalte\n"
    "• Allgemeine Hochschulorganisation OHNE Termine\n"
    "• Schlüsselwörter: 'wie', 'was', 'welche', 'Verfahren', 'Ordnung', 'Modul'\n\n"
    "**web** - Verwende für:\n"
    "• Aktuelle/neue Informationen, die nicht in Dokumenten stehen\n"
    "• Spezifische Personen, Kontakte, News\n"
    "• Als Fallback wenn rag/rag_calendar nicht passend\n\n"
    "**ENTSCHEIDUNGSLOGIK:**\n"
    "1. Enthält die Frage Zeit-/Termin-/Kalenderbezug? → rag_calendar\n"
    "2. Geht es um Dokumente/Ordnungen/Module? → rag\n"
    "3. Geht es um aktuelle/spezifische Infos? → web\n\n"
    "**BEISPIELE:**\n"
    "• 'Wann ist Mathe-Vorlesung?' → rag_calendar\n"
    "• 'Trage Vorlesung in Kalender ein' → rag_calendar\n"
    "• 'Welche Module gibt es in Informatik?' → rag\n"
    "• 'Wie ist die Prüfungsordnung?' → rag\n"
    "• 'Wer ist der neue Dekan?' → web\n\n"
    'Antworte nur mit valider JSON {"tool": "rag|web|rag_calendar", "query": "string"}.'
)


# ---------- Nodes ----------
def guard_node(state: AgentState) -> AgentState:
    print(f"AGENT guard_node was called")
    messages = [
        {"role": "system", "content": GUARD_PROMPT},
        {"role": "user", "content": state["user_msg"]},
    ]
    raw = _guard.chat(messages)
    try:
        data = GuardResult.model_validate_json(raw) if raw.strip().startswith("{") else GuardResult.model_validate_json(json.dumps(json.loads(raw)))
    except Exception:
        data = GuardResult(valid=True, reason="fallback")
    state["guard"] = data
    print(f"AGENT guard_node finished")
    return state


def route_after_guard(state: AgentState) -> str:
    if not state["guard"].valid:
        return "deny"
    return "supervisor"


def deny_node(state: AgentState) -> AgentState:
    print(f"AGENT deny_node was called")
    reason = state["guard"].reason or "Policy"
    state["answer"] = f"❌ Anfrage abgelehnt: {reason}"
    state["done"] = True
    print(f"AGENT deny_node finished")
    return state


def supervisor_node(state: AgentState) -> AgentState:
    print(f"AGENT supervisor_node was called")
    messages = [
        {"role": "system", "content": SUPERVISOR_PROMPT},
        {"role": "user", "content": state["user_msg"]},
    ]
    raw = _supervisor.chat(messages)
    try:
        plan = Plan.model_validate_json(raw) if raw.strip().startswith("{") else Plan(**json.loads(raw))
    except Exception:
        plan = Plan(tool="rag", query=state["user_msg"])
    state["plan"] = plan
    print(f"AGENT supervisor_node finished")
    return state


def route_tools(state: AgentState) -> str:
    return state["plan"].tool


def rag_node(state: AgentState) -> AgentState:
    """RAG with web search fallback"""
    print(f"AGENT rag_node was called")
    q = state["plan"].query
    ans, conf, cites = rag.answer(q)
    state["answer"], state["confidence"], state["citations"] = ans, float(conf), cites or []

    # If confidence is low, try web search as fallback
    if state.get("confidence", 0.0) < CONFIDENCE_THRESHOLD:
        web_result = search.search_and_answer(q)
        if isinstance(web_result, dict):
            state["answer"] = web_result.get("answer", state["answer"])
            if "citations" in web_result:
                state["citations"].extend(web_result["citations"])

    state["done"] = True
    print(f"AGENT rag_node finished")
    return state


def web_node(state: AgentState) -> AgentState:
    """Web search with RAG fallback"""
    print(f"AGENT web_node was called")
    q = state["plan"].query
    web_result = search.search_and_answer(q)

    if isinstance(web_result, dict):
        state["answer"] = web_result.get("answer", "")
        state["citations"] = web_result.get("citations", [])
        state["confidence"] = web_result.get("confidence", 0.8)
    else:
        state["answer"] = str(web_result)
        state["confidence"] = 0.5

    # If web search fails or confidence is low, try RAG as fallback
    if state.get("confidence", 0.0) < CONFIDENCE_THRESHOLD or not state.get("answer"):
        rag_ans, rag_conf, rag_cites = rag.answer(q)
        if rag_conf > state.get("confidence", 0.0):
            state["answer"] = rag_ans
            state["confidence"] = float(rag_conf)
            state["citations"] = rag_cites or []

    state["done"] = True
    print(f"AGENT web_node finished")
    return state


def rag_calendar_node(state: AgentState) -> AgentState:
    """Step 1: HKA timetable RAG lookup only"""
    print(f"AGENT rag_calendar_node was called")
    q = state["plan"].query

    try:
        # Use the timetables-specific RAG instead of general RAG
        timetable_ans, timetable_conf, timetable_cites = calendar_rag_answer(f"HKA Stundenplan Termine Veranstaltungen: {q}")

        # Ensure proper types
        if not isinstance(timetable_conf, (int, float)):
            timetable_conf = 0.5
        if not isinstance(timetable_cites, list):
            timetable_cites = []

    except Exception as e:
        timetable_ans = f"Fehler beim Abrufen der HKA-Daten: {e}"
        timetable_conf = 0.0
        timetable_cites = []

    # Store results for calendar agent
    state["hka_rag_results"] = {"answer": timetable_ans, "confidence": float(timetable_conf), "citations": timetable_cites or []}

    print(f"AGENT rag_calendar_node finished")
    return state


def calendar_agent_node(state: AgentState) -> AgentState:
    """Calendar operations using HKA context and intelligent action determination"""
    print(f"AGENT calendar_agent_node was called")
    import json
    from datetime import datetime, timedelta

    from src.tools.google_calendar_tool import (
        create_event_tool,
        delete_event_tool,
        list_events_tool,
        postpone_event_tool,
    )

    user_query = state["user_msg"]
    hka_context = state.get("hka_rag_results", {})

    # Create context for the calendar agent
    context_info = ""
    if hka_context.get("answer"):
        context_info = f"HKA Kontext: {hka_context['answer']}\n"
        if hka_context.get("citations"):
            context_info += f"Quellen: {', '.join(hka_context['citations'])}\n"

    # Determine what calendar action to take
    action_prompt = f"""
Basierend auf der Nutzeranfrage und dem HKA-Kontext, bestimme die passende Kalenderfunktion:

{context_info}
Nutzeranfrage: {user_query}

Verfügbare Aktionen:
- "list": Termine anzeigen/auflisten (z.B. "zeige meine Termine", "was steht heute an")
- "create": Neuen Termin/mehrere Termine erstellen (z.B. "trage Vorlesung ein", "erstelle Termine", "füge hinzu")  
- "postpone": Termin verschieben (z.B. "verschiebe Meeting", "Termin um 2h später")
- "delete": Termin löschen (z.B. "lösche Termin", "entferne Meeting")
- "answer_only": Nur Antwort geben ohne Kalenderfunktion (bei reinen Informationsanfragen)

Beachte:
- Wenn der HKA-Kontext konkrete Termine/Vorlesungszeiten enthält UND der User nach Kalendereintragung fragt → "create"
- Wenn der User explizit nach "eintragen", "hinzufügen", "erstellen" fragt → "create"
- Wenn nur Informationen gesucht werden ohne Kalenderaktion → "answer_only"

Antworte ausschließlich mit valider JSON: {{"action": "list|create|postpone|delete|answer_only", "reasoning": "warum diese Aktion"}}
"""

    try:
        messages = [{"role": "user", "content": action_prompt}]
        action_response = _calendar_agent.chat(messages)

        # Clean the response and try to parse JSON
        cleaned_response = action_response.strip()
        if not cleaned_response.startswith("{"):
            # Try to find JSON in the response
            import re

            json_match = re.search(r"\{.*\}", cleaned_response, re.DOTALL)
            if json_match:
                cleaned_response = json_match.group()

        action_data = json.loads(cleaned_response)
        action = action_data.get("action", "answer_only")
    except (json.JSONDecodeError, Exception) as e:
        print(f"Error parsing action response: {e}")
        action = "answer_only"

    # Execute the determined action
    calendar_result = ""

    if action == "list":
        # List upcoming events
        start_time = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        end_time = (datetime.now() + timedelta(days=7)).strftime("%Y-%m-%dT%H:%M:%S")

        events = list_events_tool.invoke({"start_datetime": start_time, "end_datetime": end_time, "max_results": 10})

        state["calendar_events"] = events
        calendar_result = f"📅 Gefundene Termine: {len(events)} Einträge"

    elif action == "create":
        # Create event(s) based on HKA context and user request
        create_prompt = f"""
Erstelle Kalendereinträge basierend auf:

HKA-Kontext: {hka_context.get('answer', '')}
Nutzeranfrage: {user_query}

Wichtig: Verwende realistische Zeiten. Aktuelles Datum: {datetime.now().strftime('%Y-%m-%d %H:%M')}

Extrahiere ALLE Termine aus dem HKA-Kontext und erstelle für jeden einen Eintrag.
Antworte NUR mit JSON-Array, ohne zusätzlichen Text:
[
  {{"title": "Vorlesung 1", "start": "2025-01-20T09:00:00", "end": "2025-01-20T10:30:00", "location": "Raum A123", "description": "Beschreibung"}},
  {{"title": "Vorlesung 2", "start": "2025-01-21T14:00:00", "end": "2025-01-21T15:30:00", "location": "Raum B456", "description": "Beschreibung"}}
]
"""

        try:
            messages = [{"role": "user", "content": create_prompt}]
            create_response = _calendar_agent.chat(messages)

            # Improved JSON cleaning and parsing
            cleaned_response = create_response.strip()

            # Try to extract JSON array from the response
            import re

            # Look for JSON array first
            json_match = re.search(r"\[.*?\]", cleaned_response, re.DOTALL)
            if not json_match:
                # Fallback to single object
                json_match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", cleaned_response, re.DOTALL)
                if json_match:
                    cleaned_response = f"[{json_match.group()}]"  # Wrap in array
                else:
                    raise ValueError("No valid JSON found in response")
            else:
                cleaned_response = json_match.group()

            # Additional cleaning
            cleaned_response = re.sub(r"[\n\r\t]", " ", cleaned_response)
            cleaned_response = re.sub(r"\s+", " ", cleaned_response)

            events_data = json.loads(cleaned_response)

            # Ensure it's a list
            if not isinstance(events_data, list):
                events_data = [events_data] if isinstance(events_data, dict) else []

            created_events = []
            for event_data in events_data:
                # Validate required fields
                if not event_data.get("start") or not event_data.get("end"):
                    created_events.append(f"❌ Übersprungen: Fehlende Start-/Endzeit in {event_data.get('title', 'Unbekannt')}")
                    continue

                try:
                    result = create_event_tool.invoke(
                        {
                            "start_datetime": event_data.get("start"),
                            "end_datetime": event_data.get("end"),
                            "summary": event_data.get("title", "HKA Termin"),
                            "location": event_data.get("location", ""),
                            "description": event_data.get("description", ""),
                        }
                    )
                    created_events.append(f"✅ Erstellt: {event_data.get('title', 'HKA Termin')} - {result}")
                except Exception as e:
                    created_events.append(f"❌ Fehler bei '{event_data.get('title', 'Unbekannt')}': {e}")

            if created_events:
                calendar_result = f"Kalendererstellung abgeschlossen:\n" + "\n".join(created_events)
            else:
                calendar_result = "❌ Keine Termine konnten erstellt werden."

        except (json.JSONDecodeError, ValueError, Exception) as e:
            print(f"Error creating calendar event(s): {e}")
            calendar_result = f"❌ Fehler beim Erstellen der Termine: {e}"

            # Try to provide helpful context about the parsing error
            if "Extra data" in str(e):
                calendar_result += " (JSON-Parsing-Fehler: Zusätzliche Daten im LLM-Response)"
    elif action == "postpone":
        calendar_result = postpone_event_tool.invoke({"user_query": user_query})

    elif action == "delete":
        calendar_result = delete_event_tool.invoke({"user_query": user_query})

    # Combine HKA context with calendar action result
    final_answer = ""

    # Always show HKA context if available
    if hka_context.get("answer"):
        final_answer += f"📚 **HKA Informationen:**\n{hka_context['answer']}\n\n"

    # Add calendar action result with better formatting
    if calendar_result:
        if action == "create":
            final_answer += f"📅 **Kalenderergebnis:**\n{calendar_result}"
        elif action == "list":
            final_answer += f"📅 **Ihre Termine:**\n{calendar_result}"
            # Add event details if available
            if state.get("calendar_events"):
                events = state["calendar_events"]
                if events:
                    final_answer += f"\n\n**Termindetails:**"
                    for i, event in enumerate(events[:5], 1):  # Show max 5 events
                        title = event.get("summary", "Kein Titel")
                        start = event.get("start", "Unbekannte Zeit")
                        location = event.get("location", "")
                        location_text = f" in {location}" if location else ""
                        final_answer += f"\n{i}. **{title}** - {start}{location_text}"
                    if len(events) > 5:
                        final_answer += f"\n... und {len(events) - 5} weitere Termine"
        elif action == "postpone":
            final_answer += f"📅 **Terminverschiebung:**\n{calendar_result}"
        elif action == "delete":
            final_answer += f"📅 **Terminlöschung:**\n{calendar_result}"
        else:
            final_answer += f"📅 {calendar_result}"

    # Add helpful hints based on action
    if action == "create" and "✅" in calendar_result:
        final_answer += f"\n\n💡 **Tipp:** Die Termine wurden in Ihren Google Kalender eingetragen und sind über den bereitgestellten Link erreichbar."
    elif action == "list" and state.get("calendar_events"):
        final_answer += f"\n\n💡 **Tipp:** Sie können Termine mit 'verschiebe [Terminname]' oder 'lösche [Terminname]' bearbeiten."

    # Fallback message
    if not final_answer.strip():
        final_answer = "Ich konnte keine passende Antwort oder Kalenderfunktion für Ihre Anfrage finden. Bitte versuchen Sie es mit einer konkreteren Frage zu HKA-Terminen oder Stundenplänen."

    # # Temporary simple response for testing
    # final_answer = f"TEST: Calendar agent executed with action '{action}'. HKA Context: {hka_context.get('answer', 'None')[:50]}..."

    state["answer"] = final_answer
    state["confidence"] = float(hka_context.get("confidence", 0.7))
    state["citations"] = hka_context.get("citations", [])
    state["done"] = True

    print(f"AGENT calendar_agent_node finished")
    return state


# ---------- Graph bauen ----------
def build_agent():
    g = StateGraph(AgentState)
    g.add_node("guard", guard_node)
    g.add_node("deny", deny_node)
    g.add_node("supervisor", supervisor_node)
    g.add_node("rag", rag_node)
    g.add_node("web", web_node)
    g.add_node("rag_calendar", rag_calendar_node)
    g.add_node("calendar_agent", calendar_agent_node)

    g.set_entry_point("guard")
    g.add_conditional_edges(
        "guard",
        route_after_guard,
        {"deny": "deny", "supervisor": "supervisor"},
    )

    g.add_conditional_edges(
        "supervisor",
        route_tools,
        {"rag": "rag", "web": "web", "rag_calendar": "rag_calendar"},
    )

    # New edge: rag_calendar -> calendar_agent
    g.add_edge("rag_calendar", "calendar_agent")

    g.add_edge("deny", END)
    g.add_edge("rag", END)
    g.add_edge("web", END)
    g.add_edge("calendar_agent", END)

    return g.compile()


AGENT_GRAPH = build_agent()
print(AGENT_GRAPH.get_graph().draw_mermaid())


def run_agent(user_msg: str) -> dict:
    state: AgentState = {"user_msg": user_msg}
    out = AGENT_GRAPH.invoke(state)

    # # Add debugging
    # print(f"RUN_AGENT DEBUG - Final state keys: {list(out.keys())}")
    # print(f"RUN_AGENT DEBUG - Answer: {out.get('answer', 'NO ANSWER')[:100]}...")
    # print(f"RUN_AGENT DEBUG - Done: {out.get('done')}")

    # Normalisiertes Ergebnis für Chainlit
    result = {
        "answer": out.get("answer"),
        "confidence": out.get("confidence"),
        "citations": out.get("citations"),
    }
    if out.get("calendar_events"):
        result["calendar_events"] = out["calendar_events"]

    print(f"RUN_AGENT DEBUG - Final result: {result}")
    return result
