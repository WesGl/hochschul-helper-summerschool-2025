# src/agent.py
from __future__ import annotations

import json
import os
from typing import Literal, Optional, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

from models import LLM  # dein Wrapper
from src.tools import google_calendar_tool, rag, search

GUARD_MODEL = os.getenv("GUARD_MODEL", "deepseek/deepseek-chat-v3.1:free")
SUPERVISOR_MODEL = os.getenv("SUPERVISOR_MODEL", "deepseek/deepseek-chat-v3.1:free")
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", 0.6))

_guard = LLM(GUARD_MODEL)
_supervisor = LLM(SUPERVISOR_MODEL)


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
    "Missbrauch/Off-Topic (Code, allgemeine LLM-Fragen) -> false. "
    "Antworte als kompaktes JSON {valid: bool, reason: string?}."
)

SUPERVISOR_PROMPT = (
    "Du bist ein Tool-Router für HKA-Anfragen. "
    "Wähle das beste Tool basierend auf der Anfrage:\n"
    "- 'rag_calendar': Für Termine, Stundenplan, Kalenderfragen\n"
    "- 'rag': Für allgemeine HKA-Informationen mit Web-Fallback\n"
    "- 'web': Für aktuelle/spezifische Infos mit RAG-Fallback\n"
    "Antworte als JSON {tool: rag|web|rag_calendar, query: string}."
)


# ---------- Nodes ----------
def guard_node(state: AgentState) -> AgentState:
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
    return state


def route_after_guard(state: AgentState) -> str:
    if not state["guard"].valid:
        return "deny"
    return "supervisor"


def deny_node(state: AgentState) -> AgentState:
    reason = state["guard"].reason or "Policy"
    state["answer"] = f"❌ Anfrage abgelehnt: {reason}"
    state["done"] = True
    return state


def supervisor_node(state: AgentState) -> AgentState:
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
    return state


def route_tools(state: AgentState) -> str:
    return state["plan"].tool


def rag_node(state: AgentState) -> AgentState:
    """RAG with web search fallback"""
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
    return state


def web_node(state: AgentState) -> AgentState:
    """Web search with RAG fallback"""
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
    return state


def rag_calendar_node(state: AgentState) -> AgentState:
    """Step 1: HKA timetable RAG lookup only"""
    q = state["plan"].query

    # Only do RAG lookup for HKA events/timetable
    timetable_ans, timetable_conf, timetable_cites = rag.answer(f"HKA Stundenplan Termine Veranstaltungen: {q}")

    # Store results for calendar agent
    state["hka_rag_results"] = {"answer": timetable_ans, "confidence": float(timetable_conf), "citations": timetable_cites or []}

    return state


def calendar_agent_node(state: AgentState) -> AgentState:
    """Enhanced Step 2: Calendar operations using HKA context"""
    user_intent = state["user_msg"]
    original_query = state["plan"].query
    hka_context = state.get("hka_rag_results", {})

    try:
        # Enhanced logging
        import logging

        logger = logging.getLogger(__name__)
        logger.info(f"Calendar agent processing: {user_intent}")
        logger.info(f"HKA context available: {bool(hka_context.get('answer'))}")

        calendar_result = google_calendar_tool.process_calendar_request(query=original_query, hka_context=hka_context.get("answer", ""), user_intent=user_intent)

        state["answer"] = calendar_result.get("message", "Kalenderoperation durchgeführt.")
        state["calendar_events"] = calendar_result.get("events", [])
        state["citations"] = hka_context.get("citations", [])
        state["confidence"] = max(hka_context.get("confidence", 0.5), calendar_result.get("confidence", 0.8))

        logger.info(f"Calendar operation completed with confidence: {state['confidence']}")

    except Exception as e:
        import logging

        logger = logging.getLogger(__name__)
        logger.error(f"Calendar agent error: {str(e)}")

        # Enhanced fallback with partial HKA info
        hka_info = hka_context.get("answer", "Keine HKA-Informationen verfügbar")
        state["answer"] = (
            f"HKA-Stundenplan Informationen:\n{hka_info}\n\n" f"⚠️ Kalenderfunktion temporär nicht verfügbar: {str(e)}\n" f"Versuchen Sie es später erneut oder kontaktieren Sie den Support."
        )
        state["citations"] = hka_context.get("citations", [])
        state["confidence"] = max(hka_context.get("confidence", 0.3), 0.4)

    state["done"] = True
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
    g.add_node("calendar_agent", calendar_agent_node)  # New node

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
    g.add_edge("calendar_agent", END)  # Calendar agent goes to END

    return g.compile()


AGENT_GRAPH = build_agent()
print(AGENT_GRAPH.get_graph().draw_mermaid())


def run_agent(user_msg: str) -> dict:
    state: AgentState = {"user_msg": user_msg}
    out = AGENT_GRAPH.invoke(state)
    # Normalisiertes Ergebnis für Chainlit
    result = {
        "answer": out.get("answer"),
        "confidence": out.get("confidence"),
        "citations": out.get("citations"),
    }
    if out.get("calendar_events"):
        result["calendar_events"] = out["calendar_events"]
    return result
