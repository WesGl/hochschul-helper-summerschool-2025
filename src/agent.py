# src/agent.py
from __future__ import annotations

import json
import os
from typing import Literal, Optional, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

from models import LLM  # dein Wrapper
from src.tools import ics_calendar_tool, rag, search

# GUARD_MODEL = os.getenv("GUARD_MODEL", "openai/gpt-4o-mini")
GUARD_MODEL = os.getenv("GUARD_MODEL", "deepseek/deepseek-chat-v3.1:free")

# SUPERVISOR_MODEL = os.getenv("SUPERVISOR_MODEL", "openai/gpt-4o-mini")
SUPERVISOR_MODEL = os.getenv("SUPERVISOR_MODEL", "deepseek/deepseek-chat-v3.1:free")

CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", 0.6))

_guard = LLM(GUARD_MODEL)
_supervisor = LLM(SUPERVISOR_MODEL)


# ---------- Strukturausgaben ----------
class GuardResult(BaseModel):
    valid: bool
    reason: Optional[str] = None


class Plan(BaseModel):
    tool: Literal["rag", "web", "calendar", "multi"] = "rag"
    query: str = Field(..., description="Kanonische Such-/RAG-Query")
    need_calendar: bool = False


# ---------- Agent-State ----------
class AgentState(TypedDict, total=False):
    user_msg: str
    guard: GuardResult
    plan: Plan
    answer: str
    confidence: float
    citations: list[str]
    ics_filename: Optional[str]
    ics_bytes: Optional[bytes]
    done: bool


# ---------- Prompts ----------
GUARD_PROMPT = (
    "Beurteile knapp, ob die Nutzerfrage legitime HKA-Informationen betrifft. "
    "Missbrauch/Off-Topic (Code, allgemeine LLM-Fragen) -> false. "
    "Antworte als kompaktes JSON {valid: bool, reason: string?}."
)

SUPERVISOR_PROMPT = (
    "Du bist ein Tool-Router. Bevorzuge RAG. "
    "Wenn RAG-Konfidenz < THRESHOLD, nutze Websuche. "
    "Bei Termin-Intent setze need_calendar=true. "
    "Antworte als JSON {tool: rag|web|calendar|multi, query: string, need_calendar: bool}."
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
    sys = SUPERVISOR_PROMPT.replace("THRESHOLD", str(CONFIDENCE_THRESHOLD))
    messages = [
        {"role": "system", "content": sys},
        {"role": "user", "content": state["user_msg"]},
    ]
    raw = _supervisor.chat(messages)
    try:
        plan = Plan.model_validate_json(raw) if raw.strip().startswith("{") else Plan(**json.loads(raw))
    except Exception:
        plan = Plan(tool="rag", query=state["user_msg"], need_calendar=False)
    state["plan"] = plan
    return state


def route_tools(state: AgentState) -> str:
    tool = state["plan"].tool
    if tool == "rag":
        return "rag"
    if tool == "web":
        return "web"
    if tool == "calendar":
        return "calendar"
    return "multi"


def rag_node(state: AgentState) -> AgentState:
    q = state["plan"].query
    ans, conf, cites = rag.answer(q)
    state["answer"], state["confidence"], state["citations"] = ans, float(conf), cites or []
    # optional Autonachschlag ins Web
    if state["plan"].need_calendar:
        # erzeugt zusätzlich ICS
        ics_bytes, filename = ics_calendar_tool.make_ics_from_text(q)
        state["ics_bytes"], state["ics_filename"] = ics_bytes, filename
    if state.get("confidence", 0.0) < CONFIDENCE_THRESHOLD:
        # an Web-Node weiter
        return state
    state["done"] = True
    return state


def web_node(state: AgentState) -> AgentState:
    q = state["plan"].query
    web = search.search_and_answer(q)  # erwarte dict mit "answer", evtl. "citations"
    if isinstance(web, dict):
        state["answer"] = web.get("answer") or state.get("answer")
        if "citations" in web:
            state["citations"] = web["citations"]
    if state["plan"].need_calendar:
        ics_bytes, filename = ics_calendar_tool.make_ics_from_text(q)
        state["ics_bytes"], state["ics_filename"] = ics_bytes, filename
    state["done"] = True
    return state


def ics_calendar_node(state: AgentState) -> AgentState:
    q = state["plan"].query
    ics_bytes, filename = ics_calendar_tool.make_ics_from_text(q)
    state["answer"] = "Kalenderdatei erstellt."
    state["ics_bytes"], state["ics_filename"] = ics_bytes, filename
    state["done"] = True
    return state


def google_calendar_agent_node(state: AgentState) -> AgentState:

    return state


def multi_node(state: AgentState) -> AgentState:
    # RAG -> ggf. Web -> ggf. ICS
    state = rag_node(state)
    if not state.get("done"):
        state = web_node(state)
    return state


# ---------- Graph bauen ----------
def build_agent():
    g = StateGraph(AgentState)
    g.add_node("guard", guard_node)
    g.add_node("deny", deny_node)
    g.add_node("supervisor", supervisor_node)
    g.add_node("rag", rag_node)
    g.add_node("web", web_node)
    g.add_node("calendar", ics_calendar_node)
    g.add_node("multi", multi_node)

    g.set_entry_point("guard")
    g.add_conditional_edges(
        "guard",
        route_after_guard,
        {"deny": "deny", "supervisor": "supervisor"},
    )

    # Router-Knoten als Bedingung
    g.add_conditional_edges(
        "supervisor",
        route_tools,
        {"rag": "rag", "web": "web", "calendar": "calendar", "multi": "multi"},
    )

    g.add_edge("deny", END)
    g.add_edge("rag", END)
    g.add_edge("web", END)
    g.add_edge("calendar", END)
    g.add_edge("multi", END)

    return g.compile()


AGENT_GRAPH = build_agent()


def run_agent(user_msg: str) -> dict:
    state: AgentState = {"user_msg": user_msg}
    out = AGENT_GRAPH.invoke(state)
    # Normalisiertes Ergebnis für Chainlit
    result = {
        "answer": out.get("answer"),
        "confidence": out.get("confidence"),
        "citations": out.get("citations"),
    }
    if out.get("ics_bytes") and out.get("ics_filename"):
        result["ics"] = (out["ics_filename"], out["ics_bytes"])
    return result
