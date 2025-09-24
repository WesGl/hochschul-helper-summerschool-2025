# src/router.py
from __future__ import annotations

import os
from curses import raw

from pydantic import BaseModel

from models import LLM
from src.tools import calendar, rag, search

GUARD_MODEL = os.getenv("GUARD_MODEL", "openai/gpt-4o-mini")
SUPERVISOR_MODEL = os.getenv("SUPERVISOR_MODEL", "openai/gpt-4o-mini")
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", 0.6))


_guard = LLM(GUARD_MODEL)
_supervisor = LLM(SUPERVISOR_MODEL)


class GuardResult(BaseModel):
    valid: bool
    reason: str | None = None


GUARD_PROMPT = "Beurteile knapp, ob die Nutzerfrage legitime HKA-Informationen betrifft und frei von Missbrauch ist. " "Antwort-JSON mit Feldern: valid (true/false), reason (kurz)."


SUPERVISOR_PROMPT = (
    "Du bist ein Tool-Router. Bevorzuge RAG. Wenn RAG-Konfidenz < THRESHOLD, nutze Websuche. "
    "Wenn Termin-Intent erkannt wird, erzeuge eine ICS über das Kalender-Tool. "
    "Gib ein JSON mit keys: tool (rag|web|calendar|multi), query, need_calendar (true/false)."
)


def guard_check(user_msg: str) -> GuardResult:
    msg = [
        {"role": "system", "content": GUARD_PROMPT},
        {"role": "user", "content": user_msg},
    ]
    raw = _guard.chat(msg)
    import json

    try:
        data = json.loads(raw)
        return GuardResult(**data)
    except Exception:
        # conservative: allow, but with note
        return GuardResult(valid=True, reason="fallback")


def supervise(user_msg: str):
    msg = [
        {"role": "system", "content": SUPERVISOR_PROMPT},
        {"role": "user", "content": user_msg},
    ]
    import json

    raw = _supervisor.chat(msg)
    try:
        plan = json.loads(raw)
    except Exception:
        plan = {"tool": "rag", "query": user_msg, "need_calendar": False}

    if plan.get("tool") == "rag":
        ans, conf, cites = rag.answer(plan.get("query", user_msg))
        if conf < CONFIDENCE_THRESHOLD:
            web = search.search_and_answer(plan.get("query", user_msg))
            return web
        return {"answer": ans, "confidence": conf, "citations": cites}

    if plan.get("tool") == "web":
        return search.search_and_answer(plan.get("query", user_msg))

    if plan.get("tool") == "calendar":
        ics_bytes, filename = calendar.make_ics_from_text(plan.get("query", user_msg))
        return {"answer": "Kalenderdatei erstellt.", "ics": (filename, ics_bytes)}

    # multi: RAG, then Web if needed, optionally ICS
    ans, conf, cites = rag.answer(plan.get("query", user_msg))
    if conf < CONFIDENCE_THRESHOLD:
        web = search.search_and_answer(plan.get("query", user_msg))
        return web
    return {"answer": ans, "confidence": conf, "citations": cites}
