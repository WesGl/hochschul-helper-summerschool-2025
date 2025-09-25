# src/tools/search.py
from __future__ import annotations

import json
import os
from enum import Enum
from typing import Any, Dict, List, Optional

from tavily import TavilyClient

from ..models import LLM

# ----------------- Konfiguration -----------------
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
# SEARCH_MODEL = os.getenv("RAG_MODEL", "openai/gpt-4o-mini")  # wird für Summarize & Router genutzt
SEARCH_MODEL = os.getenv("RAG_MODEL", "deepseek/deepseek-chat-v3.1:free")  # wird für Summarize & Router genutzt

_llm = LLM(SEARCH_MODEL)

client = TavilyClient(api_key=TAVILY_API_KEY)

# Platzhalter-Domains – nachträglich mit echten Domains ersetzen
DOMAINS_FACHSCHAFT: List[str] = [
    "https://iwi-hka.de/",
    "https://mmt-hka.de/",
    "https://hka-eit.de/site/%C3%BCber-uns.html",
    "https://fachschaftw.de/",
    "https://asta-hka.de/wp-content/uploads/2024/11/Fachschaftsordnung-AB.pdf",
]

DOMAINS_HKA: List[str] = [
    "https://www.h-ka.de/",
    "https://raumzeit.hka-iwi.de/timetables",
]


# ----------------- Scopes -----------------
class SearchScope(str, Enum):
    GENERAL = "general"
    FACHSCHAFT = "fachschaft"
    HKA = "hka"


def _domains_for_scope(scope: SearchScope) -> List[str]:
    if scope == SearchScope.FACHSCHAFT:
        return DOMAINS_FACHSCHAFT
    if scope == SearchScope.HKA:
        return DOMAINS_HKA
    return []  # General: keine Einschränkung


# ----------------- Prompts -----------------
BASE_SYSTEM = (
    "Du fasst Webresultate sachlich zusammen und gibst die wichtigsten Quellen an. "
    "Alle Fragen beziehen sich auf die Hochschule Karlsruhe (HKA). "
    "Wenn ein Scope vorgegeben ist, berücksichtige ihn strikt."
)

ROUTER_SYSTEM = (
    "Du bist ein präziser Router für Websuchen im HKA-Kontext. "
    "Wähle GENAU EINEN Scope passend zur Frage:\n"
    "- 'general': allgemeine Websuche (keine Domainbeschränkung)\n"
    "- 'fachschaft': nur Fachschaftsseiten (Studentenvertretungen, FS-Unterseiten, O-Phase)\n"
    "- 'hka': nur offizielle HKA-Seiten (uniweite/FB-offizielle Domains)\n\n"
    "Du darfst die Query minimal normalisieren (Klartext, Keywords). "
    'Antworte NUR als kompaktes JSON: {"scope":"general|fachschaft|hka","normalized_query":"..."}.'
)


# ----------------- Mini-Agent: Scope-Router -----------------
def _route_scope_with_llm(user_query: str) -> Dict[str, str]:
    """Lässt das LLM den Scope bestimmen und optional die Query normalisieren."""
    messages = [
        {"role": "system", "content": ROUTER_SYSTEM},
        {"role": "user", "content": user_query},
    ]
    raw = _llm.chat(messages, temperature=0.0)  # deterministisch
    # Robust parsen
    try:
        data = json.loads(raw)
        scope = str(data.get("scope", "")).strip().lower()
        norm_q = str(data.get("normalized_query", "")).strip() or user_query
        if scope not in {s.value for s in SearchScope}:
            scope = SearchScope.GENERAL.value
        return {"scope": scope, "normalized_query": norm_q}
    except Exception:
        return {"scope": SearchScope.GENERAL.value, "normalized_query": user_query}


# ----------------- Hilfen -----------------
def _build_context(results: List[Dict[str, Any]], max_snippet_chars: int = 1000) -> str:
    blocks: List[str] = []
    for r in results:
        title = r.get("title") or ""
        url = r.get("url") or ""
        content = r.get("content") or ""
        snippet = content[:max_snippet_chars]
        blocks.append(f"- {title}\n {url}\n {snippet}")
    return "\n\n".join(blocks)


def _summarize(context: str, query: str, scope: SearchScope) -> str:
    scope_note = {
        SearchScope.GENERAL: "Scope: Allgemeine Websuche (keine Domainbeschränkung).",
        SearchScope.FACHSCHAFT: "Scope: Nur Fachschafts-Webseiten.",
        SearchScope.HKA: "Scope: Nur offizielle HKA-Webseiten.",
    }[scope]
    msg = [
        {"role": "system", "content": BASE_SYSTEM},
        {"role": "user", "content": f"{scope_note}\n\nWeb-Kontext:\n{context}\n\nFrage: {query}"},
    ]
    return _llm.chat(msg, temperature=0.0)


# ----------------- Öffentliche API -----------------
def search_and_answer(
    query: str,
    scope: Optional[str] = "auto",
    *,
    top_k: int = 8,
    search_depth: str = "advanced",  # "basic" | "advanced" (falls von Tavily unterstützt)
    fallback_to_general: bool = True,  # sinnvoll: bei engen Scopes auf general fallen, wenn leer
    max_snippet_chars: int = 1000,
) -> dict:
    """
    Führt eine Tavily-Suche aus und fasst die Resultate per LLM zusammen.
    - scope="auto": interner Mini-Agent wählt general|fachschaft|hka
    - scope in {"general","fachschaft","hka"}: explizit erzwingen
    Rückgabe: {"answer": <str>, "citations": [<url>, ...]}
    """
    print(f"TOOL search_and_answer was called")
    # 1) Scope bestimmen
    if scope is None or str(scope).lower() == "auto":
        routed = _route_scope_with_llm(query)
        scope_value = routed["scope"]
        used_query = routed["normalized_query"]
    else:
        scope_value = str(scope).lower()
        if scope_value not in {s.value for s in SearchScope}:
            scope_value = SearchScope.GENERAL.value
        used_query = query

    scope_enum = SearchScope(scope_value)
    include_domains = _domains_for_scope(scope_enum)

    # 2) Tavily-Suche
    search_kwargs = {
        "query": used_query,
        "max_results": top_k,
        "include_domains": include_domains or None,
        "search_depth": search_depth,
    }
    res = client.search(**search_kwargs)
    results = res.get("results", []) or []

    # 3) Optionaler Fallback, falls enger Scope leer ist
    if not results and include_domains and fallback_to_general:
        res = client.search(query=used_query, max_results=top_k, search_depth=search_depth)
        results = res.get("results", []) or []
        scope_enum = SearchScope.GENERAL  # Kennzeichne, dass Ergebnis aus General kam

    # 4) Zusammenfassen
    if results:
        context = _build_context(results, max_snippet_chars=max_snippet_chars)
        summary = _summarize(context, used_query, scope_enum)
        citations = [r.get("url") for r in results if r.get("url")]
        print(f"TOOL search_and_answer finished")
        return {"answer": summary, "citations": citations}

    # 5) Keine Treffer
    msg = "Es wurden keine geeigneten Treffer gefunden."
    print(f"TOOL search_and_answer finished - no results")
    return {"answer": msg, "citations": []}


# Optionale direkte Wrapper
def search_general(query: str, **kwargs) -> dict:
    return search_and_answer(query, scope=SearchScope.GENERAL.value, **kwargs)


def search_fachschaft(query: str, **kwargs) -> dict:
    return search_and_answer(query, scope=SearchScope.FACHSCHAFT.value, **kwargs)


def search_hka(query: str, **kwargs) -> dict:
    return search_and_answer(query, scope=SearchScope.HKA.value, **kwargs)
