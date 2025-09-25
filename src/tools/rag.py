# src/tools/rag.py
from __future__ import annotations

import os

from chromadb import PersistentClient

from src.models import LLM

from .ingest import COLL_NAME, DB_DIR, get_embedder

# RAG_MODEL = os.getenv("RAG_MODEL", "openai/gpt-4o-mini")
RAG_MODEL = os.getenv("RAG_MODEL", "deepseek/deepseek-chat-v3.1:free")

_llm = LLM(RAG_MODEL)


SYSTEM = (
    "Du beantwortest Fragen zur Hochschule Karlsruhe. Alle Fragen beziehen sich auf die Hochschule Karlsruhe, das heißt HKA, Hochschule, Universität sind auf die Hochschule Karlsruhe bezogen. "
    "**Hierbei handelt es sich um die Kernthemen:**\n"
    "• Studien-/Prüfungsordnungen (SPO), Modulhandbücher\n"
    "• Zulassungsvoraussetzungen, Bewerbungsverfahren\n"
    "• Rechenzentrum-Anleitungen, technische Hilfen\n"
    "• Studienganginformationen, Modulinhalte\n"
    "• Allgemeine Hochschulorganisation OHNE Termine\n"
    "Wenn die Antwort nicht sicher ist, sag ehrlich 'Ich bin nicht sicher'."
)


PROMPT = "Kontext:\n{context}\n\nFrage: {question}\n" "Antworte präzise und nenne Quellen (Dateiname+Chunk)."


def retrieve(query: str, k: int = 6):
    client = PersistentClient(path=str(DB_DIR))
    coll = client.get_or_create_collection(COLL_NAME)
    embedder = get_embedder()
    qv = embedder.encode([query]).tolist()
    res = coll.query(query_embeddings=qv, n_results=k)
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    return list(zip(docs, metas))


def answer(query: str):
    print(f"Normal RAG answer requested")
    hits = retrieve(query)
    context = "\n\n".join([f"[{m['source']}#{m['chunk']}]\n{d}" for d, m in hits])
    msg = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": PROMPT.format(context=context, question=query)},
    ]
    out = _llm.chat(msg)

    # naive confidence estimation: length & number of hits
    conf = min(0.95, 0.3 + 0.1 * len(hits)) if hits else 0.2
    cites = [f"{m['source']}#{m['chunk']}" for _, m in hits]
    return out, conf, cites
