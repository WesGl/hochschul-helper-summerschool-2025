from __future__ import annotations

import os

from chromadb import PersistentClient

from src.models import LLM

from .ingest import get_embedder

RAG_MODEL = os.getenv("RAG_MODEL", "deepseek/deepseek-chat-v3.1:free")
CHROMA_COLLECTION_NAME = "timetables"
CHROMA_PERSIST_DIR = "./vectordb"
_llm = LLM(RAG_MODEL)
SYSTEM_PROMPT = (
    """Du bist ein Assistent für die Stundenpläne der Hochschule. 
        Deine Aufgabe ist es, Fragen zu Vorlesungen, Übungen und Räumen anhand 
        der bereitgestellten Dokumenten-Auszüge (Kontext) zu beantworten. 

        Richtlinien:
        - Nutze NUR die Informationen aus dem bereitgestellten Kontext. 
        - Wenn eine Information nicht im Kontext enthalten ist, sage klar: 
        "Dazu liegen mir keine Informationen vor."
        - Formuliere deine Antwort knapp, präzise und in vollständigen deutschen Sätzen.
        - Gib immer an, zu welchem Fach, Wochentag, Uhrzeit und Raum sich deine Antwort bezieht, 
        wenn diese Daten verfügbar sind.
        - Falls mehrere passende Termine im Kontext stehen, liste sie alle übersichtlich auf.
        - Zitiere die Quelle mit dem Dateinamen (`source_file`), wenn möglich.
        - Antworte niemals mit frei erfundenen Daten.

        Dein Ziel ist es, den Benutzer zuverlässig durch den Stundenplan zu leiten."""
)
USER_PROMPT = """Hier sind relevante Dokumenten-Auszüge (Kontext) aus den Stundenplänen:

            {context}

            Frage des Nutzers:
            {question}

            Bitte beantworte die Frage ausschließlich auf Grundlage des Kontexts. 
            Füge am Ende deiner Antwort IMMER die Quelle(n) aus den Metadaten (`source_file`) hinzu."""


def retrieve(query: str, k: int = 6):
    client = PersistentClient(path=str(CHROMA_PERSIST_DIR))
    coll = client.get_or_create_collection(CHROMA_COLLECTION_NAME)
    embedder = get_embedder()
    qv = embedder.encode([query]).tolist()
    res = coll.query(query_embeddings=qv, n_results=k)
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    return list(zip(docs, metas))


def answer(query: str):
    hits = retrieve(query)
    context = "\n\n".join([f"[{m['source']}#{m['chunk']}]\n{d}" for d, m in hits])
    msg = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT.format(context=context, question=query)},
    ]
    out = _llm.chat(msg)

    # naive confidence estimation: length & number of hits
    conf = min(0.95, 0.3 + 0.1 * len(hits)) if hits else 0.2
    cites = [f"{m['source']}#{m['chunk']}" for _, m in hits]
    return out, conf, cites
